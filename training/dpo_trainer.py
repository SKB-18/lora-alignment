"""
DPOTrainer: End-to-End Direct Preference Optimization Loop with LoRA/QLoRA Integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
try:
    from ..dpo.loss import DPOLoss
    from ..dpo.utils import compute_sequence_log_probs
except (ImportError, ValueError):
    from dpo.loss import DPOLoss
    from dpo.utils import compute_sequence_log_probs


@dataclass
class DPOTrainerConfig:
    """Configuration for DPOTrainer execution."""
    lr: float = 1e-3
    beta: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    reference_free: bool = False
    average_log_prob: bool = False
    device: str = "cpu"
    epochs: int = 3
    warmup_steps: int = 0


class DPOTrainer:
    """
    Direct Preference Optimization Trainer.

    Manages fine-tuning a policy model (e.g., base LM with injected LoRA/QLoRA layers)
    against a frozen reference model to align completions with human preference rankings.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        config: Optional[DPOTrainerConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.config = config or DPOTrainerConfig()
        self.device = torch.device(self.config.device)

        self.policy_model = policy_model.to(self.device)

        # Freeze reference model
        if reference_model is not None:
            self.reference_model = reference_model.to(self.device)
            self.reference_model.eval()
            for p in self.reference_model.parameters():
                p.requires_grad = False
        else:
            self.reference_model = None

        self.loss_fn = DPOLoss(
            beta=self.config.beta,
            label_smoothing=self.config.label_smoothing,
            reference_free=self.config.reference_free or (self.reference_model is None),
        )

        # Optimizer targeting only trainable parameters (e.g., LoRA adapters)
        trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("Policy model has no trainable parameters (requires_grad=True). Did you inject LoRA?")

        self.optimizer = optimizer or torch.optim.AdamW(
            trainable_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.step_count = 0
        self.history: List[Dict[str, float]] = []

    def _forward_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Helper to get token logits from model."""
        if hasattr(model, "forward"):
            out = model(input_ids)
            # Support both output objects (HuggingFace-like) and raw tensor logits
            if hasattr(out, "logits"):
                return out.logits
            elif isinstance(out, (tuple, list)):
                return out[0]
            return out
        raise TypeError(f"Model {model} must implement forward().")

    def get_batch_logps(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes sequence log-probabilities for chosen and rejected responses.
        """
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)

        chosen_mask = batch.get("chosen_attention_mask")
        if chosen_mask is not None:
            chosen_mask = chosen_mask.to(self.device)

        rejected_mask = batch.get("rejected_attention_mask")
        if rejected_mask is not None:
            rejected_mask = rejected_mask.to(self.device)

        # Forward passes
        chosen_logits = self._forward_model(model, chosen_ids, chosen_mask)
        rejected_logits = self._forward_model(model, rejected_ids, rejected_mask)

        # Labels match input_ids for causal language modeling
        chosen_labels = batch.get("chosen_labels", chosen_ids).to(self.device)
        rejected_labels = batch.get("rejected_labels", rejected_ids).to(self.device)

        chosen_logps = compute_sequence_log_probs(
            chosen_logits,
            chosen_labels,
            attention_mask=chosen_mask,
            average_log_prob=self.config.average_log_prob,
        )

        rejected_logps = compute_sequence_log_probs(
            rejected_logits,
            rejected_labels,
            attention_mask=rejected_mask,
            average_log_prob=self.config.average_log_prob,
        )

        return chosen_logps, rejected_logps

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Executes a single optimization step on a preference batch.
        """
        self.policy_model.train()

        # Policy forward pass (tracks gradients)
        policy_chosen_logps, policy_rejected_logps = self.get_batch_logps(
            self.policy_model, batch
        )

        # Reference forward pass (no gradients)
        ref_chosen_logps, ref_rejected_logps = None, None
        if self.reference_model is not None and not self.config.reference_free:
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps = self.get_batch_logps(
                    self.reference_model, batch
                )

        # Compute DPO loss and telemetry
        loss, metrics = self.loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
        )

        self.optimizer.zero_grad()
        loss.backward()

        if self.config.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy_model.parameters() if p.requires_grad],
                max_norm=self.config.max_grad_norm,
            )

        self.optimizer.step()
        self.step_count += 1

        step_record = {
            "step": self.step_count,
            **metrics,
        }
        self.history.append(step_record)
        return step_record

    def evaluate(
        self,
        dataloader: Union[DataLoader, List[Dict[str, torch.Tensor]]],
    ) -> Dict[str, float]:
        """
        Evaluates policy on validation preference samples without updating weights.
        """
        self.policy_model.eval()

        total_loss = 0.0
        total_margin = 0.0
        total_acc = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                policy_chosen, policy_rejected = self.get_batch_logps(self.policy_model, batch)

                ref_chosen, ref_rejected = None, None
                if self.reference_model is not None and not self.config.reference_free:
                    ref_chosen, ref_rejected = self.get_batch_logps(self.reference_model, batch)

                loss, metrics = self.loss_fn(
                    policy_chosen,
                    policy_rejected,
                    ref_chosen,
                    ref_rejected,
                )

                total_loss += metrics["loss"]
                total_margin += metrics["reward_margin"]
                total_acc += metrics["accuracy"]
                count += 1

        if count == 0:
            return {"loss": 0.0, "reward_margin": 0.0, "accuracy": 0.0}

        return {
            "val_loss": total_loss / count,
            "val_reward_margin": total_margin / count,
            "val_accuracy": total_acc / count,
        }

    def fit(
        self,
        train_dataloader: Union[DataLoader, List[Dict[str, torch.Tensor]]],
        epochs: Optional[int] = None,
        val_dataloader: Optional[Union[DataLoader, List[Dict[str, torch.Tensor]]]] = None,
        step_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop across specified epochs.
        """
        num_epochs = epochs or self.config.epochs
        epoch_summaries = []

        for ep in range(1, num_epochs + 1):
            epoch_loss = 0.0
            epoch_margin = 0.0
            epoch_acc = 0.0
            steps_in_epoch = 0

            for batch in train_dataloader:
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                epoch_margin += metrics["reward_margin"]
                epoch_acc += metrics["accuracy"]
                steps_in_epoch += 1

                if step_callback:
                    step_callback(metrics)

            summary = {
                "epoch": ep,
                "avg_loss": epoch_loss / max(1, steps_in_epoch),
                "avg_reward_margin": epoch_margin / max(1, steps_in_epoch),
                "avg_accuracy": epoch_acc / max(1, steps_in_epoch),
            }

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                summary.update(val_metrics)

            epoch_summaries.append(summary)

        return {
            "epoch_summaries": epoch_summaries,
            "total_steps": self.step_count,
            "final_margin": epoch_summaries[-1]["avg_reward_margin"] if epoch_summaries else 0.0,
            "final_accuracy": epoch_summaries[-1]["avg_accuracy"] if epoch_summaries else 0.0,
        }
