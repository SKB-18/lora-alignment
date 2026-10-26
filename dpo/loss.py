"""
DPO Loss Function and Implicit Reward Computations (Rafailov et al., NeurIPS 2023).
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization Loss.
    
    L_DPO = -E [ log sigma( beta * ( log(pi(yw|x)/pi_ref(yw|x)) - log(pi(yl|x)/pi_ref(yl|x)) ) ) ]
    
    Implicit Reward:
        r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        reference_free: bool = False,
    ):
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"Beta must be strictly positive, got {beta}")
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reference_free = reference_free

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor] = None,
        reference_rejected_logps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes DPO loss and telemetry diagnostics.
        
        Args:
            policy_chosen_logps: Log-probabilities of chosen responses under policy pi_theta.
            policy_rejected_logps: Log-probabilities of rejected responses under policy pi_theta.
            reference_chosen_logps: Log-probabilities under reference model pi_ref.
            reference_rejected_logps: Log-probabilities under reference model pi_ref.
        """
        if self.reference_free or reference_chosen_logps is None or reference_rejected_logps is None:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = self.beta * (pi_logratios - ref_logratios)

        if self.label_smoothing > 0.0:
            # Smoothed negative log-sigmoid
            losses = (
                -F.logsigmoid(logits) * (1.0 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)

        loss = losses.mean()

        # Telemetry metrics
        chosen_rewards = self.beta * (policy_chosen_logps - (reference_chosen_logps if reference_chosen_logps is not None else 0.0))
        rejected_rewards = self.beta * (policy_rejected_logps - (reference_rejected_logps if reference_rejected_logps is not None else 0.0))
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": reward_margin,
            "accuracy": accuracy,
        }

        return loss, metrics
