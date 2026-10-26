"""
Unit tests for DPOTrainer end-to-end loop, convergence, and parameter isolation.
"""

import pytest
import torch
import torch.nn as nn
from lora.toy_lm import ToyCausalLM
from lora.injector import inject_lora_layers, get_lora_parameters
from training.dpo_trainer import DPOTrainer, DPOTrainerConfig


def test_trainer_initialization():
    """Verify trainer correctly identifies trainable LoRA parameters and freezes reference model."""
    policy = ToyCausalLM(vocab_size=32, d_model=16, num_layers=1, num_heads=2)
    ref = ToyCausalLM(vocab_size=32, d_model=16, num_layers=1, num_heads=2)
    inject_lora_layers(policy, target_names=["q_proj", "v_proj"], r=2)

    trainer = DPOTrainer(policy_model=policy, reference_model=ref)

    # Reference model parameters must be completely frozen
    for p in trainer.reference_model.parameters():
        assert not p.requires_grad

    # Optimizer must have only LoRA params
    opt_params = trainer.optimizer.param_groups[0]["params"]
    lora_params = get_lora_parameters(policy)
    assert len(opt_params) == len(lora_params)


def test_trainer_no_trainable_params_raises():
    """Verify DPOTrainer raises ValueError if policy has no trainable parameters."""
    frozen_model = nn.Linear(10, 10)
    for p in frozen_model.parameters():
        p.requires_grad = False

    with pytest.raises(ValueError, match="Policy model has no trainable parameters"):
        DPOTrainer(policy_model=frozen_model)


def test_trainer_train_step_updates_only_lora():
    """Verify a single training step updates only LoRA parameters while base weights remain unchanged."""
    torch.manual_seed(42)
    policy = ToyCausalLM(vocab_size=32, d_model=16, num_layers=1, num_heads=2)
    ref = ToyCausalLM(vocab_size=32, d_model=16, num_layers=1, num_heads=2)
    ref.load_state_dict(policy.state_dict())

    inject_lora_layers(policy, target_names=["q_proj", "v_proj"], r=2)

    # Record initial states
    base_weight_before = policy.layers[0].attn.q_proj.weight.clone()
    lora_b_before = policy.layers[0].attn.q_proj.lora_B.clone()

    batch = {
        "chosen_input_ids": torch.randint(0, 32, (2, 8)),
        "rejected_input_ids": torch.randint(0, 32, (2, 8)),
    }

    trainer = DPOTrainer(policy, ref, config=DPOTrainerConfig(lr=0.01))
    step_metrics = trainer.train_step(batch)

    assert "loss" in step_metrics
    assert "reward_margin" in step_metrics
    assert trainer.step_count == 1

    # Base weight must be completely unchanged
    torch.testing.assert_close(policy.layers[0].attn.q_proj.weight, base_weight_before)

    # LoRA B must have received an update
    assert not torch.allclose(policy.layers[0].attn.q_proj.lora_B, lora_b_before)


def test_trainer_convergence_on_synthetic_preference():
    """Verify multi-step DPO training on preference data increases reward margin and accuracy."""
    torch.manual_seed(42)
    policy = ToyCausalLM(vocab_size=64, d_model=32, num_layers=1, num_heads=2)
    ref = ToyCausalLM(vocab_size=64, d_model=32, num_layers=1, num_heads=2)
    ref.load_state_dict(policy.state_dict())

    inject_lora_layers(policy, target_names=["q_proj", "v_proj"], r=4)

    # Create dataset where chosen completion always has target token 15
    dataset = []
    for _ in range(8):
        prefix = torch.randint(0, 10, (1, 4))
        chosen = torch.cat([prefix, torch.full((1, 4), 15)], dim=1)
        rejected = torch.cat([prefix, torch.full((1, 4), 25)], dim=1)
        dataset.append({
            "chosen_input_ids": chosen,
            "rejected_input_ids": rejected,
        })

    config = DPOTrainerConfig(lr=0.01, beta=0.2, epochs=4)
    trainer = DPOTrainer(policy, ref, config=config)

    # Evaluate before training
    init_eval = trainer.evaluate(dataset)

    # Fit for 4 epochs
    history = trainer.fit(dataset, epochs=4)

    final_eval = trainer.evaluate(dataset)

    # Reward margin should expand significantly
    assert final_eval["val_reward_margin"] > init_eval["val_reward_margin"]
    assert final_eval["val_accuracy"] >= 0.875


def test_trainer_evaluate_empty_dataset():
    """Verify trainer.evaluate handles empty datasets gracefully."""
    policy = ToyCausalLM(vocab_size=32, d_model=16, num_layers=1, num_heads=2)
    inject_lora_layers(policy, target_names=["q_proj", "v_proj"], r=2)
    trainer = DPOTrainer(policy)

    eval_res = trainer.evaluate([])
    assert eval_res["loss"] == 0.0
    assert eval_res["reward_margin"] == 0.0
    assert eval_res["accuracy"] == 0.0


def test_trainer_custom_optimizer():
    """Verify DPOTrainer accepts user-provided custom optimizer."""
    policy = ToyCausalLM(vocab_size=32, d_model=16, num_layers=1, num_heads=2)
    inject_lora_layers(policy, target_names=["q_proj", "v_proj"], r=2)

    lora_params = get_lora_parameters(policy)
    custom_opt = torch.optim.SGD(lora_params, lr=0.05, momentum=0.9)

    trainer = DPOTrainer(policy, optimizer=custom_opt)
    assert isinstance(trainer.optimizer, torch.optim.SGD)
