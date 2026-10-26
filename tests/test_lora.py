"""
Unit tests for LoRA (Low-Rank Adaptation) layers and injector.
"""

import pytest
import math
import torch
import torch.nn as nn
from lora.layer import LoRALinear
from lora.injector import (
    inject_lora_layers,
    mark_only_lora_as_trainable,
    get_lora_parameters,
    count_trainable_parameters,
)
from lora.toy_lm import ToyCausalLM


def test_lora_initialization():
    """Verify matrix A is Gaussian and B is zeros, guaranteeing Delta W = 0 at step 0."""
    layer = LoRALinear(in_features=32, out_features=16, r=4, lora_alpha=8.0)

    assert layer.r == 4
    assert layer.scaling == 2.0
    assert layer.lora_A.shape == (4, 32)
    assert layer.lora_B.shape == (16, 4)

    # A should be non-zero (Gaussian)
    assert torch.any(layer.lora_A != 0.0)
    # B should be strictly zeros
    assert torch.all(layer.lora_B == 0.0)
    # Base weight must be frozen
    assert not layer.weight.requires_grad
    # LoRA adapters must require grad
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad


def test_lora_forward_exact_match_at_step_0():
    """At step 0 (before training), LoRA output must be numerically identical to base linear layer."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16)
    lora = LoRALinear.from_linear(base, r=8, lora_alpha=16.0)

    x = torch.randn(4, 32)
    out_base = base(x)
    out_lora = lora(x)

    torch.testing.assert_close(out_base, out_lora, rtol=1e-5, atol=1e-6)


def test_lora_rank_decomposition():
    """Verify Delta W has rank at most r."""
    r = 3
    layer = LoRALinear(in_features=20, out_features=20, r=r)
    # Randomize B so Delta W is non-zero
    nn.init.normal_(layer.lora_B)

    delta_w = (layer.lora_B @ layer.lora_A) * layer.scaling
    # Compute rank via SVD
    s = torch.linalg.svdvals(delta_w)
    effective_rank = (s > 1e-5).sum().item()
    assert effective_rank <= r


def test_lora_merge_and_unmerge_weights():
    """Verify weight merging produces identical outputs and unmerging restores original weights."""
    torch.manual_seed(42)
    layer = LoRALinear(in_features=16, out_features=8, r=4, lora_alpha=8.0)
    # Give B non-zero values
    nn.init.normal_(layer.lora_B)

    x = torch.randn(3, 16)
    out_unmerged = layer(x)

    w_before_merge = layer.weight.clone()
    assert not layer.merged

    # Merge
    layer.merge_weights()
    assert layer.merged
    out_merged = layer(x)

    # Merged output must equal unmerged output
    torch.testing.assert_close(out_unmerged, out_merged, rtol=1e-5, atol=1e-6)

    # Unmerge
    layer.unmerge_weights()
    assert not layer.merged
    torch.testing.assert_close(layer.weight, w_before_merge, rtol=1e-5, atol=1e-6)


def test_lora_gradient_isolation():
    """Verify that only LoRA parameters receive gradients during backward pass, base weights stay untouched."""
    layer = LoRALinear(in_features=16, out_features=8, r=4)
    x = torch.randn(2, 16)
    loss = layer(x).sum()
    loss.backward()

    assert layer.weight.grad is None
    if layer.bias is not None:
        assert layer.bias.grad is None
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None


def test_lora_scaling_alpha():
    """Verify that lora_alpha correctly scales the low-rank update."""
    layer1 = LoRALinear(in_features=10, out_features=10, r=4, lora_alpha=4.0)
    layer2 = LoRALinear(in_features=10, out_features=10, r=4, lora_alpha=8.0)

    assert layer1.scaling == 1.0
    assert layer2.scaling == 2.0


def test_lora_rank_zero():
    """When rank r=0, layer should act as standard linear without adapter matrices."""
    layer = LoRALinear(in_features=12, out_features=6, r=0)
    assert layer.lora_A is None
    assert layer.lora_B is None
    x = torch.randn(2, 12)
    out = layer(x)
    assert out.shape == (2, 6)


def test_lora_inject_layers():
    """Verify injection into ToyCausalLM replaces target modules and calculates parameters correctly."""
    model = ToyCausalLM(vocab_size=64, d_model=32, num_layers=2, num_heads=2)
    res = inject_lora_layers(model, target_names=["q_proj", "v_proj"], r=4, lora_alpha=8.0)

    assert res["injected_count"] == 4  # 2 layers * 2 projections
    assert res["trainable_pct"] < 5.0  # Trainable params should be a small fraction

    # Check that injected modules are LoRALinear
    for name, module in model.named_modules():
        if ("q_proj" in name or "v_proj" in name) and not name.endswith(".dropout"):
            assert isinstance(module, LoRALinear)


def test_count_trainable_parameters():
    """Test parameter counter utility."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Linear(10, 5),
    )
    for p in model[0].parameters():
        p.requires_grad = False

    stats = count_trainable_parameters(model)
    assert stats["frozen_params"] == 110  # 10*10 + 10
    assert stats["trainable_params"] == 55  # 10*5 + 5
    assert 30.0 < stats["trainable_pct"] < 35.0


def test_lora_extra_repr():
    """Verify extra_repr includes configuration details."""
    layer = LoRALinear(in_features=8, out_features=4, r=2, lora_alpha=4.0)
    rep = layer.extra_repr()
    assert "r=2" in rep
    assert "lora_alpha=4.0" in rep
    assert "scaling=2.0000" in rep


def test_lora_linear_without_bias():
    """Verify LoRALinear operates correctly with bias=False."""
    layer = LoRALinear(in_features=16, out_features=8, r=4, bias=False)
    assert layer.bias is None
    x = torch.randn(2, 16)
    out = layer(x)
    assert out.shape == (2, 8)


def test_lora_double_merge_is_idempotent():
    """Verify calling merge_weights() twice doesn't double-add delta W."""
    layer = LoRALinear(in_features=10, out_features=5, r=2)
    nn.init.normal_(layer.lora_B)

    layer.merge_weights()
    w_merged_once = layer.weight.clone()

    # Second merge should be a no-op
    layer.merge_weights()
    torch.testing.assert_close(layer.weight, w_merged_once)


def test_lora_unmerge_without_merge_is_noop():
    """Verify calling unmerge_weights() when not merged is a safe no-op."""
    layer = LoRALinear(in_features=10, out_features=5, r=2)
    w_initial = layer.weight.clone()
    layer.unmerge_weights()
    torch.testing.assert_close(layer.weight, w_initial)


def test_lora_save_and_load_adapter_weights():
    """Verify exporting and reloading LoRA adapter state dict into a new model."""
    layer1 = LoRALinear(in_features=16, out_features=8, r=4)
    nn.init.normal_(layer1.lora_A)
    nn.init.normal_(layer1.lora_B)

    # Extract adapter state dict
    adapter_state = {k: v for k, v in layer1.state_dict().items() if "lora_" in k}

    # Load into fresh layer
    layer2 = LoRALinear(in_features=16, out_features=8, r=4)
    layer2.load_state_dict(adapter_state, strict=False)

    torch.testing.assert_close(layer1.lora_A, layer2.lora_A)
    torch.testing.assert_close(layer1.lora_B, layer2.lora_B)
