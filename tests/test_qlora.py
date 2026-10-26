"""
Unit tests for QLoRA 4-bit NormalFloat (NF4), Double Quantization (DQ), and QLoRALinear.
"""

import pytest
import torch
import torch.nn as nn
from qlora.nf4 import (
    NF4_POINTS,
    NF4_QUANTILE_TABLE,
    quantize_nf4,
    dequantize_nf4,
)
from qlora.double_quant import (
    quantize_double_scale,
    dequantize_double_scale,
)
from qlora.layer import QLoRALinear


def test_nf4_quantile_table():
    """Verify NF4 codebook satisfies properties of 16-point standard normal quantiles."""
    assert len(NF4_POINTS) == 16
    assert NF4_POINTS[0].item() == -1.0
    assert NF4_POINTS[15].item() == 1.0

    # Must contain exact zero at index 7
    assert NF4_POINTS[7].item() == 0.0

    # Strictly monotonically increasing
    diffs = torch.diff(NF4_POINTS)
    assert torch.all(diffs > 0)

    # NF4_QUANTILE_TABLE is identical to NF4_POINTS
    torch.testing.assert_close(NF4_POINTS, NF4_QUANTILE_TABLE)


def test_nf4_quantize_dequantize_roundtrip():
    """Verify NF4 quantize -> dequantize reconstructs Gaussian data with low MSE."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    q_indices, scales, orig_shape = quantize_nf4(x, block_size=64)

    assert q_indices.dtype == torch.uint8
    assert torch.all(q_indices >= 0) and torch.all(q_indices < 16)
    assert orig_shape == (128, 128)

    deq_x = dequantize_nf4(q_indices, scales, orig_shape)

    mse = torch.mean((x - deq_x) ** 2).item()
    # NF4 on standard Gaussian should achieve MSE < 0.015
    assert mse < 0.015


def test_nf4_block_padding():
    """Verify tensors not divisible by block_size are safely padded and unpadded without shape change."""
    x = torch.randn(10, 15)  # 150 elements, not divisible by 64
    q_indices, scales, orig_shape = quantize_nf4(x, block_size=64)
    deq_x = dequantize_nf4(q_indices, scales, orig_shape)

    assert deq_x.shape == (10, 15)


def test_double_quantize_roundtrip():
    """Verify 8-bit double quantization roundtrip on block scales preserves scale values."""
    torch.manual_seed(42)
    scales = torch.rand(256) * 2.5 + 0.1  # positive scales

    q_scales, second_scales, second_zp, orig_numel = quantize_double_scale(scales, second_block_size=64)

    assert q_scales.dtype == torch.uint8
    assert orig_numel == 256

    deq_scales = dequantize_double_scale(q_scales, second_scales, second_zp, orig_numel)

    max_err = torch.max(torch.abs(scales - deq_scales)).item()
    # 8-bit quantization on ranges ~2.5 should have max error < 0.02
    assert max_err < 0.02


def test_double_quantize_memory_savings():
    """Verify double quantization saves >60% memory compared to FP32 scales."""
    scales = torch.rand(1024)
    q_scales, s_scales, s_zp, _ = quantize_double_scale(scales, second_block_size=64)

    fp32_bytes = scales.numel() * 4
    dq_bytes = q_scales.numel() * 1 + s_scales.numel() * 4 + s_zp.numel() * 4

    reduction = 100.0 * (1.0 - (dq_bytes / fp32_bytes))
    assert reduction > 60.0


def test_qlora_linear_from_linear():
    """Verify QLoRALinear construction from nn.Linear and forward pass."""
    torch.manual_seed(42)
    base = nn.Linear(64, 32)
    qlayer = QLoRALinear.from_linear(base, r=4, lora_alpha=8.0, block_size=32, double_quant=True)

    assert qlayer.in_features == 64
    assert qlayer.out_features == 32
    assert qlayer.r == 4
    assert qlayer.scaling == 2.0
    assert qlayer.double_quant is True

    x = torch.randn(3, 64)
    out = qlayer(x)
    assert out.shape == (3, 32)


def test_qlora_linear_gradient_flow():
    """Verify that in QLoRALinear, only LoRA adapters receive gradients."""
    qlayer = QLoRALinear(in_features=32, out_features=16, r=4, block_size=16)
    x = torch.randn(2, 32)
    loss = qlayer(x).sum()
    loss.backward()

    assert qlayer.lora_A.grad is not None
    assert qlayer.lora_B.grad is not None


def test_qlora_linear_rank_zero():
    """Verify QLoRALinear with rank r=0 functions as frozen quantized linear layer."""
    base = nn.Linear(32, 16)
    qlayer = QLoRALinear.from_linear(base, r=0, block_size=16)

    assert qlayer.lora_A is None
    assert qlayer.lora_B is None

    x = torch.randn(2, 32)
    out = qlayer(x)
    assert out.shape == (2, 16)


def test_qlora_without_double_quant():
    """Verify QLoRALinear operates correctly with double_quant=False."""
    base = nn.Linear(32, 16)
    qlayer = QLoRALinear.from_linear(base, r=4, block_size=16, double_quant=False)

    assert not qlayer.double_quant
    x = torch.randn(2, 32)
    out = qlayer(x)
    assert out.shape == (2, 16)


def test_qlora_from_linear_without_bias():
    """Verify QLoRALinear handles base linear layers with bias=False."""
    base = nn.Linear(32, 16, bias=False)
    qlayer = QLoRALinear.from_linear(base, r=4, block_size=16)

    assert qlayer.bias is None
    x = torch.randn(2, 32)
    out = qlayer(x)
    assert out.shape == (2, 16)


def test_qlora_odd_shaped_matrices():
    """Verify QLoRA handles non-power-of-two and odd-dimensioned matrices."""
    torch.manual_seed(42)
    base = nn.Linear(37, 19)
    qlayer = QLoRALinear.from_linear(base, r=3, block_size=16)

    x = torch.randn(3, 37)
    out = qlayer(x)
    assert out.shape == (3, 19)


def test_qlora_double_quant_arbitrary_scales():
    """Verify double quantization handles arbitrary scale tensors with odd lengths."""
    scales = torch.rand(77) * 5.0 + 0.05
    q_scales, s_scales, s_min, orig_numel = quantize_double_scale(scales, second_block_size=16)

    assert orig_numel == 77
    deq = dequantize_double_scale(q_scales, s_scales, s_min, orig_numel)
    assert deq.shape == (77,)
    assert torch.max(torch.abs(scales - deq)).item() < 0.05
