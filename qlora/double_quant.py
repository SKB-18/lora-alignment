"""
Double Quantization (DQ) for Quantization Constants (Dettmers et al., 2023).
Reduces the memory footprint of FP32 block scales by quantizing them to 8-bit integers.
"""

from typing import Tuple, Optional
import torch


def quantize_double_scale(
    scales: torch.Tensor,
    second_block_size: int = 64,
    block_size: Optional[int] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Performs 8-bit asymmetric quantization on first-order scales.
    
    q = clamp(round((scales - min_val) / scale), 0, 255)
    dequant = q * scale + min_val
    
    Returns:
        (quantized_scales, second_scales, second_min_vals, original_numel)
    """
    if block_size is not None:
        second_block_size = block_size

    orig_numel = scales.numel()
    flat = scales.reshape(-1).float()
    device = scales.device

    remainder = orig_numel % second_block_size
    if remainder != 0:
        pad_len = second_block_size - remainder
        padded = torch.cat([flat, torch.zeros(pad_len, device=device)])
    else:
        padded = flat

    blocks = padded.view(-1, second_block_size)
    min_vals = torch.amin(blocks, dim=-1, keepdim=True)
    max_vals = torch.amax(blocks, dim=-1, keepdim=True)

    val_range = torch.clamp(max_vals - min_vals, min=eps)
    second_scales = val_range / 255.0

    # Map to [0, 255]
    q_scales = torch.clamp(torch.round((blocks - min_vals) / second_scales), 0, 255).to(torch.uint8)

    return q_scales, second_scales.squeeze(-1), min_vals.squeeze(-1), orig_numel


def dequantize_double_scale(
    q_scales: torch.Tensor,
    second_scales: torch.Tensor,
    second_zp_or_min: torch.Tensor,
    orig_numel: int,
) -> torch.Tensor:
    """
    Recovers first-order floating-point scales from 8-bit representations.
    """
    # q_scales: (N_blocks, second_block_size)
    # second_scales: (N_blocks,)
    # second_zp_or_min: (N_blocks,)
    if second_scales.dim() == 0:
        second_scales = second_scales.unsqueeze(0)
    if second_zp_or_min.dim() == 0:
        second_zp_or_min = second_zp_or_min.unsqueeze(0)

    dequant = q_scales.float() * second_scales.unsqueeze(-1) + second_zp_or_min.unsqueeze(-1)
    flat = dequant.reshape(-1)
    return flat[:orig_numel]
