"""
NormalFloat 4 (NF4) Quantization Logic (Dettmers et al., NeurIPS 2023).
"""

from typing import Tuple
import torch

# The 16 exact theoretical quantiles of standard normal distribution N(0, 1) normalized to [-1, 1].
NF4_POINTS = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=torch.float32)

NF4_QUANTILE_TABLE = NF4_POINTS


def quantize_nf4(
    x: torch.Tensor,
    block_size: int = 64,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """
    Quantizes a floating point tensor into 4-bit NormalFloat (NF4) codebook indices.
    
    Returns:
        (indices, block_scales, original_shape)
    """
    orig_shape = x.shape
    device = x.device
    table = NF4_POINTS.to(device)

    flat_x = x.reshape(-1).float()
    numel = flat_x.numel()

    # Pad if not divisible by block_size
    remainder = numel % block_size
    if remainder != 0:
        pad_len = block_size - remainder
        padded_x = torch.cat([flat_x, torch.zeros(pad_len, device=device)])
    else:
        padded_x = flat_x

    # Reshape into blocks (N_blocks, block_size)
    blocks = padded_x.view(-1, block_size)

    # Compute per-block absolute maximum scale
    block_scales = torch.amax(torch.abs(blocks), dim=-1, keepdim=True).clamp(min=eps)

    # Normalize blocks to [-1, 1]
    norm_blocks = blocks / block_scales  # (N_blocks, block_size)

    # Vectorized nearest-neighbor search in NF4 codebook
    # norm_blocks: (N_blocks, block_size, 1)
    # table: (1, 1, 16)
    diff = torch.abs(norm_blocks.unsqueeze(-1) - table.view(1, 1, 16))
    indices = torch.argmin(diff, dim=-1).to(torch.uint8)  # (N_blocks, block_size)

    return indices, block_scales.squeeze(-1), orig_shape


def dequantize_nf4(
    indices: torch.Tensor,
    block_scales: torch.Tensor,
    orig_shape: Tuple[int, ...],
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantizes NF4 indices back to floating point representation using block scales.
    """
    device = indices.device
    table = NF4_POINTS.to(device)

    # Lookup float value in table
    # indices: (N_blocks, block_size)
    float_blocks = table[indices.long()]  # (N_blocks, block_size)

    # Multiply by scale: (N_blocks, block_size) * (N_blocks, 1)
    dequant_blocks = float_blocks * block_scales.unsqueeze(-1)

    flat_dequant = dequant_blocks.reshape(-1)
    numel = 1
    for d in orig_shape:
        numel *= d

    return flat_dequant[:numel].view(orig_shape).to(target_dtype)
