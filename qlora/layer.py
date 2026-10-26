"""
QLoRALinear Layer: 4-bit NormalFloat Quantized Base Weights with Trainable LoRA Adapters.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nf4 import quantize_nf4, dequantize_nf4
from .double_quant import quantize_double_scale, dequantize_double_scale


class QLoRALinear(nn.Module):
    """
    Fused layer combining frozen 4-bit NF4 base weights with high-precision LoRA adapters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        block_size: int = 64,
        double_quant: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(r) if r > 0 else 1.0
        self.block_size = block_size
        self.double_quant = double_quant

        # Initialize default base weights
        w_init = torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features))
        indices, scales, _ = quantize_nf4(w_init, block_size=block_size)
        self.register_buffer("qweight", indices)

        if double_quant:
            q_scales, s_scales, s_zp, orig_numel = quantize_double_scale(scales)
            self.register_buffer("block_scales", q_scales)
            self.register_buffer("second_scales", s_scales)
            self.register_buffer("second_zp", s_zp)
            self.orig_scales_numel = orig_numel
        else:
            self.register_buffer("block_scales", scales)
            self.register_buffer("second_scales", torch.zeros(1, dtype=torch.float32))
            self.register_buffer("second_zp", torch.zeros(1, dtype=torch.float32))
            self.orig_scales_numel = 0

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable LoRA adapter parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, r))
            self.reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_parameters(self):
        if self.r > 0:
            nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(self.r))
            nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        base_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        block_size: int = 64,
        double_quant: bool = True,
    ) -> "QLoRALinear":
        """Builds a QLoRALinear from an existing float nn.Linear layer."""
        layer = cls(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            block_size=block_size,
            double_quant=double_quant,
            bias=base_linear.bias is not None,
        )

        w = base_linear.weight.data.clone().float()
        indices, scales, _ = quantize_nf4(w, block_size=block_size)
        layer.qweight = indices

        if double_quant:
            q_scales, s_scales, s_zp, orig_numel = quantize_double_scale(scales)
            layer.block_scales = q_scales
            layer.second_scales = s_scales
            layer.second_zp = s_zp
            layer.orig_scales_numel = orig_numel
        else:
            layer.block_scales = scales

        if base_linear.bias is not None:
            layer.bias.data.copy_(base_linear.bias.data)

        return layer

    def get_dequantized_weight(self, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantizes the NF4 base weight back to floating point."""
        if self.double_quant:
            scales = dequantize_double_scale(
                self.block_scales,
                self.second_scales,
                self.second_zp,
                self.orig_scales_numel,
            )
        else:
            scales = self.block_scales

        return dequantize_nf4(
            self.qweight,
            scales,
            (self.out_features, self.in_features),
            target_dtype=target_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = self.get_dequantized_weight(target_dtype=x.dtype)
        base_out = F.linear(x, base_weight, self.bias)

        if self.r > 0:
            lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
            return base_out + lora_out
        return base_out
