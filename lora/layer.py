"""
LoRALinear Layer: Parameter-Efficient Low-Rank Adaptation.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Implements LoRA for Linear layers (Hu et al., 2021).
    
    Forward pass:
        h = W_0 * x + (alpha / r) * (B * A * x)
        
    Properties:
    - Base weight W_0 is strictly frozen.
    - Matrix A is initialized via Gaussian N(0, 1/r).
    - Matrix B is initialized to zeros, ensuring Delta W = 0 at initialization.
    - Zero inference latency overhead via merge_weights().
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(r) if r > 0 else 1.0

        # Base linear layer weights (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # LoRA low-rank adapter matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, r))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.dropout = nn.Identity()

        self.merged = False

    def reset_parameters(self):
        """Initializes A to Gaussian and B to 0 so Delta W = 0 at step 0."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.r > 0:
            # Gaussian initialization for A
            nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(self.r))
            # Zero initialization for B
            nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        base_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> "LoRALinear":
        """Factory method constructing a LoRALinear layer wrapping an existing nn.Linear."""
        layer = cls(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=base_linear.bias is not None,
        )
        layer.weight.data.copy_(base_linear.weight.data)
        if base_linear.bias is not None:
            layer.bias.data.copy_(base_linear.bias.data)
        return layer

    def merge_weights(self):
        """Merges low-rank weights into base weights for zero-overhead inference."""
        if self.r > 0 and not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data += delta_w
            self.merged = True

    def unmerge_weights(self):
        """Recovers base weights from merged state."""
        if self.r > 0 and self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.r == 0:
            return F.linear(x, self.weight, self.bias)

        base_out = F.linear(x, self.weight, self.bias)
        # Low rank forward: (x * A^T) * B^T * scaling
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, scaling={self.scaling:.4f}, "
            f"merged={self.merged}"
        )
