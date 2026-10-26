"""
Model Injector: Injects LoRALinear layers into neural networks and configures gradient tracking.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from .layer import LoRALinear


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none"):
    """
    Freezes all base model parameters except for LoRA adapter matrices.
    """
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALinear) and m.bias is not None:
                m.bias.requires_grad = True


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Returns only trainable LoRA parameters."""
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def get_parameter_breakdown(model: nn.Module) -> Dict[str, Any]:
    """Returns total vs trainable parameter counts and ratio."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "trainable_pct": trainable_pct,
        "trainable_percent": trainable_pct,
    }


def count_trainable_parameters(model: nn.Module) -> Dict[str, Any]:
    """Alias for get_parameter_breakdown."""
    return get_parameter_breakdown(model)


def inject_lora_layers(
    model: nn.Module,
    target_names: Optional[List[str]] = None,
    target_module_names: Optional[List[str]] = None,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> Dict[str, Any]:
    """
    Recursively scans model modules and replaces target Linear layers with LoRALinear.
    """
    targets = target_names or target_module_names or ["q_proj", "v_proj", "fc1", "fc2"]
    injected_count = 0
    injected_names = []

    def _inject(module: nn.Module, prefix: str = ""):
        nonlocal injected_count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                # Check if this linear layer matches any target substring
                if any(t in full_name for t in targets):
                    lora_layer = LoRALinear.from_linear(
                        child,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    )
                    setattr(module, name, lora_layer)
                    injected_names.append(full_name)
                    injected_count += 1
            else:
                _inject(child, full_name)

    _inject(model)
    mark_only_lora_as_trainable(model)
    breakdown = get_parameter_breakdown(model)

    return {
        "injected_count": injected_count,
        "injected_names": injected_names,
        **breakdown,
    }
