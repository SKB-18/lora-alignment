"""
Comprehensive Empirical Benchmark for LoRA, QLoRA NF4, and DPO Alignment.
"""

import sys
import os
import time
import json
from typing import Dict, Any, List
import torch
import torch.nn as nn

# Reconfigure stdout for Windows console UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add parent path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lora.layer import LoRALinear
from lora.injector import inject_lora_layers, count_trainable_parameters
from lora.toy_lm import ToyCausalLM
from qlora.nf4 import quantize_nf4, dequantize_nf4, NF4_QUANTILE_TABLE
from qlora.double_quant import quantize_double_scale, dequantize_double_scale
from qlora.layer import QLoRALinear
from dpo.loss import DPOLoss
from training.dpo_trainer import DPOTrainer, DPOTrainerConfig


def benchmark_lora_parameter_efficiency() -> Dict[str, Any]:
    """Measures parameter reduction across multiple LoRA ranks."""
    print("=" * 70)
    print(" 1. LoRA Parameter Efficiency Benchmark")
    print("=" * 70)

    ranks = [2, 4, 8, 16, 32, 64]
    results = []

    for r in ranks:
        model = ToyCausalLM(vocab_size=1024, d_model=256, num_layers=4, num_heads=4)
        total_before = sum(p.numel() for p in model.parameters())

        inject_lora_layers(model, target_module_names=["q_proj", "v_proj"], r=r, lora_alpha=r * 2)
        stats = count_trainable_parameters(model)

        # Forward latency test
        x = torch.randint(0, 1024, (8, 32))
        start_t = time.perf_counter()
        for _ in range(50):
            _ = model(x)
        elapsed_ms = (time.perf_counter() - start_t) / 50.0 * 1000.0

        res = {
            "rank": r,
            "total_params": stats["total_params"],
            "trainable_params": stats["trainable_params"],
            "frozen_params": stats["frozen_params"],
            "trainable_pct": stats["trainable_pct"],
            "param_reduction_pct": 100.0 - stats["trainable_pct"],
            "latency_ms": round(elapsed_ms, 2),
        }
        results.append(res)
        print(
            f"  • Rank {r:2d} | Trainable: {res['trainable_params']:>6,d} / {res['total_params']:>8,d} "
            f"({res['trainable_pct']:5.2f}%) | Reduction: {res['param_reduction_pct']:5.2f}% | Latency: {res['latency_ms']:.2f}ms"
        )

    return {"ranks_tested": results}


def benchmark_qlora_nf4_and_double_quant() -> Dict[str, Any]:
    """Measures NF4 quantization fidelity and double quantization savings."""
    print("\n" + "=" * 70)
    print(" 2. QLoRA 4-bit NormalFloat & Double Quantization Benchmark")
    print("=" * 70)

    torch.manual_seed(42)
    # Simulate a 1024 x 1024 linear weight matrix with standard normal distribution
    weights = torch.randn(1024, 1024, dtype=torch.float32)

    # 1. NF4 Single Quantization (block size 64)
    q_indices, scales, orig_shape = quantize_nf4(weights, block_size=64)
    deq_weights = dequantize_nf4(q_indices, scales, orig_shape)
    mse_nf4 = torch.mean((weights - deq_weights) ** 2).item()

    # Memory calculations
    fp32_bytes = weights.numel() * 4
    nf4_weight_bytes = (weights.numel() + 1) // 2  # 4 bits = 0.5 bytes per element
    single_scales_bytes = scales.numel() * 4       # FP32 scale per 64 weights = 0.5 bits/param

    # 2. Double Quantization of scales (8-bit quantization with second-order scale)
    q_scales, s_scales, s_zp, orig_numel = quantize_double_scale(scales, block_size=256)
    deq_scales = dequantize_double_scale(q_scales, s_scales, s_zp, orig_numel)
    deq_dq_weights = dequantize_nf4(q_indices, deq_scales, orig_shape)
    mse_dq = torch.mean((weights - deq_dq_weights) ** 2).item()

    dq_scales_bytes = q_scales.numel() * 1 + s_scales.numel() * 4 + s_zp.numel() * 4

    total_single_bytes = nf4_weight_bytes + single_scales_bytes
    total_dq_bytes = nf4_weight_bytes + dq_scales_bytes

    bits_per_param_single = (total_single_bytes * 8) / weights.numel()
    bits_per_param_dq = (total_dq_bytes * 8) / weights.numel()
    scale_overhead_reduction = 100.0 * (1.0 - (dq_scales_bytes / single_scales_bytes))

    print(f"  • FP32 Matrix Size:            {fp32_bytes / 1024:.1f} KB (32.0 bits/param)")
    print(f"  • NF4 Single Quant Size:       {total_single_bytes / 1024:.1f} KB ({bits_per_param_single:.3f} bits/param)")
    print(f"  • NF4 + Double Quant Size:     {total_dq_bytes / 1024:.1f} KB ({bits_per_param_dq:.3f} bits/param)")
    print(f"  • Scale Footprint Reduction:   {scale_overhead_reduction:.1f}% savings on quantization constants")
    print(f"  • NF4 Single Quant MSE:        {mse_nf4:.6f}")
    print(f"  • NF4 + Double Quant MSE:      {mse_dq:.6f} (negligible difference: {abs(mse_dq - mse_nf4):.7f})")

    return {
        "fp32_kb": fp32_bytes / 1024,
        "nf4_single_kb": total_single_bytes / 1024,
        "nf4_dq_kb": total_dq_bytes / 1024,
        "bits_per_param_dq": bits_per_param_dq,
        "scale_reduction_pct": scale_overhead_reduction,
        "mse_nf4": mse_nf4,
        "mse_dq": mse_dq,
    }


def benchmark_dpo_alignment_trajectory() -> Dict[str, Any]:
    """Trains a LoRA policy model on preference pairs and verifies reward margin expansion."""
    print("\n" + "=" * 70)
    print(" 3. DPO Alignment Trajectory & Reward Margin Benchmark")
    print("=" * 70)

    torch.manual_seed(42)
    # Initialize base model and frozen reference
    base_model = ToyCausalLM(vocab_size=128, d_model=64, num_layers=2, num_heads=2)
    ref_model = ToyCausalLM(vocab_size=128, d_model=64, num_layers=2, num_heads=2)
    ref_model.load_state_dict(base_model.state_dict())

    # Inject LoRA into policy
    inject_lora_layers(base_model, target_module_names=["q_proj", "v_proj"], r=8, lora_alpha=16.0)

    # Synthetic preference dataset: chosen completions contain target token 42 repeatedly
    num_samples = 32
    seq_len = 16
    dataset = []
    for _ in range(num_samples):
        prefix = torch.randint(10, 40, (1, 8))
        chosen_suffix = torch.full((1, 8), 42)
        rejected_suffix = torch.randint(50, 90, (1, 8))

        chosen_ids = torch.cat([prefix, chosen_suffix], dim=1)
        rejected_ids = torch.cat([prefix, rejected_suffix], dim=1)

        dataset.append({
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids,
        })

    config = DPOTrainerConfig(
        lr=2e-3,
        beta=0.2,
        epochs=5,
        device="cpu",
    )
    trainer = DPOTrainer(
        policy_model=base_model,
        reference_model=ref_model,
        config=config,
    )

    # Initial zero-step evaluation
    init_eval = trainer.evaluate(dataset)
    print(f"  • Step 0 (Pre-Alignment):  Margin = {init_eval['val_reward_margin']:+.4f} | Accuracy = {init_eval['val_accuracy'] * 100:.1f}%")

    history = trainer.fit(dataset, epochs=5)

    for summary in history["epoch_summaries"]:
        print(
            f"  • Epoch {summary['epoch']}: Loss = {summary['avg_loss']:.4f} | "
            f"Margin = {summary['avg_reward_margin']:+.4f} | "
            f"Accuracy = {summary['avg_accuracy'] * 100:.1f}%"
        )

    final_eval = trainer.evaluate(dataset)
    margin_expansion = final_eval["val_reward_margin"] - init_eval["val_reward_margin"]
    print(
        f"\n  [OK] Final Reward Margin: {final_eval['val_reward_margin']:+.4f} "
        f"(+{margin_expansion:.4f} expansion) | Final Win-Rate: {final_eval['val_accuracy'] * 100:.1f}%"
    )

    return {
        "initial_margin": init_eval["val_reward_margin"],
        "final_margin": final_eval["val_reward_margin"],
        "margin_expansion": margin_expansion,
        "final_accuracy": final_eval["val_accuracy"],
    }


def main():
    print("\n" + "=" * 70)
    print(" Running Full lora-alignment Benchmark Suite")
    print("=" * 70)

    t0 = time.time()
    lora_res = benchmark_lora_parameter_efficiency()
    qlora_res = benchmark_qlora_nf4_and_double_quant()
    dpo_res = benchmark_dpo_alignment_trajectory()
    total_time = time.time() - t0

    print("\n" + "=" * 70)
    print(f" ALL BENCHMARKS COMPLETED IN {total_time:.2f}s")
    print("=" * 70)

    summary = {
        "timestamp": time.time(),
        "total_time_seconds": round(total_time, 2),
        "lora": lora_res,
        "qlora": qlora_res,
        "dpo": dpo_res,
    }

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f" Saved full benchmark metrics to {out_path}\n")


if __name__ == "__main__":
    main()
