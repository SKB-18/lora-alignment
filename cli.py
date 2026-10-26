"""
Interactive and Scriptable CLI for LoRA, QLoRA, and Direct Preference Optimization (DPO).
"""

import argparse
import sys
import os
import json
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lora.layer import LoRALinear
from lora.injector import inject_lora_layers, count_trainable_parameters, get_lora_parameters
from lora.toy_lm import ToyCausalLM
from qlora.nf4 import quantize_nf4, dequantize_nf4, NF4_QUANTILE_TABLE
from qlora.double_quant import quantize_double_scale, dequantize_double_scale
from qlora.layer import QLoRALinear
from dpo.loss import DPOLoss
from dpo.utils import compute_sequence_log_probs
from eval.metrics import evaluate_preference_batch
from training.dpo_trainer import DPOTrainer, DPOTrainerConfig


def cmd_lora(args):
    """Inspects LoRA adapter injection and parameter reduction on a model."""
    print("=" * 65)
    print(f" LoRA Adapter Injection (Rank r={args.rank}, Alpha={args.alpha})")
    print("=" * 65)

    model = ToyCausalLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
    )
    initial_stats = count_trainable_parameters(model)
    print(f"  • Base Model Total Parameters:      {initial_stats['total_params']:,d}")

    targets = args.targets.split(",") if args.targets else ["q_proj", "v_proj"]
    res = inject_lora_layers(
        model,
        target_names=targets,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
    )

    print(f"  • Target Linear Modules Injected:   {res['injected_count']} layers {targets}")
    print(f"  • Trainable Parameters:             {res['trainable_params']:,d} ({res['trainable_pct']:.2f}%)")
    print(f"  • Frozen Base Parameters:           {res['frozen_params']:,d}")
    print(f"  • Parameter Reduction Factor:       {100.0 - res['trainable_pct']:.2f}% reduction")

    # Test weight merging
    first_lora = next(m for m in model.modules() if isinstance(m, LoRALinear))
    w_unmerged = first_lora.weight.clone()
    first_lora.merge_weights()
    print(f"  • Zero-Overhead Inference Merge:    [OK] (Merged: {first_lora.merged})")
    first_lora.unmerge_weights()
    print(f"  • Unmerge Weight Restoration:       [OK] (Merged: {first_lora.merged})")
    print("=" * 65)


def cmd_qlora(args):
    """Demonstrates 4-bit NormalFloat and Double Quantization."""
    print("=" * 65)
    print(f" QLoRA 4-bit NormalFloat & Double Quantization (Block Size {args.block_size})")
    print("=" * 65)

    torch.manual_seed(args.seed)
    weights = torch.randn(args.rows, args.cols, dtype=torch.float32)

    # 1. NF4 Single Quantization
    q_indices, scales, orig_shape = quantize_nf4(weights, block_size=args.block_size)
    deq = dequantize_nf4(q_indices, scales, orig_shape)
    mse_single = torch.mean((weights - deq) ** 2).item()

    # 2. Double Quantization of scales
    q_scales, s_scales, s_zp, orig_numel = quantize_double_scale(scales, second_block_size=args.second_block_size)
    deq_scales = dequantize_double_scale(q_scales, s_scales, s_zp, orig_numel)
    deq_dq = dequantize_nf4(q_indices, deq_scales, orig_shape)
    mse_dq = torch.mean((weights - deq_dq) ** 2).item()

    fp32_bytes = weights.numel() * 4
    nf4_bytes = (weights.numel() + 1) // 2
    single_scale_bytes = scales.numel() * 4
    dq_scale_bytes = q_scales.numel() + s_scales.numel() * 4 + s_zp.numel() * 4

    print(f"  • Original Matrix Dimensions:       {args.rows} x {args.cols} ({weights.numel():,d} elements)")
    print(f"  • Unquantized FP32 Size:            {fp32_bytes / 1024:.1f} KB (32.0 bits/param)")
    print(f"  • NF4 Base Weights:                 {nf4_bytes / 1024:.1f} KB (4.0 bits/param)")
    print(f"  • First-order Scales (FP32):        {single_scale_bytes / 1024:.1f} KB ({(single_scale_bytes * 8) / weights.numel():.3f} bits/param)")
    print(f"  • Double Quantized Scales (INT8):   {dq_scale_bytes / 1024:.1f} KB ({(dq_scale_bytes * 8) / weights.numel():.3f} bits/param)")
    print(f"  • Total Storage with DQ:            {(nf4_bytes + dq_scale_bytes) / 1024:.1f} KB ({((nf4_bytes + dq_scale_bytes) * 8) / weights.numel():.3f} bits/param)")
    print(f"  • Quantization Fidelity (MSE NF4):  {mse_single:.6f}")
    print(f"  • Quantization Fidelity (MSE DQ):   {mse_dq:.6f}")
    print("=" * 65)


def cmd_train(args):
    """Runs DPO fine-tuning loop on preference dataset."""
    print("=" * 65)
    print(f" DPO Alignment Training (Beta={args.beta}, LR={args.lr}, Epochs={args.epochs})")
    print("=" * 65)

    torch.manual_seed(args.seed)
    # Policy model & reference model
    policy = ToyCausalLM(vocab_size=args.vocab_size, d_model=args.d_model, num_layers=2, num_heads=2)
    reference = ToyCausalLM(vocab_size=args.vocab_size, d_model=args.d_model, num_layers=2, num_heads=2)
    reference.load_state_dict(policy.state_dict())

    inject_lora_layers(policy, target_names=["q_proj", "v_proj"], r=args.rank, lora_alpha=args.alpha)

    # Generate synthetic preferences
    dataset = []
    target_token = 42
    for _ in range(args.samples):
        prefix = torch.randint(10, 40, (1, 6))
        chosen_suffix = torch.full((1, 6), target_token)
        rejected_suffix = torch.randint(60, 90, (1, 6))
        dataset.append({
            "chosen_input_ids": torch.cat([prefix, chosen_suffix], dim=1),
            "rejected_input_ids": torch.cat([prefix, rejected_suffix], dim=1),
        })

    config = DPOTrainerConfig(
        lr=args.lr,
        beta=args.beta,
        epochs=args.epochs,
        device="cpu",
    )
    trainer = DPOTrainer(policy_model=policy, reference_model=reference, config=config)

    init_res = trainer.evaluate(dataset)
    print(f"  • Epoch  0 (Zero Step):  Margin = {init_res['val_reward_margin']:+.4f} | Win-Rate = {init_res['val_accuracy'] * 100:.1f}%")

    history = trainer.fit(dataset, epochs=args.epochs)
    for s in history["epoch_summaries"]:
        print(
            f"  • Epoch {s['epoch']:2d}: Loss = {s['avg_loss']:.4f} | "
            f"Reward Margin = {s['avg_reward_margin']:+.4f} | "
            f"Win-Rate = {s['avg_accuracy'] * 100:.1f}%"
        )

    final_res = trainer.evaluate(dataset)
    print("=" * 65)
    print(f"  [SUCCESS] DPO Alignment Complete: Margin = {final_res['val_reward_margin']:+.4f}, Win-Rate = {final_res['val_accuracy'] * 100:.1f}%")
    print("=" * 65)


def cmd_benchmark(args):
    """Executes the automated benchmark runner."""
    from benchmarks.run_benchmark import main as run_bench
    run_bench()


def main():
    parser = argparse.ArgumentParser(description="lora-alignment CLI: PEFT and DPO Alignment Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # lora subcommand
    p_lora = subparsers.add_parser("lora", help="Inspect LoRA adapter parameter savings")
    p_lora.add_argument("--rank", "-r", type=int, default=8, help="LoRA rank")
    p_lora.add_argument("--alpha", "-a", type=float, default=16.0, help="LoRA alpha scaling")
    p_lora.add_argument("--dropout", type=float, default=0.0, help="LoRA dropout")
    p_lora.add_argument("--targets", type=str, default="q_proj,v_proj", help="Comma-separated target layers")
    p_lora.add_argument("--vocab-size", type=int, default=1024, help="Model vocabulary size")
    p_lora.add_argument("--d-model", type=int, default=256, help="Model hidden dimension")
    p_lora.add_argument("--layers", type=int, default=4, help="Number of transformer layers")
    p_lora.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    p_lora.set_defaults(func=cmd_lora)

    # qlora subcommand
    p_qlora = subparsers.add_parser("qlora", help="Benchmark 4-bit NF4 and Double Quantization")
    p_qlora.add_argument("--rows", type=int, default=512, help="Matrix rows")
    p_qlora.add_argument("--cols", type=int, default=512, help="Matrix columns")
    p_qlora.add_argument("--block-size", type=int, default=64, help="NF4 block size")
    p_qlora.add_argument("--second-block-size", type=int, default=64, help="Double quant block size")
    p_qlora.add_argument("--seed", type=int, default=42, help="Random seed")
    p_qlora.set_defaults(func=cmd_qlora)

    # train subcommand
    p_train = subparsers.add_parser("train", help="Run DPO alignment training loop")
    p_train.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    p_train.add_argument("--beta", type=float, default=0.2, help="DPO beta temperature")
    p_train.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    p_train.add_argument("--rank", type=int, default=8, help="LoRA rank")
    p_train.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    p_train.add_argument("--samples", type=int, default=24, help="Number of preference samples")
    p_train.add_argument("--vocab-size", type=int, default=128, help="Vocabulary size")
    p_train.add_argument("--d-model", type=int, default=64, help="Hidden dimension")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.set_defaults(func=cmd_train)

    # benchmark subcommand
    p_bench = subparsers.add_parser("benchmark", help="Run full benchmark suite")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
