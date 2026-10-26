# LoRA & DPO Alignment Lab (`lora-alignment`)

[![Tests](https://img.shields.io/badge/tests-46%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Production-grade parameter-efficient fine-tuning (PEFT) and preference alignment framework built from first principles in PyTorch. Implements **Low-Rank Adaptation (LoRA)**, **4-bit NormalFloat Quantized LoRA (QLoRA) with Double Quantization**, and **Direct Preference Optimization (DPO)** without reinforcement learning or explicit reward modeling.

---

## 🏛️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │               Base Linear Layer              │
                    │         W_0 in R^{d_out x d_in} (Frozen)      │
                    └──────────────────────┬───────────────────────┘
                                           │
               Input x                     │                   Output h
           ─────────────┬──────────────────┴────────────────────────────►
                        │                                  ▲
                        ▼                                  │
                 ┌──────────────┐                          │
                 │   Matrix A   │                          │
                 │  R^{r x d_in}│ (Gaussian N(0, 1/r))     │
                 └──────┬───────┘                          │
                        │                                  │
                        ▼                                  │
                 ┌──────────────┐                          │
                 │   Matrix B   │                          │
                 │ R^{d_out x r}│ (Zero Initialized)       │
                 └──────┬───────┘                          │
                        │                                  │
                        ▼                                  │
                 Scale: (alpha / r) ───────────────────────┘
```

---

## 🔬 Mathematical Formulations

### 1. Low-Rank Adaptation (LoRA)
Rather than updating the full weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, LoRA constrains the update by parameterizing the delta as low-rank matrices $B$ and $A$:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$
where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized from $\mathcal{N}(0, 1/r)$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to zero.

At step 0:
$$\Delta W = B \cdot A = 0 \implies h_{\text{step 0}} = W_0 x$$

#### Zero-Latency Inference Merging
For deployment without inference overhead, low-rank matrices are folded into base weights:
$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$

---

### 2. QLoRA 4-bit NormalFloat (NF4) & Double Quantization
NormalFloat (NF4) builds an information-theoretically optimal quantile codebook $q_i$ for zero-mean standard normal distributions $\mathcal{N}(0, 1)$:
$$q_i = \frac{1}{2} \left( Q_X\left(\frac{i}{2^k}\right) + Q_X\left(\frac{i+1}{2^k}\right) \right)$$
normalized symmetrically into $[-1.0, 1.0]$ with an exact zero at bin 7.

#### Double Quantization (DQ)
Quantization block scales $c_1^{\text{FP32}}$ are themselves quantized into 8-bit integers using second-order scales $c_2^{\text{FP32}}$:
$$c_1^{\text{INT8}} = \text{clamp}\left( \left\lfloor \frac{c_1^{\text{FP32}} - \min(c_1)}{c_2} \right\rceil, 0, 255 \right)$$
This reduces scale memory overhead from **0.50 bits/param** down to **0.127 bits/param** (74.2% scale memory savings).

---

### 3. Direct Preference Optimization (DPO)
DPO reparameterizes the Bradley-Terry preference model by expressing implicit rewards directly through policy probability ratios, eliminating explicit reward models and PPO actor-critic loops:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right) \right]$$

Implicit reward definition:
$$\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

---

## 📊 Benchmark Results

Empirical results measured on standard hardware via `benchmarks/run_benchmark.py`:

### 1. LoRA Parameter Efficiency vs. Base Model

| LoRA Rank ($r$) | Base Parameters | Trainable LoRA Params | Trainable % | Parameter Reduction | Forward Latency |
|:---------------:|:---------------:|:---------------------:|:-----------:|:-------------------:|:---------------:|
| **$r=2$**       | 2,645,504       | 8,192                 | **0.31%**   | **99.69%**          | 15.68 ms        |
| **$r=4$**       | 2,645,504       | 16,384                | **0.62%**   | **99.38%**          | 10.35 ms        |
| **$r=8$**       | 2,645,504       | 32,768                | **1.22%**   | **98.78%**          | 9.07 ms         |
| **$r=16$**      | 2,645,504       | 65,536                | **2.42%**   | **97.58%**          | 9.10 ms         |
| **$r=32$**      | 2,645,504       | 131,072               | **4.72%**   | **95.28%**          | 8.80 ms         |
| **$r=64$**      | 2,645,504       | 262,144               | **9.02%**   | **90.98%**          | 9.03 ms         |

### 2. Quantization Memory & Fidelity

| Format | Bits / Parameter | Memory (1024x1024) | Scale Savings | Reconstruction MSE |
|:-------|:----------------:|:------------------:|:-------------:|:------------------:|
| **FP32** | 32.000 bits | 4,096.0 KB | Baseline | 0.000000 |
| **NF4 Single Quant** | 4.500 bits | 576.0 KB | 0.0% | 0.008473 |
| **NF4 + Double Quant** | **4.129 bits** | **528.5 KB** | **74.2%** | 0.031403 |

### 3. DPO Alignment Trajectory

```
  Step 0 (Zero Step):  Margin = +0.0000 | Win-Rate =   0.0%
  Epoch 1: Loss 0.2247 | Margin = +2.0351 | Win-Rate =  96.9%
  Epoch 2: Loss 0.0078 | Margin = +4.9129 | Win-Rate = 100.0%
  Epoch 3: Loss 0.0043 | Margin = +5.4769 | Win-Rate = 100.0%
  Epoch 4: Loss 0.0034 | Margin = +5.7282 | Win-Rate = 100.0%
  Epoch 5: Loss 0.0028 | Margin = +5.9015 | Win-Rate = 100.0%
  [OK] Final Reward Margin: +5.9756 (+5.9756 expansion) | Win-Rate: 100.0%
```

---

## 🚀 Quickstart & CLI

```bash
# Clone and enter project
git clone https://github.com/SKB-18/lora-alignment.git
cd lora-alignment

# Install dependencies
pip install -r requirements.txt
```

### Inspect LoRA Parameter Savings
```bash
python cli.py lora --rank 8 --alpha 16.0 --targets q_proj,v_proj
```

### Benchmark NF4 & Double Quantization
```bash
python cli.py qlora --rows 512 --cols 512 --block-size 64
```

### Run DPO Alignment Fine-Tuning
```bash
python cli.py train --lr 0.002 --beta 0.2 --epochs 5 --rank 8
```

### Execute Comprehensive Empirical Benchmarks
```bash
python cli.py benchmark
```

---

## 🧪 Automated Test Suite (46 Tests)

Run complete verification:
```bash
pytest tests/ -v
```

```
tests/test_dpo.py::test_dpo_loss_formula_symmetric PASSED                [  2%]
tests/test_dpo.py::test_dpo_loss_decreases_with_preference_margin PASSED [  4%]
tests/test_dpo.py::test_dpo_gradient_direction PASSED                    [  6%]
tests/test_dpo.py::test_dpo_label_smoothing PASSED                       [  8%]
tests/test_dpo.py::test_dpo_reference_free PASSED                        [ 10%]
tests/test_dpo.py::test_dpo_invalid_beta PASSED                          [ 13%]
tests/test_dpo.py::test_eval_metrics_helpers PASSED                      [ 15%]
tests/test_dpo.py::test_dpo_loss_extreme_logits_numerical_stability PASSED [ 17%]
tests/test_lora.py::test_lora_initialization PASSED                      [ 19%]
tests/test_lora.py::test_lora_forward_exact_match_at_step_0 PASSED       [ 21%]
tests/test_lora.py::test_lora_rank_decomposition PASSED                  [ 23%]
tests/test_lora.py::test_lora_merge_and_unmerge_weights PASSED           [ 26%]
tests/test_lora.py::test_lora_gradient_isolation PASSED                  [ 28%]
tests/test_lora.py::test_lora_scaling_alpha PASSED                       [ 30%]
tests/test_lora.py::test_lora_rank_zero PASSED                           [ 32%]
tests/test_lora.py::test_lora_inject_layers PASSED                       [ 34%]
tests/test_lora.py::test_count_trainable_parameters PASSED               [ 36%]
tests/test_lora.py::test_lora_extra_repr PASSED                          [ 39%]
tests/test_lora.py::test_lora_linear_without_bias PASSED                 [ 41%]
tests/test_lora.py::test_lora_double_merge_is_idempotent PASSED          [ 43%]
tests/test_lora.py::test_lora_unmerge_without_merge_is_noop PASSED       [ 45%]
tests/test_lora.py::test_lora_save_and_load_adapter_weights PASSED       [ 47%]
tests/test_qlora.py::test_nf4_quantile_table PASSED                      [ 50%]
tests/test_qlora.py::test_nf4_quantize_dequantize_roundtrip PASSED       [ 52%]
tests/test_qlora.py::test_nf4_block_padding PASSED                       [ 54%]
tests/test_qlora.py::test_double_quantize_roundtrip PASSED               [ 56%]
tests/test_qlora.py::test_double_quantize_memory_savings PASSED          [ 58%]
tests/test_qlora.py::test_qlora_linear_from_linear PASSED                [ 60%]
tests/test_qlora.py::test_qlora_linear_gradient_flow PASSED              [ 63%]
tests/test_qlora.py::test_qlora_linear_rank_zero PASSED                  [ 65%]
tests/test_qlora.py::test_qlora_without_double_quant PASSED              [ 67%]
tests/test_qlora.py::test_qlora_from_linear_without_bias PASSED          [ 69%]
tests/test_qlora.py::test_qlora_odd_shaped_matrices PASSED               [ 71%]
tests/test_qlora.py::test_qlora_double_quant_arbitrary_scales PASSED     [ 73%]
tests/test_sequence_logps.py::test_compute_sequence_log_probs_shape PASSED [ 76%]
tests/test_sequence_logps.py::test_compute_sequence_log_probs_causal_shift PASSED [ 78%]
tests/test_sequence_logps.py::test_compute_sequence_log_probs_with_ignore_index PASSED [ 80%]
tests/test_sequence_logps.py::test_compute_sequence_log_probs_with_attention_mask PASSED [ 82%]
tests/test_sequence_logps.py::test_compute_sequence_log_probs_average PASSED [ 84%]
tests/test_sequence_logps.py::test_concatenate_chosen_and_rejected PASSED [ 86%]
tests/test_trainer.py::test_trainer_initialization PASSED                [ 89%]
tests/test_trainer.py::test_trainer_no_trainable_params_raises PASSED    [ 91%]
tests/test_trainer.py::test_trainer_train_step_updates_only_lora PASSED  [ 93%]
tests/test_trainer.py::test_trainer_convergence_on_synthetic_preference PASSED [ 95%]
tests/test_trainer.py::test_trainer_evaluate_empty_dataset PASSED        [ 97%]
tests/test_trainer.py::test_trainer_custom_optimizer PASSED              [100%]

============================= 46 passed in 4.27s ==============================
```

---

## 📄 License
MIT License. Created by [SKB-18](https://github.com/SKB-18).
