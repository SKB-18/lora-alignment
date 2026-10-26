"""
Unit tests for Direct Preference Optimization (DPO) Loss and Mathematical Properties.
"""

import pytest
import math
import torch
import torch.nn.functional as F
from dpo.loss import DPOLoss
from eval.metrics import (
    compute_implicit_rewards,
    compute_reward_margin,
    compute_preference_accuracy,
    compute_kl_drift,
    evaluate_preference_batch,
)


def test_dpo_loss_formula_symmetric():
    """When policy matches reference model (logits=0), DPO loss must equal log(2) approx 0.6931."""
    loss_fn = DPOLoss(beta=0.1)

    policy_chosen = torch.tensor([-2.5, -3.0])
    policy_rejected = torch.tensor([-4.0, -4.5])

    # Reference logps identical to policy logps
    ref_chosen = policy_chosen.clone()
    ref_rejected = policy_rejected.clone()

    loss, metrics = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    expected_loss = math.log(2.0)
    assert abs(loss.item() - expected_loss) < 1e-5
    assert abs(metrics["reward_margin"] - 0.0) < 1e-5


def test_dpo_loss_decreases_with_preference_margin():
    """As chosen reward grows larger than rejected reward, DPO loss must decrease monotonically."""
    loss_fn = DPOLoss(beta=0.5)

    ref_c = torch.tensor([-2.0])
    ref_r = torch.tensor([-2.0])

    # Margin 0: policy chosen == ref, policy rejected == ref
    loss_0, _ = loss_fn(torch.tensor([-2.0]), torch.tensor([-2.0]), ref_c, ref_r)

    # Margin +1: policy chosen is higher
    loss_1, _ = loss_fn(torch.tensor([-1.0]), torch.tensor([-2.0]), ref_c, ref_r)

    # Margin +3: policy chosen is much higher
    loss_3, _ = loss_fn(torch.tensor([1.0]), torch.tensor([-2.0]), ref_c, ref_r)

    assert loss_0.item() > loss_1.item() > loss_3.item()


def test_dpo_gradient_direction():
    """Verify DPO gradients pull chosen logps UP (negative grad) and rejected logps DOWN (positive grad)."""
    loss_fn = DPOLoss(beta=1.0)

    policy_chosen = torch.tensor([-3.0], requires_grad=True)
    policy_rejected = torch.tensor([-3.0], requires_grad=True)

    ref_chosen = torch.tensor([-3.0])
    ref_rejected = torch.tensor([-3.0])

    loss, _ = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    loss.backward()

    # Minimizing loss means: grad_chosen < 0 (increase chosen), grad_rejected > 0 (decrease rejected)
    assert policy_chosen.grad.item() < 0.0
    assert policy_rejected.grad.item() > 0.0
    assert abs(policy_chosen.grad.item() + policy_rejected.grad.item()) < 1e-6


def test_dpo_label_smoothing():
    """Verify label smoothing bounds the gradient and softens extreme margins."""
    loss_standard = DPOLoss(beta=0.1, label_smoothing=0.0)
    loss_smoothed = DPOLoss(beta=0.1, label_smoothing=0.1)

    pc = torch.tensor([5.0])
    pr = torch.tensor([-5.0])
    rc = torch.tensor([0.0])
    rr = torch.tensor([0.0])

    l_std, _ = loss_standard(pc, pr, rc, rr)
    l_smooth, _ = loss_smoothed(pc, pr, rc, rr)

    # Smoothed loss penalizes overconfidence, so loss is slightly higher
    assert l_smooth.item() > l_std.item()


def test_dpo_reference_free():
    """Verify reference_free mode executes when reference models are None."""
    loss_fn = DPOLoss(beta=0.2, reference_free=True)

    pc = torch.tensor([-1.0, -2.0])
    pr = torch.tensor([-3.0, -4.0])

    loss, metrics = loss_fn(pc, pr)
    assert loss.item() > 0.0
    assert metrics["accuracy"] == 1.0
    assert metrics["reward_margin"] > 0.0


def test_dpo_invalid_beta():
    """Verify ValueError is raised if beta <= 0."""
    with pytest.raises(ValueError, match="Beta must be strictly positive"):
        DPOLoss(beta=0.0)
    with pytest.raises(ValueError, match="Beta must be strictly positive"):
        DPOLoss(beta=-0.5)


def test_eval_metrics_helpers():
    """Test standalone evaluation functions in eval/metrics.py."""
    pc = torch.tensor([-1.0, -0.5, -2.0])
    pr = torch.tensor([-2.0, -1.5, -1.0])
    rc = torch.tensor([-1.5, -1.0, -2.0])
    rr = torch.tensor([-2.0, -1.0, -1.5])

    eval_dict = evaluate_preference_batch(pc, pr, rc, rr, beta=0.1)

    assert "chosen_reward_mean" in eval_dict
    assert "rejected_reward_mean" in eval_dict
    assert "reward_margin" in eval_dict
    assert "accuracy" in eval_dict
    assert "chosen_kl_drift" in eval_dict
    assert "rejected_kl_drift" in eval_dict
    assert 0.0 <= eval_dict["accuracy"] <= 1.0


def test_dpo_loss_extreme_logits_numerical_stability():
    """Verify DPO loss doesn't produce NaN or Inf under extreme margin conditions."""
    loss_fn = DPOLoss(beta=0.1)

    # Very large positive margin (chosen >> rejected)
    pc_huge = torch.tensor([1000.0])
    pr_huge = torch.tensor([-1000.0])
    rc = torch.tensor([0.0])
    rr = torch.tensor([0.0])

    loss_pos, _ = loss_fn(pc_huge, pr_huge, rc, rr)
    assert not torch.isnan(loss_pos)
    assert not torch.isinf(loss_pos)
    assert loss_pos.item() >= 0.0

    # Very large negative margin (rejected >> chosen)
    loss_neg, _ = loss_fn(pr_huge, pc_huge, rc, rr)
    assert not torch.isnan(loss_neg)
    assert not torch.isinf(loss_neg)
