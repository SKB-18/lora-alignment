"""
Evaluation module for DPO alignment and policy diagnostics.
"""

from .metrics import (
    compute_implicit_rewards,
    compute_reward_margin,
    compute_preference_accuracy,
    compute_kl_drift,
    evaluate_preference_batch,
)

__all__ = [
    "compute_implicit_rewards",
    "compute_reward_margin",
    "compute_preference_accuracy",
    "compute_kl_drift",
    "evaluate_preference_batch",
]
