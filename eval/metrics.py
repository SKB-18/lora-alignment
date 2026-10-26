"""
Evaluation Metrics for Direct Preference Optimization (DPO) and Preference Alignment.
"""

from typing import Dict, Optional, Tuple
import torch


def compute_implicit_rewards(
    policy_logps: torch.Tensor,
    reference_logps: Optional[torch.Tensor] = None,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Computes DPO implicit reward:
        r(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))

    If reference_logps is None, computes unreferenced policy scores scaled by beta.
    """
    if reference_logps is None:
        return beta * policy_logps
    return beta * (policy_logps - reference_logps)


def compute_reward_margin(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> float:
    """
    Computes mean reward margin: E[r(x, y_chosen) - r(x, y_rejected)].
    """
    return (chosen_rewards - rejected_rewards).mean().item()


def compute_preference_accuracy(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> float:
    """
    Computes accuracy (win-rate): percentage of samples where r(x, y_chosen) > r(x, y_rejected).
    """
    return (chosen_rewards > rejected_rewards).float().mean().item()


def compute_kl_drift(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
) -> float:
    """
    Computes empirical KL divergence drift: E_{y ~ pi}[log pi(y|x) - log pi_ref(y|x)].
    """
    return (policy_logps - reference_logps).mean().item()


def evaluate_preference_batch(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: Optional[torch.Tensor] = None,
    reference_rejected_logps: Optional[torch.Tensor] = None,
    beta: float = 0.1,
) -> Dict[str, float]:
    """
    Computes a comprehensive evaluation dictionary for a preference batch.
    """
    chosen_rewards = compute_implicit_rewards(policy_chosen_logps, reference_chosen_logps, beta=beta)
    rejected_rewards = compute_implicit_rewards(policy_rejected_logps, reference_rejected_logps, beta=beta)

    margin = compute_reward_margin(chosen_rewards, rejected_rewards)
    accuracy = compute_preference_accuracy(chosen_rewards, rejected_rewards)

    results = {
        "chosen_reward_mean": chosen_rewards.mean().item(),
        "chosen_reward_std": chosen_rewards.std().item() if chosen_rewards.numel() > 1 else 0.0,
        "rejected_reward_mean": rejected_rewards.mean().item(),
        "rejected_reward_std": rejected_rewards.std().item() if rejected_rewards.numel() > 1 else 0.0,
        "reward_margin": margin,
        "accuracy": accuracy,
    }

    if reference_chosen_logps is not None:
        results["chosen_kl_drift"] = compute_kl_drift(policy_chosen_logps, reference_chosen_logps)
    if reference_rejected_logps is not None:
        results["rejected_kl_drift"] = compute_kl_drift(policy_rejected_logps, reference_rejected_logps)

    return results
