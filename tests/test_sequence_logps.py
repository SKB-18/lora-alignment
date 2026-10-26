"""
Unit tests for causal sequence log-probability calculation and batch concatenation.
"""

import pytest
import torch
import torch.nn.functional as F
from dpo.utils import compute_sequence_log_probs, concatenate_chosen_and_rejected


def test_compute_sequence_log_probs_shape():
    """Verify return shape is [batch_size]."""
    batch_size, seq_len, vocab_size = 4, 10, 32
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    logps = compute_sequence_log_probs(logits, labels)
    assert logps.shape == (batch_size,)
    assert torch.all(logps <= 0.0)  # Log probabilities must be <= 0


def test_compute_sequence_log_probs_causal_shift():
    """Verify that token t prediction uses logits from position t-1."""
    # Construct a simple case: 1 batch, 3 tokens, vocab 4
    # Tokens: [A, B, C]
    # Prediction of B should come from logits[0]
    # Prediction of C should come from logits[1]
    logits = torch.zeros(1, 3, 4)
    # Set logits[0, 0] to high for token 1
    logits[0, 0, 1] = 100.0  # token at t=1 (B) should be 1
    # Set logits[0, 1] to high for token 2
    logits[0, 1, 2] = 100.0  # token at t=2 (C) should be 2

    labels = torch.tensor([[0, 1, 2]])
    logps = compute_sequence_log_probs(logits, labels)

    # Both next-token predictions should have prob ~ 1.0 (log prob ~ 0.0)
    assert abs(logps.item() - 0.0) < 1e-3


def test_compute_sequence_log_probs_with_ignore_index():
    """Verify tokens marked with ignore_index (-100) are omitted from log prob sum."""
    logits = torch.randn(2, 5, 10)
    labels = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, -100, -100],
    ])

    logps = compute_sequence_log_probs(logits, labels, ignore_index=-100)
    assert logps[0] != logps[1]


def test_compute_sequence_log_probs_with_attention_mask():
    """Verify attention mask 0 ignores padded tokens."""
    logits = torch.randn(1, 6, 8)
    labels = torch.tensor([[1, 2, 3, 4, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])

    logps_masked = compute_sequence_log_probs(logits, labels, attention_mask=mask)

    # Compare with sliced sequence of length 4
    sliced_logits = logits[:, :4, :]
    sliced_labels = labels[:, :4]
    logps_sliced = compute_sequence_log_probs(sliced_logits, sliced_labels)

    torch.testing.assert_close(logps_masked, logps_sliced, rtol=1e-5, atol=1e-6)


def test_compute_sequence_log_probs_average():
    """Verify average_log_prob divides sum by valid token count."""
    logits = torch.randn(1, 4, 6)
    labels = torch.tensor([[1, 2, 3, 4]])

    sum_logps = compute_sequence_log_probs(logits, labels, average_log_prob=False)
    avg_logps = compute_sequence_log_probs(logits, labels, average_log_prob=True)

    # 4 tokens -> 3 next-token prediction steps
    assert abs(avg_logps.item() - (sum_logps.item() / 3.0)) < 1e-5


def test_concatenate_chosen_and_rejected():
    """Verify padding and concatenation of chosen and rejected token tensors."""
    chosen = torch.tensor([[1, 2, 3]])
    rejected = torch.tensor([[1, 2, 3, 4, 5]])

    concat_ids, concat_mask = concatenate_chosen_and_rejected(chosen, rejected, pad_token_id=0)

    assert concat_ids.shape == (2, 5)
    assert concat_mask.shape == (2, 5)

    # First row is chosen, padded from length 3 to 5
    assert torch.equal(concat_ids[0], torch.tensor([1, 2, 3, 0, 0]))
    assert torch.equal(concat_mask[0], torch.tensor([1, 1, 1, 0, 0]))

    # Second row is rejected, unpadded
    assert torch.equal(concat_ids[1], torch.tensor([1, 2, 3, 4, 5]))
    assert torch.equal(concat_mask[1], torch.tensor([1, 1, 1, 1, 1]))
