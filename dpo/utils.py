"""
Utility functions for sequence log-probability calculation and token masking.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def compute_sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """
    Computes per-sequence log probabilities for causal language models.

    Given logits of shape [batch_size, seq_len, vocab_size] and target token labels
    of shape [batch_size, seq_len], this function shifts logits and labels by 1 token
    for next-token prediction, computes token-level log-likelihoods, and sums or averages
    them over valid tokens.

    Args:
        logits: Tensor of unnormalized next-token predictions [B, T, V].
        labels: Tensor of token ids [B, T].
        attention_mask: Optional binary attention mask [B, T] (1 for valid, 0 for pad).
        ignore_index: Label value indicating ignored tokens (default -100).
        average_log_prob: If True, divides sum of log probs by valid sequence length.

    Returns:
        Tensor of shape [batch_size] containing sequence log-probabilities.
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits of shape [B, T, V], got {logits.shape}")
    if labels.dim() != 2:
        raise ValueError(f"Expected labels of shape [B, T], got {labels.shape}")

    # Shift tokens for next-token prediction
    # Predict token t using logits from position t-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_size, seq_len, vocab_size = shift_logits.shape

    # Compute valid token mask
    valid_mask = shift_labels != ignore_index
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous().bool()
        valid_mask = valid_mask & shift_mask

    # Replace ignore_index with 0 for gather operation
    safe_labels = shift_labels.clone()
    safe_labels[~valid_mask] = 0

    # Compute per-token log probabilities: log p(x_t | x_{<t})
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log prob of target labels: [B, T, 1] -> [B, T]
    per_token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    # Zero out ignored/padded tokens
    per_token_log_probs = per_token_log_probs * valid_mask.float()

    # Sum along sequence dimension
    sequence_log_probs = per_token_log_probs.sum(dim=-1)

    if average_log_prob:
        seq_lengths = valid_mask.float().sum(dim=-1).clamp(min=1.0)
        sequence_log_probs = sequence_log_probs / seq_lengths

    return sequence_log_probs


def concatenate_chosen_and_rejected(
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask: Optional[torch.Tensor] = None,
    rejected_mask: Optional[torch.Tensor] = None,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenates chosen and rejected batches for efficient single-pass model inference.

    Pads sequences to maximum batch length and stacks [chosen_batch; rejected_batch].

    Returns:
        Tuple of (concatenated_ids, concatenated_attention_mask)
    """
    b_chosen, len_chosen = chosen_ids.shape
    b_rejected, len_rejected = rejected_ids.shape
    max_len = max(len_chosen, len_rejected)

    # Pad chosen if needed
    if len_chosen < max_len:
        pad_size = max_len - len_chosen
        chosen_pad = torch.full((b_chosen, pad_size), pad_token_id, dtype=chosen_ids.dtype, device=chosen_ids.device)
        chosen_ids = torch.cat([chosen_ids, chosen_pad], dim=1)
        if chosen_mask is not None:
            mask_pad = torch.zeros((b_chosen, pad_size), dtype=chosen_mask.dtype, device=chosen_mask.device)
            chosen_mask = torch.cat([chosen_mask, mask_pad], dim=1)
        else:
            chosen_mask = torch.cat([
                torch.ones((b_chosen, len_chosen), device=chosen_ids.device),
                torch.zeros((b_chosen, pad_size), device=chosen_ids.device),
            ], dim=1)
    elif chosen_mask is None:
        chosen_mask = torch.ones_like(chosen_ids)

    # Pad rejected if needed
    if len_rejected < max_len:
        pad_size = max_len - len_rejected
        rejected_pad = torch.full((b_rejected, pad_size), pad_token_id, dtype=rejected_ids.dtype, device=rejected_ids.device)
        rejected_ids = torch.cat([rejected_ids, rejected_pad], dim=1)
        if rejected_mask is not None:
            mask_pad = torch.zeros((b_rejected, pad_size), dtype=rejected_mask.dtype, device=rejected_mask.device)
            rejected_mask = torch.cat([rejected_mask, mask_pad], dim=1)
        else:
            rejected_mask = torch.cat([
                torch.ones((b_rejected, len_rejected), device=rejected_ids.device),
                torch.zeros((b_rejected, pad_size), device=rejected_ids.device),
            ], dim=1)
    elif rejected_mask is None:
        rejected_mask = torch.ones_like(rejected_ids)

    concat_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
    concat_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
    return concat_ids, concat_mask
