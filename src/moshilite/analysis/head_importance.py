"""Gradient-based attention head importance scoring.

Computes importance of each attention head using gradient-based saliency:
the magnitude of gradients flowing through each head's output, averaged
over a calibration dataset. Heads with low importance can be pruned.

Reference: "Are Sixteen Heads Really Better than One?" (Michel et al., 2019)
"""

import torch
import numpy as np
from torch import nn, Tensor
from typing import Callable
from dataclasses import dataclass


@dataclass
class HeadImportanceResult:
    """Attention head importance analysis result."""
    importance_scores: np.ndarray  # shape: [n_layers, n_heads]
    ranked_heads: list[tuple[int, int, float]]  # (layer, head, score), ascending

    def get_least_important(self, n: int) -> list[tuple[int, int]]:
        """Return the n least important (layer, head) pairs for pruning."""
        return [(layer, head) for layer, head, _ in self.ranked_heads[:n]]


def compute_head_importance(
    model: nn.Module,
    data: list[Tensor],
    n_layers: int,
    n_heads: int,
    head_mask_fn: Callable,
    loss_fn: Callable,
    device: str = "cuda",
) -> HeadImportanceResult:
    """Compute gradient-based importance for every attention head.

    Uses the sensitivity method: importance(head) = |grad(loss) w.r.t head_mask|.
    A head mask of 1.0 is applied to each head, and the gradient of the
    loss with respect to this mask indicates how much removing the head
    would affect the output.

    Args:
        model: Moshi Temporal Transformer model.
        data: List of (input_tokens, target_tokens) pairs for gradient computation.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.
        head_mask_fn: Function to apply head masks to the model.
            Signature: head_mask_fn(model, head_mask_tensor) -> None
            head_mask_tensor: shape [n_layers, n_heads], values in [0, 1].
        loss_fn: Loss function.
            Signature: loss_fn(model_output, targets) -> scalar loss.
        device: Device to run on.

    Returns:
        HeadImportanceResult with per-head scores and ranking.
    """
    # Initialize head mask as all-ones (no masking), requires grad
    head_mask = torch.ones(n_layers, n_heads, device=device, requires_grad=True)

    importance_acc = torch.zeros(n_layers, n_heads, device=device)

    model.eval()
    for input_tokens, targets in data:
        input_tokens = input_tokens.to(device)
        targets = targets.to(device)

        # Zero grads
        if head_mask.grad is not None:
            head_mask.grad.zero_()

        # Apply head mask
        head_mask_fn(model, head_mask)

        # Forward + backward
        outputs = model(input_tokens)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Accumulate absolute gradient as importance
        if head_mask.grad is not None:
            importance_acc += head_mask.grad.abs().detach()

    # Average over data
    importance_avg = importance_acc / len(data)
    scores = importance_avg.cpu().numpy()

    # Rank heads: flatten, sort ascending (least important first)
    ranked = []
    for layer in range(n_layers):
        for head in range(n_heads):
            ranked.append((layer, head, float(scores[layer, head])))
    ranked.sort(key=lambda x: x[2])

    return HeadImportanceResult(
        importance_scores=scores,
        ranked_heads=ranked,
    )


def compute_head_entropy(
    attention_weights: list[Tensor],
) -> np.ndarray:
    """Compute entropy of each head's attention distribution.

    Heads with near-uniform attention (high entropy) are less specialized
    and may be more pruneable.

    Args:
        attention_weights: List of attention weight tensors, one per layer.
            Each tensor shape: [batch, n_heads, seq_len, seq_len].

    Returns:
        Array of shape [n_layers, n_heads] with mean entropy per head.
    """
    entropies = []
    for attn in attention_weights:
        # attn: [batch, n_heads, seq, seq]
        # Clamp to avoid log(0)
        attn_clamped = attn.clamp(min=1e-10)
        entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # [batch, heads, seq]
        mean_entropy = entropy.mean(dim=(0, 2))  # [heads]
        entropies.append(mean_entropy.cpu().numpy())

    return np.stack(entropies)  # [n_layers, n_heads]
