"""Block Influence (BI) score computation for layer importance analysis.

BI scores measure each layer's contribution to the model's output by
comparing hidden state representations before and after each layer.
A higher BI score means the layer transforms representations more —
removing it would have higher impact.

Reference: "ShortGPT: Layers in LLMs are More Redundant Than You Think"
"""

import torch
import numpy as np
from torch import nn, Tensor
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class BIResult:
    """Block Influence analysis result."""
    bi_scores: np.ndarray  # shape: [n_layers]
    cosine_similarities: np.ndarray  # shape: [n_layers] (1 - BI)
    pairwise_cosine_matrix: Optional[np.ndarray] = None  # shape: [n_layers, n_layers]


def compute_bi_scores(
    model: nn.Module,
    data: list[Tensor],
    layer_hook_fn: Callable,
    device: str = "cuda",
    compute_pairwise: bool = True,
) -> BIResult:
    """Compute Block Influence scores for all transformer layers.

    BI(layer_i) = 1 - cosine_sim(hidden_state_before_layer_i, hidden_state_after_layer_i)

    Higher BI = more transformative layer = harder to prune.

    Args:
        model: Moshi Temporal Transformer model (loaded at validated precision).
        data: List of input token tensors to evaluate on.
        layer_hook_fn: Function that registers forward hooks on the model and
            returns collected hidden states. Signature:
                layer_hook_fn(model, input_batch) -> list[Tensor]
            Returns list of hidden state tensors, one per layer boundary
            (including input to first layer), so len = n_layers + 1.
        device: Device to run on.
        compute_pairwise: Whether to compute the full pairwise cosine matrix.

    Returns:
        BIResult with BI scores and optionally pairwise cosine matrix.
    """
    all_hidden_states = []

    model.eval()
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device)
            hidden_states = layer_hook_fn(model, batch)
            # hidden_states: list of [batch, seq_len, hidden_dim] tensors
            # Average over batch and seq_len for each layer
            averaged = [h.float().mean(dim=(0, 1)) for h in hidden_states]
            all_hidden_states.append(averaged)

    # Average across all data batches
    n_boundaries = len(all_hidden_states[0])  # n_layers + 1
    avg_states = []
    for layer_idx in range(n_boundaries):
        stacked = torch.stack([hs[layer_idx] for hs in all_hidden_states])
        avg_states.append(stacked.mean(dim=0))  # [hidden_dim]

    # Compute BI scores: cosine sim between consecutive layer boundaries
    n_layers = n_boundaries - 1
    cosine_sims = np.zeros(n_layers)
    for i in range(n_layers):
        cos = torch.nn.functional.cosine_similarity(
            avg_states[i].unsqueeze(0),
            avg_states[i + 1].unsqueeze(0),
        )
        cosine_sims[i] = cos.item()

    bi_scores = 1.0 - cosine_sims

    # Optionally compute full pairwise cosine matrix
    pairwise = None
    if compute_pairwise:
        pairwise = np.zeros((n_boundaries, n_boundaries))
        for i in range(n_boundaries):
            for j in range(n_boundaries):
                cos = torch.nn.functional.cosine_similarity(
                    avg_states[i].unsqueeze(0),
                    avg_states[j].unsqueeze(0),
                )
                pairwise[i, j] = cos.item()

    return BIResult(
        bi_scores=bi_scores,
        cosine_similarities=cosine_sims,
        pairwise_cosine_matrix=pairwise,
    )


def rank_layers_by_importance(bi_scores: np.ndarray) -> list[int]:
    """Rank layers by BI score (ascending = least important first).

    Returns list of layer indices sorted from least to most important.
    Prune from the beginning of this list.
    """
    return list(np.argsort(bi_scores))
