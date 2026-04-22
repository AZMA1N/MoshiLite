"""PCA-guided FFN channel importance analysis.

Identifies which intermediate FFN channels contribute least to the
model's representations using PCA-based variance analysis. Channels
explaining little variance can be pruned with minimal quality loss.

Reference: "SlimLLM" (PCA-guided FFN pruning approach)
"""

import torch
import numpy as np
from torch import nn, Tensor
from typing import Callable
from dataclasses import dataclass


@dataclass
class FFNImportanceResult:
    """FFN channel importance analysis result."""
    importance_scores: dict[int, np.ndarray]  # layer_idx -> [intermediate_dim] scores
    explained_variance_ratios: dict[int, np.ndarray]  # layer_idx -> PCA variance ratios
    ranked_channels: dict[int, list[int]]  # layer_idx -> channel indices (ascending importance)

    def get_least_important(self, layer_idx: int, n: int) -> list[int]:
        """Return the n least important channel indices for a given layer."""
        return self.ranked_channels[layer_idx][:n]

    def get_prune_mask(self, layer_idx: int, keep_ratio: float) -> np.ndarray:
        """Return boolean mask: True = keep, False = prune."""
        n_channels = len(self.importance_scores[layer_idx])
        n_keep = max(1, int(n_channels * keep_ratio))
        mask = np.zeros(n_channels, dtype=bool)
        # Keep the most important channels
        important = self.ranked_channels[layer_idx][-n_keep:]
        mask[important] = True
        return mask


def compute_ffn_importance(
    model: nn.Module,
    data: list[Tensor],
    layer_indices: list[int],
    ffn_hook_fn: Callable,
    device: str = "cuda",
    n_components: int = 64,
) -> FFNImportanceResult:
    """Compute PCA-guided importance scores for FFN intermediate channels.

    For each FFN layer:
    1. Collect intermediate activations (after gate/up projection, before down)
    2. Run PCA on the activation matrix [n_samples * seq_len, intermediate_dim]
    3. Channel importance = sum of variance explained by principal components
       that load heavily on that channel

    Args:
        model: Moshi Temporal Transformer model.
        data: List of input token tensors.
        layer_indices: Which transformer layer indices to analyze.
        ffn_hook_fn: Function to register hooks and collect FFN intermediate
            activations. Signature:
                ffn_hook_fn(model, input_batch, layer_idx) -> Tensor
            Returns: [batch * seq_len, intermediate_dim] activation tensor.
        device: Device to run on.
        n_components: Number of PCA components to compute.

    Returns:
        FFNImportanceResult with per-layer importance scores.
    """
    importance_scores = {}
    variance_ratios = {}
    ranked_channels = {}

    model.eval()
    with torch.no_grad():
        for layer_idx in layer_indices:
            # Collect activations across all data
            all_activations = []
            for batch in data:
                batch = batch.to(device)
                acts = ffn_hook_fn(model, batch, layer_idx)
                all_activations.append(acts.cpu().float())

            # Stack: [total_tokens, intermediate_dim]
            activation_matrix = torch.cat(all_activations, dim=0).numpy()

            # PCA analysis
            scores, var_ratios = _pca_channel_importance(
                activation_matrix, n_components
            )

            importance_scores[layer_idx] = scores
            variance_ratios[layer_idx] = var_ratios
            ranked_channels[layer_idx] = list(np.argsort(scores))

    return FFNImportanceResult(
        importance_scores=importance_scores,
        explained_variance_ratios=variance_ratios,
        ranked_channels=ranked_channels,
    )


def _pca_channel_importance(
    activations: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel importance using PCA.

    Importance of channel j = sum over top-k PCA components of
    (variance_explained_k * |loading_k_j|^2).

    This weights channels by how much they contribute to the directions
    of highest variance in the activation space.

    Args:
        activations: [n_samples, intermediate_dim] activation matrix.
        n_components: Number of PCA components.

    Returns:
        Tuple of (importance_scores [intermediate_dim], variance_ratios [n_components]).
    """
    from sklearn.decomposition import PCA

    n_components = min(n_components, activations.shape[1], activations.shape[0])

    pca = PCA(n_components=n_components)
    pca.fit(activations)

    # Loading matrix: [n_components, intermediate_dim]
    loadings = pca.components_
    variance_ratios = pca.explained_variance_ratio_

    # Channel importance = weighted sum of squared loadings
    # Weight each component by its explained variance ratio
    weighted_loadings_sq = (loadings ** 2) * variance_ratios[:, np.newaxis]
    channel_importance = weighted_loadings_sq.sum(axis=0)  # [intermediate_dim]

    return channel_importance, variance_ratios


def compute_activation_norms(
    model: nn.Module,
    data: list[Tensor],
    layer_indices: list[int],
    ffn_hook_fn: Callable,
    device: str = "cuda",
) -> dict[int, np.ndarray]:
    """Simpler alternative: rank channels by mean activation magnitude.

    Less informative than PCA but faster and doesn't require sklearn.

    Returns:
        Dict mapping layer_idx -> mean absolute activation per channel.
    """
    norms = {}

    model.eval()
    with torch.no_grad():
        for layer_idx in layer_indices:
            all_norms = []
            for batch in data:
                batch = batch.to(device)
                acts = ffn_hook_fn(model, batch, layer_idx)
                # Mean absolute activation per channel
                channel_norms = acts.abs().mean(dim=0)  # [intermediate_dim]
                all_norms.append(channel_norms.cpu())

            norms[layer_idx] = torch.stack(all_norms).mean(dim=0).numpy()

    return norms
