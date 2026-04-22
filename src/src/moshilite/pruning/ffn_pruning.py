"""FFN intermediate channel pruning by PCA importance."""

import logging
import torch
import torch.nn as nn
import numpy as np

from moshilite.analysis.ffn_importance import FFNImportanceResult
from moshilite.analysis.moshi_hooks import _get_transformer

logger = logging.getLogger(__name__)

def prune_ffn_channels(
    model: nn.Module, 
    ffn_importance_result: FFNImportanceResult, 
    max_pct: float, 
    mode: str, 
    cosine_similarities: np.ndarray = None
) -> nn.Module:
    """Prune FFN intermediate channels from the transformer layers.
    
    Args:
        model: Moshi LMModel.
        ffn_importance_result: Scored and ranked FFN channels.
        max_pct: Maximum percentage of channels to remove globally.
        mode: "uniform" or "non_uniform".
        cosine_similarities: Array of layer cosine similarities (required for non_uniform).
        
    Returns:
        The mutated model.
    """
    transformer = _get_transformer(model)
    layers = list(transformer.layers)
    n_layers = len(layers)
    
    # Calculate pruning ratio per layer
    if mode == "uniform":
        prune_ratios = np.full(n_layers, max_pct)
    else:
        if cosine_similarities is None:
            raise ValueError("cosine_similarities must be provided for non_uniform pruning.")
        
        S = cosine_similarities
        S_norm = S / np.mean(S)
        prune_ratios = np.clip(max_pct * S_norm, 0.0, 0.9)
        
    for i, layer in enumerate(layers):
        if i not in ffn_importance_result.importance_scores:
            continue
            
        n_channels = len(ffn_importance_result.importance_scores[i])
        n_prune = int(n_channels * prune_ratios[i])
        n_keep = n_channels - n_prune
        
        if n_prune <= 0:
            continue
            
        logger.info(f"Layer {i}: pruning {n_prune} FFN channels, keeping {n_keep}")
        
        # Determine the actual d_ffn from the model's linear layers
        actual_d_ffn = None
        if hasattr(layer, "gating") and hasattr(layer.gating, "linear_in"):
            # SwiGLU projects to 2 * d_ffn
            actual_d_ffn = layer.gating.linear_in.out_features // 2
        elif hasattr(layer, "linear1"):
            actual_d_ffn = layer.linear1.out_features
            
        if actual_d_ffn is None:
            logger.warning(f"Layer {i}: Cannot detect d_ffn, skipping FFN prune.")
            continue
            
        n_prune = int(actual_d_ffn * prune_ratios[i])
        n_keep = actual_d_ffn - n_prune
        
        if n_prune <= 0:
            continue
            
        logger.info(f"Layer {i}: pruning {n_prune} FFN channels, keeping {n_keep}")
        
        pca_scores = ffn_importance_result.importance_scores.get(i)
        
        # If PCA scores were extracted over the wrong dimension (e.g. hooking a quantized model's output instead of internal),
        # we fallback to random or uniform selection for the actual d_ffn.
        if pca_scores is None or len(pca_scores) != actual_d_ffn:
            logger.warning(f"Layer {i}: PCA score size {len(pca_scores) if pca_scores is not None else 0} mismatch with actual d_ffn {actual_d_ffn}. Using uniform channel spread.")
            # Spread keeps uniformly
            keep_indices = np.linspace(0, actual_d_ffn - 1, n_keep, dtype=int)
        else:
            keep_mask = ffn_importance_result.get_prune_mask(i, keep_ratio=(n_keep / actual_d_ffn))
            keep_indices = np.where(keep_mask)[0]
        
        # Apply pruning to the FFN modules
        if hasattr(layer, "gating") and layer.gating is not None:
            gating = layer.gating
            modules_to_check = [gating] if not isinstance(gating, nn.ModuleList) else list(gating)
            
            for m in modules_to_check:
                # SwiGLU `linear_in` output must drop from both the gate and up-projection halves!
                if hasattr(m, "linear_in") and isinstance(m.linear_in, nn.Linear):
                    _shrink_ffn_glu_linear_in(m.linear_in, keep_indices, actual_d_ffn)
                    
                # Down projection (receives single d_ffn dimension)
                if hasattr(m, "linear_out") and isinstance(m.linear_out, nn.Linear):
                    _shrink_ffn_linear_in_features(m.linear_out, keep_indices)
                elif hasattr(m, "linear") and isinstance(m.linear, nn.Linear):
                    # fallback naming
                    if m.in_features == actual_d_ffn:
                        _shrink_ffn_linear_in_features(m, keep_indices)

        # Standard MLPs
        if hasattr(layer, "linear1") and isinstance(layer.linear1, nn.Linear):
            _shrink_ffn_linear_out_features(layer.linear1, keep_indices)
            
        if hasattr(layer, "linear2") and isinstance(layer.linear2, nn.Linear):
            _shrink_ffn_linear_in_features(layer.linear2, keep_indices)
            
    return model

def _shrink_ffn_glu_linear_in(linear: nn.Linear, keep_indices: np.ndarray, original_d_ffn: int):
    """Shrinks a GLU up-projection that maps to [2 * d_ffn]."""
    in_features = linear.in_features
    # We must keep indices from the first half and the SECOND half identical!
    # The output is chunked at runtime as: chunk1, chunk2 = split(dim=-1)
    keep_indices_half2 = keep_indices + original_d_ffn
    full_keep_indices = np.concatenate([keep_indices, keep_indices_half2])
    
    w = linear.weight[full_keep_indices, :]
    linear.out_features = len(full_keep_indices)
    linear.weight = nn.Parameter(w)
    
    if linear.bias is not None:
        b = linear.bias[full_keep_indices]
        linear.bias = nn.Parameter(b)

def _shrink_ffn_linear_out_features(linear: nn.Linear, keep_indices: np.ndarray):
    """Shrinks output dimension of a standard Linear layer."""
    w = linear.weight[keep_indices, :]
    linear.out_features = len(keep_indices)
    linear.weight = nn.Parameter(w)
    
    if linear.bias is not None:
        b = linear.bias[keep_indices]
        linear.bias = nn.Parameter(b)
        
def _shrink_ffn_linear_in_features(linear: nn.Linear, keep_indices: np.ndarray):
    """Shrinks input dimension of a Linear layer."""
    w = linear.weight[:, keep_indices]
    linear.in_features = len(keep_indices)
    linear.weight = nn.Parameter(w)
