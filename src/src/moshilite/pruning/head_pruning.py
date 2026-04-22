"""Attention head pruning by gradient saliency ranking."""

import logging
import torch
import torch.nn as nn
import numpy as np

from moshilite.analysis.head_importance import HeadImportanceResult
from moshilite.analysis.moshi_hooks import _get_transformer

logger = logging.getLogger(__name__)

def prune_heads(
    model: nn.Module, 
    head_importance_result: HeadImportanceResult, 
    max_pct: float, 
    mode: str, 
    cosine_similarities: np.ndarray = None
) -> nn.Module:
    """Prune attention heads from the transformer layers.
    
    Args:
        model: Moshi LMModel.
        head_importance_result: Scored and ranked heads.
        max_pct: Maximum percentage of heads to remove globally.
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
        # non_uniform: scale by layer's cosine similarity. High similarity = less important = prune more.
        if cosine_similarities is None:
            raise ValueError("cosine_similarities must be provided for non_uniform pruning.")
        
        S = cosine_similarities
        # Normalize S so its mean is 1.0, and clip it
        # That way the overall average pruning ratio stays near max_pct
        S_norm = S / np.mean(S)
        prune_ratios = np.clip(max_pct * S_norm, 0.0, 0.9)  # never prune more than 90%
        
    for i, layer in enumerate(layers):
        if not hasattr(layer, "self_attn") or getattr(layer, "skip_self_attn", False):
            continue
            
        attn = layer.self_attn
        if not hasattr(attn, "num_heads"):
            logger.warning(f"Layer {i} lacks num_heads, skipping head pruning.")
            continue
            
        n_heads = attn.num_heads
        n_prune = int(n_heads * prune_ratios[i])
        n_keep = n_heads - n_prune
        
        if n_prune <= 0:
            continue
            
        # Get importance scores for heads in this layer
        scores = head_importance_result.importance_scores[i]
        
        # Rank heads in layer (ascending importance)
        ranked_heads = np.argsort(scores)
        keep_heads_idx = sorted(ranked_heads[n_prune:])
        
        logger.info(f"Layer {i}: pruning {n_prune} heads, keeping {n_keep}")
        
        # We now have the indices of the heads we want to keep.
        # Since we are operating on an unknown custom attention module (likely with a ModuleList in out_projs),
        # we will simply log the desired indices for the moment. Proper module replacement
        # requires modifying the underlying q/k/v/o linear weights inside `attn`.
        attn.num_heads = n_keep
        
        d_head = attn.embed_dim // n_heads
        new_embed_dim = n_keep * d_head
        attn.embed_dim = new_embed_dim
        
        # Example of how we might shrink standard specific linear projections:
        for attr_name in ['q_proj', 'k_proj', 'v_proj', 'in_proj']:
            if hasattr(attn, attr_name):
                proj = getattr(attn, attr_name)
                if isinstance(proj, nn.Linear):
                    _shrink_linear_for_heads(proj, keep_heads_idx, d_head, is_out_proj=False)
                    
        # out_proj is the projection back to d_model
        if hasattr(attn, 'out_proj') and isinstance(attn.out_proj, nn.Linear):
            _shrink_linear_for_heads(attn.out_proj, keep_heads_idx, d_head, is_out_proj=True)
                    
        # Moshi uses a ModuleList for out_projs, we just need to shrink the individual projections
        if hasattr(attn, "out_projs") and isinstance(attn.out_projs, nn.ModuleList):
            for proj in attn.out_projs:
                if isinstance(proj, nn.Linear):
                     _shrink_linear_for_heads(proj, keep_heads_idx, d_head, is_out_proj=True)

        # Moshi uses a ModuleList for in_projs: these combine Q, K, V.
        # The output projection shape is [3 * num_heads * d_head, in_features]
        # and it's reshaped internal to Moshi as (p h d) where p=3 (q,k,v).
        if hasattr(attn, "in_projs") and isinstance(attn.in_projs, nn.ModuleList):
            for proj in attn.in_projs:
                if isinstance(proj, nn.Linear):
                    _shrink_qkv_in_proj(proj, keep_heads_idx, d_head, original_n_heads=n_heads)

    return model

def _shrink_qkv_in_proj(linear: nn.Linear, keep_heads: list[int], d_head: int, original_n_heads: int):
    """Shrinks a combined QKV projection (Moshi's in_projs)."""
    in_features = linear.in_features
    # Weight shape is [3 * original_n_heads * d_head, in_features]
    w = linear.weight.view(3, original_n_heads, d_head, in_features)
    
    # Slice the heads: keep all 3 of Q,K,V (dim=0), select heads (dim=1), keep all d_head (dim=2), all in_features (dim=3)
    w_new = w[:, keep_heads, :, :].reshape(-1, in_features)
    
    b_new = None
    if linear.bias is not None:
        b = linear.bias.view(3, original_n_heads, d_head)
        b_new = b[:, keep_heads, :].reshape(-1)
        
    linear.out_features = 3 * len(keep_heads) * d_head
    linear.weight = nn.Parameter(w_new)
    if b_new is not None:
        linear.bias = nn.Parameter(b_new)

def _shrink_linear_for_heads(linear: nn.Linear, keep_heads: list[int], d_head: int, is_out_proj: bool = False):
    """Shrinks a linear layer's weight/bias by selecting only the specific heads."""
    in_features = linear.in_features
    out_features = linear.out_features
    
    if not is_out_proj:
        # If this projection maps to heads (e.g. q/k/v), we prune output features
        n_heads = out_features // d_head
        if out_features % d_head == 0 and n_heads > max(keep_heads):
            # Weight shape is [out_features, in_features]
            w = linear.weight.view(n_heads, d_head, in_features)
            w_new = w[keep_heads].view(-1, in_features)
            
            b_new = None
            if linear.bias is not None:
                b = linear.bias.view(n_heads, d_head)
                b_new = b[keep_heads].view(-1)
                
            linear.out_features = len(keep_heads) * d_head
            linear.weight = nn.Parameter(w_new)
            if b_new is not None:
                linear.bias = nn.Parameter(b_new)
            
    else:
        # If this is the output projection (out_proj), we prune input features
        n_heads = in_features // d_head
        if in_features % d_head == 0 and n_heads > max(keep_heads):
            w = linear.weight.view(out_features, n_heads, d_head)
            w_new = w[:, keep_heads, :].reshape(out_features, -1)
            
            linear.in_features = len(keep_heads) * d_head
            linear.weight = nn.Parameter(w_new)
