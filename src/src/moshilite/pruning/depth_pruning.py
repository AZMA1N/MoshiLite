"""Depth pruning strategies: Scattered and Contiguous layer removal.

Reads layer_tags.json and determines which layers to remove based on
the selected strategy (Scattered, Contiguous Strict, Penalized, or Relaxed).
"""

import logging
import torch.nn as nn
import numpy as np
from typing import Optional

from moshilite.analysis.dual_stream import DSRResult, TAG_GENERAL
from moshilite.analysis.moshi_hooks import _get_transformer

logger = logging.getLogger(__name__)


def prune_layers(model: nn.Module, layers_to_prune: list[int]) -> nn.Module:
    """Physically remove specified layers from the model.
    
    Mutates model.transformer.layers to drop the requested indices.
    
    Args:
        model: Moshi LMModel.
        layers_to_prune: List of layer indices to remove.
        
    Returns:
        The mutated model.
    """
    transformer = _get_transformer(model)
    layers = list(transformer.layers)
    total_layers = len(layers)
    
    keep_indices = [i for i in range(total_layers) if i not in layers_to_prune]
    logger.info(f"Pruning {len(layers_to_prune)} layers. Keeping {len(keep_indices)} layers.")
    
    new_layers = nn.ModuleList([layers[i] for i in keep_indices])
    transformer.layers = new_layers
    
    return model


def get_scattered_prune_indices(dsr_result: DSRResult, max_layers_to_prune: int) -> list[int]:
    """Strategy V1: Scattered pruning of least-influence 'general' layers.
    
    Args:
        dsr_result: Evaluated DSR tags.
        max_layers_to_prune: The maximum number of layers to remove.
        
    Returns:
        List of layer indices to prune.
    """
    pruneable = [lt for lt in dsr_result.layer_tags if lt.tag == TAG_GENERAL]
    
    # Sort by dialogue BI scores ascending (lowest influence first)
    pruneable.sort(key=lambda lt: lt.bi_dialogue)
    
    to_prune = [lt.layer_index for lt in pruneable[:max_layers_to_prune]]
    to_prune.sort() # Keep ascending order
    
    return to_prune


def get_contiguous_prune_block(
    dsr_result: DSRResult, 
    num_layers_to_prune: int, 
    mode: str = "strict",
    penalty: float = 10.0
) -> Optional[list[int]]:
    """Strategy V2: Contiguous block pruning.
    
    Finds the best contiguous block of `num_layers_to_prune` to remove.
    The "best" block minimizes a cost function (sum of BI scores).
    
    Modes:
      - 'strict': Block must contain ZERO non-general layers. 
      - 'penalized': Block can contain critical layers, but they add `penalty * N_critical` to cost.
      - 'relaxed': Ignores criticality tags; pure minimum sum of BI_dialogue.
      
    Args:
        dsr_result: Evaluated DSR tags.
        num_layers_to_prune: Exact number of sequential layers to remove.
        mode: 'strict', 'penalized', or 'relaxed'.
        penalty: Added cost per critical layer for 'penalized' mode.
        
    Returns:
        List of layer indices (the block) or None if strict mode fails to find a block.
    """
    n_layers = len(dsr_result.layer_tags)
    if num_layers_to_prune <= 0 or num_layers_to_prune >= n_layers:
        return []
        
    best_cost = float('inf')
    best_start = -1
    
    for start_idx in range(n_layers - num_layers_to_prune + 1):
        block_tags = dsr_result.layer_tags[start_idx : start_idx + num_layers_to_prune]
        
        n_critical = sum(1 for lt in block_tags if lt.tag != TAG_GENERAL)
        sum_bi = sum(lt.bi_dialogue for lt in block_tags)
        
        if mode == "strict" and n_critical > 0:
            continue
            
        if mode == "relaxed":
            cost = sum_bi
        elif mode == "penalized":
            cost = sum_bi + (penalty * n_critical)
        else: # strict
            cost = sum_bi
            
        if cost < best_cost:
            best_cost = cost
            best_start = start_idx
            
    if best_start == -1:
        if mode == "strict":
            logger.warning(f"No strict contiguous block of size {num_layers_to_prune} found.")
            return None
        return []

    return list(range(best_start, best_start + num_layers_to_prune))
