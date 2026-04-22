"""LaCo-style Layer Collapse.

Instead of hard removal, layers nominated for pruning have their weights
merged into the adjacent layer, preserving some of the residual contribution.
"""

import logging
import torch
import torch.nn as nn
from moshilite.analysis.moshi_hooks import _get_transformer

logger = logging.getLogger(__name__)


def collapse_layers(model: nn.Module, layers_to_collapse: list[int]) -> nn.Module:
    """Apply Layer Collapse (LaCo).
    
    For each layer index in `layers_to_collapse`, its parameter weights are
    added/averaged into the adjacent preceding layer (or next layer if index is 0).
    After the weights are merged, the layer is removed from the network structure.
    
    Args:
        model: Moshi LMModel.
        layers_to_collapse: List of layer indices.
        
    Returns:
        The mutated model with layers collapsed.
    """
    transformer = _get_transformer(model)
    layers = list(transformer.layers)
    total_layers = len(layers)
    
    if not layers_to_collapse:
        return model
        
    # Keep track of the layers we will ultimately keep
    keep_indices = [i for i in range(total_layers) if i not in layers_to_collapse]
    
    # To merge effectively, we map each dropped layer to its closest kept neighbor
    for drop_idx in layers_to_collapse:
        # Find closest neighbor to merge into
        # Prefer preceding layer, but use successive if preceding isn't kept (e.g. index 0)
        preceding = [i for i in keep_indices if i < drop_idx]
        successive = [i for i in keep_indices if i > drop_idx]
        
        target_idx = -1
        if preceding:
            target_idx = preceding[-1]
        elif successive:
            target_idx = successive[0]
            
        if target_idx != -1:
            logger.info(f"Collapsing layer {drop_idx} into layer {target_idx}")
            _merge_layer_weights(layers[drop_idx], layers[target_idx])
            
    # Rebuild module list with only kept layers
    new_layers = nn.ModuleList([layers[i] for i in keep_indices])
    transformer.layers = new_layers
    
    return model


def _merge_layer_weights(src_layer: nn.Module, dst_layer: nn.Module):
    """Averages the weights from src_layer into dst_layer.
    
    Both layers must have the identical architecture.
    """
    with torch.no_grad():
        for (src_name, src_param), (dst_name, dst_param) in zip(src_layer.named_parameters(), dst_layer.named_parameters()):
            if src_name == dst_name and src_param.shape == dst_param.shape:
                # Average the parameters to blend the representations
                dst_param.copy_((src_param + dst_param) / 2.0)
