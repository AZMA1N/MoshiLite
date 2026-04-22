"""Two-round variant pruning orchestrator.

Executes the pruning comparison protocol:
Round 1: 5 depth strategies + non-uniform width pruning
Round 2: Top 2 depth strategies + uniform width pruning
"""

import logging
import json
from pathlib import Path
import copy
import torch.nn as nn
import numpy as np

from moshilite.analysis.dual_stream import DSRResult
from moshilite.analysis.head_importance import HeadImportanceResult
from moshilite.analysis.ffn_importance import FFNImportanceResult
from moshilite.analysis.block_influence import BIResult
from moshilite.pruning.depth_pruning import prune_layers, get_scattered_prune_indices, get_contiguous_prune_block
from moshilite.pruning.layer_collapse import collapse_layers
from moshilite.pruning.head_pruning import prune_heads
from moshilite.pruning.ffn_pruning import prune_ffn_channels
from moshilite.eval.quick_eval import run_quick_eval
from moshilite.utils.experiment import save_eval_results

logger = logging.getLogger(__name__)


def run_variant_eval_protocol(
    base_model: nn.Module,
    experiment_id: str,
    dsr_result: DSRResult,
    bi_single: BIResult,
    head_importance: HeadImportanceResult,
    ffn_importance: FFNImportanceResult,
    config: dict,
    eval_data: list,
    full_model_targets: list = None
):
    """Executes the two-round Phase C eval protocol for pruning variants.
    
    Args:
        base_model: The original Moshi model (unpruned).
        experiment_id: ID for saving results.
        dsr_result: Parsed layer tags.
        bi_single: BI results for calculating cosine similarities.
        head_importance: Pruning rankings for attention heads.
        ffn_importance: Pruning rankings for FFN channels.
        config: Loaded stage2_pruning.yaml config dict.
        eval_data: Batch list [B, K, T] for fast eval.
        full_model_targets: Precomputed logits from the full model on eval_data.
    """
    p_config = config.get("pruning", {})
    d_config = config.get("depth", {})
    w_config = config.get("width", {})
    
    max_layer_pct = p_config.get("max_layer_pct", 0.3)
    max_head_pct = p_config.get("max_head_pct", 0.3)
    max_ffn_pct = p_config.get("max_ffn_pct", 0.3)
    
    num_layers = len(dsr_result.layer_tags)
    num_layers_to_prune = int(num_layers * max_layer_pct)
    
    # Pre-calculate layer cosine similarities for non-uniform pruning
    layer_cosine_sims = bi_single.cosine_similarities
    
    # ─── ROUND 1: Depth Sweep (Non-Uniform Width) ───
    logger.info("Starting Round 1: Depth Sweep (Non-Uniform Width)")
    
    variants_r1 = {}
    
    # Build strategy variants definitions
    strategies = {
        "v1_scattered": get_scattered_prune_indices(dsr_result, num_layers_to_prune),
        "v2a_cont_strict": get_contiguous_prune_block(dsr_result, num_layers_to_prune, "strict"),
        "v2b_cont_penalized": get_contiguous_prune_block(dsr_result, num_layers_to_prune, "penalized", penalty=d_config.get("contiguous_penalty", 10.0)),
        "v2c_cont_relaxed": get_contiguous_prune_block(dsr_result, num_layers_to_prune, "relaxed"),
        "v3_collapse": get_scattered_prune_indices(dsr_result, num_layers_to_prune)  # Uses scattered selection but collapses
    }
    
    metrics_r1 = {}
    
    for name, indices in strategies.items():
        if not indices:
            logger.warning(f"Strategy {name} produced no valid indices. Skipping.")
            continue
            
        logger.info(f"Evaluating {name}...")
        # Clone base model to prevent cumulative pruning
        m_variant = copy.deepcopy(base_model)
        
        # Apply depth pruning
        if name == "v3_collapse":
            m_variant = collapse_layers(m_variant, indices)
        else:
            m_variant = prune_layers(m_variant, indices)
            
        # Apply non_uniform width pruning
        m_variant = prune_heads(m_variant, head_importance, max_head_pct, "non_uniform", layer_cosine_sims)
        m_variant = prune_ffn_channels(m_variant, ffn_importance, max_ffn_pct, "non_uniform", layer_cosine_sims)
        
        # Evaluate
        res = run_quick_eval(m_variant, eval_data, full_model_targets)
        metrics_r1[name] = _metric_to_dict(res)
        
        # Free memory aggressively to prevent OOM
        del m_variant
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Rank Round 1
    # Sort by Mean Codebook Accuracy (descending) as primary metric
    ranked_r1 = sorted(metrics_r1.items(), key=lambda x: x[1]["mean_codebook_acc"], reverse=True)
    top_2_names = [k for k, v in ranked_r1[:2]]
    
    logger.info(f"Top 2 depth strategies: {top_2_names}")
    
    # ─── ROUND 2: Width Sweep (Uniform) on Top 2 ───
    logger.info("Starting Round 2: Width Sweep (Uniform)")
    
    metrics_r2 = {}
    
    for name in top_2_names:
        uni_name = name + "_uniform"
        indices = strategies[name]
        
        logger.info(f"Evaluating {uni_name}...")
        import gc
        import torch
        m_variant = copy.deepcopy(base_model)
        
        if name == "v3_collapse":
            m_variant = collapse_layers(m_variant, indices)
        else:
            m_variant = prune_layers(m_variant, indices)
            
        # Apply uniform width pruning
        m_variant = prune_heads(m_variant, head_importance, max_head_pct, "uniform")
        m_variant = prune_ffn_channels(m_variant, ffn_importance, max_ffn_pct, "uniform")
        
        res = run_quick_eval(m_variant, eval_data, full_model_targets)
        metrics_r2[uni_name] = _metric_to_dict(res)
        
        del m_variant
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    # Compile final report
    final_report = {
        "round_1_depth_sweep": metrics_r1,
        "round_1_ranking": [k for k, v in ranked_r1],
        "round_2_width_sweep": metrics_r2,
        "recommendation": ranked_r1[0][0]  # The absolutely best variant
    }
    
    # Save using experiment utility
    save_eval_results(experiment_id, "stage2_post_prune", final_report)
    return final_report

def _metric_to_dict(metric) -> dict:
    return {
        "params_b": metric.params_b,
        "text_perplexity": metric.text_perplexity,
        "mean_codebook_acc": metric.mean_codebook_acc,
        "output_cosine_sim": metric.output_cosine_sim
    }
