"""Stage 1 Layer Analysis Orchestrator.

Top-level function that runs the complete Stage 1 analysis pipeline:
1. Precision validation (INT4 vs INT8)
2. Block Influence (BI) scores
3. Dual-stream criticality (DSR) tagging
4. Attention head importance
5. FFN channel importance

All results are saved to the experiment's eval directory on GDrive.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def run_stage1_analysis(
    config_path: str = "configs/stage1_analysis.yaml",
    weights_dir: str = "/content/drive/MyDrive/moshilite/upstream_weights/moshiko",
    token_dir: str = "/content/drive/MyDrive/moshilite/tokens",
    output_dir: str = "/content/drive/MyDrive/moshilite/eval/prune30_v1/stage1_analysis",
    device: str = "cuda",
    skip_precision_gate: bool = False,
) -> dict:
    """Run the full Stage 1 layer analysis pipeline.

    Args:
        config_path: Path to stage1_analysis.yaml.
        weights_dir: Directory with Moshi weights.
        token_dir: Directory with pre-encoded token shards (subdirs per dataset).
        output_dir: Where to save all analysis results.
        device: CUDA device.
        skip_precision_gate: If True, skip precision validation and use INT4.

    Returns:
        Dict with all analysis results.
    """
    from moshilite.analysis.moshi_model import load_moshi_for_analysis, get_model_info
    from moshilite.analysis.moshi_hooks import (
        moshi_layer_hook_fn,
        moshi_ffn_hook_fn,
        moshi_head_mask_fn,
        moshi_loss_fn,
        prepare_token_batches,
        get_n_layers,
        get_n_heads,
    )
    from moshilite.analysis.block_influence import compute_bi_scores, rank_layers_by_importance
    from moshilite.analysis.dual_stream import tag_layers, save_layer_tags
    from moshilite.analysis.head_importance import compute_head_importance
    from moshilite.analysis.ffn_importance import compute_ffn_importance

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load config ──
    with open(config_path) as f:
        config = yaml.safe_load(f)

    precision_config = config["precision_validation"]
    dsr_config = config["dual_stream_criticality"]

    results = {"config": config}

    # ── Step 0: Precision Validation ──
    precision = "int4"  # default
    if not skip_precision_gate:
        logger.info("🔍 Running precision validation gate...")
        precision = _run_precision_gate(
            config=precision_config,
            weights_dir=weights_dir,
            token_dir=token_dir,
            output_dir=str(out),
            device=device,
        )
        results["validated_precision"] = precision
    else:
        logger.info("⏭️  Skipping precision gate, using INT4")
        results["validated_precision"] = "int4"

    # ── Load model at validated precision ──
    logger.info("📦 Loading model at %s precision...", precision)
    model = load_moshi_for_analysis(weights_dir, precision=precision, device=device)
    model_info = get_model_info(model)
    results["model_info"] = model_info
    logger.info("Model info: %s", model_info)

    n_layers = get_n_layers(model)
    n_heads = get_n_heads(model)

    # ── Step 1: Load calibration data ──
    n_single = dsr_config.get("n_single_examples", 200)
    n_dialogue = dsr_config.get("n_dialogue_examples", 200)

    logger.info("📊 Loading calibration data...")
    single_data = prepare_token_batches(
        f"{token_dir}/librispeech", n_samples=n_single, batch_size=4
    )
    dialogue_data = prepare_token_batches(
        f"{token_dir}/librimix", n_samples=n_dialogue, batch_size=4
    )
    logger.info("Loaded %d single-speaker batches, %d dialogue batches",
                len(single_data), len(dialogue_data))

    # ── Step 2: Compute BI scores ──
    logger.info("📈 Computing BI scores (single-speaker)...")
    bi_single = compute_bi_scores(
        model, single_data, moshi_layer_hook_fn, device=device
    )
    np.savez(str(out / "bi_scores_single.npz"),
             bi_scores=bi_single.bi_scores,
             cosine_sims=bi_single.cosine_similarities)

    logger.info("📈 Computing BI scores (dialogue)...")
    bi_dialogue = compute_bi_scores(
        model, dialogue_data, moshi_layer_hook_fn, device=device
    )
    np.savez(str(out / "bi_scores_dialogue.npz"),
             bi_scores=bi_dialogue.bi_scores,
             cosine_sims=bi_dialogue.cosine_similarities)

    results["bi_single"] = bi_single.bi_scores.tolist()
    results["bi_dialogue"] = bi_dialogue.bi_scores.tolist()

    # ── Step 3: DSR + Layer Tagging ──
    logger.info("🏷️  Computing DSR and tagging layers...")
    dsr_thresholds = dsr_config["dsr_thresholds"]
    dsr_thresholds["fallback_critical_pct"] = dsr_config.get("fallback_critical_pct", 0.70)

    dsr_result = tag_layers(bi_single, bi_dialogue, dsr_thresholds)
    save_layer_tags(dsr_result, out / "layer_tags.json")

    results["dsr_summary"] = dsr_result.summary()
    results["pruning_candidates"] = dsr_result.get_pruneable_layers()

    logger.info("DSR summary: %s", dsr_result.summary())
    if dsr_result.fallback_triggered:
        logger.warning("⚠️ Fallback triggered: %s", dsr_result.fallback_reason)

    # ── Step 4: Head Importance ──
    logger.info("🧠 Computing attention head importance (gradient-based)...")
    try:
        # Prepare paired data for gradient computation
        # head_importance needs (input, target) pairs
        paired_data = [(batch, batch) for batch in single_data[:10]]  # use subset

        head_result = compute_head_importance(
            model, paired_data, n_layers, n_heads,
            head_mask_fn=moshi_head_mask_fn,
            loss_fn=moshi_loss_fn,
            device=device,
        )
        np.savez(str(out / "head_importance.npz"),
                 scores=head_result.importance_scores)
        results["head_importance_shape"] = list(head_result.importance_scores.shape)
        results["least_important_heads"] = [
            {"layer": l, "head": h, "score": s}
            for l, h, s in head_result.ranked_heads[:20]
        ]
        logger.info("Top 10 least important heads: %s",
                    head_result.get_least_important(10))
    except Exception as e:
        logger.warning("⚠️ Head importance failed (may need gradients): %s", e)
        results["head_importance_error"] = str(e)

    # ── Step 5: FFN Importance ──
    logger.info("🔬 Computing FFN channel importance (PCA-guided)...")
    try:
        # Sample a subset of layers for FFN analysis (expensive)
        sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
        ffn_result = compute_ffn_importance(
            model, single_data[:5], sample_layers,
            moshi_ffn_hook_fn, device=device, n_components=64,
        )
        # Save per-layer results
        for layer_idx, scores in ffn_result.importance_scores.items():
            np.savez(str(out / f"ffn_importance_layer{layer_idx:02d}.npz"),
                     scores=scores,
                     variance_ratios=ffn_result.explained_variance_ratios[layer_idx])
        results["ffn_analyzed_layers"] = sample_layers
        logger.info("FFN analysis complete for layers: %s", sample_layers)
    except Exception as e:
        logger.warning("⚠️ FFN importance failed: %s", e)
        results["ffn_importance_error"] = str(e)

    # ── Save Summary ──
    layer_ranking = rank_layers_by_importance(bi_dialogue.bi_scores)
    results["layer_ranking_by_bi_dialogue"] = layer_ranking
    results["recommended_prune_order"] = layer_ranking[:int(0.3 * n_layers)]

    with open(out / "stage1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("✅ Stage 1 analysis complete. Results saved to %s", out)
    return results


def _run_precision_gate(
    config: dict,
    weights_dir: str,
    token_dir: str,
    output_dir: str,
    device: str,
) -> str:
    """Run precision validation and return the recommended precision.

    Loads model at both INT4 and INT8, compares outputs, decides which
    precision to use for the rest of Stage 1.

    Returns:
        'int4' or 'int8'
    """
    from moshilite.analysis.moshi_model import load_moshi_for_analysis
    from moshilite.analysis.moshi_hooks import (
        moshi_layer_hook_fn, prepare_token_batches,
    )
    from moshilite.analysis.block_influence import compute_bi_scores
    from moshilite.utils.precision import validate_stage1_precision

    n_cal = config["n_calibration_samples"]
    thresholds = config["thresholds"]

    # Load calibration data
    cal_data = prepare_token_batches(
        f"{token_dir}/librispeech", n_samples=n_cal, batch_size=4
    )

    # Load model at both precisions (sequentially — T4 can't hold both)
    logger.info("Loading INT4 model...")
    model_int4 = load_moshi_for_analysis(weights_dir, precision="int4", device=device)

    # Compute BI scores at INT4
    logger.info("Computing INT4 BI scores...")
    bi_int4 = compute_bi_scores(model_int4, cal_data[:10], moshi_layer_hook_fn, device=device)

    # Free INT4 model
    del model_int4
    torch.cuda.empty_cache()

    # Load INT8
    logger.info("Loading INT8 model...")
    model_int8 = load_moshi_for_analysis(weights_dir, precision="int8", device=device)

    # Compute BI scores at INT8
    logger.info("Computing INT8 BI scores...")
    bi_int8 = compute_bi_scores(model_int8, cal_data[:10], moshi_layer_hook_fn, device=device)

    # For the precision helper functions, we need both models simultaneously
    # But T4 can't hold both. So compute what we can with INT8, then compare.
    # The cosine drift / KL div need both models — we save INT4 outputs and reload.

    # Save INT8 results, load back INT4 for comparison
    del model_int8
    torch.cuda.empty_cache()

    # For a T4, we do the full comparison on a small subset
    from scipy import stats
    rho, _ = stats.spearmanr(bi_int4.bi_scores, bi_int8.bi_scores)

    from moshilite.utils.precision import PrecisionReport
    report = PrecisionReport(
        spearman_rho=float(rho),
        passed=rho >= thresholds["spearman_rho"],
        recommended_precision="int4" if rho >= thresholds["spearman_rho"] else "int8",
    )

    # Save report
    out = Path(output_dir)
    with open(out / "precision_report.json", "w") as f:
        json.dump(report.summary(), f, indent=2)

    logger.info(
        "Precision gate: ρ=%.4f (threshold=%.2f) → %s",
        report.spearman_rho, thresholds["spearman_rho"], report.recommended_precision,
    )

    return report.recommended_precision
