"""Dual-stream criticality analysis.

Computes DSR (Dialogue Sensitivity Ratio) per layer and tags layers
as general, dialogue-critical, or dual-stream-critical using v8 thresholds.

DSR[i] = BI_dialogue[i] / BI_single[i]

Layers with high DSR are disproportionately important for dialogue/overlap
handling and should be protected from pruning.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from moshilite.analysis.block_influence import BIResult


# Layer criticality tags
TAG_DUAL_STREAM_CRITICAL = "dual-stream-critical"
TAG_DIALOGUE_CRITICAL = "dialogue-critical"
TAG_GENERAL = "general"


@dataclass
class LayerTag:
    """Criticality tag for a single layer."""
    layer_index: int
    tag: str
    bi_single: float
    bi_dialogue: float
    dsr: float


@dataclass
class DSRResult:
    """Complete dual-stream criticality analysis result."""
    layer_tags: list[LayerTag]
    dsr_values: np.ndarray
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None

    def get_pruneable_layers(self) -> list[int]:
        """Return indices of layers tagged as 'general' (safe to prune)."""
        return [lt.layer_index for lt in self.layer_tags if lt.tag == TAG_GENERAL]

    def get_critical_layers(self) -> list[int]:
        """Return indices of layers tagged critical (avoid pruning)."""
        return [
            lt.layer_index for lt in self.layer_tags
            if lt.tag in (TAG_DUAL_STREAM_CRITICAL, TAG_DIALOGUE_CRITICAL)
        ]

    def summary(self) -> dict:
        """Summary statistics for logging."""
        tags = [lt.tag for lt in self.layer_tags]
        return {
            "n_layers": len(self.layer_tags),
            "n_general": tags.count(TAG_GENERAL),
            "n_dialogue_critical": tags.count(TAG_DIALOGUE_CRITICAL),
            "n_dual_stream_critical": tags.count(TAG_DUAL_STREAM_CRITICAL),
            "fallback_triggered": self.fallback_triggered,
            "mean_dsr": float(np.mean(self.dsr_values)),
            "max_dsr": float(np.max(self.dsr_values)),
        }


def compute_dsr(
    bi_single: BIResult,
    bi_dialogue: BIResult,
) -> np.ndarray:
    """Compute Dialogue Sensitivity Ratio per layer.

    DSR[i] = BI_dialogue[i] / BI_single[i]

    Args:
        bi_single: BI scores from single-speaker examples.
        bi_dialogue: BI scores from overlapping dialogue examples.

    Returns:
        Array of DSR values, one per layer.
    """
    # Avoid division by zero — use small epsilon for layers with ~0 single BI
    epsilon = 1e-8
    dsr = bi_dialogue.bi_scores / (bi_single.bi_scores + epsilon)
    return dsr


def tag_layers(
    bi_single: BIResult,
    bi_dialogue: BIResult,
    dsr_thresholds: dict,
) -> DSRResult:
    """Tag each layer with its criticality level using v8 criteria.

    Thresholds (from configs/stage1_analysis.yaml):
        dual_stream_critical_dsr: 2.0
        dual_stream_critical_bi_top_pct: 0.20
        dialogue_critical_dsr: 1.5
        dialogue_critical_bi_top_pct: 0.30
        fallback_critical_pct: 0.70

    Args:
        bi_single: BI scores from single-speaker data.
        bi_dialogue: BI scores from dialogue data.
        dsr_thresholds: Dict with threshold values from config.

    Returns:
        DSRResult with per-layer tags and metadata.
    """
    dsr = compute_dsr(bi_single, bi_dialogue)
    n_layers = len(dsr)

    # Extract thresholds
    ds_crit_dsr = dsr_thresholds["dual_stream_critical_dsr"]
    ds_crit_bi_pct = dsr_thresholds["dual_stream_critical_bi_top_pct"]
    dial_crit_dsr = dsr_thresholds["dialogue_critical_dsr"]
    dial_crit_bi_pct = dsr_thresholds["dialogue_critical_bi_top_pct"]
    fallback_pct = dsr_thresholds.get("fallback_critical_pct", 0.70)

    # Compute BI dialogue percentile thresholds
    bi_d = bi_dialogue.bi_scores
    bi_top20_threshold = np.percentile(bi_d, 100 * (1 - ds_crit_bi_pct))
    bi_top30_threshold = np.percentile(bi_d, 100 * (1 - dial_crit_bi_pct))

    # Tag each layer
    layer_tags = []
    for i in range(n_layers):
        # dual-stream-critical: DSR >= 2.0 AND BI_dialogue in top 20%
        if dsr[i] >= ds_crit_dsr and bi_d[i] >= bi_top20_threshold:
            tag = TAG_DUAL_STREAM_CRITICAL
        # dialogue-critical: DSR >= 1.5 OR BI_dialogue in top 30%
        elif dsr[i] >= dial_crit_dsr or bi_d[i] >= bi_top30_threshold:
            tag = TAG_DIALOGUE_CRITICAL
        else:
            tag = TAG_GENERAL

        layer_tags.append(LayerTag(
            layer_index=i,
            tag=tag,
            bi_single=float(bi_single.bi_scores[i]),
            bi_dialogue=float(bi_d[i]),
            dsr=float(dsr[i]),
        ))

    result = DSRResult(layer_tags=layer_tags, dsr_values=dsr)

    # Fallback: if >70% tagged critical, thresholds are poorly calibrated
    n_critical = sum(
        1 for lt in layer_tags
        if lt.tag in (TAG_DUAL_STREAM_CRITICAL, TAG_DIALOGUE_CRITICAL)
    )
    if n_critical / n_layers > fallback_pct:
        result.fallback_triggered = True
        result.fallback_reason = (
            f"{n_critical}/{n_layers} ({n_critical/n_layers:.0%}) layers tagged "
            f"critical, exceeding {fallback_pct:.0%} threshold. "
            f"Falling back to pure BI ranking."
        )
        result.layer_tags = _fallback_bi_ranking(bi_dialogue, n_layers)

    return result


def _fallback_bi_ranking(bi_dialogue: BIResult, n_layers: int) -> list[LayerTag]:
    """Fallback: tag bottom 30% by BI_dialogue as general, rest as critical."""
    bi_d = bi_dialogue.bi_scores
    n_general = max(1, int(0.30 * n_layers))
    sorted_indices = np.argsort(bi_d)  # ascending = lowest BI first

    tags = []
    general_set = set(sorted_indices[:n_general])
    for i in range(n_layers):
        tag = TAG_GENERAL if i in general_set else TAG_DIALOGUE_CRITICAL
        tags.append(LayerTag(
            layer_index=i,
            tag=tag,
            bi_single=0.0,  # not used in fallback
            bi_dialogue=float(bi_d[i]),
            dsr=0.0,  # not used in fallback
        ))
    return tags


def save_layer_tags(result: DSRResult, output_path: str | Path):
    """Save layer tags to JSON file for use by pruning stages."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "layer_tags": [asdict(lt) for lt in result.layer_tags],
        "summary": result.summary(),
        "fallback_triggered": result.fallback_triggered,
        "fallback_reason": result.fallback_reason,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Layer tags saved to {output_path}")


def load_layer_tags(path: str | Path) -> DSRResult:
    """Load layer tags from JSON file."""
    with open(path) as f:
        data = json.load(f)

    layer_tags = [LayerTag(**lt) for lt in data["layer_tags"]]
    dsr_values = np.array([lt.dsr for lt in layer_tags])

    return DSRResult(
        layer_tags=layer_tags,
        dsr_values=dsr_values,
        fallback_triggered=data.get("fallback_triggered", False),
        fallback_reason=data.get("fallback_reason"),
    )
