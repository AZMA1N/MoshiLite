"""Tests for dual-stream criticality tagging logic."""

import json
import numpy as np
import tempfile
from pathlib import Path

from moshilite.analysis.block_influence import BIResult
from moshilite.analysis.dual_stream import (
    compute_dsr,
    tag_layers,
    save_layer_tags,
    load_layer_tags,
    TAG_DUAL_STREAM_CRITICAL,
    TAG_DIALOGUE_CRITICAL,
    TAG_GENERAL,
)


DEFAULT_THRESHOLDS = {
    "dual_stream_critical_dsr": 2.0,
    "dual_stream_critical_bi_top_pct": 0.20,
    "dialogue_critical_dsr": 1.5,
    "dialogue_critical_bi_top_pct": 0.30,
    "fallback_critical_pct": 0.70,
}


def _make_bi(scores):
    return BIResult(
        bi_scores=np.array(scores, dtype=np.float64),
        cosine_similarities=1 - np.array(scores, dtype=np.float64),
    )


def test_compute_dsr():
    bi_single = _make_bi([0.1, 0.2, 0.3, 0.4])
    bi_dialogue = _make_bi([0.2, 0.6, 0.3, 0.8])
    dsr = compute_dsr(bi_single, bi_dialogue)
    assert dsr.shape == (4,)
    np.testing.assert_allclose(dsr[0], 2.0, rtol=0.01)
    np.testing.assert_allclose(dsr[1], 3.0, rtol=0.01)
    np.testing.assert_allclose(dsr[2], 1.0, rtol=0.01)
    np.testing.assert_allclose(dsr[3], 2.0, rtol=0.01)


def test_tag_layers_uses_all_tags():
    """With 10 layers and carefully set scores, all 3 tags appear."""
    # 10 layers: craft scores so thresholds are stable
    bi_single = _make_bi([0.10, 0.10, 0.20, 0.20, 0.30, 0.30, 0.10, 0.10, 0.20, 0.20])
    bi_dialogue = _make_bi([0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.90, 0.95, 0.80, 0.85])
    #                       DSR:  0.5   0.5   0.5   0.5   0.5   0.5   9.0   9.5   4.0   4.25
    #                       tag:  gen   gen   gen   gen   gen   gen   dsc?  dsc?  dc?   dc?
    # Top 20% BI_dialogue threshold (80th percentile of [0.05..0.95]) ≈ 0.88
    # Top 30% BI_dialogue threshold (70th percentile) ≈ 0.84
    # Layer 6: DSR=9≥2, BI=0.90≥0.88 → dual-stream-critical
    # Layer 7: DSR=9.5≥2, BI=0.95≥0.88 → dual-stream-critical
    # Layer 8: DSR=4≥2, BI=0.80<0.88 → not dsc, but DSR≥1.5 → dialogue-critical
    # Layer 9: DSR=4.25≥2, BI=0.85≥0.84(top30) → dialogue-critical (via BI or DSR)
    # Layers 0-5: DSR<1.5, BI<0.84 → general

    result = tag_layers(bi_single, bi_dialogue, DEFAULT_THRESHOLDS)
    tags = {lt.layer_index: lt.tag for lt in result.layer_tags}

    # Layers 0-5 should be general
    for i in range(6):
        assert tags[i] == TAG_GENERAL, f"Layer {i}: expected general, got {tags[i]}"

    # Layers 6-7 should be dual-stream-critical
    assert tags[6] == TAG_DUAL_STREAM_CRITICAL
    assert tags[7] == TAG_DUAL_STREAM_CRITICAL

    # Layers 8-9 should be some flavor of critical
    assert tags[8] in (TAG_DIALOGUE_CRITICAL, TAG_DUAL_STREAM_CRITICAL)
    assert tags[9] in (TAG_DIALOGUE_CRITICAL, TAG_DUAL_STREAM_CRITICAL)

    assert not result.fallback_triggered


def test_fallback_when_too_many_critical():
    """All layers have high DSR → >70% critical → fallback triggered."""
    bi_single = _make_bi([0.01] * 10)
    bi_dialogue = _make_bi([0.5] * 10)  # All DSR = 50.0
    result = tag_layers(bi_single, bi_dialogue, DEFAULT_THRESHOLDS)

    assert result.fallback_triggered
    general = result.get_pruneable_layers()
    assert len(general) == 3  # bottom 30% of 10 = 3 layers


def test_save_load_roundtrip():
    bi_single = _make_bi([0.1, 0.2, 0.3])
    bi_dialogue = _make_bi([0.3, 0.4, 0.1])
    result = tag_layers(bi_single, bi_dialogue, DEFAULT_THRESHOLDS)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "layer_tags.json"
        save_layer_tags(result, path)
        loaded = load_layer_tags(path)

    assert len(loaded.layer_tags) == len(result.layer_tags)
    for orig, loaded_tag in zip(result.layer_tags, loaded.layer_tags):
        assert orig.tag == loaded_tag.tag
        assert orig.layer_index == loaded_tag.layer_index


def test_get_pruneable_layers():
    bi_single = _make_bi([0.1, 0.2, 0.3, 0.4, 0.5])
    bi_dialogue = _make_bi([0.1, 0.2, 0.3, 0.4, 0.5])
    result = tag_layers(bi_single, bi_dialogue, DEFAULT_THRESHOLDS)
    pruneable = result.get_pruneable_layers()
    critical = result.get_critical_layers()
    # Every layer should be in exactly one category
    assert len(pruneable) + len(critical) == 5
