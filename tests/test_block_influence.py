"""Tests for BI score computation and layer ranking."""

import numpy as np
from moshilite.analysis.block_influence import BIResult, rank_layers_by_importance


def test_rank_layers_ascending():
    """Least important layers should come first."""
    bi = np.array([0.3, 0.1, 0.5, 0.2])
    ranked = rank_layers_by_importance(bi)
    assert ranked[0] == 1  # lowest BI = least important
    assert ranked[-1] == 2  # highest BI = most important


def test_bi_result_dataclass():
    result = BIResult(
        bi_scores=np.array([0.1, 0.2, 0.3]),
        cosine_similarities=np.array([0.9, 0.8, 0.7]),
    )
    assert result.bi_scores.shape == (3,)
    assert result.pairwise_cosine_matrix is None
