"""Tests for Moshi-specific hook implementations.

These tests use mock models (not the real Moshi model) to verify
that hooks register correctly, return correct shapes, and handle
edge cases properly.
"""

import io
import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn


# ──────────────────────────────────────────────────────────
#  Mock Model that mimics Moshi's StreamingTransformerLayer
# ──────────────────────────────────────────────────────────


class MockSelfAttn(nn.Module):
    """Mimics StreamingMultiheadAttention."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = n_heads
        self.out_projs = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)])

    def forward(self, q, k, v):
        return self.out_projs[0](q)


class MockGating(nn.Module):
    """Mimics SiGLU gating."""

    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ffn, bias=False)
        self.up = nn.Linear(d_model, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        return self.down(torch.sigmoid(self.gate(x)) * self.up(x))


class MockTransformerLayer(nn.Module):
    """Mimics StreamingTransformerLayer."""

    def __init__(self, d_model, n_heads, d_ffn):
        super().__init__()
        self.self_attn = MockSelfAttn(d_model, n_heads)
        self.skip_self_attn = False
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gating = MockGating(d_model, d_ffn)
        self.layer_scale_1 = nn.Identity()
        self.layer_scale_2 = nn.Identity()

    def forward(self, x, cross_attention_src=None):
        # Simplified: pre-norm residual
        h = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        h = h + self.gating(self.norm2(h))
        return h


class MockTransformer(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ffn):
        super().__init__()
        self.layers = nn.ModuleList(
            [MockTransformerLayer(d_model, n_heads, d_ffn) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MockLMOutput:
    def __init__(self, logits, text_logits, mask, text_mask):
        self.logits = logits
        self.text_logits = text_logits
        self.mask = mask
        self.text_mask = text_mask


class MockLMModel(nn.Module):
    """Mimics Moshi LMModel with transformer + embeddings + output heads."""

    def __init__(self, n_layers=4, d_model=64, n_heads=4, d_ffn=128,
                 n_q=8, card=32, text_card=100):
        super().__init__()
        self.dim = d_model
        self.n_q = n_q
        self.card = card
        self.text_card = text_card
        self.transformer = MockTransformer(n_layers, d_model, n_heads, d_ffn)
        self.emb = nn.ModuleList([nn.Embedding(card + 1, d_model) for _ in range(n_q)])
        self.text_emb = nn.Embedding(text_card + 1, d_model)
        self.text_linear = nn.Linear(d_model, text_card)
        self.linears = nn.ModuleList([nn.Linear(d_model, card) for _ in range(n_q)])

    def forward(self, codes):
        B, K, T = codes.shape
        # Embed: sum text + audio embeddings
        text_tokens = codes[:, 0].clamp(0, self.text_card)
        x = self.text_emb(text_tokens)  # [B, T, d_model]
        for i in range(min(K - 1, self.n_q)):
            audio_tokens = codes[:, i + 1].clamp(0, self.card)
            x = x + self.emb[i](audio_tokens)

        # Run transformer
        x = self.transformer(x)

        # Output heads
        text_logits = self.text_linear(x).unsqueeze(1)  # [B, 1, T, text_card]
        audio_logits = torch.stack([lin(x) for lin in self.linears], dim=1)  # [B, K, T, card]

        mask = torch.ones(B, self.n_q, T, dtype=torch.bool, device=codes.device)
        text_mask = torch.ones(B, 1, T, dtype=torch.bool, device=codes.device)

        return MockLMOutput(audio_logits, text_logits, mask, text_mask)


# ──────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    return MockLMModel(n_layers=4, d_model=64, n_heads=4, d_ffn=128).eval()


@pytest.fixture
def mock_batch():
    return torch.randint(0, 30, (2, 9, 50))  # [B=2, K=9, T=50]


def test_layer_hook_fn_returns_correct_count(mock_model, mock_batch):
    """Layer hooks should return n_layers + 1 hidden states."""
    from moshilite.analysis.moshi_hooks import moshi_layer_hook_fn

    hidden_states = moshi_layer_hook_fn(mock_model, mock_batch)
    assert len(hidden_states) == 5  # 4 layers + 1 input


def test_layer_hook_fn_shapes(mock_model, mock_batch):
    """Each hidden state should be [B, T, d_model]."""
    from moshilite.analysis.moshi_hooks import moshi_layer_hook_fn

    hidden_states = moshi_layer_hook_fn(mock_model, mock_batch)
    for hs in hidden_states:
        assert hs.shape == (2, 50, 64)


def test_ffn_hook_fn_shape(mock_model, mock_batch):
    """FFN hook should return [B*T, d_ffn] tensor."""
    from moshilite.analysis.moshi_hooks import moshi_ffn_hook_fn

    acts = moshi_ffn_hook_fn(mock_model, mock_batch, layer_idx=0)
    assert acts.dim() == 2
    assert acts.shape[0] == 2 * 50  # B * T
    assert acts.shape[1] == 128  # d_ffn


def test_ffn_hook_multiple_layers(mock_model, mock_batch):
    """FFN hook should work for different layer indices."""
    from moshilite.analysis.moshi_hooks import moshi_ffn_hook_fn

    for layer_idx in range(4):
        acts = moshi_ffn_hook_fn(mock_model, mock_batch, layer_idx=layer_idx)
        assert acts.shape == (100, 128)


def test_get_logits_shapes(mock_model, mock_batch):
    """get_logits should return (audio_logits, text_logits) with correct shapes."""
    from moshilite.analysis.moshi_hooks import moshi_get_logits

    audio, text = moshi_get_logits(mock_model, mock_batch)
    assert audio.shape == (2, 8, 50, 32)   # [B, n_q, T, card]
    assert text.shape == (2, 1, 50, 100)   # [B, 1, T, text_card]


def test_get_hidden_states(mock_model, mock_batch):
    """get_hidden_states should return states at requested layers."""
    from moshilite.analysis.moshi_hooks import moshi_get_hidden_states

    states = moshi_get_hidden_states(mock_model, mock_batch, [0, 2, 3])
    assert len(states) == 3
    for s in states:
        assert s.shape == (2, 50, 64)


def test_get_n_layers(mock_model):
    from moshilite.analysis.moshi_hooks import get_n_layers
    assert get_n_layers(mock_model) == 4


def test_get_n_heads(mock_model):
    from moshilite.analysis.moshi_hooks import get_n_heads
    assert get_n_heads(mock_model) == 4


def test_prepare_token_batches():
    """Test loading tokens from a tar shard."""
    from moshilite.analysis.moshi_hooks import prepare_token_batches

    with tempfile.TemporaryDirectory() as tmp:
        # Create a fake tar shard with token arrays
        tar_path = Path(tmp) / "shard-000000.tar"
        with tarfile.open(str(tar_path), "w") as tf:
            for i in range(10):
                # Create a token array [8, 100] (8 codebooks, 100 timesteps)
                tokens = np.random.randint(0, 1024, (8, 100), dtype=np.int64)
                buf = io.BytesIO()
                np.save(buf, tokens)
                buf.seek(0)
                info = tarfile.TarInfo(name=f"sample_{i:04d}.tokens.npy")
                info.size = len(buf.getvalue())
                tf.addfile(info, buf)

                # Also add a metadata file (should be skipped)
                meta = json.dumps({"duration_s": 8.0}).encode()
                meta_info = tarfile.TarInfo(name=f"sample_{i:04d}.meta.json")
                meta_info.size = len(meta)
                tf.addfile(meta_info, io.BytesIO(meta))

        batches = prepare_token_batches(
            tmp, n_samples=8, seq_len=50, n_codebooks=9, batch_size=4
        )
        assert len(batches) == 2  # 8 samples / 4 batch_size
        assert batches[0].shape == (4, 9, 50)
        assert batches[0].dtype == torch.long


def test_prepare_token_batches_padding():
    """Tokens shorter than seq_len should be zero-padded."""
    from moshilite.analysis.moshi_hooks import prepare_token_batches

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "shard-000000.tar"
        with tarfile.open(str(tar_path), "w") as tf:
            # Short token sequence
            tokens = np.ones((8, 10), dtype=np.int64)
            buf = io.BytesIO()
            np.save(buf, tokens)
            buf.seek(0)
            info = tarfile.TarInfo(name="short.tokens.npy")
            info.size = len(buf.getvalue())
            tf.addfile(info, buf)

        batches = prepare_token_batches(
            tmp, n_samples=1, seq_len=50, n_codebooks=9, batch_size=1
        )
        assert batches[0].shape == (1, 9, 50)
        # First 10 timesteps should be 1, rest 0 (padded)
        assert batches[0][0, 0, 9].item() == 1
        assert batches[0][0, 0, 10].item() == 0


def test_model_info():
    """get_model_info should extract architecture details."""
    from moshilite.analysis.moshi_model import get_model_info

    model = MockLMModel(n_layers=4, d_model=64, n_heads=4, d_ffn=128)
    info = get_model_info(model)
    assert info["n_layers"] == 4
    assert info["n_q"] == 8
    assert info["d_model"] == 64


def test_bi_scores_with_mock_model(mock_model, mock_batch):
    """End-to-end: compute BI scores using mock model + hooks."""
    from moshilite.analysis.block_influence import compute_bi_scores
    from moshilite.analysis.moshi_hooks import moshi_layer_hook_fn

    result = compute_bi_scores(
        mock_model, [mock_batch], moshi_layer_hook_fn,
        device="cpu", compute_pairwise=True,
    )
    assert result.bi_scores.shape == (4,)
    assert result.cosine_similarities.shape == (4,)
    assert result.pairwise_cosine_matrix.shape == (5, 5)
    # All BI scores should be between 0 and 2 (1 - cos, cos in [-1,1])
    assert np.all(result.bi_scores >= 0)
    assert np.all(result.bi_scores <= 2)


def test_full_dsr_pipeline_with_mock(mock_model):
    """End-to-end: BI → DSR → layer tags using mock model."""
    from moshilite.analysis.block_influence import compute_bi_scores
    from moshilite.analysis.dual_stream import tag_layers
    from moshilite.analysis.moshi_hooks import moshi_layer_hook_fn

    single_batch = torch.randint(0, 30, (2, 9, 50))
    dialogue_batch = torch.randint(0, 30, (2, 9, 50))

    bi_single = compute_bi_scores(
        mock_model, [single_batch], moshi_layer_hook_fn, device="cpu"
    )
    bi_dialogue = compute_bi_scores(
        mock_model, [dialogue_batch], moshi_layer_hook_fn, device="cpu"
    )

    thresholds = {
        "dual_stream_critical_dsr": 2.0,
        "dual_stream_critical_bi_top_pct": 0.20,
        "dialogue_critical_dsr": 1.5,
        "dialogue_critical_bi_top_pct": 0.30,
        "fallback_critical_pct": 0.70,
    }

    result = tag_layers(bi_single, bi_dialogue, thresholds)
    assert len(result.layer_tags) == 4
    # Every layer should have a valid tag
    for lt in result.layer_tags:
        assert lt.tag in ("general", "dialogue-critical", "dual-stream-critical")
