"""Tests for Mimi encoding pipeline (shard I/O and verification only).

Note: Actual Mimi encoding requires the model on GPU.
These tests verify the serialization, tar shard, and verification logic.
"""

import io
import json
import tarfile
import tempfile
import numpy as np
from pathlib import Path

from moshilite.data.encoding import (
    _serialize_tokens,
    _deserialize_tokens,
    _add_to_tar,
    verify_shard_integrity,
    ShardStats,
)


def test_serialize_roundtrip():
    """Tokens survive serialize → deserialize roundtrip."""
    tokens = np.random.randint(0, 1024, size=(8, 100), dtype=np.int64)
    data = _serialize_tokens(tokens)
    recovered = _deserialize_tokens(data)
    np.testing.assert_array_equal(tokens, recovered)


def test_add_to_tar():
    """Can add data to a tar and read it back."""
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = tmp.name

    with tarfile.open(tmp_path, "w") as tar:
        _add_to_tar(tar, "test.npy", b"hello world")

    with tarfile.open(tmp_path, "r") as tar:
        member = tar.getmembers()[0]
        assert member.name == "test.npy"
        content = tar.extractfile(member).read()
        assert content == b"hello world"

    Path(tmp_path).unlink()


def test_verify_shard_integrity_passes(tmp_path):
    """Verification passes when stats and shards exist."""
    dataset_name = "test_dataset"

    # Create a valid shard
    shard_path = tmp_path / f"{dataset_name}-000000.tar"
    with tarfile.open(str(shard_path), "w") as tar:
        _add_to_tar(tar, "sample.tokens.npy", _serialize_tokens(np.zeros((8, 10))))

    # Create stats file
    stats = {
        "n_files": 5,
        "n_shards": 1,
        "total_duration_hours": 1.5,
        "total_tokens": 400,
        "total_bytes": 1024,
        "shard_paths": [str(shard_path)],
    }
    with open(tmp_path / f"{dataset_name}_stats.json", "w") as f:
        json.dump(stats, f)

    assert verify_shard_integrity(tmp_path, dataset_name) is True


def test_verify_shard_integrity_fails_missing_stats(tmp_path):
    """Verification fails when stats file is missing."""
    assert verify_shard_integrity(tmp_path, "nonexistent") is False


def test_verify_shard_integrity_fails_missing_shard(tmp_path):
    """Verification fails when a referenced shard is missing."""
    dataset_name = "test_dataset"
    stats = {
        "n_files": 5,
        "n_shards": 1,
        "total_duration_hours": 1.5,
        "total_tokens": 400,
        "total_bytes": 1024,
        "shard_paths": [str(tmp_path / "missing_shard.tar")],
    }
    with open(tmp_path / f"{dataset_name}_stats.json", "w") as f:
        json.dump(stats, f)

    assert verify_shard_integrity(tmp_path, dataset_name) is False


def test_verify_shard_integrity_fails_min_files(tmp_path):
    """Verification fails when file count is below minimum."""
    dataset_name = "test_dataset"
    shard_path = tmp_path / f"{dataset_name}-000000.tar"
    with tarfile.open(str(shard_path), "w") as tar:
        _add_to_tar(tar, "sample.tokens.npy", b"data")

    stats = {
        "n_files": 5,
        "n_shards": 1,
        "total_duration_hours": 1.0,
        "total_tokens": 100,
        "total_bytes": 512,
        "shard_paths": [str(shard_path)],
    }
    with open(tmp_path / f"{dataset_name}_stats.json", "w") as f:
        json.dump(stats, f)

    # Expect at least 100 files but only have 5
    assert verify_shard_integrity(
        tmp_path, dataset_name, expected_min_files=100
    ) is False
