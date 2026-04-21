"""Tests for shard staging logic."""

import tempfile
from pathlib import Path

from moshilite.data.staging import stage_shards_for_session, cleanup_staged


def test_stage_shards_basic(tmp_path):
    """Test staging copies shards to local dir and respects budget."""
    # Create fake shards on "GDrive"
    gdrive = tmp_path / "gdrive"
    gdrive.mkdir()
    for i in range(5):
        shard = gdrive / f"shard_{i:04d}.tar"
        shard.write_bytes(b"x" * 1000)  # 1KB each

    local = tmp_path / "staged"
    manifest = [str(gdrive / f"shard_{i:04d}.tar") for i in range(5)]

    # Stage with 3KB budget → should get 3 shards
    staged, next_idx = stage_shards_for_session(
        manifest, start_index=0, max_local_gb=3e-6, local_dir=str(local)
    )
    assert len(staged) == 3
    assert next_idx == 3
    assert all(Path(p).exists() for p in staged)


def test_stage_shards_resume(tmp_path):
    """Test staging resumes from a given start index."""
    gdrive = tmp_path / "gdrive"
    gdrive.mkdir()
    for i in range(5):
        shard = gdrive / f"shard_{i:04d}.tar"
        shard.write_bytes(b"x" * 1000)

    local = tmp_path / "staged"
    manifest = [str(gdrive / f"shard_{i:04d}.tar") for i in range(5)]

    # Start from index 3 → should get shards 3, 4
    staged, next_idx = stage_shards_for_session(
        manifest, start_index=3, max_local_gb=1, local_dir=str(local)
    )
    assert len(staged) == 2
    assert next_idx == 5


def test_cleanup_staged(tmp_path):
    """Test cleanup removes the staged directory."""
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "shard.tar").write_bytes(b"data")
    cleanup_staged(str(staged))
    assert not staged.exists()
