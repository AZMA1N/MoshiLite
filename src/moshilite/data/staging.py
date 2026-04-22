"""GDrive shard staging for Colab training sessions.

Copies token + teacher target shards from GDrive to local NVMe disk at
the start of each Colab session, so training never reads from GDrive
during gradient updates.
"""

import shutil
from pathlib import Path


def stage_shards_for_session(
    shard_manifest: list[str],
    start_index: int,
    max_local_gb: float = 40.0,
    local_dir: str = "/content/staged",
) -> tuple[list[str], int]:
    """Copy shards from GDrive to local disk until budget is exhausted.

    Args:
        shard_manifest: Ordered list of all shard paths on GDrive.
        start_index: Resume from this shard index (saved in checkpoint).
        max_local_gb: Max local disk budget in GB.
        local_dir: Local directory to stage shards into.

    Returns:
        Tuple of (list of local shard paths, next_start_index).
    """
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    staged = []
    total_bytes = 0

    for i in range(start_index, len(shard_manifest)):
        src = Path(shard_manifest[i])
        size = src.stat().st_size
        if total_bytes + size > max_local_gb * 1e9:
            break
        dst = local / src.name
        shutil.copy2(src, dst)
        staged.append(str(dst))
        total_bytes += size

    print(f"✅ Staged {len(staged)} shards ({total_bytes / 1e9:.1f} GB) to {local_dir}")
    return staged, start_index + len(staged)


def cleanup_staged(local_dir: str = "/content/staged"):
    """Remove all staged shards from local disk."""
    local = Path(local_dir)
    if local.exists():
        shutil.rmtree(local)
        print(f"🧹 Cleaned up staged shards from {local_dir}")
