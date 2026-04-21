"""Tests for LibriLight duration-binned speaker sampling."""

import numpy as np
from moshilite.data.librilight_sampler import sample_speakers, SamplingResult


def _make_metadata(n_speakers=50, utterances_per_speaker=10, duration_range=(5, 60)):
    """Create fake speaker metadata for testing."""
    metadata = {}
    for i in range(n_speakers):
        speaker_id = f"speaker_{i:04d}"
        durations = np.random.uniform(
            duration_range[0], duration_range[1], utterances_per_speaker
        )
        metadata[speaker_id] = [
            {"file": f"/data/{speaker_id}/utt_{j}.flac", "duration_s": float(d), "split": "small"}
            for j, d in enumerate(durations)
        ]
    return metadata


def test_sample_speakers_basic():
    """Sampling returns speakers and files."""
    metadata = _make_metadata(n_speakers=100, utterances_per_speaker=20)
    result = sample_speakers(metadata, target_hours=5.0, n_bins=5)

    assert result.n_speakers > 0
    assert result.n_files > 0
    assert result.total_duration_hours > 0
    assert len(result.bin_stats) == 5


def test_sample_speakers_covers_all_bins():
    """Each bin contributes at least some speakers."""
    metadata = _make_metadata(n_speakers=100, utterances_per_speaker=20)
    result = sample_speakers(metadata, target_hours=10.0, n_bins=10)

    for bin_stat in result.bin_stats:
        assert bin_stat["n_selected_speakers"] >= 1


def test_sample_speakers_reproducible():
    """Same seed produces same result."""
    metadata = _make_metadata(n_speakers=50)
    result1 = sample_speakers(metadata, target_hours=2.0, seed=42)
    result2 = sample_speakers(metadata, target_hours=2.0, seed=42)

    assert result1.selected_speakers == result2.selected_speakers
    assert result1.n_files == result2.n_files


def test_sampling_result_save_load(tmp_path):
    """SamplingResult survives save/load roundtrip."""
    metadata = _make_metadata(n_speakers=20)
    result = sample_speakers(metadata, target_hours=1.0, n_bins=5)

    path = tmp_path / "sampling.json"
    result.save(path)
    loaded = SamplingResult.load(path)

    assert loaded.n_speakers == result.n_speakers
    assert loaded.n_files == result.n_files
    assert len(loaded.bin_stats) == len(result.bin_stats)
