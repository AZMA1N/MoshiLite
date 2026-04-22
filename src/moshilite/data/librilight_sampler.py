"""Duration-binned speaker sampling for LibriLight 1K-hour selection.

Selects ~1K hours from LibriLight using metadata-only speaker binning
to ensure coverage across prolific and rare speakers.

Strategy (from v8 plan):
    1. Load LibriLight metadata (speaker IDs, utterance durations)
    2. Compute total duration per speaker → sort by total hours
    3. Bin speakers into 10 equal-count bins (low → high total hours)
    4. From each bin: randomly sample speakers until ~100 hours per bin
    5. Include all utterances from selected speakers
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SamplingResult:
    """Result of LibriLight speaker sampling."""
    selected_speakers: list[str]
    selected_files: list[str]
    total_duration_hours: float
    n_speakers: int
    n_files: int
    bin_stats: list[dict]  # per-bin statistics

    def save(self, output_path: str | Path):
        """Save sampling result to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"✅ Sampling result saved to {output_path}")

    @classmethod
    def load(cls, path: str | Path) -> "SamplingResult":
        """Load sampling result from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def scan_librilight_metadata(
    librilight_dir: str | Path,
    splits: list[str] = ("small",),
) -> dict[str, list[dict]]:
    """Scan LibriLight directory structure to extract speaker metadata.

    LibriLight directory structure:
        {split}/
            {speaker_id}/
                {book_id}/
                    {utterance_id}.flac
                    {utterance_id}.json  (metadata with duration)

    Args:
        librilight_dir: Root directory of downloaded LibriLight.
        splits: Which splits to scan ("small", "medium", "large").

    Returns:
        Dict mapping speaker_id -> list of {file, duration_s, split} dicts.
    """
    librilight_dir = Path(librilight_dir)
    speakers = defaultdict(list)

    for split in splits:
        split_dir = librilight_dir / split
        if not split_dir.exists():
            print(f"⚠️ Split directory not found: {split_dir}")
            continue

        # Scan speaker directories
        for speaker_dir in sorted(split_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name

            for book_dir in speaker_dir.iterdir():
                if not book_dir.is_dir():
                    continue

                for audio_file in book_dir.glob("*.flac"):
                    # Try to get duration from companion JSON
                    meta_file = audio_file.with_suffix(".json")
                    duration_s = _get_duration(audio_file, meta_file)

                    speakers[speaker_id].append({
                        "file": str(audio_file),
                        "duration_s": duration_s,
                        "split": split,
                    })

    print(f"📊 Scanned {len(speakers)} speakers across {splits}")
    return dict(speakers)


def sample_speakers(
    speaker_metadata: dict[str, list[dict]],
    target_hours: float = 1000.0,
    n_bins: int = 10,
    hours_per_bin: Optional[float] = None,
    seed: int = 42,
) -> SamplingResult:
    """Select speakers using duration-binned sampling.

    Args:
        speaker_metadata: Output of scan_librilight_metadata().
        target_hours: Total target duration in hours.
        n_bins: Number of duration bins.
        hours_per_bin: Target hours per bin (default: target_hours / n_bins).
        seed: Random seed for reproducibility.

    Returns:
        SamplingResult with selected speakers and files.
    """
    random.seed(seed)
    np.random.seed(seed)

    if hours_per_bin is None:
        hours_per_bin = target_hours / n_bins

    # Step 1: Compute total duration per speaker
    speaker_durations = {}
    for speaker_id, utterances in speaker_metadata.items():
        total_s = sum(u["duration_s"] for u in utterances)
        speaker_durations[speaker_id] = total_s

    # Step 2: Sort speakers by total duration
    sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1])

    # Step 3: Bin speakers into equal-count bins
    n_speakers = len(sorted_speakers)
    bin_size = max(1, n_speakers // n_bins)
    bins = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n_speakers
        bin_speakers = sorted_speakers[start:end]
        bins.append(bin_speakers)

    # Step 4: Sample from each bin until target hours reached
    selected_speakers = []
    selected_files = []
    total_duration = 0.0
    bin_stats = []

    for bin_idx, bin_speakers in enumerate(bins):
        random.shuffle(bin_speakers)
        bin_duration = 0.0
        bin_selected = 0
        target_bin_s = hours_per_bin * 3600

        for speaker_id, speaker_duration_s in bin_speakers:
            if bin_duration >= target_bin_s:
                break

            selected_speakers.append(speaker_id)
            bin_selected += 1
            bin_duration += speaker_duration_s
            total_duration += speaker_duration_s

            # Include all utterances from this speaker
            for utt in speaker_metadata[speaker_id]:
                selected_files.append(utt["file"])

        bin_stats.append({
            "bin_index": bin_idx,
            "n_available_speakers": len(bin_speakers),
            "n_selected_speakers": bin_selected,
            "duration_hours": bin_duration / 3600,
            "duration_range_hours": (
                bin_speakers[0][1] / 3600 if bin_speakers else 0,
                bin_speakers[-1][1] / 3600 if bin_speakers else 0,
            ),
        })

    result = SamplingResult(
        selected_speakers=selected_speakers,
        selected_files=selected_files,
        total_duration_hours=total_duration / 3600,
        n_speakers=len(selected_speakers),
        n_files=len(selected_files),
        bin_stats=bin_stats,
    )

    print(f"✅ Selected {result.n_speakers} speakers, {result.n_files} files, "
          f"{result.total_duration_hours:.1f} hours")

    return result


# === Internal helpers ===


def _get_duration(audio_path: Path, meta_path: Path) -> float:
    """Get audio duration from metadata JSON or by probing the file."""
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if "duration" in meta:
                return float(meta["duration"])
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: probe audio file with torchaudio
    try:
        import torchaudio
        info = torchaudio.info(str(audio_path))
        return info.num_frames / info.sample_rate
    except Exception:
        # Last resort: estimate from file size (FLAC ~700kbps)
        file_size = audio_path.stat().st_size
        return file_size / (700 * 1024 / 8)  # rough estimate
