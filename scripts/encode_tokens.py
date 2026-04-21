#!/usr/bin/env python3
"""CLI: Mimi pre-encoding of audio datasets to RVQ token shards.

Usage:
    python scripts/encode_tokens.py \
        --audio-dir /content/datasets/librispeech/train-clean-100 \
        --output-dir /content/drive/MyDrive/moshilite/tokens/librispeech/train-clean-100 \
        --dataset-name librispeech-train-clean-100 \
        --shard-size-mb 256

    # Chunked mode for LibriMix (encode + verify in one step)
    python scripts/encode_tokens.py \
        --audio-dir /content/librimix_chunk/ \
        --output-dir /content/drive/MyDrive/moshilite/tokens/librimix/train-100 \
        --dataset-name librimix-train-100 \
        --verify \
        --expected-min-files 1000

    # LibriLight with speaker sampling
    python scripts/encode_tokens.py \
        --audio-dir /content/datasets/librilight \
        --output-dir /content/drive/MyDrive/moshilite/tokens/librilight \
        --dataset-name librilight-1k \
        --use-speaker-sampling \
        --target-hours 1000 \
        --splits small
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Encode audio files to Mimi RVQ token shards"
    )
    parser.add_argument(
        "--audio-dir", type=str, required=True,
        help="Directory containing audio files (searched recursively)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save WebDataset tar shards"
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True,
        help="Name prefix for shard files"
    )
    parser.add_argument(
        "--weights-dir", type=str,
        default="/content/drive/MyDrive/moshilite/upstream_weights/moshiko",
        help="Path to Moshi/Mimi weights directory"
    )
    parser.add_argument(
        "--shard-size-mb", type=float, default=256.0,
        help="Target size per shard in MB"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for encoding (cuda/cpu)"
    )

    # Verification options
    parser.add_argument(
        "--verify", action="store_true",
        help="Run shard integrity verification after encoding"
    )
    parser.add_argument(
        "--expected-min-files", type=int, default=1,
        help="Minimum number of encoded files expected (for verification)"
    )
    parser.add_argument(
        "--expected-min-hours", type=float, default=0.0,
        help="Minimum total duration in hours (for verification)"
    )

    # LibriLight speaker sampling options
    parser.add_argument(
        "--use-speaker-sampling", action="store_true",
        help="Use duration-binned speaker sampling (for LibriLight)"
    )
    parser.add_argument(
        "--target-hours", type=float, default=1000.0,
        help="Target total hours for speaker sampling"
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["small"],
        help="LibriLight splits to use for sampling"
    )
    parser.add_argument(
        "--sampling-seed", type=int, default=42,
        help="Random seed for speaker sampling"
    )

    # Verify-only mode (skip encoding)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only run verification, skip encoding"
    )

    args = parser.parse_args()

    if args.verify_only:
        from moshilite.data.encoding import verify_shard_integrity
        passed = verify_shard_integrity(
            args.output_dir,
            args.dataset_name,
            expected_min_files=args.expected_min_files,
            expected_min_duration_hours=args.expected_min_hours,
        )
        sys.exit(0 if passed else 1)

    # Load Mimi encoder
    from moshilite.data.encoding import load_mimi_encoder, encode_audio_dir, verify_shard_integrity
    print(f"🔄 Loading Mimi encoder from {args.weights_dir}...")
    mimi = load_mimi_encoder(args.weights_dir, device=args.device)

    # Handle LibriLight speaker sampling
    audio_dir = args.audio_dir
    if args.use_speaker_sampling:
        from moshilite.data.librilight_sampler import scan_librilight_metadata, sample_speakers
        print(f"🔄 Scanning LibriLight metadata in {args.audio_dir}...")
        metadata = scan_librilight_metadata(args.audio_dir, splits=args.splits)

        result = sample_speakers(
            metadata,
            target_hours=args.target_hours,
            seed=args.sampling_seed,
        )

        # Save sampling result
        result.save(f"{args.output_dir}/{args.dataset_name}_sampling.json")

        # Create a temp file list for the encoder to use
        # (encode_audio_dir processes a directory, so we write selected files)
        import tempfile
        file_list_path = tempfile.mktemp(suffix=".txt")
        with open(file_list_path, "w") as f:
            for file_path in result.selected_files:
                f.write(file_path + "\n")
        print(f"📋 Selected {result.n_files} files → {file_list_path}")

        # For now, encode the full directory — TODO: use file list filtering
        print("⚠️ Note: encoding full audio dir (speaker sampling file list "
              "filtering not yet wired into encode_audio_dir)")

    # Encode
    print(f"🔄 Encoding {args.audio_dir} → {args.output_dir}...")
    stats = encode_audio_dir(
        mimi,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        shard_size_mb=args.shard_size_mb,
        device=args.device,
    )

    # Verify if requested
    if args.verify:
        passed = verify_shard_integrity(
            args.output_dir,
            args.dataset_name,
            expected_min_files=args.expected_min_files,
            expected_min_duration_hours=args.expected_min_hours,
        )
        if not passed:
            print("❌ VERIFICATION FAILED — do NOT delete raw audio!")
            sys.exit(1)
        print("✅ Verification passed — safe to delete raw audio")


if __name__ == "__main__":
    main()
