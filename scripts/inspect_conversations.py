#!/usr/bin/env python3
"""Inspect self-play conversation .npz files.

Decodes the inner monologue text tokens using Moshi's SentencePiece tokenizer
and displays conversation content, statistics, and quality metrics.

Usage (from Colab):
    !python /content/moshilite/scripts/inspect_conversations.py \
        --data-dir /content/drive/MyDrive/moshilite/self_play_data/conversations \
        --num 10

    # With text decoding (requires sentencepiece + tokenizer model):
    !python /content/moshilite/scripts/inspect_conversations.py \
        --data-dir /content/drive/MyDrive/moshilite/self_play_data/conversations \
        --num 10 \
        --weights-dir /content/drive/MyDrive/moshilite/upstream_weights/moshiko

    # Show only stats (no text decoding):
    !python /content/moshilite/scripts/inspect_conversations.py \
        --data-dir /content/drive/MyDrive/moshilite/self_play_data/conversations \
        --num 10 \
        --stats-only
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter


def load_sentencepiece_model(weights_dir: str | None):
    """Try to load SentencePiece tokenizer from Moshi weights directory.

    Moshi uses a 32K SentencePiece vocabulary. The model file is typically
    at: <weights_dir>/tokenizer_spm_32k_3.model or similar.

    Returns None if not available (falls back to raw token display).
    """
    if weights_dir is None:
        return None

    try:
        import sentencepiece as spm
    except ImportError:
        print("⚠️  sentencepiece not installed. Install with: pip install sentencepiece")
        print("   Falling back to raw token IDs.\n")
        return None

    weights_path = Path(weights_dir)

    # Search for sentencepiece model files
    sp_candidates = (
        list(weights_path.rglob("*.model"))
        + list(weights_path.rglob("tokenizer*.model"))
    )

    # Also check HuggingFace cache structure
    if not sp_candidates:
        sp_candidates = list(weights_path.rglob("*spm*"))
        sp_candidates = [f for f in sp_candidates if f.suffix in {".model", ""}]

    if not sp_candidates:
        print(f"⚠️  No SentencePiece model found in {weights_dir}")
        print("   Falling back to raw token IDs.\n")
        return None

    # Pick the most likely one
    sp_model_path = sp_candidates[0]
    print(f"📖 Loading tokenizer from: {sp_model_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_model_path))
    print(f"   Vocabulary size: {sp.GetPieceSize()}\n")
    return sp


def decode_text_tokens(tokens: np.ndarray, sp_model) -> str:
    """Decode text token array to readable text.

    Special handling:
    - Token 0 is typically PAD/silence in Moshi's text stream
    - Token 3 is often used as a special "unset" token
    - During silence, Moshi outputs PAD tokens in the text stream
    """
    if sp_model is None:
        # Fallback: show raw token IDs, highlighting non-pad tokens
        non_pad = tokens[tokens > 3]  # skip PAD(0), BOS(1), EOS(2), UNK(3)
        if len(non_pad) == 0:
            return "[all PAD/special tokens — model was silent]"
        return f"[raw IDs, {len(non_pad)} non-special tokens]: {non_pad[:50].tolist()}..."

    # Filter out special tokens for cleaner text
    # In Moshi's SPM, tokens 0-3 are typically: <pad>, <s>, </s>, <unk>
    token_list = tokens.tolist()

    # Decode all tokens (SPM handles special tokens)
    try:
        text = sp_model.Decode(token_list)
    except Exception:
        # Some tokens might be out of vocab — decode one by one
        pieces = []
        for t in token_list:
            try:
                if 0 <= t < sp_model.GetPieceSize():
                    pieces.append(sp_model.IdToPiece(int(t)))
                else:
                    pieces.append(f"[{t}]")
            except Exception:
                pieces.append(f"[{t}]")
        text = "".join(pieces)

    return text


def compute_stats(data: dict, filename: str) -> dict:
    """Compute statistics for a single conversation .npz file."""
    text_tokens = data["text_tokens"]
    audio_tokens = data["audio_tokens"]
    num_valid = int(data["num_valid_steps"][0])

    T = len(text_tokens)
    duration_s = T / 12.5  # 12.5 Hz frame rate

    # Text token stats
    text_unique = len(np.unique(text_tokens))
    text_non_pad = np.sum(text_tokens > 3)  # non-special tokens
    text_pad_ratio = 1.0 - (text_non_pad / max(T, 1))

    # Audio token stats
    audio_unique_per_cb = [len(np.unique(audio_tokens[cb])) for cb in range(audio_tokens.shape[0])]

    # Repetition (text 4-grams)
    def ngram_rep(arr, n=4):
        if len(arr) < n:
            return 0.0
        ngrams = [tuple(arr[i:i+n]) for i in range(len(arr) - n + 1)]
        return 1.0 - len(set(ngrams)) / max(len(ngrams), 1)

    text_rep = ngram_rep(text_tokens[:num_valid].tolist())
    audio_cb0_rep = ngram_rep(audio_tokens[0, :num_valid].tolist())

    # Silence on CB0
    silence_tokens = {0, 2048}
    cb0 = audio_tokens[0, :num_valid]
    silence_ratio = float(np.isin(cb0, list(silence_tokens)).sum()) / max(len(cb0), 1)

    # Top-k logit stats (if present)
    has_logits = "text_logits_vals" in data
    if has_logits:
        text_logit_vals = data["text_logits_vals"]
        logit_entropy_approx = float(np.std(text_logit_vals))
    else:
        logit_entropy_approx = None

    # User audio stats
    has_user_audio = "user_audio_tokens" in data
    if has_user_audio:
        user_audio = data["user_audio_tokens"]
        user_silence = float(np.isin(user_audio[0, :num_valid], list(silence_tokens)).sum()) / max(num_valid, 1)
    else:
        user_silence = None

    return {
        "filename": filename,
        "num_steps": T,
        "num_valid_steps": num_valid,
        "duration_s": round(duration_s, 1),
        "text_unique_tokens": text_unique,
        "text_non_pad_tokens": int(text_non_pad),
        "text_pad_ratio": round(text_pad_ratio, 3),
        "text_4gram_repetition": round(text_rep, 3),
        "audio_cb0_4gram_repetition": round(audio_cb0_rep, 3),
        "audio_cb0_silence_ratio": round(silence_ratio, 3),
        "audio_unique_per_codebook": audio_unique_per_cb,
        "user_audio_silence_ratio": round(user_silence, 3) if user_silence is not None else None,
        "has_logits": has_logits,
    }


def print_conversation(idx: int, filepath: Path, sp_model, stats_only: bool = False):
    """Print details of a single conversation."""
    data = np.load(str(filepath))
    stats = compute_stats(data, filepath.name)

    print(f"\n{'='*80}")
    print(f"  Conversation #{idx + 1}: {filepath.name}")
    print(f"{'='*80}")

    # Basic stats
    print(f"  Duration:          {stats['duration_s']}s ({stats['num_valid_steps']} valid / {stats['num_steps']} total steps)")
    print(f"  Text tokens:       {stats['text_non_pad_tokens']} non-PAD / {stats['num_steps']} total ({stats['text_pad_ratio']:.0%} PAD)")
    print(f"  Unique text IDs:   {stats['text_unique_tokens']}")
    print(f"  Text repetition:   {stats['text_4gram_repetition']:.1%} (4-gram)")
    print(f"  Audio CB0 rep:     {stats['audio_cb0_4gram_repetition']:.1%} (4-gram)")
    print(f"  CB0 silence:       {stats['audio_cb0_silence_ratio']:.1%}")
    if stats['user_audio_silence_ratio'] is not None:
        print(f"  User silence:      {stats['user_audio_silence_ratio']:.1%}")
    print(f"  Audio diversity:   {stats['audio_unique_per_codebook']} unique tokens per CB")

    if stats_only:
        return stats

    # Decode and show text
    text_tokens = data["text_tokens"][:stats["num_valid_steps"]]

    print(f"\n  📝 Inner Monologue (decoded text):")
    print(f"  {'-'*60}")

    decoded = decode_text_tokens(text_tokens, sp_model)

    # Clean up the text for display
    # Remove excessive padding representations
    cleaned = decoded.strip()
    if not cleaned or cleaned == "":
        print("  [Empty — model produced only PAD/special tokens (silence)]")
    else:
        # Word-wrap for readability
        words = cleaned.split()
        line = "  "
        for w in words:
            if len(line) + len(w) + 1 > 78:
                print(line)
                line = "  " + w
            else:
                line = line + " " + w if line.strip() else "  " + w
        if line.strip():
            print(line)

    # Show token distribution
    print(f"\n  📊 Text token distribution (top 10):")
    token_counts = Counter(text_tokens.tolist())
    for tok, count in token_counts.most_common(10):
        pct = count / len(text_tokens) * 100
        if sp_model is not None and 0 <= tok < sp_model.GetPieceSize():
            piece = sp_model.IdToPiece(int(tok))
            print(f"     Token {tok:5d} ({piece:>10s}): {count:4d} ({pct:5.1f}%)")
        else:
            print(f"     Token {tok:5d}: {count:4d} ({pct:5.1f}%)")

    return stats


def print_summary(all_stats: list[dict]):
    """Print aggregate summary over all inspected conversations."""
    print(f"\n\n{'='*80}")
    print(f"  AGGREGATE SUMMARY ({len(all_stats)} conversations)")
    print(f"{'='*80}")

    durations = [s["duration_s"] for s in all_stats]
    text_reps = [s["text_4gram_repetition"] for s in all_stats]
    audio_reps = [s["audio_cb0_4gram_repetition"] for s in all_stats]
    silence_ratios = [s["audio_cb0_silence_ratio"] for s in all_stats]
    pad_ratios = [s["text_pad_ratio"] for s in all_stats]
    non_pad = [s["text_non_pad_tokens"] for s in all_stats]

    print(f"  Total duration:     {sum(durations):.1f}s ({sum(durations)/3600:.3f} hours)")
    print(f"  Avg duration:       {np.mean(durations):.1f}s ± {np.std(durations):.1f}s")
    print(f"  Duration range:     {min(durations):.1f}s — {max(durations):.1f}s")
    print(f"")
    print(f"  Avg text PAD ratio: {np.mean(pad_ratios):.1%} ± {np.std(pad_ratios):.1%}")
    print(f"  Avg non-PAD tokens: {np.mean(non_pad):.0f} ± {np.std(non_pad):.0f}")
    print(f"  Avg text rep (4g):  {np.mean(text_reps):.1%} ± {np.std(text_reps):.1%}")
    print(f"  Avg audio rep (4g): {np.mean(audio_reps):.1%} ± {np.std(audio_reps):.1%}")
    print(f"  Avg CB0 silence:    {np.mean(silence_ratios):.1%} ± {np.std(silence_ratios):.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect self-play conversation .npz files — "
                    "decode text, show stats, check quality."
    )
    parser.add_argument("--data-dir", type=str,
                        default="/content/drive/MyDrive/moshilite/self_play_data/conversations",
                        help="Directory containing .npz conversation files (searched recursively).")
    parser.add_argument("--num", type=int, default=10,
                        help="Number of conversations to inspect.")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="Path to Moshi weights directory (for SentencePiece text decoding). "
                             "If not provided, shows raw token IDs.")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only show statistics, skip text decoding.")
    parser.add_argument("--batch", type=str, default=None,
                        help="Specific batch to inspect (e.g. 'batch_000'). "
                             "If not set, inspects from all batches.")
    parser.add_argument("--sort-by", type=str, default=None,
                        choices=["duration", "text_rep", "audio_rep", "silence", "non_pad"],
                        help="Sort conversations by a metric before displaying.")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save stats to a JSON file.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print(f"   Make sure you've run self-play generation (Stage 4a) first.")
        return

    # Find .npz files
    if args.batch:
        search_dir = data_dir / args.batch
        if not search_dir.exists():
            print(f"❌ Batch directory not found: {search_dir}")
            return
    else:
        search_dir = data_dir

    all_files = sorted(search_dir.rglob("*.npz"))
    # Exclude rejected conversations
    all_files = [f for f in all_files if "rejected" not in f.parts]

    if not all_files:
        print(f"❌ No .npz files found in {search_dir}")
        return

    print(f"📂 Found {len(all_files)} conversation files in {search_dir}")

    # Check for batch metadata
    batch_metas = sorted(search_dir.rglob("batch_meta.json"))
    if batch_metas:
        print(f"\n📋 Batch metadata found:")
        for meta_path in batch_metas:
            with open(meta_path) as f:
                meta = json.load(f)
            batch_name = meta_path.parent.name
            print(f"   {batch_name}: {meta.get('accepted', '?')} accepted, "
                  f"{meta.get('rejected', '?')} rejected, "
                  f"{meta.get('total_audio_hours', 0):.2f} hours, "
                  f"{meta.get('acceptance_rate', 0):.1%} acceptance rate")

    # Load tokenizer
    sp_model = None
    if not args.stats_only:
        sp_model = load_sentencepiece_model(args.weights_dir)
        if sp_model is None and args.weights_dir is None:
            print("💡 Tip: pass --weights-dir to decode text tokens with SentencePiece")
            print("   e.g. --weights-dir /content/drive/MyDrive/moshilite/upstream_weights/moshiko\n")

    # Select conversations to inspect
    num_to_show = min(args.num, len(all_files))

    if args.sort_by:
        # Need to load all to sort
        print(f"\n⏳ Loading all {len(all_files)} files to sort by {args.sort_by}...")
        all_stats = []
        for f in all_files:
            try:
                data = np.load(str(f))
                s = compute_stats(data, f.name)
                s["_path"] = f
                all_stats.append(s)
            except Exception as e:
                print(f"  ⚠️ Error loading {f.name}: {e}")

        sort_keys = {
            "duration": lambda s: s["duration_s"],
            "text_rep": lambda s: s["text_4gram_repetition"],
            "audio_rep": lambda s: s["audio_cb0_4gram_repetition"],
            "silence": lambda s: s["audio_cb0_silence_ratio"],
            "non_pad": lambda s: s["text_non_pad_tokens"],
        }
        all_stats.sort(key=sort_keys[args.sort_by], reverse=True)
        selected_files = [s["_path"] for s in all_stats[:num_to_show]]
    else:
        # Even spread across the dataset
        if num_to_show < len(all_files):
            indices = np.linspace(0, len(all_files) - 1, num_to_show, dtype=int)
            selected_files = [all_files[i] for i in indices]
        else:
            selected_files = all_files[:num_to_show]

    # Inspect each conversation
    all_stats = []
    for idx, filepath in enumerate(selected_files):
        try:
            stats = print_conversation(idx, filepath, sp_model, stats_only=args.stats_only)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n⚠️  Error inspecting {filepath.name}: {e}")

    # Aggregate summary
    if len(all_stats) > 1:
        print_summary(all_stats)

    # Save to JSON
    if args.save_json:
        output = {
            "num_inspected": len(all_stats),
            "total_files": len(all_files),
            "data_dir": str(search_dir),
            "conversations": all_stats,
        }
        with open(args.save_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n💾 Stats saved to {args.save_json}")


if __name__ == "__main__":
    main()
