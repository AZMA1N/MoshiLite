#!/usr/bin/env python3
"""CLI: Self-play data generation for offline knowledge distillation.

Usage (from Colab):
    !python /content/moshilite/scripts/generate_self_play.py \
        --batch-id batch_000 \
        --num-conversations 120 \
        --steps 3750 \
        --output-dir /content/drive/MyDrive/moshilite/self_play_data/conversations \
        --device cuda
"""

import argparse
import sys
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play conversations with teacher target capture."
    )
    parser.add_argument("--batch-id", type=str, default="batch_000",
                        help="Unique batch identifier for this session.")
    parser.add_argument("--num-conversations", type=int, default=120,
                        help="Number of accepted conversations to generate.")
    parser.add_argument("--steps", type=int, default=3750,
                        help="Timesteps per conversation (3750 ≈ 5min at 12.5 Hz).")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of top logits to capture per step.")
    parser.add_argument("--output-dir", type=str,
                        default="/content/drive/MyDrive/moshilite/self_play_data/conversations",
                        help="GDrive directory for saving conversations.")
    parser.add_argument("--weights-dir", type=str,
                        default="/content/drive/MyDrive/moshilite/upstream_weights/moshiko",
                        help="Path to Moshi weights directory or HF repo.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Starting conversation index (for multi-session).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],
                        help="Model dtype. Use float16 for T4, bfloat16 for A100/L4.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for generation.")
    parser.add_argument("--temperature-text", type=float, default=0.7,
                        help="Sampling temperature for text tokens.")
    parser.add_argument("--repetition-penalty", type=float, default=1.3,
                        help="Repetition penalty for text tokens (1.0=off, >1.0=penalize).")
    parser.add_argument("--rep-window", type=int, default=50,
                        help="Window of recent non-PAD text tokens tracked for penalty.")
    args = parser.parse_args()

    # ── Load model ──
    print(f"🔧 Loading Moshi from {args.weights_dir}...")
    from moshi.models import loaders

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    import json
    from pathlib import Path
    
    weights_path = Path(args.weights_dir)
    config = None
    moshi_name = "model.safetensors"

    if weights_path.is_dir():
        print(f"Loading local weights from directory: {weights_path}")
        config_path = weights_path / "config.json"
        
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            moshi_name = config.get("moshi_name", moshi_name)
            
        model_path = weights_path / moshi_name
        
        if not model_path.exists():
            sf_files = list(weights_path.glob("*.safetensors"))
            if sf_files:
                model_path = sf_files[0]
                
        lm_model = loaders.get_moshi_lm(
            model_path, lm_kwargs=config, device=args.device, dtype=dtype
        )
    else:
        print(f"Attempting to load from HF Hub: {args.weights_dir}")
        ckpt = loaders.CheckpointInfo.from_hf_repo(args.weights_dir)
        lm_model = ckpt.get_moshi(device=args.device, dtype=dtype)

    print(f"✅ Model loaded: n_q={lm_model.n_q}, dep_q={lm_model.dep_q}, "
          f"dim={lm_model.dim}, params={sum(p.numel() for p in lm_model.parameters()) / 1e9:.2f}B")

    # ── Create LMGen ──
    from moshi.models.lm import LMGen

    lm_gen = LMGen(
        lm_model,
        use_sampling=True,
        temp=args.temperature,
        temp_text=args.temperature_text,
        top_k=250,
        top_k_text=25,
    )

    # ── Generate ──
    from moshilite.self_play.generator import generate_batch

    stats = generate_batch(
        lm_gen=lm_gen,
        num_conversations=args.num_conversations,
        steps_per_conversation=args.steps,
        top_k_logits=args.top_k,
        output_dir=args.output_dir,
        batch_id=args.batch_id,
        start_index=args.start_index,
        card=lm_model.card,
        device=args.device,
        repetition_penalty=args.repetition_penalty,
        rep_window=args.rep_window,
    )

    print(f"\n📊 Generation complete:")
    print(f"   Accepted: {stats['accepted']}")
    print(f"   Rejected: {stats['rejected']}")
    print(f"   Audio hours: {stats['total_audio_hours']:.2f}")
    print(f"   Wall time: {stats['total_wall_time_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
