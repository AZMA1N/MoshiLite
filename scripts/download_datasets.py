#!/usr/bin/env python3
"""Automated dataset downloader for MoshiLite.

Downloads all required datasets using aria2c (8 parallel connections)
to /content/datasets/ on Colab. Much faster than wget.

Usage:
    # Download everything
    python scripts/download_datasets.py --all

    # Download specific datasets
    python scripts/download_datasets.py --librispeech --ami
    python scripts/download_datasets.py --librilight-small
    python scripts/download_datasets.py --moshi-weights

    # Check what's already downloaded
    python scripts/download_datasets.py --status
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DATASETS_DIR = Path("/content/datasets")
GDRIVE_ROOT = Path("/content/drive/MyDrive/moshilite")


def _aria2c_available() -> bool:
    return subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0


def _install_aria2c():
    print("📦 Installing aria2c...")
    subprocess.run(["apt-get", "install", "-y", "-q", "aria2"], check=True)


def aria2c_download(url: str, out_dir: Path, filename: str | None = None):
    """Download a file with aria2c (8 parallel connections, auto-resume)."""
    if not _aria2c_available():
        _install_aria2c()

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        "--split=8",             # 8 parallel connections per server
        "--max-connection-per-server=8",
        "--min-split-size=10M",
        "--continue=true",       # auto-resume interrupted downloads
        "--dir", str(out_dir),
        "--max-tries=5",
        "--retry-wait=3",
    ]
    if filename:
        cmd += ["--out", filename]
    cmd.append(url)

    print(f"⬇️  Downloading {url.split('/')[-1]}...")
    subprocess.run(cmd, check=True)


def extract_tar(archive: Path, out_dir: Path):
    """Extract tar/tar.gz archive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    flag = "xzf" if str(archive).endswith(".gz") else "xf"
    print(f"📦 Extracting {archive.name}...")
    subprocess.run(["tar", flag, str(archive), "-C", str(out_dir)], check=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset download functions
# ──────────────────────────────────────────────────────────────────────────────

def download_moshi_weights():
    weights_dir = GDRIVE_ROOT / "upstream_weights/moshiko"
    if weights_dir.exists() and any(weights_dir.glob("*")):
        print("✅ Moshi weights already on GDrive"); return

    # Get HuggingFace token (set HF_TOKEN in Colab Secrets or env)
    import os
    token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    if not token:
        raise ValueError(
            "HF_TOKEN not found. Add it to Colab Secrets (🔑 sidebar) "
            "or set os.environ['HF_TOKEN'] = '<your_token>'. "
            "Also accept the license at: "
            "https://huggingface.co/kyutai/moshiko-pytorch-bf16"
        )

    print("⬇️  Downloading Moshi weights from HuggingFace (~15 GB)...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="kyutai/moshiko-pytorch-bf16",
        local_dir=str(weights_dir),
        token=token,
    )
    print("✅ Moshi weights saved to GDrive")


def download_librispeech():
    out_dir = DATASETS_DIR / "librispeech"
    if out_dir.exists() and any(out_dir.glob("**/*.flac")):
        print("✅ LibriSpeech already downloaded"); return

    archives_dir = DATASETS_DIR / "_archives"
    urls = [
        "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "https://www.openslr.org/resources/12/dev-other.tar.gz",
    ]
    for url in urls:
        fname = url.split("/")[-1]
        archive = archives_dir / fname
        if not archive.exists():
            aria2c_download(url, archives_dir)
        extract_tar(archive, out_dir)
        archive.unlink()  # free space immediately after extraction
    print("✅ LibriSpeech ready")


def download_librilight_small():
    out_dir = DATASETS_DIR / "librilight"
    if out_dir.exists() and any(out_dir.glob("**/*.flac")):
        print("✅ LibriLight (small) already downloaded"); return

    archives_dir = DATASETS_DIR / "_archives"
    aria2c_download(
        "https://dl.fbaipublicfiles.com/librilight/data/small.tar",
        archives_dir,
    )
    extract_tar(archives_dir / "small.tar", out_dir)
    (archives_dir / "small.tar").unlink()
    print("✅ LibriLight small ready")


def download_ami():
    out_dir = DATASETS_DIR / "ami"
    if out_dir.exists() and any(out_dir.glob("**/*.wav")):
        print("✅ AMI already downloaded"); return

    # AMI: use the amicorpus download script
    out_dir.mkdir(parents=True, exist_ok=True)
    print("⬇️  Downloading AMI Corpus (~20 GB)...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "gdown"
    ], check=True)

    # AMI individual headset microphone audio (IHM) — what Moshi was trained on
    script = "https://raw.githubusercontent.com/amicorpus/amicorpus/main/download_ami.sh"
    subprocess.run(["wget", "-q", "-O", "/tmp/download_ami.sh", script])
    subprocess.run(["bash", "/tmp/download_ami.sh", str(out_dir)])
    print("✅ AMI ready")


# ──────────────────────────────────────────────────────────────────────────────
# Status check
# ──────────────────────────────────────────────────────────────────────────────

def print_status():
    """Show what's downloaded and what's encoded."""
    checks = {
        "Moshi weights":          GDRIVE_ROOT / "upstream_weights/moshiko",
        "LibriSpeech (raw)":      DATASETS_DIR / "librispeech",
        "LibriLight small (raw)": DATASETS_DIR / "librilight",
        "AMI (raw)":              DATASETS_DIR / "ami",
    }
    token_checks = {
        "Tokens: LibriSpeech":   GDRIVE_ROOT / "tokens/librispeech",
        "Tokens: LibriLight":    GDRIVE_ROOT / "tokens/librilight",
        "Tokens: AMI":           GDRIVE_ROOT / "tokens/ami",
        "Tokens: LibriMix":      GDRIVE_ROOT / "tokens/librimix",
    }

    print("\n=== Raw datasets ===")
    for name, path in checks.items():
        if path.exists() and any(path.glob("**/*")):
            n = sum(1 for _ in path.rglob("*") if _.is_file())
            print(f"  ✅ {name}: {n} files")
        else:
            print(f"  ❌ {name}: not found")

    print("\n=== Encoded tokens on GDrive ===")
    import json
    for name, path in token_checks.items():
        stats_files = list(path.glob("*_stats.json")) if path.exists() else []
        if stats_files:
            total_hrs = 0.0
            total_shards = 0
            total_files = 0
            for sf in stats_files:
                stats = json.loads(sf.read_text())
                total_hrs += stats.get("total_duration_hours", 0)
                total_shards += stats.get("n_shards", 0)
                total_files += stats.get("n_files", 0)
            print(f"  ✅ {name}: {total_files} files, {total_hrs:.1f} hrs, "
                  f"{total_shards} shards ({len(stats_files)} stats files)")
        else:
            print(f"  ❌ {name}: not encoded")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download datasets for MoshiLite")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--moshi-weights", action="store_true")
    parser.add_argument("--librispeech", action="store_true")
    parser.add_argument("--librilight-small", action="store_true")
    parser.add_argument("--ami", action="store_true")
    parser.add_argument("--status", action="store_true", help="Show download status")
    args = parser.parse_args()

    if args.status:
        print_status(); return

    if args.all or args.moshi_weights:
        download_moshi_weights()
    if args.all or args.librispeech:
        download_librispeech()
    if args.all or args.librilight_small:
        download_librilight_small()
    if args.all or args.ami:
        download_ami()

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
