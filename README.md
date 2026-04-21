# 🎤 MoshiLite

**Compressing Moshi (~7.69B) to a sub-3B speech-to-speech model.**

Multi-stage compression: structured pruning → offline knowledge distillation → quantization.

## Setup

```bash
# Clone and install
git clone https://github.com/<your-username>/moshilite.git
cd moshilite
pip install -e ".[dev]"
```

### Colab Session Startup

```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys

REPO = "https://github.com/<your-username>/moshilite.git"
REPO_DIR = "/content/moshilite"

if not __import__("os").path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", REPO_DIR, "-q"], check=True)
```

## Project Structure

```
src/moshilite/
├── data/           # Dataset loading, Mimi encoding, shard staging
├── analysis/       # Layer importance (BI scores, DSR, head/FFN importance)
├── pruning/        # Structured pruning (depth, head, FFN)
├── distillation/   # KD losses, alignment, training loop, teacher precompute
├── eval/           # Metrics, codebook eval, SQA pipeline
└── utils/          # Checkpointing, experiment helpers, precision gates
```

## Stages

| Stage | Description |
|-------|-------------|
| Phase 0 | Environment setup + Mimi pre-encoding |
| Stage 1 | Layer importance analysis |
| Stage 2 | Structured pruning (~7B → ~3B) |
| Stage 3 | Depth Transformer compatibility |
| Stage 4a | Teacher pre-computation (offline) |
| Stage 4b | Student KD training |
| Stage 5 | Automated codebook eval (callback) |
| Stage 6 | Quantization (INT8/INT4/GGUF) |

## Compute

Designed for **Google Colab Free (T4)** + **Google Drive (1.9 TB)** persistent storage.