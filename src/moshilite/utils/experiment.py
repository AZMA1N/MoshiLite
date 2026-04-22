"""Experiment management helpers for MoshiLite.

Provides auto-scoped directory management, config snapshots, model manifest
tracking, and eval result routing — all keyed by experiment_id.
"""

import json
import shutil
import yaml
from pathlib import Path
from datetime import datetime

GDRIVE_ROOT = Path("/content/drive/MyDrive/moshilite")


def get_experiment_dir(experiment_id: str, subdir: str) -> Path:
    """Returns the experiment-scoped directory, creating it if needed.

    Usage:
        ckpt_dir = get_experiment_dir("prune30_kd_v1", "checkpoints")
        eval_dir = get_experiment_dir("prune30_kd_v1", "eval/stage2_post_prune")
        log_dir  = get_experiment_dir("prune30_kd_v1", "logs")
    """
    path = GDRIVE_ROOT / subdir.replace("{experiment_id}", experiment_id)
    if "{experiment_id}" not in subdir:
        # Auto-scope to experiment_id for known mutable dirs
        if subdir in ("checkpoints", "logs", "eval") or subdir.startswith("eval/"):
            path = GDRIVE_ROOT / subdir / experiment_id
        else:
            path = GDRIVE_ROOT / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config_snapshot(experiment_id: str, config: dict):
    """Auto-saves the exact config used for a run alongside its checkpoints."""
    ckpt_dir = get_experiment_dir(experiment_id, "checkpoints")
    config_path = ckpt_dir / "config.yaml"
    config["_saved_at"] = datetime.now().isoformat()
    config["_experiment_id"] = experiment_id
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ Config snapshot saved to {config_path}")


def update_model_manifest(
    model_filename: str,
    source_checkpoint: str,
    experiment_id: str,
    metrics: dict,
):
    """Auto-appends an entry to models/manifest.json on every model export."""
    models_dir = GDRIVE_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = models_dir / "manifest.json"

    manifest = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest.append({
        "model": model_filename,
        "source_checkpoint": source_checkpoint,
        "experiment_id": experiment_id,
        "metrics": metrics,
        "exported_at": datetime.now().isoformat(),
    })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Manifest updated: {model_filename} → {manifest_path}")


def save_eval_results(experiment_id: str, stage: str, results: dict):
    """Auto-routes eval results to the correct experiment/stage directory."""
    eval_dir = get_experiment_dir(experiment_id, "eval") / stage
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / "results.json"
    results["_experiment_id"] = experiment_id
    results["_stage"] = stage
    results["_saved_at"] = datetime.now().isoformat()
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Eval results saved to {results_path}")
