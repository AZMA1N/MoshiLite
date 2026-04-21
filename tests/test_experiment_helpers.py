"""Tests for experiment management helpers."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from moshilite.utils.experiment import (
    get_experiment_dir,
    save_config_snapshot,
    update_model_manifest,
    save_eval_results,
)


def test_get_experiment_dir_checkpoints(tmp_path):
    with patch("moshilite.utils.experiment.GDRIVE_ROOT", tmp_path):
        result = get_experiment_dir("exp_01", "checkpoints")
        assert result == tmp_path / "checkpoints" / "exp_01"
        assert result.exists()


def test_get_experiment_dir_eval(tmp_path):
    with patch("moshilite.utils.experiment.GDRIVE_ROOT", tmp_path):
        result = get_experiment_dir("exp_01", "eval/stage2_post_prune")
        assert result == tmp_path / "eval" / "stage2_post_prune" / "exp_01"
        assert result.exists()


def test_get_experiment_dir_immutable(tmp_path):
    with patch("moshilite.utils.experiment.GDRIVE_ROOT", tmp_path):
        result = get_experiment_dir("exp_01", "tokens")
        # tokens is not auto-scoped — should NOT include experiment_id
        assert result == tmp_path / "tokens"
        assert result.exists()


def test_save_config_snapshot(tmp_path):
    with patch("moshilite.utils.experiment.GDRIVE_ROOT", tmp_path):
        config = {"lr": 1e-4, "batch_size": 4}
        save_config_snapshot("exp_01", config)
        config_path = tmp_path / "checkpoints" / "exp_01" / "config.yaml"
        assert config_path.exists()
        import yaml
        saved = yaml.safe_load(config_path.read_text())
        assert saved["lr"] == 1e-4
        assert saved["_experiment_id"] == "exp_01"
        assert "_saved_at" in saved


def test_update_model_manifest(tmp_path):
    with patch("moshilite.utils.experiment.GDRIVE_ROOT", tmp_path):
        update_model_manifest("model_int8.pt", "step_1000.pt", "exp_01", {"pesq": 3.5})
        manifest_path = tmp_path / "models" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest) == 1
        assert manifest[0]["model"] == "model_int8.pt"
        assert manifest[0]["metrics"]["pesq"] == 3.5

        # Append a second entry
        update_model_manifest("model_int4.pt", "step_2000.pt", "exp_01", {"pesq": 3.2})
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest) == 2


def test_save_eval_results(tmp_path):
    with patch("moshilite.utils.experiment.GDRIVE_ROOT", tmp_path):
        results = {"pesq": 3.5, "stoi": 0.92}
        save_eval_results("exp_01", "stage2_post_prune", results)
        results_path = tmp_path / "eval" / "exp_01" / "stage2_post_prune" / "results.json"
        assert results_path.exists()
        saved = json.loads(results_path.read_text())
        assert saved["pesq"] == 3.5
        assert saved["_stage"] == "stage2_post_prune"
