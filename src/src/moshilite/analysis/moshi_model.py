"""Moshi model loading utilities for analysis.

Provides helpers to load the Moshi LMModel at different quantization
precisions (INT4, INT8, FP16/BF16) for use in Stage 1 analysis and
teacher pre-computation.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Default GDrive weights path
DEFAULT_WEIGHTS_DIR = "/content/drive/MyDrive/moshilite/upstream_weights/moshiko"


def load_moshi_for_analysis(
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    precision: str = "bf16",
    device: str = "cuda",
    hf_repo: str = "kyutai/moshiko-pytorch-bf16",
) -> "torch.nn.Module":
    """Load Moshi LMModel at the specified precision for analysis.

    Requires ~15 GB VRAM for initial BF16 load. Use L4 (24 GB) or better.
    After loading, quantization reduces memory to ~4 GB (INT4) or ~8 GB (INT8).

    Args:
        weights_dir: Path to directory containing Moshi checkpoint files.
            If the directory doesn't exist or is empty, downloads from HF.
        precision: One of 'bf16' (default), 'fp16', 'int8', 'int4'.
        device: Target device ('cuda' or 'cpu').
        hf_repo: HuggingFace repo ID for downloading weights.

    Returns:
        LMModel in eval mode at the requested precision.
    """
    weights_path = Path(weights_dir)
    if not weights_path.exists() or not any(weights_path.iterdir()):
        logger.info("Weights not found at %s, downloading from HuggingFace...", weights_dir)
        weights_path = _download_weights(hf_repo, weights_dir)

    # Always load BF16 first — checkpoints don't have quantization keys
    model = _load_from_checkpoint(weights_path, device=device)

    # Apply quantization after loading
    if precision == "int4":
        logger.info("Applying INT4 quantization (moshi native)...")
        model = _apply_moshi_quantize(model)
    elif precision == "int8":
        model = _quantize_int8(model, device)
    elif precision == "fp16":
        model = model.half()
    # bf16 is already the native format — no conversion needed

    model.eval()

    # Report memory usage
    if device == "cuda" and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info("✅ Moshi loaded: precision=%s, VRAM=%.1f GB", precision, alloc)
    else:
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        logger.info("✅ Moshi loaded: precision=%s, params=%.2fB", precision, n_params)

    return model


def _load_from_checkpoint(
    weights_dir: Path, device: str = "cuda",
) -> "torch.nn.Module":
    """Load model from moshi checkpoint files (BF16).

    Finds the correct LM checkpoint file in the weights directory and
    passes it to moshi's get_moshi_lm() loader.
    """
    try:
        from moshi.models import loaders
    except ImportError:
        raise ImportError(
            "The 'moshi' package is required. Install with: pip install moshi"
        )

    # get_moshi_lm() expects a FILE path, not a directory.
    weights_dir = Path(weights_dir)
    lm_file = _find_lm_checkpoint(weights_dir)

    if lm_file is None:
        logger.info("No local LM checkpoint found, letting moshi download from HF...")

    logger.info("Loading Moshi LM from: %s", lm_file)

    model = loaders.get_moshi_lm(
        str(lm_file) if lm_file else None,
        device=device,
        dtype=torch.bfloat16,
    )
    return model


def _apply_moshi_quantize(model: "torch.nn.Module") -> "torch.nn.Module":
    """Apply moshi's built-in quantization (INT4) after loading BF16 weights."""
    try:
        from moshi.utils.quantize import replace_linear_with_qlinear
        replace_linear_with_qlinear(model)
        logger.info("Applied moshi native quantization")
        return model
    except Exception as e:
        logger.warning("Moshi quantization failed: %s. Keeping BF16.", e)
        return model


def _find_lm_checkpoint(weights_dir: Path) -> Path | None:
    """Find the LM model checkpoint file in a weights directory.

    Looks for the main model file, excluding tokenizer/mimi files.
    """
    # Typical filenames:
    #   model.safetensors, moshiko-*.safetensors, checkpoint*.safetensors
    # Exclude: tokenizer*.safetensors (that's the Mimi codec)
    all_safetensors = sorted(weights_dir.glob("*.safetensors"))
    # Filter out tokenizer/mimi files
    lm_candidates = [
        f for f in all_safetensors
        if not any(x in f.name.lower() for x in ("tokenizer", "mimi"))
    ]

    if lm_candidates:
        logger.info("Found LM checkpoint: %s", lm_candidates[0].name)
        return lm_candidates[0]

    # If all files look like tokenizers, there might be a different structure
    # Check for .pt files
    pt_candidates = [
        f for f in sorted(weights_dir.glob("*.pt"))
        if not any(x in f.name.lower() for x in ("tokenizer", "mimi"))
    ]
    if pt_candidates:
        return pt_candidates[0]

    # Check for nested directories (HF snapshot_download structure)
    for subdir in weights_dir.iterdir():
        if subdir.is_dir():
            result = _find_lm_checkpoint(subdir)
            if result:
                return result

    return None


def _download_weights(hf_repo: str, target_dir: str) -> Path:
    """Download weights from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    path = snapshot_download(
        repo_id=hf_repo,
        local_dir=target_dir,
        token=token,
    )
    return Path(path)


def _quantize_int8(model: "torch.nn.Module", device: str) -> "torch.nn.Module":
    """Apply INT8 dynamic quantization."""
    try:
        # Try moshi's native quantization first
        from moshi.utils.quantize import replace_linear_with_qlinear
        replace_linear_with_qlinear(model)
        model = model.to(device)
        logger.info("Applied INT8 via moshi native quantization")
        return model
    except (ImportError, Exception) as e:
        logger.warning("Moshi native quantization failed: %s", e)

    # Fallback: PyTorch dynamic quantization
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        model = model.to(device)
        logger.info("Applied INT8 via PyTorch dynamic quantization")
        return model
    except Exception as e:
        logger.warning("Dynamic quantization failed: %s. Using full precision.", e)
        return model.to(device)


def _moshi_native_quantize(model: "torch.nn.Module", precision: str) -> "torch.nn.Module":
    """Use moshi's built-in quantization if bitsandbytes is unavailable."""
    try:
        from moshi.utils.quantize import replace_linear_with_qlinear
        replace_linear_with_qlinear(model)
        logger.info("Applied moshi native quantization")
        return model
    except ImportError:
        logger.warning("No quantization backend available, using full precision")
        return model


def get_model_info(model: "torch.nn.Module") -> dict:
    """Extract architecture info from a loaded Moshi model.

    Returns dict with n_layers, n_heads, d_model, d_ffn, total_params.
    Handles QLinear (quantized) modules that lack standard attributes.
    """
    info = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "total_params_b": sum(p.numel() for p in model.parameters()) / 1e9,
    }

    # Use model-level attributes first (always available)
    if hasattr(model, "dim"):
        info["d_model"] = model.dim
    if hasattr(model, "dep_q"):
        info["dep_q"] = model.dep_q
    if hasattr(model, "n_q"):
        info["n_q"] = model.n_q

    # Try to extract layer-level info from the transformer
    if hasattr(model, "transformer"):
        transformer = model.transformer
        if hasattr(transformer, "layers"):
            layers = list(transformer.layers)
            info["n_layers"] = len(layers)

            if layers:
                layer0 = layers[0]
                try:
                    if hasattr(layer0, "self_attn"):
                        info["n_heads"] = layer0.self_attn.num_heads
                        info.setdefault("d_model", layer0.self_attn.embed_dim)
                except AttributeError:
                    pass

                try:
                    if hasattr(layer0, "gating") and layer0.gating is not None:
                        gating = layer0.gating
                        if hasattr(gating, "linear_in"):
                            w = gating.linear_in
                            # QLinear stores shape in weight tensor
                            if hasattr(w, "out_features"):
                                info["d_ffn"] = w.out_features // 2
                            elif hasattr(w, "weight"):
                                info["d_ffn"] = w.weight.shape[0] // 2
                    elif hasattr(layer0, "linear1") and layer0.linear1 is not None:
                        lin = layer0.linear1
                        if hasattr(lin, "out_features"):
                            info["d_ffn"] = lin.out_features
                        elif hasattr(lin, "weight"):
                            info["d_ffn"] = lin.weight.shape[0]
                except AttributeError:
                    pass

    return info

