"""Unstructured pruning: Magnitude, Wanda, and SparseGPT.

Applies weight-level sparsity (zeroing individual weights) to the
Temporal Transformer's Linear layers. Unlike structured pruning, this
does NOT change tensor shapes — the model retains its full architecture
but with a fraction of weights set to zero.

All three methods operate on the Temporal Transformer only (matching
the scope of structured pruning in Stage 2 Part A).

References:
    - Magnitude: Baseline — prune smallest |W| entries.
    - Wanda: "A Simple and Effective Pruning Approach for LLMs" (Sun et al., 2024)
      Prunes by |W_ij| * ||X_j||_2 — weight magnitude × input activation norm.
    - SparseGPT: "SparseGPT: Massive Language Models Can Be Accurately Pruned
      in One-Shot" (Frantar & Alistarh, 2023) — prunes with Hessian-based
      weight reconstruction to minimize output distortion.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from moshilite.analysis.moshi_hooks import _get_transformer

logger = logging.getLogger(__name__)


@dataclass
class UnstructuredPruningResult:
    """Summary of an unstructured pruning run."""
    method: str                     # "magnitude", "wanda", or "sparsegpt"
    target_sparsity: float          # Requested sparsity (e.g. 0.30)
    actual_sparsity: float          # Achieved sparsity after pruning
    total_params: int               # Total parameters in pruned modules
    zeroed_params: int              # Number of parameters set to zero
    n_layers_pruned: int            # Number of transformer layers touched
    per_layer_sparsity: dict = field(default_factory=dict)  # layer_idx -> actual sparsity


def _get_prunable_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Get all Linear layers in the Temporal Transformer that should be pruned.

    Targets: Q/K/V projections, output projections, FFN up/down projections.
    Excludes: embedding layers, LM heads, Depth Transformer, Mimi codec.
    """
    transformer = _get_transformer(model)
    linears = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            linears.append((f"transformer.{name}", module))
    logger.info(f"Found {len(linears)} prunable Linear layers in Temporal Transformer")
    return linears


def _find_gating_children(model: nn.Module) -> dict:
    """Identify Linear modules inside gating (FFN) modules.

    Moshi's gating modules call F.linear() internally instead of
    self.linear_in(x), so standard forward hooks on the child Linear
    modules never fire. We must hook the parent gating module instead.

    Returns:
        Dict mapping Linear full name -> {
            'gating': the parent gating module,
            'role': 'linear_in' or 'linear_out'
        }
    """
    transformer = _get_transformer(model)
    result = {}

    for layer_idx, layer in enumerate(transformer.layers):
        if not hasattr(layer, 'gating') or layer.gating is None:
            continue
        gating = layer.gating
        if isinstance(gating, nn.ModuleList):
            gating = gating[0]

        prefix = f"transformer.layers.{layer_idx}.gating"
        if hasattr(gating, 'linear_in'):
            result[f"{prefix}.linear_in"] = {'gating': gating, 'role': 'linear_in'}
        if hasattr(gating, 'linear_out'):
            result[f"{prefix}.linear_out"] = {'gating': gating, 'role': 'linear_out'}

    logger.info(f"Found {len(result)} gating-child Linears requiring parent-level hooks")
    return result


def _accumulate_norm(store: dict, counts: dict, key: str, x: torch.Tensor):
    """Accumulate L2 column norms into store[key]."""
    x_flat = x.reshape(-1, x.shape[-1]).float()
    col_norms = torch.norm(x_flat, p=2, dim=0)
    if key not in store:
        store[key] = torch.zeros(col_norms.shape[0], device=col_norms.device, dtype=torch.float32)
        counts[key] = 0
    store[key] += col_norms.to(store[key].device)
    counts[key] += 1


def _compute_gating_intermediate(gating_mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Manually compute the SiGLU intermediate activation.

    gating.linear_in projects to 2*d_ffn, then:
        gate, up = chunk(2)
        intermediate = activation(gate) * up

    This is the input to gating.linear_out.
    """
    w = gating_mod.linear_in.weight.data
    b = gating_mod.linear_in.bias
    b = b.data if b is not None else None
    h = torch.nn.functional.linear(x, w, b)
    gate, up = h.chunk(2, dim=-1)
    if hasattr(gating_mod, 'activation'):
        return gating_mod.activation(gate) * up
    else:
        return torch.nn.functional.silu(gate) * up  # SiGLU default


# ──────────────────────────────────────────────────────────
#  Method 1: Magnitude Pruning
# ──────────────────────────────────────────────────────────


def prune_magnitude(
    model: nn.Module,
    sparsity: float = 0.30,
    per_layer: bool = True,
) -> UnstructuredPruningResult:
    """Prune weights by absolute magnitude.

    Zeroes out the smallest |W| entries. This is the simplest baseline
    requiring no calibration data.

    Args:
        model: Moshi LMModel (mutated in-place).
        sparsity: Fraction of weights to zero (0.30 = 30%).
        per_layer: If True, prune each layer independently to the target
            sparsity. If False, use a global threshold across all layers.

    Returns:
        UnstructuredPruningResult with pruning statistics.
    """
    linears = _get_prunable_linears(model)
    per_layer_stats = {}

    if per_layer:
        for name, linear in linears:
            W = linear.weight.data
            n_params = W.numel()
            n_prune = int(n_params * sparsity)

            if n_prune == 0:
                continue

            # Find threshold: the n_prune-th smallest absolute value
            threshold = torch.kthvalue(W.abs().flatten(), n_prune).values
            mask = W.abs() > threshold
            W.mul_(mask.to(W.dtype))

            actual = 1.0 - (mask.sum().item() / n_params)
            per_layer_stats[name] = actual
    else:
        # Global magnitude threshold
        all_weights = torch.cat([l.weight.data.abs().flatten() for _, l in linears])
        n_total = all_weights.numel()
        n_prune = int(n_total * sparsity)
        threshold = torch.kthvalue(all_weights, n_prune).values

        for name, linear in linears:
            W = linear.weight.data
            mask = W.abs() > threshold
            W.mul_(mask.to(W.dtype))
            actual = 1.0 - (mask.sum().item() / W.numel())
            per_layer_stats[name] = actual

    return _compute_result("magnitude", sparsity, linears, per_layer_stats)


# ──────────────────────────────────────────────────────────
#  Method 2: Wanda (Weights AND Activations)
# ──────────────────────────────────────────────────────────


def prune_wanda(
    model: nn.Module,
    calibration_data: list[torch.Tensor],
    sparsity: float = 0.30,
    device: str = "cuda",
) -> UnstructuredPruningResult:
    """Prune weights by Wanda: |W_ij| * ||X_j||_2.

    For each Linear layer, computes the importance of each weight as the
    product of the weight magnitude and the L2 norm of the corresponding
    input activation column (computed over calibration data). Then zeroes
    the least important weights.

    Args:
        model: Moshi LMModel (mutated in-place).
        calibration_data: List of token tensors [B, K, T] for computing
            activation norms. Typically 200 self-play conversations.
        sparsity: Fraction of weights to zero (0.30 = 30%).
        device: Torch device for calibration forward passes.

    Returns:
        UnstructuredPruningResult with pruning statistics.
    """
    transformer = _get_transformer(model)
    linears = _get_prunable_linears(model)
    gating_children = _find_gating_children(model)

    # Step 1: Collect input activation norms for each Linear layer
    activation_norms: dict[str, torch.Tensor] = {}
    hooks = []
    sample_counts: dict[str, int] = {}

    # 1a. Standard hooks on non-gating Linears (attention projections)
    def make_hook(name: str, linear: nn.Linear):
        def hook_fn(module, inputs, output):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            _accumulate_norm(activation_norms, sample_counts, name, x)
        return hook_fn

    for name, linear in linears:
        if name not in gating_children:
            hooks.append(linear.register_forward_hook(make_hook(name, linear)))

    # 1b. Hook parent gating modules for their child Linears
    #     (Moshi calls F.linear() internally, so child hooks don't fire)
    hooked_gating_ids = set()
    for name, info in gating_children.items():
        gating_mod = info['gating']
        gid = id(gating_mod)
        if gid in hooked_gating_ids:
            continue
        hooked_gating_ids.add(gid)

        # Find both linear_in and linear_out keys for this gating module
        in_key = next((n for n, i in gating_children.items()
                       if id(i['gating']) == gid and i['role'] == 'linear_in'), None)
        out_key = next((n for n, i in gating_children.items()
                        if id(i['gating']) == gid and i['role'] == 'linear_out'), None)

        def make_gating_hook(g_mod, ik, ok):
            def hook_fn(module, inputs, output):
                x = inputs[0] if isinstance(inputs, tuple) else inputs
                # Gating input = linear_in input
                if ik is not None:
                    _accumulate_norm(activation_norms, sample_counts, ik, x)
                # Compute intermediate for linear_out input
                if ok is not None and hasattr(g_mod, 'linear_in'):
                    with torch.no_grad():
                        intermediate = _compute_gating_intermediate(g_mod, x)
                        _accumulate_norm(activation_norms, sample_counts, ok, intermediate)
            return hook_fn

        hooks.append(gating_mod.register_forward_hook(
            make_gating_hook(gating_mod, in_key, out_key)
        ))

    logger.info(f"Running {len(calibration_data)} calibration batches for Wanda...")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(calibration_data, desc="Wanda calibration"):
            batch = batch.to(device)
            try:
                _ = model(batch)
            except Exception as e:
                logger.warning(f"Calibration batch failed: {e}")
                continue

    for h in hooks:
        h.remove()

    # Average the norms
    for name in activation_norms:
        if sample_counts[name] > 0:
            activation_norms[name] /= sample_counts[name]

    # Step 2: Prune by Wanda score = |W_ij| * ||X_j||_2
    per_layer_stats = {}

    for name, linear in linears:
        W = linear.weight.data  # [out_features, in_features]

        if name not in activation_norms:
            logger.warning(f"No activations captured for {name}, skipping")
            continue

        X_norm = activation_norms[name].to(W.device)  # [in_features]

        # Wanda score: element-wise |W| * ||X||  (broadcast over rows)
        wanda_score = W.abs() * X_norm.unsqueeze(0)  # [out, in]

        n_params = W.numel()
        n_prune = int(n_params * sparsity)

        if n_prune == 0:
            continue

        threshold = torch.kthvalue(wanda_score.flatten(), n_prune).values
        mask = wanda_score > threshold
        W.mul_(mask.to(W.dtype))

        actual = 1.0 - (mask.sum().item() / n_params)
        per_layer_stats[name] = actual

    return _compute_result("wanda", sparsity, linears, per_layer_stats)


# ──────────────────────────────────────────────────────────
#  Method 3: SparseGPT
# ──────────────────────────────────────────────────────────


def prune_sparsegpt(
    model: nn.Module,
    calibration_data: list[torch.Tensor],
    sparsity: float = 0.30,
    block_size: int = 128,
    percdamp: float = 0.01,
    device: str = "cuda",
) -> UnstructuredPruningResult:
    """Prune weights with SparseGPT (one-shot, Hessian-based reconstruction).

    For each Linear layer, computes the inverse Hessian from calibration
    activations, then iteratively prunes weights and updates the remaining
    weights to minimize the layer output reconstruction error.

    Memory-efficient: accumulates H = X^T @ X on-the-fly during calibration
    (~8 GB CPU RAM) instead of storing raw activations (~224 GB).

    Args:
        model: Moshi LMModel (mutated in-place).
        calibration_data: List of token tensors [B, K, T].
        sparsity: Fraction of weights to zero (0.30 = 30%).
        block_size: Column block size for blocked OBS updates.
        percdamp: Dampening factor for Hessian diagonal (numerical stability).
        device: Torch device.

    Returns:
        UnstructuredPruningResult with pruning statistics.
    """
    transformer = _get_transformer(model)
    linears = _get_prunable_linears(model)
    gating_children = _find_gating_children(model)

    # Step 1: Accumulate Hessians on-the-fly during calibration
    # For each Linear: H = sum(X^T @ X) over all calibration samples
    # Stored on CPU to save GPU VRAM (~64 MB per [4096, 4096] matrix)
    hessians: dict[str, torch.Tensor] = {}
    hessian_counts: dict[str, int] = {}
    hooks = []

    def _accumulate_hessian(name: str, x: torch.Tensor):
        """Accumulate H += X^T @ X for a Linear layer's input."""
        x_flat = x.reshape(-1, x.shape[-1]).float()  # [N, in_features], GPU
        # Compute H contribution on GPU (fast), then move to CPU (small)
        H_contrib = x_flat.t() @ x_flat  # [in_features, in_features]
        H_cpu = H_contrib.cpu()
        if name not in hessians:
            hessians[name] = torch.zeros_like(H_cpu)
            hessian_counts[name] = 0
        hessians[name] += H_cpu
        hessian_counts[name] += x_flat.shape[0]

    # 1a. Standard hooks on non-gating Linears (attention projections)
    def make_hook(name: str):
        def hook_fn(module, inputs, output):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            _accumulate_hessian(name, x)
        return hook_fn

    for name, linear in linears:
        if name not in gating_children:
            hooks.append(linear.register_forward_hook(make_hook(name)))

    # 1b. Hook parent gating modules for their child Linears
    hooked_gating_ids = set()
    for name, info in gating_children.items():
        gating_mod = info['gating']
        gid = id(gating_mod)
        if gid in hooked_gating_ids:
            continue
        hooked_gating_ids.add(gid)

        in_key = next((n for n, i in gating_children.items()
                       if id(i['gating']) == gid and i['role'] == 'linear_in'), None)
        out_key = next((n for n, i in gating_children.items()
                        if id(i['gating']) == gid and i['role'] == 'linear_out'), None)

        def make_gating_hook(g_mod, ik, ok):
            def hook_fn(module, inputs, output):
                x = inputs[0] if isinstance(inputs, tuple) else inputs
                if ik is not None:
                    _accumulate_hessian(ik, x)
                if ok is not None and hasattr(g_mod, 'linear_in'):
                    with torch.no_grad():
                        intermediate = _compute_gating_intermediate(g_mod, x)
                        _accumulate_hessian(ok, intermediate)
            return hook_fn

        hooks.append(gating_mod.register_forward_hook(
            make_gating_hook(gating_mod, in_key, out_key)
        ))

    logger.info(f"Running {len(calibration_data)} calibration batches for SparseGPT...")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(calibration_data, desc="SparseGPT calibration"):
            batch = batch.to(device)
            try:
                _ = model(batch)
            except Exception as e:
                logger.warning(f"Calibration batch failed: {e}")
                continue

    for h in hooks:
        h.remove()

    hessian_mem = sum(H.numel() * 4 for H in hessians.values()) / 1e9
    logger.info(f"Hessians accumulated: {len(hessians)} layers, {hessian_mem:.2f} GB CPU RAM")

    # Step 2: For each layer, run blocked SparseGPT pruning
    per_layer_stats = {}
    total_layers = len(linears)

    for idx, (name, linear) in enumerate(
        tqdm(linears, desc="SparseGPT pruning", total=total_layers)
    ):
        if name not in hessians:
            logger.warning(f"No Hessian for {name}, skipping")
            continue

        W = linear.weight.data.clone().float()  # [out, in]
        d_row, d_col = W.shape

        # Normalize Hessian by sample count
        H = hessians[name].float().to(W.device) / hessian_counts[name]

        # Dampening for numerical stability
        damp = percdamp * torch.diag(H).mean()
        H.diagonal().add_(damp)

        # Cholesky decomposition of H → H_inv
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            del L
        except torch.linalg.LinAlgError:
            logger.warning(f"Cholesky failed for {name}, falling back to pseudo-inverse")
            H_inv = torch.linalg.pinv(H)

        # Free Hessian to save memory
        del hessians[name]
        del H

        # Blocked SparseGPT pruning
        mask = torch.ones_like(W, dtype=torch.bool)

        for col_start in range(0, d_col, block_size):
            col_end = min(col_start + block_size, d_col)

            W_block = W[:, col_start:col_end].clone()
            H_block_diag = torch.diag(H_inv[col_start:col_end, col_start:col_end])

            # Score = W^2 / diag(H^{-1}) — small score means safe to prune
            scores = W_block.pow(2) / (H_block_diag.unsqueeze(0) + 1e-10)

            # Determine how many to prune in this block
            n_block_params = W_block.numel()
            n_prune_block = int(n_block_params * sparsity)

            if n_prune_block > 0:
                threshold = torch.kthvalue(scores.flatten(), n_prune_block).values
                prune_mask = scores <= threshold

                # Zero out pruned weights
                W_block[prune_mask] = 0
                mask[:, col_start:col_end][prune_mask] = False

                # Weight reconstruction: update future columns to compensate
                errors = W[:, col_start:col_end] - W_block
                if col_end < d_col:
                    H_cross = H_inv[col_start:col_end, col_end:]
                    correction = errors @ H_cross
                    W[:, col_end:] -= correction

            W[:, col_start:col_end] = W_block

        # Apply pruned weights back to the module
        linear.weight.data.copy_(W.to(linear.weight.dtype))

        actual = 1.0 - (mask.float().sum().item() / W.numel())
        per_layer_stats[name] = actual

        # Free H_inv and intermediate tensors
        del H_inv, W, mask
        if device == "cuda":
            torch.cuda.empty_cache()

    return _compute_result("sparsegpt", sparsity, linears, per_layer_stats)


# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────


def _compute_result(
    method: str,
    target_sparsity: float,
    linears: list[tuple[str, nn.Linear]],
    per_layer_stats: dict[str, float],
) -> UnstructuredPruningResult:
    """Compute overall pruning statistics from per-layer results."""
    total_params = 0
    zeroed_params = 0

    for name, linear in linears:
        n = linear.weight.numel()
        total_params += n
        z = (linear.weight.data == 0).sum().item()
        zeroed_params += z

    actual_sparsity = zeroed_params / total_params if total_params > 0 else 0.0

    result = UnstructuredPruningResult(
        method=method,
        target_sparsity=target_sparsity,
        actual_sparsity=actual_sparsity,
        total_params=total_params,
        zeroed_params=zeroed_params,
        n_layers_pruned=len(per_layer_stats),
        per_layer_sparsity=per_layer_stats,
    )

    logger.info(
        f"✅ {method} pruning complete: "
        f"target={target_sparsity:.1%}, actual={actual_sparsity:.1%}, "
        f"zeroed={zeroed_params:,} / {total_params:,} params across "
        f"{len(per_layer_stats)} layers"
    )

    return result


def get_model_sparsity(model: nn.Module) -> dict:
    """Compute current sparsity statistics for a model.

    Returns:
        Dict with total_params, nonzero_params, zero_params, sparsity.
    """
    linears = _get_prunable_linears(model)
    total = 0
    zeros = 0
    for name, linear in linears:
        n = linear.weight.numel()
        z = (linear.weight.data == 0).sum().item()
        total += n
        zeros += z
    return {
        "total_params": total,
        "nonzero_params": total - zeros,
        "zero_params": zeros,
        "sparsity": zeros / total if total > 0 else 0.0,
    }


def prepare_calibration_from_self_play(
    conversations_dir: str,
    n_conversations: int = 200,
    seq_len: int = 500,
    batch_size: int = 4,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Load self-play conversations as calibration data for Wanda/SparseGPT.

    Reads saved .npz conversation files and constructs token tensors
    in the format expected by the Moshi LMModel forward pass.

    Moshi expects [B, K, T] where K = n_q + 1 = 17:
        Channel 0:    text (inner monologue)
        Channels 1-8:  model audio CB0..7 (Depformer output)
        Channels 9-16: user audio CB0..7  (input to temporal transformer)

    The .npz files from self-play generation store:
        text_tokens:       [T]    — text channel
        audio_tokens:      [8, T] — model audio (Channel B)
        user_audio_tokens: [8, T] — user audio (Channel A)

    Args:
        conversations_dir: Path to directory containing .npz conversation files.
        n_conversations: Number of conversations to load.
        seq_len: Steps per sample (truncated/padded).
        batch_size: Samples per batch tensor.
        device: Target device.

    Returns:
        List of tensors [batch_size, 17, seq_len].
    """
    from pathlib import Path

    N_CHANNELS = 17  # 1 text + 8 model audio + 8 user audio

    conv_dir = Path(conversations_dir)
    npz_files = sorted(conv_dir.glob("**/*.npz"))
    npz_files = [f for f in npz_files if f.name.startswith("conv_")]

    if not npz_files:
        raise FileNotFoundError(
            f"No conversation .npz files found in {conversations_dir}"
        )

    npz_files = npz_files[:n_conversations]
    all_tokens = []

    for npz_path in npz_files:
        data = np.load(str(npz_path))
        text_tokens = data["text_tokens"]              # [T]
        audio_tokens = data["audio_tokens"]            # [8, T] — model audio
        user_audio = data.get("user_audio_tokens")     # [8, T] — user audio

        T = min(len(text_tokens), seq_len)

        # Build [17, seq_len]: [text, model_cb0..7, user_cb0..7]
        tokens = np.zeros((N_CHANNELS, seq_len), dtype=np.int64)
        tokens[0, :T] = text_tokens[:T]

        n_model_audio = min(audio_tokens.shape[0], 8)
        tokens[1:1 + n_model_audio, :T] = audio_tokens[:n_model_audio, :T]

        if user_audio is not None:
            n_user_audio = min(user_audio.shape[0], 8)
            tokens[9:9 + n_user_audio, :T] = user_audio[:n_user_audio, :T]

        all_tokens.append(tokens)

    # Group into batches
    batches = []
    for i in range(0, len(all_tokens), batch_size):
        batch = all_tokens[i:i + batch_size]
        while len(batch) < batch_size:
            batch.append(batch[-1])
        tensor = torch.tensor(np.stack(batch), dtype=torch.long)
        batches.append(tensor)

    logger.info(
        f"Loaded {len(all_tokens)} conversations as {len(batches)} calibration batches "
        f"(channels={N_CHANNELS}, seq_len={seq_len}, batch_size={batch_size})"
    )
    return batches
