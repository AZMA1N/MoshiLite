"""Moshi-specific hook implementations for layer analysis.

This module bridges the generic analysis interfaces (layer_hook_fn,
ffn_hook_fn, head_mask_fn, loss_fn) with Moshi's actual
StreamingTransformerLayer internals.

All hook functions are designed to be passed as callbacks to the
analysis functions in block_influence.py, head_importance.py,
and ffn_importance.py.
"""

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Layer Hook: captures hidden states at each layer boundary
# ──────────────────────────────────────────────────────────


def moshi_layer_hook_fn(model: nn.Module, input_batch: Tensor) -> list[Tensor]:
    """Collect hidden states at every layer boundary of the Temporal Transformer.

    Registers forward hooks on each StreamingTransformerLayer to capture
    the input and output. Returns n_layers + 1 tensors:
        [input_to_layer_0, output_of_layer_0, ..., output_of_layer_{n-1}]

    Args:
        model: Moshi LMModel (in eval mode).
        input_batch: Token tensor of shape [B, K, T] where K = n_q + 1
            (text channel + audio codebooks).

    Returns:
        List of hidden state tensors, each [B, T, d_model].
    """
    transformer = _get_transformer(model)
    layers = list(transformer.layers)
    n_layers = len(layers)

    hidden_states: dict[int, Tensor] = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inputs, output):
            if layer_idx == 0:
                # Capture input to the first layer
                inp = inputs[0] if isinstance(inputs, tuple) else inputs
                hidden_states[-1] = inp.detach().cpu()
            # Capture output of this layer
            out = output if isinstance(output, Tensor) else output[0]
            hidden_states[layer_idx] = out.detach().cpu()
        return hook_fn

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

        # Run forward pass
        with torch.no_grad():
            _ = model(input_batch)

    finally:
        for h in hooks:
            h.remove()

    # Assemble ordered list: [input, layer_0_out, layer_1_out, ..., layer_{n-1}_out]
    result = [hidden_states[-1]]  # input to first layer
    for i in range(n_layers):
        result.append(hidden_states[i])

    return result


# ──────────────────────────────────────────────────────────
#  FFN Hook: captures intermediate FFN activations
# ──────────────────────────────────────────────────────────


def moshi_ffn_hook_fn(model: nn.Module, input_batch: Tensor, layer_idx: int) -> Tensor:
    """Capture FFN intermediate activations for a specific layer.

    Hooks into the gating module's internal up-projection or the linear1
    layer to capture the intermediate representation (d_ffn-dimensional)
    before the down-projection.

    Args:
        model: Moshi LMModel.
        input_batch: Token tensor [B, K, T].
        layer_idx: Which transformer layer to hook.

    Returns:
        Activation tensor [n_tokens, d_ffn] flattened over batch and seq.
    """
    transformer = _get_transformer(model)
    layer = list(transformer.layers)[layer_idx]
    captured = {}

    def hook_fn(module, inputs, output):
        out = output if isinstance(output, Tensor) else output[0]
        captured["activations"] = out.detach().cpu()

    hook = None
    try:
        if hasattr(layer, "gating") and layer.gating is not None:
            gating = layer.gating
            if isinstance(gating, nn.ModuleList):
                gating = gating[0]
            # Hook into the internal up-projection to get d_ffn-dimensional output.
            # Moshi's gating modules (SiGLU etc.) typically have:
            #   - linear_in: projects to 2*d_ffn (gate + up combined)
            #   - linear: projects to d_ffn
            #   - up/gate: separate projections
            # We look for the first internal Linear that outputs d_ffn.
            target = _find_ffn_intermediate(gating)
            if target is not None:
                hook = target.register_forward_hook(hook_fn)
            else:
                # Fallback: hook the gating module itself
                hook = gating.register_forward_hook(hook_fn)
        elif hasattr(layer, "linear1") and layer.linear1 is not None:
            hook = layer.linear1.register_forward_hook(hook_fn)
        else:
            raise RuntimeError(f"Layer {layer_idx} has no recognizable FFN module")

        with torch.no_grad():
            _ = model(input_batch)

    finally:
        if hook is not None:
            hook.remove()

    acts = captured["activations"]
    # Flatten batch and sequence dimensions: [B, T, d_ffn] -> [B*T, d_ffn]
    if acts.dim() == 3:
        acts = acts.reshape(-1, acts.shape[-1])
    return acts


def _find_ffn_intermediate(gating: nn.Module) -> nn.Module | None:
    """Find the internal linear layer that produces d_ffn-dimensional output.

    Searches through common gating module patterns:
    - SiGLU: has gate/up projections (d_model -> d_ffn)
    - linear_in: combined projection (d_model -> 2*d_ffn)
    - Standard: linear1 (d_model -> d_ffn)
    """
    # Try common attribute names for the up-projection
    for name in ("up", "linear_up", "w2", "gate", "linear_in", "linear"):
        child = getattr(gating, name, None)
        if isinstance(child, nn.Linear) and child.out_features > child.in_features:
            return child

    # Fallback: find the first Linear with out > in
    for module in gating.modules():
        if isinstance(module, nn.Linear) and module.out_features > module.in_features:
            return module

    return None


# ──────────────────────────────────────────────────────────
#  Head Mask: per-head scaling for gradient-based importance
# ──────────────────────────────────────────────────────────


def moshi_head_mask_fn(model: nn.Module, head_mask: Tensor) -> None:
    """Apply per-head mask to the model's attention output projections.

    Since Moshi uses F.scaled_dot_product_attention (which doesn't expose
    raw attention weights), we scale the output projection weights per head
    to simulate head masking for gradient-based importance.

    Args:
        model: Moshi LMModel.
        head_mask: Tensor [n_layers, n_heads] with values in [0, 1].
            Gradient flows through this tensor to compute importance.
    """
    transformer = _get_transformer(model)
    layers = list(transformer.layers)

    for i, layer in enumerate(layers):
        if not hasattr(layer, "self_attn") or layer.skip_self_attn:
            continue

        attn = layer.self_attn
        # The output projection: out_projs is a ModuleList
        # Each out_proj: [d_model, d_model] — we scale by head
        d_head = attn.embed_dim // attn.num_heads

        for proj in attn.out_projs:
            if isinstance(proj, nn.Linear):
                # Scale each head's contribution in the output projection
                # Weight shape: [d_model, d_model]
                # Reshape as [d_model, n_heads, d_head] then scale
                w = proj.weight.view(proj.out_features, attn.num_heads, d_head)
                # head_mask[i] is [n_heads], broadcast to scale each head's columns
                scale = head_mask[i].view(1, attn.num_heads, 1)
                proj.weight.data = (w * scale).view_as(proj.weight)


def moshi_loss_fn(model_output, targets: Tensor) -> Tensor:
    """Compute cross-entropy loss from LMModel output.

    Args:
        model_output: LMOutput with .logits [B, K, T, card] and
            .text_logits [B, 1, T, text_card].
        targets: Input codes [B, K, T] (shifted internally by model).

    Returns:
        Scalar loss.
    """
    # Audio logits loss
    logits = model_output.logits  # [B, K, T, card]
    mask = model_output.mask      # [B, K, T]
    B, K, T, C = logits.shape

    # We must explicitly slice targets to dynamically match K_out (model's dep_q)!
    audio_targets = targets[:, 1 : 1 + K, :T] 
    
    audio_loss = torch.nn.functional.cross_entropy(
        logits[mask].view(-1, C),
        audio_targets[mask[:, :, :T].expand_as(audio_targets)].reshape(-1),
        reduction="mean",
    )

    # Text logits loss
    text_logits = model_output.text_logits  # [B, 1, T, text_card]
    text_mask = model_output.text_mask       # [B, 1, T]
    TC = text_logits.shape[-1]

    if text_mask.any():
        text_loss = torch.nn.functional.cross_entropy(
            text_logits[text_mask].view(-1, TC),
            targets[:, 0:1, :T][text_mask].reshape(-1),
            reduction="mean",
        )
        return audio_loss + text_loss

    return audio_loss


# ──────────────────────────────────────────────────────────
#  Logit Extraction
# ──────────────────────────────────────────────────────────


def moshi_get_logits(
    model: nn.Module, input_batch: Tensor
) -> tuple[Tensor, Tensor]:
    """Run forward pass and extract logits.

    Args:
        model: Moshi LMModel.
        input_batch: Token tensor [B, K, T].

    Returns:
        Tuple of (audio_logits [B, K, T, card], text_logits [B, 1, T, text_card]).
    """
    with torch.no_grad():
        output = model(input_batch)
    return output.logits.detach(), output.text_logits.detach()


# ──────────────────────────────────────────────────────────
#  Hidden State Extraction at Specific Layers
# ──────────────────────────────────────────────────────────


def moshi_get_hidden_states(
    model: nn.Module,
    input_batch: Tensor,
    layer_indices: list[int],
) -> list[Tensor]:
    """Extract hidden states at specific layer indices.

    Args:
        model: Moshi LMModel.
        input_batch: Token tensor [B, K, T].
        layer_indices: Which layers to capture.

    Returns:
        List of hidden state tensors [B, T, d_model], one per requested layer.
    """
    transformer = _get_transformer(model)
    layers = list(transformer.layers)
    captured: dict[int, Tensor] = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inputs, output):
            out = output if isinstance(output, Tensor) else output[0]
            captured[idx] = out.detach()
        return hook_fn

    try:
        for idx in layer_indices:
            hooks.append(layers[idx].register_forward_hook(make_hook(idx)))
        with torch.no_grad():
            _ = model(input_batch)
    finally:
        for h in hooks:
            h.remove()

    return [captured[idx] for idx in layer_indices]


# ──────────────────────────────────────────────────────────
#  Token Batch Preparation from Pre-Encoded Shards
# ──────────────────────────────────────────────────────────


def prepare_token_batches(
    token_dir: str,
    n_samples: int = 200,
    seq_len: int = 500,
    n_codebooks: int = 9,  # 1 text + 8 audio
    batch_size: int = 4,
    dataset_name: Optional[str] = None,
) -> list[Tensor]:
    """Load pre-encoded tokens from tar shards into batches for analysis.

    Reads WebDataset tar shards, extracts token arrays, truncates/pads
    to seq_len, and groups into batches.

    Args:
        token_dir: Directory containing .tar shard files.
        n_samples: Total number of samples to load.
        seq_len: Sequence length to truncate/pad to (in timesteps).
        n_codebooks: Number of codebook channels (text + audio = 9).
        batch_size: Samples per batch.
        dataset_name: Optional filter — only load shards matching this name.

    Returns:
        List of tensors, each [batch_size, n_codebooks, seq_len].
    """
    token_path = Path(token_dir)
    tar_files = sorted(token_path.glob("*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No .tar shard files found in {token_dir}")

    if dataset_name:
        tar_files = [f for f in tar_files if dataset_name in f.name]

    all_tokens = []
    for tar_path in tar_files:
        if len(all_tokens) >= n_samples:
            break
        try:
            with tarfile.open(str(tar_path), "r") as tf:
                for member in tf:
                    if len(all_tokens) >= n_samples:
                        break
                    if not member.name.endswith(".tokens.npy"):
                        continue

                    f = tf.extractfile(member)
                    if f is None:
                        continue

                    tokens = np.load(io.BytesIO(f.read()))
                    # tokens shape: [n_codebooks, T] — from encoding pipeline
                    if tokens.ndim == 2:
                        n_cb, t = tokens.shape
                        if t < seq_len:
                            tokens = np.pad(tokens, ((0, 0), (0, seq_len - t)))
                        else:
                            tokens = tokens[:, :seq_len]

                        # Ensure we have the right number of codebooks
                        if n_cb < n_codebooks:
                            # Pad with zeros (padding tokens) for missing text channel
                            tokens = np.pad(tokens, ((0, n_codebooks - n_cb), (0, 0)))
                        elif n_cb > n_codebooks:
                            tokens = tokens[:n_codebooks]

                        all_tokens.append(tokens)
        except Exception as e:
            logger.warning("Error reading %s: %s", tar_path, e)
            continue

    if not all_tokens:
        raise RuntimeError(f"No tokens loaded from {token_dir}")

    logger.info("Loaded %d token samples from %s", len(all_tokens), token_dir)

    # Group into batches
    batches = []
    for i in range(0, len(all_tokens), batch_size):
        batch = all_tokens[i : i + batch_size]
        if len(batch) < batch_size:
            # Pad last batch
            while len(batch) < batch_size:
                batch.append(batch[-1])
        tensor = torch.tensor(np.stack(batch), dtype=torch.long)
        batches.append(tensor)

    return batches


# ──────────────────────────────────────────────────────────
#  Internal Helpers
# ──────────────────────────────────────────────────────────


def _get_transformer(model: nn.Module) -> nn.Module:
    """Get the Temporal Transformer from an LMModel."""
    if hasattr(model, "transformer"):
        return model.transformer
    # Try unwrapping DataParallel or similar wrappers
    if hasattr(model, "module"):
        return _get_transformer(model.module)
    raise AttributeError(
        "Model has no 'transformer' attribute. Expected a Moshi LMModel."
    )


def get_n_layers(model: nn.Module) -> int:
    """Get the number of transformer layers."""
    transformer = _get_transformer(model)
    return len(list(transformer.layers))


def get_n_heads(model: nn.Module) -> int:
    """Get the number of attention heads per layer."""
    transformer = _get_transformer(model)
    layer0 = list(transformer.layers)[0]
    return layer0.self_attn.num_heads
