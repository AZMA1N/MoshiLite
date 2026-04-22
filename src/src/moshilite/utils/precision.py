"""INT4/INT8 precision validation gates.

Used before Stage 1 (layer analysis) and Stage 4a (teacher precompute)
to verify that quantized inference doesn't distort measurements.
"""

import torch
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class PrecisionReport:
    """Results of a precision validation gate."""
    spearman_rho: Optional[float] = None  # BI score rank correlation
    cosine_drift: Optional[float] = None  # Mean cosine sim drift per layer
    kl_divergence: Optional[float] = None  # Mean KL-div of output logits
    top50_agreement: Optional[float] = None  # Stage 4a: logit overlap rate
    hidden_cosine_sim: Optional[float] = None  # Stage 4a: per-layer hidden cosine
    codebook_agreement: Optional[float] = None  # Stage 4a: codebook pred agreement
    passed: bool = False
    recommended_precision: str = "int8"  # fallback

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def validate_stage1_precision(
    models: dict,
    calibration_data: list,
    compute_bi_fn,
    thresholds: dict,
    device: str = "cuda",
) -> PrecisionReport:
    """Stage 1 precision gate: compare INT4 vs INT8 BI scores and outputs.

    Args:
        models: Dict with 'int4' and 'int8' keys mapping to loaded models.
        calibration_data: List of token batches for calibration.
        compute_bi_fn: Function that computes BI scores given model + data.
            Signature: compute_bi_fn(model, data) -> np.ndarray of shape [n_layers].
        thresholds: Dict with keys 'spearman_rho', 'cosine_drift', 'kl_divergence'.
        device: Device to run on.

    Returns:
        PrecisionReport with gate decision.
    """
    report = PrecisionReport()

    # --- Compute BI scores at both precisions ---
    bi_int4 = compute_bi_fn(models["int4"], calibration_data)
    bi_int8 = compute_bi_fn(models["int8"], calibration_data)

    # Spearman rank correlation of BI scores
    rho, _ = stats.spearmanr(bi_int4, bi_int8)
    report.spearman_rho = float(rho)

    # --- Compute hidden state cosine drift per layer ---
    drifts = _compute_cosine_drifts(models, calibration_data, device)
    report.cosine_drift = float(np.mean(drifts))

    # --- Compute KL divergence of output logits ---
    kl = _compute_kl_divergence(models, calibration_data, device)
    report.kl_divergence = float(kl)

    # --- Gate decision ---
    report.passed = (
        report.spearman_rho >= thresholds["spearman_rho"]
        and report.cosine_drift < thresholds["cosine_drift"]
        and report.kl_divergence < thresholds["kl_divergence"]
    )
    report.recommended_precision = "int4" if report.passed else "int8"

    return report



def validate_stage4a_precision(
    model,
    calibration_data: list,
    thresholds: dict,
    aligned_teacher_layers: list[int],
    device: str = "cuda",
) -> PrecisionReport:
    """Stage 4a precision gate: stricter thresholds for teacher pre-computation.

    Args:
        model: Moshi teacher model.
        calibration_data: List of token batches.
        thresholds: Dict with keys 'top50_agreement', 'hidden_cosine_sim',
            'codebook_agreement'.
        aligned_teacher_layers: Layer indices to check hidden states for.
        device: Device to run on.

    Returns:
        PrecisionReport with gate decision.
    """
    report = PrecisionReport()

    # --- Top-50 logit agreement ---
    report.top50_agreement = _compute_top50_agreement(
        model, calibration_data, device
    )

    # --- Hidden state cosine similarity at aligned layers ---
    report.hidden_cosine_sim = _compute_aligned_hidden_cosine(
        model, calibration_data, aligned_teacher_layers, device
    )

    # --- Codebook prediction agreement ---
    report.codebook_agreement = _compute_codebook_agreement(
        model, calibration_data, device
    )

    # --- Gate decision ---
    report.passed = (
        report.top50_agreement >= thresholds["top50_agreement"]
        and report.hidden_cosine_sim >= thresholds["hidden_cosine_sim"]
        and report.codebook_agreement >= thresholds["codebook_agreement"]
    )
    report.recommended_precision = "int4" if report.passed else "int8"

    return report


# === Internal helpers ===
# These functions compare outputs between two models loaded at different
# precisions (e.g., INT4 vs INT8). They use hooks from moshi_hooks.py.


def _compute_cosine_drifts(model, data, device):
    """Compute per-layer mean cosine similarity drift between INT4 and INT8.

    Args:
        model: Dict with 'int4' and 'int8' keys mapping to loaded models,
               OR a single model (legacy, will raise).
        data: List of token batch tensors [B, K, T].
        device: Device string.

    Returns:
        np.ndarray of drift values per layer (1 - cosine_sim).
    """
    from moshilite.analysis.moshi_hooks import moshi_layer_hook_fn

    if not isinstance(model, dict):
        raise TypeError(
            "Expected dict with 'int4' and 'int8' model entries. "
            "Use validate_stage1_precision_v2() for the updated API."
        )

    model_a, model_b = model["int4"], model["int8"]
    n_batches = len(data)
    all_drifts = []

    for batch in data:
        batch = batch.to(device)
        hs_a = moshi_layer_hook_fn(model_a, batch)
        hs_b = moshi_layer_hook_fn(model_b, batch)

        # Compare layer outputs (skip input, index 1..n_layers)
        n_layers = len(hs_a) - 1
        drifts = np.zeros(n_layers)
        for i in range(n_layers):
            ha = hs_a[i + 1].float().mean(dim=(0, 1))  # [d_model]
            hb = hs_b[i + 1].float().mean(dim=(0, 1))
            cos = torch.nn.functional.cosine_similarity(
                ha.unsqueeze(0), hb.unsqueeze(0)
            ).item()
            drifts[i] = 1.0 - cos
        all_drifts.append(drifts)

    return np.mean(all_drifts, axis=0)


def _compute_kl_divergence(model, data, device):
    """Compute mean KL divergence of output logits between INT4 and INT8.

    Args:
        model: Dict with 'int4' and 'int8' model entries.
        data: List of token batch tensors.
        device: Device string.

    Returns:
        Float — mean KL divergence.
    """
    from moshilite.analysis.moshi_hooks import moshi_get_logits

    if not isinstance(model, dict):
        raise TypeError("Expected dict with 'int4' and 'int8' model entries.")

    model_a, model_b = model["int4"], model["int8"]
    kl_sum = 0.0
    count = 0

    for batch in data:
        batch = batch.to(device)
        audio_a, text_a = moshi_get_logits(model_a, batch)
        audio_b, text_b = moshi_get_logits(model_b, batch)

        # KL divergence on audio logits
        log_p = torch.nn.functional.log_softmax(audio_a.float(), dim=-1)
        q = torch.nn.functional.softmax(audio_b.float(), dim=-1)
        kl_audio = torch.nn.functional.kl_div(
            log_p.reshape(-1, log_p.shape[-1]),
            q.reshape(-1, q.shape[-1]),
            reduction="batchmean",
        ).item()

        # KL divergence on text logits
        log_p_t = torch.nn.functional.log_softmax(text_a.float(), dim=-1)
        q_t = torch.nn.functional.softmax(text_b.float(), dim=-1)
        kl_text = torch.nn.functional.kl_div(
            log_p_t.reshape(-1, log_p_t.shape[-1]),
            q_t.reshape(-1, q_t.shape[-1]),
            reduction="batchmean",
        ).item()

        kl_sum += (kl_audio + kl_text) / 2
        count += 1

    return kl_sum / max(count, 1)


def _compute_top50_agreement(model, data, device):
    """Compute % overlap of top-50 logits between INT4 and INT8.

    For each position, finds the top-50 token indices at each precision
    and computes the intersection percentage.

    Returns:
        Float in [0, 1] — mean top-50 overlap rate.
    """
    from moshilite.analysis.moshi_hooks import moshi_get_logits

    if not isinstance(model, dict):
        raise TypeError("Expected dict with 'int4' and 'int8' model entries.")

    model_a, model_b = model["int4"], model["int8"]
    agreement_sum = 0.0
    count = 0

    for batch in data:
        batch = batch.to(device)
        audio_a, text_a = moshi_get_logits(model_a, batch)
        audio_b, text_b = moshi_get_logits(model_b, batch)

        # Top-50 agreement on audio logits
        for logits_a, logits_b in [(audio_a, audio_b), (text_a, text_b)]:
            flat_a = logits_a.reshape(-1, logits_a.shape[-1])
            flat_b = logits_b.reshape(-1, logits_b.shape[-1])
            top_a = torch.topk(flat_a, k=50, dim=-1).indices  # [N, 50]
            top_b = torch.topk(flat_b, k=50, dim=-1).indices

            # Per-position overlap
            for pos in range(min(flat_a.shape[0], 1000)):  # cap for speed
                set_a = set(top_a[pos].cpu().numpy())
                set_b = set(top_b[pos].cpu().numpy())
                agreement_sum += len(set_a & set_b) / 50.0
                count += 1

    return agreement_sum / max(count, 1)


def _compute_aligned_hidden_cosine(model, data, aligned_layers, device):
    """Compute mean cosine similarity of hidden states at aligned layers.

    Args:
        model: Dict with 'int4' and 'int8' model entries.
        data: List of token batch tensors.
        aligned_layers: List of layer indices to compare.
        device: Device string.

    Returns:
        Float — mean cosine similarity across aligned layers and data.
    """
    from moshilite.analysis.moshi_hooks import moshi_get_hidden_states

    if not isinstance(model, dict):
        raise TypeError("Expected dict with 'int4' and 'int8' model entries.")

    model_a, model_b = model["int4"], model["int8"]
    sim_sum = 0.0
    count = 0

    for batch in data:
        batch = batch.to(device)
        hs_a = moshi_get_hidden_states(model_a, batch, aligned_layers)
        hs_b = moshi_get_hidden_states(model_b, batch, aligned_layers)

        for ha, hb in zip(hs_a, hs_b):
            # Average over batch and seq → [d_model]
            va = ha.float().mean(dim=(0, 1))
            vb = hb.float().mean(dim=(0, 1))
            cos = torch.nn.functional.cosine_similarity(
                va.unsqueeze(0), vb.unsqueeze(0)
            ).item()
            sim_sum += cos
            count += 1

    return sim_sum / max(count, 1)


def _compute_codebook_agreement(model, data, device):
    """Compute codebook prediction agreement rate between INT4 and INT8.

    For codebooks 1–3, compares the argmax predictions at each position.

    Returns:
        Float in [0, 1] — mean agreement rate.
    """
    from moshilite.analysis.moshi_hooks import moshi_get_logits

    if not isinstance(model, dict):
        raise TypeError("Expected dict with 'int4' and 'int8' model entries.")

    model_a, model_b = model["int4"], model["int8"]
    agree_sum = 0.0
    total = 0

    for batch in data:
        batch = batch.to(device)
        audio_a, _ = moshi_get_logits(model_a, batch)
        audio_b, _ = moshi_get_logits(model_b, batch)

        # Compare codebooks 0-2 (first 3 audio codebooks)
        n_codebooks = min(3, audio_a.shape[1])
        for cb in range(n_codebooks):
            pred_a = audio_a[:, cb].argmax(dim=-1)  # [B, T]
            pred_b = audio_b[:, cb].argmax(dim=-1)
            agree_sum += (pred_a == pred_b).float().mean().item()
            total += 1

    return agree_sum / max(total, 1)
