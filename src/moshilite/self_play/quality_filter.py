"""Quality filters for self-play conversations.

Detects degenerate outputs: repetition collapse, silence collapse,
low-energy conversations, and conversations too short for training.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class QualityReport:
    """Quality assessment for a single conversation."""
    accepted: bool
    rejection_reason: str | None  # None if accepted
    repetition_ratio: float       # fraction of repeated 4-grams
    silence_ratio: float          # fraction of silence tokens on model audio
    num_valid_steps: int          # steps with valid (non-delay) output
    details: dict                 # additional debug info


# Mimi silence token: in Moshiko, the audio codebook has card=2048.
# Token 0 is typically the silence / low-energy token.
# We heuristically define "silence" as the initial_token_id (card = 2048).
# The actual silence detection should be calibrated per-model.
_SILENCE_TOKEN_CANDIDATES = {0, 2048}


def compute_ngram_repetition(tokens: np.ndarray, n: int = 4) -> float:
    """Compute fraction of repeated n-grams in a 1D token sequence.

    Args:
        tokens: 1D array of token IDs.
        n: n-gram size.

    Returns:
        Fraction of n-grams that are repeats (0.0 = no repetition, 1.0 = all repeats).
    """
    if len(tokens) < n:
        return 0.0

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i + n]))

    total = len(ngrams)
    unique = len(set(ngrams))
    if total == 0:
        return 0.0
    return 1.0 - (unique / total)


def compute_silence_ratio(audio_tokens: np.ndarray) -> float:
    """Compute fraction of timesteps where the first model audio CB is a silence token.

    Args:
        audio_tokens: shape [dep_q=8, T] — the model's 8 audio codebooks.

    Returns:
        Fraction of timesteps that are silence on CB0.
    """
    if audio_tokens.shape[1] == 0:
        return 1.0
    cb0 = audio_tokens[0]  # first model audio codebook
    silence_mask = np.isin(cb0, list(_SILENCE_TOKEN_CANDIDATES))
    return float(silence_mask.sum()) / len(cb0)


def filter_conversation(
    text_tokens: np.ndarray,
    audio_tokens: np.ndarray,
    num_valid_steps: int,
    min_steps: int = 100,
    max_repetition: float = 0.92,  # For all-tokens repetition (including PAD)
    max_silence: float = 0.80,
    max_meaningful_repetition: float = 0.50,  # For non-PAD text tokens only
    max_second_half_repetition: float = 0.50,  # For non-PAD tokens in second half
) -> QualityReport:
    """Assess a single self-play conversation for quality.

    Args:
        text_tokens: shape [T] — Inner Monologue text tokens.
        audio_tokens: shape [dep_q=8, T] — model audio codebook tokens.
        num_valid_steps: number of timesteps with valid output (after delay fill).
        min_steps: reject if fewer than this many valid steps.
        max_repetition: reject if text 4-gram repetition ratio exceeds this.
        max_silence: reject if silence ratio on CB0 exceeds this.

    Returns:
        QualityReport with acceptance decision and details.
    """
    details = {}

    # Check 1: Minimum length
    if num_valid_steps < min_steps:
        return QualityReport(
            accepted=False,
            rejection_reason=f"too_short ({num_valid_steps} < {min_steps})",
            repetition_ratio=0.0,
            silence_ratio=0.0,
            num_valid_steps=num_valid_steps,
            details=details,
        )

    # Check 2: Text repetition
    rep_ratio = compute_ngram_repetition(text_tokens[:num_valid_steps], n=4)
    details["text_4gram_repetition"] = rep_ratio

    if rep_ratio > max_repetition:
        return QualityReport(
            accepted=False,
            rejection_reason=f"text_repetition ({rep_ratio:.2f} > {max_repetition})",
            repetition_ratio=rep_ratio,
            silence_ratio=0.0,
            num_valid_steps=num_valid_steps,
            details=details,
        )

    # Check 2b: Repetition among meaningful (non-PAD) text tokens
    # Standard repetition metric is inflated by PAD (~85-90% of text stream).
    # This checks actual content tokens for degenerate loops.
    meaningful = text_tokens[:num_valid_steps]
    meaningful = meaningful[meaningful > 3]
    if len(meaningful) >= 4:
        meaningful_rep = compute_ngram_repetition(meaningful, n=4)
        details["text_meaningful_4gram_repetition"] = meaningful_rep
        if meaningful_rep > max_meaningful_repetition:
            return QualityReport(
                accepted=False,
                rejection_reason=f"meaningful_text_repetition ({meaningful_rep:.2f} > {max_meaningful_repetition})",
                repetition_ratio=rep_ratio,
                silence_ratio=0.0,
                num_valid_steps=num_valid_steps,
                details=details,
            )

    # Check 2c: Second-half repetition (catches late-conversation loops)
    half = num_valid_steps // 2
    second_half = text_tokens[half:num_valid_steps]
    second_half_meaningful = second_half[second_half > 3]
    if len(second_half_meaningful) >= 4:
        sh_rep = compute_ngram_repetition(second_half_meaningful, n=4)
        details["text_meaningful_4gram_repetition_2nd_half"] = sh_rep
        if sh_rep > max_second_half_repetition:
            return QualityReport(
                accepted=False,
                rejection_reason=f"second_half_repetition ({sh_rep:.2f} > {max_second_half_repetition})",
                repetition_ratio=rep_ratio,
                silence_ratio=0.0,
                num_valid_steps=num_valid_steps,
                details=details,
            )

    # Check 3: Audio CB0 repetition
    audio_rep = compute_ngram_repetition(audio_tokens[0, :num_valid_steps], n=4)
    details["audio_cb0_4gram_repetition"] = audio_rep

    if audio_rep > max_repetition:
        return QualityReport(
            accepted=False,
            rejection_reason=f"audio_repetition ({audio_rep:.2f} > {max_repetition})",
            repetition_ratio=audio_rep,
            silence_ratio=0.0,
            num_valid_steps=num_valid_steps,
            details=details,
        )

    # Check 4: Silence collapse
    sil_ratio = compute_silence_ratio(audio_tokens[:, :num_valid_steps])
    details["silence_ratio"] = sil_ratio

    if sil_ratio > max_silence:
        return QualityReport(
            accepted=False,
            rejection_reason=f"silence_collapse ({sil_ratio:.2f} > {max_silence})",
            repetition_ratio=rep_ratio,
            silence_ratio=sil_ratio,
            num_valid_steps=num_valid_steps,
            details=details,
        )

    # All checks passed
    return QualityReport(
        accepted=True,
        rejection_reason=None,
        repetition_ratio=rep_ratio,
        silence_ratio=sil_ratio,
        num_valid_steps=num_valid_steps,
        details=details,
    )
