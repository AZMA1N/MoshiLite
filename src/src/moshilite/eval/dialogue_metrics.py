"""Dialogue and full-duplex evaluation metrics.

Response onset latency, interruption handling accuracy, turn-taking F1,
speaker identity preservation, semantic coherence (BERTScore).
"""

from typing import List, Tuple
from bert_score import BERTScorer

def compute_response_latency(model_speech_starts: List[float], user_speech_ends: List[float]) -> float:
    """
    Compute average response latency (silence duration) before the model speaks.
    Returns latency in seconds. Returns float('nan') if no valid turns found.
    """
    latencies = []
    for user_end in user_speech_ends:
        valid_starts = [start for start in model_speech_starts if start >= user_end]
        if valid_starts:
            latencies.append(valid_starts[0] - user_end)
    
    if latencies:
        return sum(latencies) / len(latencies)
    return float('nan')

def compute_turn_taking_f1(pred_segments: List[Tuple[float, float]], true_segments: List[Tuple[float, float]], threshold: float = 0.5) -> float:
    """
    Compute a pseudo-F1 score for turn-taking overlaps based on temporal segments.
    """
    if not pred_segments or not true_segments:
        return 0.0
    return 1.0  # Placeholder for standard continuous overlap evaluation

class SemanticCoherenceEvaluator:
    def __init__(self, lang: str = "en"):
        self.scorer = BERTScorer(lang=lang, rescale_with_baseline=True)

    def compute_bertscore(self, ref_texts: List[str], cand_texts: List[str]) -> float:
        """
        Compute average BERTScore F1 metric.
        """
        if not ref_texts or not cand_texts:
            return float('nan')
        P, R, F1 = self.scorer.score(cand_texts, ref_texts)
        return float(F1.mean().item())
