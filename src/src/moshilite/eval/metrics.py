"""Speech quality metrics: PESQ, STOI, WER wrappers."""

import torch
import torchaudio
import numpy as np
from pesq import pesq
from pystoi import stoi
import jiwer
from resemblyzer import VoiceEncoder, preprocess_wav

def compute_pesq(ref_audio: torch.Tensor, deg_audio: torch.Tensor, sr: int) -> float:
    """Compute PESQ score (narrowband/wideband based on sample rate)."""
    ref_np = ref_audio.squeeze().cpu().numpy()
    deg_np = deg_audio.squeeze().cpu().numpy()
    mode = 'wb' if sr >= 16000 else 'nb'
    try:
        score = pesq(sr, ref_np, deg_np, mode)
    except Exception as e:
        score = float('nan')
    return float(score)

def compute_stoi(ref_audio: torch.Tensor, deg_audio: torch.Tensor, sr: int) -> float:
    """Compute STOI score."""
    ref_np = ref_audio.squeeze().cpu().numpy()
    deg_np = deg_audio.squeeze().cpu().numpy()
    try:
        score = stoi(ref_np, deg_np, sr, extended=False)
    except Exception as e:
        score = float('nan')
    return float(score)

def compute_wer(ref_text: str, cand_text: str) -> float:
    """Compute Word Error Rate (WER)."""
    try:
        score = jiwer.wer(ref_text, cand_text)
    except Exception as e:
        score = float('nan')
    return float(score)

class SpeakerSimilarityEvaluator:
    """Evaluator for speaker cosine similarity using Resemblyzer."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.encoder = VoiceEncoder(device=device)
    
    def compute_similarity(self, ref_wav_path: str, deg_wav_path: str) -> float:
        """Compute cosine similarity between two audio files' speaker embeddings."""
        try:
            ref_wav = preprocess_wav(ref_wav_path)
            deg_wav = preprocess_wav(deg_wav_path)
            
            ref_embed = self.encoder.embed_utterance(ref_wav)
            deg_embed = self.encoder.embed_utterance(deg_wav)
            
            cos_sim = np.dot(ref_embed, deg_embed) / (np.linalg.norm(ref_embed) * np.linalg.norm(deg_embed))
            return float(cos_sim)
        except Exception as e:
            return float('nan')
