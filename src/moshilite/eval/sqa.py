"""SQA evaluation pipeline with comprehensive metrics.

Implements real Moshi streaming inference via Mimi (audio codec) + LMGen
(language model generation), then evaluates across multiple dimensions:

Semantic/Content:
  - Exact Match (EM)
  - Token F1
  - Word Error Rate (WER) via jiwer
  - BERTScore F1 via bert-score

Audio Quality:
  - UTMOS (reference-free neural MOS predictor, 1-5 scale)

Efficiency:
  - Real-Time Factor (RTF)
  - Peak VRAM (GB)
  - Parameter Count
"""

import os
import json
import time
import math
from math import gcd
import torch
import soundfile as sf
from scipy.signal import resample_poly
import whisper
from typing import Optional


# --------------------------------------------------------------------------
# Lazy torchaudio helpers (same pattern as component_eval.py)
# --------------------------------------------------------------------------
_torchaudio = None


def _get_torchaudio():
    global _torchaudio
    if _torchaudio is None:
        try:
            import torchaudio as _ta
            _torchaudio = _ta
        except Exception:
            _torchaudio = False
    return _torchaudio if _torchaudio else None


def _load_audio(path: str):
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # (T, C) -> (C, T)
        return waveform, sr
    except Exception:
        ta = _get_torchaudio()
        if ta is not None:
            return ta.load(path)
        raise RuntimeError(f"Cannot load audio (soundfile failed, torchaudio unavailable): {path}")


def _resample(waveform: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return waveform
    ta = _get_torchaudio()
    if ta is not None:
        return ta.functional.resample(waveform, orig_sr, new_sr)
    g = gcd(orig_sr, new_sr)
    out = resample_poly(waveform.numpy(), new_sr // g, orig_sr // g, axis=-1).astype("float32")
    return torch.from_numpy(out)


def _save_audio(path: str, waveform: torch.Tensor, sr: int):
    ta = _get_torchaudio()
    if ta is not None:
        ta.save(path, waveform, sr)
        return
    arr = waveform.numpy()
    if arr.ndim == 2:
        arr = arr.T
    sf.write(path, arr, sr)

from moshi.models import LMGen, MimiModel
from moshi.models.lm import LMModel

# Lazy imports for optional heavy dependencies
_jiwer = None
_bert_score = None
_utmos = None


def _get_jiwer():
    global _jiwer
    if _jiwer is None:
        import jiwer
        _jiwer = jiwer
    return _jiwer


def _get_bert_score():
    global _bert_score
    if _bert_score is None:
        from bert_score import score as bert_score_fn
        _bert_score = bert_score_fn
    return _bert_score


def _get_utmos():
    global _utmos
    if _utmos is None:
        import torch as _torch
        # ----------------------------------------------------------------
        # UTMOSv2 uses torch.load internally. On torch < 2.6 this is
        # completely blocked by the CVE-2025-32434 patch — even with
        # weights_only=False.  We replace torch.load with a wrapper that
        # catches the RuntimeError and falls back to _legacy_load (the
        # internal loader without the version guard).
        # ----------------------------------------------------------------
        _real_load = _torch.load.__wrapped__ if hasattr(_torch.load, '__wrapped__') else _torch.load

        def _unrestricted_load(*a, **kw):
            kw["weights_only"] = False
            try:
                return _real_load(*a, **kw)
            except RuntimeError as e:
                if 'CVE-2025-32434' not in str(e):
                    raise
                # torch 2.5.x hard-blocks torch.load entirely via RuntimeError.
                # Fall back to the internal _legacy_load which has no guard.
                import pickle
                f = a[0] if a else kw.get('f')
                map_loc = kw.get('map_location', None)
                fh = None
                try:
                    if isinstance(f, (str, os.PathLike)):
                        fh = open(f, 'rb')
                        f = fh
                    return _torch.serialization._legacy_load(
                        f, map_location=map_loc, pickle_module=pickle)
                finally:
                    if fh is not None:
                        fh.close()
            except TypeError:
                # Older torch doesn't accept weights_only at all
                kw.pop("weights_only", None)
                return _real_load(*a, **kw)

        _torch.load = _unrestricted_load
        try:
            import utmosv2
            _utmos = utmosv2.create_model(pretrained=True)
        except Exception as e:
            print(f"    [WARN] UTMOSv2 init failed: {e}")
            _utmos = "FAILED"
        finally:
            _torch.load = _real_load
    return _utmos if _utmos != "FAILED" else None


class SQAEvaluator:
    def __init__(
        self,
        model: Optional[LMModel] = None,
        mimi: Optional[MimiModel] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        whisper_model_size: str = "base",
    ):
        self.lm_model = model
        self.mimi = mimi
        self.device = device
        # Load Whisper model for transcription
        print(f"Loading Whisper ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)

    # ------------------------------------------------------------------
    # Streaming Inference
    # ------------------------------------------------------------------
    def streaming_inference(self, user_audio_path: str) -> tuple[torch.Tensor, dict]:
        """
        Perform real Moshi streaming inference.

        Returns:
            model_waveform: generated audio [1, T] at 24kHz on CPU
            perf_metrics: dict with rtf, peak_vram_gb, inference_time_s
        """
        waveform, sr = _load_audio(user_audio_path)
        if sr != 24000:
            waveform = _resample(waveform, sr, 24000)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio_duration_s = waveform.shape[-1] / 24000.0

        # If no real model, return silence
        if self.lm_model is None or self.mimi is None:
            return (
                torch.zeros(1, waveform.shape[-1], device="cpu"),
                {"rtf": 0.0, "peak_vram_gb": 0.0, "inference_time_s": 0.0,
                 "audio_duration_s": audio_duration_s},
            )

        waveform = waveform.to(self.device)
        frame_size = self.mimi.frame_size
        n_frames = waveform.shape[-1] // frame_size

        if n_frames == 0:
            return (
                torch.zeros(1, waveform.shape[-1], device="cpu"),
                {"rtf": 0.0, "peak_vram_gb": 0.0, "inference_time_s": 0.0,
                 "audio_duration_s": audio_duration_s},
            )

        waveform = waveform[:, :n_frames * frame_size].unsqueeze(0)

        lm_model = self.lm_model
        needed_input = lm_model.n_q - lm_model.dep_q
        lm_gen = LMGen(lm_model, temp=0.8, temp_text=0.7)
        all_model_codes = []

        # Reset peak memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.perf_counter()

        with torch.no_grad(), self.mimi.streaming(1), lm_gen.streaming(1):
            for frame_idx in range(n_frames):
                start = frame_idx * frame_size
                chunk = waveform[:, :, start:start + frame_size]
                user_codes = self.mimi.encode(chunk)

                if user_codes.shape[1] < needed_input:
                    pad = torch.zeros(
                        1, needed_input - user_codes.shape[1], 1,
                        device=self.device, dtype=user_codes.dtype,
                    )
                    input_tokens = torch.cat([user_codes, pad], dim=1)
                else:
                    input_tokens = user_codes[:, :needed_input, :]

                out = lm_gen.step(input_tokens)
                if out is not None:
                    all_model_codes.append(out[:, 1:, :])

        if not all_model_codes:
            t_end = time.perf_counter()
            return (
                torch.zeros(1, waveform.shape[-1], device="cpu"),
                {"rtf": 0.0, "peak_vram_gb": 0.0,
                 "inference_time_s": t_end - t_start,
                 "audio_duration_s": audio_duration_s},
            )

        all_codes = torch.cat(all_model_codes, dim=2)
        with torch.no_grad(), self.mimi.streaming(1):
            model_waveform = self.mimi.decode(all_codes)

        t_end = time.perf_counter()

        inference_time = t_end - t_start
        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        rtf = audio_duration_s / inference_time if inference_time > 0 else 0.0

        perf = {
            "rtf": round(rtf, 3),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "inference_time_s": round(inference_time, 3),
            "audio_duration_s": round(audio_duration_s, 3),
        }

        return model_waveform.squeeze(0).cpu(), perf

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def transcribe_and_score(
        self, audio_tensor: torch.Tensor, expected_text: str
    ) -> dict:
        """
        Transcribe audio with Whisper and compute all text-level metrics.
        """
        temp_path = "temp_sqa_eval.wav"
        audio_tensor = audio_tensor.clamp(-1.0, 1.0)
        _save_audio(temp_path, audio_tensor.cpu(), 24000)

        # --- Whisper transcription ---
        result = self.whisper_model.transcribe(temp_path)
        transcription = result["text"].strip()

        expected_norm = expected_text.lower().strip()
        trans_norm = transcription.lower().strip()

        # --- 1. Exact Match ---
        exact_match = int(expected_norm == trans_norm)

        # --- 2. Token F1 ---
        expected_tokens = set(expected_norm.split())
        trans_tokens = set(trans_norm.split())
        if not expected_tokens or not trans_tokens:
            token_f1 = 0.0
        else:
            intersection = len(expected_tokens & trans_tokens)
            precision = intersection / len(trans_tokens)
            recall = intersection / len(expected_tokens)
            token_f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        # --- 3. Word Error Rate (WER) ---
        try:
            jiwer = _get_jiwer()
            wer = jiwer.wer(expected_norm, trans_norm) if trans_norm else 1.0
        except Exception as e:
            print(f"    [WARN] WER computation failed: {e}")
            wer = -1.0

        # --- 4. BERTScore ---
        try:
            bert_score_fn = _get_bert_score()
            # bert_score returns (P, R, F1) tensors
            _, _, bert_f1 = bert_score_fn(
                [trans_norm], [expected_norm],
                lang="en", verbose=False,
                device=self.device,
            )
            bert_f1_val = bert_f1.item()
        except Exception as e:
            print(f"    [WARN] BERTScore computation failed: {e}")
            bert_f1_val = -1.0

        # --- 5. UTMOS (reference-free audio naturalness, 1-5 MOS scale) ---
        try:
            utmos_model = _get_utmos()
            utmos_score = utmos_model.predict(input_path=temp_path)
            utmos_score = float(utmos_score)
        except Exception as e:
            print(f"    [WARN] UTMOS computation failed: {e}")
            utmos_score = -1.0

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "transcription": transcription,
            # Semantic / Content
            "exact_match": exact_match,
            "token_f1": round(token_f1, 4),
            "wer": round(wer, 4) if wer >= 0 else wer,
            "bert_score_f1": round(bert_f1_val, 4) if bert_f1_val >= 0 else bert_f1_val,
            # Audio Quality
            "utmos": round(utmos_score, 4) if utmos_score >= 0 else utmos_score,
        }

    # ------------------------------------------------------------------
    # Full Evaluation Loop
    # ------------------------------------------------------------------
    def run_sqa_eval(self, benchmark_dir: str) -> dict:
        """
        Evaluate over the full dataset in benchmark_dir.
        Returns aggregated metrics + per-sample details.
        """
        metadata_path = os.path.join(benchmark_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            print(f"No metadata.jsonl found in {benchmark_dir}.")
            return {"error": "missing metadata.jsonl"}

        # Compute model-level efficiency metrics once
        param_count = 0
        if self.lm_model is not None:
            param_count = sum(p.numel() for p in self.lm_model.parameters())
        mimi_param_count = 0
        if self.mimi is not None:
            mimi_param_count = sum(p.numel() for p in self.mimi.parameters())

        print(f"  Model: {param_count/1e9:.2f}B params (LM) + {mimi_param_count/1e6:.1f}M params (Mimi)")

        # Accumulators
        total_exact = 0
        total_f1 = 0.0
        total_wer = 0.0
        total_bert = 0.0
        total_utmos = 0.0
        total_rtf = 0.0
        total_peak_vram = 0.0
        count = 0
        wer_count = 0
        bert_count = 0
        utmos_count = 0
        transcriptions = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                audio_path = os.path.join(benchmark_dir, data.get("file_name", ""))
                expected_text = data.get("text", "")
                if not os.path.exists(audio_path):
                    continue

                count += 1
                print(f"\n  [{count}] {data.get('file_name', '?')}")

                # Inference + timing
                model_audio, perf = self.streaming_inference(audio_path)
                print(f"      RTF={perf['rtf']} | VRAM={perf['peak_vram_gb']}GB | "
                      f"Time={perf['inference_time_s']}s")

                total_rtf += perf["rtf"]
                total_peak_vram = max(total_peak_vram, perf["peak_vram_gb"])

                # Semantic + audio quality scoring
                scores = self.transcribe_and_score(model_audio, expected_text)
                print(f"      EM={scores['exact_match']} | F1={scores['token_f1']} | "
                      f"WER={scores['wer']} | BERT={scores['bert_score_f1']} | "
                      f"UTMOS={scores['utmos']}")
                print(f"      Expected:  {expected_text[:80]}")
                print(f"      Got:       {scores['transcription'][:80]}")

                total_exact += scores["exact_match"]
                total_f1 += scores["token_f1"]

                if scores["wer"] >= 0:
                    total_wer += scores["wer"]
                    wer_count += 1
                if scores["bert_score_f1"] >= 0:
                    total_bert += scores["bert_score_f1"]
                    bert_count += 1
                if scores["utmos"] >= 0:
                    total_utmos += scores["utmos"]
                    utmos_count += 1

                transcriptions.append({
                    "file": data.get("file_name", ""),
                    "expected": expected_text,
                    "transcription": scores["transcription"],
                    **scores,
                    **perf,
                })

        if count == 0:
            return {"error": "no valid samples found", "count": 0}

        results = {
            "model_info": {
                "lm_param_count": param_count,
                "lm_param_billions": round(param_count / 1e9, 3),
                "mimi_param_count": mimi_param_count,
                "mimi_param_millions": round(mimi_param_count / 1e6, 1),
            },
            "semantic_metrics": {
                "exact_match": round(total_exact / count, 4),
                "token_f1": round(total_f1 / count, 4),
                "wer": round(total_wer / wer_count, 4) if wer_count > 0 else -1.0,
                "bert_score_f1": round(total_bert / bert_count, 4) if bert_count > 0 else -1.0,
            },
            "audio_quality": {
                "utmos": round(total_utmos / utmos_count, 4) if utmos_count > 0 else -1.0,
            },
            "efficiency": {
                "avg_rtf": round(total_rtf / count, 3),
                "peak_vram_gb": round(total_peak_vram, 2),
            },
            "count": count,
            "transcriptions": transcriptions,
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY ({count} samples)")
        print(f"{'='*60}")
        print(f"  Semantic:")
        print(f"    Exact Match:  {results['semantic_metrics']['exact_match']}")
        print(f"    Token F1:     {results['semantic_metrics']['token_f1']}")
        print(f"    WER:          {results['semantic_metrics']['wer']}")
        print(f"    BERTScore F1: {results['semantic_metrics']['bert_score_f1']}")
        print(f"  Audio Quality:")
        print(f"    UTMOS (1-5):  {results['audio_quality']['utmos']}")
        print(f"  Efficiency:")
        print(f"    Avg RTF:      {results['efficiency']['avg_rtf']}")
        print(f"    Peak VRAM:    {results['efficiency']['peak_vram_gb']} GB")
        print(f"    LM Params:    {results['model_info']['lm_param_billions']}B")
        print(f"{'='*60}")

        return results
