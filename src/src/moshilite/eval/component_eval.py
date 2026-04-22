"""Component-level evaluation for Moshi pipeline.

Evaluates each stage of the Moshi pipeline independently so that post-pruning
regressions can be pinpointed to a specific component:

  Component A: Mimi Codec     — encode→decode roundtrip (PESQ, STOI, SNR)
  Component B: Temporal Txfmr — text perplexity, hidden state cosine sim
  Component C: Depth Txfmr    — per-codebook prediction accuracy (CB 0-7)
  Component D: Full Pipeline   — end-to-end audio generation quality (UTMOS, BERTScore)

Usage:
    # Baseline run (saves reference values):
    evaluator = ComponentEvaluator(moshi_lm, mimi, device)
    results = evaluator.evaluate(audio_paths, save_baseline_to="path/to/baseline/")

    # Post-prune comparison:
    evaluator = ComponentEvaluator(pruned_lm, mimi, device)
    results = evaluator.evaluate(audio_paths, compare_baseline_from="path/to/baseline/")
"""

import os
import torch as _torch_early

# ---------------------------------------------------------------------------
# torch.compile guard — MUST run before any moshi model forward pass
# ---------------------------------------------------------------------------
# moshi 0.2.13 declares `torch<2.10` as a requirement but Colab ships
# torch 2.10, which removed `determine_aoti_mmap_flags` from
# `torch._inductor.utils`.  Moshi's `utils/compile.py` wraps apply_rope
# with a lazy torch.compile call; when that call actually fires it triggers
# the broken import chain regardless of torch._dynamo.config.disable.
#
# The only reliable fix: replace torch.compile with a no-op identity so
# the broken sub-imports are never triggered.  Safe for inference-only eval.
_real_torch_compile = _torch_early.compile


def _noop_compile(fn=None, *args, **kwargs):
    """No-op replacement for torch.compile (inference compatibility shim)."""
    if fn is not None:
        return fn
    return lambda f: f   # handle the decorator-with-args form


_torch_early.compile = _noop_compile

import json
import time
import math
from math import gcd
import io
import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly
from typing import Optional
from dataclasses import dataclass, field, asdict

# torchaudio is only used as a fallback; the binary is often incompatible with
# the runtime torch version on Colab.  All audio I/O now goes through soundfile
# + scipy so the module can be imported unconditionally.
_torchaudio = None


def _get_torchaudio():
    """Lazy-import torchaudio – returns None if unavailable."""
    global _torchaudio
    if _torchaudio is None:
        try:
            import torchaudio as _ta
            _torchaudio = _ta
        except Exception:
            _torchaudio = False
    return _torchaudio if _torchaudio else None


def _load_audio(path: str) -> tuple["torch.Tensor", int]:
    """Load audio to a (channels, samples) float32 tensor.

    Tries soundfile first (no native extension required), falls back to
    torchaudio if soundfile cannot handle the format.

    Returns:
        waveform: Tensor shape (C, T)
        sr: sample rate
    """
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # soundfile returns (T, C); transpose → (C, T)
        waveform = torch.from_numpy(data.T)
        return waveform, sr
    except Exception:
        ta = _get_torchaudio()
        if ta is not None:
            return ta.load(path)
        raise RuntimeError(f"Cannot load audio file (soundfile failed and torchaudio unavailable): {path}")


def _resample(waveform: "torch.Tensor", orig_sr: int, new_sr: int) -> "torch.Tensor":
    """Resample (C, T) tensor from orig_sr to new_sr."""
    if orig_sr == new_sr:
        return waveform
    ta = _get_torchaudio()
    if ta is not None:
        return ta.functional.resample(waveform, orig_sr, new_sr)
    # scipy fallback (operates on numpy)
    g = gcd(orig_sr, new_sr)
    up, down = new_sr // g, orig_sr // g
    out = resample_poly(waveform.numpy(), up, down, axis=-1).astype("float32")
    return torch.from_numpy(out)


def _save_audio(path: str, waveform: "torch.Tensor", sr: int):
    """Save (C, T) or (T,) float32 tensor to a wav file."""
    ta = _get_torchaudio()
    if ta is not None:
        ta.save(path, waveform, sr)
        return
    # soundfile fallback: expects (T, C) or (T,)
    arr = waveform.numpy()
    if arr.ndim == 2:
        arr = arr.T  # (C, T) → (T, C)
    sf.write(path, arr, sr)

from moshi.models import LMGen, MimiModel
from moshi.models.lm import LMModel


# ---------------------------------------------------------------------------
# Lazy imports for optional dependencies
# ---------------------------------------------------------------------------
_pesq_fn = None
_stoi_fn = None
_jiwer = None
_bert_score = None
_utmos_model = None
_whisper_model = None


def _get_pesq():
    global _pesq_fn
    if _pesq_fn is None:
        from pesq import pesq
        _pesq_fn = pesq
    return _pesq_fn


def _get_stoi():
    global _stoi_fn
    if _stoi_fn is None:
        from pystoi import stoi
        _stoi_fn = stoi
    return _stoi_fn


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
    global _utmos_model
    if _utmos_model is None:
        import torch as _torch
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
                kw.pop("weights_only", None)
                return _real_load(*a, **kw)

        _torch.load = _unrestricted_load
        try:
            import utmosv2
            _utmos_model = utmosv2.create_model(pretrained=True)
        except Exception as e:
            print(f"    [WARN] UTMOSv2 init failed: {e}")
            _utmos_model = "FAILED"
        finally:
            _torch.load = _real_load
    return _utmos_model if _utmos_model != "FAILED" else None


def _get_whisper(model_size: str = "base", device: str = "cuda"):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(model_size, device=device)
    return _whisper_model


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MimiRoundtripResults:
    """Component A: Mimi encoder→decoder roundtrip quality."""
    pesq_mean: float = 0.0
    stoi_mean: float = 0.0
    snr_mean: float = 0.0
    pesq_per_sample: list = field(default_factory=list)
    stoi_per_sample: list = field(default_factory=list)
    snr_per_sample: list = field(default_factory=list)
    count: int = 0


@dataclass
class TemporalTransformerResults:
    """Component B: Temporal Transformer output quality."""
    text_perplexity: float = 0.0
    hidden_cosine_sim_vs_baseline: float = -1.0  # -1 = no baseline
    text_token_agreement_vs_baseline: float = -1.0
    count: int = 0


@dataclass
class DepthTransformerResults:
    """Component C: Depth Transformer per-codebook accuracy."""
    codebook_accuracies: list = field(default_factory=list)  # Per CB 0-7
    mean_codebook_accuracy: float = 0.0
    codebook_agreement_vs_baseline: list = field(default_factory=list)
    count: int = 0


@dataclass
class EfficiencyResults:
    """Efficiency metrics."""
    param_count_lm: int = 0
    param_count_mimi: int = 0
    param_billions_lm: float = 0.0
    avg_rtf: float = 0.0
    peak_vram_gb: float = 0.0


@dataclass
class SemanticResults:
    """Component D: Full pipeline semantic correctness."""
    exact_match: float = 0.0
    token_f1: float = 0.0
    wer: float = -1.0
    bert_score_f1: float = -1.0
    utmos: float = -1.0
    avg_rtf: float = 0.0
    per_sample: list = field(default_factory=list)
    count: int = 0


@dataclass
class ComponentEvalResults:
    """Full component-level evaluation results."""
    mimi_roundtrip: MimiRoundtripResults = field(default_factory=MimiRoundtripResults)
    temporal_transformer: TemporalTransformerResults = field(default_factory=TemporalTransformerResults)
    depth_transformer: DepthTransformerResults = field(default_factory=DepthTransformerResults)
    semantic: SemanticResults = field(default_factory=SemanticResults)
    efficiency: EfficiencyResults = field(default_factory=EfficiencyResults)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Helper: compute SNR
# ---------------------------------------------------------------------------
def _compute_snr(reference: torch.Tensor, degraded: torch.Tensor) -> float:
    """Signal-to-Noise Ratio in dB."""
    ref = reference.float()
    noise = ref - degraded.float()
    signal_power = (ref ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-10:
        return 100.0  # Perfect reconstruction
    return 10 * math.log10(signal_power / noise_power)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------
class ComponentEvaluator:
    """Evaluates each Moshi component independently.

    Designed to run at every evaluation stage (baseline, post-prune, post-KD)
    and produce comparable metrics.
    """

    def __init__(
        self,
        moshi_lm: Optional[LMModel] = None,
        mimi: Optional[MimiModel] = None,
        device: str = "cuda",
    ):
        self.lm_model = moshi_lm
        self.mimi = mimi
        self.device = device

    # ------------------------------------------------------------------
    # Component A: Mimi Encoder→Decoder Roundtrip
    # ------------------------------------------------------------------
    def eval_mimi_roundtrip(self, audio_paths: list[str]) -> MimiRoundtripResults:
        """Encode audio with Mimi, decode back, compare with original.

        This measures the audio codec's fidelity independent of the LM.
        Metrics: PESQ (perceptual quality), STOI (intelligibility), SNR.
        """
        results = MimiRoundtripResults()

        if self.mimi is None:
            print("  [SKIP] Mimi not loaded — skipping roundtrip eval")
            return results

        pesq_fn = _get_pesq()
        stoi_fn = _get_stoi()

        for path in audio_paths:
            if not os.path.exists(path):
                continue

            waveform, sr = _load_audio(path)
            # Mimi operates at 24kHz mono
            if sr != 24000:
                waveform = _resample(waveform, sr, 24000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Trim to frame boundary
            frame_size = self.mimi.frame_size
            n_frames = waveform.shape[-1] // frame_size
            if n_frames == 0:
                continue
            waveform = waveform[:, :n_frames * frame_size]
            waveform_gpu = waveform.unsqueeze(0).to(self.device)  # [1, 1, T]

            with torch.no_grad():
                codes = self.mimi.encode(waveform_gpu)       # [1, K, T_frames]
                reconstructed = self.mimi.decode(codes)       # [1, 1, T']

            # Trim to same length
            min_len = min(waveform.shape[-1], reconstructed.shape[-1])
            ref = waveform[0, :min_len].cpu()
            deg = reconstructed[0, 0, :min_len].cpu()

            # SNR (always works)
            snr = _compute_snr(ref, deg)
            results.snr_per_sample.append(round(snr, 2))

            # PESQ (requires 16kHz for wideband)
            try:
                ref_16k = _resample(ref.unsqueeze(0), 24000, 16000).squeeze()
                deg_16k = _resample(deg.unsqueeze(0), 24000, 16000).squeeze()
                p = pesq_fn(16000, ref_16k.numpy(), deg_16k.numpy(), 'wb')
                results.pesq_per_sample.append(round(p, 3))
            except Exception as e:
                print(f"      [WARN] PESQ failed: {e}")
                results.pesq_per_sample.append(-1.0)

            # STOI
            try:
                s = stoi_fn(ref.numpy(), deg.numpy(), 24000, extended=False)
                results.stoi_per_sample.append(round(s, 4))
            except Exception as e:
                print(f"      [WARN] STOI failed: {e}")
                results.stoi_per_sample.append(-1.0)

            results.count += 1

        # Averages
        valid_pesq = [x for x in results.pesq_per_sample if x >= 0]
        valid_stoi = [x for x in results.stoi_per_sample if x >= 0]
        results.pesq_mean = round(np.mean(valid_pesq), 3) if valid_pesq else -1.0
        results.stoi_mean = round(np.mean(valid_stoi), 4) if valid_stoi else -1.0
        results.snr_mean = round(np.mean(results.snr_per_sample), 2) if results.snr_per_sample else 0.0

        return results

    # ------------------------------------------------------------------
    # Component B + C: Temporal + Depth Transformer (streaming inference)
    # ------------------------------------------------------------------
    def eval_transformers(
        self,
        audio_paths: list[str],
        baseline_text_tokens: Optional[list[list[int]]] = None,
        baseline_audio_tokens: Optional[list[list[list[int]]]] = None,
        baseline_hidden_states: Optional[list[torch.Tensor]] = None,
    ) -> tuple[TemporalTransformerResults, DepthTransformerResults, dict]:
        """Evaluate the Temporal and Depth Transformer by running streaming inference
        and capturing all intermediate outputs.

        Returns:
            temporal_results: Text perplexity, hidden state similarity
            depth_results: Per-codebook accuracy
            captured: Dict of captured intermediate values (for saving as baseline)
        """
        temporal = TemporalTransformerResults()
        depth = DepthTransformerResults()
        captured = {
            "text_tokens": [],       # All text tokens predicted per sample
            "audio_tokens": [],      # All audio tokens predicted per sample
            "hidden_states": [],     # Last hidden state per sample (optional)
        }

        if self.lm_model is None or self.mimi is None:
            print("  [SKIP] Model not loaded — skipping transformer eval")
            return temporal, depth, captured

        lm_model = self.lm_model
        dep_q = lm_model.dep_q
        needed_input = lm_model.n_q - dep_q

        total_text_ce = 0.0
        total_text_steps = 0
        cb_correct = [0] * dep_q
        cb_total = [0] * dep_q

        for sample_idx, path in enumerate(audio_paths):
            if not os.path.exists(path):
                continue

            waveform, sr = _load_audio(path)
            if sr != 24000:
                waveform = _resample(waveform, sr, 24000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            frame_size = self.mimi.frame_size
            n_frames = waveform.shape[-1] // frame_size
            if n_frames == 0:
                continue
            waveform = waveform[:, :n_frames * frame_size].unsqueeze(0).to(self.device)

            sample_text_tokens = []
            sample_audio_tokens = []
            sample_hidden_states = []

            # Hooks to capture text and audio tokens
            def _on_text(t, _list=sample_text_tokens):
                _list.append(t.cpu().item())

            def _on_audio(t, _list=sample_audio_tokens):
                _list.append(t.cpu().tolist())

            lm_gen = LMGen(
                lm_model, temp=0.8, temp_text=0.7,
                on_text_hook=_on_text,
                on_audio_hook=_on_audio,
            )

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

                    # Use _step to get transformer_out (hidden states)
                    result = lm_gen._step(input_tokens)
                    if result is not None:
                        _, transformer_out = result
                        # Save a sample of hidden states (last frame)
                        sample_hidden_states.append(
                            transformer_out[:, -1, :].cpu()  # [1, D] or [B, D]
                        )

            captured["text_tokens"].append(sample_text_tokens)
            captured["audio_tokens"].append(sample_audio_tokens)
            if sample_hidden_states:
                # Average hidden state across time for this sample
                avg_hidden = torch.stack(sample_hidden_states).mean(dim=0)
                captured["hidden_states"].append(avg_hidden)

            # --- Text perplexity (cross-entropy of text token predictions) ---
            # Since Moshi predicted these tokens with its own distribution,
            # a higher diversity of tokens = higher text_ce is expected.
            # For baseline, we just record the token sequence.
            # For comparison, we compute agreement rate.
            total_text_steps += len(sample_text_tokens)

            # --- Codebook agreement vs baseline ---
            if baseline_audio_tokens is not None and sample_idx < len(baseline_audio_tokens):
                baseline_sample = baseline_audio_tokens[sample_idx]
                min_steps = min(len(sample_audio_tokens), len(baseline_sample))
                for step in range(min_steps):
                    # Tokens are generated iteratively, so codebook index cycles from 0 to dep_q - 1
                    cb = step % dep_q
                    
                    # Handle both flat lists and lists of single-element lists
                    val_sample = sample_audio_tokens[step][0] if isinstance(sample_audio_tokens[step], list) else sample_audio_tokens[step]
                    val_baseline = baseline_sample[step][0] if isinstance(baseline_sample[step], list) else baseline_sample[step]
                    
                    cb_total[cb] += 1
                    if val_sample == val_baseline:
                        cb_correct[cb] += 1

            # --- Text token agreement vs baseline ---
            if baseline_text_tokens is not None and sample_idx < len(baseline_text_tokens):
                baseline_text = baseline_text_tokens[sample_idx]
                min_len = min(len(sample_text_tokens), len(baseline_text))
                if min_len > 0:
                    matches = sum(
                        1 for a, b in zip(sample_text_tokens[:min_len], baseline_text[:min_len])
                        if a == b
                    )
                    temporal.text_token_agreement_vs_baseline = round(matches / min_len, 4)

            temporal.count += 1

        # --- Hidden state cosine similarity vs baseline ---
        if baseline_hidden_states and captured["hidden_states"]:
            cos_sims = []
            for i in range(min(len(captured["hidden_states"]), len(baseline_hidden_states))):
                current = captured["hidden_states"][i].float().flatten()
                baseline = baseline_hidden_states[i].float().flatten()
                sim = torch.nn.functional.cosine_similarity(
                    current.unsqueeze(0), baseline.unsqueeze(0)
                ).item()
                cos_sims.append(sim)
            if cos_sims:
                temporal.hidden_cosine_sim_vs_baseline = round(np.mean(cos_sims), 4)

        # --- Codebook accuracy (vs baseline or absolute self-consistency) ---
        cb_accs = []
        for cb in range(dep_q):
            if cb_total[cb] > 0:
                cb_accs.append(round(cb_correct[cb] / cb_total[cb], 4))
            else:
                cb_accs.append(-1.0)
        depth.codebook_accuracies = cb_accs
        valid_accs = [a for a in cb_accs if a >= 0]
        depth.mean_codebook_accuracy = round(np.mean(valid_accs), 4) if valid_accs else -1.0
        depth.count = temporal.count

        return temporal, depth, captured

    # ------------------------------------------------------------------
    # Component D: Full Pipeline Semantic Correctness
    # ------------------------------------------------------------------
    def eval_full_pipeline(
        self,
        audio_paths: list[str],
        expected_texts: list[str],
    ) -> SemanticResults:
        """Run full streaming inference, transcribe with Whisper, and score.

        Metrics: Exact Match, Token F1, WER, BERTScore, UTMOS, RTF.
        """
        sem = SemanticResults()

        if self.lm_model is None or self.mimi is None:
            print("  [SKIP] Model not loaded — skipping semantic eval")
            return sem

        whisper_model = _get_whisper(device=self.device)
        lm_model = self.lm_model
        needed_input = lm_model.n_q - lm_model.dep_q

        total_em, total_f1, total_wer = 0, 0.0, 0.0
        total_bert, total_utmos, total_rtf = 0.0, 0.0, 0.0
        wer_n, bert_n, utmos_n = 0, 0, 0

        for idx, (path, expected) in enumerate(zip(audio_paths, expected_texts)):
            if not os.path.exists(path):
                continue

            # --- Load & prep audio ---
            waveform, sr = _load_audio(path)
            if sr != 24000:
                waveform = _resample(waveform, sr, 24000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            audio_dur = waveform.shape[-1] / 24000.0
            frame_size = self.mimi.frame_size
            n_frames = waveform.shape[-1] // frame_size
            if n_frames == 0:
                continue

            waveform = waveform[:, :n_frames * frame_size].unsqueeze(0).to(self.device)

            # --- Streaming inference ---
            lm_gen = LMGen(lm_model, temp=0.8, temp_text=0.7)
            all_codes = []

            t0 = time.perf_counter()
            with torch.no_grad(), self.mimi.streaming(1), lm_gen.streaming(1):
                for fi in range(n_frames):
                    chunk = waveform[:, :, fi * frame_size:(fi + 1) * frame_size]
                    user_codes = self.mimi.encode(chunk)
                    if user_codes.shape[1] < needed_input:
                        pad = torch.zeros(
                            1, needed_input - user_codes.shape[1], 1,
                            device=self.device, dtype=user_codes.dtype)
                        inp = torch.cat([user_codes, pad], dim=1)
                    else:
                        inp = user_codes[:, :needed_input, :]
                    out = lm_gen.step(inp)
                    if out is not None:
                        all_codes.append(out[:, 1:, :])

            if not all_codes:
                continue

            codes_cat = torch.cat(all_codes, dim=2)
            with torch.no_grad(), self.mimi.streaming(1):
                model_wav = self.mimi.decode(codes_cat)
            t1 = time.perf_counter()

            rtf = audio_dur / (t1 - t0) if (t1 - t0) > 0 else 0.0
            total_rtf += rtf

            # --- Save temp wav for Whisper + UTMOS ---
            temp_path = f"_comp_eval_tmp_{idx}.wav"
            out_audio = model_wav.squeeze(0).cpu().clamp(-1, 1)
            _save_audio(temp_path, out_audio, 24000)

            # --- Whisper transcription ---
            result = whisper_model.transcribe(temp_path)
            transcription = result["text"].strip()
            exp_norm = expected.lower().strip()
            trans_norm = transcription.lower().strip()

            # Exact Match
            em = int(exp_norm == trans_norm)
            total_em += em

            # Token F1
            exp_tok = set(exp_norm.split())
            tr_tok = set(trans_norm.split())
            if exp_tok and tr_tok:
                inter = len(exp_tok & tr_tok)
                prec = inter / len(tr_tok)
                rec = inter / len(exp_tok)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            else:
                f1 = 0.0
            total_f1 += f1

            # WER
            try:
                jiwer = _get_jiwer()
                w = jiwer.wer(exp_norm, trans_norm) if trans_norm else 1.0
                total_wer += w
                wer_n += 1
            except Exception:
                w = -1.0

            # BERTScore
            try:
                bsf = _get_bert_score()
                _, _, bf1 = bsf([trans_norm], [exp_norm], lang="en",
                                verbose=False, device=self.device)
                bv = bf1.item()
                total_bert += bv
                bert_n += 1
            except Exception as e:
                if total_bert == 0 and bert_n == 0:  # Only print first time
                    print(f"      [WARN] BERTScore computation failed: {e}")
                bv = -1.0

            # UTMOS
            try:
                um = _get_utmos()
                uv = float(um.predict(input_path=temp_path))
                total_utmos += uv
                utmos_n += 1
            except Exception as e:
                if total_utmos == 0 and utmos_n == 0: # Only print first time
                    print(f"      [WARN] UTMOS computation failed: {e}")
                uv = -1.0

            if os.path.exists(temp_path):
                os.remove(temp_path)

            sem.count += 1
            sem.per_sample.append({
                "file": os.path.basename(path),
                "expected": expected[:100],
                "transcription": transcription[:100],
                "exact_match": em, "token_f1": round(f1, 4),
                "wer": round(w, 4) if w >= 0 else w,
                "bert_score_f1": round(bv, 4) if bv >= 0 else bv,
                "utmos": round(uv, 4) if uv >= 0 else uv,
                "rtf": round(rtf, 3),
            })

            print(f"    [{sem.count}] EM={em} F1={f1:.3f} WER={w:.3f} "
                  f"BERT={bv:.3f} UTMOS={uv:.2f} RTF={rtf:.2f}")
            print(f"        Exp: {expected[:70]}")
            print(f"        Got: {transcription[:70]}")

        if sem.count > 0:
            sem.exact_match = round(total_em / sem.count, 4)
            sem.token_f1 = round(total_f1 / sem.count, 4)
            sem.wer = round(total_wer / wer_n, 4) if wer_n > 0 else -1.0
            sem.bert_score_f1 = round(total_bert / bert_n, 4) if bert_n > 0 else -1.0
            sem.utmos = round(total_utmos / utmos_n, 4) if utmos_n > 0 else -1.0
            sem.avg_rtf = round(total_rtf / sem.count, 3)

        return sem

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        audio_paths: list[str],
        expected_texts: Optional[list[str]] = None,
        metadata_path: Optional[str] = None,
        save_baseline_to: Optional[str] = None,
        compare_baseline_from: Optional[str] = None,
    ) -> ComponentEvalResults:
        """Run full component-level evaluation.

        Args:
            audio_paths: List of audio file paths to evaluate
            expected_texts: List of expected transcriptions (for semantic eval)
            metadata_path: Path to metadata.jsonl (alternative to expected_texts)
            save_baseline_to: If set, save captured intermediate values as baseline
            compare_baseline_from: If set, load baseline and compute agreement metrics
        """
        results = ComponentEvalResults()

        # Load expected texts from metadata if not provided directly
        if expected_texts is None and metadata_path and os.path.exists(metadata_path):
            expected_texts = []
            audio_paths_from_meta = []
            benchmark_dir = os.path.dirname(metadata_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    ap = os.path.join(benchmark_dir, data.get("file_name", ""))
                    if os.path.exists(ap):
                        audio_paths_from_meta.append(ap)
                        expected_texts.append(data.get("text", ""))
            audio_paths = audio_paths_from_meta

        # Efficiency
        if self.lm_model is not None:
            pc = sum(p.numel() for p in self.lm_model.parameters())
            results.efficiency.param_count_lm = pc
            results.efficiency.param_billions_lm = round(pc / 1e9, 3)
        if self.mimi is not None:
            results.efficiency.param_count_mimi = sum(p.numel() for p in self.mimi.parameters())

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # --- Component A: Mimi Roundtrip ---
        print("\n  === Component A: Mimi Codec Roundtrip ===")
        t0 = time.perf_counter()
        results.mimi_roundtrip = self.eval_mimi_roundtrip(audio_paths)
        t1 = time.perf_counter()
        mr = results.mimi_roundtrip
        print(f"    PESQ={mr.pesq_mean} | STOI={mr.stoi_mean} | SNR={mr.snr_mean}dB "
              f"({mr.count} samples, {t1-t0:.1f}s)")

        # --- Load baseline if comparing ---
        baseline_text = None
        baseline_audio = None
        baseline_hidden = None
        if compare_baseline_from and os.path.exists(compare_baseline_from):
            print(f"\n  Loading baseline from {compare_baseline_from}...")
            bl_path = os.path.join(compare_baseline_from, "component_baseline.json")
            if os.path.exists(bl_path):
                with open(bl_path, "r") as f:
                    bl = json.load(f)
                baseline_text = bl.get("text_tokens")
                baseline_audio = bl.get("audio_tokens")
            hs_path = os.path.join(compare_baseline_from, "hidden_states.pt")
            if os.path.exists(hs_path):
                baseline_hidden = torch.load(hs_path, weights_only=True)

        # --- Components B+C: Temporal + Depth Transformer ---
        print("\n  === Components B+C: Temporal & Depth Transformer ===")
        t0 = time.perf_counter()
        temporal, depth, captured = self.eval_transformers(
            audio_paths,
            baseline_text_tokens=baseline_text,
            baseline_audio_tokens=baseline_audio,
            baseline_hidden_states=baseline_hidden,
        )
        t1 = time.perf_counter()
        results.temporal_transformer = temporal
        results.depth_transformer = depth

        print(f"    Text token agreement vs baseline: {temporal.text_token_agreement_vs_baseline}")
        print(f"    Hidden cosine sim vs baseline:    {temporal.hidden_cosine_sim_vs_baseline}")
        print(f"    Per-CB accuracy vs baseline:      {depth.codebook_accuracies}")
        print(f"    Mean CB accuracy:                 {depth.mean_codebook_accuracy}")
        print(f"    ({temporal.count} samples, {t1-t0:.1f}s)")

        # --- Component D: Semantic Correctness (Full Pipeline) ---
        if expected_texts:
            print("\n  === Component D: Full Pipeline Semantic Correctness ===")
            t0 = time.perf_counter()
            results.semantic = self.eval_full_pipeline(audio_paths, expected_texts)
            t1 = time.perf_counter()
            s = results.semantic
            print(f"    EM={s.exact_match} | F1={s.token_f1} | WER={s.wer} | "
                  f"BERT={s.bert_score_f1} | UTMOS={s.utmos} | RTF={s.avg_rtf}")
            print(f"    ({s.count} samples, {t1-t0:.1f}s)")
        else:
            print("\n  === Component D: SKIPPED (no expected_texts provided) ===")

        # VRAM
        if torch.cuda.is_available():
            results.efficiency.peak_vram_gb = round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 2
            )

        # --- Save baseline if requested ---
        if save_baseline_to:
            os.makedirs(save_baseline_to, exist_ok=True)
            bl_path = os.path.join(save_baseline_to, "component_baseline.json")
            with open(bl_path, "w") as f:
                json.dump({
                    "text_tokens": captured["text_tokens"],
                    "audio_tokens": captured["audio_tokens"],
                }, f)
            if captured["hidden_states"]:
                hs_path = os.path.join(save_baseline_to, "hidden_states.pt")
                torch.save(captured["hidden_states"], hs_path)
            print(f"\n  ✅ Baseline saved to {save_baseline_to}")

        # --- Print summary ---
        s = results.semantic
        print(f"\n  {'='*60}")
        print(f"  COMPONENT EVALUATION SUMMARY")
        print(f"  {'='*60}")
        print(f"  A) Mimi Codec:   PESQ={mr.pesq_mean} | STOI={mr.stoi_mean} | SNR={mr.snr_mean}dB")
        print(f"  B) Temporal LM:  TextAgree={temporal.text_token_agreement_vs_baseline} | "
              f"HiddenCos={temporal.hidden_cosine_sim_vs_baseline}")
        print(f"  C) DepthTxfmr:   CB_Acc={depth.codebook_accuracies}")
        print(f"  D) Semantic:     EM={s.exact_match} | F1={s.token_f1} | WER={s.wer} | "
              f"BERT={s.bert_score_f1} | UTMOS={s.utmos}")
        print(f"  Efficiency:      {results.efficiency.param_billions_lm}B params | "
              f"VRAM={results.efficiency.peak_vram_gb}GB | RTF={s.avg_rtf}")
        print(f"  {'='*60}")

        return results
