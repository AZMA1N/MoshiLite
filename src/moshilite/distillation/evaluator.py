"""Mimicking evaluator for offline KD — measures how well the student
reproduces the teacher's predictions on held-out self-play data.

Metrics computed (all masked over valid timesteps):
    text_token_acc        – argmax accuracy on text head
    audio_cb0_token_acc   – argmax accuracy on audio CB0 head
    text_top5_agree       – teacher token in student's top-5 (text)
    audio_cb0_top5_agree  – teacher token in student's top-5 (audio CB0)
    text_kl_div           – KL(teacher || student) on text logits (sparse top-K)
    audio_cb0_kl_div      – KL(teacher || student) on audio CB0 logits
    text_perplexity       – exp(CE) of student on teacher text tokens
    audio_cb0_perplexity  – exp(CE) of student on teacher audio CB0 tokens
    val_loss_l2           – composite distillation loss (same as training)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from typing import Optional

from .losses import DistillationLoss, SparseLogitKDLoss


class MimickingEvaluator:
    """Evaluate a student model's ability to mimic the teacher on
    held-out self-play validation data.

    Usage::

        evaluator = MimickingEvaluator(device="cuda")
        results = evaluator.evaluate(student_model, val_loader)
        # results is a dict with all 9 metrics
    """

    def __init__(
        self,
        alpha: float = 0.7,
        delta: float = 0.3,
        temperature: float = 3.0,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.device = device
        self.use_amp = use_amp
        self.criterion = DistillationLoss(
            alpha=alpha, delta=delta, temperature=temperature,
        )
        self.kd_loss_fn = SparseLogitKDLoss(temperature=temperature)

    def _build_codes(self, batch: dict) -> torch.Tensor:
        """Construct [B, 17, T] input codes from batch data.

        Same layout as StudentTrainer._build_student_input():
            codes[:, 0, :]    = text tokens (Inner Monologue)
            codes[:, 1:9, :]  = model audio CB0..7
            codes[:, 9:17, :] = user audio CB0..7
        """
        text = batch["text_tokens"].to(self.device)            # [B, T]
        audio = batch["audio_tokens"].to(self.device)           # [B, 8, T]
        user_audio = batch["user_audio_tokens"].to(self.device) # [B, 8, T]

        B, T = text.shape
        codes = torch.zeros(B, 17, T, dtype=torch.long, device=self.device)
        codes[:, 0, :] = text
        codes[:, 1:9, :] = audio
        codes[:, 9:17, :] = user_audio
        return codes

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
    ) -> dict:
        """Run full mimicking evaluation on the validation set.

        Args:
            model: The student LMModel (already on device).
            val_loader: DataLoader yielding collated self-play batches.

        Returns:
            Dict with all 9 evaluation metrics.
        """
        model.eval()
        amp_dtype = torch.float16 if self.use_amp else torch.float32

        # Accumulators
        total_text_correct = 0
        total_audio_correct = 0
        total_text_top5 = 0
        total_audio_top5 = 0
        total_text_ce = 0.0
        total_audio_ce = 0.0
        total_text_kl = 0.0
        total_audio_kl = 0.0
        total_val_loss = 0.0
        total_valid = 0       # total valid timesteps
        n_batches = 0

        for batch in val_loader:
            codes = self._build_codes(batch)
            mask = batch["mask"].to(self.device)       # [B, T]
            mask_f = mask.float()
            valid_count = mask_f.sum().item()

            if valid_count == 0:
                continue

            with autocast('cuda', dtype=amp_dtype, enabled=self.use_amp):
                lm_output = model(codes)

                # Student predictions — text head and audio CB0
                s_text = lm_output.text_logits[:, 0, :, :]   # [B, T, text_card]
                s_audio = lm_output.logits[:, 0, :, :]       # [B, T, audio_card]

            # Cast to float32 for metric computation
            s_text = s_text.float()
            s_audio = s_audio.float()

            # Teacher targets
            t_text_tokens = batch["text_tokens"].to(self.device)       # [B, T]
            t_audio_tokens = batch["audio_tokens"].to(self.device)     # [B, 8, T]
            t_audio_cb0 = t_audio_tokens[:, 0, :]                     # [B, T]
            t_text_vals = batch["text_logits_vals"].to(self.device)    # [B, T, K]
            t_text_idxs = batch["text_logits_idxs"].to(self.device)   # [B, T, K]
            t_audio_vals = batch["audio_cb0_logits_vals"].to(self.device)
            t_audio_idxs = batch["audio_cb0_logits_idxs"].to(self.device)

            # ── 1. Token Accuracy (argmax) ──
            text_pred = s_text.argmax(dim=-1)       # [B, T]
            audio_pred = s_audio.argmax(dim=-1)     # [B, T]
            total_text_correct += ((text_pred == t_text_tokens) * mask).sum().item()
            total_audio_correct += ((audio_pred == t_audio_cb0) * mask).sum().item()

            # ── 2. Top-5 Agreement ──
            text_top5 = s_text.topk(5, dim=-1).indices   # [B, T, 5]
            audio_top5 = s_audio.topk(5, dim=-1).indices  # [B, T, 5]
            # Check if teacher's token is in student's top-5
            text_in_top5 = (text_top5 == t_text_tokens.unsqueeze(-1)).any(dim=-1)
            audio_in_top5 = (audio_top5 == t_audio_cb0.unsqueeze(-1)).any(dim=-1)
            total_text_top5 += (text_in_top5 * mask).sum().item()
            total_audio_top5 += (audio_in_top5 * mask).sum().item()

            # ── 3. Cross-Entropy / Perplexity ──
            B, T, V_text = s_text.shape
            _, _, V_audio = s_audio.shape

            text_ce_per = F.cross_entropy(
                s_text.reshape(B * T, V_text),
                t_text_tokens.reshape(B * T),
                reduction="none",
            ).reshape(B, T)
            audio_ce_per = F.cross_entropy(
                s_audio.reshape(B * T, V_audio),
                t_audio_cb0.reshape(B * T),
                reduction="none",
            ).reshape(B, T)
            # Guard against inf/NaN from degenerate logits
            total_text_ce += (text_ce_per * mask_f).nan_to_num(0.0).sum().item()
            total_audio_ce += (audio_ce_per * mask_f).nan_to_num(0.0).sum().item()

            # ── 4. KL Divergence (sparse top-K) ──
            text_kl_val = self.kd_loss_fn(
                s_text, t_text_vals, t_text_idxs, mask,
            ).item()
            audio_kl_val = self.kd_loss_fn(
                s_audio, t_audio_vals, t_audio_idxs, mask,
            ).item()
            # Guard: replace NaN/inf with 0 for accumulation
            if math.isfinite(text_kl_val):
                total_text_kl += text_kl_val * valid_count
            if math.isfinite(audio_kl_val):
                total_audio_kl += audio_kl_val * valid_count

            # ── 5. Composite Val Loss (L2) ──
            losses = self.criterion(
                student_text_logits=s_text,
                student_audio_cb0_logits=s_audio,
                teacher_text_logits_vals=t_text_vals,
                teacher_text_logits_idxs=t_text_idxs,
                teacher_audio_cb0_logits_vals=t_audio_vals,
                teacher_audio_cb0_logits_idxs=t_audio_idxs,
                teacher_text_tokens=t_text_tokens,
                teacher_audio_tokens=t_audio_tokens,
                mask=mask,
            )
            val_loss_item = losses["total"].item()
            if math.isfinite(val_loss_item):
                total_val_loss += val_loss_item
            total_valid += valid_count
            n_batches += 1

        # ── Aggregate ──
        if total_valid == 0:
            return {k: 0.0 for k in [
                "text_token_acc", "audio_cb0_token_acc",
                "text_top5_agree", "audio_cb0_top5_agree",
                "text_kl_div", "audio_cb0_kl_div",
                "text_perplexity", "audio_cb0_perplexity",
                "val_loss_l2",
            ]}

        avg_text_ce = total_text_ce / total_valid
        avg_audio_ce = total_audio_ce / total_valid

        # Clamp CE for safe exp(); cap at e^20 ≈ 485M to avoid overflow
        safe_text_ppl = math.exp(min(avg_text_ce, 20.0)) if math.isfinite(avg_text_ce) else float('inf')
        safe_audio_ppl = math.exp(min(avg_audio_ce, 20.0)) if math.isfinite(avg_audio_ce) else float('inf')

        results = {
            "text_token_acc":       round(total_text_correct / total_valid, 4),
            "audio_cb0_token_acc":  round(total_audio_correct / total_valid, 4),
            "text_top5_agree":      round(total_text_top5 / total_valid, 4),
            "audio_cb0_top5_agree": round(total_audio_top5 / total_valid, 4),
            "text_kl_div":          round(total_text_kl / total_valid, 4),
            "audio_cb0_kl_div":     round(total_audio_kl / total_valid, 4),
            "text_perplexity":      round(safe_text_ppl, 2),
            "audio_cb0_perplexity": round(safe_audio_ppl, 2),
            "val_loss_l2":          round(total_val_loss / n_batches, 4),
        }

        return results

    @staticmethod
    def print_comparison(
        post_kd: dict,
        pre_kd: dict,
        post_label: str = "Post-KD",
        pre_label: str = "Pruned",
    ):
        """Print a formatted side-by-side comparison table.

        Args:
            post_kd: Results dict from evaluate() on post-KD model.
            pre_kd: Results dict from evaluate() on pruned (pre-KD) model.
            post_label: Column header for post-KD results.
            pre_label: Column header for pre-KD results.
        """
        # Higher is better for these; lower is better for the rest.
        higher_better = {
            "text_token_acc", "audio_cb0_token_acc",
            "text_top5_agree", "audio_cb0_top5_agree",
        }
        lower_better = {
            "text_kl_div", "audio_cb0_kl_div",
            "text_perplexity", "audio_cb0_perplexity",
            "val_loss_l2",
        }

        display_names = {
            "text_token_acc":       "Text Token Acc",
            "audio_cb0_token_acc":  "Audio CB0 Acc",
            "text_top5_agree":      "Text Top-5 Agree",
            "audio_cb0_top5_agree": "Audio Top-5 Agree",
            "text_kl_div":          "Text KL Div",
            "audio_cb0_kl_div":     "Audio KL Div",
            "text_perplexity":      "Text Perplexity",
            "audio_cb0_perplexity": "Audio Perplexity",
            "val_loss_l2":          "Val Loss (L2)",
        }

        print("\n" + "=" * 68)
        print(f"  {post_label} vs {pre_label} -- MIMICKING EVALUATION")
        print("=" * 68)
        header = f"{'Metric':<20} {post_label:>12} {pre_label:>12} {'Delta':>12}"
        print(header)
        print("-" * len(header))

        wins = 0
        total = 0
        for key in display_names:
            name = display_names[key]
            kd_val = post_kd.get(key, 0.0)
            pr_val = pre_kd.get(key, 0.0)
            diff = kd_val - pr_val

            if key in higher_better:
                arrow = "+" if diff > 0 else "-" if diff < 0 else "="
                if diff > 0:
                    wins += 1
            elif key in lower_better:
                arrow = "+" if diff < 0 else "-" if diff > 0 else "="
                if diff < 0:
                    wins += 1
            else:
                arrow = ""

            delta_str = f"{arrow} {diff:+.4f}"
            print(f"{name:<20} {kd_val:>12.4f} {pr_val:>12.4f} {delta_str:>12}")
            total += 1

        print("-" * len(header))
        print(f"  Post-KD wins on {wins}/{total} metrics.")
        if wins > total // 2:
            print("  SUCCESS: Distillation improved the student overall.")
        else:
            print("  WARNING: Distillation did not consistently improve metrics.")
            print("  Consider tuning alpha/delta/temperature or increasing data.")
        print("=" * 68)
