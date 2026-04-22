"""KD training loop for pruned student model.

Handles:
  - Teacher-forced forward pass through student
  - L2 distillation loss (logit KD + hard label)
  - FP16 mixed precision + gradient checkpointing
  - Local-only checkpointing (fast NVMe writes, no GDrive I/O)
  - Best weights cached in memory for instant restore
  - Validation loss tracking
  - W&B logging (optional)

Disk strategy:
  - Checkpoints saved to local NVMe only (~13 GB each, fast writes)
  - Only 'best' and 'latest' kept locally (no step-specific or final)
  - GDrive receives only training_summary.json (~1 KB)
  - Heavy model export is done in the notebook Cell 7
"""

import json
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ..distillation.losses import DistillationLoss
from ..data.dataset import get_self_play_dataloader


class StudentTrainer:
    """Offline KD trainer for the pruned student model.

    The student receives teacher-forced token sequences and is trained to
    match the teacher's output distributions (logits + hard labels).

    Architecture integration:
        The student is a pruned LMModel (same architecture as teacher, fewer layers).
        Its forward() method takes codes [B, K=17, T] and returns LMOutput with:
            - logits: [B, dep_q=8, T, card=2048] (audio codebook logits)
            - text_logits: [B, 1, T, text_card=32000]

    Usage::

        trainer = StudentTrainer(
            student_model=student,
            data_dir="/content/staged/self_play",
            checkpoint_dir="/content/drive/MyDrive/moshilite/checkpoints/kd_poc",
            device="cuda",
        )
        trainer.train(num_epochs=25)
    """

    def __init__(
        self,
        student_model: nn.Module,
        data_dir: str | Path,
        checkpoint_dir: str | Path,
        # Training hyperparameters
        lr: float = 1e-4,
        batch_size: int = 4,
        gradient_accumulation: int = 4,
        alpha: float = 0.7,         # logit KD weight
        delta: float = 0.3,         # hard label weight
        temperature: float = 3.0,   # KD temperature
        loss_config: str | None = None,  # "L1"-"L5" (overrides alpha/delta)
        max_steps_per_sample: int | None = None,
        val_fraction: float = 0.1,
        # Checkpointing
        checkpoint_every: int = 200,  # steps
        val_every: int = 100,         # steps
        # Hardware
        device: str = "cuda",
        use_amp: bool = True,
        # W&B
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ):
        self.student = student_model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Local NVMe staging dir for fast checkpoint writes (avoid slow GDrive FUSE)
        self._local_ckpt_dir = Path("/content/ckpt_staging")
        self._local_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Detect model dtype for correct autocast/scaler behavior
        model_dtype = next(student_model.parameters()).dtype
        self.amp_dtype = model_dtype if model_dtype in (torch.float16, torch.bfloat16) else torch.float16
        # GradScaler is only for float16 — bfloat16 has float32's exponent range
        self.use_scaler = use_amp and (self.amp_dtype == torch.float16)

        # Data
        self.train_loader = get_self_play_dataloader(
            data_dir, split="train", batch_size=batch_size,
            max_steps=max_steps_per_sample, val_fraction=val_fraction,
        )
        self.val_loader = get_self_play_dataloader(
            data_dir, split="val", batch_size=batch_size,
            max_steps=max_steps_per_sample, val_fraction=val_fraction,
        )

        # Loss
        if loss_config:
            self.criterion = DistillationLoss(
                loss_config=loss_config, temperature=temperature,
            )
            print(f"📋 Loss config: {loss_config} "
                  f"(α={self.criterion.alpha}, β={self.criterion.beta}, "
                  f"γ={self.criterion.gamma}, δ={self.criterion.delta})")
        else:
            self.criterion = DistillationLoss(
                alpha=alpha, delta=delta, temperature=temperature,
            )

        # Collect trainable parameters: student + any loss module params
        # (e.g. hidden state projection layers for L3/L5)
        trainable_params = list(
            filter(lambda p: p.requires_grad, self.student.parameters())
        )
        for loss_param in self.criterion.parameters():
            if loss_param.requires_grad:
                trainable_params.append(loss_param)

        # Optimizer (8-bit AdamW if bitsandbytes available, else standard)
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                trainable_params, lr=lr, weight_decay=0.01,
            )
            print("✅ Using 8-bit AdamW (bitsandbytes)")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                trainable_params, lr=lr, weight_decay=0.01,
            )
            print("⚠️  bitsandbytes not available, using standard AdamW")

        self.scaler = GradScaler("cuda", enabled=self.use_scaler)
        print(f"🔧 AMP dtype: {self.amp_dtype}, GradScaler: {'ON' if self.use_scaler else 'OFF'}")
        self.gradient_accumulation = gradient_accumulation
        self.checkpoint_every = checkpoint_every
        self.val_every = val_every

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        # W&B
        self.wandb_run = None
        if wandb_project:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project, name=wandb_run_name,
                    config={
                        "lr": lr, "batch_size": batch_size,
                        "gradient_accumulation": gradient_accumulation,
                        "alpha": alpha, "delta": delta, "temperature": temperature,
                    },
                )
            except Exception as e:
                print(f"⚠️  W&B init failed: {e}")

        # Try to resume from latest checkpoint
        self._try_resume()

    def _build_student_input(self, batch: dict) -> torch.Tensor:
        """Construct the student's input codes tensor from batch data.

        The student's LMModel.forward() expects codes [B, K=17, T] where:
            codes[:, 0, :]  = text tokens (Inner Monologue)
            codes[:, 1:9, :] = model audio CB0..7 (generated by Depformer)
            codes[:, 9:17, :] = user audio CB0..7 (Channel A input)

        In teacher-forced training, we use the teacher's token sequences.
        """
        text = batch["text_tokens"].to(self.device)           # [B, T]
        audio = batch["audio_tokens"].to(self.device)          # [B, 8, T]
        user_audio = batch["user_audio_tokens"].to(self.device)  # [B, 8, T]

        B, T = text.shape

        # codes shape: [B, 17, T]
        codes = torch.zeros(B, 17, T, dtype=torch.long, device=self.device)
        codes[:, 0, :] = text             # text channel
        codes[:, 1:9, :] = audio          # model audio channels
        codes[:, 9:17, :] = user_audio    # user audio channels

        return codes

    def _train_step(self, batch: dict) -> dict:
        """Run one forward + backward pass on a batch."""
        codes = self._build_student_input(batch)
        mask = batch["mask"].to(self.device)  # [B, T]

        with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            # Student forward pass (teacher-forced)
            lm_output = self.student(codes)

            # Extract student logits
            # text_logits: [B, 1, T, text_card] → [B, T, text_card]
            student_text_logits = lm_output.text_logits[:, 0, :, :]
            # logits: [B, dep_q=8, T, card] — take CB0
            student_audio_cb0_logits = lm_output.logits[:, 0, :, :]  # [B, T, card]

            # CB1-7 logits for codebook CE (L4/L5)
            student_audio_cb1_7 = None
            if self.criterion.gamma > 0:
                student_audio_cb1_7 = lm_output.logits[:, 1:, :, :]  # [B, 7, T, card]

            # Compute loss
            losses = self.criterion(
                student_text_logits=student_text_logits,
                student_audio_cb0_logits=student_audio_cb0_logits,
                teacher_text_logits_vals=batch["text_logits_vals"].to(self.device),
                teacher_text_logits_idxs=batch["text_logits_idxs"].to(self.device),
                teacher_audio_cb0_logits_vals=batch["audio_cb0_logits_vals"].to(self.device),
                teacher_audio_cb0_logits_idxs=batch["audio_cb0_logits_idxs"].to(self.device),
                teacher_text_tokens=batch["text_tokens"].to(self.device),
                teacher_audio_tokens=batch["audio_tokens"].to(self.device),
                mask=mask,
                student_audio_cb1_7_logits=student_audio_cb1_7,
            )

            loss = losses["total"] / self.gradient_accumulation

        if self.use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()}

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation and return average loss."""
        self.student.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            codes = self._build_student_input(batch)
            mask = batch["mask"].to(self.device)

            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                lm_output = self.student(codes)
                student_text_logits = lm_output.text_logits[:, 0, :, :]
                student_audio_cb0_logits = lm_output.logits[:, 0, :, :]

                # CB1-7 logits for codebook CE (L4/L5)
                student_audio_cb1_7 = None
                if self.criterion.gamma > 0:
                    student_audio_cb1_7 = lm_output.logits[:, 1:, :, :]  # [B, 7, T, card]

                losses = self.criterion(
                    student_text_logits=student_text_logits,
                    student_audio_cb0_logits=student_audio_cb0_logits,
                    teacher_text_logits_vals=batch["text_logits_vals"].to(self.device),
                    teacher_text_logits_idxs=batch["text_logits_idxs"].to(self.device),
                    teacher_audio_cb0_logits_vals=batch["audio_cb0_logits_vals"].to(self.device),
                    teacher_audio_cb0_logits_idxs=batch["audio_cb0_logits_idxs"].to(self.device),
                    teacher_text_tokens=batch["text_tokens"].to(self.device),
                    teacher_audio_tokens=batch["audio_tokens"].to(self.device),
                    mask=mask,
                    student_audio_cb1_7_logits=student_audio_cb1_7,
                )
                total_loss += losses["total"].item()
                n_batches += 1

        self.student.train()
        return total_loss / max(1, n_batches)

    def _save_checkpoint(self, tag: str = "latest"):
        """Save training state to local NVMe only (no GDrive writes).

        Only 'best' and 'latest' are saved. Each is ~13 GB.
        Local peak: ~26 GB (well within Colab's ~78 GB limit).
        """
        ckpt_data = {
            "global_step": self.global_step,
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses[-100:],  # keep last 100
            "val_losses": self.val_losses,
        }

        local_path = self._local_ckpt_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt_data, str(local_path))
        print(f"💾 Checkpoint saved: {local_path.name} (step {self.global_step})")

    def _cleanup_local_staging(self):
        """Remove all local checkpoint files to free disk space."""
        import shutil
        if self._local_ckpt_dir.exists():
            shutil.rmtree(str(self._local_ckpt_dir), ignore_errors=True)
            print(f"🧹 Cleaned up local staging dir ({self._local_ckpt_dir})")

    def _try_resume(self):
        """Try to resume from local checkpoint (survives runtime restarts within same session)."""
        local_path = self._local_ckpt_dir / "checkpoint_latest.pt"
        if not local_path.exists():
            return

        print(f"🔄 Resuming from local checkpoint...")
        try:
            ckpt = torch.load(str(local_path), map_location=self.device)
            self.student.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.global_step = ckpt["global_step"]
            self.best_val_loss = ckpt["best_val_loss"]
            self.train_losses = ckpt.get("train_losses", [])
            self.val_losses = ckpt.get("val_losses", [])
            print(f"✅ Resumed at step {self.global_step}, "
                  f"best_val_loss={self.best_val_loss:.4f}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint (file may be corrupt): {e}")
            print("   Starting training from scratch.")
            self.global_step = 0
            self.best_val_loss = float("inf")

    def train(self, num_epochs: int = 25):
        """Run the full training loop.

        Args:
            num_epochs: Number of passes over the training data.
        """
        self.student.train()
        total_steps_per_epoch = len(self.train_loader) // self.gradient_accumulation
        total_steps = total_steps_per_epoch * num_epochs
        print(f"\n🚀 Training: {num_epochs} epochs × {total_steps_per_epoch} steps/epoch "
              f"= {total_steps} total steps")

        t_start = time.time()
        accum_losses = []

        for epoch in range(num_epochs):
            epoch_pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=True,
            )

            for batch_idx, batch in enumerate(epoch_pbar):
                losses = self._train_step(batch)
                accum_losses.append(losses["total"])

                # Gradient accumulation step
                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    if self.use_scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    avg_loss = sum(accum_losses) / len(accum_losses)
                    self.train_losses.append(avg_loss)
                    accum_losses = []

                    epoch_pbar.set_postfix({
                        "step": self.global_step,
                        "loss": f"{avg_loss:.4f}",
                    })

                    # W&B logging
                    if self.wandb_run:
                        import wandb
                        log_data = {
                            "train/loss": avg_loss,
                            "train/logit_text": losses.get("logit_text", 0),
                            "train/logit_audio": losses.get("logit_audio", 0),
                            "train/hard_text": losses.get("hard_text", 0),
                            "train/hard_audio": losses.get("hard_audio", 0),
                            "train/step": self.global_step,
                            "train/epoch": epoch + 1,
                        }
                        if "codebook_ce" in losses:
                            log_data["train/codebook_ce"] = losses["codebook_ce"]
                        if "hidden_state" in losses:
                            log_data["train/hidden_state"] = losses["hidden_state"]
                        wandb.log(log_data)

                    # Validation
                    if self.global_step % self.val_every == 0:
                        val_loss = self._validate()
                        self.val_losses.append({
                            "step": self.global_step,
                            "loss": val_loss,
                        })
                        print(f"\n📊 Val loss @ step {self.global_step}: {val_loss:.4f}")

                        if self.wandb_run:
                            import wandb
                            wandb.log({
                                "val/loss": val_loss,
                                "val/step": self.global_step,
                            })

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
                            self._save_checkpoint("best")

                    # Periodic checkpoint (local only)
                    if self.global_step % self.checkpoint_every == 0:
                        self._save_checkpoint("latest")

        # Save final 'latest' locally
        try:
            self._save_checkpoint("latest")
        except Exception as e:
            print(f"⚠️  Final checkpoint save failed (non-critical): {e}")

        # Restore best weights so the caller gets the best model, not the last epoch
        if hasattr(self, '_best_state'):
            self.student.load_state_dict(self._best_state)
            del self._best_state  # free memory
            print("✅ Restored best checkpoint weights to student model")

        # Clean up local staging to free ~26 GB disk space
        self._cleanup_local_staging()

        elapsed = time.time() - t_start
        print(f"\n✅ Training complete: {self.global_step} steps, "
              f"{elapsed / 60:.1f} min, best val loss: {self.best_val_loss:.4f}")

        # Save training summary
        summary = {
            "total_steps": self.global_step,
            "num_epochs": num_epochs,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "elapsed_seconds": elapsed,
            "val_losses": self.val_losses,
        }
        summary_path = self.checkpoint_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📝 Summary saved to {summary_path}")

        return summary
