"""KD loss functions for offline distillation from self-play teacher targets.

Loss configurations (from project plan v10):
    L1: Pure logit KD (KL-divergence only)
    L2: Logit KD + hard label CE  ← default
    L3: Logit KD + hidden state MSE + hard label CE  (requires hidden state data)
    L4: Logit KD + codebook CE + hard label CE
    L5: Full (all four losses)  (requires hidden state data)

Usage:
    # Select by config string (weights auto-set per plan):
    loss = DistillationLoss(loss_config="L4", temperature=3.0)

    # Or manually set weights (backward compatible):
    loss = DistillationLoss(alpha=0.7, delta=0.3, temperature=3.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Pre-defined weight configurations from project plan v10 §Stage 4-ablation
LOSS_CONFIGS = {
    "L1": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
    "L2": {"alpha": 0.7, "beta": 0.0, "gamma": 0.0, "delta": 0.3},
    "L3": {"alpha": 0.5, "beta": 0.3, "gamma": 0.0, "delta": 0.2},
    "L4": {"alpha": 0.5, "beta": 0.0, "gamma": 0.3, "delta": 0.2},
    "L5": {"alpha": 0.4, "beta": 0.2, "gamma": 0.2, "delta": 0.2},
}


class SparseLogitKDLoss(nn.Module):
    """KL-divergence loss using sparse top-K teacher logits.

    The teacher logits are stored as sparse top-K (values, indices) to save
    storage. This loss reconstructs a partial probability distribution over
    the top-K tokens and computes KL-divergence against the student's
    corresponding logits.

    This works for both text logits (card=32000) and audio CB0 logits (card=2048).
    """

    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,     # [B, T, vocab_size]
        teacher_vals: torch.Tensor,        # [B, T, K] — top-K logit values
        teacher_idxs: torch.Tensor,        # [B, T, K] — top-K logit indices
        mask: torch.Tensor,               # [B, T] — valid positions
    ) -> torch.Tensor:
        """Compute sparse KL-divergence loss.

        Args:
            student_logits: Full student logits over vocabulary.
            teacher_vals: Teacher's top-K logit values (pre-softmax).
            teacher_idxs: Teacher's top-K logit indices.
            mask: Boolean mask of valid timesteps.

        Returns:
            Scalar loss (averaged over valid positions).
        """
        T_temp = self.temperature

        # Teacher: softmax over top-K only
        teacher_probs = F.softmax(teacher_vals / T_temp, dim=-1)  # [B, T, K]

        # Student: gather logits at teacher's top-K indices
        # student_logits: [B, T, vocab], teacher_idxs: [B, T, K]
        student_at_teacher = student_logits.gather(
            dim=-1, index=teacher_idxs
        )  # [B, T, K]

        # Student: log-softmax over the same K positions
        student_log_probs = F.log_softmax(student_at_teacher / T_temp, dim=-1)

        # KL(teacher || student) at each position
        kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
            log_target=False,
        ).sum(dim=-1)  # [B, T]

        # Scale by T² (standard KD scaling)
        kl = kl * (T_temp ** 2)

        # Mask and average
        kl = kl * mask.float()
        return kl.sum() / mask.float().sum().clamp(min=1.0)


class HardLabelLoss(nn.Module):
    """Cross-entropy loss against teacher's hard (argmax) token predictions.

    Works for both text tokens and audio codebook tokens.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        student_logits: torch.Tensor,   # [B, T, vocab_size]
        teacher_tokens: torch.Tensor,   # [B, T] — teacher's chosen token IDs
        mask: torch.Tensor,             # [B, T] — valid positions
    ) -> torch.Tensor:
        """Compute masked cross-entropy loss.

        Args:
            student_logits: Student's full logits.
            teacher_tokens: Teacher's hard token choices (argmax).
            mask: Boolean mask of valid timesteps.

        Returns:
            Scalar loss (averaged over valid positions).
        """
        B, T, V = student_logits.shape

        # Flatten for cross_entropy
        logits_flat = student_logits.reshape(B * T, V)
        targets_flat = teacher_tokens.reshape(B * T)
        mask_flat = mask.reshape(B * T).float()

        # Per-element CE
        ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # [B*T]
        ce = ce * mask_flat

        return ce.sum() / mask_flat.sum().clamp(min=1.0)


class CodebookCELoss(nn.Module):
    """Cross-entropy loss for depth transformer codebook predictions (CB 1–7).

    Computes masked CE between the student's codebook logits and the teacher's
    hard codebook token predictions for codebooks 1 through 7.  This encourages
    the student's depth transformer to reproduce the teacher's inter-codebook
    dependency modeling.
    """

    def forward(
        self,
        student_cb_logits: torch.Tensor,   # [B, 7, T, card]
        teacher_cb_tokens: torch.Tensor,   # [B, 7, T]
        mask: torch.Tensor,                # [B, T]
    ) -> torch.Tensor:
        """Compute masked cross-entropy across codebooks 1–7.

        Args:
            student_cb_logits: Student's logits for CB 1–7.
            teacher_cb_tokens: Teacher's hard token choices for CB 1–7.
            mask: Boolean mask of valid timesteps.

        Returns:
            Scalar loss (averaged over valid positions and codebooks).
        """
        B, n_cb, T, V = student_cb_logits.shape

        # Expand mask: [B, T] → [B, n_cb, T] → flatten
        mask_flat = (
            mask.unsqueeze(1).expand(B, n_cb, T).reshape(B * n_cb * T).float()
        )

        logits_flat = student_cb_logits.reshape(B * n_cb * T, V)
        targets_flat = teacher_cb_tokens.reshape(B * n_cb * T)

        ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        ce = ce * mask_flat

        return ce.sum() / mask_flat.sum().clamp(min=1.0)


class HiddenStateAlignmentLoss(nn.Module):
    """MSE loss between projected student hidden states and teacher hidden states.

    Uses learnable linear projections to map student hidden dim → teacher hidden
    dim at K aligned layer pairs.  Projection parameters are trained alongside
    the student.

    Note:
        Requires teacher hidden states to be stored in the self-play dataset
        (not available by default — only generated if loss ablation picks L3/L5).
        Also requires forward hooks on the student to capture hidden states
        during training.
    """

    def __init__(
        self,
        student_hidden_dim: int = 4096,
        teacher_hidden_dim: int = 4096,
        num_aligned_layers: int = 4,
    ):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(student_hidden_dim, teacher_hidden_dim, bias=False)
            for _ in range(num_aligned_layers)
        ])
        self.num_aligned_layers = num_aligned_layers

    def forward(
        self,
        student_hidden_states: list[torch.Tensor],  # K × [B, T, student_dim]
        teacher_hidden_states: list[torch.Tensor],   # K × [B, T, teacher_dim]
        mask: torch.Tensor,                          # [B, T]
    ) -> torch.Tensor:
        """Compute projected MSE between aligned hidden state pairs.

        Args:
            student_hidden_states: List of K student hidden state tensors.
            teacher_hidden_states: List of K teacher hidden state tensors.
            mask: Boolean mask of valid timesteps.

        Returns:
            Scalar loss (averaged over aligned layers and valid positions).
        """
        if len(student_hidden_states) != self.num_aligned_layers:
            raise ValueError(
                f"Expected {self.num_aligned_layers} hidden state tensors, "
                f"got {len(student_hidden_states)}"
            )

        mask_f = mask.float().unsqueeze(-1)  # [B, T, 1]
        n_valid = mask_f.sum().clamp(min=1.0)
        total_loss = torch.tensor(0.0, device=mask.device)

        for proj, s_hidden, t_hidden in zip(
            self.projections, student_hidden_states, teacher_hidden_states
        ):
            projected = proj(s_hidden)  # [B, T, teacher_dim]
            diff = projected - t_hidden.detach()
            mse_per_elem = diff.pow(2)  # [B, T, teacher_dim]
            # Mask, average over hidden dim, then over valid positions
            mse = (mse_per_elem * mask_f).sum() / (n_valid * t_hidden.shape[-1])
            total_loss = total_loss + mse

        return total_loss / self.num_aligned_layers


class DistillationLoss(nn.Module):
    """Combined KD loss with configurable components (L1–L5).

    Supports all five loss configurations from the project plan:
        L1: α·L_logit                                          (pure soft matching)
        L2: α·L_logit + δ·L_hard                               (+ hard label CE)
        L3: α·L_logit + β·L_hidden + δ·L_hard                  (+ hidden state MSE)
        L4: α·L_logit + γ·L_codebook + δ·L_hard                (+ codebook CE)
        L5: α·L_logit + β·L_hidden + γ·L_codebook + δ·L_hard   (all)

    Both text head and audio CB0 head are distilled via logit KD + hard labels.
    Codebook CE targets CB 1–7 (from the Depth Transformer).
    Hidden state alignment requires pre-stored teacher hidden states.

    Usage:
        # By config name (recommended — weights auto-set per plan):
        loss = DistillationLoss(loss_config="L4", temperature=3.0)

        # Manual weights (backward compatible with existing code):
        loss = DistillationLoss(alpha=0.7, delta=0.3, temperature=3.0)
    """

    def __init__(
        self,
        loss_config: str | None = None,
        alpha: float = 0.7,        # logit KD weight
        beta: float = 0.0,         # hidden state weight
        gamma: float = 0.0,        # codebook CE weight
        delta: float = 0.3,        # hard label weight
        temperature: float = 3.0,  # KD temperature
        # Hidden state alignment params (only used if β > 0)
        student_hidden_dim: int = 4096,
        teacher_hidden_dim: int = 4096,
        num_aligned_layers: int = 4,
    ):
        super().__init__()

        # If loss_config string provided, override weights from plan
        if loss_config is not None:
            if loss_config not in LOSS_CONFIGS:
                raise ValueError(
                    f"Unknown loss config '{loss_config}'. "
                    f"Choose from: {list(LOSS_CONFIGS.keys())}"
                )
            cfg = LOSS_CONFIGS[loss_config]
            alpha = cfg["alpha"]
            beta = cfg["beta"]
            gamma = cfg["gamma"]
            delta = cfg["delta"]

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.loss_config = loss_config

        # Core losses (always available)
        self.logit_kd = SparseLogitKDLoss(temperature=temperature)
        self.hard_label = HardLabelLoss()

        # Codebook CE — only instantiated when needed (L4, L5)
        self.codebook_ce = CodebookCELoss() if gamma > 0 else None

        # Hidden state alignment — only instantiated when needed (L3, L5)
        self.hidden_state = None
        if beta > 0:
            self.hidden_state = HiddenStateAlignmentLoss(
                student_hidden_dim=student_hidden_dim,
                teacher_hidden_dim=teacher_hidden_dim,
                num_aligned_layers=num_aligned_layers,
            )

    def forward(
        self,
        student_text_logits: torch.Tensor,            # [B, T, text_card]
        student_audio_cb0_logits: torch.Tensor,        # [B, T, audio_card]
        teacher_text_logits_vals: torch.Tensor,        # [B, T, K]
        teacher_text_logits_idxs: torch.Tensor,        # [B, T, K]
        teacher_audio_cb0_logits_vals: torch.Tensor,   # [B, T, K]
        teacher_audio_cb0_logits_idxs: torch.Tensor,   # [B, T, K]
        teacher_text_tokens: torch.Tensor,             # [B, T]
        teacher_audio_tokens: torch.Tensor,            # [B, 8, T]
        mask: torch.Tensor,                            # [B, T]
        # ── Optional inputs for L4/L5 ──
        student_audio_cb1_7_logits: torch.Tensor | None = None,  # [B, 7, T, card]
        # ── Optional inputs for L3/L5 ──
        student_hidden_states: list[torch.Tensor] | None = None,
        teacher_hidden_states: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined distillation loss.

        Args:
            student_text_logits: Student text head logits.
            student_audio_cb0_logits: Student audio CB0 logits.
            teacher_text_logits_vals: Sparse top-K teacher text logit values.
            teacher_text_logits_idxs: Sparse top-K teacher text logit indices.
            teacher_audio_cb0_logits_vals: Sparse top-K teacher audio CB0 values.
            teacher_audio_cb0_logits_idxs: Sparse top-K teacher audio CB0 indices.
            teacher_text_tokens: Teacher's hard text token choices.
            teacher_audio_tokens: Teacher's hard audio tokens (all 8 codebooks).
            mask: Boolean mask of valid timesteps.
            student_audio_cb1_7_logits: Student logits for CB 1–7 (required for L4/L5).
            student_hidden_states: Student hidden states at aligned layers (L3/L5).
            teacher_hidden_states: Teacher hidden states at aligned layers (L3/L5).

        Returns:
            Dict with 'total' and per-component losses (all detached except 'total').
        """
        losses = {}
        total = torch.tensor(0.0, device=mask.device, dtype=student_text_logits.dtype)

        # ── Logit KD (all configs) ──
        if self.alpha > 0:
            l_logit_text = self.logit_kd(
                student_text_logits, teacher_text_logits_vals,
                teacher_text_logits_idxs, mask,
            )
            l_logit_audio = self.logit_kd(
                student_audio_cb0_logits, teacher_audio_cb0_logits_vals,
                teacher_audio_cb0_logits_idxs, mask,
            )
            total = total + self.alpha * (l_logit_text + l_logit_audio)
            losses["logit_text"] = l_logit_text.detach()
            losses["logit_audio"] = l_logit_audio.detach()

        # ── Hard Label CE (L2, L3, L4, L5) ──
        if self.delta > 0:
            l_hard_text = self.hard_label(
                student_text_logits, teacher_text_tokens, mask,
            )
            l_hard_audio = self.hard_label(
                student_audio_cb0_logits, teacher_audio_tokens[:, 0, :], mask,
            )
            total = total + self.delta * (l_hard_text + l_hard_audio)
            losses["hard_text"] = l_hard_text.detach()
            losses["hard_audio"] = l_hard_audio.detach()

        # ── Codebook CE (L4, L5) ──
        if self.gamma > 0:
            if self.codebook_ce is None:
                raise RuntimeError("CodebookCELoss not initialized but γ > 0")
            if student_audio_cb1_7_logits is None:
                raise ValueError(
                    f"Loss config requires codebook CE (γ={self.gamma}) but "
                    f"student_audio_cb1_7_logits was not provided. "
                    f"Pass lm_output.logits[:, 1:, :, :] from the student."
                )
            l_codebook = self.codebook_ce(
                student_audio_cb1_7_logits,
                teacher_audio_tokens[:, 1:, :],
                mask,
            )
            total = total + self.gamma * l_codebook
            losses["codebook_ce"] = l_codebook.detach()

        # ── Hidden State Alignment (L3, L5) ──
        if self.beta > 0:
            if self.hidden_state is None:
                raise RuntimeError(
                    "HiddenStateAlignmentLoss not initialized but β > 0"
                )
            if student_hidden_states is None or teacher_hidden_states is None:
                raise ValueError(
                    f"Loss config requires hidden state alignment (β={self.beta}) "
                    f"but hidden states were not provided. L3/L5 require: "
                    f"(1) teacher hidden states in the .npz dataset, "
                    f"(2) student hidden state capture via forward hooks."
                )
            l_hidden = self.hidden_state(
                student_hidden_states, teacher_hidden_states, mask,
            )
            total = total + self.beta * l_hidden
            losses["hidden_state"] = l_hidden.detach()

        losses["total"] = total
        return losses
