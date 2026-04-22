"""Dataset and DataLoader for self-play teacher targets.

Loads .npz files from self-play generation (Stage 4a) and serves them
as training samples for student offline distillation (Stage 4b).

Each sample contains:
  - Input context: text tokens + audio tokens (teacher-forced sequence)
  - Teacher targets: sparse top-K logits (text + audio CB0) + hard labels
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional


class SelfPlayDataset(Dataset):
    """PyTorch Dataset wrapping self-play .npz conversation files.

    Each __getitem__ returns a dict with tensors ready for the KD loss:
        text_tokens:           [T] int64   — teacher text token sequence
        audio_tokens:          [8, T] int64 — teacher audio codebook tokens
        user_audio_tokens:     [8, T] int64 — Channel A tokens (for student input)
        text_logits_vals:      [T, K] float32 — sparse top-K text logit values
        text_logits_idxs:      [T, K] int64   — sparse top-K text logit indices
        audio_cb0_logits_vals: [T, K] float32 — sparse top-K audio CB0 logit values
        audio_cb0_logits_idxs: [T, K] int64   — sparse top-K audio CB0 logit indices
        num_valid_steps:       int             — number of valid timesteps
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_steps: Optional[int] = None,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Path to conversation directory (contains .npz files,
                      possibly in subdirectories like batch_000/, batch_001/).
            max_steps: If set, truncate conversations to this many steps.
            split: "train" or "val".
            val_fraction: Fraction of data to hold out for validation.
            seed: RNG seed for train/val split.
        """
        self.data_dir = Path(data_dir)
        self.max_steps = max_steps

        # Collect all .npz files (search recursively for multi-batch support)
        all_files = sorted(self.data_dir.rglob("*.npz"))
        # Exclude files in 'rejected/' subdirectories
        all_files = [f for f in all_files if "rejected" not in f.parts]

        if len(all_files) == 0:
            raise FileNotFoundError(
                f"No .npz files found in {self.data_dir}. "
                f"Run self-play generation first (Stage 4a)."
            )

        # Deterministic train/val split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_files))
        val_size = max(1, int(len(all_files) * val_fraction))

        if split == "val":
            selected = indices[:val_size]
        elif split == "train":
            selected = indices[val_size:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.files = [all_files[i] for i in selected]
        print(f"📂 SelfPlayDataset [{split}]: {len(self.files)} conversations "
              f"from {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))

        num_valid = int(data["num_valid_steps"][0])
        T = num_valid
        if self.max_steps is not None:
            T = min(T, self.max_steps)

        return {
            "text_tokens": torch.from_numpy(
                data["text_tokens"][:T].astype(np.int64)
            ),
            "audio_tokens": torch.from_numpy(
                data["audio_tokens"][:, :T].astype(np.int64)
            ),
            "user_audio_tokens": torch.from_numpy(
                data["user_audio_tokens"][:, :T].astype(np.int64)
            ),
            "text_logits_vals": torch.from_numpy(
                data["text_logits_vals"][:T].astype(np.float32)
            ),
            "text_logits_idxs": torch.from_numpy(
                data["text_logits_idxs"][:T].astype(np.int64)
            ),
            "audio_cb0_logits_vals": torch.from_numpy(
                data["audio_cb0_logits_vals"][:T].astype(np.float32)
            ),
            "audio_cb0_logits_idxs": torch.from_numpy(
                data["audio_cb0_logits_idxs"][:T].astype(np.int64)
            ),
            "num_valid_steps": T,
        }


def collate_self_play(batch: list[dict]) -> dict:
    """Collate variable-length conversations into a padded batch.

    Pads all sequences to the max length in the batch.
    Returns a mask tensor indicating valid positions.
    """
    max_T = max(item["num_valid_steps"] for item in batch)
    B = len(batch)
    top_k = batch[0]["text_logits_vals"].shape[-1]

    result = {
        "text_tokens": torch.zeros(B, max_T, dtype=torch.long),
        "audio_tokens": torch.zeros(B, 8, max_T, dtype=torch.long),
        "user_audio_tokens": torch.zeros(B, 8, max_T, dtype=torch.long),
        "text_logits_vals": torch.zeros(B, max_T, top_k),
        "text_logits_idxs": torch.zeros(B, max_T, top_k, dtype=torch.long),
        "audio_cb0_logits_vals": torch.zeros(B, max_T, top_k),
        "audio_cb0_logits_idxs": torch.zeros(B, max_T, top_k, dtype=torch.long),
        "mask": torch.zeros(B, max_T, dtype=torch.bool),
        "lengths": torch.zeros(B, dtype=torch.long),
    }

    for i, item in enumerate(batch):
        T = item["num_valid_steps"]
        result["text_tokens"][i, :T] = item["text_tokens"]
        result["audio_tokens"][i, :, :T] = item["audio_tokens"]
        result["user_audio_tokens"][i, :, :T] = item["user_audio_tokens"]
        result["text_logits_vals"][i, :T] = item["text_logits_vals"]
        result["text_logits_idxs"][i, :T] = item["text_logits_idxs"]
        result["audio_cb0_logits_vals"][i, :T] = item["audio_cb0_logits_vals"]
        result["audio_cb0_logits_idxs"][i, :T] = item["audio_cb0_logits_idxs"]
        result["mask"][i, :T] = True
        result["lengths"][i] = T

    return result


def get_self_play_dataloader(
    data_dir: str | Path,
    split: str = "train",
    batch_size: int = 4,
    max_steps: int | None = None,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for self-play training data.

    Args:
        data_dir: Path to self-play conversation directory.
        split: "train" or "val".
        batch_size: Batch size.
        max_steps: Optional max timesteps per conversation.
        val_fraction: Fraction of data for validation.
        num_workers: DataLoader workers (0 for Colab compatibility).
        seed: RNG seed for split.

    Returns:
        DataLoader yielding padded batches.
    """
    dataset = SelfPlayDataset(
        data_dir=data_dir,
        max_steps=max_steps,
        split=split,
        val_fraction=val_fraction,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_self_play,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
