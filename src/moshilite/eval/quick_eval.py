"""Fast inference-only evaluation for pruning candidates."""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass

from moshilite.analysis.moshi_hooks import moshi_get_logits, moshi_loss_fn
from moshilite.analysis.moshi_model import get_model_info

logger = logging.getLogger(__name__)


@dataclass
class QuickEvalMetrics:
    """Results from quick post-prune evaluation."""
    params_b: float
    text_perplexity: float
    codebook_accuracies: list[float]  # Accuracies for CB 0 through 7
    mean_codebook_acc: float
    output_cosine_sim: float


def run_quick_eval(
    model: nn.Module,
    data: list[torch.Tensor],
    full_model_targets: Optional[list[torch.Tensor]] = None,
    device: str = "cuda"
) -> QuickEvalMetrics:
    """Runs a quick evaluation pipeline to rank pruning variants.
    
    Args:
        model: Mutated Moshi model.
        data: List of token tensors [B, K, SeqLen].
        full_model_targets: Optional list of full-model output logits for cosine similarity.
        device: Torch device.
        
    Returns:
        QuickEvalMetrics containing the results.
    """
    model.eval()
    
    # 1. Parameter count
    info = get_model_info(model)
    params_b = info.get("total_params_b", 0.0)
    
    # Trackers
    total_loss = 0.0
    total_tokens_text = 0
    total_tokens_audio = 0
    
    cb_correct = [0] * 8
    cb_total = [0] * 8
    
    cosine_sims = []

    with torch.no_grad():
        for i, batch in enumerate(data):
            batch = batch.to(device)
            # targets are the batch shifted or we can just use the batch directly 
            # if we are doing next-token prediction
            
            # Extract outputs
            try:
                outputs = model(batch)
            except Exception as e:
                logger.error(f"Forward pass failed! {e}")
                return QuickEvalMetrics(params_b, float('inf'), [0]*8, 0.0, 0.0)
                
            audio_logits = outputs.logits      # [B, K, T, card]
            text_logits = outputs.text_logits  # [B, 1, T, t_card]
            
            # Loss for perplexity
            loss = moshi_loss_fn(outputs, batch)
            total_loss += loss.item() * batch.size(0)
            total_tokens_text += batch.size(0)
            
            # Accuracies
            # text_preds = text_logits.argmax(dim=-1)
            audio_preds = audio_logits.argmax(dim=-1)  # [B, K, T]
            
            # Compare to targets (batch shifted by 1 in time)
            targets = batch[:, 1:, :audio_preds.shape[-1]]
            
            # Limit K to what we have
            n_cb_eval = min(8, audio_preds.shape[1], targets.shape[1])
            mask = outputs.mask[:, :n_cb_eval, :audio_preds.shape[-1]]
            
            for cb in range(n_cb_eval):
                cb_p = audio_preds[:, cb, :]
                cb_t = targets[:, cb, :]
                cb_m = mask[:, cb, :]
                if cb_m.any():
                    correct = (cb_p[cb_m] == cb_t[cb_m]).sum().item()
                    total = cb_m.sum().item()
                    cb_correct[cb] += correct
                    cb_total[cb] += total
                    
            # Cosine similarity vs full model targets
            if full_model_targets is not None and i < len(full_model_targets):
                target_logits = full_model_targets[i].to(device)
                
                # Reshape to 2D to compute cosine sim across vocabulary
                pred_flat = audio_logits.view(-1, audio_logits.shape[-1]).float()
                targ_flat = target_logits.view(-1, target_logits.shape[-1]).float()
                
                # Compute sample-wise cosine similarity and average
                sim = torch.nn.functional.cosine_similarity(pred_flat, targ_flat)
                cosine_sims.append(sim.mean().item())

    # Average perplexity
    mean_loss = total_loss / max(1, total_tokens_text)
    perplexity = np.exp(mean_loss)
    
    # Average accuracies
    cb_accs = []
    for c, t in zip(cb_correct, cb_total):
        if t > 0:
            cb_accs.append(c / t)
        else:
            cb_accs.append(0.0)
            
    mean_acc = sum(cb_accs) / max(1, len([a for a in cb_accs if a > 0]))
    
    # Average cosine sim
    mean_cos = sum(cosine_sims) / max(1, len(cosine_sims)) if cosine_sims else 0.0
    
    logger.info(f"Eval completed: Params {params_b:.2f}B | PPL {perplexity:.2f} | CB_Acc {mean_acc:.3f} | CosSim {mean_cos:.3f}")

    return QuickEvalMetrics(
        params_b=params_b,
        text_perplexity=perplexity,
        codebook_accuracies=cb_accs,
        mean_codebook_acc=mean_acc,
        output_cosine_sim=mean_cos
    )
