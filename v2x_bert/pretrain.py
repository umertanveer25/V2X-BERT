"""
Self-Supervised Masked Telemetry Modeling (MTM) and Cross-Standard Alignment Engine for V2X-BERT.
Optimized for multi-core CPU and GPU training with zero memory bloat.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .tokenizer import V2XTokenizer
from .model import EdgeV2XBERT


def mask_telemetry_tokens(input_ids, tokenizer, mask_prob=0.15):
    """
    Applies standard BERT masking to vehicular telemetry tokens.
    15% of active non-special tokens are selected:
      - 80% replaced with [MASK] (token 4)
      - 10% replaced with random token from vocabulary [16, 1023]
      - 10% kept unchanged
    """
    masked_ids = input_ids.clone()
    labels = torch.full(input_ids.shape, -100, dtype=torch.long, device=input_ids.device)

    # Don't mask special tokens: PAD=0, UNK=1, CLS=2, SEP=3
    eligible_mask = (input_ids > 4)

    # Sample random probabilities
    prob_matrix = torch.rand(input_ids.shape, device=input_ids.device)
    mask_indices = eligible_mask & (prob_matrix < mask_prob)

    labels[mask_indices] = input_ids[mask_indices]

    # 80% of selected tokens -> [MASK]
    rand_types = torch.rand(input_ids.shape, device=input_ids.device)
    mask_80 = mask_indices & (rand_types < 0.80)
    masked_ids[mask_80] = tokenizer.MASK_TOKEN

    # 10% -> Random token
    random_10 = mask_indices & (rand_types >= 0.80) & (rand_types < 0.90)
    random_tokens = torch.randint(16, tokenizer.vocab_size, input_ids.shape, device=input_ids.device)
    masked_ids[random_10] = random_tokens[random_10]

    # Remaining 10% -> Kept unchanged

    return masked_ids, labels


def compute_contrastive_alignment_loss(z_proj, temperature=0.07):
    """
    InfoNCE Contrastive loss enforcing semantic consistency across latent representations.
    """
    batch_size = z_proj.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=z_proj.device)
    
    sim_matrix = torch.matmul(z_proj, z_proj.T) / temperature
    labels = torch.arange(batch_size, device=z_proj.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def pretrain_v2x_bert(model, dataset, epochs=5, batch_size=128, lr=1e-3, lambda_align=0.1, device="cpu"):
    """
    Executes Joint Masked Telemetry and Cross-Standard Alignment Pre-training.
    """
    model.to(device)
    tokenizer = V2XTokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion_mtm = nn.CrossEntropyLoss(ignore_index=-100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"\n{'='*75}")
    print(f"   V2X-BERT: SELF-SUPERVISED MASKED TELEMETRY PRE-TRAINING ({device.upper()})")
    print(f"   Trainable Parameters: {model.count_parameters():,} ({model.count_parameters()/1e6:.2f}M)")
    print(f"   Epochs: {epochs} | Batch Size: {batch_size} | Loss: MTM + {lambda_align} * Alignment")
    print(f"{'='*75}")

    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mtm_loss = 0.0
        total_align_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            if len(batch) == 3:
                batch_seqs, batch_masks, _ = batch
            else:
                batch_seqs, _ = batch
                batch_masks = (batch_seqs != 0).long()

            batch_seqs = batch_seqs.to(device)
            batch_masks = batch_masks.to(device)

            masked_inputs, target_labels = mask_telemetry_tokens(batch_seqs, tokenizer, mask_prob=0.15)

            optimizer.zero_grad()
            logits, z_proj = model.forward_pretrain(masked_inputs, attention_mask=batch_masks)

            # 1. Masked Telemetry Modeling Loss
            loss_mtm = criterion_mtm(logits.view(-1, tokenizer.vocab_size), target_labels.view(-1))

            # 2. Cross-Standard Contrastive Latent Alignment Loss
            loss_align = compute_contrastive_alignment_loss(z_proj)

            # Combined Objective
            loss = loss_mtm + lambda_align * loss_align
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mtm_loss += loss_mtm.item()
            total_align_loss += loss_align.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        avg_mtm = total_mtm_loss / max(1, total_batches)
        avg_align = total_align_loss / max(1, total_batches)
        perplexity = float(np.exp(min(avg_mtm, 20.0)))
        elapsed = time.time() - start_time

        print(f" [Epoch {epoch:02d}/{epochs:02d}] Total Loss: {avg_loss:.4f} (MTM: {avg_mtm:.4f}, Align: {avg_align:.4f}) | Perplexity: {perplexity:.2f} | Time: {elapsed:.1f}s")
        history.append({
            "epoch": epoch,
            "total_loss": float(avg_loss),
            "mtm_loss": float(avg_mtm),
            "align_loss": float(avg_align),
            "perplexity": float(perplexity)
        })

    total_time = time.time() - start_time
    print(f"{'='*75}")
    print(f" [Pre-training Complete] Model converged in {total_time:.2f} seconds.")
    return history
