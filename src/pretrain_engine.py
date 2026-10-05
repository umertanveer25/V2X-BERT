"""
Self-Supervised Masked Telemetry Modeling (MTM) Pre-training Engine for V2X-BERT.
Optimized for multithreaded CPU training with zero memory bloat.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .v2x_tokenizer import V2XTokenizer
from .v2x_bert_model import EdgeV2XBERT


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

    # Remaining 10% -> Unchanged

    return masked_ids, labels


def pretrain_v2x_bert(model, dataset, epochs=5, batch_size=128, lr=1e-3, device="cpu"):
    """
    Executes Masked Telemetry Pre-training on the standards-tokenized corpus.
    """
    model.to(device)
    tokenizer = V2XTokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"\n{'='*75}")
    print(f"   V2X-BERT: SELF-SUPERVISED MASKED TELEMETRY PRE-TRAINING (CPU)")
    print(f"   Architecture Parameters: {model.count_parameters():,} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"{'='*75}")

    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch_seqs, _ in dataloader:
            batch_seqs = batch_seqs.to(device)
            masked_inputs, target_labels = mask_telemetry_tokens(batch_seqs, tokenizer, mask_prob=0.15)

            optimizer.zero_grad()
            logits = model.forward_pretrain(masked_inputs)

            # Flatten for CrossEntropyLoss
            loss = criterion(logits.view(-1, tokenizer.vocab_size), target_labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        elapsed = time.time() - start_time
        perplexity = np.exp(min(avg_loss, 20.0))

        print(f" [Epoch {epoch:02d}/{epochs:02d}] MTM Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f} | Time: {elapsed:.1f}s")
        history.append({"epoch": epoch, "loss": avg_loss, "perplexity": perplexity})

    print(f"{'='*75}")
    print(f" [Pre-training Complete] Model converged in {time.time() - start_time:.2f} seconds.")
    return history
