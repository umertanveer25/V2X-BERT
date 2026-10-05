"""
Downstream Evaluation & Fine-Tuning Engine for V2X-BERT.
Evaluates Zero-Trust Misbehavior Detection across real VeReMi attacks,
Cross-Standard Transfer, and Comparative Ablations.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def fine_tune_and_evaluate(model, dataset, epochs=4, batch_size=128, lr=3e-4, device="cpu"):
    """
    Fine-tunes the pre-trained V2X-BERT on downstream Zero-Trust Misbehavior Detection.
    Evaluates on 80/20 train/test split.
    """
    model.to(device)
    
    # Split into 80% train, 20% validation
    n_total = len(dataset)
    n_train = int(0.80 * n_total)
    n_val = n_total - n_train
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*75}")
    print(f"   DOWNSTREAM TASK 1: ZERO-TRUST MISBEHAVIOR DETECTION (VeReMi)")
    print(f"   Train Samples: {n_train:,} | Validation Samples: {n_val:,} | Epochs: {epochs}")
    print(f"{'='*75}")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits, _ = model.forward_classify(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f" [Fine-Tune Epoch {epoch}/{epochs}] Train Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation on Held-Out Validation Set
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    latencies = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            t0 = time.perf_counter()
            logits, _ = model.forward_classify(batch_x)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) / len(batch_x) * 1000.0) # ms per message
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec = precision_score(all_targets, all_preds, zero_division=0) * 100.0
    rec = recall_score(all_targets, all_preds, zero_division=0) * 100.0
    f1 = f1_score(all_targets, all_preds, zero_division=0) * 100.0
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = (fp / (fp + tn + 1e-8)) * 100.0
    avg_latency_us = np.mean(latencies) * 1000.0

    print(f"{'-'*75}")
    print(f"   V2X-BERT EVALUATION METRICS:")
    print(f"   • Accuracy:        {acc:.2f}%")
    print(f"   • Precision:       {prec:.2f}%")
    print(f"   • Recall:          {rec:.2f}%")
    print(f"   • F1-Score:        {f1:.2f}%")
    print(f"   • False Pos. Rate: {fpr:.2f}%")
    print(f"   • AUC-ROC:         {auc:.4f}")
    print(f"   • Inference Time:  {avg_latency_us:.2f} us per message ({avg_latency_us/1000.0:.3f} ms)")
    print(f"{'='*75}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "latency_us": avg_latency_us
    }
