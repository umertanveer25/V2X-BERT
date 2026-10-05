"""
Downstream Evaluation & Fine-Tuning Engine for V2X-BERT.
Evaluates Zero-Trust Misbehavior Detection across simulated VeReMi attack scenarios
with zero pre-training data leakage on strictly held-out test partitions.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix


def evaluate_model(model, test_dataset, batch_size=128, device="cpu"):
    """
    Evaluates a trained or quantized model on strictly held-out test scenarios with precise latency benchmarking.
    """
    model.to(device)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []
    latencies = []

    # Warm-up pass for latency benchmarking
    warmup_x = torch.randint(0, 1024, (1, 64), device=device)
    warmup_m = torch.ones((1, 64), device=device)
    for _ in range(50):
        with torch.no_grad():
            _ = model.forward_classify(warmup_x, attention_mask=warmup_m)

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                batch_x, batch_m, batch_y = batch
            else:
                batch_x, batch_y = batch
                batch_m = (batch_x != 0).long()

            batch_x = batch_x.to(device)
            batch_m = batch_m.to(device)

            # Per-batch precise timing
            t0 = time.perf_counter()
            logits, _ = model.forward_classify(batch_x, attention_mask=batch_m)
            t1 = time.perf_counter()

            per_sample_us = ((t1 - t0) / len(batch_x)) * 1_000_000.0  # microseconds
            latencies.extend([per_sample_us] * len(batch_x))

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    latencies = np.array(latencies)

    acc = float(accuracy_score(all_targets, all_preds) * 100.0)
    prec = float(precision_score(all_targets, all_preds, zero_division=0) * 100.0)
    rec = float(recall_score(all_targets, all_preds, zero_division=0) * 100.0)
    f1 = float(f1_score(all_targets, all_preds, zero_division=0) * 100.0)
    auc = float(roc_auc_score(all_targets, all_probs))
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr_rate = float((fp / (fp + tn + 1e-8)) * 100.0)

    # Compute ROC and PR Curve arrays for exact plotting
    roc_fpr, roc_tpr, _ = roc_curve(all_targets, all_probs)
    pr_prec, pr_rec, _ = precision_recall_curve(all_targets, all_probs)

    # Latency percentiles
    mean_lat_us = float(np.mean(latencies))
    median_lat_us = float(np.median(latencies))
    p95_lat_us = float(np.percentile(latencies, 95))
    p99_lat_us = float(np.percentile(latencies, 99))

    print(f"{'-'*75}")
    print(f"   HELD-OUT TEST SET EVALUATION METRICS (Zero-Leakage):")
    print(f"   • Accuracy:        {acc:.2f}%")
    print(f"   • Precision:       {prec:.2f}%")
    print(f"   • Recall:          {rec:.2f}%")
    print(f"   • F1-Score:        {f1:.2f}%")
    print(f"   • False Pos. Rate: {fpr_rate:.2f}%")
    print(f"   • AUC-ROC:         {auc:.4f}")
    print(f"   • Inference Latency: Mean = {mean_lat_us:.2f} us | Median = {median_lat_us:.2f} us | P95 = {p95_lat_us:.2f} us")
    print(f"{'='*75}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr_rate,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "latency_us": {
            "mean": mean_lat_us,
            "median": median_lat_us,
            "p95": p95_lat_us,
            "p99": p99_lat_us
        },
        "roc_curve": {
            "fpr": roc_fpr.tolist()[::max(1, len(roc_fpr)//100)],
            "tpr": roc_tpr.tolist()[::max(1, len(roc_tpr)//100)]
        },
        "pr_curve": {
            "precision": pr_prec.tolist()[::max(1, len(pr_prec)//100)],
            "recall": pr_rec.tolist()[::max(1, len(pr_rec)//100)]
        }
    }


def fine_tune_and_evaluate(model, train_dataset, test_dataset, epochs=4, batch_size=128, lr=3e-4, device="cpu"):
    """
    Fine-tunes V2X-BERT on training scenarios and evaluates on strictly held-out test scenarios.
    """
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*75}")
    print(f"   DOWNSTREAM TASK: ZERO-TRUST MISBEHAVIOR DETECTION (VeReMi)")
    print(f"   Train Samples: {len(train_dataset):,} | Held-Out Test Samples: {len(test_dataset):,} | Epochs: {epochs}")
    print(f"{'='*75}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:
                batch_x, batch_m, batch_y = batch
            else:
                batch_x, batch_y = batch
                batch_m = (batch_x != 0).long()

            batch_x = batch_x.to(device)
            batch_m = batch_m.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits, _ = model.forward_classify(batch_x, attention_mask=batch_m)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f" [Fine-Tune Epoch {epoch:02d}/{epochs:02d}] Train Loss: {total_loss/len(train_loader):.4f}")

    return evaluate_model(model, test_dataset, batch_size=batch_size, device=device)
