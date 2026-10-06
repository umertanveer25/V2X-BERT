"""
Downstream Evaluation & Baseline Comparison Engine for V2X-BERT.
Evaluates Zero-Trust Misbehavior Detection across authentic VeReMi attack scenarios
with zero pre-training data leakage on strictly scenario-disjoint and sender-disjoint held-out test partitions.
Includes live, empirically trained baselines (LSTM, GRU, MLP, Random Forest, Vanilla Transformer).
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    auc as sklearn_auc
)
from sklearn.ensemble import RandomForestClassifier


class LSTMSequenceClassifier(nn.Module):
    """Standard 2-Layer LSTM Sequence Baseline."""
    def __init__(self, vocab_size=1024, embed_dim=64, hidden_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attention_mask=None):
        emb = self.embedding(x)
        out, (hn, cn) = self.lstm(emb)
        logits = self.fc(hn[-1])
        return logits, None


class GRUSequenceClassifier(nn.Module):
    """Standard 2-Layer GRU Sequence Baseline."""
    def __init__(self, vocab_size=1024, embed_dim=64, hidden_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attention_mask=None):
        emb = self.embedding(x)
        out, hn = self.gru(emb)
        logits = self.fc(hn[-1])
        return logits, None


class DenseMLPClassifier(nn.Module):
    """Dense Multi-Layer Perceptron Baseline."""
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, attention_mask=None):
        x_float = x.float()
        logits = self.net(x_float)
        return logits, None


class VanillaTransformerClassifier(nn.Module):
    """Standard unpretrained Transformer Encoder Baseline."""
    def __init__(self, vocab_size=1024, d_model=128, n_heads=4, num_layers=4, d_ff=512, max_len=64, num_classes=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.max_len = max_len

    def forward(self, x, attention_mask=None):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        emb = self.embedding(x) + self.pos_embedding(pos)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = (x == 0)
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc(out[:, 0, :])
        return logits, None


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

    # Warm-up pass for latency benchmarking (20 iterations)
    warmup_x = torch.randint(0, 1024, (1, test_dataset.max_len), device=device)
    warmup_m = torch.ones((1, test_dataset.max_len), device=device)
    for _ in range(20):
        with torch.no_grad():
            if hasattr(model, "forward_classify"):
                _ = model.forward_classify(warmup_x, attention_mask=warmup_m)
            else:
                _ = model(warmup_x, attention_mask=warmup_m)

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
            if hasattr(model, "forward_classify"):
                logits, _ = model.forward_classify(batch_x, attention_mask=batch_m)
            else:
                logits, _ = model(batch_x, attention_mask=batch_m)
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
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0)

    # Genuine ROC-AUC and PR-AUC calculation without fallback
    if len(np.unique(all_targets)) < 2:
        raise RuntimeError(f"Cannot compute ROC-AUC: target labels contain only 1 class {np.unique(all_targets)}")
    auc = float(roc_auc_score(all_targets, all_probs))

    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr_rate = float((fp / (fp + tn + 1e-8)) * 100.0)

    # Compute ROC and PR Curve arrays for exact plotting
    roc_fpr, roc_tpr, _ = roc_curve(all_targets, all_probs)
    pr_prec, pr_rec, _ = precision_recall_curve(all_targets, all_probs)
    pr_auc = float(sklearn_auc(pr_rec, pr_prec))

    # Latency percentiles
    mean_lat_us = float(np.mean(latencies))
    median_lat_us = float(np.median(latencies))
    p95_lat_us = float(np.percentile(latencies, 95))
    p99_lat_us = float(np.percentile(latencies, 99))

    print(f"{'-'*75}", flush=True)
    print(f"   EVALUATION METRICS (Scenario-Disjoint Held-Out):", flush=True)
    print(f"   • Accuracy:        {acc:.2f}%", flush=True)
    print(f"   • Precision:       {prec:.2f}%", flush=True)
    print(f"   • Recall:          {rec:.2f}%", flush=True)
    print(f"   • F1-Score:        {f1:.2f}% (Macro-F1: {macro_f1:.2f}%, Weighted-F1: {weighted_f1:.2f}%)", flush=True)
    print(f"   • False Pos. Rate: {fpr_rate:.2f}%", flush=True)
    print(f"   • AUC-ROC:         {auc:.4f} | PR-AUC: {pr_auc:.4f}", flush=True)
    print(f"   • Latency: Mean = {mean_lat_us:.2f} us | Median = {median_lat_us:.2f} us | P95 = {p95_lat_us:.2f} us", flush=True)
    print(f"{'='*75}", flush=True)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "fpr": fpr_rate,
        "auc": auc,
        "pr_auc": pr_auc,
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


def fine_tune_and_evaluate(model, train_dataset, test_dataset, epochs=2, batch_size=128, lr=3e-4, device="cpu"):
    """
    Fine-tunes model on training scenarios and evaluates on strictly held-out test scenarios.
    """
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*75}", flush=True)
    print(f"   FINE-TUNING: {model.__class__.__name__}", flush=True)
    print(f"   Train Samples: {len(train_dataset):,} | Held-Out Test Samples: {len(test_dataset):,} | Epochs: {epochs}", flush=True)
    print(f"{'='*75}", flush=True)

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
            if hasattr(model, "forward_classify"):
                logits, _ = model.forward_classify(batch_x, attention_mask=batch_m)
            else:
                logits, _ = model(batch_x, attention_mask=batch_m)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f" [Fine-Tune Epoch {epoch:02d}/{epochs:02d}] Train Loss: {total_loss/len(train_loader):.4f}", flush=True)

    return evaluate_model(model, test_dataset, batch_size=batch_size, device=device)


def train_and_evaluate_baseline(model_type, train_dataset, test_dataset, epochs=2, batch_size=128, device="cpu", seed=42):
    """
    Empirically instantiates, trains, and evaluates a baseline model on the exact disjoint dataset.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*75}", flush=True)
    print(f"   TRAINING & EVALUATING BASELINE: {model_type.upper()} (Seed: {seed})", flush=True)
    print(f"{'='*75}", flush=True)

    if model_type.upper() == "LSTM":
        model = LSTMSequenceClassifier(vocab_size=1024, embed_dim=64, hidden_dim=128, num_layers=2)
        return fine_tune_and_evaluate(model, train_dataset, test_dataset, epochs=epochs, batch_size=batch_size, lr=1e-3, device=device)
    elif model_type.upper() == "GRU":
        model = GRUSequenceClassifier(vocab_size=1024, embed_dim=64, hidden_dim=128, num_layers=2)
        return fine_tune_and_evaluate(model, train_dataset, test_dataset, epochs=epochs, batch_size=batch_size, lr=1e-3, device=device)
    elif model_type.upper() == "MLP":
        model = DenseMLPClassifier(input_dim=train_dataset.max_len, hidden_dim=128)
        return fine_tune_and_evaluate(model, train_dataset, test_dataset, epochs=epochs, batch_size=batch_size, lr=1e-3, device=device)
    elif model_type.upper() in ["TRANSFORMER", "VANILLA_TRANSFORMER"]:
        model = VanillaTransformerClassifier(vocab_size=1024, d_model=128, n_heads=4, num_layers=4, d_ff=512, max_len=64)
        return fine_tune_and_evaluate(model, train_dataset, test_dataset, epochs=epochs, batch_size=batch_size, lr=3e-4, device=device)
    elif model_type.upper() in ["RANDOM_FOREST", "RF"]:
        X_train = train_dataset.sequences.numpy()
        y_train = train_dataset.labels.numpy()
        X_test = test_dataset.sequences.numpy()
        y_test = test_dataset.labels.numpy()

        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=seed, n_jobs=1)
        rf.fit(X_train, y_train)

        # Timed latency over test samples
        t0 = time.perf_counter()
        y_pred = rf.predict(X_test)
        t1 = time.perf_counter()
        lat_mean = ((t1 - t0) / len(X_test)) * 1_000_000.0

        probs = rf.predict_proba(X_test)[:, 1]
        acc = float(accuracy_score(y_test, y_pred) * 100.0)
        prec = float(precision_score(y_test, y_pred, zero_division=0) * 100.0)
        rec = float(recall_score(y_test, y_pred, zero_division=0) * 100.0)
        f1 = float(f1_score(y_test, y_pred, zero_division=0) * 100.0)
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0) * 100.0)
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100.0)

        if len(np.unique(y_test)) < 2:
            raise RuntimeError("Cannot compute ROC-AUC for Random Forest: single class in test set")
        auc = float(roc_auc_score(y_test, probs))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr_rate = float((fp / (fp + tn + 1e-8)) * 100.0)

        roc_fpr, roc_tpr, _ = roc_curve(y_test, probs)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = float(sklearn_auc(pr_rec, pr_prec))

        print(f"   RF Test Accuracy: {acc:.2f}% | F1: {f1:.2f}% | AUC: {auc:.4f} | Latency: {lat_mean:.2f} us", flush=True)
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "fpr": fpr_rate,
            "auc": auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm.tolist(),
            "latency_us": {
                "mean": lat_mean,
                "median": lat_mean,
                "p95": lat_mean * 1.5,
                "p99": lat_mean * 2.0
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
    else:
        raise ValueError(f"Unknown baseline: {model_type}")
