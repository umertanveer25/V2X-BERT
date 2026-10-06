"""
DAIR-V2X Vehicle-to-Infrastructure (V2I/V2V) Cooperative Perception Experiment Runner.
Evaluates V2X-BERT and empirical baselines on cooperative multi-message telemetry streams (VIC + RSU Fusion).
"""

import os
import sys
import json
import csv
import argparse
import time
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.v2x_bert_model import EdgeV2XBERT
from src.v2x_tokenizer import V2XTokenizer
from src.dair_v2x_loader import load_dair_v2x_dataset
from src.pretrain_engine import pretrain_v2x_bert
from src.evaluate_downstream import fine_tune_and_evaluate, evaluate_model, train_and_evaluate_baseline


def run_dair_v2x_experiment(samples=2000, pretrain_epochs=2, finetune_epochs=2, batch_size=64, device="cpu"):
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f"   DAIR-V2X VEHICLE-INFRASTRUCTURE COOPERATIVE BENCHMARK EXPERIMENT")
    print(f"   Vehicle (VIC) & Roadside Unit (RSU) Cooperative Perception Fusion")
    print(f"   Samples: {samples:,} | Pre-train Epochs: {pretrain_epochs} | Fine-tune Epochs: {finetune_epochs}")
    print(f"{'#'*80}\n")

    # 1. Load DAIR-V2X paired cooperative dataset
    train_ds, test_ds = load_dair_v2x_dataset(num_samples=samples, seq_len=5, window_tokens=64, test_ratio=0.20)

    # 2. Instantiate V2X-BERT Model
    model = EdgeV2XBERT(vocab_size=1024, d_model=128, n_heads=4, num_layers=4, d_ff=512, max_len=64)

    # 3. Self-Supervised Cooperative Pre-Training
    pretrain_hist = pretrain_v2x_bert(
        model=model,
        dataset=train_ds,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        lr=1e-3,
        lambda_align=0.1,
        device=device
    )

    # 4. Downstream Cooperative Anomaly & Misbehavior Detection on Held-Out Test Set
    eval_v2x_bert = fine_tune_and_evaluate(
        model=model,
        train_dataset=train_ds,
        test_dataset=test_ds,
        epochs=finetune_epochs,
        batch_size=batch_size,
        lr=3e-4,
        device=device
    )

    # 5. INT8 Quantized Edge Evaluation
    quant_model = model.to("cpu").quantize_int8()
    eval_quant = evaluate_model(quant_model, test_ds, batch_size=batch_size, device="cpu")

    # 6. Untrained Baseline Ablation
    untrained_model = EdgeV2XBERT(vocab_size=1024, d_model=128, n_heads=4, num_layers=4, d_ff=512, max_len=64)
    eval_untrained = fine_tune_and_evaluate(
        model=untrained_model,
        train_dataset=train_ds,
        test_dataset=test_ds,
        epochs=finetune_epochs,
        batch_size=batch_size,
        lr=3e-4,
        device=device
    )

    # 7. Live Baseline: GRU Sequence Classifier
    eval_gru = train_and_evaluate_baseline("GRU", train_ds, test_ds, epochs=finetune_epochs, batch_size=batch_size, device=device)

    # 8. Live Baseline: LSTM Sequence Classifier
    eval_lstm = train_and_evaluate_baseline("LSTM", train_ds, test_ds, epochs=finetune_epochs, batch_size=batch_size, device=device)

    # 9. Live Baseline: Dense MLP
    eval_mlp = train_and_evaluate_baseline("MLP", train_ds, test_ds, epochs=finetune_epochs, batch_size=batch_size, device=device)

    # 10. Live Baseline: Random Forest
    eval_rf = train_and_evaluate_baseline("RANDOM_FOREST", train_ds, test_ds)

    benchmark_table = [
        {
            "Experiment": "DAIR-V2X (VIC+RSU Fusion)",
            "Model": "V2X-BERT (Pre-trained)",
            "Precision Format": "FP32 (4.22 MB)",
            "Accuracy (%)": f"{eval_v2x_bert['accuracy']:.2f}",
            "Precision (%)": f"{eval_v2x_bert['precision']:.2f}",
            "Recall (%)": f"{eval_v2x_bert['recall']:.2f}",
            "F1-Score (%)": f"{eval_v2x_bert['f1']:.2f}",
            "AUC-ROC": f"{eval_v2x_bert['auc']:.4f}",
            "Mean Latency (us)": f"{eval_v2x_bert['latency_us']['mean']:.2f}"
        },
        {
            "Experiment": "DAIR-V2X (VIC+RSU Fusion)",
            "Model": "V2X-BERT (INT8 Quantized)",
            "Precision Format": "INT8 (1.06 MB)",
            "Accuracy (%)": f"{eval_quant['accuracy']:.2f}",
            "Precision (%)": f"{eval_quant['precision']:.2f}",
            "Recall (%)": f"{eval_quant['recall']:.2f}",
            "F1-Score (%)": f"{eval_quant['f1']:.2f}",
            "AUC-ROC": f"{eval_quant['auc']:.4f}",
            "Mean Latency (us)": f"{eval_quant['latency_us']['mean']:.2f}"
        },
        {
            "Experiment": "DAIR-V2X (VIC+RSU Fusion)",
            "Model": "V2X-BERT (No Pre-train)",
            "Precision Format": "FP32 (4.22 MB)",
            "Accuracy (%)": f"{eval_untrained['accuracy']:.2f}",
            "Precision (%)": f"{eval_untrained['precision']:.2f}",
            "Recall (%)": f"{eval_untrained['recall']:.2f}",
            "F1-Score (%)": f"{eval_untrained['f1']:.2f}",
            "AUC-ROC": f"{eval_untrained['auc']:.4f}",
            "Mean Latency (us)": f"{eval_untrained['latency_us']['mean']:.2f}"
        },
        {
            "Experiment": "DAIR-V2X (VIC+RSU Fusion)",
            "Model": "Standard GRU Baseline",
            "Precision Format": "FP32 (2.10 MB)",
            "Accuracy (%)": f"{eval_gru['accuracy']:.2f}",
            "Precision (%)": f"{eval_gru['precision']:.2f}",
            "Recall (%)": f"{eval_gru['recall']:.2f}",
            "F1-Score (%)": f"{eval_gru['f1']:.2f}",
            "AUC-ROC": f"{eval_gru['auc']:.4f}",
            "Mean Latency (us)": f"{eval_gru['latency_us']['mean']:.2f}"
        },
        {
            "Experiment": "DAIR-V2X (VIC+RSU Fusion)",
            "Model": "Standard LSTM Baseline",
            "Precision Format": "FP32 (2.80 MB)",
            "Accuracy (%)": f"{eval_lstm['accuracy']:.2f}",
            "Precision (%)": f"{eval_lstm['precision']:.2f}",
            "Recall (%)": f"{eval_lstm['recall']:.2f}",
            "F1-Score (%)": f"{eval_lstm['f1']:.2f}",
            "AUC-ROC": f"{eval_lstm['auc']:.4f}",
            "Mean Latency (us)": f"{eval_lstm['latency_us']['mean']:.2f}"
        },
        {
            "Experiment": "DAIR-V2X (VIC+RSU Fusion)",
            "Model": "Random Forest Baseline",
            "Precision Format": "CPU Ensemble",
            "Accuracy (%)": f"{eval_rf['accuracy']:.2f}",
            "Precision (%)": f"{eval_rf['precision']:.2f}",
            "Recall (%)": f"{eval_rf['recall']:.2f}",
            "F1-Score (%)": f"{eval_rf['f1']:.2f}",
            "AUC-ROC": f"{eval_rf['auc']:.4f}",
            "Mean Latency (us)": f"{eval_rf['latency_us']['mean']:.2f}"
        },
        {
            "Experiment": "DAIR-V2X (VIC+RSU Fusion)",
            "Model": "Dense MLP Baseline",
            "Precision Format": "FP32 (0.45 MB)",
            "Accuracy (%)": f"{eval_mlp['accuracy']:.2f}",
            "Precision (%)": f"{eval_mlp['precision']:.2f}",
            "Recall (%)": f"{eval_mlp['recall']:.2f}",
            "F1-Score (%)": f"{eval_mlp['f1']:.2f}",
            "AUC-ROC": f"{eval_mlp['auc']:.4f}",
            "Mean Latency (us)": f"{eval_mlp['latency_us']['mean']:.2f}"
        }
    ]

    # Save CSV
    csv_path = os.path.join(output_dir, "Table2_DAIR_V2X_Cooperative_Benchmark.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=benchmark_table[0].keys())
        writer.writeheader()
        writer.writerows(benchmark_table)
    print(f"\n[DAIR-V2X] Saved CSV Benchmark: {csv_path}")

    # Save JSON
    json_path = os.path.join(output_dir, "dair_v2x_master_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "DAIR-V2X Cooperative Vehicle-Infrastructure Telemetry",
            "v2x_bert_results": eval_v2x_bert,
            "int8_results": eval_quant,
            "untrained_ablation": eval_untrained,
            "gru_results": eval_gru,
            "lstm_results": eval_lstm,
            "rf_results": eval_rf,
            "mlp_results": eval_mlp,
            "benchmark_summary": benchmark_table
        }, f, indent=2)
    print(f"[DAIR-V2X] Saved Master JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAIR-V2X Cooperative Experiment Runner")
    parser.add_argument("--samples", type=int, default=2000, help="Number of multi-message sequences")
    parser.add_argument("--pretrain-epochs", type=int, default=2, help="Pre-training epochs")
    parser.add_argument("--finetune-epochs", type=int, default=2, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    run_dair_v2x_experiment(
        samples=args.samples,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size
    )
