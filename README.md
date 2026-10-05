# V2X-BERT: Compact Standards-Aware Bidirectional Transformer for Vehicular Telemetry

[![PyPI Version](https://img.shields.io/badge/PyPI-v1.0.0-blue.svg)](https://pypi.org/project/v2x-bert/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameters](https://img.shields.io/badge/Parameters-1.11M-orange.svg)](https://github.com/umertanveer25/V2X-BERT)
[![Memory FP32/INT8](https://img.shields.io/badge/Memory-4.43MB%20%2F%201.11MB-purple.svg)](https://github.com/umertanveer25/V2X-BERT)

**V2X-BERT** is a domain-specific, compact bidirectional Transformer architecture designed for multi-message vehicular communications. By tokenizing structured **SAE J2735:2020** (BSM, SPaT, MAP, PSM) and **ETSI EN 302 637-2** (CAM, DENM, CDD) protocols into a discrete semantic vocabulary ($|\mathcal{V}| = 1,024$), V2X-BERT captures the temporal grammar, physical kinematic invariants, and cross-standard semantic alignment of Connected & Automated Vehicles (CAVs).

---

## 🌟 Key Architectural Features

- **Standards-Informed Schema-Aware Tokenizer ($|\mathcal{V}| = 1,024$):**
  - **Speed ($[32, 127]$):** 96 linear velocity quantization bins ($0$ to $180\,\text{km/h}$).
  - **Acceleration ($[128, 255]$):** 128 linear acceleration bins ($-12.0$ to $+8.0\,\text{m/s}^2$).
  - **Heading ($[256, 327]$):** 72 angular bins ($5^\circ$ resolution).
  - **Status & Bitmasks ($[328, 399]$):** Brake, ABS, TCS, SCS, Hazard light flags.
  - **Signal Phase & Timing ($[400, 449]$):** SPaT states and countdown countdowns.
  - **Hazard Event Causes ($[450, 499]$):** DENM cause codes.
  - **Polar Spatial Neighborhood ($[500, 1011]$):** 512 discrete spatial grid bins ($16$ distance tiers $\times 32$ angular sectors).

- **Ultra-Compact Edge-Native Architecture:**
  - **Trainable Parameters:** Exactly **$1,106,882$ parameters** ($\approx 1.11\,\text{M}$).
  - **Memory Footprint:** **$4.43\,\text{MB}$** (FP32) / **$1.11\,\text{MB}$** (INT8 Quantized).
  - **Edge Latency:** Sub-millisecond inference per multi-message sequence on automotive embedded CPUs.

- **Joint Pre-Training Objectives:**
  - **Masked Telemetry Modeling (MTM):** Canonical BERT 80/10/10 masking reconstructing corrupted kinematic slots.
  - **Cross-Standard Latent Alignment:** InfoNCE contrastive projection aligning SAE J2735 and ETSI representations.

- **Leakage-Free Validation Protocol:**
  - Evaluated under strictly **scenario-disjoint and sender-disjoint** splits on standardized VeReMi message logs.

---

## 📊 Benchmark & Performance Summary

| Model Architecture | Pre-trained | Precision | Accuracy (%) | F1-Score (%) | AUC-ROC | Parameters | Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2X-BERT (Pre-trained)** | **Yes (MTM + Align)** | **FP32** | **97.85** | **97.72** | **0.988** | **1.11M** | **4.43 MB** |
| **V2X-BERT (INT8 Quantized)** | **Yes (MTM + Align)** | **INT8** | **97.60** | **97.45** | **0.984** | **1.11M** | **1.11 MB** |
| V2X-BERT (Untrained Ablation) | No | FP32 | 94.10 | 93.85 | 0.942 | 1.11M | 4.43 MB |
| Standard LSTM Baseline | No | FP32 | 91.45 | 89.99 | 0.932 | 2.80M | 11.2 MB |
| Dense MLP Baseline | No | FP32 | 86.10 | 84.65 | 0.884 | 0.45M | 1.80 MB |

---

## 🚀 Quick Start & Installation

### Option 1: Install via pip
```bash
pip install v2x-bert
```

### Option 2: Install from Source
```bash
git clone https://github.com/umertanveer25/V2X-BERT.git
cd V2X-BERT
pip install -e .
```

### Python Usage Example
```python
import torch
import v2x_bert
from v2x_bert import V2XTokenizer, load_model

# 1. Initialize Tokenizer and Model (1.11M params)
tokenizer = V2XTokenizer()
model = load_model(pretrained=True, device="cpu")

# 2. Tokenize real vehicular telemetry
bsm_tokens = tokenizer.encode_bsm(
    speed=28.5,      # km/h
    accel=-3.2,      # m/s^2
    heading=180.0,   # degrees
    dx=12.5, dy=2.0, # meters relative to ego vehicle
    brake=1, abs_flag=0
)

# 3. Package into BERT input sequence
input_ids, attention_mask = tokenizer.encode_sequence([bsm_tokens], max_len=64)

# 4. Downstream Zero-Trust Misbehavior Classification
with torch.no_grad():
    logits, attentions = model.forward_classify(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    prob_malicious = torch.softmax(logits, dim=-1)[0, 1].item()

print(f"Malicious Telemetry Probability: {prob_malicious:.4f}")
```

---

## 📂 Repository Structure

```text
V2X-BERT/
├── v2x_bert/                 # Core installable Python package
│   ├── __init__.py           # Package exports & load_model factory
│   ├── tokenizer.py          # Standards-informed discrete tokenizer
│   ├── model.py              # EdgeV2XBERT architecture (1.11M params)
│   ├── pretrain.py           # Joint MTM & Alignment pre-training engine
│   ├── evaluate.py           # Downstream evaluation engine (zero leakage)
│   └── data.py               # Disjoint dataset loader
├── schemas/                  # Official ASN.1 Schema definitions
│   ├── SAE_J2735_2020.asn    # SAE J2735:2020 message set
│   └── ETSI_CAM_CDD_DENM.asn # ETSI EN 302 637-2 specifications
├── experiments/              # Full experiment & figure scripts
│   ├── run_v2x_bert_pipeline.py
│   └── generate_v2x_bert_figures.py
├── results/                  # Generated CSV tables, JSON logs, & 300 DPI figures
├── paper/                    # IEEE Transactions LaTeX manuscript
├── pyproject.toml            # PEP 621 Standard Build Specification
└── README.md
```

---

## 📜 Citation

If you use V2X-BERT in your research, please cite:

```bibtex
@article{tanveer2026v2xbert,
  author={Tanveer, Muhammad Umer and Salam, Abdu},
  title={{V2X-BERT}: A Compact Standards-Aware Bidirectional Transformer for Vehicular Telemetry and Misbehavior Detection},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026}
}
```
