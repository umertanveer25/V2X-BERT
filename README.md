# V2X-BERT: A Domain-Specific Transformer for Standards-Aware Vehicular Communication and Masked Telemetry Modeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Standards](https://img.shields.io/badge/Standards-SAE%20J2735%20%7C%20ETSI%20CAM-purple.svg)]()
[![Hardware](https://img.shields.io/badge/Edge%20OBU-0.35ms%20Latency-brightgreen.svg)]()

---

## 📌 Executive Abstract & Vision

Connected and Automated Vehicles (CAVs) exchange structured, high-frequency cooperative telemetry defined by international standards—**SAE J2735** (BSM, SPaT, MAP, PSM) in North America and **ETSI EN 302 637-2** (CAM, DENM, CDD) in Europe. Existing misbehavior detection algorithms treat vehicular data as flat, unstructured numerical vectors, discarding the rich spatiotemporal grammar, cross-message causal dependencies, and cross-standard semantic equivalences of connected traffic.

**V2X-BERT** introduces the first foundational domain-specific bidirectional Transformer pre-trained directly on standardized multi-message vehicular communications:
* **Schema-Aware V2X Tokenizer ($|\mathcal{V}| = 1,024$)**: Maps continuous velocities, non-linear acceleration bins, angular heading sectors, discrete protocol flags, and relative spatial grids into a compact dictionary.
* **Self-Supervised Masked Telemetry Modeling (MTM)**: Pre-trains on multi-vehicle dialogues by reconstructing randomly masked telemetry fields ($15\%$ masking probability).
* **Edge-Native Automotive Efficiency**: With only **$1.8\text{M}$ parameters**, V2X-BERT achieves an ultra-low inference latency of **$0.35\,\text{ms}$** ($<1\,\text{MB}$ memory footprint), enabling seamless deployment on automotive On-Board Units (OBUs).
* **Multi-Source Empirical Validation**: Validated on the full **$7.12\,\text{GB}$ VeReMi benchmark** and real-world **DAIR-V2X** cooperative traces, achieving **$97.85\%$ accuracy** and an AUC-ROC of **$0.988$**.

---

## 📂 Repository Structure

```tree
V2X-BERT/
├── schemas/                      # Official ASN.1 Standards Definitions
│   ├── SAE_J2735_2020.asn        # SAE J2735 (BSM, SPaT, MAP, PSM)
│   └── ETSI_CAM_CDD_DENM.asn     # ETSI (CAM, DENM, Common Data Dictionary)
├── src/
│   ├── v2x_tokenizer.py          # Schema-Aware multi-message tokenizer (|V| = 1024)
│   ├── data_loader.py            # Chunked streaming loader from authentic 7.12 GB VeReMi archive
│   ├── v2x_bert_model.py         # 1.8M Parameter Edge-Native Transformer Architecture
│   ├── pretrain_engine.py        # Self-Supervised Masked Telemetry Pre-training (MTM)
│   └── evaluate_downstream.py    # Zero-Trust misbehavior evaluation & cross-standard transfer
├── experiments/
│   ├── run_v2x_bert_pipeline.py  # Master end-to-end execution pipeline
│   └── generate_v2x_bert_figures.py # 300 DPI publication figure generator (Figs 1-6)
├── data/
│   └── dair-v2x/                 # Cloned official DAIR-V2X cooperative dataset
├── results/                      # Generated CSV benchmark tables, JSON, and PNG figures
├── paper/                        # Complete IEEE Transactions LaTeX manuscript & bibliography
├── requirements.txt              # Dependencies
└── README.md                     # Comprehensive documentation
```

---

## 🚀 Quickstart & Pipeline Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/umertanveer25/V2X-BERT.git
cd V2X-BERT
pip install -r requirements.txt
```

### 2. Run Full Pre-training & Downstream Benchmark Pipeline
```bash
python experiments/run_v2x_bert_pipeline.py
```

### 3. Generate 300 DPI Publication Figures
```bash
python experiments/generate_v2x_bert_figures.py
```

---

## 📜 Citation

```bibtex
@article{tanveer2026v2xbert,
  title={V2X-BERT: A Domain-Specific Transformer for Standards-Aware Vehicular Communication and Masked Telemetry Modeling},
  author={Tanveer, Muhammad Umer},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026},
  publisher={IEEE}
}
```
