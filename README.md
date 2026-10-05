<div align="center">

# 🚗 V2X-BERT: Standards-Aware Foundational Transformer for Connected Vehicle Telemetry & Masked Telemetry Modeling

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![IEEE Transactions](https://img.shields.io/badge/IEEE%20T--ITS-Submitted%202026-00629B?style=for-the-badge&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Standards: SAE J2735](https://img.shields.io/badge/Standards-SAE%20J2735%20%7C%20ETSI%20CAM-blueviolet?style=for-the-badge)]()
[![Hardware: Edge OBU](https://img.shields.io/badge/Edge%20OBU-0.35ms%20Latency-success?style=for-the-badge)]()

<p align="center">
  <b>The First Foundational Bidirectional Transformer Pre-Trained on Structured International V2X Communication Standards (SAE J2735 & ETSI EN 302 637-2) for Real-Time Zero-Trust Vehicular Cybersecurity.</b>
</p>

[Key Features](#-key-features) •
[Architecture](#-architectural-pipeline) •
[Standards Grammar](#-standards-aware-tokenization) •
[Benchmark Results](#-empirical-benchmark-results) •
[Visualizations](#-publication-figures--visualizations) •
[Quickstart](#-quickstart--reproduction) •
[Citation](#-citation)

</div>

---

## 📌 Executive Overview

Connected and Automated Vehicles (CAVs) exchange high-frequency cooperative telemetry defined by rigorous international protocols—**SAE J2735** (Basic Safety Messages, Signal Phase and Timing, MAP) in North America and **ETSI EN 302 637-2** (Cooperative Awareness Messages, Decentralized Environmental Notifications) in Europe.

Existing Machine Learning and Deep Learning models for vehicular misbehavior detection treat telemetry streams as **unstructured flat numerical vectors**, completely discarding:
1. **Hierarchical Protocol Grammars**: Loss of ASN.1 field constraints, quantized kinematic resolutions, and safety-critical bitmasks.
2. **Cross-Message Spatial-Temporal Causality**: Inability to reason across heterogeneous message types (e.g., validating a vehicle's BSM acceleration against a traffic light's SPaT red phase).
3. **Edge Deployment Viability**: Massive 7B Large Language Models (LLMs) take $>500\,\text{ms}$, violating the automotive safety-critical response deadline ($<10\,\text{ms}$).

### 💡 The V2X-BERT Solution:
**`V2X-BERT`** is a compact **1.8 Million-Parameter** domain-specific Transformer pre-trained via self-supervised **Masked Telemetry Modeling (MTM)** on structured multi-message vehicular streams. It achieves **$0.35\,\text{ms}$ sub-millisecond inference** on automotive On-Board Units (OBUs) with a **$<1\,\text{MB}$ memory footprint** and delivers state-of-the-art **$97.85\%$ accuracy** and **$0.988$ AUC-ROC** on the authentic full-scale VeReMi benchmark ($7.12\,\text{GB}$).

---

## ✨ Key Features

* **🌐 Cross-Standard Schema Alignment**: Natively ingests and aligns American **SAE J2735 (BSM, SPaT, MAP, PSM)** and European **ETSI (CAM, DENM, CDD)** into a unified semantic embedding space.
* **🔤 Schema-Aware Discrete Tokenizer ($|\mathcal{V}| = 1,024$)**: Structured quantization of continuous velocity, non-linear acceleration bins, angular heading sectors, discrete protocol flags, and relative spatial grids.
* **🎭 Self-Supervised Masked Telemetry Modeling (MTM)**: Pre-trains on multi-vehicle dialogues by reconstructing randomly corrupted telemetry fields ($15\%$ masking), forcing the model to learn physical kinematic laws and cooperative traffic dynamics.
* **⚡ Edge-Native Automotive Efficiency**: Custom 4-layer Pre-LN Transformer encoder running in **$0.35\,\text{ms}$** ($<350\,\mu\text{s}$) on embedded CPU microcontrollers.
* **🛡️ Zero-Trust Misbehavior Detection**: Detects sophisticated GPS spoofing, constant speed falsification, sudden deceleration, Sybil nodes, and data replay attacks.

---

## 🏗️ Architectural Pipeline

```
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   Heterogeneous V2X Telemetry Streams                            │
 │   • SAE J2735: BSM (Kinematics) | SPaT (Traffic Lights) | MAP (Topology) | PSM (Pedestrians)      │
 │   • ETSI EN 302 637: CAM (Cooperative Awareness) | DENM (Hazard Alerts) | CPM (LiDAR Perception) │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                ASN.1 Bit-Level Standards Parser                                  │
 │         Extracts exact physical resolution units (0.02 m/s speed, 0.0125° heading, bitmasks)     │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                          Schema-Aware V2X Tokenizer (|V| = 1,024 Tokens)                         │
 │   [CLS] [BSM] [SPD_85] [ACC_NEG_3] [HDG_90] [GRID_X+05] [BRAKE_ABS] [SPAT] [RED] [COUNT_8S] ... │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                             Edge-V2X-BERT Transformer Encoder Stack                              │
 │            • 4 Transformer Blocks | Hidden Dim d_model = 128 | 4 Attention Heads | d_ff = 512    │
 │            • Pre-LayerNorm Architecture | Positional & Standards Type Embeddings                 │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                      Downstream Task Heads                                       │
 │  1. Masked Telemetry Head (MTM)  │  2. Zero-Trust IDS Head (VeReMi)  │  3. Cross-Standard Align  │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

<div align="center">
  <img src="results/Fig1_V2X_BERT_Architecture_and_Tokenization.png" alt="Figure 1: V2X-BERT Architecture" width="95%"/>
  <p><i>Figure 1: Complete V2X-BERT architectural pipeline from raw ASN.1 message streams to downstream task heads.</i></p>
</div>

---

## 🔠 Standards-Aware Tokenization

Instead of arbitrary natural language tokens, `V2X-BERT` discretizes vehicular telemetry into **$1,024$ domain-specific semantic tokens**:

| Token Sub-Range | Token IDs | Category | Description & Quantization Scheme | Example Tokens |
| :--- | :---: | :--- | :--- | :--- |
| **Special & Headers** | `0` – `31` | Message Types | Protocol identifiers & BERT control tokens | `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`, `[BSM]`, `[CAM]`, `[SPAT]`, `[DENM]`, `[PSM]` |
| **Kinematic Speed** | `32` – `127` | Speed Dynamics | $0\text{ to } 180\,\text{km/h}$ in $1.875\,\text{km/h}$ discrete bins | `[SPD_0]`, `[SPD_30]`, `[SPD_60]`, `[SPD_120]` |
| **Acceleration / Jerk** | `128` – `255`| Acceleration | Non-linear log-linear bins ($-12.0\text{ to }+8.0\,\text{m/s}^2$) | `[ACC_EMERGENCY_BRAKE]`, `[ACC_SMOOTH]`, `[ACC_LAUNCH]` |
| **Heading & Yaw** | `256` – `327`| Heading Angle | $72$ discrete angular sectors ($5^\circ$ resolution) | `[HDG_0_5]`, `[HDG_90_95]`, `[HDG_180_185]` |
| **Status & Bitmasks** | `328` – `399`| Vehicle Flags | Binary protocol bitmasks | `[BRAKE_ACTIVE]`, `[ABS_ACTIVE]`, `[TCS_ACTIVE]`, `[HAZARD_LIGHTS]` |
| **SPaT Signal States**| `400` – `449`| Infrastructure | Traffic signal phases & remaining countdowns | `[SPAT_RED]`, `[SPAT_YELLOW]`, `[SPAT_GREEN]`, `[COUNTDOWN_5S]` |
| **DENM Event Codes** | `450` – `499`| Safety Alerts | Event-driven hazard notifications | `[HAZARD_OBSTACLE]`, `[BLACK_ICE]`, `[ACCIDENT_AHEAD]` |
| **Spatial Grid Delta**| `500` – `1023`| Topologies | 524 polar Euclidean relative neighborhood bins | `[GRID_DIST_10M_ANG_45]`, `[PLATOON_GAP_TIGHT]` |

---

## 📊 Empirical Benchmark Results

Evaluated on the full **$7.12\,\text{GB}$ authentic VeReMi benchmark** ($250,000+$ BSM transactions) across 19 cyberattack classifications:

### Table 1: Comparative Performance on VeReMi Benchmark

| Model Architecture | Pre-training Paradigm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC | Inference Latency | Memory Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V2X-BERT` (Proposed)** | **Self-Supervised MTM** | **$\mathbf{97.85\%}$** | **$\mathbf{98.10\%}$** | **$\mathbf{97.35\%}$** | **$\mathbf{97.72\%}$** | **$\mathbf{0.988}$** | **$\mathbf{0.35\,\text{ms}}$** | **$<1.0\,\text{MB}$** |
| `V2X-BERT` (Untrained Ablation) | None (Random Init) | $94.10\%$ | $93.80\%$ | $93.90\%$ | $93.85\%$ | $0.942$ | $0.35\,\text{ms}$ | $<1.0\,\text{MB}$ |
| Standard LSTM (2 Layers) | None (Supervised) | $91.45\%$ | $90.80\%$ | $89.20\%$ | $89.99\%$ | $0.915$ | $0.48\,\text{ms}$ | $3.4\,\text{MB}$ |
| Dense MLP Baseline | None (Supervised) | $86.10\%$ | $85.20\%$ | $84.10\%$ | $84.65\%$ | $0.862$ | $0.12\,\text{ms}$ | $0.6\,\text{MB}$ |
| MistralBSM (7B LLM Prompting) | Pre-trained LLM | $95.20\%$ | $94.10\%$ | $94.50\%$ | $94.30\%$ | $0.961$ | $480.00\,\text{ms}$ *(Violates OBU)* | $>14.0\,\text{GB}$ |

---

## 📈 Publication Figures & Visualizations

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="results/Fig2_Masked_Telemetry_Pretraining_Loss.png" width="100%"/><br/>
      <b>Figure 2:</b> Self-Supervised Masked Telemetry Modeling (MTM) Pre-training Loss & Perplexity Trajectory.
    </td>
    <td align="center" width="50%">
      <img src="results/Fig3_Attention_Heads_Semantic_Matrix.png" width="100%"/><br/>
      <b>Figure 3:</b> Cross-Message Self-Attention Map (BSM Kinematics $\leftrightarrow$ SPaT Infrastructure Phase).
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="results/Fig4_Downstream_Misbehavior_ROC_and_PR.png" width="100%"/><br/>
      <b>Figure 4:</b> Receiver Operating Characteristic (ROC) on VeReMi Zero-Trust Attacks.
    </td>
    <td align="center" width="50%">
      <img src="results/Fig5_Attack_Type_Breakdown_Confusion_Matrix.png" width="100%"/><br/>
      <b>Figure 5:</b> Multi-Class Attack Breakdown Confusion Matrix (%) across VeReMi Categories.
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="results/Fig6_Edge_OBU_Latency_and_Parameter_Scaling.png" width="70%"/><br/>
      <b>Figure 6:</b> Model Complexity vs. Automotive Edge OBU Real-Time Latency Trade-off.
    </td>
  </tr>
</table>

---

## 🔬 Mathematical Formulation

### 1. Multi-Head Self-Attention
For token representations $H \in \mathbb{R}^{T \times d_{\text{model}}}$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 2. Self-Supervised Masked Telemetry Loss
During pre-training, $15\%$ of active telemetry tokens $\mathcal{M}$ are masked. The model minimizes the cross-entropy loss over masked positions:
$$\mathcal{L}_{\text{MTM}}(\theta) = - \sum_{i \in \mathcal{M}} \log P(x_i = \tilde{x}_i \mid \mathbf{x}_{\backslash i}; \theta)$$

### 3. Cross-Standard Alignment Loss
Aligns normalized embeddings of equivalent SAE J2735 ($z_{\text{SAE}}$) and ETSI ($z_{\text{ETSI}}$) frames via cosine similarity:
$$\mathcal{L}_{\text{Align}} = 1 - \frac{z_{\text{SAE}} \cdot z_{\text{ETSI}}}{\|z_{\text{SAE}}\| \|z_{\text{ETSI}}\|}$$

---

## 🛠️ Quickstart & Reproduction

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/umertanveer25/V2X-BERT.git
cd V2X-BERT
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Executes data ingestion from the authentic 7.12 GB VeReMi archive, Masked Telemetry Pre-training, and downstream Zero-Trust evaluation:
```bash
python experiments/run_v2x_bert_pipeline.py
```

### 3. Generate 300 DPI Publication Figures
```bash
python experiments/generate_v2x_bert_figures.py
```

---

## 📂 Repository Structure

```tree
V2X-BERT/
├── schemas/                      # Official ASN.1 Standards Definitions
│   ├── SAE_J2735_2020.asn        # SAE J2735 (BSM, SPaT, MAP, PSM)
│   └── ETSI_CAM_CDD_DENM.asn     # ETSI (CAM, DENM, Common Data Dictionary)
├── src/
│   ├── v2x_tokenizer.py          # Schema-Aware multi-message tokenizer (|V| = 1024)
│   ├── data_loader.py            # Zero-RAM chunked streaming loader from VeReMi archive
│   ├── v2x_bert_model.py         # 1.8M Parameter Edge-Native Transformer Architecture
│   ├── pretrain_engine.py        # Self-Supervised Masked Telemetry Pre-training (MTM)
│   └── evaluate_downstream.py    # Zero-Trust misbehavior evaluation & cross-standard transfer
├── experiments/
│   ├── run_v2x_bert_pipeline.py  # Master execution orchestrator & baseline comparator
│   └── generate_v2x_bert_figures.py # 300 DPI publication figure generator (Figs 1-6)
├── results/                      # Output CSV benchmark tables, JSON, and PNG figures
├── paper/                        # Complete IEEE Transactions LaTeX manuscript & bibliography
├── requirements.txt              # Environment dependencies
└── README.md                     # Documentation
```

---

## 📜 Citation

If you build upon `V2X-BERT` in your research, please cite our manuscript:

```bibtex
@article{tanveer2026v2xbert,
  title={{V2X-BERT}: A Domain-Specific Transformer for Standards-Aware Vehicular Communication and Masked Telemetry Modeling},
  author={Tanveer, Muhammad Umer},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={XX},
  number={X},
  pages={1--14},
  year={2026},
  publisher={IEEE}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
