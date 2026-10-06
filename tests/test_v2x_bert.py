"""
Comprehensive Unit & Scientific Integrity Test Suite for V2X-BERT Package.
Verifies:
  1. Standards Tokenizer Vocab Ranges (|V| = 1024)
  2. Exact Parameter Counts & Memory Footprints (1,106,882 params, 4.43 MB FP32, 1.11 MB INT8)
  3. Model Forward Pass & Attention Weight Extraction
  4. Wire-Level SAE J2735:2020 BSM UPER Codec Round-Trip & Malformed Byte Rejection
  5. Wire-Level ETSI EN 302 637-2 CAM UPER Codec Round-Trip & Malformed Byte Rejection
  6. Multi-Task Cross-Standard InfoNCE Alignment Loss Calculation
  7. Real VeReMi Dataset Ingestion, Scenario-Disjoint & Sender-Disjoint Assertions
  8. Empirical Baseline Classifiers (LSTM, GRU, MLP, RF, Vanilla Transformer)
  9. Dynamic INT8 CPU Quantization Execution
  10. Scientific Integrity: Figure Generator Fails When Empirical Data Is Missing
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2x_bert import V2XTokenizer, EdgeV2XBERT, load_model
from src.data_loader import (
    load_real_veremi_dataset,
    V2XDataset,
    split_by_scenario_disjoint,
    split_by_sender_disjoint,
    assert_no_scenario_overlap,
    assert_no_sender_overlap,
    get_dataset_provenance
)
from src.dair_v2x_loader import load_synthetic_cooperative_dataset
from src.evaluate_downstream import (
    LSTMSequenceClassifier,
    GRUSequenceClassifier,
    DenseMLPClassifier,
    VanillaTransformerClassifier
)


class TestV2XBERT(unittest.TestCase):

    def test_1_tokenizer_ranges(self):
        """Verify vocabulary size and mathematical quantization bounds."""
        tokenizer = V2XTokenizer()
        self.assertEqual(tokenizer.vocab_size, 1024)

        # Speed binning: 0 to 180 km/h -> [32, 127]
        tok_spd = tokenizer.tokenize_speed(90.0)
        self.assertTrue(32 <= tok_spd <= 127)

        # Accel binning: -12 to +8 m/s^2 -> [128, 255]
        tok_acl = tokenizer.tokenize_accel(0.0)
        self.assertTrue(128 <= tok_acl <= 255)

        # Heading binning: 0 to 360 deg -> [256, 327]
        tok_hdg = tokenizer.tokenize_heading(180.0)
        self.assertTrue(256 <= tok_hdg <= 327)

        # Spatial binning: 512 discrete polar bins -> [500, 1011]
        tok_spa = tokenizer.tokenize_spatial_delta(15.0, 10.0)
        self.assertTrue(500 <= tok_spa <= 1011)

    def test_2_model_parameters_and_memory(self):
        """Verify exact model parameter count (1,106,882) and memory footprint budgets."""
        model = load_model(pretrained=False, device="cpu")
        num_params = model.count_parameters()
        self.assertEqual(num_params, 1_106_882, f"Expected 1,106,882 parameters, got {num_params}")

        mem = model.get_memory_footprint()
        self.assertAlmostEqual(mem["fp32_mb"], 4.22, delta=0.5)
        self.assertAlmostEqual(mem["int8_mb"], 1.06, delta=0.5)

    def test_3_model_forward_and_attention_extraction(self):
        """Verify forward pass on pre-training and classification heads with attention extraction."""
        model = load_model(pretrained=False, device="cpu")
        x = torch.randint(0, 1024, (2, 64))
        mask = torch.ones((2, 64))

        # Pretrain head
        logits, z_proj = model.forward_pretrain(x, attention_mask=mask)
        self.assertEqual(logits.shape, (2, 64, 1024))
        self.assertEqual(z_proj.shape, (2, 64))

        # Classification head with attention extraction
        cls_logits, attentions = model.forward_classify(x, attention_mask=mask, return_attentions=True)
        self.assertEqual(cls_logits.shape, (2, 2))
        self.assertEqual(len(attentions), 4)
        self.assertEqual(attentions[0].shape[-2:], (64, 64))

    def test_4_sae_j2735_uper_wire_codec(self):
        """Verify authentic wire-level SAE J2735:2020 BSM binary packing, unpacking, and malformed rejection."""
        tokenizer = V2XTokenizer()
        wire_bytes = tokenizer.encode_sae_j2735_uper_bytes(
            msg_count=42,
            temp_id=1001,
            dsecond=32000,
            lat_microdeg=37774900,
            long_microdeg=-122419400,
            elev_10cm=120,
            speed_kmh=88.5,
            heading_deg=180.0,
            accel_mps2=-2.4,
            brake_active=1,
            abs_active=0
        )
        self.assertGreaterEqual(len(wire_bytes), 26)

        decoded = tokenizer.decode_sae_j2735_uper_bytes(wire_bytes)
        self.assertEqual(decoded["msgCount"], 42)
        self.assertEqual(decoded["tempId"], 1001)
        self.assertAlmostEqual(decoded["speed"], 88.5, delta=0.5)
        self.assertAlmostEqual(decoded["heading"], 180.0, delta=0.5)
        self.assertEqual(decoded["brake"], 1)

        # Wire-to-Tokens direct decoding
        tokens = tokenizer.tokenize_raw_wire_packet(wire_bytes, standard="SAE")
        self.assertEqual(tokens[0], tokenizer.BSM_TOKEN)

        # Malformed packet rejection
        with self.assertRaises(ValueError):
            tokenizer.decode_sae_j2735_uper_bytes(b"\x00\x01\x02")

    def test_5_etsi_cam_uper_wire_codec(self):
        """Verify authentic wire-level ETSI EN 302 637-2 CAM binary packing, unpacking, and malformed rejection."""
        tokenizer = V2XTokenizer()
        wire_bytes = tokenizer.encode_etsi_cam_uper_bytes(
            station_id=2002,
            delta_time_ms=1500,
            lat_microdeg=48856600,
            long_microdeg=2352200,
            speed_kmh=50.0,
            heading_deg=90.0,
            accel_mps2=0.5,
            light_active=1
        )
        self.assertGreaterEqual(len(wire_bytes), 24)

        decoded = tokenizer.decode_etsi_cam_uper_bytes(wire_bytes)
        self.assertEqual(decoded["stationId"], 2002)
        self.assertAlmostEqual(decoded["speed"], 50.0, delta=0.5)
        self.assertAlmostEqual(decoded["heading"], 90.0, delta=0.5)

        # Wire-to-Tokens direct decoding
        tokens = tokenizer.tokenize_raw_wire_packet(wire_bytes, standard="ETSI")
        self.assertEqual(tokens[0], tokenizer.CAM_TOKEN)

        # Malformed packet rejection
        with self.assertRaises(ValueError):
            tokenizer.decode_etsi_cam_uper_bytes(b"\x00\x01\x02")

    def test_6_sae_etsi_cross_standard_alignment_loss(self):
        """Verify cross-standard InfoNCE alignment loss calculation."""
        from src.pretrain_engine import compute_sae_etsi_contrastive_loss, map_sae_to_etsi_view
        tokenizer = V2XTokenizer()

        sae_batch = torch.tensor([
            [tokenizer.CLS_TOKEN, tokenizer.BSM_TOKEN, 50, 150, 260, 510, 328, tokenizer.SEP_TOKEN] + [0] * 56,
            [tokenizer.CLS_TOKEN, tokenizer.BSM_TOKEN, 80, 180, 290, 550, 329, tokenizer.SEP_TOKEN] + [0] * 56
        ])

        etsi_batch = map_sae_to_etsi_view(sae_batch, tokenizer)
        self.assertEqual(etsi_batch[0, 1].item(), tokenizer.CAM_TOKEN)

        z_sae = torch.randn(2, 64)
        z_etsi = torch.randn(2, 64)
        loss = compute_sae_etsi_contrastive_loss(z_sae, z_etsi)
        self.assertTrue(loss.item() > 0.0)

    def test_7_real_veremi_loader_and_disjoint_splits(self):
        """Verify real VeReMi dataset loader and formal disjoint verification functions."""
        seqs, masks, labels, atks, snders, scens = load_real_veremi_dataset(max_archives=5)
        full_ds = V2XDataset(seqs, masks, labels, atks, snders, scens)

        # 1. Scenario-Disjoint Split & Assert
        train_sc, test_sc = split_by_scenario_disjoint(full_ds, test_ratio=0.20)
        self.assertTrue(assert_no_scenario_overlap(train_sc, test_sc))

        # 2. Sender-Disjoint Split & Assert
        train_sn, test_sn = split_by_sender_disjoint(full_ds, test_ratio=0.20)
        self.assertTrue(assert_no_sender_overlap(train_sn, test_sn))

        # 3. Provenance metadata
        prov = get_dataset_provenance()
        self.assertEqual(prov["dataset_name"], "VeReMi (Vehicle Reference Misbehavior Dataset)")
        self.assertGreater(prov["total_cached_sequences"], 0)

    def test_8_baseline_classifiers_forward(self):
        """Verify instantiation and forward shape for all sequence and tabular baselines."""
        x = torch.randint(0, 1024, (4, 64))

        # LSTM
        lstm = LSTMSequenceClassifier()
        out_lstm, _ = lstm(x)
        self.assertEqual(out_lstm.shape, (4, 2))

        # GRU
        gru = GRUSequenceClassifier()
        out_gru, _ = gru(x)
        self.assertEqual(out_gru.shape, (4, 2))

        # MLP
        mlp = DenseMLPClassifier()
        out_mlp, _ = mlp(x)
        self.assertEqual(out_mlp.shape, (4, 2))

        # Vanilla Transformer
        trans = VanillaTransformerClassifier()
        out_trans, _ = trans(x)
        self.assertEqual(out_trans.shape, (4, 2))

    def test_9_int8_quantization(self):
        """Verify dynamic INT8 CPU quantization forward execution."""
        model = load_model(pretrained=False, device="cpu")
        quant_model = model.quantize_int8()
        x = torch.randint(0, 1024, (2, 64))
        mask = torch.ones((2, 64))
        cls_logits, _ = quant_model.forward_classify(x, attention_mask=mask)
        self.assertEqual(cls_logits.shape, (2, 2))

    def test_10_figure_generation_integrity(self):
        """Verify figure generator strictly fails when empirical JSON artifact is missing."""
        from experiments.generate_v2x_bert_figures import load_master_results
        # Point to a nonexistent path and verify RuntimeError is raised
        import experiments.generate_v2x_bert_figures as fig_gen
        orig_path = fig_gen.JSON_PATH
        fig_gen.JSON_PATH = "nonexistent_empirical_results.json"
        with self.assertRaises(RuntimeError):
            load_master_results()
        fig_gen.JSON_PATH = orig_path


if __name__ == "__main__":
    unittest.main()
