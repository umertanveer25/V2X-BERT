"""
Comprehensive Unit Test Suite for V2X-BERT Package.
Verifies:
  1. Standards Tokenizer Vocab Ranges
  2. Exact Parameter Counts & Memory Footprints (1.11M params)
  3. Wire-Level ASN.1 UPER / DER Binary Serialization & Deserialization
  4. Real VeReMi Dataset Ingestion & Zero-Leakage Disjoint Splits
  5. Multi-Task Cross-Standard InfoNCE Alignment Loss
  6. DAIR-V2X Cooperative Vehicle-Infrastructure Dataset Ingestion
  7. Edge OBU INT8 Dynamic Quantization
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2x_bert import V2XTokenizer, EdgeV2XBERT, load_model
from src.data_loader import load_real_veremi_dataset, V2XDataset, split_by_scenario_disjoint, split_by_sender_disjoint
from src.dair_v2x_loader import load_dair_v2x_dataset


class TestV2XBERT(unittest.TestCase):

    def test_tokenizer_ranges(self):
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

    def test_model_parameters(self):
        """Verify model parameter count (~1.11M) and memory budgets."""
        model = load_model(pretrained=False, device="cpu")
        num_params = model.count_parameters()
        self.assertTrue(1_100_000 <= num_params <= 1_120_000, f"Got {num_params}")

        mem = model.get_memory_footprint()
        self.assertTrue(mem["fp32_mb"] < 5.0)
        self.assertTrue(mem["int8_mb"] < 1.5)

    def test_model_forward(self):
        """Verify forward pass on pre-training and classification heads."""
        model = load_model(pretrained=False, device="cpu")
        x = torch.randint(0, 1024, (2, 64))
        mask = torch.ones((2, 64))

        # Pretrain head
        logits, z_proj = model.forward_pretrain(x, attention_mask=mask)
        self.assertEqual(logits.shape, (2, 64, 1024))
        self.assertEqual(z_proj.shape, (2, 64))

        # Classification head
        cls_logits, attentions = model.forward_classify(x, attention_mask=mask)
        self.assertEqual(cls_logits.shape, (2, 2))
        self.assertEqual(len(attentions), 4)

    def test_sae_j2735_uper_wire_codec(self):
        """Verify authentic wire-level SAE J2735:2020 BSM binary packing and unpacking."""
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

    def test_etsi_cam_uper_wire_codec(self):
        """Verify authentic wire-level ETSI EN 302 637-2 CAM binary packing and unpacking."""
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

    def test_sae_etsi_cross_standard_alignment_loss(self):
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

    def test_real_veremi_loader_and_disjoint_splits(self):
        """Verify real VeReMi dataset loader and formal disjoint verification."""
        seqs, masks, labels, atks, snders, scens = load_real_veremi_dataset(max_archives=5)
        full_ds = V2XDataset(seqs, masks, labels, atks, snders, scens)

        # 1. Scenario-Disjoint Split
        train_sc, test_sc = split_by_scenario_disjoint(full_ds, test_ratio=0.20)
        self.assertEqual(len(set(train_sc.scenario_ids).intersection(set(test_sc.scenario_ids))), 0)

        # 2. Sender-Disjoint Split
        train_sn, test_sn = split_by_sender_disjoint(full_ds, test_ratio=0.20)
        self.assertEqual(len(set(train_sn.sender_ids).intersection(set(test_sn.sender_ids))), 0)

    def test_dair_v2x_loader(self):
        """Verify DAIR-V2X cooperative vehicle-infrastructure loader."""
        train_ds, test_ds = load_dair_v2x_dataset(num_samples=200, test_ratio=0.20)
        self.assertEqual(len(train_ds), 160)
        self.assertEqual(len(test_ds), 40)
        self.assertEqual(train_ds[0][0].shape[0], 64)

    def test_int8_quantization(self):
        """Verify dynamic INT8 CPU quantization forward execution."""
        model = load_model(pretrained=False, device="cpu")
        quant_model = model.quantize_int8()
        x = torch.randint(0, 1024, (2, 64))
        mask = torch.ones((2, 64))
        cls_logits, _ = quant_model.forward_classify(x, attention_mask=mask)
        self.assertEqual(cls_logits.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
