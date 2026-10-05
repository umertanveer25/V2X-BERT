"""
Unit Tests for V2X-BERT Package.
Verifies tokenization consistency, parameter counts, forward passes, and INT8 quantization.
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2x_bert import V2XTokenizer, EdgeV2XBERT, load_model


class TestV2XBERT(unittest.TestCase):

    def test_tokenizer_ranges(self):
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
        model = load_model(pretrained=False, device="cpu")
        num_params = model.count_parameters()
        
        # Verify exact parameter count (~1.11M)
        self.assertTrue(1_100_000 <= num_params <= 1_120_000, f"Got {num_params}")
        
        mem = model.get_memory_footprint()
        self.assertTrue(mem["fp32_mb"] < 5.0)
        self.assertTrue(mem["int8_mb"] < 1.5)

    def test_model_forward(self):
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

    def test_standards_dictionary_encoding(self):
        tokenizer = V2XTokenizer()
        
        # SAE J2735 BSM Dict
        bsm_dict = {
            "messageId": "BSM",
            "speed": 65.0,
            "accel": -1.5,
            "heading": 90.0,
            "dx": 12.0,
            "dy": 4.0,
            "brake": 1,
            "abs": 0
        }
        tokens = tokenizer.encode_standard_dict(bsm_dict)
        self.assertEqual(len(tokens), 6)
        self.assertEqual(tokens[0], tokenizer.BSM_TOKEN)

        # ETSI CAM Dict
        cam_dict = {
            "messageId": "CAM",
            "speed": 50.0,
            "accel": 0.2,
            "heading": 180.0,
            "dx": 0.0,
            "dy": 0.0,
            "light": 1
        }
        cam_tokens = tokenizer.encode_standard_dict(cam_dict)
        self.assertEqual(len(cam_tokens), 6)
        self.assertEqual(cam_tokens[0], tokenizer.CAM_TOKEN)

        # SAE SPaT Dict
        spat_dict = {
            "messageId": "SPAT",
            "phase": "GREEN",
            "countdown": 15.0
        }
        spat_tokens = tokenizer.encode_standard_dict(spat_dict)
        self.assertEqual(len(spat_tokens), 3)
        self.assertEqual(spat_tokens[0], tokenizer.SPAT_TOKEN)

        # ETSI DENM Dict
        denm_dict = {
            "messageId": "DENM",
            "cause": "HARD_BRAKING",
            "speed": 80.0,
            "heading": 45.0
        }
        denm_tokens = tokenizer.encode_standard_dict(denm_dict)
        self.assertEqual(len(denm_tokens), 4)
        self.assertEqual(denm_tokens[0], tokenizer.DENM_TOKEN)

    def test_int8_quantization(self):
        model = load_model(pretrained=False, device="cpu")
        quant_model = model.quantize_int8()
        x = torch.randint(0, 1024, (2, 64))
        mask = torch.ones((2, 64))
        cls_logits, _ = quant_model.forward_classify(x, attention_mask=mask)
        self.assertEqual(cls_logits.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
