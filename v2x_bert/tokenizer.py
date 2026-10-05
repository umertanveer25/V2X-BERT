"""
Schema-Aware Standards-Informed Multi-Message V2X Tokenizer for V2X-BERT.
Supports SAE J2735:2020 (BSM, SPaT, MAP, PSM) and ETSI EN 302 637-2 (CAM, DENM, CDD).
Vocabulary size |V| = 1,024 discrete semantic tokens.
"""

import math
import numpy as np
import torch


class V2XTokenizer:
    def __init__(self):
        self.vocab_size = 1024

        # Special & Message Identifiers [0 - 31]
        self.PAD_TOKEN = 0
        self.UNK_TOKEN = 1
        self.CLS_TOKEN = 2
        self.SEP_TOKEN = 3
        self.MASK_TOKEN = 4

        # Standard Message Identifiers
        self.BSM_TOKEN = 5   # SAE J2735 BSM (Basic Safety Message)
        self.CAM_TOKEN = 6   # ETSI CAM (Cooperative Awareness Message)
        self.SPAT_TOKEN = 7  # SAE J2735 SPaT (Signal Phase and Timing)
        self.MAP_TOKEN = 8   # SAE/ETSI MAP (Intersection Geometry)
        self.DENM_TOKEN = 9  # ETSI DENM (Decentralized Environmental Notification)
        self.PSM_TOKEN = 10  # SAE PSM / ETSI VAM (Vulnerable Road Users)
        self.CPM_TOKEN = 11  # ETSI CPM (Collective Perception Message)

        # Token Sub-Ranges
        self.SPEED_OFFSET = 32         # 32 to 127 (96 linear bins: 0 to 180 km/h)
        self.ACCEL_OFFSET = 128        # 128 to 255 (128 linear bins: -12.0 to +8.0 m/s^2)
        self.HEADING_OFFSET = 256      # 256 to 327 (72 bins: 5 deg angular resolution)
        self.STATUS_OFFSET = 328       # 328 to 399 (Safety bitmasks & vehicle events)
        self.SPAT_OFFSET = 400         # 400 to 449 (Signal phases & countdown timers)
        self.DENM_OFFSET = 450         # 450 to 499 (Hazard event cause codes)
        self.SPATIAL_OFFSET = 500      # 500 to 1011 (512 polar relative spatial grid bins: 16 distance tiers x 32 angular sectors)

        # Status Flag Mapping
        self.FLAG_TOKENS = {
            "BRAKE_OFF": self.STATUS_OFFSET + 0,
            "BRAKE_ACTIVE": self.STATUS_OFFSET + 1,
            "ABS_ACTIVE": self.STATUS_OFFSET + 2,
            "TCS_ACTIVE": self.STATUS_OFFSET + 3,
            "SCS_ACTIVE": self.STATUS_OFFSET + 4,
            "HAZARD_LIGHTS": self.STATUS_OFFSET + 5,
            "SIREN_ACTIVE": self.STATUS_OFFSET + 6,
            "AIRBAG_ARMED": self.STATUS_OFFSET + 7,
        }

        # SPaT Signal States
        self.SPAT_PHASES = {
            "RED": self.SPAT_OFFSET + 0,
            "YELLOW": self.SPAT_OFFSET + 1,
            "GREEN": self.SPAT_OFFSET + 2,
            "FLASHING_RED": self.SPAT_OFFSET + 3,
            "FLASHING_YELLOW": self.SPAT_OFFSET + 4,
        }

        # DENM Cause Codes
        self.DENM_CAUSES = {
            "HARD_BRAKING": self.DENM_OFFSET + 1,
            "ROAD_HAZARD": self.DENM_OFFSET + 2,
            "BLACK_ICE": self.DENM_OFFSET + 3,
            "ACCIDENT_AHEAD": self.DENM_OFFSET + 4,
            "VRU_CROSSING": self.DENM_OFFSET + 5,
            "CONGESTION": self.DENM_OFFSET + 6,
        }

    def tokenize_speed(self, speed_kmh: float) -> int:
        """Quantizes continuous speed (0-180 km/h) into 96 discrete linear bins."""
        clamped = max(0.0, min(180.0, float(speed_kmh)))
        bin_idx = int(clamped / (180.0 / 96.0))
        bin_idx = min(95, max(0, bin_idx))
        return self.SPEED_OFFSET + bin_idx

    def tokenize_accel(self, accel_mps2: float) -> int:
        """Quantizes continuous acceleration (-12 to +8 m/s^2) into 128 linear bins."""
        clamped = max(-12.0, min(8.0, float(accel_mps2)))
        norm = (clamped + 12.0) / 20.0  # Linear normalization [0, 1]
        bin_idx = int(norm * 127.0)
        bin_idx = min(127, max(0, bin_idx))
        return self.ACCEL_OFFSET + bin_idx

    def tokenize_heading(self, heading_deg: float) -> int:
        """Quantizes heading angle (0-360 deg) into 72 bins (5 deg resolution)."""
        deg = float(heading_deg) % 360.0
        bin_idx = int(deg / 5.0)
        bin_idx = min(71, max(0, bin_idx))
        return self.HEADING_OFFSET + bin_idx

    def tokenize_spatial_delta(self, dx: float, dy: float) -> int:
        """
        Quantizes relative Euclidean offset into polar spatial grid token.
        16 distance tiers (0-160 m) x 32 angular sectors (11.25 deg) = 512 discrete grid bins.
        """
        dist = math.sqrt(dx * dx + dy * dy)
        angle = (math.atan2(dy, dx) * 180.0 / math.pi) % 360.0
        
        dist_tier = min(15, int(dist / 10.0))  # 0 to 15
        angle_sector = min(31, int(angle / (360.0 / 32.0)))  # 0 to 31
        grid_id = dist_tier * 32 + angle_sector  # 0 to 511
        grid_id = min(511, max(0, grid_id))
        return self.SPATIAL_OFFSET + grid_id

    def encode_bsm(self, speed: float, accel: float, heading: float, dx: float = 0.0, dy: float = 0.0, brake: int = 0, abs_flag: int = 0) -> list:
        """
        Converts a single SAE J2735 BSM frame into a standards-informed token sequence.
        Format: [BSM, SpeedToken, AccelToken, HeadingToken, SpatialToken, StatusToken]
        """
        status_tok = self.FLAG_TOKENS["ABS_ACTIVE"] if abs_flag else (
            self.FLAG_TOKENS["BRAKE_ACTIVE"] if brake else self.FLAG_TOKENS["BRAKE_OFF"]
        )
        return [
            self.BSM_TOKEN,
            self.tokenize_speed(speed),
            self.tokenize_accel(accel),
            self.tokenize_heading(heading),
            self.tokenize_spatial_delta(dx, dy),
            status_tok
        ]

    def encode_cam(self, speed: float, accel: float, heading: float, dx: float = 0.0, dy: float = 0.0, light_status: int = 0) -> list:
        """
        Converts an ETSI CAM frame into a standards-informed token sequence.
        Format: [CAM, SpeedToken, AccelToken, HeadingToken, SpatialToken, LightToken]
        """
        status_tok = self.FLAG_TOKENS["HAZARD_LIGHTS"] if light_status else self.FLAG_TOKENS["BRAKE_OFF"]
        return [
            self.CAM_TOKEN,
            self.tokenize_speed(speed),
            self.tokenize_accel(accel),
            self.tokenize_heading(heading),
            self.tokenize_spatial_delta(dx, dy),
            status_tok
        ]

    def encode_spat(self, signal_phase: str, time_to_change_sec: float) -> list:
        """
        Converts an SAE J2735 SPaT (Signal Phase & Timing) frame into token sequence.
        Format: [SPAT, PhaseToken, CountdownToken]
        """
        phase_token = self.SPAT_PHASES.get(signal_phase, self.SPAT_PHASES["RED"])
        countdown = min(30.0, max(0.0, float(time_to_change_sec)))
        countdown_token = self.SPAT_OFFSET + 10 + int(countdown)
        return [self.SPAT_TOKEN, phase_token, countdown_token]

    def encode_denm(self, cause_code: str, speed: float, heading: float) -> list:
        """
        Converts an ETSI DENM safety event alert into token sequence.
        Format: [DENM, CauseToken, SpeedToken, HeadingToken]
        """
        cause_token = self.DENM_CAUSES.get(cause_code, self.DENM_CAUSES["ROAD_HAZARD"])
        return [self.DENM_TOKEN, cause_token, self.tokenize_speed(speed), self.tokenize_heading(heading)]

    def encode_sequence(self, message_token_lists: list, max_len: int = 64):
        """
        Packages multiple consecutive V2X frames into a unified BERT input sequence.
        Adds [CLS] at beginning and [SEP] between frames.
        Returns:
            input_ids: torch.Tensor of shape (max_len,)
            attention_mask: torch.Tensor of shape (max_len,) where 1 is active token, 0 is padding.
        """
        flat_tokens = [self.CLS_TOKEN]
        for msg in message_token_lists:
            flat_tokens.extend(msg)
            flat_tokens.append(self.SEP_TOKEN)

        # Truncate or Pad to max_len
        if len(flat_tokens) > max_len:
            flat_tokens = flat_tokens[:max_len]
            attention_mask = [1] * max_len
        else:
            n_tokens = len(flat_tokens)
            flat_tokens.extend([self.PAD_TOKEN] * (max_len - n_tokens))
            attention_mask = [1] * n_tokens + [0] * (max_len - n_tokens)

        return torch.tensor(flat_tokens, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)
