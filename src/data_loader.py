"""
Strictly Leakage-Free Real VeReMi & Multi-Standard Dataset Loader for V2X-BERT.
Parses authentic VeReMi (SecureComm 2018) raw simulation archives (.tgz) and generates
strictly scenario-disjoint and sender-disjoint train/val/test partitions.
"""

import os
import tarfile
import json
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from .v2x_tokenizer import V2XTokenizer


class V2XDataset(Dataset):
    """
    Holds curated standards-tokenized V2X sequences, attention masks, labels, and provenance metadata.
    """
    def __init__(self, sequences, attention_masks, labels, attack_types, sender_ids, scenario_ids, max_len=64):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.attack_types = list(attack_types)
        self.sender_ids = list(sender_ids)
        self.vehicle_ids = self.sender_ids  # Alias for sender vehicle station IDs
        self.scenario_ids = list(scenario_ids)
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.attention_masks[idx], self.labels[idx]


def load_real_veremi_dataset(
    veremi_dir="data/veremi/securecomm2018",
    max_archives=40,
    seq_len=5,
    window_tokens=64,
    cache_path="data/real_veremi_cache.npz",
    force_rebuild=False
):
    """
    Loads authentic VeReMi (SecureComm 2018) dataset from raw .tgz archives.
    Parses real BSM frames (type 3), ground truth GPS kinematics (type 2), and attack taxonomy.
    Attack Types:
      - A0: Benign (Label 0)
      - A1: Constant Position Attack (Label 1)
      - A2: Constant Offset Attack (Label 1)
      - A4: Random Offset Attack (Label 1)
      - A8: Delayed / Replay Attack (Label 1)
    """
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"[VeReMi Loader] Loading real VeReMi dataset from cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        return (
            cache["sequences"],
            cache["masks"],
            cache["labels"],
            cache["attack_types"].tolist(),
            cache["sender_ids"].tolist(),
            cache["scenario_ids"].tolist()
        )

    print(f"[VeReMi Loader] Parsing authentic raw VeReMi .tgz archives from {veremi_dir}...")
    archives = sorted(glob.glob(os.path.join(veremi_dir, "*.tgz")))
    if not archives:
        # Fallback search if path is relative to project root
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        archives = sorted(glob.glob(os.path.join(root_dir, veremi_dir, "*.tgz")))

    if not archives:
        raise FileNotFoundError(f"No VeReMi .tgz archives found in {veremi_dir}")

    tokenizer = V2XTokenizer()
    selected_archives = archives[:max_archives]

    sequences = []
    masks = []
    labels = []
    attack_types = []
    sender_ids = []
    scenario_ids = []

    attack_name_map = {
        "A0": "Benign",
        "A1": "ConstantPosition",
        "A2": "ConstantOffset",
        "A4": "RandomOffset",
        "A8": "DataReplay"
    }

    for arch_path in selected_archives:
        # Simulation Scenario ID derived from archive name (e.g., veins_maat.uc1.14505201.180205_165350)
        arch_base = os.path.basename(arch_path)
        parts = arch_base.split(".")
        scenario_id = parts[2] if len(parts) >= 3 else arch_base.replace(".tgz", "")

        with tarfile.open(arch_path, "r:gz") as t:
            for member in t.getmembers():
                if not (member.name.endswith(".json") and "JSONlog-" in member.name):
                    continue

                fname = os.path.basename(member.name)
                # JSONlog-ReceiverID-SenderID-AttackCode.json
                log_parts = fname.replace(".json", "").split("-")
                if len(log_parts) < 4:
                    continue

                rcv_id = log_parts[1]
                snd_id = log_parts[2]
                atk_code = log_parts[3]

                sender_global_id = f"VEH_{scenario_id}_{snd_id}"
                is_attack = 0 if atk_code == "A0" else 1
                atk_name = attack_name_map.get(atk_code, f"Attack_{atk_code}")

                f = t.extractfile(member)
                if f is None:
                    continue

                bsm_entries = []
                for line in f:
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if not line_str:
                        continue
                    try:
                        entry = json.loads(line_str)
                        if entry.get("type") == 3:  # BSM telemetry packet
                            bsm_entries.append(entry)
                    except Exception:
                        continue

                if len(bsm_entries) < seq_len:
                    continue

                # Group into consecutive sliding window sequences of length seq_len
                for start_idx in range(0, len(bsm_entries) - seq_len + 1, seq_len):
                    window = bsm_entries[start_idx : start_idx + seq_len]
                    msg_token_list = []
                    prev_speed = 0.0

                    pos0 = window[0].get("pos", [0.0, 0.0, 0.0])
                    for step, msg in enumerate(window):
                        pos = msg.get("pos", [0.0, 0.0, 0.0])
                        spd = msg.get("spd", [0.0, 0.0, 0.0])

                        # Compute Euclidean speed & kinematics
                        vx, vy = spd[0], spd[1]
                        speed_mps = math.sqrt(vx * vx + vy * vy)
                        speed_kmh = max(0.0, min(180.0, speed_mps * 3.6))

                        # Heading
                        heading_deg = (math.atan2(vy, vx) * 180.0 / math.pi) % 360.0

                        # Acceleration estimate (dt = 0.1s standard BSM period)
                        accel_mps2 = (speed_mps - prev_speed) / 0.1 if step > 0 else 0.0
                        accel_mps2 = max(-12.0, min(8.0, accel_mps2))
                        prev_speed = speed_mps

                        is_braking = 1 if accel_mps2 < -1.5 else 0
                        abs_active = 1 if accel_mps2 < -4.0 else 0

                        # Relative trajectory offset from start of window
                        dx = max(-250.0, min(250.0, float(pos[0]) - float(pos0[0])))
                        dy = max(-250.0, min(250.0, float(pos[1]) - float(pos0[1])))

                        bsm_tokens = tokenizer.encode_bsm(
                            speed=speed_kmh,
                            accel=accel_mps2,
                            heading=heading_deg,
                            dx=dx,
                            dy=dy,
                            brake=is_braking,
                            abs_flag=abs_active
                        )
                        msg_token_list.append(bsm_tokens)

                    seq_ids, seq_mask = tokenizer.encode_sequence(msg_token_list, max_len=window_tokens)

                    sequences.append(seq_ids.numpy())
                    masks.append(seq_mask.numpy())
                    labels.append(is_attack)
                    attack_types.append(atk_name)
                    sender_ids.append(sender_global_id)
                    scenario_ids.append(scenario_id)

    sequences = np.array(sequences)
    masks = np.array(masks)
    labels = np.array(labels)

    print(f"[VeReMi Loader] Successfully parsed {len(sequences):,} real sequences from {len(selected_archives)} archives.")
    print(f"[VeReMi Loader] Real Class Balance: Benign = {np.sum(labels == 0):,} ({np.mean(labels == 0)*100:.1f}%) | Attacks = {np.sum(labels == 1):,} ({np.mean(labels == 1)*100:.1f}%)")

    # Cache to disk for instant loading
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        sequences=sequences,
        masks=masks,
        labels=labels,
        attack_types=np.array(attack_types),
        sender_ids=np.array(sender_ids),
        scenario_ids=np.array(scenario_ids)
    )
    print(f"[VeReMi Loader] Saved compressed cache to {cache_path}")

    return sequences, masks, labels, attack_types, sender_ids, scenario_ids


def split_by_sender_disjoint(full_dataset: V2XDataset, test_ratio: float = 0.20, seed: int = 42):
    """
    Partitions the dataset strictly by Sender StationID / Vehicle Pseudonym.
    Guarantees that test senders never appeared in the training set (Zero Transmitter Leakage).
    """
    unique_senders = np.array(sorted(list(set(full_dataset.sender_ids))))
    np.random.seed(seed)
    np.random.shuffle(unique_senders)

    n_test_senders = max(1, int(len(unique_senders) * test_ratio))
    test_senders_set = set(unique_senders[:n_test_senders])
    train_senders_set = set(unique_senders[n_test_senders:])

    train_idx = [i for i, s in enumerate(full_dataset.sender_ids) if s in train_senders_set]
    test_idx = [i for i, s in enumerate(full_dataset.sender_ids) if s in test_senders_set]

    train_ds = V2XDataset(
        full_dataset.sequences[train_idx].numpy(),
        full_dataset.attention_masks[train_idx].numpy(),
        full_dataset.labels[train_idx].numpy(),
        [full_dataset.attack_types[i] for i in train_idx],
        [full_dataset.sender_ids[i] for i in train_idx],
        [full_dataset.scenario_ids[i] for i in train_idx],
        max_len=full_dataset.max_len
    )

    test_ds = V2XDataset(
        full_dataset.sequences[test_idx].numpy(),
        full_dataset.attention_masks[test_idx].numpy(),
        full_dataset.labels[test_idx].numpy(),
        [full_dataset.attack_types[i] for i in test_idx],
        [full_dataset.sender_ids[i] for i in test_idx],
        [full_dataset.scenario_ids[i] for i in test_idx],
        max_len=full_dataset.max_len
    )

    # Formal mathematical verification
    overlap = len(set(train_ds.sender_ids).intersection(set(test_ds.sender_ids)))
    assert overlap == 0, f"FATAL DATA LEAKAGE: {overlap} senders found in both train and test!"

    return train_ds, test_ds


def split_by_scenario_disjoint(full_dataset: V2XDataset, test_ratio: float = 0.20, seed: int = 42):
    """
    Partitions the dataset strictly by Simulation Scenario ID.
    Guarantees that test simulation runs never appeared in the training set (Zero Scenario Leakage).
    """
    unique_scenarios = np.array(sorted(list(set(full_dataset.scenario_ids))))
    np.random.seed(seed)
    np.random.shuffle(unique_scenarios)

    n_test_scenarios = max(1, int(len(unique_scenarios) * test_ratio))
    test_scenarios_set = set(unique_scenarios[:n_test_scenarios])
    train_scenarios_set = set(unique_scenarios[n_test_scenarios:])

    train_idx = [i for i, sc in enumerate(full_dataset.scenario_ids) if sc in train_scenarios_set]
    test_idx = [i for i, sc in enumerate(full_dataset.scenario_ids) if sc in test_scenarios_set]

    train_ds = V2XDataset(
        full_dataset.sequences[train_idx].numpy(),
        full_dataset.attention_masks[train_idx].numpy(),
        full_dataset.labels[train_idx].numpy(),
        [full_dataset.attack_types[i] for i in train_idx],
        [full_dataset.sender_ids[i] for i in train_idx],
        [full_dataset.scenario_ids[i] for i in train_idx],
        max_len=full_dataset.max_len
    )

    test_ds = V2XDataset(
        full_dataset.sequences[test_idx].numpy(),
        full_dataset.attention_masks[test_idx].numpy(),
        full_dataset.labels[test_idx].numpy(),
        [full_dataset.attack_types[i] for i in test_idx],
        [full_dataset.sender_ids[i] for i in test_idx],
        [full_dataset.scenario_ids[i] for i in test_idx],
        max_len=full_dataset.max_len
    )

    overlap_sc = len(set(train_ds.scenario_ids).intersection(set(test_ds.scenario_ids)))
    assert overlap_sc == 0, f"FATAL DATA LEAKAGE: {overlap_sc} scenarios in both train and test!"

    return train_ds, test_ds


def assert_no_scenario_overlap(train_dataset: V2XDataset, test_dataset: V2XDataset) -> bool:
    """
    Formally verifies zero scenario ID overlap between training and testing partitions.
    """
    train_scenarios = set(train_dataset.scenario_ids)
    test_scenarios = set(test_dataset.scenario_ids)
    overlap = train_scenarios.intersection(test_scenarios)
    if len(overlap) > 0:
        raise AssertionError(f"Scenario leakage detected! Overlapping scenarios ({len(overlap)}): {overlap}")
    return True


def assert_no_sender_overlap(train_dataset: V2XDataset, test_dataset: V2XDataset) -> bool:
    """
    Formally verifies zero sender StationID overlap between training and testing partitions.
    """
    train_senders = set(train_dataset.sender_ids)
    test_senders = set(test_dataset.sender_ids)
    overlap = train_senders.intersection(test_senders)
    if len(overlap) > 0:
        raise AssertionError(f"Sender leakage detected! Overlapping senders ({len(overlap)}): {overlap}")
    return True


def get_dataset_provenance(veremi_dir="data/veremi/securecomm2018", cache_path="data/real_veremi_cache.npz") -> dict:
    """
    Computes cryptographic hash and extraction metadata for the real VeReMi dataset.
    """
    import hashlib
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    resolved_cache = cache_path if os.path.isabs(cache_path) else os.path.join(root_dir, cache_path)

    dataset_hash = "UNKNOWN"
    total_seqs = 0
    if os.path.exists(resolved_cache):
        hasher = hashlib.sha256()
        with open(resolved_cache, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        dataset_hash = hasher.hexdigest()
        cache = np.load(resolved_cache, allow_pickle=True)
        total_seqs = len(cache["sequences"])

    resolved_veremi = veremi_dir if os.path.isabs(veremi_dir) else os.path.join(root_dir, veremi_dir)
    archives = sorted(glob.glob(os.path.join(resolved_veremi, "*.tgz")))

    return {
        "dataset_name": "VeReMi (Vehicle Reference Misbehavior Dataset)",
        "dataset_version": "SecureComm 2018 Official Release",
        "archive_count": len(archives),
        "total_cached_sequences": total_seqs,
        "sha256_hash": dataset_hash,
        "cache_path": cache_path
    }
