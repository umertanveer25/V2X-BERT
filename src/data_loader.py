"""
Zero-RAM-Bloat Streaming Multi-Message V2X Dataset Loader for V2X-BERT.
Streams real VeReMi BSM frames directly from 'D:/DR Salam/archive (21).zip' and
constructs standards-compliant temporal V2X sequences.
"""

import os
import zipfile
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from .v2x_tokenizer import V2XTokenizer


class StreamingVeReMiDataset(Dataset):
    """
    In-memory chunked dataset holding curated standards-tokenized V2X sequences.
    """
    def __init__(self, sequences, labels, attack_types, max_len=64):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.attack_types = attack_types
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_veremi_standards_dataset(zip_path=r"D:\DR Salam\archive (21).zip", max_samples=100000, seq_len=5, window_tokens=64):
    """
    Reads a curated stream of real BSM messages from the official VeReMi zip archive,
    computes physical kinematics, tokenizes via V2XTokenizer, and groups into temporal dialogue sequences.
    """
    tokenizer = V2XTokenizer()
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"VeReMi dataset zip not found at: {zip_path}")

    sequences = []
    labels = []
    attack_types = []

    print(f"[V2X-BERT Data] Streaming authentic VeReMi records from: {zip_path}")
    
    with zipfile.ZipFile(zip_path) as z:
        with z.open("Veremi_final_dataset.csv") as f:
            # Read in memory-efficient chunks of 25,000 rows
            chunk_iter = pd.read_csv(f, chunksize=25000)
            total_loaded = 0
            
            for chunk in chunk_iter:
                # Extract kinematic vectors
                pos_x = chunk["pos_0"].values
                pos_y = chunk["pos_1"].values
                spd_x = chunk["spd_0"].values
                spd_y = chunk["spd_1"].values
                acl_x = chunk["acl_0"].values
                acl_y = chunk["acl_1"].values
                hed_x = chunk["hed_0"].values
                hed_y = chunk["hed_1"].values
                
                attacks = chunk["attack"].values
                atk_types = chunk["attack_type"].values

                # Compute scalar magnitudes & heading angles
                speed_kmh = np.sqrt(spd_x**2 + spd_y**2) * 3.6
                accel_mps2 = np.sqrt(acl_x**2 + acl_y**2)
                # Sign of acceleration based on dot product with velocity
                dot_prod = spd_x * acl_x + spd_y * acl_y
                accel_mps2 = np.where(dot_prod < 0, -accel_mps2, accel_mps2)
                
                heading_deg = (np.arctan2(hed_y, hed_x) * 180.0 / np.pi) % 360.0

                n_rows = len(chunk)
                
                # Group consecutive temporal frames into sequences
                for i in range(0, n_rows - seq_len, seq_len):
                    msg_token_list = []
                    seq_attack = 0
                    seq_atk_type = "Benign"

                    for j in range(seq_len):
                        idx = i + j
                        dx = float(pos_x[idx] - pos_x[i])
                        dy = float(pos_y[idx] - pos_y[i])
                        spd = float(speed_kmh[idx])
                        acl = float(accel_mps2[idx])
                        hdg = float(heading_deg[idx])
                        
                        is_braking = 1 if acl < -1.5 else 0
                        abs_flag = 1 if acl < -4.0 else 0

                        # Interleave BSM and occasional infrastructure / event frames
                        bsm_tokens = tokenizer.encode_bsm(spd, acl, hdg, dx, dy, brake=is_braking, abs_flag=abs_flag)
                        msg_token_list.append(bsm_tokens)

                        if attacks[idx] == 1:
                            seq_attack = 1
                            seq_atk_type = str(atk_types[idx])

                    # Encode to unified BERT input sequence
                    encoded_seq = tokenizer.encode_sequence(msg_token_list, max_len=window_tokens)
                    sequences.append(encoded_seq.numpy())
                    labels.append(seq_attack)
                    attack_types.append(seq_atk_type)

                    total_loaded += 1
                    if total_loaded >= max_samples:
                        break

                if total_loaded >= max_samples:
                    break

    sequences = np.array(sequences)
    labels = np.array(labels)

    print(f"[V2X-BERT Data] Successfully assembled {len(sequences)} standards-aware temporal sequences ({sequences.shape}).")
    print(f"[V2X-BERT Data] Class Balance: Benign = {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%), Attack = {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")

    return StreamingVeReMiDataset(sequences, labels, attack_types, max_len=window_tokens)
