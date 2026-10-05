"""
Edge-V2X-BERT: Domain-Specific Compact Transformer for Standards-Aware Vehicular Communication.
Designed for CPU pre-training and sub-millisecond automotive On-Board Unit (OBU) inference.
Total parameters: ~1.8 Million.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class V2XPositionalEmbedding(nn.Module):
    """
    Sinusoidal / Learned Positional Embeddings for temporal V2X sequences.
    """
    def __init__(self, d_model=128, max_len=128):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.pos_embedding(positions)


class V2XTransformerEncoderBlock(nn.Module):
    """
    Pre-LN Transformer Block with Multi-Head Self-Attention and GELU Feed-Forward Network.
    """
    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, key_padding_mask=None):
        # Pre-LN Multi-Head Attention
        norm_x = self.ln1(x)
        attn_out, attn_weights = self.attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # Pre-LN Feed-Forward
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


class EdgeV2XBERT(nn.Module):
    """
    Edge-Native V2X-BERT Foundational Architecture.
    """
    def __init__(self, vocab_size=1024, d_model=128, n_heads=4, num_layers=4, d_ff=512, max_len=64, num_classes=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # Token & Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = V2XPositionalEmbedding(d_model, max_len)
        self.emb_layer_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer Encoder Stack
        self.layers = nn.ModuleList([
            V2XTransformerEncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 1. Masked Telemetry Modeling (MTM) Pre-training Head
        self.mtm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )

        # 2. Zero-Trust Misbehavior Detection (Classification) Head
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        # 3. Cross-Standard Alignment Projection Head (SAE J2735 <-> ETSI)
        self.cross_standard_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_encoder(self, input_ids):
        """Passes input IDs through embedding layer and transformer encoder blocks."""
        padding_mask = (input_ids == 0)
        
        # Compute embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(input_ids)
        x = self.emb_dropout(self.emb_layer_norm(x))

        all_attentions = []
        for layer in self.layers:
            x, attn_w = layer(x, key_padding_mask=padding_mask)
            all_attentions.append(attn_w)

        x = self.final_norm(x)
        return x, all_attentions

    def forward_pretrain(self, masked_input_ids):
        """
        Self-Supervised Masked Telemetry Modeling forward pass.
        Returns predicted logits over vocabulary |V| for all token positions.
        """
        hidden_states, _ = self.forward_encoder(masked_input_ids)
        logits = self.mtm_head(hidden_states)
        return logits

    def forward_classify(self, input_ids):
        """
        Downstream Zero-Trust Misbehavior Classification using [CLS] token (position 0).
        """
        hidden_states, attentions = self.forward_encoder(input_ids)
        cls_repr = hidden_states[:, 0, :]  # Extract [CLS] representation
        logits = self.classifier_head(cls_repr)
        return logits, attentions

    def forward_cross_standard(self, sae_ids, etsi_ids):
        """
        Extracts projected normalized embeddings for cross-standard alignment testing.
        """
        h_sae, _ = self.forward_encoder(sae_ids)
        h_etsi, _ = self.forward_encoder(etsi_ids)

        z_sae = F.normalize(self.cross_standard_proj(h_sae[:, 0, :]), dim=-1)
        z_etsi = F.normalize(self.cross_standard_proj(h_etsi[:, 0, :]), dim=-1)
        return z_sae, z_etsi
