"""
V2X-BERT: Compact Standards-Aware Bidirectional Transformer for Vehicular Telemetry.
"""

from .tokenizer import V2XTokenizer
from .model import EdgeV2XBERT, V2XTransformerEncoderBlock, V2XPositionalEmbedding
from .pretrain import pretrain_v2x_bert, mask_telemetry_tokens
from .evaluate import fine_tune_and_evaluate
from .data import load_veremi_standards_dataset, V2XDataset

__version__ = "1.0.0"
__author__ = "Umer Tanveer and Abdu Salam"

__all__ = [
    "V2XTokenizer",
    "EdgeV2XBERT",
    "V2XTransformerEncoderBlock",
    "V2XPositionalEmbedding",
    "pretrain_v2x_bert",
    "mask_telemetry_tokens",
    "fine_tune_and_evaluate",
    "load_veremi_standards_dataset",
    "load_real_veremi_dataset",
    "V2XDataset",
]


def load_model(pretrained: bool = True, device: str = "cpu") -> EdgeV2XBERT:
    """
    Convenience factory to load default V2X-BERT architecture (1.11M parameters).
    """
    model = EdgeV2XBERT(
        vocab_size=1024,
        d_model=128,
        n_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=64,
        num_classes=2,
        dropout=0.1
    )
    model.to(device)
    return model
