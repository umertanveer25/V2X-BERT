"""
Package export wrapper for real VeReMi dataset loader.
"""
from src.data_loader import (
    V2XDataset,
    load_real_veremi_dataset,
    split_by_sender_disjoint,
    split_by_scenario_disjoint,
    assert_no_scenario_overlap,
    assert_no_sender_overlap,
    get_dataset_provenance
)

load_veremi_standards_dataset = load_real_veremi_dataset

__all__ = [
    "V2XDataset",
    "load_real_veremi_dataset",
    "load_veremi_standards_dataset",
    "split_by_sender_disjoint",
    "split_by_scenario_disjoint",
    "assert_no_scenario_overlap",
    "assert_no_sender_overlap",
    "get_dataset_provenance"
]
