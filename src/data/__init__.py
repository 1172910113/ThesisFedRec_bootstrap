"""Shared processed-data contracts and deterministic preprocessing utilities."""

from src.data.contracts import (
    DatasetBundle,
    InteractionSplits,
    InteractionTable,
    PartyData,
)
from src.data.loaders import (
    load_ml1m_ratings,
    resolve_data_path,
    resolve_data_root,
)
from src.data.negative_sampling import sample_negative_items
from src.data.partition import build_hfl_partition
from src.data.preprocess import preprocess_ml1m, preprocess_ml1m_from_root
from src.data.split import leave_one_out_split

__all__ = [
    "DatasetBundle",
    "InteractionSplits",
    "InteractionTable",
    "PartyData",
    "build_hfl_partition",
    "leave_one_out_split",
    "load_ml1m_ratings",
    "preprocess_ml1m",
    "preprocess_ml1m_from_root",
    "resolve_data_path",
    "resolve_data_root",
    "sample_negative_items",
]
