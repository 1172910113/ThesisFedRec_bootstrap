"""Shared processed-data contracts and deterministic preprocessing utilities."""

from src.data.amazon_books import (
    audit_amazon_books_metadata,
    iter_amazon_books_reviews,
    preprocess_amazon_books,
)
from src.data.contracts import (
    DatasetBundle,
    FeatureAccess,
    FeaturePartition,
    FeatureView,
    GraphView,
    InteractionSplits,
    InteractionTable,
    PartyData,
)
from src.data.feature_scenarios import (
    build_complete_feature_scenario,
    build_single_view_absence_scenario,
)
from src.data.features import (
    feature_partition_checksum,
    resolve_party_feature_views,
    validate_feature_partition,
)
from src.data.loaders import (
    load_ml1m_ratings,
    resolve_data_path,
    resolve_data_root,
)
from src.data.ml1m_features import build_ml1m_item_genre_view
from src.data.negative_sampling import sample_negative_items
from src.data.partition import build_hfl_partition
from src.data.preprocess import preprocess_ml1m, preprocess_ml1m_from_root
from src.data.split import leave_one_out_split

__all__ = [
    "DatasetBundle",
    "FeatureAccess",
    "FeaturePartition",
    "FeatureView",
    "GraphView",
    "InteractionSplits",
    "InteractionTable",
    "PartyData",
    "audit_amazon_books_metadata",
    "build_complete_feature_scenario",
    "build_hfl_partition",
    "build_ml1m_item_genre_view",
    "build_single_view_absence_scenario",
    "feature_partition_checksum",
    "iter_amazon_books_reviews",
    "leave_one_out_split",
    "load_ml1m_ratings",
    "preprocess_amazon_books",
    "preprocess_ml1m",
    "preprocess_ml1m_from_root",
    "resolve_data_path",
    "resolve_data_root",
    "resolve_party_feature_views",
    "sample_negative_items",
    "validate_feature_partition",
]
