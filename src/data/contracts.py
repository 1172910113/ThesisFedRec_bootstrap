"""Lightweight contracts for the unified processed-data layer."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

RawId = str | int
Metadata = dict[str, Any]
PartitionKind = Literal["none", "hfl", "vfl", "hyfl"]


def _validate_int_array(name: str, values: np.ndarray) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.dtype != np.int64:
        raise TypeError(f"{name} must have dtype int64")


def _validate_sorted_unique(name: str, values: np.ndarray) -> None:
    _validate_int_array(name, values)
    if values.size and not np.array_equal(values, np.unique(values)):
        raise ValueError(f"{name} must be sorted and contain unique IDs")


@dataclass(slots=True, kw_only=True)
class InteractionTable:
    """Columnar interaction storage using aligned NumPy arrays."""

    interaction_ids: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray
    timestamps: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        for name in ("interaction_ids", "user_ids", "item_ids", "timestamps"):
            _validate_int_array(name, getattr(self, name))

        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a NumPy array")
        if self.values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if not np.issubdtype(self.values.dtype, np.floating):
            raise TypeError("values must use a floating-point dtype")

        lengths = {
            len(self.interaction_ids),
            len(self.user_ids),
            len(self.item_ids),
            len(self.timestamps),
            len(self.values),
        }
        if len(lengths) != 1:
            raise ValueError("all interaction columns must have the same length")

        expected_ids = np.arange(len(self), dtype=np.int64)
        if not np.array_equal(self.interaction_ids, expected_ids):
            raise ValueError(
                "interaction_ids must be contiguous, start at zero, and match row order"
            )
        if np.any(self.user_ids < 0) or np.any(self.item_ids < 0):
            raise ValueError("processed user_ids and item_ids must be non-negative")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("interaction values must be finite")

    def __len__(self) -> int:
        return int(self.interaction_ids.size)


@dataclass(slots=True, kw_only=True)
class InteractionSplits:
    """Interaction IDs assigned to train, validation, and test."""

    train_interaction_ids: np.ndarray
    valid_interaction_ids: np.ndarray
    test_interaction_ids: np.ndarray

    def __post_init__(self) -> None:
        _validate_sorted_unique(
            "train_interaction_ids", self.train_interaction_ids
        )
        _validate_sorted_unique(
            "valid_interaction_ids", self.valid_interaction_ids
        )
        _validate_sorted_unique("test_interaction_ids", self.test_interaction_ids)

        if np.intersect1d(
            self.train_interaction_ids, self.valid_interaction_ids
        ).size:
            raise ValueError("train and validation splits overlap")
        if np.intersect1d(
            self.train_interaction_ids, self.test_interaction_ids
        ).size:
            raise ValueError("train and test splits overlap")
        if np.intersect1d(
            self.valid_interaction_ids, self.test_interaction_ids
        ).size:
            raise ValueError("validation and test splits overlap")

    def validate_coverage(self, num_interactions: int) -> None:
        """Require the splits to cover every interaction exactly once."""
        combined = np.concatenate(
            (
                self.train_interaction_ids,
                self.valid_interaction_ids,
                self.test_interaction_ids,
            )
        )
        expected = np.arange(num_interactions, dtype=np.int64)
        if not np.array_equal(np.sort(combined), expected):
            raise ValueError("splits must cover every interaction exactly once")


@dataclass(slots=True, kw_only=True)
class PartyData:
    """Sample and view availability for one federated party."""

    party_id: int
    train_interaction_ids: np.ndarray
    valid_interaction_ids: np.ndarray
    test_interaction_ids: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray
    feature_view_names: tuple[str, ...]
    graph_view_names: tuple[str, ...]
    owns_labels: bool
    metadata: Metadata

    def __post_init__(self) -> None:
        if isinstance(self.party_id, bool) or self.party_id < 0:
            raise ValueError("party_id must be a non-negative integer")
        for name in (
            "train_interaction_ids",
            "valid_interaction_ids",
            "test_interaction_ids",
            "user_ids",
            "item_ids",
        ):
            _validate_sorted_unique(name, getattr(self, name))
        if len(set(self.feature_view_names)) != len(self.feature_view_names):
            raise ValueError("feature_view_names must be unique")
        if len(set(self.graph_view_names)) != len(self.graph_view_names):
            raise ValueError("graph_view_names must be unique")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")


def _validate_id_map(name: str, mapping: dict[RawId, int]) -> None:
    if not isinstance(mapping, dict):
        raise TypeError(f"{name} must be a dictionary")
    for raw_id in mapping:
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise TypeError(f"{name} raw IDs must be strings or integers")
    processed_ids = sorted(mapping.values())
    if processed_ids != list(range(len(mapping))):
        raise ValueError(f"{name} values must be contiguous IDs starting at zero")


@dataclass(slots=True, kw_only=True)
class DatasetBundle:
    """One processed base dataset with an optional selected partition."""

    name: str
    schema_version: str
    interactions: InteractionTable
    splits: InteractionSplits
    user_id_map: dict[RawId, int]
    item_id_map: dict[RawId, int]
    metadata: Metadata
    partition_name: str | None = None
    partition_kind: PartitionKind = "none"
    parties: tuple[PartyData, ...] = ()
    feature_views: dict[str, object] = field(default_factory=dict)
    graph_views: dict[str, object] = field(default_factory=dict)
    continual_views: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must not be empty")
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")

        _validate_id_map("user_id_map", self.user_id_map)
        _validate_id_map("item_id_map", self.item_id_map)
        self.splits.validate_coverage(len(self.interactions))

        expected_users = np.arange(len(self.user_id_map), dtype=np.int64)
        expected_items = np.arange(len(self.item_id_map), dtype=np.int64)
        if not np.array_equal(np.unique(self.interactions.user_ids), expected_users):
            raise ValueError("interaction user IDs must match the user mapping")
        if not np.array_equal(np.unique(self.interactions.item_ids), expected_items):
            raise ValueError("interaction item IDs must match the item mapping")

        if self.partition_kind not in {"none", "hfl", "vfl", "hyfl"}:
            raise ValueError(f"unsupported partition_kind: {self.partition_kind}")
        if self.partition_kind == "none":
            if self.partition_name is not None or self.parties:
                raise ValueError(
                    "an unpartitioned bundle cannot have a partition name or parties"
                )
        elif not self.partition_name:
            raise ValueError("a partitioned bundle requires partition_name")

        if self.parties:
            party_ids = [party.party_id for party in self.parties]
            if party_ids != list(range(len(self.parties))):
                raise ValueError("parties must be ordered by contiguous party_id")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
