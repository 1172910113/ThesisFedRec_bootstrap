"""Lightweight contracts for the unified processed-data layer."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

RawId = str | int
Metadata = dict[str, Any]
PartitionKind = Literal["none", "hfl", "vfl", "hyfl"]
SCHEMA_VERSION = "1.0"


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
class FeatureView:
    """Canonical feature rows aligned to one processed entity namespace."""

    name: str
    entity_kind: Literal["user", "item", "interaction"]
    entity_ids: np.ndarray
    modality: str
    values: np.ndarray | tuple[str, ...]
    feature_names: tuple[str, ...]
    missing_mask: np.ndarray | None
    metadata: Metadata

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("FeatureView name must be a string")
        if not self.name.strip():
            raise ValueError("FeatureView name must not be empty")
        if self.entity_kind not in ("user", "item", "interaction"):
            raise ValueError(
                f"unsupported FeatureView entity_kind: {self.entity_kind}"
            )

        _validate_sorted_unique("FeatureView entity_ids", self.entity_ids)
        if np.any(self.entity_ids < 0):
            raise ValueError("FeatureView entity_ids must be non-negative")

        if not isinstance(self.modality, str):
            raise TypeError("FeatureView modality must be a string")
        if not self.modality.strip():
            raise ValueError("FeatureView modality must not be empty")

        if not isinstance(self.feature_names, tuple) or any(
            not isinstance(feature_name, str)
            for feature_name in self.feature_names
        ):
            raise TypeError("FeatureView feature_names must be a tuple of strings")
        if any(not feature_name.strip() for feature_name in self.feature_names):
            raise ValueError("FeatureView feature_names must not contain empty names")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("FeatureView feature_names must be unique")

        if isinstance(self.values, np.ndarray):
            if self.values.ndim == 0:
                raise ValueError("FeatureView values must have a row dimension")
            values_shape = self.values.shape
            if self.feature_names:
                if self.values.ndim != 2:
                    raise ValueError(
                        "FeatureView values with feature_names must be two-dimensional"
                    )
                if len(self.feature_names) != self.values.shape[1]:
                    raise ValueError(
                        "FeatureView feature_names must match the values columns"
                    )
        elif isinstance(self.values, tuple) and all(
            isinstance(value, str) for value in self.values
        ):
            values_shape = (len(self.values),)
            if self.feature_names:
                raise ValueError(
                    "FeatureView feature_names must be empty for string values"
                )
        else:
            raise TypeError(
                "FeatureView values must be a NumPy array or a tuple of strings"
            )

        if values_shape[0] != len(self.entity_ids):
            raise ValueError(
                "FeatureView entity_ids and values must have the same row count"
            )

        if self.missing_mask is not None:
            if not isinstance(self.missing_mask, np.ndarray):
                raise TypeError("FeatureView missing_mask must be a NumPy array")
            if self.missing_mask.dtype != np.bool_:
                raise TypeError("FeatureView missing_mask must have boolean dtype")
            if self.missing_mask.shape != values_shape:
                raise ValueError(
                    "FeatureView missing_mask must match the values shape"
                )

        if not isinstance(self.metadata, dict):
            raise TypeError("FeatureView metadata must be a dictionary")


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
    feature_views: dict[str, FeatureView] = field(default_factory=dict)
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

        if not isinstance(self.feature_views, dict):
            raise TypeError("feature_views must be a dictionary")
        entity_limits = {
            "user": len(self.user_id_map),
            "item": len(self.item_id_map),
            "interaction": len(self.interactions),
        }
        for view_name, view in self.feature_views.items():
            if not isinstance(view, FeatureView):
                raise TypeError("feature_views values must be FeatureView objects")
            if view_name != view.name:
                raise ValueError(
                    "feature_views keys must match their FeatureView names"
                )
            if view.entity_ids.size and np.any(
                view.entity_ids >= entity_limits[view.entity_kind]
            ):
                raise ValueError(
                    f"FeatureView {view.name!r} references an out-of-range "
                    f"{view.entity_kind} ID"
                )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
