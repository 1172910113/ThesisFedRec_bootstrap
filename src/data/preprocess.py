"""Deterministic MovieLens-1M base preprocessing."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.data.contracts import DatasetBundle, InteractionTable
from src.data.loaders import load_ml1m_ratings, resolve_data_path
from src.data.split import leave_one_out_split

SCHEMA_VERSION = "1.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select(
    columns: dict[str, np.ndarray], selection: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[selection] for name, values in columns.items()}


def _valid_record_mask(columns: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (columns["raw_user_ids"] > 0)
        & (columns["raw_item_ids"] > 0)
        & (columns["timestamps"] >= 0)
        & np.isfinite(columns["ratings"])
    )


def _filter_minimum_interactions(
    columns: dict[str, np.ndarray], minimum: int
) -> dict[str, np.ndarray]:
    if minimum < 3:
        raise ValueError("min_interactions_per_user must be at least 3")
    user_ids, counts = np.unique(columns["raw_user_ids"], return_counts=True)
    retained_users = user_ids[counts >= minimum]
    return _select(columns, np.isin(columns["raw_user_ids"], retained_users))


def _apply_duplicate_policy(
    columns: dict[str, np.ndarray], policy: str
) -> dict[str, np.ndarray]:
    if policy == "keep_all":
        return columns
    if policy != "keep_latest":
        raise ValueError(f"unsupported duplicate policy: {policy}")

    order = np.lexsort(
        (
            columns["source_indices"],
            columns["timestamps"],
            columns["raw_item_ids"],
            columns["raw_user_ids"],
        )
    )
    ordered_users = columns["raw_user_ids"][order]
    ordered_items = columns["raw_item_ids"][order]
    is_last = np.ones(order.size, dtype=bool)
    is_last[:-1] = (
        (ordered_users[:-1] != ordered_users[1:])
        | (ordered_items[:-1] != ordered_items[1:])
    )
    return _select(columns, order[is_last])


def _map_by_first_appearance(
    raw_ids: np.ndarray,
) -> tuple[dict[int, int], np.ndarray]:
    unique_ids, first_positions, inverse = np.unique(
        raw_ids, return_index=True, return_inverse=True
    )
    first_appearance_order = np.argsort(first_positions, kind="stable")
    processed_for_unique = np.empty(unique_ids.size, dtype=np.int64)
    processed_for_unique[first_appearance_order] = np.arange(
        unique_ids.size, dtype=np.int64
    )
    processed_ids = processed_for_unique[inverse]
    mapping = {
        int(unique_ids[index]): int(processed_for_unique[index])
        for index in range(unique_ids.size)
    }
    return mapping, processed_ids


def preprocess_ml1m(
    ratings_path: str | Path,
    *,
    min_interactions_per_user: int = 3,
    duplicate_policy: str = "keep_latest",
    implicit_feedback: str = "all_observed",
    seed: int = 2026,
    source_reference: str | None = None,
) -> DatasetBundle:
    """Load and preprocess one MovieLens-1M ratings.dat file."""
    path = Path(ratings_path)
    raw_columns = load_ml1m_ratings(path)
    counts: dict[str, int] = {"raw_interactions": len(raw_columns["ratings"])}

    columns = _select(raw_columns, _valid_record_mask(raw_columns))
    counts["after_invalid_filter"] = len(columns["ratings"])

    columns = _filter_minimum_interactions(
        columns, min_interactions_per_user
    )
    counts["after_initial_user_filter"] = len(columns["ratings"])

    columns = _apply_duplicate_policy(columns, duplicate_policy)
    counts["after_duplicate_policy"] = len(columns["ratings"])

    # Deduplication can reduce a user's retained count below the split minimum.
    columns = _filter_minimum_interactions(
        columns, min_interactions_per_user
    )
    if not len(columns["ratings"]):
        raise ValueError("no interactions remain after preprocessing")

    if implicit_feedback != "all_observed":
        raise ValueError(f"unsupported implicit feedback rule: {implicit_feedback}")

    canonical_order = np.argsort(columns["source_indices"], kind="stable")
    columns = _select(columns, canonical_order)

    user_id_map, user_ids = _map_by_first_appearance(columns["raw_user_ids"])
    item_id_map, item_ids = _map_by_first_appearance(columns["raw_item_ids"])
    interaction_ids = np.arange(len(user_ids), dtype=np.int64)
    interactions = InteractionTable(
        interaction_ids=interaction_ids,
        user_ids=user_ids,
        item_ids=item_ids,
        timestamps=columns["timestamps"].astype(np.int64, copy=False),
        values=np.ones(len(user_ids), dtype=np.float32),
    )
    splits = leave_one_out_split(interactions)
    counts["retained_interactions"] = len(interactions)
    counts["users"] = len(user_id_map)
    counts["items"] = len(item_id_map)

    preprocessing_configuration: dict[str, Any] = {
        "min_interactions_per_user": min_interactions_per_user,
        "duplicate_policy": duplicate_policy,
        "implicit_feedback": implicit_feedback,
        "canonical_order": "source_order",
    }
    metadata: dict[str, Any] = {
        "raw_ratings_reference": source_reference or path.name,
        "raw_ratings_sha256": _sha256_file(path),
        "preprocessing_configuration": preprocessing_configuration,
        "preprocessing_seed": seed,
        "mapping_policy": "first_appearance_in_canonical_source_order",
        "minimum_interaction_rule": {
            "entity": "user",
            "minimum": min_interactions_per_user,
            "rechecked_after_deduplication": True,
        },
        "duplicate_policy": duplicate_policy,
        "implicit_feedback_conversion": {
            "strategy": implicit_feedback,
            "positive_value": 1.0,
        },
        "split_policy": {
            "strategy": "leave_one_out_temporal",
            "order": ["timestamp", "interaction_id"],
        },
        "counts": counts,
    }
    return DatasetBundle(
        name="ml1m",
        schema_version=SCHEMA_VERSION,
        interactions=interactions,
        splits=splits,
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        metadata=metadata,
    )


def preprocess_ml1m_from_root(
    ratings_relative_path: str | Path,
    *,
    data_root: str | Path | None = None,
    min_interactions_per_user: int = 3,
    duplicate_policy: str = "keep_latest",
    implicit_feedback: str = "all_observed",
    seed: int = 2026,
) -> DatasetBundle:
    """Resolve a relative ML1M path and run base preprocessing."""
    ratings_path = resolve_data_path(
        ratings_relative_path,
        data_root=data_root,
    )
    return preprocess_ml1m(
        ratings_path,
        min_interactions_per_user=min_interactions_per_user,
        duplicate_policy=duplicate_policy,
        implicit_feedback=implicit_feedback,
        seed=seed,
        source_reference=Path(ratings_relative_path).as_posix(),
    )
