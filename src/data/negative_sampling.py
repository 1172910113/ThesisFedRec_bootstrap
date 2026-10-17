"""Shared deterministic negative-item sampling."""

from collections.abc import Sequence

import numpy as np

from src.data.contracts import DatasetBundle

_SPLIT_FIELDS = {
    "train": "train_interaction_ids",
    "valid": "valid_interaction_ids",
    "test": "test_interaction_ids",
}


def sample_negative_items(
    bundle: DatasetBundle,
    positive_interaction_ids: Sequence[int] | np.ndarray,
    *,
    negatives_per_positive: int,
    strategy: str = "uniform",
    seed: int = 2026,
    excluded_splits: Sequence[str] = ("train", "valid", "test"),
    replace: bool = False,
) -> dict[int, tuple[int, ...]]:
    """Return negative item IDs keyed by positive interaction ID."""
    if negatives_per_positive < 0:
        raise ValueError("negatives_per_positive must be non-negative")
    if strategy != "uniform":
        raise ValueError(f"unsupported negative-sampling strategy: {strategy}")

    positive_ids = np.asarray(positive_interaction_ids, dtype=np.int64)
    if positive_ids.ndim != 1:
        raise ValueError("positive_interaction_ids must be one-dimensional")
    if np.unique(positive_ids).size != positive_ids.size:
        raise ValueError("positive_interaction_ids must be unique")
    if np.any(positive_ids < 0) or np.any(positive_ids >= len(bundle.interactions)):
        raise ValueError("positive_interaction_ids contain an out-of-range ID")

    unknown_splits = set(excluded_splits) - set(_SPLIT_FIELDS)
    if unknown_splits:
        names = ", ".join(sorted(unknown_splits))
        raise ValueError(f"unknown excluded split names: {names}")

    known_by_user: list[set[int]] = [
        set() for _ in range(len(bundle.user_id_map))
    ]
    for split_name in excluded_splits:
        interaction_ids = getattr(bundle.splits, _SPLIT_FIELDS[split_name])
        users = bundle.interactions.user_ids[interaction_ids]
        items = bundle.interactions.item_ids[interaction_ids]
        for user_id, item_id in zip(users.tolist(), items.tolist(), strict=True):
            known_by_user[user_id].add(item_id)

    num_items = len(bundle.item_id_map)
    rng = np.random.default_rng(seed)
    samples: dict[int, tuple[int, ...]] = {}
    excluded_by_user: dict[int, np.ndarray] = {}
    for interaction_id in np.sort(positive_ids):
        user_id = int(bundle.interactions.user_ids[interaction_id])
        if user_id not in excluded_by_user:
            excluded_by_user[user_id] = np.sort(
                np.fromiter(known_by_user[user_id], dtype=np.int64)
            )
        positive_item_id = int(bundle.interactions.item_ids[interaction_id])
        excluded = excluded_by_user[user_id]
        position = np.searchsorted(excluded, positive_item_id)
        if position == excluded.size or excluded[position] != positive_item_id:
            # np.insert returns a target-specific copy, preserving the user cache.
            excluded = np.insert(excluded, position, positive_item_id)
        eligible_count = num_items - excluded.size
        if not replace and negatives_per_positive > eligible_count:
            raise ValueError(
                f"user {user_id} has only {eligible_count} eligible negative items"
            )
        if eligible_count == 0 and negatives_per_positive:
            raise ValueError(f"user {user_id} has no eligible negative items")

        ranks = rng.choice(
            eligible_count,
            size=negatives_per_positive,
            replace=replace,
        )
        selected = _eligible_items_from_ranks(ranks, excluded)
        samples[int(interaction_id)] = tuple(
            int(item_id) for item_id in np.sort(selected)
        )
    return samples


def _eligible_items_from_ranks(
    ranks: np.ndarray, excluded_item_ids: np.ndarray
) -> np.ndarray:
    """Map ranks in the ascending complement of sorted unique exclusions."""
    gaps = excluded_item_ids - np.arange(excluded_item_ids.size, dtype=np.int64)
    return ranks + np.searchsorted(gaps, ranks, side="right")
