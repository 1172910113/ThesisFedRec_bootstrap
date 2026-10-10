"""Shared deterministic top-K ranking evaluation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import log2
from typing import Literal

import numpy as np

from src.data.contracts import DatasetBundle
from src.data.negative_sampling import sample_negative_items

EvaluationSplit = Literal["valid", "test"]
ScoreFunction = Callable[
    [np.ndarray, np.ndarray],
    Sequence[float] | np.ndarray,
]


@dataclass(frozen=True, slots=True)
class EvaluationCandidate:
    """One held-out positive and its fixed negative item IDs."""

    target_interaction_id: int
    user_id: int
    positive_item_id: int
    negative_item_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "target_interaction_id",
            "user_id",
            "positive_item_id",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if any(
            isinstance(item_id, bool)
            or not isinstance(item_id, int)
            or item_id < 0
            for item_id in self.negative_item_ids
        ):
            raise ValueError("negative_item_ids must contain non-negative integers")
        if len(set(self.negative_item_ids)) != len(self.negative_item_ids):
            raise ValueError("negative_item_ids must not contain duplicates")
        if self.positive_item_id in self.negative_item_ids:
            raise ValueError("the positive item cannot also be a negative item")

    @property
    def item_ids(self) -> tuple[int, ...]:
        """Return the positive once followed by the fixed negative items."""
        return (self.positive_item_id, *self.negative_item_ids)


@dataclass(frozen=True, slots=True)
class EvaluationCandidateSet:
    """Reusable candidates for exactly one validation or test target set."""

    split: EvaluationSplit
    base_seed: int
    sampling_seed: int
    candidates: tuple[EvaluationCandidate, ...]

    def __post_init__(self) -> None:
        if self.split not in {"valid", "test"}:
            raise ValueError("split must be 'valid' or 'test'")
        for name in ("base_seed", "sampling_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        interaction_ids = [
            candidate.target_interaction_id for candidate in self.candidates
        ]
        if len(set(interaction_ids)) != len(interaction_ids):
            raise ValueError("candidate target interaction IDs must be unique")


def derive_evaluation_seed(base_seed: int, split: EvaluationSplit) -> int:
    """Derive disjoint, inspectable validation and test seed streams."""
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or not 0 <= base_seed < 2**32
    ):
        raise ValueError("base_seed must be an integer in [0, 2**32)")
    if split not in {"valid", "test"}:
        raise ValueError("split must be 'valid' or 'test'")
    stream_id = 0 if split == "valid" else 1
    return 2 * base_seed + stream_id


def build_evaluation_candidates(
    bundle: DatasetBundle,
    target_interaction_ids: Sequence[int] | np.ndarray,
    *,
    split: EvaluationSplit,
    negatives_per_positive: int,
    strategy: str = "uniform",
    base_seed: int = 2026,
    excluded_splits: Sequence[str] = ("train", "valid", "test"),
) -> EvaluationCandidateSet:
    """Generate one fixed no-replacement candidate set for a data split."""
    target_ids = _target_ids(target_interaction_ids)
    split_ids = getattr(bundle.splits, f"{split}_interaction_ids", None)
    if split_ids is None:
        raise ValueError("split must be 'valid' or 'test'")
    if not np.all(np.isin(target_ids, split_ids)):
        raise ValueError(f"all target interaction IDs must belong to {split}")

    ordered_target_ids = np.sort(target_ids)
    sampling_seed = derive_evaluation_seed(base_seed, split)
    negatives = sample_negative_items(
        bundle,
        ordered_target_ids,
        negatives_per_positive=negatives_per_positive,
        strategy=strategy,
        seed=sampling_seed,
        excluded_splits=excluded_splits,
        replace=False,
    )

    candidates = tuple(
        EvaluationCandidate(
            target_interaction_id=int(interaction_id),
            user_id=int(bundle.interactions.user_ids[interaction_id]),
            positive_item_id=int(bundle.interactions.item_ids[interaction_id]),
            negative_item_ids=negatives[int(interaction_id)],
        )
        for interaction_id in ordered_target_ids
    )
    return EvaluationCandidateSet(
        split=split,
        base_seed=base_seed,
        sampling_seed=sampling_seed,
        candidates=candidates,
    )


def rank_item_ids(
    item_ids: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[int, ...]:
    """Rank by descending score and then ascending processed item ID."""
    items = np.asarray(item_ids)
    values = np.asarray(scores, dtype=np.float64)
    if items.ndim != 1 or values.ndim != 1:
        raise ValueError("item_ids and scores must be one-dimensional")
    if len(items) != len(values) or not len(items):
        raise ValueError("item_ids and scores must have the same non-zero length")
    if not np.issubdtype(items.dtype, np.integer):
        raise TypeError("item_ids must contain integers")
    items = items.astype(np.int64, copy=False)
    if np.any(items < 0) or np.unique(items).size != items.size:
        raise ValueError("item_ids must be unique non-negative integers")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")

    order = np.lexsort((items, -values))
    return tuple(int(item_id) for item_id in items[order])


def hit_rate_at_k(positive_rank: int, k: int) -> float:
    """Return leave-one-out HR@K for a one-based positive rank."""
    _validate_rank_and_k(positive_rank, k)
    return float(positive_rank <= k)


def ndcg_at_k(positive_rank: int, k: int) -> float:
    """Return leave-one-out NDCG@K for a one-based positive rank."""
    _validate_rank_and_k(positive_rank, k)
    if positive_rank > k:
        return 0.0
    return 1.0 / log2(positive_rank + 1)


def evaluate_candidates(
    candidate_set: EvaluationCandidateSet,
    score_function: ScoreFunction,
    *,
    k_values: Sequence[int],
) -> dict[str, float | int]:
    """Score fixed candidates and average HR/NDCG across target interactions."""
    ks = _k_values(k_values)
    if not candidate_set.candidates:
        raise ValueError("at least one evaluation candidate is required")

    hr_totals = {k: 0.0 for k in ks}
    ndcg_totals = {k: 0.0 for k in ks}
    for candidate in candidate_set.candidates:
        item_ids = np.asarray(candidate.item_ids, dtype=np.int64)
        user_ids = np.full(item_ids.shape, candidate.user_id, dtype=np.int64)
        scores = np.asarray(score_function(user_ids, item_ids), dtype=np.float64)
        ranked_items = rank_item_ids(item_ids, scores)
        positive_rank = ranked_items.index(candidate.positive_item_id) + 1
        for k in ks:
            hr_totals[k] += hit_rate_at_k(positive_rank, k)
            ndcg_totals[k] += ndcg_at_k(positive_rank, k)

    count = len(candidate_set.candidates)
    result: dict[str, float | int] = {"num_targets": count}
    for k in ks:
        result[f"hr@{k}"] = hr_totals[k] / count
        result[f"ndcg@{k}"] = ndcg_totals[k] / count
    return result


def _target_ids(
    target_interaction_ids: Sequence[int] | np.ndarray,
) -> np.ndarray:
    target_ids = np.asarray(target_interaction_ids)
    if target_ids.ndim != 1 or not target_ids.size:
        raise ValueError("target_interaction_ids must be a non-empty 1D sequence")
    if not np.issubdtype(target_ids.dtype, np.integer):
        raise TypeError("target_interaction_ids must contain integers")
    target_ids = target_ids.astype(np.int64, copy=False)
    if np.any(target_ids < 0) or np.unique(target_ids).size != target_ids.size:
        raise ValueError("target_interaction_ids must be unique and non-negative")
    return target_ids


def _k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    values = tuple(k_values)
    if not values:
        raise ValueError("k_values must not be empty")
    if any(
        isinstance(k, bool)
        or not isinstance(k, (int, np.integer))
        or k <= 0
        for k in values
    ):
        raise ValueError("k_values must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("k_values must not contain duplicates")
    return tuple(sorted(int(k) for k in values))


def _validate_rank_and_k(positive_rank: int, k: int) -> None:
    for name, value in (("positive_rank", positive_rank), ("k", k)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
