import numpy as np
import pytest

from src.data.contracts import (
    DatasetBundle,
    InteractionSplits,
    InteractionTable,
)
from src.data.negative_sampling import sample_negative_items


def _make_sampling_bundle(
    *, include_genuine_negative: bool
) -> DatasetBundle:
    if include_genuine_negative:
        user_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        item_ids = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
        train_ids = np.array([0, 3], dtype=np.int64)
        valid_ids = np.array([1, 4], dtype=np.int64)
        test_ids = np.array([2, 5], dtype=np.int64)
        user_id_map = {10: 0, 20: 1}
        item_id_map = {100: 0, 101: 1, 102: 2, 103: 3}
    else:
        user_ids = np.array([0, 0, 0], dtype=np.int64)
        item_ids = np.array([0, 1, 2], dtype=np.int64)
        train_ids = np.array([0], dtype=np.int64)
        valid_ids = np.array([1], dtype=np.int64)
        test_ids = np.array([2], dtype=np.int64)
        user_id_map = {10: 0}
        item_id_map = {100: 0, 101: 1, 102: 2}

    num_interactions = len(user_ids)
    return DatasetBundle(
        name="synthetic",
        schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=np.arange(num_interactions, dtype=np.int64),
            user_ids=user_ids,
            item_ids=item_ids,
            timestamps=np.arange(num_interactions, dtype=np.int64),
            values=np.ones(num_interactions, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=train_ids,
            valid_interaction_ids=valid_ids,
            test_interaction_ids=test_ids,
        ),
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        metadata={},
    )


def test_negative_samples_exclude_all_known_positives_and_are_deterministic(
    processed_bundle: DatasetBundle,
) -> None:
    positive_ids = processed_bundle.splits.test_interaction_ids

    first = sample_negative_items(
        processed_bundle,
        positive_ids,
        negatives_per_positive=3,
        seed=23,
    )
    second = sample_negative_items(
        processed_bundle,
        positive_ids[::-1],
        negatives_per_positive=3,
        seed=23,
    )

    assert first == second
    for interaction_id, negative_items in first.items():
        user_id = int(processed_bundle.interactions.user_ids[interaction_id])
        known_items = set(
            processed_bundle.interactions.item_ids[
                processed_bundle.interactions.user_ids == user_id
            ].tolist()
        )
        assert len(negative_items) == 3
        assert tuple(sorted(negative_items)) == negative_items
        assert known_items.isdisjoint(negative_items)
        positive_item = int(
            processed_bundle.interactions.item_ids[interaction_id]
        )
        assert positive_item not in negative_items


@pytest.mark.parametrize("replace", [False, True])
def test_target_exclusion_rejects_empty_true_negative_pool(
    replace: bool,
) -> None:
    bundle = _make_sampling_bundle(include_genuine_negative=False)

    with pytest.raises(ValueError, match="eligible negative items"):
        sample_negative_items(
            bundle,
            [2],
            negatives_per_positive=1,
            excluded_splits=("train", "valid"),
            replace=replace,
        )


def test_target_exclusion_returns_only_genuine_negative() -> None:
    bundle = _make_sampling_bundle(include_genuine_negative=True)

    samples = sample_negative_items(
        bundle,
        [2],
        negatives_per_positive=1,
        excluded_splits=("train", "valid"),
        seed=23,
    )

    assert samples == {2: (3,)}


def test_target_filtering_does_not_mutate_cached_user_pool() -> None:
    bundle = _make_sampling_bundle(include_genuine_negative=True)

    first = sample_negative_items(
        bundle,
        [1, 2],
        negatives_per_positive=2,
        excluded_splits=("train",),
        seed=23,
    )
    second = sample_negative_items(
        bundle,
        [2, 1],
        negatives_per_positive=2,
        excluded_splits=("train",),
        seed=23,
    )

    assert first == second == {1: (2, 3), 2: (1, 3)}
