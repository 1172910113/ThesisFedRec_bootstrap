from collections.abc import Sequence
from copy import deepcopy
from itertools import combinations

import numpy as np
import pytest

from src.data.contracts import (
    DatasetBundle,
    InteractionSplits,
    InteractionTable,
)
from src.data.negative_sampling import sample_negative_items


# Test-only materialized oracle copied from the accepted sampler at
# 705e06921e16c85be39044786d57cf9d05548f0f. Keep it independent of production
# helpers: only the function name differs from that revision's function.
_SPLIT_FIELDS = {
    "train": "train_interaction_ids",
    "valid": "valid_interaction_ids",
    "test": "test_interaction_ids",
}


def _materialized_baseline(
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

    all_items = np.arange(len(bundle.item_id_map), dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples: dict[int, tuple[int, ...]] = {}
    eligible_by_user: dict[int, np.ndarray] = {}
    for interaction_id in np.sort(positive_ids):
        user_id = int(bundle.interactions.user_ids[interaction_id])
        if user_id not in eligible_by_user:
            known_items = np.fromiter(
                known_by_user[user_id],
                dtype=np.int64,
            )
            eligible_by_user[user_id] = np.setdiff1d(
                all_items,
                known_items,
                assume_unique=True,
            )
        positive_item_id = int(bundle.interactions.item_ids[interaction_id])
        eligible = eligible_by_user[user_id]
        eligible = eligible[eligible != positive_item_id]
        if not replace and negatives_per_positive > eligible.size:
            raise ValueError(
                f"user {user_id} has only {eligible.size} eligible negative items"
            )
        if eligible.size == 0 and negatives_per_positive:
            raise ValueError(f"user {user_id} has no eligible negative items")

        selected = rng.choice(
            eligible,
            size=negatives_per_positive,
            replace=replace,
        )
        samples[int(interaction_id)] = tuple(
            int(item_id) for item_id in np.sort(selected)
        )
    return samples


def _make_catalog_bundle(num_items: int = 180) -> DatasetBundle:
    """Two sparse histories plus one full history covering the mapped catalog."""
    histories = (
        (0, 1, 2, 5, num_items - 2, num_items - 1),
        (3, 4, 7, num_items - 1),
        tuple(range(num_items)),
    )
    users, items = [], []
    train, valid, test = [], [], []
    for user_id, history in enumerate(histories):
        start = len(items)
        users.extend([user_id] * len(history))
        items.extend(history)
        train.extend(range(start, len(items) - 2))
        valid.append(len(items) - 2)
        test.append(len(items) - 1)
    count = len(items)
    return DatasetBundle(
        name="synthetic-catalog",
        schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=np.arange(count, dtype=np.int64),
            user_ids=np.array(users, dtype=np.int64),
            item_ids=np.array(items, dtype=np.int64),
            timestamps=np.arange(count, dtype=np.int64),
            values=np.ones(count, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=np.array(train, dtype=np.int64),
            valid_interaction_ids=np.array(valid, dtype=np.int64),
            test_interaction_ids=np.array(test, dtype=np.int64),
        ),
        user_id_map={f"user-{i}": i for i in range(len(histories))},
        item_id_map={f"item-{i}": i for i in range(num_items)},
        metadata={},
    )


_REAL_DEFAULT_RNG = np.random.default_rng
_TRAINING_SEED = int(
    np.random.SeedSequence([2026, 3, 17, 2])
    .generate_state(1, dtype=np.uint64)[0]
)


class _RecordingGenerator:
    """Observe real NumPy choice calls, including their post-error RNG state."""

    def __init__(self, generator: np.random.Generator) -> None:
        self.generator = generator
        self.calls = []
        self.integer_populations = []
        self.states = [deepcopy(generator.bit_generator.state)]

    def choice(self, population, *args, **kwargs):
        is_integer = np.isscalar(population)
        self.integer_populations.append(is_integer)
        count = int(population) if is_integer else len(population)
        self.calls.append((count, args, dict(kwargs)))
        try:
            return self.generator.choice(population, *args, **kwargs)
        finally:
            self.states.append(deepcopy(self.generator.bit_generator.state))


def _observe_call(monkeypatch, sampler, bundle, targets, **kwargs):
    generators, factory_calls = [], []

    def record_factory(*args, **options):
        factory_calls.append((args, dict(options)))
        generator = _RecordingGenerator(_REAL_DEFAULT_RNG(*args, **options))
        generators.append(generator)
        return generator

    with monkeypatch.context() as patch:
        patch.setattr(np.random, "default_rng", record_factory)
        try:
            result = sampler(bundle, targets, **kwargs)
        except Exception as error:
            return None, (type(error), str(error)), factory_calls, generators
    return result, None, factory_calls, generators


def _assert_matches_baseline(monkeypatch, bundle, targets, **kwargs):
    arrays = [
        getattr(bundle.interactions, name)
        for name in ("interaction_ids", "user_ids", "item_ids", "timestamps", "values")
    ] + [getattr(bundle.splits, name) for name in _SPLIT_FIELDS.values()]
    snapshots = [array.copy() for array in arrays]
    targets_before, kwargs_before = deepcopy(targets), deepcopy(kwargs)
    mappings_before = deepcopy((bundle.user_id_map, bundle.item_id_map))
    global_state = np.random.get_state()
    observations = []
    for sampler in (_materialized_baseline, sample_negative_items):
        observations.append(
            _observe_call(monkeypatch, sampler, bundle, targets, **kwargs)
        )
        for array, snapshot in zip(arrays, snapshots, strict=True):
            np.testing.assert_array_equal(array, snapshot)
            assert array.dtype == snapshot.dtype
        np.testing.assert_equal(targets, targets_before)
        assert kwargs == kwargs_before
        assert (bundle.user_id_map, bundle.item_id_map) == mappings_before
        np.testing.assert_equal(np.random.get_state(), global_state)

    expected, actual = observations
    assert actual[:3] == expected[:3]  # Results, error type/message, RNG factories.
    assert len(actual[2]) <= 1
    assert len(actual[3]) == len(expected[3])
    for observed, baseline in zip(actual[3], expected[3], strict=True):
        assert observed.calls == baseline.calls
        assert observed.states == baseline.states
    if actual[0] is not None:
        assert list(actual[0]) == list(expected[0])
        for key, values in actual[0].items():
            assert type(key) is int
            assert type(values) is tuple
            assert all(type(value) is int for value in values)
            assert values == tuple(sorted(values))
    return actual


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


_EXCLUSION_POLICIES = tuple(
    subset
    for size in range(4)
    for subset in combinations(("train", "valid", "test"), size)
) + (("train", "valid", "train", "test"), ("test", "train", "valid"))


@pytest.mark.parametrize("excluded_splits", _EXCLUSION_POLICIES)
@pytest.mark.parametrize("seed", [0, 2026, 4052, 4053, _TRAINING_SEED])
@pytest.mark.parametrize("replace", [False, True])
def test_public_results_and_rng_match_baseline_across_policies(
    monkeypatch: pytest.MonkeyPatch, excluded_splits, seed, replace
) -> None:
    bundle = _make_catalog_bundle()
    targets = np.array([0, 1, 4, 5, 6, 8, 9], dtype=np.int64)
    for count in (4, 99):
        ordered_result = None
        for order in (targets, targets[::-1], targets[[3, 0, 6, 2, 5, 1, 4]]):
            result, error, factories, generators = _assert_matches_baseline(
                monkeypatch, bundle, order,
                negatives_per_positive=count, seed=seed,
                excluded_splits=excluded_splits, replace=replace,
            )
            assert error is None
            assert len(factories) == 1
            assert len(generators[0].calls) == len(targets)
            if ordered_result is None:
                ordered_result = result
            assert result == ordered_result
            assert list(result) == targets.tolist()


@pytest.mark.parametrize("include_genuine_negative", [False, True])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_empty_singleton_and_oversized_pools_match_baseline(
    monkeypatch, include_genuine_negative, replace, count
) -> None:
    bundle = _make_sampling_bundle(
        include_genuine_negative=include_genuine_negative
    )
    result, error, factories, generators = _assert_matches_baseline(
        monkeypatch, bundle, [2], negatives_per_positive=count,
        excluded_splits=("train", "valid"), replace=replace,
    )
    capacity = int(include_genuine_negative)
    assert len(factories) == 1
    if not replace and count > capacity:
        assert error == (
            ValueError, f"user 0 has only {capacity} eligible negative items"
        )
    elif not capacity and count:
        assert error == (ValueError, "user 0 has no eligible negative items")
    else:
        assert error is None
        assert result == {2: (3,) * count}
        assert len(generators[0].calls) == 1  # Even zero out of an empty pool.
    if error is not None:
        assert generators[0].calls == []


@pytest.mark.parametrize("count", [0, 99])
def test_empty_targets_still_construct_one_rng(monkeypatch, count) -> None:
    result, error, factories, generators = _assert_matches_baseline(
        monkeypatch, _make_catalog_bundle(), np.array([], dtype=np.int64),
        negatives_per_positive=count,
    )
    assert result == {}
    assert error is None
    assert len(factories) == 1
    assert generators[0].calls == []


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("count", [174, 175])
def test_exact_and_insufficient_capacity_match_baseline(
    monkeypatch, replace, count
) -> None:
    result, error, _, generators = _assert_matches_baseline(
        monkeypatch, _make_catalog_bundle(), [5],
        negatives_per_positive=count, replace=replace,
    )
    if count == 175 and not replace:
        assert error == (ValueError, "user 0 has only 174 eligible negative items")
        assert generators[0].calls == []
    else:
        assert error is None
        assert len(result[5]) == count
        if replace:
            assert len(set(result[5])) < count
        else:
            assert result[5] == tuple(i for i in range(180) if i not in {
                0, 1, 2, 5, 178, 179
            })


@pytest.mark.parametrize("replace", [False, True])
def test_later_capacity_failure_preserves_prior_rng_consumption(
    monkeypatch, replace
) -> None:
    # User 1 succeeds, then user 2's complete catalog history leaves no pool.
    result, error, factories, generators = _assert_matches_baseline(
        monkeypatch, _make_catalog_bundle(), [10, 6],
        negatives_per_positive=4, replace=replace,
    )
    assert result is None
    message = "no" if replace else "only 0"
    assert error == (ValueError, f"user 2 has {message} eligible negative items")
    assert len(factories) == 1
    assert len(generators[0].calls) == 1
    assert generators[0].states[0] != generators[0].states[-1]


@pytest.mark.parametrize(
    "targets, options, message, factory_count",
    [
        ([[1, 1]], dict(negatives_per_positive=-1, strategy="other",
                       excluded_splits=("unknown",)),
         "negatives_per_positive must be non-negative", 0),
        ([[1, 1]], dict(strategy="other", excluded_splits=("unknown",)),
         "unsupported negative-sampling strategy: other", 0),
        ([[1, 1]], dict(excluded_splits=("unknown",)),
         "positive_interaction_ids must be one-dimensional", 0),
        (1, {}, "positive_interaction_ids must be one-dimensional", 0),
        ([-1, -1], dict(excluded_splits=("unknown",)),
         "positive_interaction_ids must be unique", 0),
        ([1, 1], {}, "positive_interaction_ids must be unique", 0),
        ([-1, 0], dict(excluded_splits=("unknown",), seed=-1),
         "positive_interaction_ids contain an out-of-range ID", 0),
        ([999999], {}, "positive_interaction_ids contain an out-of-range ID", 0),
        ([0], dict(excluded_splits=("z", "a"), seed=-1),
         "unknown excluded split names: a, z", 0),
        ([], dict(excluded_splits=("unknown",)),
         "unknown excluded split names: unknown", 0),
        (["not-an-id"], dict(excluded_splits=("unknown",)), None, 0),
        (None, {}, None, 0),
        ([0], dict(negatives_per_positive="4"), None, 0),
        ([0], dict(negatives_per_positive=1.5), None, 1),
        ([0], dict(excluded_splits=None), None, 0),
        ([10], dict(seed=-1), None, 1),
        ([], dict(seed=-1), None, 1),
    ],
)
def test_validation_and_error_precedence_match_baseline(
    monkeypatch, targets, options, message, factory_count
) -> None:
    kwargs = {"negatives_per_positive": 4, **options}
    result, error, factories, _ = _assert_matches_baseline(
        monkeypatch, _make_catalog_bundle(), targets, **kwargs
    )
    assert result is None
    assert error is not None
    assert len(factories) == factory_count
    if message is not None:
        assert error == (ValueError, message)


@pytest.mark.parametrize("targets", [[True, False], [1.9, 2.1], ["1", "2"]])
def test_existing_input_coercions_are_not_tightened(monkeypatch, targets) -> None:
    result, error, _, _ = _assert_matches_baseline(
        monkeypatch, _make_catalog_bundle(), targets,
        negatives_per_positive=np.int64(4), seed=np.uint64(_TRAINING_SEED),
        excluded_splits=["valid", "train", "valid"],
    )
    assert error is None
    assert result


@pytest.mark.parametrize("seed", [0, 4053, _TRAINING_SEED])
@pytest.mark.parametrize(
    "num_items, count",
    [(10003, 4998)] + [(20003, k) for k in (0, 4, 99, 399, 400, 1000, 19997)],
)
def test_sparse_and_dense_numpy_choice_paths_match_baseline(
    monkeypatch, seed, num_items, count
) -> None:
    # NumPy 1.26.4 uses its dense path above population 10,000 and sample
    # size population//50 (default shuffle). Cover both sides, and full draws.
    result, error, _, generators = _assert_matches_baseline(
        monkeypatch, _make_catalog_bundle(num_items), [5, 0],
        negatives_per_positive=count, seed=seed,
    )
    assert error is None
    assert list(result) == [0, 5]
    assert len(generators[0].calls) == 2


# Frozen before changing production, from the AST-verified baseline oracle
# under Python 3.11.16 / NumPy 1.26.4, not from the rank-based implementation.
@pytest.mark.parametrize(
    "targets, options, expected",
    [
        ([9, 5, 4, 0], dict(negatives_per_positive=4, seed=2026), {
            0: (8, 34, 115, 149),
            4: (65, 67, 114, 148),
            5: (34, 117, 127, 152),
            9: (52, 129, 163, 170),
        }),
        ([4], dict(negatives_per_positive=99, seed=4052), {4: (
            3, 8, 9, 11, 12, 13, 14, 17, 19, 20, 24, 26, 27, 29, 30, 32,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 49, 51, 55,
            57, 58, 59, 60, 61, 62, 65, 68, 69, 71, 73, 74, 77, 78, 79,
            80, 81, 85, 87, 88, 89, 91, 93, 94, 95, 97, 98, 100, 101, 103,
            104, 109, 110, 111, 115, 117, 118, 119, 120, 122, 124, 125, 126,
            127, 130, 133, 137, 139, 141, 143, 144, 146, 148, 149, 153, 154,
            155, 156, 159, 160, 166, 167, 168, 170, 171, 173, 176, 177,
        )}),
        ([9, 8, 5], dict(negatives_per_positive=4, seed=_TRAINING_SEED,
                         excluded_splits=("train",)), {
            5: (30, 65, 84, 93),
            8: (11, 72, 84, 98),
            9: (35, 83, 105, 163),
        }),
    ],
)
def test_frozen_baseline_outputs(monkeypatch, targets, options, expected) -> None:
    result, error, _, _ = _assert_matches_baseline(
        monkeypatch, _make_catalog_bundle(), targets, **options
    )
    assert error is None
    assert result == expected
    assert list(result) == list(expected)


def test_only_exclusion_sized_arrays_and_integer_choice_populations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.data.negative_sampling as sampling

    bundle = _make_catalog_bundle(20003)
    targets = np.array([0, 1, 4, 5, 6, 8, 9], dtype=np.int64)
    real_arange, real_fromiter = np.arange, np.fromiter
    real_map = sampling._eligible_items_from_ranks
    ranges, known_arrays, exclusions = [], [], []

    def bounded_arange(stop, *args, **kwargs):
        # No catalog-sized Python allocation; exclusion-sized arrays are valid.
        assert 0 <= stop <= 6
        ranges.append(stop)
        return real_arange(stop, *args, **kwargs)

    def record_known(*args, **kwargs):
        array = real_fromiter(*args, **kwargs)
        known_arrays.append(array)
        assert array.size <= 6
        return array

    def record_mapping(ranks, excluded):
        exclusions.append((excluded, excluded.copy()))
        return real_map(ranks, excluded)

    def reject_complement(*args, **kwargs):
        pytest.fail("the production sampler must not materialize complements")

    with monkeypatch.context() as patch:
        patch.setattr(np, "arange", bounded_arange)
        patch.setattr(np, "fromiter", record_known)
        patch.setattr(np, "setdiff1d", reject_complement)
        patch.setattr(sampling, "_eligible_items_from_ranks", record_mapping)
        result, error, factories, generators = _observe_call(
            patch, sample_negative_items, bundle, targets,
            negatives_per_positive=99, excluded_splits=("train",),
        )
    assert error is None
    assert len(result) == len(targets)
    assert len(factories) == 1
    assert generators[0].integer_populations == [True] * len(targets)
    assert len(known_arrays) == 2  # Only the two encountered users, once each.
    assert sorted(array.size for array in known_arrays) == [2, 4]
    assert len(ranges) == len(targets)
    for excluded, snapshot in exclusions:
        np.testing.assert_array_equal(excluded, snapshot)
        assert np.all(excluded[1:] > excluded[:-1])
