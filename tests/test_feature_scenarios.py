import random
from dataclasses import fields, replace
from functools import partial
from itertools import permutations

import numpy as np
import pytest

from src.data import (
    DatasetBundle,
    FeatureView,
    InteractionSplits,
    InteractionTable,
    PartyData,
    build_complete_feature_scenario,
    build_hfl_partition,
    build_single_view_absence_scenario,
    feature_partition_checksum,
    validate_feature_partition,
)


@pytest.fixture
def scenario_bundle() -> DatasetBundle:
    """Synthetic in-memory data, with no loader or cache dependency."""
    views = {
        "item_tabular": FeatureView(
            name="item_tabular", entity_kind="item",
            entity_ids=np.array([0, 2], dtype=np.int64), modality="tabular",
            values=np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32),
            feature_names=("category", "brand"),
            missing_mask=np.array([[False, True], [True, False]]), metadata={},
        ),
        "item_text": FeatureView(
            name="item_text", entity_kind="item",
            entity_ids=np.array([0, 1, 2], dtype=np.int64), modality="text",
            values=("", "", ""), feature_names=(),
            missing_mask=np.ones(3, dtype=np.bool_), metadata={},
        ),
        "empty_view": FeatureView(
            name="empty_view", entity_kind="interaction",
            entity_ids=np.array([], dtype=np.int64), modality="tabular",
            values=np.empty((0, 1), dtype=np.float32), feature_names=("context",),
            missing_mask=np.empty((0, 1), dtype=np.bool_), metadata={},
        ),
        "user_info": FeatureView(
            name="user_info", entity_kind="user",
            entity_ids=np.array([0], dtype=np.int64), modality="tabular",
            values=np.array([[1.0]], dtype=np.float32), feature_names=("field",),
            missing_mask=None, metadata={},
        ),
    }
    bundle = DatasetBundle(
        name="arbitrary_dataset", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=np.arange(12, dtype=np.int64),
            user_ids=np.repeat(np.arange(4, dtype=np.int64), 3),
            item_ids=np.tile(np.arange(3, dtype=np.int64), 4),
            timestamps=np.arange(12, dtype=np.int64),
            values=np.ones(12, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=np.arange(0, 12, 3, dtype=np.int64),
            valid_interaction_ids=np.arange(1, 12, 3, dtype=np.int64),
            test_interaction_ids=np.arange(2, 12, 3, dtype=np.int64),
        ),
        user_id_map={f"user-{i}": i for i in range(4)},
        item_id_map={f"item-{i}": i for i in range(3)},
        metadata={}, feature_views=views,
    )
    return replace(
        bundle, partition_kind="hfl", partition_name="source_sample_partition",
        parties=build_hfl_partition(
            bundle, mode="grouped_round_robin", num_parties=3, seed=2026
        ),
    )


@pytest.fixture
def scenario_parties(scenario_bundle: DatasetBundle) -> tuple[PartyData, ...]:
    return tuple(
        replace(
            party, party_id=party_id,
            feature_view_names=("user_info",) if i != 1 else (),
            graph_view_names=("unchanged_graph",), owns_labels=i != 1,
            metadata={"nested": {"original_party": party.party_id}},
        )
        for i, (party, party_id) in enumerate(
            zip(scenario_bundle.parties, (101, 7, 42))
        )
    )


@pytest.fixture(params=["complete", "absence"])
def scenario_builder(request):
    if request.param == "complete":
        return build_complete_feature_scenario
    return partial(
        build_single_view_absence_scenario,
        missing_view="item_text", missing_party_count=1, seed=2026,
    )


def test_complete_access_and_hand_calculated_statistics(
    scenario_bundle, scenario_parties, monkeypatch,
):
    def unexpected_randomness(*args, **kwargs):
        pytest.fail("complete access must not construct or use a random generator")

    monkeypatch.setattr(np.random, "default_rng", unexpected_randomness)
    partition, parties, stats = build_complete_feature_scenario(
        scenario_bundle, scenario_parties,
        view_names=["item_text", "item_tabular"], name="complete",
    )

    assert tuple(partition.party_access) == (7, 42, 101)
    assert tuple(party.party_id for party in parties) == (7, 42, 101)
    for party in parties:
        assert party.feature_view_names == ("item_tabular", "item_text")
        accesses = partition.access_for_party(party.party_id)
        assert tuple(
            access.view_name for access in accesses
        ) == party.feature_view_names
        assert all(access.entity_ids is None for access in accesses)
        assert all(access.feature_names is None for access in accesses)
    assert stats == {
        "party_count": 3,
        "selected_views": ("item_tabular", "item_text"),
        "visible_view_counts_by_party": {7: 2, 42: 2, 101: 2},
        "visible_party_counts_by_view": {"item_tabular": 3, "item_text": 3},
        "featureless_party_count": 0,
        "available_party_view_pairs": 6,
        "possible_party_view_pairs": 6,
        "assignment_checksum": feature_partition_checksum(partition),
    }
    assert tuple(stats["visible_view_counts_by_party"]) == (7, 42, 101)
    assert tuple(stats["visible_party_counts_by_view"]) == ("item_tabular", "item_text")
    assert partition.metadata == {
        "recipe": "complete", "recipe_version": 1,
        "party_ids": [7, 42, 101], "view_names": ["item_tabular", "item_text"],
    }
    validate_feature_partition(partition, scenario_bundle, parties)


def test_complete_access_ignores_input_order(scenario_bundle, scenario_parties):
    first, _, first_stats = build_complete_feature_scenario(
        scenario_bundle, scenario_parties,
        view_names=("item_text", "item_tabular"), name="first",
    )
    second, parties, second_stats = build_complete_feature_scenario(
        scenario_bundle, list(reversed(scenario_parties)),
        view_names=["item_tabular", "item_text"], name="second",
    )
    assert first.party_access == second.party_access
    assert first.metadata == second.metadata
    assert first_stats == second_stats
    assert tuple(second.party_access) == (7, 42, 101)
    assert tuple(party.party_id for party in parties) == (7, 42, 101)


@pytest.mark.parametrize("missing_count", [0, 1, 3])
def test_absence_exact_counts_and_zero_equivalence(
    scenario_bundle, scenario_parties, missing_count,
):
    selected = ("item_text", "item_tabular")
    partition, parties, stats = build_single_view_absence_scenario(
        scenario_bundle, scenario_parties, view_names=selected, name="absence",
        missing_view="item_text", missing_party_count=missing_count, seed=2026,
    )
    positions = np.random.default_rng(2026).choice(3, size=missing_count, replace=False)
    expected_hidden = {(7, 42, 101)[position] for position in positions}
    hidden = {
        party.party_id for party in parties
        if "item_text" not in party.feature_view_names
    }
    assert hidden == expected_hidden
    assert len(hidden) == missing_count
    for party in parties:
        expected_names = (
            ("item_tabular",) if party.party_id in hidden else tuple(sorted(selected))
        )
        assert party.feature_view_names == expected_names
        assert tuple(
            access.view_name for access in partition.access_for_party(party.party_id)
        ) == expected_names
    assert stats["visible_view_counts_by_party"] == {
        party_id: 1 if party_id in hidden else 2 for party_id in (7, 42, 101)
    }
    assert stats["visible_party_counts_by_view"] == {
        "item_tabular": 3, "item_text": 3 - missing_count,
    }
    assert stats["available_party_view_pairs"] == 6 - missing_count
    assert stats["possible_party_view_pairs"] == 6
    assert stats["featureless_party_count"] == 0
    assert partition.metadata == {
        "recipe": "single_view_absence", "recipe_version": 1,
        "party_ids": [7, 42, 101], "view_names": ["item_tabular", "item_text"],
        "missing_view": "item_text", "missing_party_count": missing_count, "seed": 2026,
    }
    validate_feature_partition(partition, scenario_bundle, parties)
    if missing_count == 0:
        complete, complete_parties, complete_stats = build_complete_feature_scenario(
            scenario_bundle, scenario_parties,
            view_names=selected, name="different_name",
        )
        assert partition.party_access == complete.party_access
        assert [party.feature_view_names for party in parties] == [
            party.feature_view_names for party in complete_parties
        ]
        assert stats == complete_stats
        assert (
            feature_partition_checksum(partition) == feature_partition_checksum(complete)
        )


@pytest.mark.parametrize("seed", [0, 2026, 2**32 - 1])
def test_absence_seed_reproducibility_and_input_order(
    scenario_bundle, scenario_parties, seed,
):
    build = partial(
        build_single_view_absence_scenario, scenario_bundle,
        missing_view="item_text", missing_party_count=2, seed=seed, name="ordered",
    )
    expected, _, expected_stats = build(
        scenario_parties, view_names=("item_text", "item_tabular"),
    )
    for parties_in in permutations(scenario_parties):
        for views_in in permutations(("item_text", "item_tabular")):
            partition, parties, stats = build(parties_in, view_names=views_in)
            assert partition.party_access == expected.party_access
            assert partition.metadata == expected.metadata
            assert stats == expected_stats
            assert tuple(partition.party_access) == (7, 42, 101)
            assert tuple(party.party_id for party in parties) == (7, 42, 101)
            assert tuple(stats["visible_view_counts_by_party"]) == (7, 42, 101)
            assert tuple(stats["visible_party_counts_by_view"]) == (
                "item_tabular", "item_text",
            )
            validate_feature_partition(partition, scenario_bundle, parties)


def test_both_recipes_ignore_and_preserve_global_rng_state(
    scenario_bundle, scenario_parties, scenario_builder,
):
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    checksums = []
    try:
        for _ in range(2):
            np.random.random(17)
            random.random()
            before_numpy = np.random.get_state()
            before_python = random.getstate()
            partition, _, _ = scenario_builder(
                scenario_bundle, scenario_parties,
                view_names=("item_text", "item_tabular"), name="rng_isolation",
            )
            after_numpy = np.random.get_state()
            assert before_numpy[0] == after_numpy[0]
            np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
            assert before_numpy[2:] == after_numpy[2:]
            assert before_python == random.getstate()
            checksums.append(feature_partition_checksum(partition))
        assert checksums[0] == checksums[1]
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


@pytest.mark.parametrize("missing_count", [1, 3])
def test_single_selected_view_can_leave_featureless_parties(
    scenario_bundle, scenario_parties, missing_count,
):
    partition, parties, stats = build_single_view_absence_scenario(
        scenario_bundle, scenario_parties, view_names=("item_text",),
        missing_view="item_text", missing_party_count=missing_count,
        seed=0, name="one_view",
    )
    assert tuple(partition.party_access) == (7, 42, 101)
    assert sum(
        accesses == () for accesses in partition.party_access.values()
    ) == missing_count
    assert sum(party.feature_view_names == () for party in parties) == missing_count
    assert stats["featureless_party_count"] == missing_count
    assert stats["visible_party_counts_by_view"] == {"item_text": 3 - missing_count}
    assert stats["available_party_view_pairs"] == 3 - missing_count
    assert stats["possible_party_view_pairs"] == 3
    validate_feature_partition(partition, scenario_bundle, parties)


def test_zero_row_and_source_missing_views_count_as_assigned(
    scenario_bundle, scenario_parties,
):
    partition, parties, stats = build_complete_feature_scenario(
        scenario_bundle, scenario_parties,
        view_names=("user_info", "item_text", "empty_view"), name="source_missing",
    )
    assert stats["selected_views"] == ("empty_view", "item_text", "user_info")
    assert stats["visible_party_counts_by_view"] == {
        "empty_view": 3, "item_text": 3, "user_info": 3,
    }
    assert stats["featureless_party_count"] == 0
    assert stats["available_party_view_pairs"] == 9
    assert stats["possible_party_view_pairs"] == 9
    validate_feature_partition(partition, scenario_bundle, parties)


def test_new_parties_preserve_ownership_and_source_data(
    scenario_bundle, scenario_parties, scenario_builder,
):
    original_bundle_parties = scenario_bundle.parties
    original_summaries = [party.feature_view_names for party in scenario_parties]
    array_fields = [
        (record, field.name)
        for record in (
            scenario_bundle.interactions, scenario_bundle.splits,
            *scenario_bundle.feature_views.values(), *scenario_parties,
        )
        for field in fields(record)
        if isinstance(getattr(record, field.name), np.ndarray)
    ]
    arrays = [getattr(record, field_name) for record, field_name in array_fields]
    copies = [array.copy() for array in arrays]
    partition, parties, stats = scenario_builder(
        scenario_bundle, scenario_parties,
        view_names=("item_text", "item_tabular"), name="no_source_mutation",
    )
    originals = {party.party_id: party for party in scenario_parties}
    assert any(len(party.user_ids) > 1 for party in parties)
    for party in parties:
        original = originals[party.party_id]
        assert party is not original
        for field in fields(party):
            if field.name != "feature_view_names":
                assert getattr(party, field.name) is getattr(original, field.name)
        assert "user_info" not in party.feature_view_names
    assert scenario_bundle.parties is original_bundle_parties
    assert [
        party.feature_view_names for party in scenario_parties
    ] == original_summaries
    for (record, field_name), array, before in zip(array_fields, arrays, copies):
        assert getattr(record, field_name) is array
        np.testing.assert_array_equal(array, before)
    assert scenario_bundle.feature_views["item_text"].values == ("", "", "")
    # Coarse records are mutable and shallow; immutable assignment identity is separate.
    parties[0].feature_view_names = ("caller_changed_summary",)
    assert scenario_parties[1].feature_view_names == original_summaries[1]
    assert feature_partition_checksum(partition) == stats["assignment_checksum"]
    with pytest.raises(TypeError):
        partition.party_access[7] = ()


@pytest.mark.parametrize(
    "view_names, error",
    [
        ((), ValueError), (None, TypeError), ({"item_text"}, TypeError),
        ("item_text", TypeError), (b"item_text", TypeError),
        ((None,), TypeError), (("item_text", 1), TypeError),
        (("item_text", "item_text"), ValueError),
        (("unknown",), ValueError), (("",), ValueError),
    ],
)
def test_rejects_invalid_selected_views(
    scenario_bundle, scenario_parties, scenario_builder, view_names, error,
):
    with pytest.raises(error, match="view_names|FeatureView"):
        scenario_builder(
            scenario_bundle, scenario_parties, view_names=view_names, name="invalid",
        )


@pytest.mark.parametrize(
    "name, error",
    [(None, TypeError), (1, TypeError), ("", ValueError), (" \t", ValueError)],
)
def test_rejects_invalid_names(
    scenario_bundle, scenario_parties, scenario_builder, name, error,
):
    with pytest.raises(error, match="name"):
        scenario_builder(
            scenario_bundle, scenario_parties, view_names=("item_text",), name=name,
        )


def test_rejects_invalid_party_sequences(
    scenario_bundle, scenario_parties, scenario_builder,
):
    for parties, error in (
        ((), ValueError), (None, TypeError), ("party", TypeError),
        ((object(),), TypeError),
        ((scenario_parties[0], scenario_parties[0]), ValueError),
    ):
        with pytest.raises(error, match="parties"):
            scenario_builder(
                scenario_bundle, parties, view_names=("item_text",), name="invalid",
            )


@pytest.mark.parametrize("party_id", [True, -1, 1.5, "7", np.int64(7), None])
def test_rejects_non_python_or_negative_party_ids(
    scenario_bundle, scenario_parties, scenario_builder, party_id,
):
    party = replace(scenario_parties[0])
    party.party_id = party_id  # PartyData is mutable; the builder must check the ID.
    with pytest.raises(ValueError, match="party IDs"):
        scenario_builder(
            scenario_bundle, (party,), view_names=("item_text",), name="invalid",
        )


@pytest.mark.parametrize(
    "missing_view, error",
    [
        ("user_info", ValueError), ("unknown", ValueError), ("", ValueError),
        (None, TypeError), ([], TypeError),
    ],
)
def test_rejects_missing_view_outside_selected_views(
    scenario_bundle, scenario_parties, missing_view, error,
):
    with pytest.raises(error, match="missing_view"):
        build_single_view_absence_scenario(
            scenario_bundle, scenario_parties, view_names=("item_text",),
            missing_view=missing_view, missing_party_count=1, seed=0, name="invalid",
        )


@pytest.mark.parametrize(
    "count, error",
    [
        (True, TypeError), (1.0, TypeError), ("1", TypeError),
        (np.int64(1), TypeError), (-1, ValueError), (4, ValueError),
    ],
)
def test_rejects_invalid_absence_counts(
    scenario_bundle, scenario_parties, count, error,
):
    with pytest.raises(error, match="missing_party_count"):
        build_single_view_absence_scenario(
            scenario_bundle, scenario_parties, view_names=("item_text",),
            missing_view="item_text", missing_party_count=count, seed=0, name="invalid",
        )


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize(
    "seed, error",
    [
        (True, TypeError), (1.0, TypeError), ("1", TypeError),
        (np.int64(1), TypeError), (-1, ValueError), (2**32, ValueError),
    ],
)
def test_seed_is_validated_even_at_absence_boundaries(
    scenario_bundle, scenario_parties, count, seed, error,
):
    with pytest.raises(error, match="seed"):
        build_single_view_absence_scenario(
            scenario_bundle, scenario_parties, view_names=("item_text",),
            missing_view="item_text", missing_party_count=count,
            seed=seed, name="invalid",
        )
