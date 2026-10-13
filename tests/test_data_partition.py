from dataclasses import replace

import numpy as np

from src.data.contracts import DatasetBundle, FeatureView, PartyData
from src.data.partition import build_hfl_partition


def _assert_party_interactions_match_users(
    bundle: DatasetBundle, party: PartyData
) -> None:
    interaction_ids = np.concatenate(
        (
            party.train_interaction_ids,
            party.valid_interaction_ids,
            party.test_interaction_ids,
        )
    )
    interaction_users = np.unique(bundle.interactions.user_ids[interaction_ids])
    np.testing.assert_array_equal(interaction_users, party.user_ids)
    interaction_items = np.unique(bundle.interactions.item_ids[interaction_ids])
    np.testing.assert_array_equal(interaction_items, party.item_ids)


def _assert_partition_covers_splits(
    bundle: DatasetBundle, parties: tuple[PartyData, ...]
) -> None:
    for split_name in ("train", "valid", "test"):
        assigned = np.concatenate(
            [
                getattr(party, f"{split_name}_interaction_ids")
                for party in parties
            ]
        )
        expected = getattr(bundle.splits, f"{split_name}_interaction_ids")
        np.testing.assert_array_equal(np.sort(assigned), expected)


def test_per_user_hfl_mapping_is_stable_and_seed_independent(
    processed_bundle: DatasetBundle,
) -> None:
    first = build_hfl_partition(
        processed_bundle,
        mode="per_user",
        seed=7,
    )
    second = build_hfl_partition(
        processed_bundle,
        mode="per_user",
        seed=999,
    )

    assert len(first) == len(processed_bundle.user_id_map)
    assert all(len(party.user_ids) == 1 for party in first)
    first_mapping = {
        int(party.user_ids[0]): party.party_id for party in first
    }
    second_mapping = {
        int(party.user_ids[0]): party.party_id for party in second
    }
    assert first_mapping == second_mapping

    for first_party, second_party in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_party.user_ids, second_party.user_ids)
        np.testing.assert_array_equal(
            first_party.train_interaction_ids,
            second_party.train_interaction_ids,
        )
        np.testing.assert_array_equal(
            first_party.valid_interaction_ids,
            second_party.valid_interaction_ids,
        )
        np.testing.assert_array_equal(
            first_party.test_interaction_ids,
            second_party.test_interaction_ids,
        )

    for party in first:
        _assert_party_interactions_match_users(processed_bundle, party)
    _assert_partition_covers_splits(processed_bundle, first)


def test_grouped_round_robin_is_deterministic_and_keeps_users_intact(
    processed_bundle: DatasetBundle,
) -> None:
    first = build_hfl_partition(
        processed_bundle,
        mode="grouped_round_robin",
        num_parties=2,
        seed=19,
    )
    second = build_hfl_partition(
        processed_bundle,
        mode="grouped_round_robin",
        num_parties=2,
        seed=19,
    )

    assert len(first) == 2
    for first_party, second_party in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_party.user_ids, second_party.user_ids)
        np.testing.assert_array_equal(
            first_party.train_interaction_ids,
            second_party.train_interaction_ids,
        )
        np.testing.assert_array_equal(
            first_party.valid_interaction_ids,
            second_party.valid_interaction_ids,
        )
        np.testing.assert_array_equal(
            first_party.test_interaction_ids,
            second_party.test_interaction_ids,
        )
        _assert_party_interactions_match_users(processed_bundle, first_party)

    assigned_users = np.concatenate([party.user_ids for party in first])
    np.testing.assert_array_equal(
        np.sort(assigned_users),
        np.arange(len(processed_bundle.user_id_map), dtype=np.int64),
    )
    _assert_partition_covers_splits(processed_bundle, first)


def test_party_feature_view_summary_does_not_change_sample_ownership(
    processed_bundle: DatasetBundle,
) -> None:
    view = FeatureView(
        name="item_numeric",
        entity_kind="item",
        entity_ids=np.array([0], dtype=np.int64),
        modality="tabular",
        values=np.array([[1.0]], dtype=np.float32),
        feature_names=("value",),
        missing_mask=None,
        metadata={},
    )
    featured_bundle = replace(
        processed_bundle,
        feature_views={view.name: view},
    )

    without_features = build_hfl_partition(processed_bundle, seed=7)
    with_features = build_hfl_partition(featured_bundle, seed=7)

    for plain_party, featured_party in zip(
        without_features, with_features, strict=True
    ):
        assert plain_party.feature_view_names == ()
        assert featured_party.feature_view_names == (view.name,)
        np.testing.assert_array_equal(
            plain_party.train_interaction_ids,
            featured_party.train_interaction_ids,
        )
        np.testing.assert_array_equal(
            plain_party.valid_interaction_ids,
            featured_party.valid_interaction_ids,
        )
        np.testing.assert_array_equal(
            plain_party.test_interaction_ids,
            featured_party.test_interaction_ids,
        )
        np.testing.assert_array_equal(
            plain_party.user_ids,
            featured_party.user_ids,
        )
        np.testing.assert_array_equal(
            plain_party.item_ids,
            featured_party.item_ids,
        )
