"""Horizontal federation partitioning."""

from collections.abc import Mapping, Sequence

import numpy as np

from src.data.contracts import DatasetBundle, PartyData


def _group_split_ids(
    interaction_ids: np.ndarray,
    interaction_parties: np.ndarray,
    num_parties: int,
) -> list[np.ndarray]:
    if interaction_ids.size == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(num_parties)]

    assigned_parties = interaction_parties[interaction_ids]
    order = np.argsort(assigned_parties, kind="stable")
    ordered_ids = interaction_ids[order]
    counts = np.bincount(assigned_parties, minlength=num_parties)
    boundaries = np.cumsum(counts)[:-1]
    return [
        values.astype(np.int64, copy=False)
        for values in np.split(ordered_ids, boundaries)
    ]


def build_hfl_partition(
    bundle: DatasetBundle,
    *,
    mode: str = "per_user",
    num_parties: int | None = None,
    seed: int = 2026,
    party_feature_views: Mapping[int, Sequence[str]] | None = None,
    party_graph_views: Mapping[int, Sequence[str]] | None = None,
) -> tuple[PartyData, ...]:
    """Partition users and their interactions into HFL parties."""
    user_ids = np.unique(bundle.interactions.user_ids)

    if mode == "per_user":
        if num_parties not in (None, len(user_ids)):
            raise ValueError(
                "per_user mode uses exactly one independently identified party per user"
            )
        resolved_num_parties = len(user_ids)
    elif mode == "grouped_round_robin":
        if num_parties is None:
            raise ValueError("grouped_round_robin requires num_parties")
        if num_parties <= 0 or num_parties > len(user_ids):
            raise ValueError(
                "num_parties must be positive and no greater than the user count"
            )
        resolved_num_parties = num_parties
    else:
        raise ValueError(f"unsupported HFL partition mode: {mode}")

    user_to_party = np.empty(len(bundle.user_id_map), dtype=np.int64)
    if mode == "per_user":
        user_to_party[user_ids] = np.arange(
            resolved_num_parties, dtype=np.int64
        )
    else:
        rng = np.random.default_rng(seed)
        permuted_users = rng.permutation(user_ids)
        user_to_party[permuted_users] = (
            np.arange(len(permuted_users), dtype=np.int64)
            % resolved_num_parties
        )

    interaction_parties = user_to_party[bundle.interactions.user_ids]
    train_by_party = _group_split_ids(
        bundle.splits.train_interaction_ids,
        interaction_parties,
        resolved_num_parties,
    )
    valid_by_party = _group_split_ids(
        bundle.splits.valid_interaction_ids,
        interaction_parties,
        resolved_num_parties,
    )
    test_by_party = _group_split_ids(
        bundle.splits.test_interaction_ids,
        interaction_parties,
        resolved_num_parties,
    )

    default_feature_views = tuple(sorted(bundle.feature_views))
    default_graph_views = tuple(sorted(bundle.graph_views))
    parties: list[PartyData] = []
    for party_id in range(resolved_num_parties):
        party_users = np.flatnonzero(user_to_party == party_id).astype(np.int64)
        all_interaction_ids = np.concatenate(
            (
                train_by_party[party_id],
                valid_by_party[party_id],
                test_by_party[party_id],
            )
        )
        party_items = np.unique(
            bundle.interactions.item_ids[all_interaction_ids]
        ).astype(np.int64, copy=False)
        feature_names = tuple(
            party_feature_views.get(party_id, default_feature_views)
            if party_feature_views is not None
            else default_feature_views
        )
        graph_names = tuple(
            party_graph_views.get(party_id, default_graph_views)
            if party_graph_views is not None
            else default_graph_views
        )
        parties.append(
            PartyData(
                party_id=party_id,
                train_interaction_ids=train_by_party[party_id],
                valid_interaction_ids=valid_by_party[party_id],
                test_interaction_ids=test_by_party[party_id],
                user_ids=party_users,
                item_ids=party_items,
                feature_view_names=feature_names,
                graph_view_names=graph_names,
                owns_labels=True,
                metadata={"hfl_mode": mode},
            )
        )

    _validate_hfl_partition(bundle, tuple(parties))
    return tuple(parties)


def _validate_hfl_partition(
    bundle: DatasetBundle, parties: tuple[PartyData, ...]
) -> None:
    for split_name in ("train", "valid", "test"):
        party_ids = np.concatenate(
            [
                getattr(party, f"{split_name}_interaction_ids")
                for party in parties
            ]
        )
        expected_ids = getattr(bundle.splits, f"{split_name}_interaction_ids")
        if not np.array_equal(np.sort(party_ids), expected_ids):
            raise ValueError(f"HFL parties do not exactly cover the {split_name} split")

    assigned_users = np.concatenate([party.user_ids for party in parties])
    expected_users = np.arange(len(bundle.user_id_map), dtype=np.int64)
    if not np.array_equal(np.sort(assigned_users), expected_users):
        raise ValueError("every user must belong to exactly one HFL party")
