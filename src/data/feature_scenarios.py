"""In-memory whole-view availability recipes, independent of sample ownership."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from src.data.contracts import (
    DatasetBundle,
    FeatureAccess,
    FeaturePartition,
    PartyData,
)
from src.data.features import feature_partition_checksum, validate_feature_partition


def _ordered_inputs(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    view_names: Sequence[str],
    name: str,
) -> tuple[tuple[PartyData, ...], tuple[str, ...]]:
    if not isinstance(name, str):
        raise TypeError("scenario name must be a string")
    if not name.strip():
        raise ValueError("scenario name must not be empty")
    if not isinstance(parties, Sequence) or isinstance(parties, (str, bytes)):
        raise TypeError("parties must be a sequence of PartyData objects")
    if not parties:
        raise ValueError("parties must not be empty")

    party_ids: set[int] = set()
    for party in parties:
        if not isinstance(party, PartyData):
            raise TypeError("parties must contain PartyData objects")
        if (
            isinstance(party.party_id, bool)
            or not isinstance(party.party_id, int)
            or party.party_id < 0
        ):
            raise ValueError("party IDs must be non-negative Python integers")
        if party.party_id in party_ids:
            raise ValueError("parties must have unique party IDs")
        party_ids.add(party.party_id)

    if not isinstance(view_names, Sequence) or isinstance(view_names, (str, bytes)):
        raise TypeError("view_names must be a sequence of strings, not a string/bytes")
    if not view_names:
        raise ValueError("view_names must not be empty")
    if any(not isinstance(view_name, str) for view_name in view_names):
        raise TypeError("view_names must contain strings")
    if len(set(view_names)) != len(view_names):
        raise ValueError("view_names must be unique")
    for view_name in view_names:
        if view_name not in bundle.feature_views:
            raise ValueError(f"unknown canonical FeatureView {view_name!r}")

    return (
        tuple(sorted(parties, key=lambda party: party.party_id)),
        tuple(sorted(view_names)),
    )


def _scenario_result(
    bundle: DatasetBundle,
    parties: tuple[PartyData, ...],
    view_names: tuple[str, ...],
    party_access: dict[int, tuple[FeatureAccess, ...]],
    *,
    name: str,
    metadata: dict[str, object],
) -> tuple[FeaturePartition, tuple[PartyData, ...], dict[str, object]]:
    partition = FeaturePartition(
        name=name,
        party_access=party_access,
        metadata={
            **metadata,
            "recipe_version": 1,
            "party_ids": [party.party_id for party in parties],
            "view_names": list(view_names),
        },
    )
    scenario_parties = tuple(
        replace(
            party,
            feature_view_names=tuple(
                access.view_name for access in party_access[party.party_id]
            ),
        )
        for party in parties
    )
    validate_feature_partition(partition, bundle, scenario_parties)

    counts_by_party = {
        party_id: len(accesses) for party_id, accesses in party_access.items()
    }
    counts_by_view = dict.fromkeys(view_names, 0)
    for accesses in party_access.values():
        for access in accesses:
            counts_by_view[access.view_name] += 1
    statistics: dict[str, object] = {
        "party_count": len(parties),
        "selected_views": view_names,
        "visible_view_counts_by_party": counts_by_party,
        "visible_party_counts_by_view": counts_by_view,
        "featureless_party_count": sum(
            count == 0 for count in counts_by_party.values()
        ),
        "available_party_view_pairs": sum(counts_by_party.values()),
        "possible_party_view_pairs": len(parties) * len(view_names),
        "assignment_checksum": feature_partition_checksum(partition),
    }
    return partition, scenario_parties, statistics


def build_complete_feature_scenario(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    *,
    view_names: Sequence[str],
    name: str,
) -> tuple[FeaturePartition, tuple[PartyData, ...], dict[str, object]]:
    """Grant every party whole access to explicitly selected canonical views.

    No randomness is used. Results are ordered by party ID and view name.
    Returned PartyData records replace only feature_view_names; unchanged
    arrays and metadata are shared, not deeply copied or made immutable.
    The bundle and supplied parties are never changed.
    """
    ordered_parties, selected_views = _ordered_inputs(bundle, parties, view_names, name)
    accesses = tuple(FeatureAccess(view_name=view_name) for view_name in selected_views)
    party_access = {party.party_id: accesses for party in ordered_parties}
    return _scenario_result(
        bundle, ordered_parties, selected_views, party_access,
        name=name, metadata={"recipe": "complete"},
    )


def build_single_view_absence_scenario(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    *,
    view_names: Sequence[str],
    missing_view: str,
    missing_party_count: int,
    seed: int,
    name: str,
) -> tuple[FeaturePartition, tuple[PartyData, ...], dict[str, object]]:
    """Hide one selected view from exactly K parties using a local NumPy RNG.

    Sample positions without replacement in sorted party-ID order. Input
    order and global RNG state have no effect. Zero/all-party absence is valid.
    New PartyData records replace only feature_view_names and share unchanged
    arrays/metadata; they are not deeply immutable or fully independent copies.
    Availability statistics count assignments, not observed feature coverage.
    """
    ordered_parties, selected_views = _ordered_inputs(bundle, parties, view_names, name)
    if not isinstance(missing_view, str):
        raise TypeError("missing_view must be a string")
    if missing_view not in selected_views:
        raise ValueError("missing_view must be one of the selected view_names")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a Python integer, excluding bool")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must satisfy 0 <= seed < 2**32")
    if isinstance(missing_party_count, bool) or not isinstance(
        missing_party_count, int
    ):
        raise TypeError("missing_party_count must be a Python integer, excluding bool")
    if not 0 <= missing_party_count <= len(ordered_parties):
        raise ValueError("missing_party_count must be between zero and the party count")

    positions = np.random.default_rng(seed).choice(
        len(ordered_parties), size=missing_party_count, replace=False
    )
    hidden_party_ids = {ordered_parties[position].party_id for position in positions}
    party_access = {
        party.party_id: tuple(
            FeatureAccess(view_name=view_name)
            for view_name in selected_views
            if view_name != missing_view or party.party_id not in hidden_party_ids
        )
        for party in ordered_parties
    }
    return _scenario_result(
        bundle, ordered_parties, selected_views, party_access,
        name=name,
        metadata={
            "recipe": "single_view_absence",
            "missing_view": missing_view,
            "missing_party_count": missing_party_count,
            "seed": seed,
        },
    )
