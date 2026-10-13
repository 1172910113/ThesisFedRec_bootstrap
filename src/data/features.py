"""Validation and deterministic identity for feature-access partitions."""

import hashlib
import json
from collections.abc import Sequence

import numpy as np

from src.data.contracts import (
    DatasetBundle,
    FeaturePartition,
    PartyData,
)


def validate_feature_partition(
    partition: FeaturePartition,
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
) -> None:
    """Validate party feature access against canonical views and party IDs."""
    parties_by_id: dict[int, PartyData] = {}
    for party in parties:
        if not isinstance(party, PartyData):
            raise TypeError("parties must contain PartyData objects")
        if party.party_id in parties_by_id:
            raise ValueError("supplied parties must have unique party IDs")
        parties_by_id[party.party_id] = party

    for party_id, accesses in partition.party_access.items():
        if party_id not in parties_by_id:
            raise ValueError(
                f"FeaturePartition references unknown party_id {party_id}"
            )
        for access in accesses:
            if access.view_name not in bundle.feature_views:
                raise ValueError(
                    "FeaturePartition references unknown FeatureView "
                    f"{access.view_name!r}"
                )
            view = bundle.feature_views[access.view_name]

            if access.entity_ids is not None:
                unknown_entity_ids = access.entity_ids[
                    ~np.isin(access.entity_ids, view.entity_ids)
                ]
                if unknown_entity_ids.size:
                    raise ValueError(
                        f"FeatureAccess for {access.view_name!r} references "
                        f"unknown entity ID {int(unknown_entity_ids[0])}"
                    )

            if access.feature_names is not None:
                if not view.feature_names:
                    raise ValueError(
                        f"FeatureView {access.view_name!r} does not support "
                        "field-level access"
                    )
                unknown_feature_names = sorted(
                    set(access.feature_names) - set(view.feature_names)
                )
                if unknown_feature_names:
                    raise ValueError(
                        f"FeatureAccess for {access.view_name!r} references "
                        f"unknown feature field {unknown_feature_names[0]!r}"
                    )

    for party_id, party in parties_by_id.items():
        detailed_names = tuple(
            sorted(
                access.view_name
                for access in partition.access_for_party(party_id)
            )
        )
        summary_names = tuple(sorted(party.feature_view_names))
        if detailed_names != summary_names:
            raise ValueError(
                "PartyData feature_view_names do not match FeaturePartition "
                f"access for party_id {party_id}"
            )


def feature_partition_checksum(partition: FeaturePartition) -> str:
    """Hash access definitions in a canonical, insertion-order-free form."""
    canonical_access = [
        {
            "party_id": party_id,
            "views": [
                {
                    "view_name": access.view_name,
                    "entity_ids": (
                        None
                        if access.entity_ids is None
                        else [int(value) for value in access.entity_ids]
                    ),
                    "feature_names": (
                        None
                        if access.feature_names is None
                        else sorted(access.feature_names)
                    ),
                }
                for access in sorted(
                    partition.party_access[party_id],
                    key=lambda value: value.view_name,
                )
            ],
        }
        for party_id in sorted(partition.party_access)
    ]
    serialized = json.dumps(
        canonical_access,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
