"""Feature-access validation, assignment identity, and local payload resolution."""

import hashlib
import json
from collections.abc import Sequence

import numpy as np

from src.data.contracts import (
    DatasetBundle,
    FeatureAccess,
    FeaturePartition,
    FeatureView,
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


def _validate_referenced_views(
    bundle: DatasetBundle, partition: FeaturePartition
) -> None:
    if not isinstance(bundle.feature_views, dict):
        raise TypeError("feature_views must be a dictionary")
    entity_limits = {
        "user": len(bundle.user_id_map),
        "item": len(bundle.item_id_map),
        "interaction": len(bundle.interactions),
    }
    referenced_names = {
        access.view_name
        for accesses in partition.party_access.values()
        for access in accesses
    }
    for name in sorted(referenced_names):
        if name not in bundle.feature_views:
            raise ValueError(
                f"FeaturePartition references unknown FeatureView {name!r}"
            )
        view = bundle.feature_views[name]
        if not isinstance(view, FeatureView):
            raise TypeError("referenced feature_views must be FeatureView objects")
        if name != view.name:
            raise ValueError("feature_views keys must match their FeatureView names")
        # Reuse constructor validation in a temporary base-class wrapper. No
        # canonical object is mutated, and these shared references never escape.
        FeatureView(
            name=view.name,
            entity_kind=view.entity_kind,
            entity_ids=view.entity_ids,
            modality=view.modality,
            values=view.values,
            feature_names=view.feature_names,
            missing_mask=view.missing_mask,
            metadata=view.metadata,
        )
        if np.any(view.entity_ids >= entity_limits[view.entity_kind]):
            raise ValueError(
                f"FeatureView {name!r} references an out-of-range {view.entity_kind} ID"
            )


def _validate_delivery_dtype(array: np.ndarray, name: str) -> None:
    # Check both the reported dtype and the backing dtype after stripping a
    # subclass: a subclass property must not conceal object/metadata references.
    for dtype in (array.dtype, np.asarray(array).dtype):
        if (
            dtype.hasobject
            or dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.metadata is not None
        ):
            raise TypeError(f"{name} has unsupported delivery dtype {dtype}")


def _copy_selected_array(
    array: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray | None = None,
) -> np.ndarray:
    # Strip subclass state before indexing, and own only the selected storage.
    plain = np.asarray(array)
    selected = plain[rows] if columns is None else plain[np.ix_(rows, columns)]
    return np.array(selected, copy=True, order="C", subok=False)


def _resolve_feature_view(view: FeatureView, access: FeatureAccess) -> FeatureView:
    # Check delivered source dtypes before slicing, even for empty selections.
    for name in ("entity_ids", "values", "missing_mask"):
        array = getattr(view, name)
        if isinstance(array, np.ndarray):
            _validate_delivery_dtype(array, f"FeatureView {view.name!r} {name}")

    selected_ids = access.entity_ids
    rows = (
        np.arange(len(view.entity_ids), dtype=np.intp)
        if selected_ids is None
        else np.searchsorted(np.asarray(view.entity_ids), selected_ids)
    )
    columns = None
    feature_names: tuple[str, ...] = ()
    if view.feature_names:
        columns = np.array(
            [
                index for index, name in enumerate(view.feature_names)
                if access.feature_names is None or name in access.feature_names
            ],
            dtype=np.intp,
        )
        feature_names = tuple(view.feature_names[index] for index in columns)

    values = (
        _copy_selected_array(view.values, rows, columns)
        if isinstance(view.values, np.ndarray)
        else tuple(view.values[index] for index in rows)
    )
    output_strings = (view.name, view.entity_kind, view.modality, *feature_names)
    if isinstance(values, tuple):
        output_strings += values
    if any(type(value) is not str for value in output_strings):
        raise TypeError("resolved schema and text must use plain built-in strings")

    return FeatureView(
        name=view.name,
        entity_kind=view.entity_kind,
        entity_ids=_copy_selected_array(view.entity_ids, rows),
        modality=view.modality,
        values=values,
        feature_names=feature_names,
        missing_mask=(
            None if view.missing_mask is None
            else _copy_selected_array(view.missing_mask, rows, columns)
        ),
        metadata={},
    )


def resolve_party_feature_views(
    bundle: DatasetBundle,
    partition: FeaturePartition,
    parties: Sequence[PartyData],
    *,
    party_id: int,
) -> dict[str, FeatureView]:
    """Return detached local projections, never attaching them to the bundle.

    Validate the whole partition's current references/structure; delivery dtype
    restrictions apply only to the requested party. Rows and columns follow
    canonical order. Arrays own selected storage; metadata is always fresh/empty.
    Schema/text string subclasses are rejected if they would be returned.

    Callers must keep canonical data unchanged during resolution and across
    calls intended to share one source state. There is no cache, locking, atomic
    multi-party snapshot, or cryptographic privacy guarantee. Provenance and
    assignment identity remain with the trusted caller, not the local payload.
    """
    if not isinstance(bundle, DatasetBundle):
        raise TypeError("bundle must be a DatasetBundle")
    if not isinstance(partition, FeaturePartition):
        raise TypeError("partition must be a FeaturePartition")
    if not isinstance(parties, Sequence) or isinstance(parties, (str, bytes)):
        raise TypeError("parties must be a sequence of PartyData objects")
    if isinstance(party_id, bool) or not isinstance(party_id, int) or party_id < 0:
        raise ValueError("party_id must be a non-negative Python integer")
    party_ids: set[int] = set()
    for party in parties:
        if not isinstance(party, PartyData):
            raise TypeError("parties must contain PartyData objects")
        if (
            isinstance(party.party_id, bool)
            or not isinstance(party.party_id, int)
            or party.party_id < 0
        ):
            raise ValueError("roster party IDs must be non-negative Python integers")
        if party.party_id in party_ids:
            raise ValueError("supplied parties must have unique party IDs")
        party_ids.add(party.party_id)
    if party_id not in party_ids:
        raise ValueError(f"unknown requested party_id {party_id}")

    _validate_referenced_views(bundle, partition)
    validate_feature_partition(partition, bundle, parties)
    resolved: dict[str, FeatureView] = {}
    for access in sorted(
        partition.access_for_party(party_id), key=lambda entry: entry.view_name
    ):
        view = _resolve_feature_view(bundle.feature_views[access.view_name], access)
        resolved[view.name] = view
    return resolved
