"""Trusted, in-memory feature snapshots and indexed detached delivery."""

import hashlib
import json
from collections.abc import Sequence
from types import MappingProxyType

import numpy as np

from src.data.contracts import (
    DatasetBundle, FeatureAccess, FeaturePartition, FeatureView, PartyData,
)
from src.data.features import feature_partition_checksum, validate_feature_partition
from src.data.graph import graph_domain_fingerprint


class FeatureRuntime:
    """Opaque coordinator handle; private storage is not a public interface."""

    __slots__ = ("_binding_id", "_views", "_indexes", "_access_by_party")

    def __init__(self) -> None:
        raise TypeError("use prepare_feature_runtime to create a FeatureRuntime")

    @property
    def binding_id(self) -> str:
        """Trusted-local snapshot identity; retired handles cannot be read."""
        _require_open(self)
        return self._binding_id


def _require_open(runtime: FeatureRuntime) -> None:
    if not isinstance(runtime, FeatureRuntime):
        raise TypeError("runtime must be a FeatureRuntime")
    if runtime._binding_id is None:
        raise ValueError("FeatureRuntime is closed")


def check_feature_runtime_binding(runtime: FeatureRuntime, *, binding_id: str) -> None:
    """Check an explicit expected identity without inspecting caller sources."""
    _require_open(runtime)
    if type(binding_id) is not str:
        raise TypeError("binding_id must be a plain built-in string")
    if binding_id != runtime._binding_id:
        raise ValueError("FeatureRuntime binding mismatch")


def _plain_name(name: str, value: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a plain built-in string")
    if not value.strip():
        raise ValueError(f"{name} must be nonblank")


def _names(name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple of plain strings")
    for value in values:
        _plain_name(name, value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def _party_id(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("party_id must be an exact non-negative built-in integer")


def _validate_roster(parties: Sequence[PartyData]) -> dict[int, tuple[str, ...]]:
    if not isinstance(parties, Sequence) or isinstance(parties, (str, bytes)):
        raise TypeError("parties must be a sequence of PartyData objects")
    roster = {}
    for party in parties:
        if not isinstance(party, PartyData):
            raise TypeError("parties must contain PartyData objects")
        _party_id(party.party_id)
        if party.party_id in roster:
            raise ValueError("roster must have unique party IDs")
        _names("PartyData feature_view_names", party.feature_view_names)
        roster[party.party_id] = tuple(sorted(party.feature_view_names))
    return roster


def _copy_partition(partition: FeaturePartition) -> FeaturePartition:
    accesses = {}
    for party_id, entries in partition.party_access.items():
        _party_id(party_id)
        if not isinstance(entries, tuple):
            raise TypeError("access entries must be tuples")
        copied = []
        for entry in entries:
            if not isinstance(entry, FeatureAccess):
                raise TypeError("access entries must contain FeatureAccess objects")
            _plain_name("FeatureAccess view_name", entry.view_name)
            if entry.feature_names is not None:
                _names("FeatureAccess feature_names", entry.feature_names)
            copied.append(FeatureAccess(
                view_name=entry.view_name, entity_ids=entry.entity_ids,
                feature_names=(
                    None if entry.feature_names is None
                    else tuple(name for name in entry.feature_names)
                ),
            ))
        accesses[party_id] = tuple(sorted(copied, key=lambda entry: entry.view_name))
    return FeaturePartition(name="runtime", party_access=accesses, metadata={})


def _plain_array(name: str, array: np.ndarray) -> None:
    if type(array) is not np.ndarray:
        raise TypeError(f"{name} must be a plain NumPy ndarray")
    dtype = array.dtype
    if (dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None
            or dtype.metadata is not None):
        raise TypeError(f"{name} has unsupported delivery dtype")


def _validate_views(views: dict[str, FeatureView], limits: dict[str, int]) -> None:
    for name, view in views.items():
        if not isinstance(view, FeatureView):
            raise TypeError("referenced views must be FeatureView objects")
        for field in ("name", "entity_kind", "modality"):
            _plain_name(f"FeatureView {field}", getattr(view, field))
        _names("FeatureView feature_names", view.feature_names)
        _plain_array("FeatureView entity_ids", view.entity_ids)
        if isinstance(view.values, np.ndarray):
            _plain_array("FeatureView values", view.values)
        elif isinstance(view.values, tuple):
            # Empty text strings are legal content, unlike schema names.
            if any(type(text) is not str for text in view.values):
                raise TypeError("FeatureView text must use plain built-in strings")
        else:
            raise TypeError("FeatureView values must be ndarray or tuple of strings")
        if view.missing_mask is not None:
            _plain_array("FeatureView missing_mask", view.missing_mask)
        # Validate original types/structure BEFORE copying. No metadata traversal.
        FeatureView(
            name=view.name, entity_kind=view.entity_kind, entity_ids=view.entity_ids,
            modality=view.modality, values=view.values, feature_names=view.feature_names,
            missing_mask=view.missing_mask, metadata=view.metadata,
        )
        if name != view.name:
            raise ValueError("feature_views keys must match FeatureView names")
        if np.any(view.entity_ids >= limits[view.entity_kind]):
            raise ValueError(f"FeatureView {name!r} references an out-of-range ID")


def _json_digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _array_identity(array: np.ndarray) -> dict:
    return {
        "shape": list(array.shape), "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _content_identity(view: FeatureView) -> str:
    return _json_digest({
        "format": "thesisfedrec.feature_content", "version": 1,
        "name": view.name, "entity_kind": view.entity_kind, "modality": view.modality,
        "feature_names": list(view.feature_names),
        "entity_ids": _array_identity(view.entity_ids),
        "values": (
            {"kind": "array", "array": _array_identity(view.values)}
            if isinstance(view.values, np.ndarray)
            else {"kind": "text", "items": list(view.values)}
        ),
        "missing_mask": (
            None if view.missing_mask is None else _array_identity(view.missing_mask)
        ),
    })


def _immutable_array(array: np.ndarray) -> np.ndarray:
    # Unlike frombuffer, the explicit-shape constructor also accepts plain V0
    # arrays. Bytes backing still prevents re-enabling element writes.
    return np.ndarray(shape=array.shape, dtype=array.dtype,
                      buffer=array.tobytes(order="C"), order="C")


def _copy_view(view: FeatureView) -> FeatureView:
    return FeatureView(
        name=view.name, entity_kind=view.entity_kind, modality=view.modality,
        entity_ids=_immutable_array(view.entity_ids),
        values=(
            _immutable_array(view.values) if isinstance(view.values, np.ndarray)
            else tuple(text for text in view.values)
        ),
        feature_names=tuple(name for name in view.feature_names),
        missing_mask=(
            None if view.missing_mask is None else _immutable_array(view.missing_mask)
        ),
        metadata={},
    )


def _build_index(view: FeatureView) -> tuple:
    # One lookup pair per source view, shared by every party. No full-row vectors.
    return (
        MappingProxyType({int(entity): row for row, entity in enumerate(view.entity_ids)}),
        MappingProxyType({name: column for column, name in enumerate(view.feature_names)}),
    )


def _selection(view: FeatureView, access: FeatureAccess, index: tuple) -> tuple:
    row_index, field_index = index
    ids = access.entity_ids
    try:
        rows = (
            slice(None) if ids is None
            else np.array([row_index[int(entity)] for entity in ids], dtype=np.intp)
        )
    except KeyError as error:
        raise ValueError(f"unknown feature entity ID {error.args[0]}") from None
    if access.feature_names is None:
        return rows, None, view.feature_names
    if not view.feature_names:
        raise ValueError(f"FeatureView {view.name!r} does not support field-level access")
    try:
        columns = np.array(sorted(field_index[name] for name in access.feature_names),
                           dtype=np.intp)
    except KeyError as error:
        raise ValueError(f"unknown feature field {error.args[0]!r}") from None
    return rows, columns, tuple(view.feature_names[column] for column in columns)


def _validate_owned_references(partition, views, indexes, roster) -> None:
    for party_id, entries in partition.party_access.items():
        if party_id not in roster:
            raise ValueError(f"unknown party_id {party_id}")
        for entry in entries:
            if entry.view_name not in views:
                raise ValueError(f"unknown FeatureView {entry.view_name!r}")
            _selection(views[entry.view_name], entry, indexes[entry.view_name])
    for party_id, names in roster.items():
        if tuple(entry.view_name for entry in partition.access_for_party(party_id)) != names:
            raise ValueError(f"feature_view_names do not match access for party_id {party_id}")


def prepare_feature_runtime(
    bundle: DatasetBundle, partition: FeaturePartition, parties: Sequence[PartyData],
    *, source_id: str, split_id: str, recipe_id: str,
) -> FeatureRuntime:
    """Validate, snapshot and index referenced views while caller inputs are stable.

    Context strings assert provenance, not verified files/splits. Domain identity
    covers processed namespaces; interaction cardinality is not row semantics.
    Input changes require a new handle. No metadata or sample payload is retained.
    """
    if not isinstance(bundle, DatasetBundle):
        raise TypeError("bundle must be a DatasetBundle")
    if not isinstance(partition, FeaturePartition):
        raise TypeError("partition must be a FeaturePartition")
    for name, value in (("source_id", source_id), ("split_id", split_id),
                        ("recipe_id", recipe_id)):
        _plain_name(name, value)
    roster = _validate_roster(parties)
    owned_partition = _copy_partition(partition)
    if not isinstance(bundle.feature_views, dict):
        raise TypeError("feature_views must be a dictionary")
    names = sorted({entry.view_name for entries in owned_partition.party_access.values()
                    for entry in entries})
    if any(name not in bundle.feature_views for name in names):
        raise ValueError("partition references unknown FeatureView")
    source_views = {name: bundle.feature_views[name] for name in names}
    domain_id = graph_domain_fingerprint(bundle)
    limits = {"user": len(bundle.user_id_map), "item": len(bundle.item_id_map),
              "interaction": len(bundle.interactions)}
    _validate_views(source_views, limits)
    validate_feature_partition(partition, bundle, parties)
    checksum = feature_partition_checksum(partition)
    identities = {name: _content_identity(view) for name, view in source_views.items()}
    views = {name: _copy_view(view) for name, view in source_views.items()}
    _validate_views(views, limits)
    if any(_content_identity(view) != identities[name] for name, view in views.items()):
        raise ValueError("copied feature content identity mismatch")
    if feature_partition_checksum(owned_partition) != checksum:
        raise ValueError("copied feature access identity mismatch")
    indexes = {name: _build_index(view) for name, view in views.items()}
    _validate_owned_references(owned_partition, views, indexes, roster)
    access_by_party = {
        party_id: owned_partition.access_for_party(party_id) for party_id in sorted(roster)
    }
    binding_id = _json_digest({
        "format": "thesisfedrec.feature_runtime", "version": 1,
        "source_id": source_id, "split_id": split_id, "recipe_id": recipe_id,
        "domain_fingerprint": domain_id, "interaction_count": limits["interaction"],
        "feature_fingerprints": identities, "partition_checksum": checksum,
        "roster": [{"party_id": party_id, "view_names": list(roster[party_id])}
                   for party_id in sorted(roster)],
    })
    # Do not publish a handle until every original and owned-data check succeeds.
    runtime = object.__new__(FeatureRuntime)
    runtime._binding_id = binding_id
    runtime._views = MappingProxyType(views)
    runtime._indexes = MappingProxyType(indexes)
    runtime._access_by_party = MappingProxyType(access_by_party)
    return runtime


def _copy_selected(array, rows, columns=None) -> np.ndarray:
    if columns is None:
        selected = array[rows]
    elif isinstance(rows, slice):
        selected = array[rows, columns]
    else:
        selected = array[np.ix_(rows, columns)]
    return np.array(selected, copy=True, order="C", subok=False)


def _project_view(view, access, index) -> FeatureView:
    rows, columns, names = _selection(view, access, index)
    if isinstance(view.values, np.ndarray):
        values = _copy_selected(view.values, rows, columns)
    elif isinstance(rows, slice):
        values = tuple(text for text in view.values)
    else:
        values = tuple(view.values[row] for row in rows)
    return FeatureView(
        name=view.name, entity_kind=view.entity_kind, modality=view.modality,
        entity_ids=_copy_selected(view.entity_ids, rows), values=values,
        feature_names=names, missing_mask=(
            None if view.missing_mask is None
            else _copy_selected(view.missing_mask, rows, columns)
        ), metadata={},
    )


def project_party_features(
    runtime: FeatureRuntime, *, party_id: int, binding_id: str,
) -> dict[str, FeatureView]:
    """Return detached authorized payloads using only the prepared snapshot/indexes."""
    check_feature_runtime_binding(runtime, binding_id=binding_id)
    _party_id(party_id)
    if party_id not in runtime._access_by_party:
        raise ValueError(f"unknown requested party_id {party_id}")
    return {
        access.view_name: _project_view(
            runtime._views[access.view_name], access, runtime._indexes[access.view_name],
        ) for access in runtime._access_by_party[party_id]
    }


def close_feature_runtime(runtime: FeatureRuntime) -> None:
    """Idempotently retire a correctly typed handle and release owned references."""
    if not isinstance(runtime, FeatureRuntime):
        raise TypeError("runtime must be a FeatureRuntime")
    runtime._binding_id = None
    runtime._views = None
    runtime._indexes = None
    runtime._access_by_party = None
