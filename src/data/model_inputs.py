"""Strict, owned binary deliveries for trusted framework callers, not models.

No parent runtime, metadata or mutable input objects survive preparation.
The local map establishes column identities; this module neither creates model
state nor authenticates the caller's permissions/provenance assertions.
"""

from collections.abc import Mapping
import hashlib

import numpy as np

from .contracts import FeatureView
from .local_field_maps import (
    LocalFieldMap,
    _canonical_bytes,
    _validate_hash,
    _validate_int,
    _validate_local_map,
    _validate_text,
)


class BinaryInputAdapter:
    """Prepare-only trusted handle; private storage is not a public interface."""

    __slots__ = ("_binding_id", "_item_count", "_field_count", "_views")

    def __new__(cls):
        raise TypeError("use prepare_binary_input_adapter()")

    @property
    def binding_id(self) -> str:
        """Read the prepared identity; retired handles have no live identity."""
        _require_adapter(self)
        return self._binding_id


def _require_adapter(adapter: BinaryInputAdapter) -> None:
    if type(adapter) is not BinaryInputAdapter:
        raise TypeError("adapter must be a BinaryInputAdapter")
    if getattr(adapter, "_binding_id", None) is None:
        raise ValueError("binary input adapter is closed or unprepared")


def _check_array(value: np.ndarray, name: str, dtypes: tuple) -> None:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a plain ndarray")
    dtype = value.dtype
    if (
        dtype.metadata is not None
        or dtype.fields is not None
        or dtype.subdtype is not None
        or not dtype.isnative
        or dtype not in dtypes
    ):
        raise TypeError(f"{name} has an unsupported dtype")


def _validate_view_types(name: str, view: FeatureView) -> None:
    if type(view) is not FeatureView:
        raise TypeError("authorized_views values must be exact FeatureView objects")
    for label in ("name", "entity_kind", "modality"):
        _validate_text(getattr(view, label), f"view.{label}")
    if view.name != name:
        raise ValueError("authorized view key must equal view.name")
    if view.entity_kind != "item" or view.modality != "tabular":
        raise ValueError("only item/tabular views are supported")
    if not isinstance(view.metadata, dict):
        raise TypeError("view.metadata must be a dict")
    if type(view.feature_names) is not tuple:
        raise TypeError("feature_names must be an exact tuple")
    for field in view.feature_names:
        _validate_text(field, "feature name")
    _check_array(view.entity_ids, "entity_ids", (np.dtype("int64"),))
    _check_array(view.values, "values", tuple(map(np.dtype, ("bool", "int64", "float32", "float64"))))
    if view.missing_mask is not None:
        _check_array(view.missing_mask, "missing_mask", (np.dtype("bool"),))


def _validate_view_shape(view: FeatureView, item_count: int) -> None:
    if view.entity_ids.ndim != 1:
        raise ValueError("entity_ids must have rank 1")
    if view.values.ndim != 2:
        raise ValueError("values must have rank 2")
    if view.values.shape != (len(view.entity_ids), len(view.feature_names)):
        raise ValueError("values shape must match entity rows and feature columns")
    if len(set(view.feature_names)) != len(view.feature_names):
        raise ValueError("feature_names must be unique")
    if view.missing_mask is not None and view.missing_mask.shape != view.values.shape:
        raise ValueError("missing_mask shape must equal values shape")
    ids = view.entity_ids
    if np.any(ids < 0) or np.any(ids >= item_count):
        raise ValueError("source entity ID outside public item domain")
    if np.any(ids[1:] <= ids[:-1]):
        raise ValueError("source entity IDs must be sorted and unique")


def _validate_authorized_rows(rows: tuple[int, ...] | None, item_count: int) -> None:
    if rows is None:
        return
    if type(rows) is not tuple:
        raise TypeError("authorized rows must be an exact tuple or None")
    for entity_id in rows:
        _validate_int(entity_id, "authorized entity ID")
    if any(entity_id >= item_count for entity_id in rows):
        raise ValueError("authorized entity ID outside public item domain")
    if any(left >= right for left, right in zip(rows, rows[1:])):
        raise ValueError("authorized rows must be sorted and unique")


def _validate_content(view: FeatureView) -> None:
    # Check every selected cell before conversion, including masked/unused rows.
    if not np.all(np.isfinite(view.values)):
        raise ValueError("binary source values must be finite, including masked cells")
    if np.any((view.values != 0) & (view.values != 1)):
        raise ValueError("binary source values must be exactly 0 or 1")


def _array_hash(values: np.ndarray) -> str:
    canonical = values.astype(values.dtype.newbyteorder("<"), copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _make_view_snapshot(view: FeatureView, rows: tuple[int, ...] | None, columns: tuple) -> tuple:
    """Own only lookup integers, immutable numeric bytes and local column pairs."""
    codes = np.array(view.values, dtype=np.int64, order="C", copy=True)
    valid = np.ones(view.values.shape, dtype=np.bool_)
    if view.missing_mask is not None:
        valid[view.missing_mask] = False
        codes[view.missing_mask] = 0
    return (
        None if rows is None else frozenset(rows),
        {int(entity_id): row for row, entity_id in enumerate(view.entity_ids)},
        codes.tobytes(order="C"),
        valid.tobytes(order="C"),
        tuple(view.values.shape),
        columns,
    )


def prepare_binary_input_adapter(
    authorized_views: Mapping[str, FeatureView],
    *,
    party_id: int,
    public_item_count: int,
    runtime_binding_id: str,
    local_map: LocalFieldMap,
    authorized_rows_by_view: Mapping[str, tuple[int, ...] | None],
) -> BinaryInputAdapter:
    """Validate/hash once and own a projection snapshot with explicit rights.

    Keep all inputs stable during this call. None rights mean the full public
    domain, not merely rows present in the projection. No catalogue is expanded.
    """
    _validate_int(party_id, "party_id")
    _validate_int(public_item_count, "public_item_count")
    _validate_hash(runtime_binding_id, "runtime_binding_id")
    _validate_local_map(local_map)
    if local_map.party_id != party_id:
        raise ValueError("local map belongs to another party")
    if not isinstance(authorized_views, Mapping) or not isinstance(authorized_rows_by_view, Mapping):
        raise TypeError("authorized views and row permissions must be mappings")

    # Validate all outer types before sorting or reading contents. These shallow
    # dictionaries are temporary: no caller view/mapping escapes in the handle.
    views = dict(authorized_views)
    permissions = dict(authorized_rows_by_view)
    for name in views:
        _validate_text(name, "view key")
    for name in permissions:
        _validate_text(name, "permission key")
    for name, view in views.items():
        _validate_view_types(name, view)
    for rows in permissions.values():
        _validate_authorized_rows(rows, public_item_count)
    for view in views.values():
        _validate_view_shape(view, public_item_count)

    assignments = local_map.field_assignments
    expected_names = {name for name, _ in assignments}
    if set(views) != expected_names or set(permissions) != expected_names:
        raise ValueError("view and permission names must exactly match the local map")
    actual_pairs = {(name, field) for name, view in views.items() for field in view.feature_names}
    if actual_pairs != set(assignments):
        raise ValueError("view fields must exactly match local map assignments")
    for name, view in views.items():
        rows = permissions[name]
        if rows is not None:
            allowed = frozenset(rows)
            if any(int(entity_id) not in allowed for entity_id in view.entity_ids):
                raise ValueError("projection contains an unauthorized source row")
    for view in views.values():
        _validate_content(view)

    records = []
    snapshots = []
    columns_by_view = {name: [] for name in views}
    for local_id, (name, field) in enumerate(assignments):
        columns_by_view[name].append((local_id, field))
    for name in sorted(views):
        view = views[name]
        field_index = {field: column for column, field in enumerate(view.feature_names)}
        columns = tuple((local_id, field_index[field]) for local_id, field in columns_by_view[name])
        records.append({
            "view_name": name,
            "feature_names": list(view.feature_names),
            "entity_count": len(view.entity_ids),
            "field_count": len(view.feature_names),
            "values_dtype": view.values.dtype.name,
            "entity_ids_sha256": _array_hash(view.entity_ids),
            "values_sha256": _array_hash(view.values),
            "missing_mask_sha256": None if view.missing_mask is None else _array_hash(view.missing_mask),
            "authorized_rows": None if permissions[name] is None else list(permissions[name]),
        })
        snapshots.append(_make_view_snapshot(view, permissions[name], columns))
    envelope = {
        "format": "thesisfedrec.binary_input_adapter",
        "schema_version": 1,
        "policy_revision": "strict-binary-intersection.v1",
        "party_id": party_id,
        "public_item_count": public_item_count,
        "schema_epoch": local_map.schema_epoch,
        "map_id": local_map.map_id,
        "runtime_binding_id": runtime_binding_id,
        "views": records,
    }
    binding = hashlib.sha256(_canonical_bytes(envelope)).hexdigest()
    adapter = object.__new__(BinaryInputAdapter)
    adapter._binding_id = binding
    adapter._item_count = public_item_count
    adapter._field_count = len(assignments)
    adapter._views = tuple(snapshots)
    return adapter


def deliver_binary_field_units(
    adapter: BinaryInputAdapter,
    *,
    entity_ids: list[int] | tuple[int, ...],
    expected_binding_id: str,
) -> dict[str, object]:
    """Own fresh output arrays, preserving requested row order and multiplicity."""
    _require_adapter(adapter)
    _validate_hash(expected_binding_id, "expected_binding_id")
    if expected_binding_id != adapter._binding_id:
        raise ValueError("binary input adapter binding mismatch")
    if type(entity_ids) not in (list, tuple):
        raise TypeError("entity_ids must be an exact list or tuple")
    requested = tuple(entity_ids)
    for entity_id in requested:
        if type(entity_id) is not int:
            raise TypeError("requested entity IDs must be exact built-in ints")
    if any(entity_id < 0 or entity_id >= adapter._item_count for entity_id in requested):
        raise ValueError("requested entity ID outside public item domain")
    # Full request authorization precedes output allocation or source lookup.
    for allowed, *_ in adapter._views:
        if allowed is not None and any(entity_id not in allowed for entity_id in requested):
            raise ValueError("request is not authorized by every selected view")

    shape = (len(requested), adapter._field_count)
    ids = np.array(requested, dtype=np.int64, order="C", copy=True)
    local_ids = np.full(shape, -1, dtype=np.int64)
    codes = np.zeros(shape, dtype=np.int64)
    valid = np.zeros(shape, dtype=np.bool_)
    for _, row_index, code_bytes, valid_bytes, source_shape, columns in adapter._views:
        # Headers only over owned immutable bytes, never a source-sized copy/scan.
        source_codes = np.frombuffer(code_bytes, dtype=np.int64).reshape(source_shape)
        source_valid = np.frombuffer(valid_bytes, dtype=np.bool_).reshape(source_shape)
        for output_row, entity_id in enumerate(requested):
            source_row = row_index.get(entity_id)
            if source_row is None:
                continue
            for local_id, source_column in columns:
                if source_valid[source_row, source_column]:
                    local_ids[output_row, local_id] = local_id
                    codes[output_row, local_id] = source_codes[source_row, source_column]
                    valid[output_row, local_id] = True
    return {
        "entity_kind": "item",
        "entity_ids": ids,
        "local_field_ids": local_ids,
        "binary_values": codes,
        "valid_unit_mask": valid,
    }


def close_binary_input_adapter(adapter: BinaryInputAdapter) -> None:
    """Release owned buffers/indexes; idempotent, without reclaiming deliveries."""
    if type(adapter) is not BinaryInputAdapter:
        raise TypeError("adapter must be a BinaryInputAdapter")
    adapter._binding_id = None
    adapter._views = None
    adapter._item_count = None
    adapter._field_count = None
