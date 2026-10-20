"""Owned feature snapshots: independent projections, binding and resource checks."""

from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
import random
import sys
import weakref

import numpy as np
import pytest
import torch

from src.data import (
    FeatureAccess, FeaturePartition, FeatureRuntime, FeatureView,
    build_complete_feature_scenario, build_single_view_absence_scenario,
    check_feature_runtime_binding, close_feature_runtime,
    feature_partition_checksum, graph_domain_fingerprint, prepare_feature_runtime,
    project_party_features, resolve_party_feature_views,
)
from src.data import feature_runtime as runtime_module
from src.data import features as strict_module
from test_feature_resolution import _assignment, _party, feature_bundle


CONTEXT = dict(source_id="synthetic/source-v1", split_id="split-v1", recipe_id="recipe-v1")
ARRAY_FIELDS = ("entity_ids", "values", "missing_mask")


def _access(name="item_signals", rows=None, fields=None):
    return FeatureAccess(view_name=name, entity_ids=(
        None if rows is None else np.array(rows, dtype=np.int64)
    ), feature_names=fields)


def _prepare(bundle, accesses=None, **context):
    if accesses is None:
        accesses = {
            41: (_access("item_text"), _access(fields=("flag", "score"), rows=(2, 11))),
            7: tuple(_access(name) for name in reversed(bundle.feature_views)),
            901: (),
        }
    partition, parties = _assignment(accesses)
    runtime = prepare_feature_runtime(bundle, partition, parties, **(CONTEXT | context))
    return runtime, partition, parties


def _project(runtime, party_id=7):
    return project_party_features(runtime, party_id=party_id, binding_id=runtime.binding_id)


def _equal(actual, expected):
    assert tuple(actual) == tuple(expected)
    for name, view in actual.items():
        other = expected[name]
        for field in ("name", "entity_kind", "modality", "feature_names"):
            assert getattr(view, field) == getattr(other, field)
        for field in ARRAY_FIELDS:
            left, right = getattr(view, field), getattr(other, field)
            if isinstance(right, np.ndarray):
                assert type(left) is np.ndarray
                assert left.dtype.str == right.dtype.str and left.shape == right.shape
                assert left.tobytes() == right.tobytes()
                assert left.flags.owndata and left.flags.c_contiguous
                assert left.base is None
            else:
                assert left == right
        assert view.metadata == {}


def _reference(view, access):
    """Python membership/positions oracle, independent of runtime lookup logic."""
    selected = None if access.entity_ids is None else set(access.entity_ids.tolist())
    rows = [i for i, entity in enumerate(view.entity_ids.tolist())
            if selected is None or entity in selected]
    columns = [i for i, name in enumerate(view.feature_names)
               if access.feature_names is None or name in access.feature_names]

    def array(values):
        if view.feature_names and values is not view.entity_ids:
            # Explicit separate row/column construction, including empty axes.
            output = np.empty((len(rows), len(columns)), dtype=values.dtype)
            for i, row in enumerate(rows):
                for j, column in enumerate(columns):
                    output[i, j] = values[row, column]
            return output
        output = np.empty((len(rows), *values.shape[1:]), dtype=values.dtype)
        for i, row in enumerate(rows):
            output[i] = values[row]
        return output

    return FeatureView(
        name=view.name, entity_kind=view.entity_kind, modality=view.modality,
        entity_ids=array(view.entity_ids),
        values=(array(view.values) if isinstance(view.values, np.ndarray)
                else tuple(view.values[row] for row in rows)),
        feature_names=tuple(view.feature_names[column] for column in columns),
        missing_mask=None if view.missing_mask is None else array(view.missing_mask),
        metadata={},
    )


def _check_oracles(runtime, bundle, partition, parties, party_id):
    actual = _project(runtime, party_id)
    expected = {
        entry.view_name: _reference(bundle.feature_views[entry.view_name], entry)
        for entry in sorted(partition.access_for_party(party_id), key=lambda a: a.view_name)
    }
    _equal(actual, expected)
    _equal(actual, resolve_party_feature_views(bundle, partition, parties, party_id=party_id))
    return actual


def test_hand_expected_sparse_ids_fields_and_mask(feature_bundle):
    runtime, partition, parties = _prepare(feature_bundle)
    local = _check_oracles(runtime, feature_bundle, partition, parties, 41)
    assert list(local) == ["item_signals", "item_text"]
    np.testing.assert_array_equal(local["item_signals"].entity_ids, [2, 11])
    assert local["item_signals"].feature_names == ("score", "flag")
    np.testing.assert_array_equal(local["item_signals"].values, [[10, 1], [30, 1]])
    np.testing.assert_array_equal(local["item_signals"].missing_mask, [[False, False], [True, False]])
    assert local["item_text"].values == ("two", "seven", "eleven")
    for party_id in (7, 901):
        _check_oracles(runtime, feature_bundle, partition, parties, party_id)


@pytest.mark.parametrize("rows", [None, (), (2, 11), (2, 7, 11)])
@pytest.mark.parametrize("fields", [None, (), ("flag", "score"), ("flag", "count", "score")])
def test_named_selections(feature_bundle, rows, fields):
    runtime, partition, parties = _prepare(feature_bundle, {7: (_access(rows=rows, fields=fields),)})
    _check_oracles(runtime, feature_bundle, partition, parties, 7)


@pytest.mark.parametrize("name,rows", [
    ("item_text", None), ("item_text", ()), ("item_text", (7, 11)),
    ("user_tensor", None), ("user_tensor", ()), ("user_tensor", (2,)),
    ("context", None), ("context", ()), ("context", (8, 23)),
])
def test_unnamed_tensor_text_and_missingness(feature_bundle, name, rows):
    runtime, partition, parties = _prepare(feature_bundle, {7: (_access(name, rows),)})
    output = _check_oracles(runtime, feature_bundle, partition, parties, 7)[name]
    if name == "user_tensor":
        assert output.missing_mask is None
    if name == "context" and rows is None:
        assert np.isnan(output.values[1]) and not output.missing_mask[1]
        assert output.values[0] == 0 and not output.missing_mask[0]


@pytest.mark.parametrize("name", ["item_text", "user_tensor", "context"])
@pytest.mark.parametrize("fields", [(), ("not-named",)])
def test_fields_on_unnamed_views_fail_even_when_empty(feature_bundle, name, fields):
    with pytest.raises(ValueError, match="field-level"):
        _prepare(feature_bundle, {7: (), 41: (_access(name, (), fields),)})


@pytest.mark.parametrize("explicit", [False, True])
def test_featureless_and_empty_roster(feature_bundle, explicit):
    partition, parties = _assignment({7: ()})
    if not explicit:
        partition = replace(partition, party_access={})
    runtime = prepare_feature_runtime(feature_bundle, partition, parties, **CONTEXT)
    assert _project(runtime) == {}
    assert _project(runtime) is not _project(runtime)
    empty, _, _ = _prepare(feature_bundle, {})
    with pytest.raises(ValueError, match="unknown"):
        _project(empty)


@pytest.mark.parametrize("name", ["item_signals", "item_text", "user_tensor"])
def test_zero_row_canonical_views(feature_bundle, name):
    view = feature_bundle.feature_views[name]
    view.entity_ids = view.entity_ids[:0]
    view.values = view.values[:0]
    if view.missing_mask is not None:
        view.missing_mask = view.missing_mask[:0]
    runtime, partition, parties = _prepare(feature_bundle, {7: (_access(name),)})
    _check_oracles(runtime, feature_bundle, partition, parties, 7)


def _digest(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _reference_content(view):
    def identity(array):
        return {"shape": list(array.shape), "dtype": array.dtype.str,
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    return _digest({
        "format": "thesisfedrec.feature_content", "version": 1,
        "name": view.name, "entity_kind": view.entity_kind, "modality": view.modality,
        "feature_names": list(view.feature_names), "entity_ids": identity(view.entity_ids),
        "values": ({"kind": "array", "array": identity(view.values)}
                   if isinstance(view.values, np.ndarray)
                   else {"kind": "text", "items": list(view.values)}),
        "missing_mask": None if view.missing_mask is None else identity(view.missing_mask),
    })


def test_independently_serialized_binding_and_input_nonmutation(feature_bundle):
    original = deepcopy(feature_bundle)
    runtime, partition, parties = _prepare(feature_bundle)
    expected = _digest({
        "format": "thesisfedrec.feature_runtime", "version": 1, **CONTEXT,
        "domain_fingerprint": graph_domain_fingerprint(feature_bundle),
        "interaction_count": 24,
        "feature_fingerprints": {name: _reference_content(view)
                                 for name, view in feature_bundle.feature_views.items()},
        "partition_checksum": feature_partition_checksum(partition),
        "roster": [{"party_id": party.party_id, "view_names": sorted(party.feature_view_names)}
                   for party in sorted(parties, key=lambda party: party.party_id)],
    })
    assert runtime.binding_id == expected
    for name, view in feature_bundle.feature_views.items():
        assert _reference_content(view) == _reference_content(original.feature_views[name])
        assert view.metadata == original.feature_views[name].metadata
        assert view.entity_ids.flags.writeable
    assert feature_bundle.user_id_map == original.user_id_map
    assert feature_bundle.item_id_map == original.item_id_map


def test_order_metadata_layout_and_sample_payload_do_not_change_binding(feature_bundle):
    runtime, partition, parties = _prepare(feature_bundle)
    old = runtime.binding_id
    for view in feature_bundle.feature_views.values():
        view.metadata["unrelated"] = object()
        if isinstance(view.values, np.ndarray):
            view.values = np.asfortranarray(view.values)
    reversed_access = {
        party_id: tuple(FeatureAccess(view_name=entry.view_name, entity_ids=entry.entity_ids,
                                     feature_names=None if entry.feature_names is None
                                     else tuple(reversed(entry.feature_names)))
                        for entry in reversed(entries))
        for party_id, entries in reversed(list(partition.party_access.items()))
    }
    partition = FeaturePartition(name="changed", party_access=reversed_access,
                                 metadata={"ignored": object()})
    for party in parties:
        party.train_interaction_ids = np.array([999], dtype=np.int64)
        party.feature_view_names = tuple(reversed(party.feature_view_names))
        party.metadata["ignored"] = object()
    # Binding intentionally asserts split/row semantics rather than scanning them.
    feature_bundle.interactions.timestamps[:] = -1
    feature_bundle.splits.train_interaction_ids = np.array([], dtype=np.int64)
    assert prepare_feature_runtime(feature_bundle, partition, tuple(reversed(parties)),
                                   **CONTEXT).binding_id == old


@pytest.mark.parametrize("change", [
    "values", "entity_ids", "mask", "mask_none", "dtype", "modality", "fields",
    "kind", "text", "domain", "interaction_count", "access", "roster",
    "source_id", "split_id", "recipe_id",
])
def test_rebuild_binding_tracks_verified_changes(feature_bundle, change):
    runtime, partition, parties = _prepare(feature_bundle, {7: (_access(), _access("item_text"))})
    before = _project(runtime)
    view = feature_bundle.feature_views["item_signals"]
    context = dict(CONTEXT)
    if change == "values":
        view.values[0, 0] += 1
    elif change == "entity_ids":
        view.entity_ids[0] = 1
    elif change == "mask":
        view.missing_mask[0, 0] = True
    elif change == "mask_none":
        view.missing_mask = None
    elif change == "dtype":
        view.values = view.values.astype(">f4")
    elif change == "modality":
        view.modality = "changed"
    elif change == "fields":
        view.feature_names = ("new", "count", "flag")
    elif change == "kind":
        view.entity_kind = "interaction"
    elif change == "text":
        feature_bundle.feature_views["item_text"].values = ("", "七", "eleven")
    elif change == "domain":
        feature_bundle.item_id_map = {f"changed-{i}": i for i in range(12)}
    elif change == "interaction_count":
        feature_bundle.interactions = replace(
            feature_bundle.interactions,
            **{field: getattr(feature_bundle.interactions, field)[:-1]
               for field in ("interaction_ids", "user_ids", "item_ids", "timestamps", "values")},
        )
    elif change == "access":
        partition = replace(partition, party_access={7: (_access(rows=()), _access("item_text"))})
    elif change == "roster":
        parties = (*parties, _party(12345))
    else:
        context[change] = "changed"
    rebuilt = prepare_feature_runtime(feature_bundle, partition, parties, **context)
    assert rebuilt.binding_id != runtime.binding_id
    with pytest.raises(ValueError, match="binding mismatch"):
        project_party_features(rebuilt, party_id=7, binding_id=runtime.binding_id)
    _equal(_project(runtime), before)


def test_signed_zero_nan_payload_and_none_explicit_full_identity(feature_bundle):
    view = feature_bundle.feature_views["context"]
    runtime, partition, parties = _prepare(feature_bundle, {7: (_access("context"),)})
    first = runtime.binding_id
    view.values[0] = -0.0
    second = prepare_feature_runtime(feature_bundle, partition, parties, **CONTEXT).binding_id
    assert first != second
    view.values.view(np.uint64)[1] = 0x7FF8000000000001
    third = prepare_feature_runtime(feature_bundle, partition, parties, **CONTEXT).binding_id
    assert third != second
    explicit, _, _ = _prepare(feature_bundle, {7: (_access("context", (1, 8, 23)),)})
    assert explicit.binding_id != third


class _String(str):
    pass


class _Integer(int):
    pass


class _Array(np.ndarray):
    pass


@pytest.mark.parametrize("field", ["source_id", "split_id", "recipe_id"])
@pytest.mark.parametrize("value", [None, 7, "", " \t", _String("s")])
def test_context_validation(feature_bundle, field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        _prepare(feature_bundle, **{field: value})


@pytest.mark.parametrize("field", ARRAY_FIELDS)
@pytest.mark.parametrize("restriction", ["subclass", "masked", "metadata", "empty_metadata"])
def test_array_preflight_before_empty_projection(feature_bundle, field, restriction):
    source = feature_bundle.feature_views["item_signals"]
    array = getattr(source, field)
    if restriction == "subclass":
        array = array.view(_Array)
    elif restriction == "masked":
        array = np.ma.array(array, mask=False)
    else:
        array = array.view(np.dtype(array.dtype, metadata=(
            {} if restriction == "empty_metadata" else {"private": [1]}
        )))
    setattr(source, field, array)
    with pytest.raises(TypeError, match="plain NumPy|unsupported delivery dtype"):
        _prepare(feature_bundle, {7: (), 41: (_access(rows=(), fields=()),)})


@pytest.mark.parametrize("dtype", [object, np.dtype([("private", "i4")]), np.dtype([("private", object)])])
def test_unsafe_value_dtypes_even_for_empty_selection(feature_bundle, dtype):
    feature_bundle.feature_views["item_signals"].values = np.zeros((3, 3), dtype=dtype)
    with pytest.raises(TypeError, match="unsupported delivery dtype"):
        _prepare(feature_bundle, {7: (_access(rows=(), fields=()),)})


@pytest.mark.parametrize("dtype", [
    "f4", ">f4", "f8", "i2", "c8", "?", "S3", "U3", "M8[D]", "m8[D]", "V4", "V0",
])
@pytest.mark.parametrize("empty", [False, True])
def test_delivery_safe_plain_dtypes_preserve_exact_storage(feature_bundle, dtype, empty):
    source = feature_bundle.feature_views["item_signals"]
    source.values = np.zeros((3, 3), dtype=dtype)
    access = _access(rows=() if empty else (2, 11), fields=("flag", "score"))
    runtime, partition, parties = _prepare(feature_bundle, {7: (access,)})
    _check_oracles(runtime, feature_bundle, partition, parties, 7)
    owned = runtime._views["item_signals"].values
    assert owned.dtype.str == source.values.dtype.str
    assert owned.tobytes() == source.values.tobytes()
    with pytest.raises(ValueError):
        owned.setflags(write=True)


@pytest.mark.parametrize("layout", ["fortran", "strided", "negative_stride", "read_only"])
def test_noncontiguous_source_layout(feature_bundle, layout):
    source = feature_bundle.feature_views["item_signals"]
    if layout == "fortran":
        source.values = np.asfortranarray(source.values)
    elif layout == "strided":
        backing = np.zeros((6, 6), dtype=source.values.dtype)
        backing[::2, ::2] = source.values
        source.values = backing[::2, ::2]
    elif layout == "negative_stride":
        source.values = source.values[::-1, ::-1].copy()[::-1, ::-1]
    else:
        source.values.setflags(write=False)
    runtime, partition, parties = _prepare(feature_bundle, {7: (_access(),)})
    _check_oracles(runtime, feature_bundle, partition, parties, 7)
    owned = runtime._views["item_signals"].values
    assert owned.flags.c_contiguous and not np.shares_memory(owned, source.values)


@pytest.mark.parametrize("field", ["name", "entity_kind", "modality", "feature_names", "text", "access", "fields", "roster"])
def test_plain_schema_and_text_even_when_unselected(feature_bundle, field):
    access = _access(rows=(), fields=())
    source = feature_bundle.feature_views["item_signals"]
    if field in ("name", "entity_kind", "modality"):
        setattr(source, field, _String(getattr(source, field)))
    elif field == "feature_names":
        source.feature_names = (_String("score"), "count", "flag")
    elif field == "text":
        source = feature_bundle.feature_views["item_text"]
        source.values = (_String("two"), "seven", "eleven")
        access = _access("item_text", ())
    elif field == "access":
        access = FeatureAccess(view_name=_String("item_signals"), entity_ids=np.array([], np.int64))
    elif field == "fields":
        access = _access(rows=(), fields=(_String("score"),))
    partition, parties = _assignment({7: (), 41: (access,)})
    if field == "roster":
        parties[1].feature_view_names = (_String("item_signals"),)
    with pytest.raises(TypeError, match="plain"):
        prepare_feature_runtime(feature_bundle, partition, parties, **CONTEXT)


@pytest.mark.parametrize("bad_id", [True, False, np.int64(7), _Integer(7), -1, 7.0, "7", None])
def test_strict_party_ids_requests_and_roster(feature_bundle, bad_id):
    runtime, partition, parties = _prepare(feature_bundle)
    with pytest.raises(ValueError, match="party_id"):
        _project(runtime, bad_id)
    parties[0].party_id = bad_id
    with pytest.raises(ValueError, match="party_id"):
        prepare_feature_runtime(feature_bundle, partition, parties, **CONTEXT)


def test_partition_integer_subclass_rejected(feature_bundle):
    with pytest.raises(ValueError, match="party_id"):
        _prepare(feature_bundle, {_Integer(7): (_access(),)})


@pytest.mark.parametrize("failure", [
    "unknown_view", "unknown_row", "unknown_field", "unknown_party",
    "duplicate_party", "summary", "duplicate_summary", "summary_list",
    "bad_view", "view_key", "unsorted_ids", "domain_ids", "values_shape", "mask_shape",
])
def test_all_parties_and_referenced_views_validated_before_handle(feature_bundle, failure):
    partition, parties = _assignment({7: (), 41: (_access(),)})
    view = feature_bundle.feature_views["item_signals"]
    if failure == "unknown_view":
        feature_bundle.feature_views.clear()
    elif failure == "unknown_row":
        partition = replace(partition, party_access={7: (), 41: (_access(rows=(3,)),)})
    elif failure == "unknown_field":
        partition = replace(partition, party_access={7: (), 41: (_access(fields=("bad",)),)})
    elif failure == "unknown_party":
        parties = parties[:1]
    elif failure == "duplicate_party":
        parties = (*parties, parties[0])
    elif failure == "summary":
        parties[1].feature_view_names = ()
    elif failure == "duplicate_summary":
        parties[1].feature_view_names = ("item_signals", "item_signals")
    elif failure == "summary_list":
        parties[1].feature_view_names = ["item_signals"]
    elif failure == "bad_view":
        feature_bundle.feature_views["item_signals"] = object()
    elif failure == "view_key":
        view.name = "changed"
    elif failure == "unsorted_ids":
        view.entity_ids[:] = [7, 2, 11]
    elif failure == "domain_ids":
        view.entity_ids[-1] = 12
    elif failure == "values_shape":
        view.values.shape = (9,)
    elif failure == "mask_shape":
        view.missing_mask = np.zeros(3, dtype=bool)
    with pytest.raises((TypeError, ValueError)):
        prepare_feature_runtime(feature_bundle, partition, parties, **CONTEXT)


@pytest.mark.parametrize("failure", ["structure", "domain", "valid_content", "copy_access"])
def test_owned_copy_revalidation_rejects_partial_handle(feature_bundle, monkeypatch, failure):
    runtime, partition, parties = _prepare(feature_bundle)
    original = _project(runtime)
    copy_view = runtime_module._copy_view
    copy_partition = runtime_module._copy_partition

    def bad_copy(view):
        copied = copy_view(view)
        if view.name == "item_signals":
            if failure == "structure":
                copied.values = np.zeros(3, dtype=np.float32)
            elif failure == "domain":
                copied.entity_ids = np.array([2, 7, 12], dtype=np.int64)
            elif failure == "valid_content":
                copied.values = np.zeros_like(copied.values)
        return copied

    def bad_access(value):
        copied = copy_partition(value)
        entries = dict(copied.party_access)
        entries[41] = (_access(rows=()), _access("item_text"))
        return replace(copied, party_access=entries)

    monkeypatch.setattr(runtime_module, "_copy_view", bad_copy)
    if failure == "copy_access":
        monkeypatch.setattr(runtime_module, "_copy_partition", bad_access)
    with pytest.raises(ValueError):
        prepare_feature_runtime(feature_bundle, partition, parties, **CONTEXT)
    _equal(_project(runtime), original)


def test_unreferenced_views_ignored_and_originals_released(feature_bundle):
    feature_bundle.feature_views["unreferenced"] = object()
    runtime, partition, parties = _prepare(feature_bundle, {7: (_access(),), 41: ()})
    before = _project(runtime)
    source = feature_bundle.feature_views["item_signals"]
    arrays = [weakref.ref(getattr(source, field)) for field in ARRAY_FIELDS]

    class Sentinel:
        def __deepcopy__(self, memo):
            raise AssertionError("metadata must not be traversed")

    sentinel = Sentinel()
    marker = weakref.ref(sentinel)
    source.metadata["sentinel"] = sentinel
    partition.metadata["sentinel"] = sentinel
    # Preparation must not retain arbitrary nested metadata.
    second = prepare_feature_runtime(feature_bundle, partition, parties, **CONTEXT)
    assert second.binding_id == runtime.binding_id
    samples = weakref.ref(parties[0].train_interaction_ids)
    feature_bundle.feature_views.clear()
    del source, parties, partition, sentinel
    gc.collect()
    assert all(ref() is None for ref in arrays)
    assert samples() is None and marker() is None
    _equal(_project(second), before)


def test_no_retained_domain_maps_or_interaction_arrays(feature_bundle):
    class WeakMap(dict):
        pass
    feature_bundle.user_id_map = WeakMap(feature_bundle.user_id_map)
    feature_bundle.item_id_map = WeakMap(feature_bundle.item_id_map)
    originals = [weakref.ref(feature_bundle.user_id_map), weakref.ref(feature_bundle.item_id_map)]
    originals.extend(weakref.ref(getattr(feature_bundle.interactions, name)) for name in (
        "interaction_ids", "user_ids", "item_ids", "timestamps", "values",
    ))
    runtime, _, _ = _prepare(feature_bundle)
    expected = _project(runtime)
    feature_bundle.user_id_map = {}
    feature_bundle.item_id_map = {}
    feature_bundle.interactions = None
    gc.collect()
    assert all(ref() is None for ref in originals)
    _equal(_project(runtime), expected)


@pytest.mark.parametrize("field", ARRAY_FIELDS)
@pytest.mark.parametrize("mutation", ["contents", "shape", "dtype"])
def test_source_array_and_header_isolation(feature_bundle, field, mutation):
    runtime, _, _ = _prepare(feature_bundle)
    expected, binding = _project(runtime), runtime.binding_id
    array = getattr(feature_bundle.feature_views["item_signals"], field)
    if mutation == "contents":
        array.fill(0)
    elif mutation == "shape":
        array.shape = (array.size,)
    else:
        array.dtype = np.uint8
    feature_bundle.feature_views["item_signals"].name = "changed"
    feature_bundle.feature_views["item_signals"].metadata["changed"] = object()
    assert runtime.binding_id == binding
    _equal(_project(runtime), expected)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("field", ARRAY_FIELDS)
@pytest.mark.parametrize("mutation", ["contents", "shape", "dtype", "flags"])
def test_output_isolation_across_parties_and_future_calls(feature_bundle, empty, field, mutation):
    access = _access(rows=() if empty else None, fields=() if empty else None)
    runtime, partition, parties = _prepare(feature_bundle, {7: (access,), 41: (access,)})
    first, second = _project(runtime), _project(runtime, 41)
    array = getattr(first["item_signals"], field)
    assert not np.shares_memory(array, getattr(second["item_signals"], field))
    if mutation == "contents":
        array.fill(0)
    elif mutation == "shape":
        array.shape = (array.size,)
    elif mutation == "dtype":
        array.dtype = np.uint8
    else:
        array.setflags(write=False)
    first["item_signals"].metadata["changed"] = [1]
    first["item_signals"].feature_names = ()
    _equal(second, resolve_party_feature_views(feature_bundle, partition, parties, party_id=41))
    for party_id in (7, 41):
        _check_oracles(runtime, feature_bundle, partition, parties, party_id)


def test_runtime_lifecycle_releases_owned_buffers_and_keeps_outputs(feature_bundle):
    with pytest.raises(TypeError, match="prepare_feature_runtime"):
        FeatureRuntime()
    runtime, _, _ = _prepare(feature_bundle)
    identity, output = runtime.binding_id, _project(runtime)
    buffers = [weakref.ref(getattr(runtime._views["item_signals"], name)) for name in ARRAY_FIELDS]
    for name in ARRAY_FIELDS:
        array = getattr(runtime._views["item_signals"], name)
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 0
    del array
    with pytest.raises(AttributeError):
        runtime.binding_id = "changed"
    with pytest.raises(ValueError, match="unknown"):
        _project(runtime, 99999)
    with pytest.raises(ValueError, match="binding mismatch"):
        project_party_features(runtime, party_id=7, binding_id="wrong")
    with pytest.raises(TypeError, match="binding_id"):
        check_feature_runtime_binding(runtime, binding_id=_String(identity))
    close_feature_runtime(runtime)
    close_feature_runtime(runtime)
    assert all(ref() is None for ref in buffers)
    for operation in (
        lambda: runtime.binding_id,
        lambda: check_feature_runtime_binding(runtime, binding_id=identity),
        lambda: project_party_features(runtime, party_id=7, binding_id=identity),
    ):
        with pytest.raises(ValueError, match="closed"):
            operation()
    for operation in (close_feature_runtime,
                      lambda r: check_feature_runtime_binding(r, binding_id=identity),
                      lambda r: project_party_features(r, party_id=7, binding_id=identity)):
        with pytest.raises(TypeError, match="FeatureRuntime"):
            operation(None)
    assert output["item_signals"].values.flags.owndata


@pytest.mark.parametrize("invalid_input", ["bundle", "partition", "sequence", "party"])
def test_top_level_input_types(feature_bundle, invalid_input):
    partition, parties = _assignment({7: (_access(),)})
    args = [feature_bundle, partition, parties]
    if invalid_input == "bundle":
        args[0] = None
    elif invalid_input == "partition":
        args[1] = None
    elif invalid_input == "sequence":
        args[2] = iter(parties)
    else:
        args[2] = (object(),)
    with pytest.raises(TypeError):
        prepare_feature_runtime(*args, **CONTEXT)


def test_preserves_python_numpy_and_torch_rng(feature_bundle):
    originals = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(5678)
        before = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
        runtime, _, _ = _prepare(feature_bundle)
        _project(runtime)
        close_feature_runtime(runtime)
        assert random.getstate() == before[0]
        after = np.random.get_state()
        assert after[0] == before[1][0] and after[2:] == before[1][2:]
        np.testing.assert_array_equal(after[1], before[1][1])
        assert torch.equal(torch.random.get_rng_state(), before[2])
    finally:
        random.setstate(originals[0])
        np.random.set_state(originals[1])
        torch.random.set_rng_state(originals[2])


@pytest.mark.parametrize("recipe", ["complete", "absence"])
def test_existing_whole_view_recipes(feature_bundle, recipe):
    parties = (_party(7), _party(41), _party(901))
    kwargs = dict(view_names=("item_text", "item_signals"), name="synthetic")
    if recipe == "complete":
        partition, parties, _ = build_complete_feature_scenario(feature_bundle, parties, **kwargs)
    else:
        partition, parties, _ = build_single_view_absence_scenario(
            feature_bundle, parties, **kwargs, missing_view="item_text",
            missing_party_count=1, seed=2026,
        )
    runtime = prepare_feature_runtime(feature_bundle, partition, parties, **CONTEXT)
    for party in parties:
        _check_oracles(runtime, feature_bundle, partition, parties, party.party_id)


def test_no_global_work_on_repeated_delivery(feature_bundle, monkeypatch):
    counters = {}

    def instrument(module, name):
        original = getattr(module, name)
        key = module.__name__ + "." + name
        counters[key] = 0
        def wrapped(*args, **kwargs):
            counters[key] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(module, name, wrapped)

    for name in ("_validate_roster", "_copy_partition", "_validate_views",
                 "validate_feature_partition", "feature_partition_checksum",
                 "graph_domain_fingerprint", "_content_identity", "_json_digest",
                 "_array_identity", "_copy_view", "_build_index", "_validate_owned_references"):
        instrument(runtime_module, name)
    instrument(strict_module, "resolve_party_feature_views")
    instrument(strict_module, "validate_feature_partition")

    class CountingDict(dict):
        def items(self):
            counters["namespace_items"] += 1
            return super().items()
        def __iter__(self):
            counters["namespace_iter"] += 1
            return super().__iter__()

    counters.update(namespace_items=0, namespace_iter=0)
    feature_bundle.item_id_map = CountingDict(feature_bundle.item_id_map)
    feature_bundle.user_id_map = CountingDict(feature_bundle.user_id_map)
    runtime, partition, parties = _prepare(feature_bundle)
    before = dict(counters)
    assert before[runtime_module.__name__ + "._build_index"] == 4
    assert before["namespace_items"] > 0 and before["namespace_iter"] > 0
    visited = []
    original_project = runtime_module._project_view
    def projected(view, *args):
        visited.append(view.name)
        return original_project(view, *args)
    monkeypatch.setattr(runtime_module, "_project_view", projected)
    for _ in range(25):
        _project(runtime, 41)
        _project(runtime, 901)
    assert counters == before
    assert visited == ["item_signals", "item_text"] * 25
    strict_module.resolve_party_feature_views(feature_bundle, partition, parties, party_id=41)
    assert counters[strict_module.__name__ + ".validate_feature_partition"] == 1


def test_bounded_buffer_accounting(feature_bundle, record_property):
    rows, fields = 4096, 12
    view = FeatureView(
        name="bounded", entity_kind="item", modality="numeric",
        entity_ids=np.arange(rows, dtype=np.int64),
        values=np.arange(rows * fields, dtype=np.float32).reshape(rows, fields),
        feature_names=tuple(f"f{i}" for i in range(fields)),
        missing_mask=np.zeros((rows, fields), dtype=bool), metadata={},
    )
    # Only current domain maps and referenced views matter here; no raw data scan.
    feature_bundle.item_id_map = {f"item-{i}": i for i in range(rows)}
    feature_bundle.feature_views = {view.name: view}
    runtime, _, _ = _prepare(feature_bundle, {
        party: (_access("bounded"),) for party in (7, 41, 901, 1234, 9999)
    })
    source_arrays = [getattr(view, name) for name in ARRAY_FIELDS]
    snapshot_arrays = [getattr(runtime._views["bounded"], name) for name in ARRAY_FIELDS]
    source_bytes = sum(a.nbytes for a in source_arrays)
    snapshot_bytes = sum(a.nbytes for a in snapshot_arrays)
    assert source_bytes == snapshot_bytes == rows * (8 + fields * (4 + 1))
    assert len(runtime._views) == len(runtime._indexes) == 1
    assert all(entry.entity_ids is None and entry.feature_names is None
               for entries in runtime._access_by_party.values() for entry in entries)
    for original, owned in zip(source_arrays, snapshot_arrays):
        assert not np.shares_memory(original, owned)
        assert not owned.flags.writeable
    full = _project(runtime)["bounded"]
    output_bytes = sum(getattr(full, name).nbytes for name in ARRAY_FIELDS)
    assert output_bytes == snapshot_bytes
    partial, _, _ = _prepare(feature_bundle, {
        7: (_access("bounded", tuple(range(128)), ("f11", "f0", "f3")),),
    })
    selected = _project(partial)["bounded"]
    selected_bytes = sum(getattr(selected, name).nbytes for name in ARRAY_FIELDS)
    assert selected_bytes == 128 * (8 + 3 * (4 + 1))
    assert selected.feature_names == ("f0", "f3", "f11")
    # Python indexes are additional storage, not hidden inside ndarray counts.
    dictionaries = [gc.get_referents(proxy)[0] for proxy in runtime._indexes["bounded"]]
    objects = {id(obj): obj for mapping in dictionaries for pair in mapping.items() for obj in pair}
    index_bytes = sum(sys.getsizeof(mapping) for mapping in dictionaries) + sum(
        sys.getsizeof(obj) for obj in objects.values()
    )
    report = {
        "rows": rows, "fields": fields, "parties": 5, "snapshot_views": 1,
        "source_arrays": len(source_arrays), "snapshot_arrays": len(snapshot_arrays),
        "source_payload_bytes": source_bytes, "snapshot_payload_bytes": snapshot_bytes,
        "source_plus_snapshot_payload_bytes": source_bytes + snapshot_bytes,
        "full_output_arrays": 3, "full_output_payload_bytes": output_bytes,
        "selected_output_arrays": 3, "selected_output_payload_bytes": selected_bytes,
        "row_index_entries": len(dictionaries[0]), "field_index_entries": len(dictionaries[1]),
        "index_dict_and_unique_entry_shallow_bytes": index_bytes,
        "limits": "Excludes ndarray/bytes headers, runtime containers and temporary validation/indexing buffers; not peak RSS.",
    }
    record_property("bounded_buffer_inventory", json.dumps(report, sort_keys=True))
