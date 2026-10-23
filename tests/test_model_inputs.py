"""Strict binary delivery: independent identities, permissions and owned snapshots."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
import hashlib
import json
import random
import struct
from types import MappingProxyType

import numpy as np
import pytest

from src.data.contracts import FeatureView
from src.data.local_field_maps import prepare_local_field_map
from src.data.model_inputs import (
    BinaryInputAdapter,
    close_binary_input_adapter,
    deliver_binary_field_units,
    prepare_binary_input_adapter,
)
from src.data import model_inputs as adapter_module


RUNTIME_ID = "a" * 64
ASSIGNMENTS = (("vB", "b"), ("vA", "x"), ("vA", "y"))
ARRAY_KEYS = ("entity_ids", "local_field_ids", "binary_values", "valid_unit_mask")
REQUEST = [3, 0, 3, 2, 4]
EXPECTED_IDS = [[0, -1, -1], [0, 1, -1], [0, -1, -1], [-1, 1, 2], [-1, -1, -1]]
EXPECTED_CODES = [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]]
EXPECTED_VALID = [[True, False, False], [True, True, False], [True, False, False],
                  [False, True, True], [False, False, False]]


class IntSubclass(int):
    pass


class StrSubclass(str):
    pass


class ArraySubclass(np.ndarray):
    pass


class TupleSubclass(tuple):
    pass


class ListSubclass(list):
    pass


def _view(name="v", rows=(0, 2), values=None, fields=("x", "y"), mask=None):
    if values is None:
        values = np.array([[0, 1], [1, 0]], dtype=np.int64)
    return FeatureView(
        name=name, entity_kind="item", entity_ids=np.array(rows, dtype=np.int64),
        modality="tabular", values=values, feature_names=fields,
        missing_mask=mask, metadata={},
    )


def _fixture(dtype=np.int64):
    """The task-card fixture, without deriving any expected output from production."""
    views = {
        "vA": _view("vA", values=np.array([[0, 1], [1, 0]], dtype=dtype),
                     mask=np.array([[False, True], [False, False]], dtype=bool)),
        "vB": _view("vB", rows=(0, 3), values=np.array([[1], [0]], dtype=dtype),
                     fields=("b",)),
    }
    local_map = prepare_local_field_map(
        party_id=41, schema_epoch=3, field_assignments=ASSIGNMENTS,
    )
    return views, local_map, {"vA": None, "vB": (0, 2, 3, 4)}


def _prepare(views=None, local_map=None, permissions=None, **overrides):
    if views is None:
        views, fixture_map, fixture_permissions = _fixture()
        if local_map is None:
            local_map = fixture_map
        if permissions is None:
            permissions = fixture_permissions
    arguments = dict(party_id=41, public_item_count=5, runtime_binding_id=RUNTIME_ID,
                     local_map=local_map, authorized_rows_by_view=permissions)
    arguments.update(overrides)
    return prepare_binary_input_adapter(views, **arguments)


def _deliver(adapter, request=REQUEST, **overrides):
    arguments = dict(entity_ids=request, expected_binding_id=adapter.binding_id)
    arguments.update(overrides)
    return deliver_binary_field_units(adapter, **arguments)


def _assert_same(left, right):
    assert type(left) is dict and set(left) == {"entity_kind", *ARRAY_KEYS}
    assert left["entity_kind"] == right["entity_kind"] == "item"
    for name in ARRAY_KEYS:
        np.testing.assert_array_equal(left[name], right[name])
        assert left[name].dtype == right[name].dtype
        assert left[name].shape == right[name].shape


def _assert_owned(output, rows, fields):
    assert type(output) is dict and set(output) == {"entity_kind", *ARRAY_KEYS}
    assert type(output["entity_kind"]) is str and output["entity_kind"] == "item"
    for name in ARRAY_KEYS:
        value = output[name]
        assert type(value) is np.ndarray
        assert value.shape == ((rows,) if name == "entity_ids" else (rows, fields))
        assert value.dtype == (np.bool_ if name == "valid_unit_mask" else np.int64)
        assert value.dtype.isnative and value.flags.owndata and value.flags.c_contiguous
        assert value.base is None
    for i, first in enumerate(ARRAY_KEYS):
        for second in ARRAY_KEYS[i + 1:]:
            assert not np.shares_memory(output[first], output[second])


def _json_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _fixture_binding_oracle():
    """Literal semantic JSON plus struct-packed little-endian input, no prod helper."""
    map_semantic = {
        "format": "thesisfedrec.local_field_map", "schema_version": 1,
        "entity_kind": "item", "map_algorithm": "explicit-order.v1",
        "party_id": 41, "schema_epoch": 3,
        "rows": [
            {"local_id": 0, "view_name": "vB", "field_name": "b"},
            {"local_id": 1, "view_name": "vA", "field_name": "x"},
            {"local_id": 2, "view_name": "vA", "field_name": "y"},
        ],
    }
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    envelope = {
        "format": "thesisfedrec.binary_input_adapter", "schema_version": 1,
        "policy_revision": "strict-binary-intersection.v1", "party_id": 41,
        "public_item_count": 5, "schema_epoch": 3,
        "map_id": _json_digest(map_semantic), "runtime_binding_id": RUNTIME_ID,
        "views": [
            {"view_name": "vA", "feature_names": ["x", "y"],
             "entity_count": 2, "field_count": 2, "values_dtype": "int64",
             "entity_ids_sha256": sha(struct.pack("<qq", 0, 2)),
             "values_sha256": sha(struct.pack("<qqqq", 0, 1, 1, 0)),
             "missing_mask_sha256": sha(bytes([0, 1, 0, 0])),
             "authorized_rows": None},
            {"view_name": "vB", "feature_names": ["b"],
             "entity_count": 2, "field_count": 1, "values_dtype": "int64",
             "entity_ids_sha256": sha(struct.pack("<qq", 0, 3)),
             "values_sha256": sha(struct.pack("<qq", 1, 0)),
             "missing_mask_sha256": None, "authorized_rows": [0, 2, 3, 4]},
        ],
    }
    return _json_digest(envelope)


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.float32, np.float64])
def test_frozen_hand_worked_projection_and_four_raw_types(dtype):
    views, local_map, permissions = _fixture(dtype)
    adapter = _prepare(views, local_map, permissions)
    output = _deliver(adapter)
    _assert_owned(output, 5, 3)
    np.testing.assert_array_equal(output["entity_ids"], REQUEST)
    np.testing.assert_array_equal(output["local_field_ids"], EXPECTED_IDS)
    np.testing.assert_array_equal(output["binary_values"], EXPECTED_CODES)
    np.testing.assert_array_equal(output["valid_unit_mask"], EXPECTED_VALID)
    with pytest.raises(ValueError):
        _deliver(adapter, [1])  # vA allows this ID, but vB does not.
    with pytest.raises(ValueError):
        _deliver(adapter, [5])
    absent = _deliver(adapter, [4])
    assert not absent["valid_unit_mask"].any()
    close_binary_input_adapter(adapter)


def test_binding_matches_independent_semantic_and_array_byte_oracle():
    adapter = _prepare()
    assert adapter.binding_id == _fixture_binding_oracle()
    assert len(adapter.binding_id) == 64
    assert adapter.binding_id == adapter.binding_id.lower()
    with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
        adapter.binding_id = "b" * 64
    close_binary_input_adapter(adapter)


def test_mapping_order_is_not_identity_and_column_order_is_representation_identity():
    views, local_map, permissions = _fixture()
    original = _prepare(views, local_map, permissions)
    reversed_mapping = _prepare(
        MappingProxyType(dict(reversed(tuple(views.items())))), local_map,
        MappingProxyType(dict(reversed(tuple(permissions.items())))),
    )
    assert original.binding_id == reversed_mapping.binding_id
    views["vA"].values = views["vA"].values[:, ::-1]
    views["vA"].missing_mask = views["vA"].missing_mask[:, ::-1]
    views["vA"].feature_names = ("y", "x")
    reordered = _prepare(views, local_map, permissions)
    assert reordered.binding_id != original.binding_id
    _assert_same(_deliver(original), _deliver(reordered))
    for adapter in (original, reversed_mapping, reordered):
        close_binary_input_adapter(adapter)


@pytest.mark.parametrize("change", ["mask", "raw_dtype", "runtime", "epoch", "domain", "masked_value"])
def test_representation_or_context_change_requires_new_binding(change):
    views, local_map, permissions = _fixture()
    before = _prepare(views, local_map, permissions)
    overrides = {}
    if change == "mask":
        views["vB"].missing_mask = np.zeros((2, 1), dtype=bool)
    elif change == "raw_dtype":
        views["vA"].values = views["vA"].values.astype(np.float64)
    elif change == "runtime":
        overrides["runtime_binding_id"] = "b" * 64
    elif change == "epoch":
        local_map = prepare_local_field_map(party_id=41, schema_epoch=4,
                                            field_assignments=ASSIGNMENTS)
    elif change == "domain":
        permissions["vA"] = (0, 1, 2, 3, 4)
    else:
        views["vA"].values[0, 1] = 0  # Still valid binary, but its masked raw bytes change.
    after = _prepare(views, local_map, permissions, **overrides)
    assert before.binding_id != after.binding_id
    _assert_same(_deliver(before), _deliver(after))
    with pytest.raises(ValueError):
        _deliver(after, expected_binding_id=before.binding_id)
    close_binary_input_adapter(before)
    close_binary_input_adapter(after)


@pytest.mark.parametrize("readonly", [False, True])
def test_strided_plain_arrays_and_readonly_inputs_are_not_modified(readonly):
    views, local_map, permissions = _fixture(np.float64)
    parents = []
    for view in views.values():
        for name in ("entity_ids", "values", "missing_mask"):
            source = getattr(view, name)
            if source is None:
                continue
            parent = np.empty(tuple(dimension * 2 for dimension in source.shape), dtype=source.dtype)
            parent.fill(0)
            index = tuple(slice(None, None, 2) for _ in source.shape)
            parent[index] = source
            strided = parent[index]
            if readonly:
                strided.setflags(write=False)
            setattr(view, name, strided)
            parents.append((parent, parent.tobytes(), strided, strided.strides, strided.flags.writeable))
    adapter = _prepare(views, local_map, permissions)
    output = _deliver(adapter)
    np.testing.assert_array_equal(output["binary_values"], EXPECTED_CODES)
    for parent, before, strided, strides, writeable in parents:
        assert parent.tobytes() == before
        assert strided.strides == strides and strided.flags.writeable == writeable
        assert all(not np.shares_memory(strided, output[key]) for key in ARRAY_KEYS)
    close_binary_input_adapter(adapter)


def test_negative_zero_is_observed_binary_zero_and_binding_retains_raw_sign_bit():
    views, local_map, permissions = _fixture(np.float64)
    positive = _prepare(views, local_map, permissions)
    views["vA"].values[0, 0] = -0.0
    negative = _prepare(views, local_map, permissions)
    assert positive.binding_id != negative.binding_id
    _assert_same(_deliver(positive), _deliver(negative))
    assert _deliver(negative, [0])["valid_unit_mask"][0, 1]
    for adapter in (positive, negative):
        close_binary_input_adapter(adapter)


@pytest.mark.parametrize("query", [[], ()])
def test_empty_request_preserves_exact_owned_shapes(query):
    adapter = _prepare()
    first = _deliver(adapter, query)
    second = _deliver(adapter, query)
    _assert_owned(first, 0, 3)
    _assert_owned(second, 0, 3)
    assert all(first[key] is not second[key] for key in ARRAY_KEYS)
    close_binary_input_adapter(adapter)


@pytest.mark.parametrize("count,query", [(0, []), (5, [4, 0, 4])])
def test_no_fields_requires_no_views_and_delivers_zero_columns(count, query):
    local_map = prepare_local_field_map(party_id=41, schema_epoch=3, field_assignments=())
    adapter = _prepare({}, local_map, {}, public_item_count=count)
    output = _deliver(adapter, query)
    _assert_owned(output, len(query), 0)
    np.testing.assert_array_equal(output["entity_ids"], query)
    if count == 0:
        with pytest.raises(ValueError):
            _deliver(adapter, [0])
    close_binary_input_adapter(adapter)


def test_zero_source_rows_and_all_masked_keep_invalid_ids_and_codes_sanitized():
    views, local_map, permissions = _fixture()
    views["vA"].entity_ids = np.empty(0, dtype=np.int64)
    views["vA"].values = np.empty((0, 2), dtype=np.int64)
    views["vA"].missing_mask = np.empty((0, 2), dtype=bool)
    views["vB"].missing_mask = np.ones((2, 1), dtype=bool)
    adapter = _prepare(views, local_map, permissions)
    output = _deliver(adapter)
    assert (output["local_field_ids"] == -1).all()
    assert (output["binary_values"] == 0).all()
    assert not output["valid_unit_mask"].any()
    close_binary_input_adapter(adapter)


@pytest.mark.parametrize("count,permission", [(0, None), (5, None), (5, ())])
def test_empty_source_keeps_explicit_empty_authorization_distinct_from_none(count, permission):
    view = _view(rows=(), values=np.empty((0, 1), dtype=np.int64), fields=("x",))
    local_map = prepare_local_field_map(party_id=41, schema_epoch=0,
                                        field_assignments=(("v", "x"),))
    adapter = _prepare({"v": view}, local_map, {"v": permission}, public_item_count=count)
    _assert_owned(_deliver(adapter, []), 0, 1)
    if count == 0 or permission == ():
        with pytest.raises(ValueError):
            _deliver(adapter, [0])
    else:
        output = _deliver(adapter, [0])
        np.testing.assert_array_equal(output["local_field_ids"], [[-1]])
        np.testing.assert_array_equal(output["binary_values"], [[0]])
        np.testing.assert_array_equal(output["valid_unit_mask"], [[False]])
    close_binary_input_adapter(adapter)


def test_none_authorization_supports_huge_sparse_catalogue_without_expansion():
    count = 2**63 - 1
    view = _view(rows=(0, count - 1))
    local_map = prepare_local_field_map(party_id=41, schema_epoch=0,
                                        field_assignments=(("v", "x"), ("v", "y")))
    adapter = _prepare({"v": view}, local_map, {"v": None}, public_item_count=count)
    output = _deliver(adapter, [count - 1, 1, 0])
    np.testing.assert_array_equal(output["binary_values"], [[1, 0], [0, 0], [0, 1]])
    np.testing.assert_array_equal(output["valid_unit_mask"], [[True, True], [False, False], [True, True]])
    with pytest.raises(ValueError):
        _deliver(adapter, [count])
    close_binary_input_adapter(adapter)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.25, 2.0, -1.0])
@pytest.mark.parametrize("cell", [(0, 1), (1, 0)])
def test_masked_and_future_unqueried_nonbinary_cells_fail_during_prepare(bad, cell):
    views, local_map, permissions = _fixture(np.float64)
    views["vA"].values[cell] = bad
    before = views["vA"].values.tobytes()
    with pytest.raises(ValueError):
        _prepare(views, local_map, permissions)
    assert views["vA"].values.tobytes() == before


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.uint64, np.float16, np.complex128,
                                   object, "U1", np.dtype([("a", "i8")]), ">i8", ">f4",
                                   np.dtype("i8", metadata={"source": "not-allowed"})])
def test_values_reject_unsupported_dtype_before_conversion(dtype):
    views, local_map, permissions = _fixture()
    views["vA"].values = np.zeros((2, 2), dtype=dtype)
    with pytest.raises(TypeError):
        _prepare(views, local_map, permissions)


@pytest.mark.parametrize("replacement", [
    [[0, 1], [1, 0]], ((0, 1), (1, 0)),
    np.array([[0, 1], [1, 0]]).view(ArraySubclass),
    np.ma.array([[0, 1], [1, 0]], mask=False),
])
def test_values_require_plain_ndarray(replacement):
    views, local_map, permissions = _fixture()
    views["vA"].values = replacement
    with pytest.raises(TypeError):
        _prepare(views, local_map, permissions)


@pytest.mark.parametrize("name,replacement,error", [
    ("entity_ids", [0, 2], TypeError),
    ("entity_ids", np.array([0, 2], dtype=np.int32), TypeError),
    ("entity_ids", np.array([0, 2], dtype=">i8"), TypeError),
    ("entity_ids", np.array([0, 2], dtype=np.dtype("i8", metadata={"x": 1})), TypeError),
    ("entity_ids", np.array([0, 2], dtype=np.int64).view(ArraySubclass), TypeError),
    ("entity_ids", np.array([[0, 2]], dtype=np.int64), ValueError),
    ("entity_ids", np.array([2, 0], dtype=np.int64), ValueError),
    ("entity_ids", np.array([0, 0], dtype=np.int64), ValueError),
    ("entity_ids", np.array([-1, 0], dtype=np.int64), ValueError),
    ("entity_ids", np.array([0, 5], dtype=np.int64), ValueError),
    ("entity_ids", np.array([0], dtype=np.int64), ValueError),
    ("values", np.array(1, dtype=np.int64), ValueError),
    ("values", np.ones(2, dtype=np.int64), ValueError),
    ("values", np.ones((2, 2, 1), dtype=np.int64), ValueError),
    ("values", np.ones((2, 1), dtype=np.int64), ValueError),
    ("missing_mask", [[False, False], [False, False]], TypeError),
    ("missing_mask", np.zeros((2, 2), dtype=np.int64), TypeError),
    ("missing_mask", np.zeros((2, 2), dtype=bool).view(ArraySubclass), TypeError),
    ("missing_mask", np.zeros((2, 2), dtype=np.dtype("?", metadata={"x": 1})), TypeError),
    ("missing_mask", np.zeros(2, dtype=bool), ValueError),
    ("missing_mask", np.zeros((2, 1), dtype=bool), ValueError),
    ("feature_names", ["x", "y"], TypeError),
    ("feature_names", TupleSubclass(("x", "y")), TypeError),
    ("feature_names", (StrSubclass("x"), "y"), TypeError),
    ("feature_names", ("x", "x"), ValueError),
    ("feature_names", ("x",), ValueError),
    ("feature_names", ("x", ""), ValueError),
    ("feature_names", ("x", " y"), ValueError),
    ("feature_names", ("x", "y\x00"), ValueError),
    ("feature_names", ("x", "y\x85"), ValueError),
    ("entity_kind", "user", ValueError),
    ("entity_kind", 1, TypeError),
    ("modality", "numeric", ValueError),
    ("modality", StrSubclass("tabular"), TypeError),
    ("name", "wrong", ValueError),
    ("name", "vA\x9f", ValueError),
    ("metadata", (), TypeError),
])
def test_mutable_view_is_revalidated_without_trusting_construction(name, replacement, error):
    views, local_map, permissions = _fixture()
    setattr(views["vA"], name, replacement)
    with pytest.raises(error):
        _prepare(views, local_map, permissions)


@pytest.mark.parametrize("container", [[], (), None, object()])
def test_view_and_permission_containers_must_be_mappings(container):
    views, local_map, permissions = _fixture()
    with pytest.raises(TypeError):
        prepare_binary_input_adapter(
            container, party_id=41, public_item_count=5, runtime_binding_id=RUNTIME_ID,
            local_map=local_map, authorized_rows_by_view=permissions,
        )
    with pytest.raises(TypeError):
        _prepare(views, local_map, container)


def test_values_require_exact_featureview_not_subclass_or_duck_record():
    class ViewSubclass(FeatureView):
        pass

    views, local_map, permissions = _fixture()
    original = views["vA"]
    views["vA"] = ViewSubclass(**{
        name: getattr(original, name) for name in FeatureView.__slots__
    })
    with pytest.raises(TypeError):
        _prepare(views, local_map, permissions)
    views["vA"] = {"name": "vA"}
    with pytest.raises(TypeError):
        _prepare(views, local_map, permissions)


@pytest.mark.parametrize("field,value,error", [
    ("party_id", True, TypeError), ("party_id", np.int64(41), TypeError),
    ("party_id", IntSubclass(41), TypeError), ("party_id", -1, ValueError),
    ("party_id", 2**63, ValueError), ("party_id", 42, ValueError),
    ("public_item_count", True, TypeError), ("public_item_count", np.int64(5), TypeError),
    ("public_item_count", IntSubclass(5), TypeError),
    ("public_item_count", -1, ValueError), ("public_item_count", 2**63, ValueError),
    ("runtime_binding_id", None, TypeError),
    ("runtime_binding_id", StrSubclass(RUNTIME_ID), TypeError),
    ("runtime_binding_id", "", ValueError),
    ("runtime_binding_id", "A" * 64, ValueError),
    ("runtime_binding_id", "g" * 64, ValueError),
    ("runtime_binding_id", "a" * 63, ValueError),
    ("local_map", {}, TypeError),
])
def test_prepare_scalar_identity_types_and_ranges(field, value, error):
    with pytest.raises(error):
        _prepare(**{field: value})


@pytest.mark.parametrize("assignments", [
    (("vA", "x"), ("vA", "y")),
    ASSIGNMENTS + (("vB", "extra"),),
    (("vB", "b"), ("vA", "x"), ("vA", "renamed")),
    (("other", "b"), ("vA", "x"), ("vA", "y")),
    (),
])
def test_field_pair_set_must_match_exactly_without_rename_or_expansion(assignments):
    local_map = prepare_local_field_map(party_id=41, schema_epoch=3,
                                        field_assignments=assignments)
    with pytest.raises(ValueError):
        _prepare(local_map=local_map)


@pytest.mark.parametrize("field,value", [
    ("map_id", "b" * 64), ("schema_epoch", 4),
    ("field_assignments", (("vB", "b"), ("vA", "x"), ("vA", "x"))),
])
def test_adapter_rejects_damaged_local_map_even_when_record_exists(field, value):
    views, local_map, permissions = _fixture()
    object.__setattr__(local_map, field, value)
    with pytest.raises(ValueError):
        _prepare(views, local_map, permissions)


@pytest.mark.parametrize("permissions,error", [
    ({"vA": None}, ValueError),
    ({"vA": None, "vB": None, "extra": None}, ValueError),
    ({"vA": None, "vB": [0, 2, 3, 4]}, TypeError),
    ({"vA": None, "vB": TupleSubclass((0, 2, 3, 4))}, TypeError),
    ({"vA": None, "vB": (False, 2, 3, 4)}, TypeError),
    ({"vA": None, "vB": (np.int64(0), 2, 3, 4)}, TypeError),
    ({"vA": None, "vB": (IntSubclass(0), 2, 3, 4)}, TypeError),
    ({"vA": None, "vB": (0, 3, 2, 4)}, ValueError),
    ({"vA": None, "vB": (0, 2, 3, 3)}, ValueError),
    ({"vA": None, "vB": (-1, 0, 2, 3)}, ValueError),
    ({"vA": None, "vB": (0, 2, 3, 5)}, ValueError),
    ({"vA": None, "vB": (0, 2, 4)}, ValueError),
    ({"vA": None, "vB": ()}, ValueError),
])
def test_authorization_mapping_is_exact_and_all_source_rows_must_be_authorized(permissions, error):
    views, local_map, _ = _fixture()
    with pytest.raises(error):
        _prepare(views, local_map, permissions)


@pytest.mark.parametrize("query,error", [
    (None, TypeError), (np.array([0], dtype=np.int64), TypeError), ({0}, TypeError),
    (ListSubclass([0]), TypeError), (TupleSubclass((0,)), TypeError),
    ([False], TypeError), ([np.int64(0)], TypeError), ([IntSubclass(0)], TypeError),
    ([0.0], TypeError), (["0"], TypeError), ([0, 3, -1], ValueError),
    ([0, 3, 5], ValueError), ([0, 3, 2**63], ValueError),
    ([0, 3, 1], ValueError),
])
def test_entire_request_validates_without_partially_returning(query, error):
    adapter = _prepare()
    before = _deliver(adapter)
    with pytest.raises(error):
        _deliver(adapter, query)
    _assert_same(_deliver(adapter), before)
    close_binary_input_adapter(adapter)


@pytest.mark.parametrize("binding,error", [
    (None, TypeError), (StrSubclass("a" * 64), TypeError),
    ("", ValueError), ("B" * 64, ValueError), ("g" * 64, ValueError),
    ("b" * 64, ValueError),
])
def test_empty_delivery_does_not_bypass_binding(binding, error):
    adapter = _prepare()
    with pytest.raises(error):
        _deliver(adapter, [], expected_binding_id=binding)
    close_binary_input_adapter(adapter)


def test_invalid_zero_axis_inputs_still_validate_schema_and_raw_types():
    local_map = prepare_local_field_map(party_id=41, schema_epoch=3,
                                        field_assignments=(("v", "x"),))
    view = _view(rows=(), values=np.empty((0, 1), dtype=np.float16), fields=("x",))
    with pytest.raises(TypeError):
        _prepare({"v": view}, local_map, {"v": None})
    no_fields = prepare_local_field_map(party_id=41, schema_epoch=3, field_assignments=())
    zero_columns = _view(rows=(), values=np.empty((0, 0), dtype=np.int64), fields=())
    with pytest.raises(ValueError):
        _prepare({"v": zero_columns}, no_fields, {"v": None})


def _owned_graph_ids(root):
    """Walk only instance-owned records/containers, not classes/modules/global references."""
    seen = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Mapping):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif type(current) in (tuple, list):
            pending.extend(current)
        elif isinstance(current, np.ndarray):
            if current.base is not None:
                pending.append(current.base)
        elif type(current).__module__.startswith("src.data."):
            for cls in type(current).__mro__:
                slots = getattr(cls, "__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                for name in slots:
                    if hasattr(current, name):
                        pending.append(getattr(current, name))
            if hasattr(current, "__dict__"):
                pending.append(vars(current))
    return seen


def test_snapshot_has_no_parent_metadata_gt_mapping_or_array_header_references():
    class ForbiddenGT:
        def __deepcopy__(self, memo):
            raise AssertionError("metadata/GT must not be copied")

        def __iter__(self):
            raise AssertionError("metadata/GT must not be traversed")

    views, local_map, permissions = _fixture()
    gt = ForbiddenGT()
    views["vA"].metadata["UNIQUE_GT_NOT_MODEL_INPUT_7e4b"] = gt
    forbidden = [views, permissions, gt]
    for view in views.values():
        forbidden.extend((view, view.metadata, view.entity_ids, view.values))
        if view.missing_mask is not None:
            forbidden.append(view.missing_mask)
    adapter = _prepare(views, local_map, permissions)
    original = _deliver(adapter)
    assert not ({id(value) for value in forbidden} & _owned_graph_ids(adapter))
    views["vA"].metadata.clear()
    gt.changed = "unrelated future GT"
    views["vA"].values[:] = 1
    views["vA"].values.shape = (4,)
    views["vA"].values.dtype = np.int32
    views["vA"].entity_ids[:] = [1, 4]
    views["vA"].missing_mask[:] = True
    permissions["vB"] = ()
    views.clear()
    _assert_same(_deliver(adapter), original)
    close_binary_input_adapter(adapter)


def test_metadata_dict_is_not_traversed_even_during_preparation():
    class OpaqueMetadata(dict):
        def __iter__(self):
            raise AssertionError("metadata iteration is not authorized")

        def items(self):
            raise AssertionError("metadata values are not authorized")

        def keys(self):
            raise AssertionError("metadata keys are not authorized")

        def values(self):
            raise AssertionError("metadata values are not authorized")

    views, local_map, permissions = _fixture()
    views["vA"].metadata = OpaqueMetadata(unique_private_GT=object())
    adapter = _prepare(views, local_map, permissions)
    assert adapter.binding_id == _fixture_binding_oracle()
    np.testing.assert_array_equal(_deliver(adapter)["binary_values"], EXPECTED_CODES)
    close_binary_input_adapter(adapter)


def test_all_outputs_are_independent_even_after_header_and_element_mutation():
    adapter = _prepare()
    first, second = _deliver(adapter), _deliver(adapter)
    expected = {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in second.items()}
    for key in ARRAY_KEYS:
        assert not np.shares_memory(first[key], second[key])
        first[key].fill(False if key == "valid_unit_mask" else 99)
        first[key].shape = (first[key].size,)
        if key != "valid_unit_mask":
            first[key].dtype = np.int32
    _assert_same(second, expected)
    _assert_same(_deliver(adapter), expected)
    close_binary_input_adapter(adapter)
    _assert_same(second, expected)


def test_close_is_explicit_independent_and_releases_owned_payloads():
    first, second = _prepare(), _prepare()
    delivered = _deliver(first)
    before_ids = _owned_graph_ids(first)
    close_binary_input_adapter(first)
    close_binary_input_adapter(first)
    assert len(_owned_graph_ids(first)) < len(before_ids)
    with pytest.raises(ValueError):
        _deliver(first, [])
    _assert_same(_deliver(second), delivered)
    close_binary_input_adapter(second)


def test_handle_has_no_public_construction_and_wrong_objects_are_rejected():
    with pytest.raises(TypeError):
        BinaryInputAdapter()
    for wrong in (None, {}, object()):
        with pytest.raises(TypeError):
            deliver_binary_field_units(wrong, entity_ids=[], expected_binding_id="a" * 64)
        with pytest.raises(TypeError):
            close_binary_input_adapter(wrong)


def test_late_prepare_failure_does_not_modify_inputs_or_existing_handle():
    views, local_map, permissions = _fixture(np.float64)
    first = _prepare(views, local_map, permissions)
    expected = _deliver(first)
    views["vB"].values[1, 0] = 0.5  # vA is valid; later sorted view is invalid.
    before = {name: view.values.tobytes() for name, view in views.items()}
    with pytest.raises(ValueError):
        _prepare(views, local_map, permissions)
    assert {name: view.values.tobytes() for name, view in views.items()} == before
    _assert_same(_deliver(first), expected)
    close_binary_input_adapter(first)


def test_invalid_request_and_permissions_fail_before_output_allocation_or_lookup(monkeypatch):
    adapter = _prepare()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid request must fail before allocation/source lookup")

    for name in ("array", "full", "zeros", "frombuffer"):
        monkeypatch.setattr(np, name, forbidden)
    for request, error in (([0, 3, 1], ValueError), ([0, 3, 5], ValueError),
                           ([0, 3, False], TypeError)):
        with pytest.raises(error):
            _deliver(adapter, request)
    close_binary_input_adapter(adapter)


def test_all_source_structures_validate_before_any_content_scan(monkeypatch):
    views, local_map, permissions = _fixture(np.float64)
    views["vA"].values[0, 0] = np.nan
    views["vB"].values = np.zeros((2, 1), dtype=np.float16)

    def forbidden(*args, **kwargs):
        raise AssertionError("all structures must pass before any raw content scan")

    monkeypatch.setattr(adapter_module, "_validate_content", forbidden)
    with pytest.raises(TypeError):
        _prepare(views, local_map, permissions)


@pytest.mark.parametrize("container", ["views", "permissions"])
@pytest.mark.parametrize("key", [41, StrSubclass("vA"), "vA\x85"])
def test_mapping_keys_require_legal_exact_strings(container, key):
    views, local_map, permissions = _fixture()
    target = views if container == "views" else permissions
    target[key] = target.pop("vA")
    with pytest.raises(TypeError if type(key) is not str else ValueError):
        _prepare(views, local_map, permissions)


def test_repeated_delivery_never_revalidates_rehashes_or_rebuilds_source(monkeypatch):
    adapter = _prepare()
    expected = _deliver(adapter)

    def forbidden(*args, **kwargs):
        raise AssertionError("delivery must only use prepared owned indexes and scalars")

    for name in ("_validate_view_types", "_validate_view_shape", "_validate_content",
                 "_array_hash", "_make_view_snapshot", "_validate_local_map", "_canonical_bytes"):
        monkeypatch.setattr(adapter_module, name, forbidden)
    monkeypatch.setattr(hashlib, "sha256", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "seed", forbidden)
    monkeypatch.setattr(random, "seed", forbidden)
    for _ in range(3):
        _assert_same(_deliver(adapter), expected)
    _assert_owned(_deliver(adapter, []), 0, 3)
    close_binary_input_adapter(adapter)


def test_prepare_delivery_and_close_preserve_python_and_numpy_rng():
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    adapter = _prepare()
    _deliver(adapter)
    _deliver(adapter, [])
    close_binary_input_adapter(adapter)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
