"""PW8 public numeric delivery and source-slot geometry, without a trainer.

Expected rows, masks, inverses and linear gradients are frozen literals, then
cross-checked with independent NumPy loops. No expected value calls a new helper.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass
import inspect
import json
from pathlib import Path
import random
import sys
from types import MappingProxyType

import numpy as np
import pytest
import torch

from src.federation import doi4_batch as batch

FROZEN_FIXTURE_TEXT = r'''{
  "fixture_id": "PW8-B-DOI4-BATCH-K0-literal-v1",
  "context_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "online_party_ids": [7, 41, 90],
  "receiver_party_ids": [7, 90],
  "receiver_item_ids": [[4, 1, 4], [1, 3]],
  "source_party_ids_by_receiver": [[7, 41], [41, 90]],
  "source_party_ids": [7, 41, 90],
  "source_item_ids": [[1, 4], [1, 3, 4], [1, 3]],
  "receiver_source_row_indices": [[[1, 0, 1], [2, 0, 2]], [[0, 1], [0, 1]]],
  "slot_count": 2,
  "compatible_dim": 3,
  "source_H": {
    "7": [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
    "41": [[[21, 22, 23], [24, 25, 26]], [[27, 28, 29], [30, 31, 32]], [[33, 34, 35], [36, 37, 38]]],
    "90": [[[41, 42, 43], [44, 45, 46]], [[47, 48, 49], [50, 51, 52]]]
  },
  "source_available": {"7": [true, false], "41": [true, false, true], "90": [false, true]},
  "expected_tokens": [
    [[[0, 0, 0], [0, 0, 0], [33, 34, 35], [36, 37, 38]], [[1, 2, 3], [4, 5, 6], [21, 22, 23], [24, 25, 26]], [[0, 0, 0], [0, 0, 0], [33, 34, 35], [36, 37, 38]]],
    [[[21, 22, 23], [24, 25, 26], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [47, 48, 49], [50, 51, 52]]]
  ],
  "expected_masks": [[[false, false, true, true], [true, true, true, true], [false, false, true, true]], [[true, true, false, false], [false, false, true, true]]],
  "expected_slot_source_party_ids": [[7, 7, 41, 41], [41, 41, 90, 90]],
  "expected_slot_indices": [[0, 1, 0, 1], [0, 1, 0, 1]],
  "linear_weights": [
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]], [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 35, 36]]],
    [[[37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]], [[49, 50, 51], [52, 53, 54], [55, 56, 57], [58, 59, 60]]]
  ],
  "expected_source_gradients": {
    "7": [[[13, 14, 15], [16, 17, 18]], [[0, 0, 0], [0, 0, 0]]],
    "41": [[[56, 58, 60], [62, 64, 66]], [[0, 0, 0], [0, 0, 0]], [[38, 40, 42], [44, 46, 48]]],
    "90": [[[0, 0, 0], [0, 0, 0]], [[55, 56, 57], [58, 59, 60]]]
  },
  "tolerances": {"float64_atol": 1e-12, "float64_rtol": 1e-12, "float32_atol": 1e-5, "float32_rtol": 1e-5},
  "oracle": "Literal source-row/slot outputs and literal gradients from summing fixed receiver linear weights over each valid source row; independent NumPy loop cross-check never calls the new helpers.",
  "composition_case": {"source_party": 7, "receiver_party": 90, "source_items": [0, 1], "receiver_items": [1, 0, 1], "field_embedding": 0.2, "value_embeddings": [0.3, 0.5], "w_value": 2.0, "queries": [0.5, -0.25], "expected_source_H": [[[1.0], [1.0]], [[1.4], [1.4]]], "expected_logits": [1.625, 1.425, 2.025], "labels": [1, 0, 1], "group_offsets": [0, 2, 3], "loss_row_weights": [0.25, 0.25, 0.5], "oracle": "One observed field makes both T slots identical. Zero F query/key produces a uniform mean. Scalar scorer is 2*(item+0.5*H)+0.025. BCE derivatives use the independent analytic sigmoid and group weights."},
  "boundary_cases": "Literal strict packets, reflected plan mutations, scalar/zero-axis capacity, malformed late publication allocation sentinel, absent None versus evaluated allFalse including E0, strided/nonleaf/Parameter, owning output storage, huge finite inactive values, exact zero valid slots, RNG/input/grad/version preservation, direct-construction rejection and AST dependency/signature checks are retained in test source. No external old oracle or training."
}
'''
FIXTURE = json.loads(FROZEN_FIXTURE_TEXT)
CONTEXT = FIXTURE["context_id"]
MAX_INT = 2**63 - 1


class IntSubclass(int):
    pass


class StrSubclass(str):
    pass


class TupleSubclass(tuple):
    pass


class DictSubclass(dict):
    pass


class ArraySubclass(np.ndarray):
    pass


class ForbiddenMetadata:
    def __getattribute__(self, name):
        raise AssertionError("metadata/GT sentinel was read")


def _packet(rows=(4, 1, 4), width=2):
    entity_ids = np.array(rows, dtype=np.int64)
    ids = np.tile(np.arange(width, dtype=np.int64), (len(rows), 1))
    values = np.zeros((len(rows), width), dtype=np.int64)
    mask = np.ones((len(rows), width), dtype=bool)
    if len(rows) and width:
        values[0, 0] = 1
        ids[-1, -1], values[-1, -1], mask[-1, -1] = -1, 0, False
    return dict(entity_kind="item", entity_ids=entity_ids,
                local_field_ids=ids, binary_values=values, valid_unit_mask=mask)


def _tensorize(payload=None, **overrides):
    kwargs = dict(expected_entity_ids=(4, 1, 4), public_item_count=5,
                  local_field_count=2)
    kwargs.update(overrides)
    return batch.tensorize_binary_delivery(_packet() if payload is None else payload,
                                           **kwargs)


def _plan(**overrides):
    kwargs = dict(source_party_ids_by_receiver={7: (7, 41), 90: (41, 90)},
                  online_party_ids=(7, 41, 90), public_item_count=5,
                  context_id=CONTEXT)
    receiver = overrides.pop("receiver_item_ids", {7: (4, 1, 4), 90: (1, 3)})
    kwargs.update(overrides)
    return batch.prepare_doi4_batch_requests(receiver, **kwargs)


def _publications(dtype=torch.float64, *, requires_grad=False):
    return {source: (tuple(FIXTURE["source_item_ids"][index]),
                     torch.tensor(FIXTURE["source_H"][str(source)], dtype=dtype,
                                  requires_grad=requires_grad),
                     torch.tensor(FIXTURE["source_available"][str(source)],
                                  dtype=torch.bool))
            for index, source in enumerate((7, 41, 90))}


def _pack(plan=None, publications=None, **overrides):
    kwargs = dict(source_publications=_publications() if publications is None else publications,
                  slot_count=2, compatible_dim=3, dtype=torch.float64,
                  expected_context_id=CONTEXT)
    kwargs.update(overrides)
    return batch.pack_doi4_source_slots(_plan() if plan is None else plan, **kwargs)


def _close(actual, expected, dtype=torch.float64):
    actual_array = actual.detach().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
    expected_array = np.asarray(expected)
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    np.testing.assert_allclose(actual_array, expected_array, atol=tol, rtol=tol)
    return float(np.max(np.abs(actual_array - expected_array))) if actual_array.size else 0.0


def _independent_linear_gradients():
    """Only frozen rows/pools/source tables/coefficients enter this oracle."""
    result = {source: np.zeros_like(np.array(FIXTURE["source_H"][str(source)], dtype=np.float64))
              for source in (7, 41, 90)}
    for receiver_index, rows in enumerate(FIXTURE["receiver_item_ids"]):
        pools = FIXTURE["source_party_ids_by_receiver"][receiver_index]
        weights = FIXTURE["linear_weights"][receiver_index]
        for row_index, item_id in enumerate(rows):
            for source_index, source in enumerate(pools):
                unique_rows = FIXTURE["source_item_ids"][(7, 41, 90).index(source)]
                source_row = unique_rows.index(item_id)
                if FIXTURE["source_available"][str(source)][source_row]:
                    for slot in range(2):
                        for coordinate in range(3):
                            result[source][source_row, slot, coordinate] += weights[row_index][2 * source_index + slot][coordinate]
    return result


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_frozen_literal_geometry_slots_and_one_shared_backward(dtype, record_property):
    plan = _plan()
    assert plan.context_id == CONTEXT and plan.public_item_count == 5
    for name in ("online_party_ids", "receiver_party_ids", "receiver_item_ids",
                 "source_party_ids_by_receiver", "source_party_ids", "source_item_ids",
                 "receiver_source_row_indices"):
        def tuples(value):
            return tuple(tuples(part) for part in value) if type(value) is list else value
        assert getattr(plan, name) == tuples(FIXTURE[name])
    publications = _publications(dtype, requires_grad=True)
    snapshots = {source: (tokens.detach().clone(), available.clone(), tokens._version)
                 for source, (_, tokens, available) in publications.items()}
    outputs = _pack(plan, publications, dtype=dtype)
    max_forward = max_gradient = 0.0
    objective = None
    oracle = _independent_linear_gradients()
    for index, output in enumerate(outputs):
        assert type(output) is batch.ReceiverSlotBatch
        assert output.receiver_party_id == (7, 90)[index]
        assert output.entity_ids == tuple(FIXTURE["receiver_item_ids"][index])
        assert output.slot_source_party_ids == tuple(FIXTURE["expected_slot_source_party_ids"][index])
        assert output.slot_indices == tuple(FIXTURE["expected_slot_indices"][index])
        max_forward = max(max_forward, _close(output.compatible_tokens, FIXTURE["expected_tokens"][index], dtype))
        assert output.valid_token_mask.tolist() == FIXTURE["expected_masks"][index]
        assert output.compatible_tokens.is_contiguous() and output.valid_token_mask.is_contiguous()
        assert output.compatible_tokens._base is None and output.valid_token_mask._base is None
        assert output.compatible_tokens.requires_grad
        assert torch.equal(output.compatible_tokens[~output.valid_token_mask],
                           torch.zeros_like(output.compatible_tokens[~output.valid_token_mask]))
        coefficients = torch.tensor(FIXTURE["linear_weights"][index], dtype=dtype)
        term = (coefficients * output.compatible_tokens).sum()
        objective = term if objective is None else objective + term
    for source in (7, 41, 90):
        np.testing.assert_array_equal(oracle[source], FIXTURE["expected_source_gradients"][str(source)])
        assert publications[source][1].grad is None
    objective.backward()
    for source, (_, tokens, available) in publications.items():
        expected = FIXTURE["expected_source_gradients"][str(source)]
        max_gradient = max(max_gradient, _close(tokens.grad, expected, dtype))
        assert torch.equal(tokens.detach(), snapshots[source][0])
        assert torch.equal(available, snapshots[source][1])
        assert tokens._version == snapshots[source][2]
    record_property("literal_forward_max_abs_error", max_forward)
    record_property("linear_shared_H_gradient_max_abs_error", max_gradient)
    record_property("frozen_float_tolerances", json.dumps(FIXTURE["tolerances"], sort_keys=True))


def test_plan_ignores_dict_insertion_order_and_owns_recursive_builtin_values():
    receiver = {90: (1, 3), 7: (4, 1, 4)}
    pools = {90: (41, 90), 7: (7, 41)}
    plan = _plan(receiver_item_ids=receiver, source_party_ids_by_receiver=pools)
    expected = _plan()
    for field in fields(plan):
        assert getattr(plan, field.name) == getattr(expected, field.name)
    receiver[7] = (0,)
    pools.clear()
    assert plan.receiver_item_ids == ((4, 1, 4), (1, 3))
    assert plan.source_party_ids_by_receiver == ((7, 41), (41, 90))
    def recursive(value):
        assert type(value) in (str, int, tuple)
        if type(value) is tuple:
            for entry in value:
                recursive(entry)
    for field in fields(plan):
        recursive(getattr(plan, field.name))


def test_tensorize_preserves_observed_zero_and_copies_every_output():
    payload = _packet()
    first, second = _tensorize(payload), _tensorize(payload)
    assert first.entity_ids.tolist() == [4, 1, 4]
    assert first.local_field_ids.tolist() == [[0, 1], [0, 1], [0, -1]]
    assert first.binary_values.tolist() == [[1, 0], [0, 0], [0, 0]]
    assert first.valid_unit_mask.tolist() == [[True, True], [True, True], [True, False]]
    arrays = [payload[name] for name in ("entity_ids", "local_field_ids", "binary_values", "valid_unit_mask")]
    tensors = [getattr(output, name) for output in (first, second)
               for name in ("entity_ids", "local_field_ids", "binary_values", "valid_unit_mask")]
    pointers = [value.untyped_storage().data_ptr() for value in tensors]
    assert len(set(pointers)) == len(pointers)
    assert not set(pointers) & {array.ctypes.data for array in arrays}
    for tensor in tensors:
        assert tensor.device.type == "cpu" and tensor.is_contiguous()
        assert tensor._base is None and not tensor.requires_grad and tensor.grad_fn is None
    for name, tensor in zip(("entity_ids", "local_field_ids", "binary_values", "valid_unit_mask"), tensors[:4]):
        assert tensor.dtype == (torch.bool if name == "valid_unit_mask" else torch.int64)
    payload["entity_ids"][0] = 0
    payload["binary_values"].fill(1)
    first.binary_values.fill_(7)
    first.entity_ids.resize_(1)
    assert second.entity_ids.tolist() == [4, 1, 4]
    assert second.binary_values.tolist() == [[1, 0], [0, 0], [0, 0]]


def test_tensorize_accepts_readonly_strided_plain_arrays_and_no_alias():
    payload = _packet()
    for name in ("entity_ids", "local_field_ids", "binary_values", "valid_unit_mask"):
        source = payload[name]
        shape = source.shape[:-1] + (source.shape[-1] * 2,)
        storage = np.zeros(shape, dtype=source.dtype)
        view = storage[..., ::2]
        view[...] = source
        view.setflags(write=False)
        payload[name] = view
        assert not view.flags.writeable
    result = _tensorize(payload)
    assert result.entity_ids.tolist() == [4, 1, 4]
    assert result.binary_values.tolist() == [[1, 0], [0, 0], [0, 0]]
    assert result.local_field_ids.is_contiguous() and result.local_field_ids._base is None


@pytest.mark.parametrize("rows,width,item_count", [((4, 1), 0, 5), ((), 2, 5), ((), 0, 0), ((), 3, 0)])
def test_tensorize_empty_axes_validate_and_keep_shapes(rows, width, item_count):
    result = _tensorize(_packet(rows, width), expected_entity_ids=rows,
                        public_item_count=item_count, local_field_count=width)
    assert result.entity_ids.shape == (len(rows),)
    assert result.local_field_ids.shape == (len(rows), width)
    assert result.binary_values.shape == (len(rows), width)
    assert result.valid_unit_mask.shape == (len(rows), width)


@pytest.mark.parametrize("name,value,error", [
    ("expected_entity_ids", [4, 1, 4], TypeError),
    ("expected_entity_ids", TupleSubclass((4, 1, 4)), TypeError),
    ("expected_entity_ids", (True, 1, 4), TypeError),
    ("expected_entity_ids", (np.int64(4), 1, 4), TypeError),
    ("expected_entity_ids", (IntSubclass(4), 1, 4), TypeError),
    ("expected_entity_ids", (-1, 1, 4), ValueError),
    ("expected_entity_ids", (5, 1, 4), ValueError),
    ("expected_entity_ids", (MAX_INT + 1, 1, 4), ValueError),
    ("expected_entity_ids", (1, 4, 4), ValueError),
    ("public_item_count", True, TypeError),
    ("public_item_count", np.int64(5), TypeError),
    ("public_item_count", IntSubclass(5), TypeError),
    ("public_item_count", -1, ValueError),
    ("public_item_count", MAX_INT + 1, ValueError),
    ("public_item_count", 0, ValueError),
    ("local_field_count", False, TypeError),
    ("local_field_count", np.int64(2), TypeError),
    ("local_field_count", -1, ValueError),
    ("local_field_count", MAX_INT + 1, ValueError),
])
def test_tensorize_rejects_exact_scalar_and_request_violations(name, value, error):
    with pytest.raises(error):
        _tensorize(**{name: value})


@pytest.mark.parametrize("variant,error", [
    ("mapping_proxy", TypeError), ("dict_subclass", TypeError), ("extra_metadata", ValueError),
    ("missing_key", ValueError), ("key_subclass", TypeError), ("wrong_kind_type", TypeError), ("wrong_kind", ValueError),
    ("ids_rank", ValueError), ("shape", ValueError), ("ids_mismatch", ValueError),
    ("valid_wrong_id", ValueError), ("valid_nonbinary", ValueError),
    ("invalid_nonzero", ValueError), ("invalid_legal_id", ValueError),
    ("negative_active_id", ValueError), ("nonarray", TypeError),
])
def test_tensorize_strict_whitelist_shapes_and_dense_column_placeholders(variant, error):
    payload = _packet()
    if variant == "mapping_proxy":
        payload = MappingProxyType(payload)
    elif variant == "dict_subclass":
        payload = DictSubclass(payload)
    elif variant == "extra_metadata":
        payload["metadata"] = ForbiddenMetadata()
    elif variant == "missing_key":
        del payload["valid_unit_mask"]
    elif variant == "key_subclass":
        payload[StrSubclass("entity_ids")] = payload.pop("entity_ids")
    elif variant == "wrong_kind_type":
        payload["entity_kind"] = StrSubclass("item")
    elif variant == "wrong_kind":
        payload["entity_kind"] = "user"
    elif variant == "ids_rank":
        payload["entity_ids"] = payload["entity_ids"].reshape(3, 1)
    elif variant == "shape":
        payload["binary_values"] = payload["binary_values"][:, :1]
    elif variant == "ids_mismatch":
        payload["entity_ids"][0] = 1
    elif variant == "valid_wrong_id":
        payload["local_field_ids"][0, 0] = 1
    elif variant == "valid_nonbinary":
        payload["binary_values"][0, 0] = 2
    elif variant == "invalid_nonzero":
        payload["binary_values"][-1, -1] = 1
    elif variant == "invalid_legal_id":
        payload["local_field_ids"][-1, -1] = 1
    elif variant == "negative_active_id":
        payload["local_field_ids"][0, 0] = -1
    elif variant == "nonarray":
        payload["binary_values"] = [[1, 0], [0, 0], [0, 0]]
    with pytest.raises(error):
        _tensorize(payload)


@pytest.mark.parametrize("name", ["entity_ids", "local_field_ids", "binary_values", "valid_unit_mask"])
@pytest.mark.parametrize("variant", ["subclass", "metadata", "wrong_dtype", "object"])
def test_tensorize_rejects_every_nonplain_or_noncanonical_array(name, variant):
    payload = _packet()
    source = payload[name]
    if variant == "subclass":
        payload[name] = source.view(ArraySubclass)
    elif variant == "metadata":
        payload[name] = source.astype(np.dtype(source.dtype, metadata={"untrusted": "GT"}))
    elif variant == "wrong_dtype":
        payload[name] = source.astype(np.int8 if name == "valid_unit_mask" else np.float64)
    else:
        payload[name] = source.astype(object)
    with pytest.raises(TypeError):
        _tensorize(payload)


@pytest.mark.parametrize("name", ["entity_ids", "local_field_ids", "binary_values"])
def test_tensorize_rejects_nonnative_endian(name):
    payload = _packet()
    payload[name] = payload[name].astype(np.dtype(">i8" if sys.byteorder == "little" else "<i8"))
    with pytest.raises(TypeError):
        _tensorize(payload)


@pytest.mark.parametrize("name", ["local_field_ids", "binary_values", "valid_unit_mask"])
def test_empty_tensorize_still_rejects_wrong_dtype(name):
    payload = _packet((), 0)
    payload[name] = payload[name].astype(np.float32)
    with pytest.raises(TypeError):
        _tensorize(payload, expected_entity_ids=(), public_item_count=0, local_field_count=0)


@pytest.mark.parametrize("receiver,pools,online,expected_sources,expected_rows", [
    ({}, {}, (), (), ()),
    ({7: ()}, {7: ()}, (7,), (), ()),
    ({7: (4, 1, 4)}, {7: ()}, (7,), (), ()),
    ({7: ()}, {7: (41,)}, (7, 41), (41,), ((),)),
    ({7: (1,)}, {7: (41,)}, (7, 41), (41,), ((1,),)),
])
def test_plan_empty_domains_and_explicit_pool_no_implicit_own(receiver, pools, online, expected_sources, expected_rows):
    plan = _plan(receiver_item_ids=receiver, source_party_ids_by_receiver=pools,
                 online_party_ids=online)
    assert plan.source_party_ids == expected_sources
    assert plan.source_item_ids == expected_rows
    if receiver and next(iter(pools.values())) == (41,):
        assert plan.source_party_ids_by_receiver == ((41,),)


@pytest.mark.parametrize("variant,error", [
    ("receiver_mapping", TypeError), ("pool_mapping", TypeError), ("receiver_bool", TypeError),
    ("receiver_npint", TypeError), ("receiver_negative", ValueError),
    ("receiver_unknown", ValueError), ("keysets", ValueError),
    ("rows_list", TypeError), ("rows_bool", TypeError), ("rows_negative", ValueError),
    ("rows_outside", ValueError), ("pool_list", TypeError), ("pool_unsorted", ValueError),
    ("pool_duplicate", ValueError), ("pool_unknown", ValueError), ("pool_bool", TypeError),
    ("online_list", TypeError), ("online_unsorted", ValueError), ("online_duplicate", ValueError),
    ("online_npint", TypeError), ("online_negative", ValueError),
])
def test_prepare_rejects_container_id_and_roster_violations(variant, error):
    receiver, pools, online = {7: (4, 1, 4), 90: (1, 3)}, {7: (7, 41), 90: (41, 90)}, (7, 41, 90)
    if variant == "receiver_mapping": receiver = MappingProxyType(receiver)
    elif variant == "pool_mapping": pools = DictSubclass(pools)
    elif variant == "receiver_bool": receiver, pools = {True: ()}, {True: ()}
    elif variant == "receiver_npint": receiver, pools = {np.int64(7): ()}, {np.int64(7): ()}
    elif variant == "receiver_negative": receiver, pools = {-1: ()}, {-1: ()}
    elif variant == "receiver_unknown": receiver, pools = {3: ()}, {3: ()}
    elif variant == "keysets": del pools[90]
    elif variant == "rows_list": receiver[7] = [4, 1, 4]
    elif variant == "rows_bool": receiver[7] = (True,)
    elif variant == "rows_negative": receiver[7] = (-1,)
    elif variant == "rows_outside": receiver[7] = (5,)
    elif variant == "pool_list": pools[7] = [7, 41]
    elif variant == "pool_unsorted": pools[7] = (41, 7)
    elif variant == "pool_duplicate": pools[7] = (7, 7)
    elif variant == "pool_unknown": pools[7] = (7, 42)
    elif variant == "pool_bool": pools[7] = (True,)
    elif variant == "online_list": online = [7, 41, 90]
    elif variant == "online_unsorted": online = (90, 7, 41)
    elif variant == "online_duplicate": online = (7, 7, 41, 90)
    elif variant == "online_npint": online = (np.int64(7), 41, 90)
    elif variant == "online_negative": online = (-1, 7, 41, 90)
    with pytest.raises(error):
        _plan(receiver_item_ids=receiver, source_party_ids_by_receiver=pools,
              online_party_ids=online)


@pytest.mark.parametrize("value,error", [(None, TypeError), (StrSubclass(CONTEXT), TypeError),
    ("A" * 64, ValueError), ("a" * 63, ValueError), ("g" * 64, ValueError),
    (" " + CONTEXT, ValueError)])
@pytest.mark.parametrize("where", ["prepare", "pack"])
def test_context_exact_format_is_checked_even_empty_domains(value, error, where):
    if where == "prepare":
        with pytest.raises(error):
            _plan(receiver_item_ids={}, source_party_ids_by_receiver={},
                  online_party_ids=(), context_id=value)
    else:
        empty = _plan(receiver_item_ids={}, source_party_ids_by_receiver={}, online_party_ids=())
        with pytest.raises(error):
            _pack(empty, {}, expected_context_id=value)


def test_expected_context_mismatch_is_rejected_without_claiming_message_authentication():
    with pytest.raises(ValueError):
        _pack(expected_context_id="b" * 64)
    # The caller tag has no mathematical-origin proof: a different finite H is
    # accepted under the same tag and yields its different declared values.
    publications = _publications()
    publications[7][1][0, 0, 0] = -8
    assert _pack(publications=publications)[0].compatible_tokens[1, 0, 0].item() == -8


@pytest.mark.parametrize("name,value,error", [
    ("slot_count", True, TypeError), ("slot_count", np.int64(2), TypeError),
    ("slot_count", IntSubclass(2), TypeError), ("slot_count", 0, ValueError),
    ("slot_count", -1, ValueError), ("slot_count", MAX_INT + 1, ValueError),
    ("compatible_dim", False, TypeError), ("compatible_dim", np.int64(3), TypeError),
    ("compatible_dim", 0, ValueError), ("compatible_dim", MAX_INT + 1, ValueError),
    ("dtype", torch.float16, TypeError), ("dtype", torch.int64, TypeError),
    ("dtype", np.float64, TypeError), ("dtype", "float64", TypeError),
])
def test_pack_scalar_schema_is_checked_even_empty_domains(name, value, error):
    plan = _plan(receiver_item_ids={}, source_party_ids_by_receiver={}, online_party_ids=())
    with pytest.raises(error):
        _pack(plan, {}, **{name: value})


@pytest.mark.parametrize("variant,error", [
    ("mapping", TypeError), ("missing_source", ValueError), ("extra_source", ValueError),
    ("bool_source", TypeError), ("np_source", TypeError), ("list_record", TypeError),
    ("short_record", ValueError), ("ids_list", TypeError), ("swapped_rows", ValueError),
    ("ids_bool", TypeError), ("tokens_array", TypeError), ("tokens_dtype", TypeError),
    ("tokens_rank", ValueError), ("tokens_shape", ValueError), ("tokens_meta", TypeError),
    ("tokens_sparse", TypeError), ("tokens_nested", TypeError), ("tokens_nan_masked", ValueError),
    ("tokens_inf_unreferenced", ValueError), ("available_array", TypeError),
    ("available_dtype", TypeError), ("available_rank", ValueError),
    ("available_shape", ValueError), ("available_meta", TypeError),
    ("none_with_true", ValueError),
])
def test_pack_rejects_publication_schema_rows_dtype_layout_and_all_cell_nonfinite(variant, error):
    publications = _publications()
    ids, tokens, available = publications[90]
    if variant == "mapping": publications = MappingProxyType(publications)
    elif variant == "missing_source": del publications[90]
    elif variant == "extra_source": publications[91] = publications[90]
    elif variant == "bool_source": publications[True] = publications.pop(90)
    elif variant == "np_source": publications[np.int64(90)] = publications.pop(90)
    elif variant == "list_record": publications[90] = list(publications[90])
    elif variant == "short_record": publications[90] = (ids, tokens)
    elif variant == "ids_list": publications[90] = (list(ids), tokens, available)
    elif variant == "swapped_rows": publications[90] = ((3, 1), tokens, available)
    elif variant == "ids_bool": publications[90] = ((True, 3), tokens, available)
    elif variant == "tokens_array": publications[90] = (ids, tokens.numpy(), available)
    elif variant == "tokens_dtype": publications[90] = (ids, tokens.float(), available)
    elif variant == "tokens_rank": publications[90] = (ids, tokens[:, 0], available)
    elif variant == "tokens_shape": publications[90] = (ids, tokens[:, :1], available)
    elif variant == "tokens_meta": publications[90] = (ids, tokens.to("meta"), available)
    elif variant == "tokens_sparse": publications[90] = (ids, tokens.to_sparse(), available)
    elif variant == "tokens_nested":
        publications[90] = (ids, torch.nested.nested_tensor([tokens[0], tokens[1]]), available)
    elif variant == "tokens_nan_masked": tokens[0, 0, 0] = float("nan")
    elif variant == "tokens_inf_unreferenced": tokens[-1, -1, -1] = float("inf")
    elif variant == "available_array": publications[90] = (ids, tokens, available.numpy())
    elif variant == "available_dtype": publications[90] = (ids, tokens, available.to(torch.int64))
    elif variant == "available_rank": publications[90] = (ids, tokens, available[:, None])
    elif variant == "available_shape": publications[90] = (ids, tokens, available[:1])
    elif variant == "available_meta": publications[90] = (ids, tokens, available.to("meta"))
    elif variant == "none_with_true": publications[90] = (ids, None, available)
    with pytest.raises(error):
        _pack(publications=publications)


@pytest.mark.parametrize("name,value", [
    ("source_party_ids", (7, 90)),
    ("source_party_ids", (7, 41, 41, 90)),
    ("online_party_ids", (41, 7, 90)),
    ("receiver_party_ids", (90, 7)),
    ("receiver_item_ids", ((4, 2, 4), (1, 3))),
    ("source_item_ids", ((1, 4), (1, 2, 4), (1, 3))),
    ("source_item_ids", ((1, 4), (1, 3, 4), (3, 1))),
    ("source_party_ids_by_receiver", ((7,), (41, 90))),
    ("receiver_source_row_indices", (((0, 0, 1), (2, 0, 2)), ((0, 1), (0, 1)))),
    ("receiver_source_row_indices", (((1, 0, 1),), ((0, 1), (0, 1)))),
    ("receiver_source_row_indices", (((1, 0, 7), (2, 0, 2)), ((0, 1), (0, 1)))),
    ("public_item_count", 3),
])
def test_pack_revalidates_reflected_plan_content_including_in_range_wrong_inverse(name, value):
    plan = _plan()
    object.__setattr__(plan, name, value)
    with pytest.raises(ValueError):
        _pack(plan)


@pytest.mark.parametrize("name,value", [("receiver_item_ids", [(4, 1, 4), (1, 3)]),
    ("source_party_ids", (7, np.int64(41), 90)), ("public_item_count", True),
    ("receiver_source_row_indices", (((True, 0, 1), (2, 0, 2)), ((0, 1), (0, 1))))])
def test_pack_revalidates_reflected_plan_member_exact_types(name, value):
    plan = _plan()
    object.__setattr__(plan, name, value)
    with pytest.raises(TypeError):
        _pack(plan)


@pytest.mark.parametrize("bad_plan", [None, object(), {}, TupleSubclass(())])
def test_pack_rejects_nonrecord_plan(bad_plan):
    with pytest.raises(TypeError):
        batch.pack_doi4_source_slots(bad_plan, source_publications={}, slot_count=2,
                                    compatible_dim=3, dtype=torch.float64,
                                    expected_context_id=CONTEXT)


def _forbid_output_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("receiver output allocated before full validation")
    for name in ("tensor", "empty", "zeros", "full", "cat", "stack", "index_select"):
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "index_select", forbidden)


@pytest.mark.parametrize("variant", ["late_nan", "late_wrong_ids", "late_shape", "wrong_inverse", "context"])
def test_all_publications_and_plan_are_validated_before_any_receiver_allocation(monkeypatch, variant):
    plan, publications = _plan(), _publications()
    expected_context = CONTEXT
    ids, tokens, available = publications[90]
    if variant == "late_nan": tokens[-1, -1, -1] = float("nan")
    elif variant == "late_wrong_ids": publications[90] = ((3, 1), tokens, available)
    elif variant == "late_shape": publications[90] = (ids, tokens[:, :1], available)
    elif variant == "wrong_inverse": object.__setattr__(plan, "receiver_source_row_indices", (((0, 0, 1), (2, 0, 2)), ((0, 1), (0, 1))))
    else: expected_context = "b" * 64
    _forbid_output_allocation(monkeypatch)
    with pytest.raises(ValueError):
        _pack(plan, publications, expected_context_id=expected_context)


def test_tensorize_validates_whole_packet_before_copy_allocation(monkeypatch):
    payload = _packet()
    payload["binary_values"][-1, -1] = 7
    _forbid_output_allocation(monkeypatch)
    with pytest.raises(ValueError):
        _tensorize(payload)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("empty", [False, True])
def test_none_absence_vs_present_allfalse_keep_distinct_connected_graphs(dtype, empty):
    rows = () if empty else (1, 1)
    plan = _plan(receiver_item_ids={7: rows}, source_party_ids_by_receiver={7: (41,)},
                 online_party_ids=(7, 41))
    unique = () if empty else (1,)
    shape = (len(unique), 2, 3)
    availability = torch.zeros(len(unique), dtype=torch.bool)
    absent = _pack(plan, {41: (unique, None, availability)}, dtype=dtype)[0]
    assert absent.compatible_tokens.shape == (len(rows), 2, 3)
    assert not absent.compatible_tokens.requires_grad
    assert absent.compatible_tokens.grad_fn is None
    assert not bool(absent.valid_token_mask.any())
    huge = torch.finfo(dtype).max / 2
    present_H = torch.full(shape, huge, dtype=dtype, requires_grad=True)
    present = _pack(plan, {41: (unique, present_H, availability)}, dtype=dtype)[0]
    assert present.compatible_tokens.requires_grad
    assert torch.equal(present.compatible_tokens, torch.zeros_like(present.compatible_tokens))
    assert bool(torch.isfinite(present.compatible_tokens).all())
    present.compatible_tokens.sum().backward()
    assert present_H.grad is not None
    assert torch.equal(present_H.grad, torch.zeros_like(present_H))
    assert bool(torch.isfinite(present_H.grad).all())


def test_none_source_has_no_dependency_on_other_present_sources():
    plan = _plan(receiver_item_ids={7: (1,)}, source_party_ids_by_receiver={7: (7, 41)},
                 online_party_ids=(7, 41))
    tokens = torch.ones((1, 2, 3), dtype=torch.float64, requires_grad=True)
    output = _pack(plan, {7: ((1,), None, torch.tensor([False])),
                          41: ((1,), tokens, torch.tensor([True]))})[0]
    output.compatible_tokens[:, :2].sum().backward()
    assert tokens.grad is not None and torch.equal(tokens.grad, torch.zeros_like(tokens))
    assert output.valid_token_mask.tolist() == [[False, False, True, True]]


@pytest.mark.parametrize("rows,pool", [((1, 3), ()), ((), ()), ((), (41,))])
def test_pack_empty_receiver_rows_or_source_pool(rows, pool):
    plan = _plan(receiver_item_ids={7: rows}, source_party_ids_by_receiver={7: pool},
                 online_party_ids=(7, 41))
    publications = {} if not pool else {41: ((), None, torch.zeros(0, dtype=torch.bool))}
    output, = _pack(plan, publications)
    assert output.compatible_tokens.shape == (len(rows), len(pool) * 2, 3)
    assert output.valid_token_mask.shape == (len(rows), len(pool) * 2)
    assert output.entity_ids == rows
    assert output.slot_source_party_ids == tuple(source for source in pool for _ in range(2))
    assert output.slot_indices == (0, 1) * len(pool)


def test_zero_public_domain_allows_only_empty_rows_and_empty_receiver_domain_returns_tuple():
    plan = _plan(receiver_item_ids={7: ()}, source_party_ids_by_receiver={7: (7,)},
                 online_party_ids=(7,), public_item_count=0)
    output, = _pack(plan, {7: ((), None, torch.zeros(0, dtype=torch.bool))})
    assert output.entity_ids == () and output.compatible_tokens.shape == (0, 2, 3)
    with pytest.raises(ValueError):
        _plan(public_item_count=0)
    empty = _plan(receiver_item_ids={}, source_party_ids_by_receiver={}, online_party_ids=(), public_item_count=0)
    assert _pack(empty, {}) == ()


@pytest.mark.parametrize("variant", ["parameter", "nonleaf_strided"])
def test_pack_keeps_parameter_and_nonleaf_strided_source_paths(variant):
    plan = _plan(receiver_item_ids={7: (4, 1, 4)}, source_party_ids_by_receiver={7: (41,)},
                 online_party_ids=(7, 41))
    leaf = torch.tensor([[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]],
                        dtype=torch.float64, requires_grad=True)
    if variant == "parameter":
        tokens = torch.nn.Parameter(leaf.detach().reshape(2, 2, 3).clone())
        target = tokens
        expected = [[[1., 1., 1.], [1., 1., 1.]], [[2., 2., 2.], [2., 2., 2.]]]
    else:
        tokens = (leaf * 3).reshape(2, 3, 2).transpose(1, 2)
        assert not tokens.is_contiguous() and not tokens.is_leaf
        target = leaf
        expected = [[3.] * 6, [6.] * 6]
    available_storage = torch.tensor([True, False, True, False])
    available = available_storage[::2]
    assert not available.is_contiguous()
    output, = _pack(plan, {41: ((1, 4), tokens, available)})
    output.compatible_tokens.sum().backward()
    _close(target.grad, expected)
    assert output.valid_token_mask.tolist() == [[True, True]] * 3


def test_valid_zero_norm_is_retained_and_mask_not_inferred_from_values():
    plan = _plan(receiver_item_ids={7: (1,)}, source_party_ids_by_receiver={7: (41,)},
                 online_party_ids=(7, 41))
    tokens = torch.zeros((1, 2, 3), dtype=torch.float64, requires_grad=True)
    output, = _pack(plan, {41: ((1,), tokens, torch.tensor([True]))})
    assert output.valid_token_mask.tolist() == [[True, True]]
    output.compatible_tokens.sum().backward()
    assert torch.equal(tokens.grad, torch.ones_like(tokens))


def test_output_storages_are_owning_independent_across_receivers_calls_and_sources():
    publications = _publications()
    first, second = _pack(publications=publications), _pack(publications=publications)
    tensors = [getattr(output, name) for outputs in (first, second) for output in outputs
               for name in ("compatible_tokens", "valid_token_mask")]
    pointers = [value.untyped_storage().data_ptr() for value in tensors]
    input_pointers = {value.untyped_storage().data_ptr()
                      for _, tokens, available in publications.values()
                      for value in (tokens, available)}
    assert len(set(pointers)) == len(pointers) and not set(pointers) & input_pointers
    first[0].compatible_tokens.fill_(-99)
    first[0].valid_token_mask.fill_(False)
    first[0].compatible_tokens.resize_(1)
    _close(first[1].compatible_tokens, FIXTURE["expected_tokens"][1])
    _close(second[0].compatible_tokens, FIXTURE["expected_tokens"][0])
    for source in (7, 41, 90):
        _close(publications[source][1], FIXTURE["source_H"][str(source)])
        assert publications[source][2].tolist() == FIXTURE["source_available"][str(source)]


def test_all_three_functions_preserve_rng_inputs_versions_grads_and_prior_outputs():
    payload, publications = _packet(), _publications(requires_grad=True)
    for _, tokens, _ in publications.values():
        tokens.grad = torch.full_like(tokens, -0.25)
    saved_arrays = {name: value.copy() for name, value in payload.items() if isinstance(value, np.ndarray)}
    saved_tensors = {source: (tokens.detach().clone(), available.clone(), tokens._version,
                              available._version, tokens.grad.clone())
                     for source, (_, tokens, available) in publications.items()}
    prior = _pack(publications=publications)
    prior_copies = [(value.compatible_tokens.detach().clone(), value.valid_token_mask.clone()) for value in prior]
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    _tensorize(payload)
    plan = _plan()
    _pack(plan, publications)
    assert random.getstate() == python_rng
    new_numpy = np.random.get_state()
    assert new_numpy[0] == numpy_rng[0] and new_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(new_numpy[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    for name, expected in saved_arrays.items():
        np.testing.assert_array_equal(payload[name], expected)
    for source, (_, tokens, available) in publications.items():
        values, mask, version, mask_version, gradient = saved_tensors[source]
        assert torch.equal(tokens.detach(), values) and torch.equal(available, mask)
        assert tokens._version == version and available._version == mask_version
        assert torch.equal(tokens.grad, gradient)
    for output, (tokens, mask) in zip(prior, prior_copies):
        assert torch.equal(output.compatible_tokens.detach(), tokens)
        assert torch.equal(output.valid_token_mask, mask)


@pytest.mark.parametrize("function", ["tensorize", "pack_source", "pack_receiver"])
def test_int64_capacity_checks_include_zero_axis_stride_products_without_huge_allocation(function, monkeypatch):
    if function == "tensorize":
        payload = _packet((), 0)
        _forbid_output_allocation(monkeypatch)
        # Shape mismatch is a valid earlier error, but must precede any output.
        with pytest.raises(ValueError):
            _tensorize(payload, expected_entity_ids=(), local_field_count=MAX_INT)
    elif function == "pack_source":
        plan = _plan(receiver_item_ids={7: ()}, source_party_ids_by_receiver={7: (41,)},
                     online_party_ids=(7, 41))
        publications = {41: ((), None, torch.zeros(0, dtype=torch.bool))}
        _forbid_output_allocation(monkeypatch)
        with pytest.raises(ValueError):
            _pack(plan, publications, slot_count=MAX_INT, compatible_dim=2)
    else:
        plan = _plan(receiver_item_ids={7: ()}, source_party_ids_by_receiver={7: (7, 41)},
                     online_party_ids=(7, 41))
        publications = {source: ((), None, torch.zeros(0, dtype=torch.bool)) for source in (7, 41)}
        _forbid_output_allocation(monkeypatch)
        with pytest.raises(ValueError):
            _pack(plan, publications, slot_count=MAX_INT, compatible_dim=1)


@pytest.mark.parametrize("record_name", ["BinaryTensorBatch", "Doi4BatchRequests", "ReceiverSlotBatch"])
def test_results_are_frozen_slotted_keyword_only_and_factory_only(record_name):
    cls = getattr(batch, record_name)
    assert is_dataclass(cls) and hasattr(cls, "__slots__")
    assert cls.__dataclass_params__.frozen
    assert all(field.kw_only for field in fields(cls))
    with pytest.raises(TypeError):
        cls()
    with pytest.raises(TypeError):
        cls(**{field.name: None for field in fields(cls)})
    with pytest.raises(TypeError):
        cls(*([None] * len(fields(cls))))
    instance = {"BinaryTensorBatch": lambda: _tensorize(), "Doi4BatchRequests": _plan,
                "ReceiverSlotBatch": lambda: _pack()[0]}[record_name]()
    assert not hasattr(instance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(instance, fields(cls)[0].name, None)


def test_exact_frozen_signatures_record_fields_and_production_dependency_direction():
    expected = {
        "tensorize_binary_delivery": ("payload", "expected_entity_ids", "public_item_count", "local_field_count"),
        "prepare_doi4_batch_requests": ("receiver_item_ids", "source_party_ids_by_receiver", "online_party_ids", "public_item_count", "context_id"),
        "pack_doi4_source_slots": ("requests", "source_publications", "slot_count", "compatible_dim", "dtype", "expected_context_id"),
    }
    for name, names in expected.items():
        signature = inspect.signature(getattr(batch, name))
        assert tuple(signature.parameters) == names
        for index, parameter in enumerate(signature.parameters.values()):
            assert parameter.default is inspect.Parameter.empty
            assert parameter.kind == (inspect.Parameter.POSITIONAL_OR_KEYWORD if index == 0 else inspect.Parameter.KEYWORD_ONLY)
    expected_fields = {
        "BinaryTensorBatch": ("entity_ids", "local_field_ids", "binary_values", "valid_unit_mask"),
        "Doi4BatchRequests": ("context_id", "public_item_count", "online_party_ids", "receiver_party_ids", "receiver_item_ids", "source_party_ids_by_receiver", "source_party_ids", "source_item_ids", "receiver_source_row_indices"),
        "ReceiverSlotBatch": ("receiver_party_id", "entity_ids", "slot_source_party_ids", "slot_indices", "compatible_tokens", "valid_token_mask"),
    }
    for name, names in expected_fields.items():
        assert tuple(field.name for field in fields(getattr(batch, name))) == names
    source = Path(batch.__file__).read_text()
    tree = ast.parse(source)
    public_functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")}
    assert public_functions == set(expected)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            modules = [node.module or ""]
        else:
            continue
        assert all(module.split(".")[0] in sys.stdlib_module_names | {"numpy", "torch"} for module in modules)
        assert not any(module.startswith("src") for module in modules)
    assert "torch.from_numpy(" not in source and "torch.as_tensor(" not in source
    assert ".backward(" not in source and "autograd.grad(" not in source


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_real_adapter_tensorize_readout_pack_fuse_loss_has_independent_scalar_gradient_expectation(dtype, record_property):
    from src.data.contracts import FeatureView
    from src.data.local_field_maps import prepare_local_field_map
    from src.data.model_inputs import (close_binary_input_adapter, deliver_binary_field_units,
                                       prepare_binary_input_adapter)
    from src.methods.doi4.composition import (encode_readout_binary_units,
                                             fuse_score_group_loss_pri_none)
    case = FIXTURE["composition_case"]
    view = FeatureView(name="tiny", entity_kind="item", entity_ids=np.array([0, 1], dtype=np.int64),
                       modality="tabular", values=np.array([[0], [1]], dtype=np.int64),
                       feature_names=("field",), missing_mask=None,
                       metadata={"unconsumed": "GT"})
    local_map = prepare_local_field_map(party_id=7, schema_epoch=0,
                                      field_assignments=(("tiny", "field"),))
    adapter = prepare_binary_input_adapter({"tiny": view}, party_id=7, public_item_count=2,
                                          runtime_binding_id=CONTEXT, local_map=local_map,
                                          authorized_rows_by_view={"tiny": None})
    try:
        payload = deliver_binary_field_units(adapter, entity_ids=(0, 1), expected_binding_id=adapter.binding_id)
    finally:
        close_binary_input_adapter(adapter)
    delivered = batch.tensorize_binary_delivery(payload, expected_entity_ids=(0, 1), public_item_count=2, local_field_count=1)
    field = torch.tensor([[0.2]], dtype=dtype, requires_grad=True)
    values = torch.tensor([[0.3], [0.5]], dtype=dtype, requires_grad=True)
    queries = torch.tensor([[0.5], [-0.25]], dtype=dtype, requires_grad=True)
    key = torch.tensor([[1.0]], dtype=dtype, requires_grad=True)
    value = torch.tensor([[2.0]], dtype=dtype, requires_grad=True)
    H, available, _ = encode_readout_binary_units(delivered.local_field_ids, delivered.binary_values,
                                                delivered.valid_unit_mask, field_embeddings=field,
                                                value_embeddings=values, queries=queries,
                                                w_key=key, w_value=value)
    H.retain_grad()
    h_error = _close(H, case["expected_source_H"], dtype)
    plan = batch.prepare_doi4_batch_requests({90: (1, 0, 1)}, source_party_ids_by_receiver={90: (7,)},
                                            online_party_ids=(7, 90), public_item_count=2, context_id=CONTEXT)
    output, = batch.pack_doi4_source_slots(plan, source_publications={7: ((0, 1), H, available)},
                                           slot_count=2, compatible_dim=1, dtype=dtype, expected_context_id=CONTEXT)
    loss, logits, fused, fused_available, _ = fuse_score_group_loss_pri_none(
        output.compatible_tokens, output.valid_token_mask,
        torch.tensor([1., 0., 1.], dtype=dtype), torch.tensor([0, 2, 3], dtype=torch.int64),
        user_embeddings=torch.ones((3, 1), dtype=dtype),
        item_embeddings=torch.tensor([[0.1], [0.2], [0.3]], dtype=dtype),
        w_query=torch.zeros((1, 1), dtype=dtype), w_key=torch.zeros((1, 1), dtype=dtype),
        w_value=torch.ones((1, 1), dtype=dtype), w_output=torch.ones((1, 1), dtype=dtype),
        w_fused=torch.tensor([[0.5]], dtype=dtype), score_weight=torch.tensor([2.], dtype=dtype),
        score_bias=torch.tensor(0.025, dtype=dtype))
    expected_logits = np.array(case["expected_logits"], dtype=np.float64)
    weights = np.array(case["loss_row_weights"], dtype=np.float64)
    labels = np.array(case["labels"], dtype=np.float64)
    expected_loss = float(np.sum(weights * (np.logaddexp(0, expected_logits) - labels * expected_logits)))
    gradients = weights * (1 / (1 + np.exp(-expected_logits)) - labels)
    expected_H_gradient = [[[0.5 * gradients[1]], [0.5 * gradients[1]]],
                           [[0.5 * (gradients[0] + gradients[2])], [0.5 * (gradients[0] + gradients[2])]]]
    forward_error = max(h_error, _close(logits, expected_logits, dtype), _close(loss, expected_loss, dtype),
                        _close(fused, [[1.4], [1.0], [1.4]], dtype))
    assert fused_available.tolist() == [True, True, True]
    loss.backward()
    gradient_error = max(_close(H.grad, expected_H_gradient, dtype),
                         _close(field.grad, [[2 * gradients.sum()]], dtype),
                         _close(values.grad, [[2 * gradients[1]], [2 * (gradients[0] + gradients[2])]], dtype),
                         _close(value.grad, [[0.5 * gradients[1] + 0.7 * (gradients[0] + gradients[2])]], dtype))
    assert key.grad is not None and torch.equal(key.grad, torch.zeros_like(key))
    assert queries.grad is not None and torch.equal(queries.grad, torch.zeros_like(queries))
    record_property("adapter_math_forward_max_abs_error", forward_error)
    record_property("adapter_math_owner_gradient_max_abs_error", gradient_error)


@pytest.mark.parametrize("name,value,error", [
    ("public_item_count", True, TypeError), ("public_item_count", np.int64(5), TypeError),
    ("public_item_count", -1, ValueError), ("public_item_count", MAX_INT + 1, ValueError),
])
def test_prepare_validates_public_count_even_empty_receiver_domain(name, value, error):
    with pytest.raises(error):
        _plan(receiver_item_ids={}, source_party_ids_by_receiver={},
              online_party_ids=(), **{name: value})


@pytest.mark.parametrize("name", ["entity_ids", "local_field_ids", "binary_values", "valid_unit_mask"])
def test_tensorize_rejects_structured_array_dtypes_without_reading_fields(name):
    payload = _packet()
    payload[name] = np.zeros(payload[name].shape, dtype=[("raw_GT", np.int64)])
    with pytest.raises(TypeError):
        _tensorize(payload)


def test_empty_receiver_keeps_present_nonempty_source_graph_with_huge_finite_values():
    plan = _plan(receiver_item_ids={7: (), 90: (1,)},
                 source_party_ids_by_receiver={7: (41,), 90: (41,)},
                 online_party_ids=(7, 41, 90))
    tokens = torch.full((1, 2, 3), torch.finfo(torch.float64).max / 2,
                        dtype=torch.float64, requires_grad=True)
    empty_output, nonempty_output = _pack(plan, {41: ((1,), tokens, torch.tensor([False]))})
    assert empty_output.compatible_tokens.shape == (0, 2, 3)
    assert empty_output.compatible_tokens.requires_grad
    assert nonempty_output.compatible_tokens.shape == (1, 2, 3)
    assert torch.equal(nonempty_output.compatible_tokens, torch.zeros_like(nonempty_output.compatible_tokens))
    empty_output.compatible_tokens.sum().backward()
    assert tokens.grad is not None and torch.equal(tokens.grad, torch.zeros_like(tokens))
    assert bool(torch.isfinite(tokens.grad).all())


@pytest.mark.parametrize("rows", [(), (4, 1, 4)])
def test_true_zero_source_axis_does_not_invent_a_slot_count_dimension(rows):
    plan = _plan(receiver_item_ids={7: rows}, source_party_ids_by_receiver={7: ()},
                 online_party_ids=(7,))
    output, = _pack(plan, {}, slot_count=MAX_INT, compatible_dim=2)
    assert output.compatible_tokens.shape == (len(rows), 0, 2)
    assert output.valid_token_mask.shape == (len(rows), 0)
    assert output.slot_source_party_ids == output.slot_indices == ()
