"""Independent integer/tensor packing checks; no model, training or data I/O."""

from dataclasses import FrozenInstanceError, fields, replace
import random

import numpy as np
import pytest
import torch

import src.federation.trace_batch as trace_batch_module
from src.federation.round_drain import PartyGroupBatch
from src.federation.trace_batch import (
    PackedPartyGroupBatch,
    pack_party_group_batch,
)


LIMIT = 2**63 - 1
PAYLOAD_FIELDS = ("user_ids", "item_ids", "labels", "group_offsets")


class IntSubclass(int):
    pass


class TupleSubclass(tuple):
    pass


class BatchSubclass(PartyGroupBatch):
    pass


def _batch(**changes):
    arguments = dict(
        party_id=41,
        start=2,
        stop=4,
        positive_ids=(203, 101),
        user_ids=(3, 17),
        positive_item_ids=(2, 0),
        negative_item_ids=((8, 7), (7, 7)),
        loss_coefficient=(2, 7),
    )
    arguments.update(changes)
    return PartyGroupBatch(**arguments)


def _assert_payload(packed, *, users, items, labels, offsets, dtype):
    assert packed.user_ids.tolist() == users
    assert packed.item_ids.tolist() == items
    assert packed.labels.tolist() == labels
    assert packed.group_offsets.tolist() == offsets
    assert packed.labels.dtype is dtype
    assert all(getattr(packed, name).dtype is torch.int64 for name in (
        "user_ids", "item_ids", "group_offsets",
    ))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_group_major_order_target_first_and_unreduced_mass(dtype):
    batch = _batch()
    result = pack_party_group_batch(batch, dtype=dtype)
    assert type(result) is PackedPartyGroupBatch
    assert result.party_id == 41
    assert result.positive_ids == (203, 101)
    assert result.loss_coefficient == (2, 7)
    # Independent literal oracle: order is not positive-ID or item-ID sorting.
    # Duplicate negatives are preserved because this descriptor has no replace flag.
    _assert_payload(
        result,
        users=[3, 3, 3, 17, 17, 17],
        items=[2, 8, 7, 0, 7, 7],
        labels=[1, 0, 0, 1, 0, 0],
        offsets=[0, 3, 6],
        dtype=dtype,
    )
    assert batch == _batch()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_negatives_preserves_all_positive_groups(dtype):
    batch = _batch(negative_item_ids=((), ()))
    result = pack_party_group_batch(batch, dtype=dtype)
    _assert_payload(
        result, users=[3, 17], items=[2, 0], labels=[1, 1],
        offsets=[0, 1, 2], dtype=dtype,
    )
    assert result.loss_coefficient == (2, 7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tail_keeps_one_group_and_original_round_denominator(dtype):
    batch = _batch(
        start=4, stop=5, positive_ids=(277,), user_ids=(3,),
        positive_item_ids=(6,), negative_item_ids=((8,),),
        loss_coefficient=(1, 7),
    )
    result = pack_party_group_batch(batch, dtype=dtype)
    _assert_payload(
        result, users=[3, 3], items=[6, 8], labels=[1, 0],
        offsets=[0, 2], dtype=dtype,
    )
    assert result.positive_ids == (277,)
    assert result.loss_coefficient == (1, 7)


def test_repeated_users_positive_items_and_cross_group_negatives_are_legal():
    batch = _batch(user_ids=(3, 3), positive_item_ids=(2, 2))
    result = pack_party_group_batch(batch, dtype=torch.float32)
    _assert_payload(
        result, users=[3, 3, 3, 3, 3, 3], items=[2, 8, 7, 2, 7, 7],
        labels=[1, 0, 0, 1, 0, 0], offsets=[0, 3, 6], dtype=torch.float32,
    )


def test_full_int64_scalar_domain_does_not_confuse_ids_with_positions():
    batch = _batch(
        party_id=LIMIT, start=LIMIT - 2, stop=LIMIT,
        positive_ids=(LIMIT, 0), user_ids=(LIMIT, 0),
        positive_item_ids=(LIMIT, 1),
        negative_item_ids=((0, 0), (LIMIT, LIMIT)),
        loss_coefficient=(2, LIMIT),
    )
    result = pack_party_group_batch(batch, dtype=torch.float64)
    assert result.party_id == LIMIT
    assert result.positive_ids == (LIMIT, 0)
    assert result.loss_coefficient == (2, LIMIT)
    _assert_payload(
        result,
        users=[LIMIT, LIMIT, LIMIT, 0, 0, 0],
        items=[LIMIT, 0, 0, 1, LIMIT, LIMIT],
        labels=[1, 0, 0, 1, 0, 0], offsets=[0, 3, 6],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_owned_cpu_storage_mutable_outputs_and_cross_call_isolation(dtype):
    batch = _batch()
    first = pack_party_group_batch(batch, dtype=dtype)
    second = pack_party_group_batch(batch, dtype=dtype)
    tensors = [getattr(result, name) for result in (first, second)
               for name in PAYLOAD_FIELDS]
    assert len({tensor.untyped_storage().data_ptr() for tensor in tensors}) == 8
    for tensor in tensors:
        assert tensor.device.type == "cpu"
        assert tensor.layout is torch.strided
        assert tensor.is_contiguous()
        assert tensor._base is None
        assert tensor.storage_offset() == 0
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    first.user_ids.resize_(1).fill_(100)
    first.item_ids.fill_(99)
    first.labels.zero_()
    first.group_offsets.fill_(10)
    _assert_payload(
        second, users=[3, 3, 3, 17, 17, 17], items=[2, 8, 7, 0, 7, 7],
        labels=[1, 0, 0, 1, 0, 0], offsets=[0, 3, 6], dtype=dtype,
    )
    third = pack_party_group_batch(batch, dtype=dtype)
    for name in PAYLOAD_FIELDS:
        assert torch.equal(getattr(second, name), getattr(third, name))
    assert batch == _batch()


def test_exact_frozen_slotted_keyword_only_record_fields():
    result = pack_party_group_batch(_batch(), dtype=torch.float32)
    assert tuple(field.name for field in fields(result)) == (
        "party_id", "positive_ids", "loss_coefficient", "user_ids", "item_ids",
        "labels", "group_offsets",
    )
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.party_id = 7
    with pytest.raises(TypeError):
        PackedPartyGroupBatch(41, (203, 101), (2, 7), None, None, None, None)


@pytest.mark.parametrize("dtype", [
    None, "float32", 0, np.float32, torch.float16, torch.bfloat16,
    torch.int64, torch.bool, torch.complex64,
])
def test_invalid_dtype_is_rejected_before_batch_validation(dtype):
    with pytest.raises(TypeError, match="dtype"):
        pack_party_group_batch(None, dtype=dtype)


def test_dtype_does_not_invoke_arbitrary_equality():
    class NotADtype:
        def __eq__(self, other):
            raise AssertionError("unsupported dtype equality must not be invoked")

    with pytest.raises(TypeError, match="dtype"):
        pack_party_group_batch(_batch(), dtype=NotADtype())


@pytest.mark.parametrize("batch", [None, {}, (), object()])
def test_wrong_batch_type_is_rejected(batch):
    with pytest.raises(TypeError, match="exact PartyGroupBatch"):
        pack_party_group_batch(batch, dtype=torch.float32)


def test_batch_subclass_is_rejected():
    base = _batch()
    subclass = BatchSubclass(**{field.name: getattr(base, field.name)
                              for field in fields(base)})
    with pytest.raises(TypeError, match="exact PartyGroupBatch"):
        pack_party_group_batch(subclass, dtype=torch.float32)


@pytest.mark.parametrize("field", ["party_id", "start", "stop"])
@pytest.mark.parametrize("value", [True, np.int64(2), IntSubclass(2), 2.0, "2", None])
def test_scalar_types_are_not_coerced(field, value):
    with pytest.raises(TypeError, match=field):
        pack_party_group_batch(_batch(**{field: value}), dtype=torch.float64)


@pytest.mark.parametrize("field", ["party_id", "start", "stop"])
@pytest.mark.parametrize("value", [-1, LIMIT + 1])
def test_scalar_values_must_fit_nonnegative_int64(field, value):
    with pytest.raises(ValueError, match=field):
        pack_party_group_batch(_batch(**{field: value}), dtype=torch.float64)


@pytest.mark.parametrize("field", [
    "positive_ids", "user_ids", "positive_item_ids", "negative_item_ids",
    "loss_coefficient",
])
@pytest.mark.parametrize("container", [list, TupleSubclass])
def test_outer_containers_must_be_exact_tuples(field, container):
    batch = _batch()
    bad = replace(batch, **{field: container(getattr(batch, field))})
    with pytest.raises(TypeError, match=field):
        pack_party_group_batch(bad, dtype=torch.float32)


@pytest.mark.parametrize("field", [
    "positive_ids", "user_ids", "positive_item_ids", "loss_coefficient",
])
@pytest.mark.parametrize("value", [False, np.int64(1), IntSubclass(1), 1.0, "1", None])
def test_tuple_scalar_types_are_not_coerced(field, value):
    batch = _batch()
    original = getattr(batch, field)
    bad = replace(batch, **{field: (value, *original[1:])})
    with pytest.raises(TypeError, match=field):
        pack_party_group_batch(bad, dtype=torch.float64)


@pytest.mark.parametrize("field", [
    "positive_ids", "user_ids", "positive_item_ids", "loss_coefficient",
])
@pytest.mark.parametrize("value", [-1, LIMIT + 1])
def test_tuple_scalar_values_must_fit_nonnegative_int64(field, value):
    batch = _batch()
    original = getattr(batch, field)
    bad = replace(batch, **{field: (value, *original[1:])})
    with pytest.raises(ValueError, match=field):
        pack_party_group_batch(bad, dtype=torch.float32)


@pytest.mark.parametrize("row", [[8, 7], TupleSubclass((8, 7)), None, 8])
def test_negative_rows_must_be_exact_tuples(row):
    with pytest.raises(TypeError, match="negative_item_ids row"):
        pack_party_group_batch(_batch(negative_item_ids=(row, (7, 7))),
                               dtype=torch.float32)


@pytest.mark.parametrize("value", [False, np.int64(8), IntSubclass(8), 8.0, "8", None])
def test_negative_ids_require_exact_builtin_integers(value):
    with pytest.raises(TypeError, match="negative_item_ids row"):
        pack_party_group_batch(_batch(negative_item_ids=((value, 7), (7, 7))),
                               dtype=torch.float32)


@pytest.mark.parametrize("value", [-1, LIMIT + 1])
def test_negative_ids_must_fit_nonnegative_int64(value):
    with pytest.raises(ValueError, match="negative_item_ids row"):
        pack_party_group_batch(_batch(negative_item_ids=((value, 7), (7, 7))),
                               dtype=torch.float32)


@pytest.mark.parametrize("changes, message", [
    ({"user_ids": (3,)}, "align"),
    ({"positive_item_ids": (2,)}, "align"),
    ({"negative_item_ids": ((8, 7),)}, "align"),
    ({"positive_ids": (203,)}, "align"),
    ({"start": 4}, "start/stop"),
    ({"start": 5}, "start/stop"),
    ({"stop": 5}, "start/stop"),
    ({"positive_ids": (203, 203)}, "unique"),
    ({"negative_item_ids": ((8,), (7, 7))}, "same length"),
    ({"negative_item_ids": ((), (7, 7))}, "same length"),
    ({"negative_item_ids": ((8, 2), (7, 7))}, "own negative"),
    ({"negative_item_ids": ((8, 7), (0, 0))}, "own negative"),
    ({"loss_coefficient": ()}, "loss_coefficient"),
    ({"loss_coefficient": (2,)}, "loss_coefficient"),
    ({"loss_coefficient": (2, 7, 1)}, "loss_coefficient"),
    ({"loss_coefficient": (1, 7)}, "loss_coefficient"),
    ({"loss_coefficient": (2, 1)}, "loss_coefficient"),
    ({"loss_coefficient": (2, 0)}, "loss_coefficient"),
    ({"loss_coefficient": (2, LIMIT + 1)}, "loss_coefficient"),
])
def test_malformed_values_fail_before_tensor_allocation(monkeypatch, changes, message):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid batch reached tensor allocation")

    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)
    with pytest.raises(ValueError, match=message):
        pack_party_group_batch(_batch(**changes), dtype=torch.float32)


def test_empty_batch_is_rejected_without_inventing_q_or_output(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("empty batch reached allocation")

    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)
    batch = _batch(
        start=0, stop=0, positive_ids=(), user_ids=(), positive_item_ids=(),
        negative_item_ids=(), loss_coefficient=(0, 0),
    )
    with pytest.raises(ValueError, match="at least one"):
        pack_party_group_batch(batch, dtype=torch.float32)


def test_integer_capacity_boundaries_without_allocating_large_arrays():
    # Exercise the same production preallocation arithmetic using only Python
    # scalar counts; a genuine tuple of LIMIT entries must never be allocated.
    assert trace_batch_module._expanded_size(1, LIMIT - 1) == LIMIT
    assert trace_batch_module._expanded_size(LIMIT - 1, 0) == LIMIT - 1
    with pytest.raises(ValueError, match="capacity"):
        trace_batch_module._expanded_size(1, LIMIT)
    with pytest.raises(ValueError, match="capacity"):
        trace_batch_module._expanded_size(2, LIMIT // 2)
    with pytest.raises(ValueError, match="capacity"):
        trace_batch_module._expanded_size(LIMIT, 0)


def test_capacity_guard_is_used_before_any_allocation(monkeypatch):
    observed = []

    def reject_capacity(groups, negatives):
        observed.append((groups, negatives))
        raise ValueError("capacity sentinel")

    def forbidden(*args, **kwargs):
        raise AssertionError("tensor allocation preceded capacity check")

    monkeypatch.setattr(trace_batch_module, "_expanded_size", reject_capacity)
    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)
    with pytest.raises(ValueError, match="capacity sentinel"):
        pack_party_group_batch(_batch(), dtype=torch.float32)
    assert observed == [(2, 2)]


def test_repeated_packing_does_not_sample_hash_or_advance_global_rng(monkeypatch):
    import src.data.negative_sampling as negative_sampling
    import src.federation.round_drain as round_drain

    def forbidden(*args, **kwargs):
        raise AssertionError("packing must not sample, hash, prepare a plan or use RNG")

    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    monkeypatch.setattr(negative_sampling, "sample_negative_items", forbidden)
    monkeypatch.setattr(round_drain, "sha256", forbidden)
    monkeypatch.setattr(round_drain, "prepare_round_drain_plan", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    first = pack_party_group_batch(_batch(), dtype=torch.float64)
    second = pack_party_group_batch(_batch(), dtype=torch.float64)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())
    for name in PAYLOAD_FIELDS:
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_dtype_is_explicit_and_keyword_only():
    with pytest.raises(TypeError):
        pack_party_group_batch(_batch())
    with pytest.raises(TypeError):
        pack_party_group_batch(_batch(), torch.float32)
