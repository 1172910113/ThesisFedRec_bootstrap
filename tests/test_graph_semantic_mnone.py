"""NEW bounded M-none numerical-wrapper tests, not collector/training validation.

Dense operators occur only in the fixed three-row NumPy oracle. No legacy event
reference, finite differences, training loop, or resource benchmark is run here.
"""

import json
import math
import random
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from src.methods.graph_semantic import mnone, path_operator, public_basis, u4_kernel
from src.methods.graph_semantic.mnone import (
    K1Sums,
    UnprotectedMNoneContribution,
    add_k1_contribution,
    empty_k1_sums,
    finalize_k1_bank,
    prepare_mnone_contribution,
)
from src.methods.graph_semantic.path_operator import prepare_path_factors
from src.methods.graph_semantic.u4_kernel import prepare_source_kernel


MAX_MASS = 2**53 - 1
RECORDS = [UnprotectedMNoneContribution, K1Sums]


def _source(*, kind="valid", dtype=torch.float32, zero_projection=False, public_grad=False):
    """Fixed NEW fixture: public rows4/1 share one aux; mapped row3 is isolated."""
    local_ids = () if kind == "empty" else (4, 1, 3)
    triples = {
        "valid": [(0, 2, 7), (1, 2, 7), (1, 2, 7)],
        "empty": [],
        "isolated": [],
        "singleton": [(0, 2, 7), (1, 2, 8)],
    }[kind]
    paths = prepare_path_factors(
        torch.tensor(local_ids, dtype=torch.long),
        torch.tensor(triples, dtype=torch.long).reshape(-1, 3),
        num_items=5,
        dtype=dtype,
    )
    e = torch.tensor(
        [[1, 0, -1], [.5, 2, 0], [3, -1, 1], [0, 1, .5], [2, 0, 1]],
        dtype=dtype,
        requires_grad=public_grad,
    )
    w = torch.tensor([1, -.5, .25], dtype=dtype, requires_grad=public_grad)
    b = torch.tensor(.1, dtype=dtype, requires_grad=public_grad)
    basis = torch.zeros((5, 2), dtype=dtype)
    basis[0 if zero_projection else 4, 0] = 1
    basis[2 if zero_projection else 1, 1] = 1
    basis.requires_grad_(public_grad)
    return prepare_source_kernel(paths, e, w, b, basis), (e, w, b, basis)


def _final_a():
    return torch.tensor([math.log(3), -math.log(3)], dtype=torch.float32)


def _dense_expected(a, *, zero_projection=False):
    """Independent literal incidence and dense normalization; no kernel outputs."""
    r = np.array([[1.], [1.], [0.]], dtype=np.float64)
    item_degree = r.sum(axis=1)
    auxiliary_degree = r.sum(axis=0)
    inverse_item = np.zeros(3, dtype=np.float64)
    inverse_item[item_degree > 0] = 1 / np.sqrt(item_degree[item_degree > 0])
    s = inverse_item[:, None] * r / np.sqrt(auxiliary_degree)[None, :]
    operator = s @ s.T
    np.fill_diagonal(operator, 0)
    four = operator @ operator
    local_basis = np.zeros((3, 2), dtype=np.float64)
    if not zero_projection:
        local_basis[0, 0], local_basis[1, 1] = 1, 1
    gates = 1 / (1 + np.exp(-np.asarray(a, dtype=np.float64)))
    expected = np.stack([
        gates[0] * (local_basis.T @ operator @ local_basis),
        gates[1] * (local_basis.T @ four @ local_basis),
    ])
    return r, s, operator, four, expected


def _array(value):
    return value.detach().cpu().numpy().copy()


def _state(value):
    """Observe existing bytes/versions/gradients without mutating the tensor."""
    gradient = value.grad if value.is_leaf or value.retains_grad else None
    return (
        tuple(value.shape), value.dtype, value.device, value.layout,
        value.is_contiguous(), value.requires_grad, value.grad_fn, value._version,
        value.detach().contiguous().numpy().tobytes(),
        None if gradient is None else gradient.detach().contiguous().numpy().tobytes(),
    )


def _states(values):
    return [_state(value) for value in values]


def _source_tensors(source):
    paths = source.paths
    return [
        source.item_embeddings, source.weight, source.bias, source.basis,
        source.f2, source.f4, paths.local_item_ids, paths.sorted_item_ids,
        paths.sorted_local_rows, paths.item_rows, paths.auxiliary_columns,
        paths.weights, paths.h,
    ]


def _assert_unchanged(values, expected):
    assert _states(values) == expected


def _rng_state():
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
        torch.get_rng_state().numpy().tobytes(),
    )


def _record(record_type, *, p=2, mass=1, fill=0):
    dtype = torch.float32 if record_type is UnprotectedMNoneContribution else torch.float64
    return record_type(
        values=torch.full((2, p, p), fill, dtype=dtype),
        mass=torch.tensor(mass, dtype=dtype),
    )


def _owned_distinct(*values):
    assert all(value._base is None for value in values)
    assert len({value.untyped_storage().data_ptr() for value in values}) == len(values)
    assert all(not value.requires_grad and value.grad_fn is None for value in values)


@pytest.mark.parametrize("record_type", RECORDS)
def test_record_api_is_frozen_slotted_keyword_only_and_copies_originals(record_type):
    dtype = torch.float32 if record_type is UnprotectedMNoneContribution else torch.float64
    original_values = torch.arange(8, dtype=dtype).reshape(2, 2, 2)
    original_mass = torch.tensor(1, dtype=dtype)
    before = _states([original_values, original_mass])
    record = record_type(values=original_values, mass=original_mass)
    assert [field.name for field in fields(record)] == ["values", "mass"]
    assert not hasattr(record, "__dict__")
    assert all(field.kw_only for field in fields(record))
    with pytest.raises(TypeError):
        record_type(original_values, original_mass)
    with pytest.raises(FrozenInstanceError):
        record.mass = original_mass
    _owned_distinct(record.values, record.mass)
    assert {record.values.data_ptr(), record.mass.data_ptr()}.isdisjoint(
        {original_values.data_ptr(), original_mass.data_ptr()}
    )
    _assert_unchanged([original_values, original_mass], before)
    expected = record.values.clone()
    original_values.zero_()
    original_mass.zero_()
    assert torch.equal(record.values, expected) and record.mass.item() == 1
    record.values.fill_(9)
    record.mass.zero_()
    assert torch.count_nonzero(original_values).item() == 0
    assert original_mass.item() == 0


BAD_RECORD_CASES = [
    "values_object", "mass_object", "values_dtype", "mass_dtype", "values_integer",
    "values_sparse", "values_meta", "mass_meta", "rank", "channels", "nonsquare",
    "zero_rank", "contiguous", "mass_shape", "values_nan", "values_inf",
    "mass_nan", "mass_inf", "values_attached", "mass_attached", "negative_mass",
    "fractional_mass", "over_mass", "zero_inconsistent",
]


def _bad_record_input(record_type, case):
    dtype = torch.float32 if record_type is UnprotectedMNoneContribution else torch.float64
    other_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    values, mass = torch.zeros((2, 2, 2), dtype=dtype), torch.tensor(1, dtype=dtype)
    error = TypeError if case in {
        "values_object", "mass_object", "values_dtype", "mass_dtype", "values_integer",
        "values_sparse", "values_meta", "mass_meta",
    } else ValueError
    if case == "values_object":
        values = np.zeros((2, 2, 2))
    elif case == "mass_object":
        mass = 1.0
    elif case == "values_dtype":
        values = values.to(other_dtype)
    elif case == "mass_dtype":
        mass = mass.to(other_dtype)
    elif case == "values_integer":
        values = values.to(torch.int64)
    elif case == "values_sparse":
        values = values.to_sparse()
    elif case == "values_meta":
        values = torch.empty((2, 2, 2), dtype=dtype, device="meta")
    elif case == "mass_meta":
        mass = torch.empty((), dtype=dtype, device="meta")
    elif case == "rank":
        values = torch.zeros((2, 2), dtype=dtype)
    elif case == "channels":
        values = torch.zeros((1, 2, 2), dtype=dtype)
    elif case == "nonsquare":
        values = torch.zeros((2, 2, 3), dtype=dtype)
    elif case == "zero_rank":
        values = torch.zeros((2, 0, 0), dtype=dtype)
    elif case == "contiguous":
        values = values.transpose(1, 2)
    elif case == "mass_shape":
        mass = mass.reshape(1)
    elif case == "values_nan":
        values[0, 0, 0] = float("nan")
    elif case == "values_inf":
        values[0, 0, 0] = float("inf")
    elif case == "mass_nan":
        mass.fill_(float("nan"))
    elif case == "mass_inf":
        mass.fill_(float("inf"))
    elif case == "values_attached":
        values.requires_grad_()
    elif case == "mass_attached":
        mass.requires_grad_()
    elif case == "negative_mass":
        mass.fill_(-1)
    elif case == "fractional_mass":
        mass.fill_(.5)
    elif case == "over_mass":
        mass.fill_(2 if dtype == torch.float32 else 2**53)
    elif case == "zero_inconsistent":
        values.fill_(1)
        mass.zero_()
    return values, mass, error


@pytest.mark.parametrize("record_type", RECORDS)
@pytest.mark.parametrize("case", BAD_RECORD_CASES)
def test_record_rejects_invalid_original_inputs_before_copy(record_type, case):
    values, mass, error = _bad_record_input(record_type, case)
    observed = [v for v in (values, mass) if isinstance(v, torch.Tensor)
                and v.layout == torch.strided and v.device.type == "cpu"]
    before = _states(observed)
    with pytest.raises(error):
        record_type(values=values, mass=mass)
    _assert_unchanged(observed, before)


@pytest.mark.parametrize("record_type", RECORDS)
@pytest.mark.parametrize("mass", [0, 1])
def test_zero_values_are_legal_for_each_legal_structural_mass(record_type, mass):
    record = _record(record_type, mass=mass)
    assert record.mass.item() == mass
    assert torch.count_nonzero(record.values).item() == 0


@pytest.mark.parametrize("p", [1, 2, 16])
def test_empty_sums_and_zero_banks_are_independent_and_have_exact_numeric_bytes(p):
    first, second = empty_k1_sums(p=p), empty_k1_sums(p=p)
    bank1, bank2 = finalize_k1_bank(first), finalize_k1_bank(first)
    assert first.values.shape == (2, p, p) and first.mass.shape == ()
    assert first.values.dtype == first.mass.dtype == torch.float64
    assert bank1.dtype == torch.float32 and bank1.shape == (2, p, p)
    _owned_distinct(first.values, first.mass, second.values, second.mass, bank1, bank2)
    assert torch.count_nonzero(first.values).item() == 0
    assert first.mass.item() == 0 and torch.count_nonzero(bank1).item() == 0
    bank1.fill_(9)
    assert torch.count_nonzero(bank2).item() == 0
    assert torch.count_nonzero(first.values).item() == 0
    if p == 16:
        contribution = _record(UnprotectedMNoneContribution, p=p, mass=0)
        source_bytes = sum(t.numel() * t.element_size() for t in (contribution.values, contribution.mass))
        assert source_bytes == 2052
        assert bank2.numel() * bank2.element_size() == 2048


@pytest.mark.parametrize("p", [True, np.int64(2), 2., "2", None])
def test_empty_sums_requires_exact_python_integer(p):
    with pytest.raises(TypeError):
        empty_k1_sums(p=p)


@pytest.mark.parametrize("p", [0, -1, 2**31])
def test_empty_sums_checks_range_with_python_ints_before_any_allocation(monkeypatch, p):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid p must be rejected before allocation")
    for name in ("zeros", "empty", "full"):
        monkeypatch.setattr(torch, name, forbidden)
    with pytest.raises(ValueError):
        empty_k1_sums(p=p)


def test_memory_error_is_not_relabeled_as_arithmetic_conclusion(monkeypatch):
    def exhausted(*args, **kwargs):
        raise MemoryError("injected allocator failure, not a size experiment")
    monkeypatch.setattr(torch, "zeros", exhausted)
    with pytest.raises(MemoryError):
        empty_k1_sums(p=1)


MUTATIONS = [
    ("values_nan", ValueError), ("values_rank", ValueError),
    ("values_contiguous", ValueError), ("values_attached", ValueError),
    ("values_dtype", TypeError), ("mass_rank", ValueError),
    ("mass_fractional", ValueError), ("mass_negative", ValueError),
    ("mass_over", ValueError), ("mass_attached", ValueError),
    ("zero_inconsistent", ValueError),
]


def _mutate(record, case):
    with torch.no_grad():
        if case == "values_nan":
            record.values.fill_(float("nan"))
        elif case == "values_rank":
            record.values.resize_(8)
        elif case == "values_contiguous":
            record.values.transpose_(1, 2)
        elif case == "values_attached":
            record.values.requires_grad_()
        elif case == "values_dtype":
            other = torch.float64 if record.values.dtype == torch.float32 else torch.float32
            record.values.data = record.values.to(other)
        elif case == "mass_rank":
            record.mass.resize_(1)
        elif case == "mass_fractional":
            record.mass.fill_(.5)
        elif case == "mass_negative":
            record.mass.fill_(-1)
        elif case == "mass_over":
            record.mass.fill_(2 if record.mass.dtype == torch.float32 else 2**53)
        elif case == "mass_attached":
            record.mass.requires_grad_()
        elif case == "zero_inconsistent":
            record.values.fill_(1)
            record.mass.zero_()


@pytest.mark.parametrize("case,error", MUTATIONS)
@pytest.mark.parametrize("operation", ["add", "finalize"])
def test_each_operation_revalidates_mutable_current_sum_buffers(case, error, operation):
    sums = _record(K1Sums)
    contribution = _record(UnprotectedMNoneContribution)
    _mutate(sums, case)
    observed = [sums.values, sums.mass, contribution.values, contribution.mass]
    before = _states(observed)
    with pytest.raises(error):
        if operation == "add":
            add_k1_contribution(sums, contribution)
        else:
            finalize_k1_bank(sums)
    _assert_unchanged(observed, before)


@pytest.mark.parametrize("case,error", MUTATIONS)
def test_add_revalidates_mutable_current_contribution_buffers(case, error):
    sums, contribution = empty_k1_sums(p=2), _record(UnprotectedMNoneContribution)
    _mutate(contribution, case)
    observed = [sums.values, sums.mass, contribution.values, contribution.mass]
    before = _states(observed)
    with pytest.raises(error):
        add_k1_contribution(sums, contribution)
    _assert_unchanged(observed, before)


@pytest.mark.parametrize("operation", ["add_sums", "add_contribution", "finalize"])
@pytest.mark.parametrize("invalid", [None, object(), torch.zeros(())])
def test_public_arithmetic_rejects_wrong_record_objects(operation, invalid):
    sums, contribution = empty_k1_sums(p=2), _record(UnprotectedMNoneContribution)
    with pytest.raises(TypeError):
        if operation == "add_sums":
            add_k1_contribution(invalid, contribution)
        elif operation == "add_contribution":
            add_k1_contribution(sums, invalid)
        else:
            finalize_k1_bank(invalid)


def test_public_arithmetic_requires_exact_record_classes_and_matching_p():
    class DerivedSums(K1Sums):
        pass
    class DerivedContribution(UnprotectedMNoneContribution):
        pass
    sums, contribution = empty_k1_sums(p=2), _record(UnprotectedMNoneContribution)
    derived_sum = DerivedSums(values=sums.values, mass=sums.mass)
    derived_contribution = DerivedContribution(values=contribution.values, mass=contribution.mass)
    with pytest.raises(TypeError):
        finalize_k1_bank(derived_sum)
    with pytest.raises(TypeError):
        add_k1_contribution(sums, derived_contribution)
    with pytest.raises(ValueError):
        add_k1_contribution(sums, _record(UnprotectedMNoneContribution, p=1))


def test_hand_sum_mean_and_zero_present_vs_absent_have_no_roster_semantics():
    seven = np.array([[[1, 0], [0, 2]], [[0, 3], [3, 0]]], dtype=np.float32)
    thirty_one = np.array([[[3, 0], [0, 4]], [[0, 1], [1, 0]]], dtype=np.float32)
    expected_sum = seven.astype(np.float64) + thirty_one.astype(np.float64)
    expected_bank = np.array([[[2, 0], [0, 3]], [[0, 2], [2, 0]]], dtype=np.float32)
    c7 = UnprotectedMNoneContribution(values=torch.from_numpy(seven.copy()), mass=torch.tensor(1.))
    c31 = UnprotectedMNoneContribution(values=torch.from_numpy(thirty_one.copy()), mass=torch.tensor(1.))
    c73 = _record(UnprotectedMNoneContribution, mass=0)
    initial = empty_k1_sums(p=2)
    first = add_k1_contribution(initial, c7)
    second = add_k1_contribution(first, c31)
    with_zero = add_k1_contribution(second, c73)
    observed = [initial.values, initial.mass, first.values, first.mass, second.values,
                second.mass, with_zero.values, with_zero.mass, c7.values, c7.mass,
                c31.values, c31.mass, c73.values, c73.mass]
    before = _states(observed)
    bank = finalize_k1_bank(with_zero)
    np.testing.assert_allclose(_array(second.values), expected_sum, atol=1e-12, rtol=0)
    np.testing.assert_allclose(_array(bank), expected_bank, atol=1e-6, rtol=1e-5)
    assert second.mass.item() == with_zero.mass.item() == 2
    assert torch.equal(second.values, with_zero.values)
    assert torch.equal(finalize_k1_bank(second), bank)
    repeated = add_k1_contribution(first, c7)
    np.testing.assert_array_equal(_array(repeated.values), seven.astype(np.float64) * 2)
    assert repeated.mass.item() == 2  # Pure math does not deduplicate releases.
    _assert_unchanged(observed, before)
    _owned_distinct(*observed, bank, repeated.values, repeated.mass)
    print("MNONE_SUM_ORACLE " + json.dumps({
        "expected_sum": expected_sum.tolist(), "expected_bank": expected_bank.tolist(),
        "actual_sum": _array(with_zero.values).tolist(), "actual_bank": _array(bank).tolist(),
        "sum_max_error": float(np.max(np.abs(_array(with_zero.values) - expected_sum))),
        "bank_max_error": float(np.max(np.abs(_array(bank) - expected_bank))),
        "numeric_equality_is_not_roster_completeness": True,
    }, sort_keys=True))


def test_mean_uses_float64_then_single_float32_cast_without_forcing_symmetry():
    first_values = torch.tensor([[[1, 2], [5, -4]], [[-3, .25], [7, 0]]], dtype=torch.float32)
    second_values = torch.tensor([[[4, -8], [2, 3]], [[9, .5], [-1, 2]]], dtype=torch.float32)
    contributions = [UnprotectedMNoneContribution(values=v, mass=torch.tensor(1.))
                     for v in (first_values, second_values)]
    sums = empty_k1_sums(p=2)
    for contribution in contributions:
        sums = add_k1_contribution(sums, contribution)
    expected_sum = _array(first_values).astype(np.float64) + _array(second_values).astype(np.float64)
    expected_bank = (expected_sum / 2).astype(np.float32)
    np.testing.assert_allclose(_array(sums.values), expected_sum, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(_array(finalize_k1_bank(sums)), expected_bank)
    assert expected_bank[0, 0, 1] != expected_bank[0, 1, 0]


def test_maximum_exact_integer_count_and_overflow_do_not_need_large_rosters():
    zero = _record(UnprotectedMNoneContribution, p=1, mass=0)
    one = _record(UnprotectedMNoneContribution, p=1, mass=1)
    below = _record(K1Sums, p=1, mass=MAX_MASS - 1)
    maximum = add_k1_contribution(below, one)
    assert maximum.mass.item() == MAX_MASS
    assert add_k1_contribution(maximum, zero).mass.item() == MAX_MASS
    observed = [below.values, below.mass, maximum.values, maximum.mass, one.values, one.mass]
    before = _states(observed)
    with pytest.raises(ArithmeticError):
        add_k1_contribution(maximum, one)
    _assert_unchanged(observed, before)
    assert torch.count_nonzero(finalize_k1_bank(maximum)).item() == 0
    with pytest.raises(ValueError):
        K1Sums(values=torch.zeros((2, 1, 1), dtype=torch.float64),
               mass=torch.tensor(2**53, dtype=torch.float64))


def test_finite_float64_bank_cast_overflow_is_arithmetic_failure_without_mutation():
    sums = K1Sums(
        values=torch.full((2, 1, 1), torch.finfo(torch.float64).max, dtype=torch.float64),
        mass=torch.tensor(1., dtype=torch.float64),
    )
    observed = [sums.values, sums.mass]
    before = _states(observed)
    with pytest.raises(ArithmeticError):
        finalize_k1_bank(sums)
    _assert_unchanged(observed, before)


class _InjectNonfiniteArithmetic(TorchDispatchMode):
    """One narrow post-input-validation fault; not a numeric scale experiment."""
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.fired = False

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if (not self.fired and function._schema.name == "aten::" + self.operation
                and isinstance(result, torch.Tensor) and result.dtype == torch.float64
                and result.shape == (2, 2, 2)):
            self.fired = True
            return torch.full_like(result, float("inf"))
        return result


@pytest.mark.parametrize("operation", ["add", "div"])
def test_nonfinite_computed_arithmetic_is_not_mislabeled_as_bad_caller_input(operation):
    sums, contribution = _record(K1Sums, fill=1), _record(UnprotectedMNoneContribution, fill=1)
    observed = [sums.values, sums.mass, contribution.values, contribution.mass]
    before = _states(observed)
    fault = _InjectNonfiniteArithmetic(operation)
    with fault, pytest.raises(ArithmeticError):
        if operation == "add":
            add_k1_contribution(sums, contribution)
        else:
            finalize_k1_bank(sums)
    assert fault.fired
    _assert_unchanged(observed, before)


@pytest.mark.parametrize("zero_projection", [False, True])
def test_final_source_wrapper_matches_new_independent_dense_numpy_oracle(zero_projection):
    source, _ = _source(zero_projection=zero_projection)
    a = _final_a()
    r, s, operator, four, expected = _dense_expected(_array(a), zero_projection=zero_projection)
    np.testing.assert_array_equal(r, [[1], [1], [0]])
    np.testing.assert_allclose(operator, [[0, .5, 0], [.5, 0, 0], [0, 0, 0]], atol=1e-15)
    np.testing.assert_allclose(four, np.diag([.25, .25, 0]), atol=1e-15)
    actual = prepare_mnone_contribution(source, a)
    initial = prepare_mnone_contribution(source, torch.zeros(2))
    np.testing.assert_allclose(_array(actual.values), expected, atol=1e-6, rtol=1e-5)
    assert actual.mass.item() == 1 and source.paths.structurally_valid
    assert not actual.values.requires_grad and actual.values.grad_fn is None
    if zero_projection:
        assert torch.count_nonzero(actual.values).item() == 0
    else:
        assert not torch.equal(actual.values, initial.values)
        np.testing.assert_allclose(expected, [
            [[0, .375], [.375, 0]], [[.0625, 0], [0, .0625]],
        ], atol=1e-6, rtol=1e-5)
    print("MNONE_ORACLE " + json.dumps({
        "zero_projection": zero_projection, "r": r.tolist(), "s": s.tolist(),
        "m": operator.tolist(), "m_squared": four.tolist(), "final_a": _array(a).tolist(),
        "expected": expected.tolist(), "actual": _array(actual.values).tolist(),
        "maximum_error": float(np.max(np.abs(_array(actual.values) - expected))),
        "mass": actual.mass.item(), "atol": 1e-6, "rtol": 1e-5,
    }, sort_keys=True))


@pytest.mark.parametrize("kind", ["empty", "isolated", "singleton"])
def test_invalid_structure_still_recomputes_snapshot_once_and_emits_owned_zeros(monkeypatch, kind):
    source, _ = _source(kind=kind)
    a = _final_a()
    calls = []
    real = mnone.detached_operator_snapshot
    def counted(actual_source, actual_a):
        calls.append((actual_source, actual_a))
        return real(actual_source, actual_a)
    monkeypatch.setattr(mnone, "detached_operator_snapshot", counted)
    contribution = prepare_mnone_contribution(source, a)
    assert len(calls) == 1 and calls[0][0] is source and calls[0][1] is a
    assert contribution.mass.item() == 0 and not source.paths.structurally_valid
    assert torch.count_nonzero(contribution.values).item() == 0
    assert not torch.signbit(contribution.values).any().item()
    _owned_distinct(contribution.values, contribution.mass)


@pytest.mark.parametrize("kind", ["valid", "empty"])
@pytest.mark.parametrize("enabled", [True, False])
def test_nonleaf_final_a_source_gradients_rng_and_ambient_gradmode_are_preserved(monkeypatch, kind, enabled):
    with torch.enable_grad():
        source, public = _source(kind=kind, public_grad=True)
        leaf = torch.tensor([math.log(3) / 2, -math.log(3) / 2], requires_grad=True)
        a = leaf * 2
        a.retain_grad()
        leaf.grad = torch.tensor([4., 5.])
        a.grad = torch.tensor([6., 7.])
        for value in public:
            value.grad = torch.full_like(value, .75)
    observed = _source_tensors(source) + list(public) + [leaf, a]
    before, rng = _states(observed), _rng_state()
    modes = []
    real = mnone.detached_operator_snapshot
    def counted(actual_source, actual_a):
        assert actual_source is source and actual_a is a
        modes.append(torch.is_grad_enabled())
        return real(actual_source, actual_a)
    monkeypatch.setattr(mnone, "detached_operator_snapshot", counted)
    outer_mode = torch.is_grad_enabled()
    with torch.set_grad_enabled(enabled):
        result = prepare_mnone_contribution(source, a)
        assert torch.is_grad_enabled() is enabled
        sums = add_k1_contribution(empty_k1_sums(p=2), result)
        bank = finalize_k1_bank(sums)
        assert torch.is_grad_enabled() is enabled
    assert torch.is_grad_enabled() is outer_mode
    assert modes == [enabled] and _rng_state() == rng
    _assert_unchanged(observed, before)
    _owned_distinct(result.values, result.mass, sums.values, sums.mass, bank)


def test_borrowed_source_and_final_a_may_be_strided_and_grad_connected():
    source, _ = _source()
    source = replace(
        source,
        item_embeddings=source.item_embeddings.T.contiguous().T,
        basis=source.basis.T.contiguous().T,
        f2=source.f2.T.contiguous().T,
        f4=source.f4.T.contiguous().T,
    )
    leaf = torch.tensor([math.log(3), 9., -math.log(3), 8.], requires_grad=True)
    a = leaf[::2]
    assert not a.is_contiguous() and a.grad_fn is not None
    assert not source.item_embeddings.is_contiguous() and not source.basis.is_contiguous()
    result = prepare_mnone_contribution(source, a)
    expected = _dense_expected(_array(a))[-1]
    np.testing.assert_allclose(_array(result.values), expected, atol=1e-6, rtol=1e-5)
    assert result.mass.item() == 1


SOURCE_FIELDS = ["item_embeddings", "weight", "bias", "basis", "f2", "f4", "weights", "h"]


@pytest.mark.parametrize("kind", ["valid", "empty"])
@pytest.mark.parametrize("field", SOURCE_FIELDS)
@pytest.mark.parametrize("fault,error", [("dtype", TypeError), ("nonfinite", ValueError), ("shape", ValueError)])
def test_every_source_float_buffer_is_validated_before_summary_even_when_empty(monkeypatch, kind, field, fault, error):
    source, _ = _source(kind=kind)
    value = getattr(source.paths if field in {"weights", "h"} else source, field)
    if fault == "dtype":
        bad = value.double()
    elif fault == "shape":
        bad = value.reshape(-1, 1) if value.ndim != 2 else value.reshape(-1)
    elif value.numel():
        bad = value.clone()
        bad.reshape(-1)[0] = float("nan")
    else:
        # A nonfinite nonempty replacement must still fail before the sketch.
        bad = torch.full((1,), float("nan"))
    if field in {"weights", "h"}:
        source = replace(source, paths=replace(source.paths, **{field: bad}))
    else:
        source = replace(source, **{field: bad})
    def forbidden(*args, **kwargs):
        raise AssertionError("source/final_a validation must precede summary")
    monkeypatch.setattr(mnone, "detached_operator_snapshot", forbidden)
    with pytest.raises(error):
        prepare_mnone_contribution(source, _final_a())


@pytest.mark.parametrize("kind", ["valid", "empty"])
@pytest.mark.parametrize("case,error", [
    ("object", TypeError), ("dtype", TypeError), ("meta", TypeError),
    ("sparse", TypeError), ("shape", ValueError), ("nan", ValueError),
])
def test_final_a_is_validated_before_summary_in_every_structure_branch(monkeypatch, kind, case, error):
    source, _ = _source(kind=kind)
    a = _final_a()
    if case == "object":
        a = [1., 2.]
    elif case == "dtype":
        a = a.double()
    elif case == "meta":
        a = torch.empty((2,), device="meta")
    elif case == "sparse":
        a = a.to_sparse()
    elif case == "shape":
        a = a.reshape(1, 2)
    else:
        a[0] = float("nan")
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid a must fail before snapshot")
    monkeypatch.setattr(mnone, "detached_operator_snapshot", forbidden)
    with pytest.raises(error):
        prepare_mnone_contribution(source, a)


@pytest.mark.parametrize("kind", ["valid", "empty"])
def test_float64_source_is_unsupported_by_wrapper_without_changing_analytical_kernel(monkeypatch, kind):
    source, _ = _source(kind=kind, dtype=torch.float64)
    def forbidden(*args, **kwargs):
        raise AssertionError("float64 source must fail before summary")
    monkeypatch.setattr(mnone, "detached_operator_snapshot", forbidden)
    with pytest.raises(TypeError):
        prepare_mnone_contribution(source, _final_a())


@pytest.mark.parametrize("case", ["source_object", "source_subclass", "paths_object", "structural_flag"])
def test_source_and_paths_require_the_existing_exact_records(case):
    source, _ = _source()
    if case == "source_object":
        source = object()
    elif case == "source_subclass":
        class DerivedSource(u4_kernel.SourceKernel):
            pass
        source = DerivedSource(**{field.name: getattr(source, field.name) for field in fields(source)})
    elif case == "paths_object":
        source = replace(source, paths=object())
    else:
        source = replace(source, paths=replace(source.paths, structurally_valid=1))
    with pytest.raises(TypeError):
        prepare_mnone_contribution(source, _final_a())


@pytest.mark.parametrize("kind", ["valid", "empty"])
@pytest.mark.parametrize("fault,error", [("nonfinite", ArithmeticError), ("dtype", TypeError), ("shape", ValueError)])
def test_computed_summary_postconditions_are_checked_before_structural_zero(monkeypatch, kind, fault, error):
    source, _ = _source(kind=kind)
    calls = []
    if fault == "nonfinite":
        returned = torch.full((2, 2, 2), float("inf"))
    elif fault == "dtype":
        returned = torch.zeros((2, 2, 2), dtype=torch.float64)
    else:
        returned = torch.zeros((2, 2))
    def faulty(actual_source, actual_a):
        calls.append(actual_source)
        return returned
    monkeypatch.setattr(mnone, "detached_operator_snapshot", faulty)
    observed = _source_tensors(source) + [returned]
    before = _states(observed)
    with pytest.raises(error):
        prepare_mnone_contribution(source, _final_a())
    assert calls == [source]
    _assert_unchanged(observed, before)


class _ObserveTinyCalls(TorchDispatchMode):
    def __init__(self, forbidden_clone_pointers):
        super().__init__()
        self.forbidden_clone_pointers = forbidden_clone_pointers
        self.full_public_clones = 0
        self.catalogue_square_results = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function._schema.name == "aten::clone" and args:
            value = args[0]
            if isinstance(value, torch.Tensor) and value.data_ptr() in self.forbidden_clone_pointers:
                self.full_public_clones += 1
        result = function(*args, **(kwargs or {}))
        values = result if isinstance(result, (tuple, list)) else (result,)
        self.catalogue_square_results += sum(
            isinstance(value, torch.Tensor) and tuple(value.shape) == (5, 5)
            for value in values
        )
        return result


def test_wrapper_reuses_one_summary_without_global_builders_or_public_tensor_clones(monkeypatch):
    source, _ = _source()
    a = _final_a()
    real, returned = mnone.detached_operator_snapshot, []
    def counted(actual_source, actual_a):
        result = real(actual_source, actual_a)
        returned.append(result)
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError("wrapper must not prepare graph/path/basis/source again")
    monkeypatch.setattr(mnone, "detached_operator_snapshot", counted)
    monkeypatch.setattr(path_operator, "prepare_path_factors", forbidden)
    monkeypatch.setattr(u4_kernel, "prepare_source_kernel", forbidden)
    for name in ("prepare_public_basis", "validate_public_basis", "build_public_basis"):
        if hasattr(public_basis, name):
            monkeypatch.setattr(public_basis, name, forbidden)
    observation = _ObserveTinyCalls({source.item_embeddings.data_ptr(), source.basis.data_ptr()})
    with observation:
        result = prepare_mnone_contribution(source, a)
    assert len(returned) == 1
    assert result.values.data_ptr() != returned[0].data_ptr()
    assert observation.full_public_clones == observation.catalogue_square_results == 0
    source_buffers = [source.paths.weights, source.paths.h, source.f2, source.f4]
    print("MNONE_BUFFER_OBSERVATION " + json.dumps({
        "kind": "actual_tiny_fixture_numeric_buffers_only_not_benchmark",
        "n": 5, "m": 3, "d": 3, "p": 2,
        "source_local_float_bytes": sum(t.numel() * t.element_size() for t in source_buffers),
        "contribution_numeric_bytes": sum(t.numel() * t.element_size() for t in (result.values, result.mass)),
        "snapshot_calls": len(returned), "global_builder_calls": 0,
        "observed_full_public_clones": observation.full_public_clones,
        "observed_catalogue_square_results": observation.catalogue_square_results,
        "serialized_bytes": None, "wire_bytes": None,
    }, sort_keys=True))


def test_finite_syntactic_source_intermediate_overflow_is_arithmetic_with_original_cause():
    source, _ = _source()
    # Deliberate test-only corrupt finite basis. This tests a numerical failure
    # guard, not provenance, Gram validation or validity of the modified source.
    source = replace(source, basis=torch.full_like(source.basis, torch.finfo(torch.float32).max))
    observed = _source_tensors(source)
    before = _states(observed)
    with pytest.raises(ArithmeticError) as captured:
        prepare_mnone_contribution(source, _final_a())
    assert isinstance(captured.value.__cause__, ValueError)
    assert str(captured.value.__cause__) == "values must be finite CPU torch.float32 with strided layout"
    _assert_unchanged(observed, before)


@pytest.mark.parametrize("error", [ValueError("unrelated snapshot failure"), MemoryError("snapshot allocation failure")])
def test_unrelated_snapshot_failures_are_propagated_without_broad_conversion(monkeypatch, error):
    source, _ = _source()
    a = _final_a()
    observed = _source_tensors(source) + [a]
    before = _states(observed)
    def faulty(*args, **kwargs):
        raise error
    monkeypatch.setattr(mnone, "detached_operator_snapshot", faulty)
    with pytest.raises(type(error)) as captured:
        prepare_mnone_contribution(source, a)
    assert captured.value is error
    _assert_unchanged(observed, before)


@pytest.mark.parametrize("case,error", [
    ("num_items_type", TypeError), ("num_items_mismatch", ValueError),
    ("auxiliary_count_type", TypeError), ("auxiliary_count_negative", ValueError),
    ("local_ids_dtype", TypeError), ("local_ids_shape", ValueError),
    ("local_ids_range", ValueError), ("sorted_ids_shape", ValueError),
    ("sorted_rows_range", ValueError), ("item_rows_range", ValueError),
    ("auxiliary_columns_range", ValueError), ("edge_length_mismatch", ValueError),
])
def test_relevant_existing_path_metadata_is_checked_without_rebuilding_paths(monkeypatch, case, error):
    source, _ = _source()
    paths = source.paths
    if case == "num_items_type":
        paths = replace(paths, num_items=True)
    elif case == "num_items_mismatch":
        paths = replace(paths, num_items=4)
    elif case == "auxiliary_count_type":
        paths = replace(paths, auxiliary_count=np.int64(1))
    elif case == "auxiliary_count_negative":
        paths = replace(paths, auxiliary_count=-1)
    elif case == "local_ids_dtype":
        paths = replace(paths, local_item_ids=paths.local_item_ids.float())
    elif case == "local_ids_shape":
        paths = replace(paths, local_item_ids=paths.local_item_ids.reshape(1, -1))
    elif case == "local_ids_range":
        paths = replace(paths, local_item_ids=torch.tensor([5, 1, 3]))
    elif case == "sorted_ids_shape":
        paths = replace(paths, sorted_item_ids=paths.sorted_item_ids[:2])
    elif case == "sorted_rows_range":
        paths = replace(paths, sorted_local_rows=torch.tensor([3, 1, 2]))
    elif case == "item_rows_range":
        paths = replace(paths, item_rows=torch.full_like(paths.item_rows, 3))
    elif case == "auxiliary_columns_range":
        paths = replace(paths, auxiliary_columns=torch.full_like(paths.auxiliary_columns, 1))
    elif case == "edge_length_mismatch":
        paths = replace(paths, weights=torch.ones(paths.weights.numel() + 1))
    source = replace(source, paths=paths)
    def forbidden(*args, **kwargs):
        raise AssertionError("malformed metadata must fail before snapshot")
    monkeypatch.setattr(mnone, "detached_operator_snapshot", forbidden)
    with pytest.raises(error):
        prepare_mnone_contribution(source, _final_a())
