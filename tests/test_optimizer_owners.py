"""Short independent CPU Adam recurrence and owner-runtime boundary checks.

The oracle uses literal Python scalar arithmetic, not Adam or runtime helpers.
Counter injections are synthetic validation fixtures, not restoration support.
"""

from dataclasses import FrozenInstanceError, fields
import inspect
import math
import random

import numpy as np
import pytest
import torch

from src.federation.optimizer_owners import (
    OptimizerOwners,
    OptimizerOwnersSnapshot,
    OptimizerStepError,
    OptimizerStepReceipt,
    apply_optimizer_gradients,
    close_optimizer_owners,
    prepare_optimizer_owners,
    snapshot_optimizer_owners,
)


PROFILE = dict(
    learning_rate=0.025, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=0.0
)
COUNTER_MAX = 2**24 - 1
VERSION_MAX = 2**63 - 1


def _parameter(values, dtype=torch.float64):
    return torch.nn.Parameter(torch.tensor(values, dtype=dtype))


def _prepare(parameters=None, **profile):
    if parameters is None:
        parameters = {"z": {"weight": _parameter([1.0, -2.0])}}
    return prepare_optimizer_owners(parameters, **(PROFILE | profile))


def _gradients(parameters, value=0.25):
    return {
        owner: {name: torch.full_like(parameter, value) for name, parameter in group.items()}
        for owner, group in parameters.items()
    }


def _none_gradients(parameters):
    return {owner: dict.fromkeys(group) for owner, group in parameters.items()}


def _state(owners, owner, parameter):
    # Direct private inspection is restricted to lifecycle and fault fixtures.
    return owners._optimizers[owner].state[parameter]


def _equal_snapshots(left, right):
    for name in (
        "generation", "learning_rate", "betas", "epsilon", "weight_decay",
        "owner_update_versions", "parameter_steps",
    ):
        assert getattr(left, name) == getattr(right, name)
    for name in ("parameters", "exp_avg", "exp_avg_sq"):
        a, b = getattr(left, name), getattr(right, name)
        assert a.keys() == b.keys()
        for owner in a:
            assert a[owner].keys() == b[owner].keys()
            for parameter in a[owner]:
                x, y = a[owner][parameter], b[owner][parameter]
                if x is None:
                    assert y is None
                else:
                    assert torch.equal(x, y)


def _scalar_adam(values, moments, gradients, step):
    """Independent equation, including bias correction and epsilon position."""
    first, second = moments
    lr, (beta1, beta2), epsilon = 0.025, (0.9, 0.999), 1e-8
    out, m_out, v_out = [], [], []
    for parameter, mean, square, gradient in zip(values, first, second, gradients):
        mean = beta1 * mean + (1.0 - beta1) * gradient
        square = beta2 * square + (1.0 - beta2) * gradient * gradient
        corrected_mean = mean / (1.0 - beta1**step)
        corrected_square = square / (1.0 - beta2**step)
        parameter -= lr * corrected_mean / (math.sqrt(corrected_square) + epsilon)
        out.append(parameter)
        m_out.append(mean)
        v_out.append(square)
    return out, (m_out, v_out)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_three_step_independent_scalar_adam_oracle(dtype, record_property):
    parameters = {
        "source": {"user": _parameter([0.5, -0.75], dtype),
                   "encoder": _parameter([[1.0, -2.0], [3.0, 0.25]], dtype)},
        "shared": {"items": _parameter([[0.2, -0.4], [0.6, -0.8]], dtype)},
    }
    literal_gradients = [
        {"source": {"user": None, "encoder": [[0.2, -0.3], [0.4, -0.5]]},
         "shared": {"items": [[0.4, -0.2], [0.3, -0.1]]}},
        {"source": {"user": [0.1, -0.2], "encoder": None},
         "shared": {"items": [[0.0, 0.0], [0.2, -0.4]]}},
        {"source": {"user": [0.0, 0.0], "encoder": [[0.0, 0.0], [0.0, 0.0]]},
         "shared": {"items": None}},
    ]
    expected = {
        (owner, name): (parameter.detach().flatten().tolist(),
                        ([0.0] * parameter.numel(), [0.0] * parameter.numel()), 0)
        for owner, group in parameters.items() for name, parameter in group.items()
    }
    owners = _prepare(parameters)
    optimizer_ids = {owner: id(optimizer) for owner, optimizer in owners._optimizers.items()}
    parameter_ids = {(owner, name): id(p) for owner, group in parameters.items() for name, p in group.items()}
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    max_parameter_error, max_moment_error = 0.0, 0.0
    counter_dtypes = set()
    prior_items = None
    for generation, literal in enumerate(literal_gradients):
        gradients = {
            owner: {name: None if values is None else torch.tensor(values, dtype=dtype)
                    for name, values in group.items()} for owner, group in literal.items()
        }
        receipt = apply_optimizer_gradients(owners, gradients, expected_generation=generation)
        snapshot = snapshot_optimizer_owners(owners)
        assert receipt.generation_before == generation
        assert receipt.generation_after == generation + 1
        assert receipt.stepped_owner_ids == (("shared", "source") if generation < 2 else ("source",))
        assert receipt.owner_update_versions == (("shared", min(generation + 1, 2)), ("source", generation + 1))
        for owner, group in parameters.items():
            assert id(owners._optimizers[owner]) == optimizer_ids[owner]
            for name, parameter in group.items():
                assert id(parameter) == parameter_ids[owner, name]
                assert parameter.grad is None
                values, moments, step = expected[owner, name]
                gradient = gradients[owner][name]
                if gradient is not None:
                    step += 1
                    values, moments = _scalar_adam(values, moments, gradient.flatten().tolist(), step)
                    expected[owner, name] = values, moments, step
                expected_tensor = torch.tensor(values, dtype=dtype)
                max_parameter_error = max(max_parameter_error, (parameter.detach().flatten() - expected_tensor).abs().max().item())
                torch.testing.assert_close(parameter.flatten(), expected_tensor, atol=tolerance, rtol=tolerance)
                assert snapshot.parameter_steps[owner][name] == step
                if step:
                    counter_dtypes.add(str(_state(owners, owner, parameter)["step"].dtype))
                    for field, expected_moment in zip(("exp_avg", "exp_avg_sq"), moments):
                        actual_moment = getattr(snapshot, field)[owner][name].flatten()
                        expected_tensor = torch.tensor(expected_moment, dtype=dtype)
                        max_moment_error = max(max_moment_error, (actual_moment - expected_tensor).abs().max().item())
                        torch.testing.assert_close(actual_moment, expected_tensor, atol=tolerance, rtol=tolerance)
                else:
                    assert snapshot.exp_avg[owner][name] is None
                    assert snapshot.exp_avg_sq[owner][name] is None
        if generation == 0:
            # No user path in a source-only owner does not create user state.
            assert snapshot.parameter_steps["source"]["user"] == 0
            prior_items = snapshot.parameters["shared"]["items"].clone()
        if generation == 1:
            # Dense zero gradient in an unrequested row still moves its old momentum.
            assert not torch.equal(parameters["shared"]["items"][0], prior_items[0])
    assert owners.generation == 3
    record_property("oracle", "independent Python scalar Adam recurrence; three calls")
    record_property("dtype", str(dtype))
    record_property("atol_rtol", tolerance)
    record_property("maximum_parameter_absolute_error", max_parameter_error)
    record_property("maximum_moment_absolute_error", max_moment_error)
    record_property("actual_adam_counter_dtypes", ",".join(sorted(counter_dtypes)))


def test_profile_parameter_identity_canonical_order_and_no_constructor():
    with pytest.raises(TypeError):
        OptimizerOwners()
    parameters = {"z-local": {"z": _parameter([3.0]), "a": _parameter([2.0])},
                  "a-shared": {"w": _parameter([1.0])}}
    owners = _prepare(parameters)
    assert owners.status == "active" and owners.generation == 0 and owners.last_failure is None
    snapshot = snapshot_optimizer_owners(owners)
    assert list(snapshot.parameters) == ["a-shared", "z-local"]
    assert list(snapshot.parameters["z-local"]) == ["a", "z"]
    assert snapshot.owner_update_versions == (("a-shared", 0), ("z-local", 0))
    assert snapshot.parameter_steps == {"a-shared": {"w": 0}, "z-local": {"a": 0, "z": 0}}
    for owner, optimizer in owners._optimizers.items():
        group = optimizer.param_groups[0]
        assert type(optimizer) is torch.optim.Adam
        assert len(optimizer.param_groups) == 1
        assert [id(p) for p in group["params"]] == [id(parameters[owner][name]) for name in sorted(parameters[owner])]
        for key in ("amsgrad", "maximize", "foreach", "fused", "capturable", "differentiable"):
            assert group[key] is False
        assert group["lr"] == 0.025 and group["betas"] == (0.9, 0.999)
        assert group["eps"] == 1e-8 and group["weight_decay"] == 0.0
    for name in ("generation", "status", "last_failure"):
        with pytest.raises(AttributeError):
            setattr(owners, name, None)


def test_frozen_keyword_only_exact_observation_fields_and_receipt_steps(record_property):
    parameters = {"b": {"none": _parameter([1.0]), "used": _parameter([2.0])}}
    owners = _prepare(parameters)
    receipt = apply_optimizer_gradients(owners, {"b": {"used": torch.tensor([0.0], dtype=torch.float64), "none": None}}, expected_generation=0)
    snapshot = snapshot_optimizer_owners(owners)
    assert tuple(f.name for f in fields(receipt)) == ("generation_before", "generation_after", "stepped_owner_ids", "owner_update_versions", "parameter_steps")
    assert tuple(f.name for f in fields(snapshot)) == ("generation", "learning_rate", "betas", "epsilon", "weight_decay", "owner_update_versions", "parameters", "parameter_steps", "exp_avg", "exp_avg_sq")
    for record in (receipt, snapshot):
        assert not hasattr(record, "__dict__")
        assert all(f.kw_only for f in fields(record))
        with pytest.raises(FrozenInstanceError):
            setattr(record, "generation" if isinstance(record, OptimizerOwnersSnapshot) else "generation_before", -1)
    assert type(receipt) is OptimizerStepReceipt
    assert type(snapshot) is OptimizerOwnersSnapshot
    actual_dtype = _state(owners, "b", parameters["b"]["used"])["step"].dtype
    assert actual_dtype in (torch.float32, torch.float64)
    record_property("actual_adam_counter_dtype", str(actual_dtype))
    assert receipt.parameter_steps == (("b", "none", 0), ("b", "used", 1))
    for field in ("parameters", "parameter_steps", "exp_avg", "exp_avg_sq"):
        assert type(getattr(snapshot, field)) is dict
        assert type(getattr(snapshot, field)["b"]) is dict
    signature = inspect.signature(prepare_optimizer_owners)
    assert all(signature.parameters[name].default is inspect.Parameter.empty for name in PROFILE)


def test_receipt_flat_all_parameters_canonical_order_and_all_none_call(record_property):
    parameters = {
        "source": {"user": _parameter([1.0]), "encoder": _parameter([2.0])},
        "shared": {"items": _parameter([3.0])},
        "idle": {"weight": _parameter([4.0])},
    }
    owners = _prepare(parameters)
    receipt = apply_optimizer_gradients(
        owners,
        {"source": {"user": None, "encoder": torch.tensor([0.25], dtype=torch.float64)},
         "shared": {"items": None}, "idle": {"weight": None}},
        expected_generation=0,
    )
    expected = (("idle", "weight", 0), ("shared", "items", 0),
                ("source", "encoder", 1), ("source", "user", 0))
    expected_snapshot = {"idle": {"weight": 0}, "shared": {"items": 0},
                         "source": {"encoder": 1, "user": 0}}
    assert receipt.stepped_owner_ids == ("source",)
    assert receipt.owner_update_versions == (("idle", 0), ("shared", 0), ("source", 1))
    before = snapshot_optimizer_owners(owners)
    all_none = apply_optimizer_gradients(owners, _none_gradients(parameters), expected_generation=1)
    after = snapshot_optimizer_owners(owners)
    assert (receipt.generation_before, receipt.generation_after) == (0, 1)
    assert (all_none.generation_before, all_none.generation_after) == (1, 2)
    assert all_none.stepped_owner_ids == ()
    assert all_none.owner_update_versions == receipt.owner_update_versions
    for result in (receipt, all_none):
        assert type(result.parameter_steps) is tuple
        assert result.parameter_steps == expected
        assert all(type(row) is tuple and len(row) == 3 for row in result.parameter_steps)
        for owner_id, parameter_name, step in result.parameter_steps:
            assert type(owner_id) is str and type(parameter_name) is str
            assert type(step) is int
    for snapshot in (before, after):
        assert type(snapshot.parameter_steps) is dict
        assert snapshot.parameter_steps == expected_snapshot
        for group in snapshot.parameter_steps.values():
            assert type(group) is dict
            assert all(type(step) is int for step in group.values())
    record_property("flat_receipt_parameter_steps", repr(receipt.parameter_steps))
    record_property("snapshot_parameter_steps", repr(after.parameter_steps))


def test_all_none_advances_only_generation_and_never_calls_step(monkeypatch):
    parameters = {"a": {"w": _parameter([1.0])}, "b": {"w": _parameter([2.0])}}
    owners = _prepare(parameters)
    before = snapshot_optimizer_owners(owners)
    def forbidden():
        raise AssertionError("all-None owner must not step")
    for optimizer in owners._optimizers.values():
        monkeypatch.setattr(optimizer, "step", forbidden)
    receipt = apply_optimizer_gradients(owners, _none_gradients(parameters), expected_generation=0)
    assert receipt.generation_after == 1 and receipt.stepped_owner_ids == ()
    assert receipt.owner_update_versions == (("a", 0), ("b", 0))
    after = snapshot_optimizer_owners(owners)
    assert after.parameter_steps == before.parameter_steps
    assert all(value is None for group in after.exp_avg.values() for value in group.values())
    for owner in parameters:
        assert torch.equal(before.parameters[owner]["w"], after.parameters[owner]["w"])


def test_all_none_preserves_initialized_moments_and_parameters_exactly():
    parameters = {"o": {"w": _parameter([1.0, -2.0])}}
    owners = _prepare(parameters)
    apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=0)
    before = snapshot_optimizer_owners(owners)
    receipt = apply_optimizer_gradients(owners, _none_gradients(parameters), expected_generation=1)
    after = snapshot_optimizer_owners(owners)
    assert receipt.stepped_owner_ids == () and after.generation == 2
    assert before.parameter_steps == after.parameter_steps
    assert before.owner_update_versions == after.owner_update_versions
    for field in ("parameters", "exp_avg", "exp_avg_sq"):
        assert torch.equal(getattr(before, field)["o"]["w"], getattr(after, field)["o"]["w"])


def test_input_mappings_and_snapshots_do_not_alias_runtime_or_each_other():
    parameter = _parameter([1.0, 2.0])
    parameters = {"o": {"w": parameter}}
    owners = _prepare(parameters)
    parameters["o"]["w"] = _parameter([9.0, 9.0])
    parameters["extra"] = {"other": _parameter([3.0])}
    del parameters["o"]
    gradient = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True) * 2
    before_gradient = gradient.detach().clone()
    apply_optimizer_gradients(owners, {"o": {"w": gradient}}, expected_generation=0)
    assert torch.equal(gradient, before_gradient)
    first, second = snapshot_optimizer_owners(owners), snapshot_optimizer_owners(owners)
    buffers = [parameter]
    for snapshot in (first, second):
        for name in ("parameters", "exp_avg", "exp_avg_sq"):
            tensor = getattr(snapshot, name)["o"]["w"]
            assert tensor.grad_fn is None and not tensor.requires_grad
            buffers.append(tensor)
    assert len({t.untyped_storage().data_ptr() for t in buffers}) == len(buffers)
    for name in ("parameters", "exp_avg", "exp_avg_sq"):
        getattr(first, name)["o"]["w"].fill_(99.0)
        getattr(first, name)["o"]["w"].resize_(1)
        getattr(first, name)["extra"] = {}
    first.parameter_steps["o"]["w"] = 999
    first.parameters.clear()
    _equal_snapshots(second, snapshot_optimizer_owners(owners))
    assert parameter.grad is None


def test_noncontiguous_nonleaf_gradients_and_zero_element_parameters(monkeypatch):
    noncontiguous = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64).t())
    parameters = {"o": {"view": noncontiguous, "empty": _parameter([]), "other-empty": _parameter([])}}
    owners = _prepare(parameters)
    upstream = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64, requires_grad=True)
    gradient = (upstream * 2).t()
    assert not noncontiguous.is_contiguous() and not gradient.is_contiguous() and not gradient.is_leaf
    original, version = gradient.detach().clone(), gradient._version
    observed_gradients = []
    original_step = owners._optimizers["o"].step
    def observe():
        observed_gradients.append(noncontiguous.grad)
        assert noncontiguous.grad.grad_fn is None and not noncontiguous.grad.requires_grad
        assert noncontiguous.grad.untyped_storage().data_ptr() != gradient.untyped_storage().data_ptr()
        return original_step()
    monkeypatch.setattr(owners._optimizers["o"], "step", observe)
    apply_optimizer_gradients(owners, {"o": {"view": gradient, "empty": torch.empty(0, dtype=torch.float64), "other-empty": None}}, expected_generation=0)
    assert len(observed_gradients) == 1
    assert upstream.grad is None and gradient._version == version and torch.equal(gradient, original)
    assert snapshot_optimizer_owners(owners).parameter_steps == {"o": {"empty": 1, "other-empty": 0, "view": 1}}
    assert all(p.grad is None for p in parameters["o"].values())


def test_gradient_aliases_to_each_other_are_copied_independently(monkeypatch):
    parameters = {"o": {"a": _parameter([1.0, 2.0]), "b": _parameter([3.0, 4.0])}}
    owners = _prepare(parameters)
    gradient = torch.tensor([0.2, 0.3], dtype=torch.float64)
    before = gradient.clone()
    original_step = owners._optimizers["o"].step
    def observe():
        storage = [parameters["o"][name].grad.untyped_storage().data_ptr() for name in ("a", "b")]
        assert len(set(storage + [gradient.untyped_storage().data_ptr()])) == 3
        return original_step()
    monkeypatch.setattr(owners._optimizers["o"], "step", observe)
    apply_optimizer_gradients(owners, {"o": {"a": gradient, "b": gradient.view_as(gradient)}}, expected_generation=0)
    assert torch.equal(gradient, before)


def test_late_gradient_copy_failure_occurs_before_any_owner_step(monkeypatch):
    parameters = {"a": {"w": _parameter([1.0])}, "z": {"w": _parameter([2.0])}}
    owners = _prepare(parameters)
    before = snapshot_optimizer_owners(owners)
    gradients = _gradients(parameters)
    failed_storage = gradients["z"]["w"].untyped_storage().data_ptr()
    original_clone = torch.Tensor.clone
    calls = []
    failure = MemoryError("synthetic last-gradient buffer allocation")
    def fail_last_clone(tensor, *args, **kwargs):
        if tensor.untyped_storage().data_ptr() == failed_storage:
            raise failure
        return original_clone(tensor, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", fail_last_clone)
        for owner, optimizer in owners._optimizers.items():
            patch.setattr(optimizer, "step", lambda owner=owner: calls.append(owner))
        with pytest.raises(MemoryError) as caught:
            apply_optimizer_gradients(owners, gradients, expected_generation=0)
        assert caught.value is failure
    assert calls == [] and owners.status == "active" and owners.last_failure is None
    _equal_snapshots(before, snapshot_optimizer_owners(owners))
    assert all(p.grad is None for group in parameters.values() for p in group.values())


def test_scalar_mixed_dtypes_and_explicit_zero_betas_are_supported():
    parameters = {"o": {"f32": _parameter(1.0, torch.float32), "f64": _parameter(2.0)}}
    owners = _prepare(parameters, betas=(0.0, 0.0))
    gradients = {"o": {"f32": torch.tensor(0.25), "f64": torch.tensor(-0.5, dtype=torch.float64)}}
    apply_optimizer_gradients(owners, gradients, expected_generation=0)
    snapshot = snapshot_optimizer_owners(owners)
    assert snapshot.betas == (0.0, 0.0)
    assert snapshot.parameters["o"]["f32"].shape == ()
    assert snapshot.parameters["o"]["f64"].shape == ()
    assert snapshot.parameters["o"]["f32"].item() == pytest.approx(1.0 - 0.025 * 0.25 / (0.25 + 1e-8), abs=1e-5, rel=1e-5)
    assert snapshot.parameters["o"]["f64"].item() == pytest.approx(2.0 + 0.025 * 0.5 / (0.5 + 1e-8), abs=1e-12, rel=1e-12)
    assert snapshot.exp_avg["o"]["f32"].item() == 0.25
    assert snapshot.exp_avg_sq["o"]["f64"].item() == 0.25


class _DictionarySubclass(dict):
    pass


class _StringSubclass(str):
    pass


class _ParameterSubclass(torch.nn.Parameter):
    pass


class _TensorSubclass(torch.Tensor):
    pass


@pytest.mark.parametrize("field,value,error", [
    ("learning_rate", 1, TypeError), ("learning_rate", True, TypeError),
    ("learning_rate", np.float64(0.1), TypeError), ("learning_rate", 0.0, ValueError),
    ("learning_rate", -0.1, ValueError), ("learning_rate", float("nan"), ValueError),
    ("learning_rate", float("inf"), ValueError), ("epsilon", 1, TypeError),
    ("epsilon", 0.0, ValueError), ("epsilon", -1.0, ValueError),
    ("epsilon", float("nan"), ValueError), ("epsilon", float("inf"), ValueError),
    ("weight_decay", 0, TypeError), ("weight_decay", False, TypeError),
    ("weight_decay", 0.1, ValueError), ("weight_decay", -0.1, ValueError),
    ("weight_decay", float("nan"), ValueError), ("betas", [0.9, 0.999], TypeError),
    ("betas", (0.9,), ValueError), ("betas", (0.9, 0.999, 0.5), ValueError),
    ("betas", (1, 0.999), TypeError), ("betas", (0.9, False), TypeError),
    ("betas", (np.float64(0.9), 0.999), TypeError),
    ("betas", (-0.1, 0.999), ValueError), ("betas", (0.9, 1.0), ValueError),
    ("betas", (float("nan"), 0.999), ValueError), ("betas", (0.9, float("inf")), ValueError),
])
def test_invalid_explicit_profile(field, value, error):
    with pytest.raises(error):
        _prepare(**{field: value})


@pytest.mark.parametrize("case,error", [
    ("outer-subclass", TypeError), ("outer-list", TypeError), ("empty", ValueError),
    ("owner-empty", ValueError), ("owner-int", TypeError), ("owner-subclass", TypeError),
    ("inner-subclass", TypeError), ("inner-list", TypeError), ("inner-empty", ValueError),
    ("name-empty", ValueError), ("name-int", TypeError), ("name-subclass", TypeError),
])
def test_parameter_mapping_contract(case, error):
    parameter = _parameter([1.0])
    inputs = {
        "outer-subclass": _DictionarySubclass(o={"w": parameter}), "outer-list": [], "empty": {},
        "owner-empty": {"": {"w": parameter}}, "owner-int": {1: {"w": parameter}},
        "owner-subclass": {_StringSubclass("o"): {"w": parameter}},
        "inner-subclass": {"o": _DictionarySubclass(w=parameter)}, "inner-list": {"o": []},
        "inner-empty": {"o": {}}, "name-empty": {"o": {"": parameter}},
        "name-int": {"o": {1: parameter}}, "name-subclass": {"o": {_StringSubclass("w"): parameter}},
    }
    with pytest.raises(error):
        _prepare(inputs[case])


@pytest.mark.parametrize("case,error", [
    ("tensor", TypeError), ("subclass", TypeError), ("no-grad", ValueError),
    ("half", TypeError), ("meta", TypeError), ("sparse", TypeError),
    ("nan", ValueError), ("inf", ValueError), ("entry-grad", ValueError),
])
def test_parameter_value_contract(case, error):
    parameter = _parameter([1.0])
    if case == "tensor": parameter = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    elif case == "subclass": parameter = _ParameterSubclass(torch.tensor([1.0], dtype=torch.float64))
    elif case == "no-grad": parameter.requires_grad_(False)
    elif case == "half": parameter = _parameter([1.0], torch.float16)
    elif case == "meta": parameter = torch.nn.Parameter(torch.empty(1, device="meta", dtype=torch.float64))
    elif case == "sparse": parameter = torch.nn.Parameter(torch.sparse_coo_tensor([[0]], [1.0], (1,), dtype=torch.float64))
    elif case in ("nan", "inf"): parameter = _parameter([float(case)])
    elif case == "entry-grad": parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(error):
        _prepare({"o": {"w": parameter}})


@pytest.mark.parametrize("across_owners", [False, True])
@pytest.mark.parametrize("same_object", [False, True])
def test_duplicate_parameter_and_shared_nonempty_storage_rejected(across_owners, same_object):
    buffer = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    first = torch.nn.Parameter(buffer[:2])
    second = first if same_object else torch.nn.Parameter(buffer[2:])
    parameters = {"o": {"a": first}, "p": {"b": second}} if across_owners else {"o": {"a": first, "b": second}}
    with pytest.raises(ValueError):
        _prepare(parameters)


@pytest.mark.parametrize("case,error", [
    ("outer-subclass", TypeError), ("inner-subclass", TypeError),
    ("missing-owner", ValueError), ("extra-owner", ValueError),
    ("missing-name", ValueError), ("extra-name", ValueError),
    ("bad-owner-type", TypeError), ("bad-name-type", TypeError),
    ("list", TypeError), ("parameter", TypeError), ("subclass", TypeError),
    ("dtype", TypeError), ("meta", TypeError), ("sparse", TypeError),
    ("shape", ValueError), ("nan", ValueError), ("inf", ValueError),
    ("own-storage", ValueError), ("other-storage", ValueError),
])
def test_late_invalid_gradient_rejects_whole_package_before_any_step(case, error, monkeypatch):
    parameters = {"a": {"w": _parameter([1.0, 2.0])}, "z": {"w": _parameter([3.0, 4.0])}}
    owners = _prepare(parameters)
    before = snapshot_optimizer_owners(owners)
    gradients = _gradients(parameters)
    if case == "outer-subclass": gradients = _DictionarySubclass(gradients)
    elif case == "inner-subclass": gradients["z"] = _DictionarySubclass(gradients["z"])
    elif case == "missing-owner": del gradients["z"]
    elif case == "extra-owner": gradients["x"] = {"w": None}
    elif case == "missing-name": del gradients["z"]["w"]
    elif case == "extra-name": gradients["z"]["x"] = None
    elif case == "bad-owner-type": gradients[_StringSubclass("z")] = gradients.pop("z")
    elif case == "bad-name-type": gradients["z"] = {_StringSubclass("w"): None}
    else:
        gradients["z"]["w"] = {
            "list": [0.1, 0.2], "parameter": _parameter([0.1, 0.2]),
            "subclass": torch.ones(2, dtype=torch.float64).as_subclass(_TensorSubclass),
            "dtype": torch.ones(2, dtype=torch.float32),
            "meta": torch.empty(2, dtype=torch.float64, device="meta"),
            "sparse": torch.sparse_coo_tensor([[0, 1]], [0.1, 0.2], (2,), dtype=torch.float64),
            "shape": torch.zeros(1, 2, dtype=torch.float64),
            "nan": torch.tensor([0.1, float("nan")], dtype=torch.float64),
            "inf": torch.tensor([0.1, float("inf")], dtype=torch.float64),
            "own-storage": parameters["z"]["w"].detach(),
            "other-storage": parameters["a"]["w"].detach(),
        }[case]
    calls = []
    for owner, optimizer in owners._optimizers.items():
        monkeypatch.setattr(optimizer, "step", lambda owner=owner: calls.append(owner))
    with pytest.raises(error):
        apply_optimizer_gradients(owners, gradients, expected_generation=0)
    assert calls == [] and owners.status == "active" and owners.last_failure is None
    _equal_snapshots(before, snapshot_optimizer_owners(owners))
    assert all(p.grad is None for group in parameters.values() for p in group.values())


@pytest.mark.parametrize("generation,error", [(True, TypeError), (0.0, TypeError), (np.int64(0), TypeError), (-1, ValueError), (1, ValueError)])
def test_expected_generation_exact_type_and_stale_rejection(generation, error):
    owners = _prepare()
    before = snapshot_optimizer_owners(owners)
    with pytest.raises(error):
        apply_optimizer_gradients(owners, {"z": {"weight": None}}, expected_generation=generation)
    _equal_snapshots(before, snapshot_optimizer_owners(owners))
    assert apply_optimizer_gradients(owners, {"z": {"weight": None}}, expected_generation=0).generation_after == 1
    with pytest.raises(ValueError):
        apply_optimizer_gradients(owners, {"z": {"weight": None}}, expected_generation=0)


@pytest.mark.parametrize("kind", ["parameter", "exp_avg", "exp_avg_sq"])
def test_nonfinite_existing_state_rejected_before_first_owner_step(kind, monkeypatch):
    parameters = {"a": {"w": _parameter([1.0])}, "z": {"w": _parameter([2.0])}}
    owners = _prepare(parameters)
    apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=0)
    tensor = parameters["z"]["w"] if kind == "parameter" else _state(owners, "z", parameters["z"]["w"])[kind]
    with torch.no_grad(): tensor.fill_(float("nan"))
    first_before = parameters["a"]["w"].detach().clone()
    calls = []
    for owner, optimizer in owners._optimizers.items():
        monkeypatch.setattr(optimizer, "step", lambda owner=owner: calls.append(owner))
    with pytest.raises(ValueError):
        apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=1)
    assert calls == [] and torch.equal(parameters["a"]["w"], first_before)
    assert owners.status == "active" and owners.generation == 1
    assert bool(torch.isnan(tensor).all())  # Caller corruption was not rolled back.


@pytest.mark.parametrize("case,error", [
    ("mean-shape", ValueError), ("square-shape", ValueError),
    ("mean-dtype", TypeError), ("square-dtype", TypeError),
    ("missing-step", ValueError), ("missing-mean", ValueError), ("extra-state", ValueError),
])
def test_malformed_existing_last_owner_state_rejects_before_any_step(case, error, monkeypatch):
    parameters = {"a": {"w": _parameter([1.0, 2.0])}, "z": {"w": _parameter([3.0, 4.0])}}
    owners = _prepare(parameters)
    apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=0)
    state = _state(owners, "z", parameters["z"]["w"])
    if case in ("mean-shape", "square-shape"):
        key = "exp_avg" if case == "mean-shape" else "exp_avg_sq"
        state[key] = torch.ones(1, 2, dtype=torch.float64)
    elif case in ("mean-dtype", "square-dtype"):
        key = "exp_avg" if case == "mean-dtype" else "exp_avg_sq"
        state[key] = state[key].float()
    elif case == "missing-step": del state["step"]
    elif case == "missing-mean": del state["exp_avg"]
    elif case == "extra-state": state["unsupported"] = torch.tensor(0.0)
    before_values = {owner: group["w"].detach().clone() for owner, group in parameters.items()}
    calls = []
    for owner, optimizer in owners._optimizers.items():
        monkeypatch.setattr(optimizer, "step", lambda owner=owner: calls.append(owner))
    with pytest.raises(error):
        apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=1)
    assert calls == [] and owners.status == "active" and owners.generation == 1
    for owner in parameters:
        assert torch.equal(parameters[owner]["w"], before_values[owner])
        assert parameters[owner]["w"].grad is None


@pytest.mark.parametrize("kind", ["shape", "dtype", "storage", "requires_grad", "value", "grad"])
def test_external_authoritative_parameter_changes_are_not_silently_accepted(kind):
    parameter = _parameter([1.0, 2.0])
    owners = _prepare({"o": {"w": parameter}})
    if kind == "shape": parameter.data = parameter.data.reshape(1, 2)
    elif kind == "dtype": parameter.data = parameter.data.float()
    elif kind == "storage": parameter.data = parameter.data.clone()
    elif kind == "requires_grad": parameter.requires_grad_(False)
    elif kind == "value":
        with torch.no_grad(): parameter.add_(1.0)
    elif kind == "grad": parameter.grad = torch.zeros_like(parameter)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        apply_optimizer_gradients(owners, {"o": {"w": None}}, expected_generation=0)
    assert owners.generation == 0 and owners.status == "active"


@pytest.mark.parametrize("bad_step", [
    -1.0, 1.5, float("nan"), float("inf"), float(COUNTER_MAX + 1),
    "int-dtype", "rank-one", "meta", "python-int",
])
def test_invalid_existing_adam_counter_rejected_without_updating(bad_step):
    parameter = _parameter([1.0])
    owners = _prepare({"o": {"w": parameter}})
    apply_optimizer_gradients(owners, {"o": {"w": torch.ones_like(parameter)}}, expected_generation=0)
    bad = {
        "int-dtype": torch.tensor(1, dtype=torch.int64),
        "rank-one": torch.tensor([1.0], dtype=torch.float32),
        "meta": torch.empty((), device="meta"), "python-int": 1,
    }.get(bad_step) if isinstance(bad_step, str) else torch.tensor(bad_step, dtype=torch.float64)
    _state(owners, "o", parameter)["step"] = bad
    before = parameter.detach().clone()
    with pytest.raises((ValueError, TypeError)):
        apply_optimizer_gradients(owners, {"o": {"w": None}}, expected_generation=1)
    assert torch.equal(parameter, before) and owners.generation == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_counter_cap_active_rejection_none_preservation_and_last_safe_step(dtype, record_property):
    parameter = _parameter([1.0])
    owners = _prepare({"o": {"w": parameter}})
    gradients = {"o": {"w": torch.ones_like(parameter)}}
    apply_optimizer_gradients(owners, gradients, expected_generation=0)
    state = _state(owners, "o", parameter)
    state["step"] = torch.tensor(COUNTER_MAX - 1, dtype=dtype)
    receipt = apply_optimizer_gradients(owners, gradients, expected_generation=1)
    assert receipt.parameter_steps == (("o", "w", COUNTER_MAX),)
    assert state["step"].dtype is dtype
    record_property("actual_injected_adam_counter_dtype", str(state["step"].dtype))
    before = snapshot_optimizer_owners(owners)
    with pytest.raises(ValueError):
        apply_optimizer_gradients(owners, gradients, expected_generation=2)
    _equal_snapshots(before, snapshot_optimizer_owners(owners))
    receipt = apply_optimizer_gradients(owners, {"o": {"w": None}}, expected_generation=2)
    assert receipt.stepped_owner_ids == () and receipt.generation_after == 3
    assert snapshot_optimizer_owners(owners).parameter_steps == {"o": {"w": COUNTER_MAX}}


def test_last_owner_counter_cap_rejects_before_first_owner_step(monkeypatch):
    parameters = {"a": {"w": _parameter([1.0])}, "z": {"w": _parameter([2.0])}}
    owners = _prepare(parameters)
    apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=0)
    _state(owners, "z", parameters["z"]["w"])["step"] = torch.tensor(COUNTER_MAX, dtype=torch.float32)
    before = snapshot_optimizer_owners(owners)
    calls = []
    for owner, optimizer in owners._optimizers.items():
        monkeypatch.setattr(optimizer, "step", lambda owner=owner: calls.append(owner))
    with pytest.raises(ValueError):
        apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=1)
    assert calls == [] and owners.status == "active"
    _equal_snapshots(before, snapshot_optimizer_owners(owners))


@pytest.mark.parametrize("which", ["generation", "owner-version"])
def test_python_counter_overflow_rejects_whole_package_before_step(which, monkeypatch):
    parameters = {"a": {"w": _parameter([1.0])}, "z": {"w": _parameter([2.0])}}
    owners = _prepare(parameters)
    if which == "generation": owners._generation = VERSION_MAX
    else: owners._owner_versions["z"] = VERSION_MAX
    calls = []
    for owner, optimizer in owners._optimizers.items():
        monkeypatch.setattr(optimizer, "step", lambda owner=owner: calls.append(owner))
    with pytest.raises(ValueError):
        apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=owners.generation)
    assert calls == [] and owners.status == "active"
    assert torch.equal(parameters["a"]["w"], torch.tensor([1.0], dtype=torch.float64))


def test_generation_and_owner_version_last_safe_update_then_none_at_owner_cap():
    parameter = _parameter([1.0])
    owners = _prepare({"o": {"w": parameter}})
    owners._generation = VERSION_MAX - 2
    owners._owner_versions["o"] = VERSION_MAX - 1
    receipt = apply_optimizer_gradients(owners, {"o": {"w": torch.ones_like(parameter)}}, expected_generation=VERSION_MAX - 2)
    assert receipt.owner_update_versions == (("o", VERSION_MAX),)
    receipt = apply_optimizer_gradients(owners, {"o": {"w": None}}, expected_generation=VERSION_MAX - 1)
    assert receipt.generation_after == VERSION_MAX and receipt.stepped_owner_ids == ()


def test_second_owner_exception_retains_partial_state_and_breaks_handle(monkeypatch):
    parameters = {"a": {"w": _parameter([1.0])}, "m": {"w": _parameter([2.0])}, "z": {"w": _parameter([3.0])}}
    owners = _prepare(parameters)
    original_step = owners._optimizers["m"].step
    failure = MemoryError("synthetic second owner failure after physical update")
    def fail_after_update():
        original_step()
        raise failure
    monkeypatch.setattr(owners._optimizers["m"], "step", fail_after_update)
    def forbidden_later_step():
        raise AssertionError("later owner must not run after a partial failure")
    monkeypatch.setattr(owners._optimizers["z"], "step", forbidden_later_step)
    with pytest.raises(OptimizerStepError) as caught:
        apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=0)
    assert caught.value.__cause__ is failure
    assert owners.status == "broken" and owners.generation == 0
    assert owners.last_failure == (("a", "m"), ("a",), "m", "MemoryError", str(failure))
    assert parameters["a"]["w"].item() < 1.0 and parameters["m"]["w"].item() < 2.0
    assert parameters["z"]["w"].item() == 3.0
    assert _state(owners, "a", parameters["a"]["w"])["step"].item() == 1.0
    assert _state(owners, "m", parameters["m"]["w"])["step"].item() == 1.0
    assert all(p.grad is None for group in parameters.values() for p in group.values())
    with pytest.raises(RuntimeError): snapshot_optimizer_owners(owners)
    with pytest.raises(RuntimeError): apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=0)
    diagnostic = owners.last_failure
    close_optimizer_owners(owners)
    assert owners.status == "closed" and owners.last_failure == diagnostic
    close_optimizer_owners(owners)


def test_actual_finite_gradient_overflow_records_returned_owner_and_partial_state():
    parameters = {"a": {"w": _parameter([1.0], torch.float32)}, "z": {"w": _parameter([2.0], torch.float32)}}
    owners = _prepare(parameters)
    gradients = {"a": {"w": torch.tensor([0.2])}, "z": {"w": torch.tensor([1e30])}}
    with pytest.raises(OptimizerStepError) as caught:
        apply_optimizer_gradients(owners, gradients, expected_generation=0)
    assert isinstance(caught.value.__cause__, ValueError)
    assert owners.status == "broken" and owners.generation == 0
    assert owners.last_failure[:3] == (("a", "z"), ("a", "z"), "z")
    assert parameters["a"]["w"].item() < 1.0
    assert bool(torch.isinf(_state(owners, "z", parameters["z"]["w"])["exp_avg_sq"]).all())
    assert all(p.grad is None for group in parameters.values() for p in group.values())


@pytest.mark.parametrize("kind", ["active-not-incremented", "none-incremented"])
def test_post_step_counter_contract_violation_breaks_handle(kind, monkeypatch):
    parameters = {"o": {"active": _parameter([1.0]), "none": _parameter([2.0])}}
    owners = _prepare(parameters)
    apply_optimizer_gradients(owners, _gradients(parameters), expected_generation=0)
    original_step = owners._optimizers["o"].step
    def corrupt():
        original_step()
        name, increment = ("active", -1) if kind == "active-not-incremented" else ("none", 1)
        _state(owners, "o", parameters["o"][name])["step"].add_(increment)
    monkeypatch.setattr(owners._optimizers["o"], "step", corrupt)
    with pytest.raises(OptimizerStepError):
        apply_optimizer_gradients(owners, {"o": {"active": torch.ones(1, dtype=torch.float64), "none": None}}, expected_generation=1)
    assert owners.status == "broken" and owners.generation == 1
    assert owners.last_failure[:3] == (("o",), ("o",), "o")


def test_preflight_error_preserves_existing_grad_and_close_preserves_caller_state():
    parameter = _parameter([1.0])
    owners = _prepare({"o": {"w": parameter}})
    existing_grad = torch.tensor([0.5], dtype=torch.float64)
    parameter.grad = existing_grad
    with pytest.raises(ValueError):
        apply_optimizer_gradients(owners, {"o": {"w": None}}, expected_generation=0)
    assert parameter.grad is existing_grad and owners.generation == 0
    storage, version, value = parameter.untyped_storage().data_ptr(), parameter._version, parameter.detach().clone()
    close_optimizer_owners(owners)
    close_optimizer_owners(owners)
    assert owners.status == "closed" and owners.generation == 0 and owners.last_failure is None
    assert parameter.grad is existing_grad and torch.equal(parameter, value)
    assert parameter.untyped_storage().data_ptr() == storage and parameter._version == version
    assert not owners._parameters and not owners._optimizers
    with pytest.raises(RuntimeError): snapshot_optimizer_owners(owners)
    with pytest.raises(RuntimeError): apply_optimizer_gradients(owners, {"o": {"w": None}}, expected_generation=0)


@pytest.mark.parametrize("function", [snapshot_optimizer_owners, close_optimizer_owners])
def test_nonhandle_rejected(function):
    with pytest.raises(TypeError): function(object())


def test_prepare_apply_snapshot_close_do_not_advance_rng_or_call_forward_backward(monkeypatch):
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer-owner runtime must not invoke model or autograd execution")
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    parameter = _parameter([1.0, -1.0])
    owners = _prepare({"any opaque owner": {"not a role": parameter}})
    apply_optimizer_gradients(owners, {"any opaque owner": {"not a role": torch.tensor([0.3, -0.2], dtype=torch.float64)}}, expected_generation=0)
    snapshot_optimizer_owners(owners)
    close_optimizer_owners(owners)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0] and np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
