from __future__ import annotations

import random
from itertools import product
from math import exp, fsum, sqrt

import numpy as np
import pytest
import torch

from src.methods.doi4 import compatible_readout
from src.methods.doi4.binary_fields import encode_binary_field_units


# These fixtures and tolerances were persisted in B's evidence directory before
# any test execution. Expected math below never calls a production helper.
FD_STEP = 1e-6
FD_ATOL = 2e-7
FD_RTOL = 1e-6


def _inputs(dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    return [
        torch.tensor([[0, 1, 0, -1], [2, 0, 1, 2], [1, -1, 0, 2]]),
        torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]]),
        torch.tensor(
            [[True, True, True, False], [True, False, True, False],
             [False, False, True, True]]
        ),
        torch.tensor(
            [[0.25, -0.4, 0.6], [-0.5, 0.3, 0.15],
             [0.8, -0.2, -0.35], [1.2, -0.9, 0.05]], dtype=dtype
        ),
        torch.tensor([[0.1, 0.2, -0.15], [-0.3, 0.5, 0.4]], dtype=dtype),
    ]


def _arrays(args: list[torch.Tensor]) -> list[np.ndarray]:
    return [arg.detach().numpy() for arg in args]


def _numpy_forward(
    ids: np.ndarray,
    codes: np.ndarray,
    mask: np.ndarray,
    fields: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    result = np.zeros((*ids.shape, fields.shape[1]), dtype=fields.dtype)
    for e, p in product(range(ids.shape[0]), range(ids.shape[1])):
        if mask[e, p]:
            for d in range(fields.shape[1]):
                result[e, p, d] = fields[ids[e, p], d] + values[codes[e, p], d]
    return result


def _numpy_scatter(
    ids: np.ndarray,
    codes: np.ndarray,
    mask: np.ndarray,
    fields: np.ndarray,
    values: np.ndarray,
    upstream: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    field_grad, value_grad = np.zeros_like(fields), np.zeros_like(values)
    for e, p in product(range(ids.shape[0]), range(ids.shape[1])):
        if mask[e, p]:
            for d in range(fields.shape[1]):
                field_grad[ids[e, p], d] += upstream[e, p, d]
                value_grad[codes[e, p], d] += upstream[e, p, d]
    return field_grad, value_grad


def _upstream(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(int(np.prod(shape)), dtype=dtype).reshape(shape) / 17 - 0.8


def _boundary_inputs(name: str, dtype: torch.dtype) -> list[torch.Tensor]:
    args = _inputs(dtype)
    if name == "singleton":
        args[:3] = [
            torch.tensor([[1], [0]]), torch.tensor([[0], [1]]),
            torch.tensor([[True], [False]]),
        ]
    elif name in ("empty_entities", "empty_units"):
        shape = (0, 4) if name == "empty_entities" else (2, 0)
        args[:3] = [
            torch.empty(shape, dtype=torch.int64),
            torch.empty(shape, dtype=torch.int64),
            torch.empty(shape, dtype=torch.bool),
        ]
    elif name == "empty_field_table":
        args[:3] = [
            torch.full((2, 2), -1), torch.tensor([[0, 1], [1, 0]]),
            torch.zeros((2, 2), dtype=torch.bool),
        ]
        args[3] = torch.empty((0, 3), dtype=dtype)
    elif name == "all_masked":
        args[:3] = [
            torch.tensor([[-1, 1], [2, -1]]), torch.tensor([[0, 1], [1, 0]]),
            torch.zeros((2, 2), dtype=torch.bool),
        ]
    return args


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "case", ["mixed", "singleton", "empty_entities", "empty_units",
             "empty_field_table", "all_masked"]
)
def test_forward_and_scatter_gradient_match_independent_oracle(
    case: str, dtype: torch.dtype, request: pytest.FixtureRequest
) -> None:
    args = _boundary_inputs(case, dtype)
    args[3].requires_grad_()
    args[4].requires_grad_()
    arrays = _arrays(args)
    expected = _numpy_forward(*arrays)
    result = encode_binary_field_units(*args)
    assert result.shape == expected.shape
    assert result.dtype == dtype and result.device.type == "cpu"
    assert result.layout == torch.strided
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    np.testing.assert_allclose(result.detach().numpy(), expected, atol=tolerance, rtol=tolerance)
    forward_error = np.abs(result.detach().numpy() - expected)
    request.node.user_properties.append((
        "forward_max_absolute_error", float(forward_error.max()) if forward_error.size else 0.0
    ))
    assert torch.count_nonzero(result[~args[2]]) == 0
    upstream = _upstream(tuple(result.shape), dtype)
    gradients = torch.autograd.grad((result * upstream).sum(), args[3:])
    expected_gradients = _numpy_scatter(*arrays, upstream.numpy())
    for name, actual, oracle in zip(("field", "value"), gradients, expected_gradients):
        np.testing.assert_allclose(actual.numpy(), oracle, atol=tolerance, rtol=tolerance)
        error = np.abs(actual.numpy() - oracle)
        request.node.user_properties.append((
            f"{name}_scatter_max_absolute_error", float(error.max()) if error.size else 0.0
        ))
    # The fixture includes a field row absent from every active unit.
    if case == "mixed":
        assert torch.count_nonzero(gradients[0][3]) == 0


def test_known_zero_is_a_valid_observation_and_has_its_own_value_row() -> None:
    args = _inputs()
    args[:3] = [torch.tensor([[0, 0]]), torch.tensor([[0, 1]]), torch.ones((1, 2), dtype=torch.bool)]
    result = encode_binary_field_units(*args)
    np.testing.assert_allclose(result.detach().numpy(), _numpy_forward(*_arrays(args)), atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(result[0, 0]) > 0
    torch.testing.assert_close(result[0, 1] - result[0, 0], args[4][1] - args[4][0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_latent_coordinate_is_valid(dtype: torch.dtype) -> None:
    args = _inputs(dtype)
    args[3] = args[3][:, :1].detach().requires_grad_()
    args[4] = args[4][:, :1].detach().requires_grad_()
    result = encode_binary_field_units(*args)
    expected = _numpy_forward(*_arrays(args))
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    assert result.shape == (3, 4, 1)
    np.testing.assert_allclose(result.detach().numpy(), expected, atol=tolerance, rtol=tolerance)
    gradients = torch.autograd.grad(result.sum(), args[3:])
    expected_gradients = _numpy_scatter(*_arrays(args), np.ones_like(expected))
    for actual, oracle in zip(gradients, expected_gradients):
        np.testing.assert_array_equal(actual.numpy(), oracle)


def _smooth_numpy_loss(args: list[np.ndarray], upstream: np.ndarray) -> float:
    encoded = _numpy_forward(*args)
    return float(np.sum(upstream * np.sin(encoded) + 0.07 * encoded**2))


def binary_fields_finite_difference_report() -> dict[str, object]:
    """Execute the frozen new-module gradient check and return measured errors."""
    args = _inputs()
    args[3].requires_grad_()
    args[4].requires_grad_()
    arrays = _arrays(args)
    expected = _numpy_forward(*arrays)
    upstream = _upstream(tuple(expected.shape), torch.float64)
    result = encode_binary_field_units(*args)
    loss = (upstream * result.sin() + 0.07 * result.square()).sum()
    actual = torch.autograd.grad(loss, args[3:])
    analytic = _numpy_scatter(
        *arrays, upstream.numpy() * np.cos(expected) + 0.14 * expected
    )
    report: dict[str, object] = {"step": FD_STEP, "atol": FD_ATOL, "rtol": FD_RTOL}
    for index, name, gradient, analytic_gradient in zip(
        (3, 4), ("field_embeddings", "value_embeddings"), actual, analytic
    ):
        difference = np.empty_like(arrays[index])
        for coordinate in np.ndindex(arrays[index].shape):
            losses = []
            for sign in (1, -1):
                perturbed = [array.copy() for array in arrays]
                perturbed[index][coordinate] += sign * FD_STEP
                losses.append(_smooth_numpy_loss(perturbed, upstream.numpy()))
            difference[coordinate] = (losses[0] - losses[1]) / (2 * FD_STEP)
        np.testing.assert_allclose(gradient.numpy(), analytic_gradient, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(gradient.numpy(), difference, atol=FD_ATOL, rtol=FD_RTOL)
        error = np.abs(gradient.numpy() - difference)
        report[name] = {
            "max_absolute_error": float(error.max()),
            "max_tolerance_ratio": float((error / (FD_ATOL + FD_RTOL * np.abs(difference))).max()),
        }
    return report


def test_both_table_gradients_match_frozen_independent_central_differences(
    request: pytest.FixtureRequest,
) -> None:
    report = binary_fields_finite_difference_report()
    for name in ("field_embeddings", "value_embeddings"):
        assert np.isfinite(report[name]["max_absolute_error"])
        assert report[name]["max_tolerance_ratio"] <= 1
        for metric, value in report[name].items():
            request.node.user_properties.append((f"{name}_central_difference_{metric}", value))


@pytest.mark.parametrize("index", range(5))
def test_rejects_non_tensor_without_numpy_conversion(index: int) -> None:
    args = _inputs()
    args[index] = args[index].numpy()
    with pytest.raises(TypeError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize(
    "index,dtype", [
        (0, torch.int32), (0, torch.float64), (0, torch.bool),
        (1, torch.int32), (1, torch.uint8), (1, torch.float32),
        (2, torch.int64), (2, torch.float32),
        (3, torch.float16), (3, torch.bfloat16), (3, torch.int64),
        (4, torch.float16), (4, torch.complex64), (4, torch.int64),
    ]
)
def test_rejects_wrong_dtype(index: int, dtype: torch.dtype) -> None:
    args = _inputs()
    args[index] = args[index].to(dtype)
    with pytest.raises(TypeError):
        encode_binary_field_units(*args)


def test_rejects_mixed_table_dtypes_without_promotion() -> None:
    args = _inputs()
    args[3] = args[3].float()
    with pytest.raises(TypeError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize("index", range(5))
def test_rejects_non_cpu_device_before_content_access(index: int) -> None:
    args = _inputs()
    args[index] = args[index].to(device="meta")
    with pytest.raises(TypeError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize("index", range(5))
def test_rejects_sparse_layout(index: int) -> None:
    args = _inputs()
    args[index] = args[index].to_sparse()
    with pytest.raises(TypeError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize("index", range(5))
def test_rejects_wrong_rank(index: int) -> None:
    args = _inputs()
    args[index] = args[index].reshape(-1)
    with pytest.raises(ValueError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize(
    "case", ["ids_width", "codes_width", "mask_rows", "field_zero_width",
             "value_zero_width", "value_one_row", "value_three_rows", "value_width"]
)
def test_rejects_shape_mismatches_and_zero_latent_width(case: str) -> None:
    args = _inputs()
    if case == "ids_width":
        args[0] = args[0][:, :-1]
    elif case == "codes_width":
        args[1] = args[1][:, :-1]
    elif case == "mask_rows":
        args[2] = args[2][:-1]
    elif case == "field_zero_width":
        args[3] = args[3][:, :0]
    elif case == "value_zero_width":
        args[4] = args[4][:, :0]
    elif case == "value_one_row":
        args[4] = args[4][:1]
    elif case == "value_three_rows":
        args[4] = torch.cat((args[4], args[4][:1]))
    else:
        args[4] = args[4][:, :-1]
    with pytest.raises(ValueError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("bad_id", [-2, 4, torch.iinfo(torch.int64).max])
def test_rejects_out_of_range_ids_even_when_masked(valid: bool, bad_id: int) -> None:
    args = _inputs()
    args[0][0, 0] = bad_id
    args[2][0, 0] = valid
    with pytest.raises(ValueError):
        encode_binary_field_units(*args)


def test_minus_one_padding_cannot_be_valid() -> None:
    args = _inputs()
    args[0][0, 0] = -1
    with pytest.raises(ValueError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("bad_code", [-1, 2, torch.iinfo(torch.int64).max])
def test_rejects_nonbinary_values_even_when_masked(valid: bool, bad_code: int) -> None:
    args = _inputs()
    args[1][0, 0] = bad_code
    args[2][0, 0] = valid
    with pytest.raises(ValueError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize("case", ["zero_id_masked", "padding_active", "illegal_masked_code"])
def test_empty_field_table_still_validates_payload(case: str) -> None:
    args = _boundary_inputs("empty_field_table", torch.float64)
    if case == "zero_id_masked":
        args[0][0, 0] = 0
    elif case == "padding_active":
        args[2][0, 0] = True
    else:
        args[1][0, 0] = 2
    with pytest.raises(ValueError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize("index", [3, 4])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_rejects_whole_table_nonfiniteness_even_when_all_units_masked(index: int, bad: float) -> None:
    args = _inputs()
    args[2].fill_(False)
    args[index][-1, -1] = bad
    with pytest.raises(ValueError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize("empty", ["empty_entities", "empty_units"])
@pytest.mark.parametrize(
    "case,error", [
        ("field_nan", ValueError), ("value_inf", ValueError),
        ("value_shape", ValueError), ("zero_width", ValueError),
        ("payload_shape", ValueError), ("ids_dtype", TypeError),
        ("codes_dtype", TypeError), ("mask_dtype", TypeError),
        ("field_dtype", TypeError), ("value_dtype", TypeError),
        ("device", TypeError), ("layout", TypeError),
    ]
)
def test_empty_payloads_cannot_bypass_validation(empty: str, case: str, error: type[Exception]) -> None:
    args = _boundary_inputs(empty, torch.float64)
    if case == "field_nan":
        args[3][-1, -1] = float("nan")
    elif case == "value_inf":
        args[4][-1, -1] = float("inf")
    elif case == "value_shape":
        args[4] = args[4][:1]
    elif case == "zero_width":
        args[3] = args[3][:, :0]
        args[4] = args[4][:, :0]
    elif case == "payload_shape":
        shape = (0, 1) if empty == "empty_entities" else (3, 0)
        args[1] = torch.empty(shape, dtype=torch.int64)
    elif case == "ids_dtype":
        args[0] = args[0].to(torch.int32)
    elif case == "codes_dtype":
        args[1] = args[1].to(torch.float64)
    elif case == "mask_dtype":
        args[2] = args[2].to(torch.int64)
    elif case == "field_dtype":
        args[3] = args[3].to(torch.float16)
    elif case == "value_dtype":
        args[4] = args[4].to(torch.float16)
    elif case == "device":
        args[2] = args[2].to(device="meta")
    else:
        args[3] = args[3].to_sparse()
    with pytest.raises(error):
        encode_binary_field_units(*args)


def test_type_validation_precedes_illegal_content() -> None:
    args = _inputs()
    args[0][0, 0] = -2
    args[4] = args[4].to(torch.float16)
    with pytest.raises(TypeError):
        encode_binary_field_units(*args)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "case", ["all_masked", "empty_entities", "empty_units", "empty_field_table", "inactive_rows", "active_cancellation"]
)
def test_large_finite_unused_tables_avoid_unnecessary_overflow_and_keep_gradients(
    case: str, dtype: torch.dtype
) -> None:
    boundary = case if case not in ("inactive_rows", "active_cancellation") else "mixed"
    args = _boundary_inputs(boundary, dtype)
    large = torch.finfo(dtype).max * 0.75
    args[3].fill_(large)
    args[4].fill_(large)
    if case == "inactive_rows":
        args[:3] = [torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 1]]), torch.tensor([[True, False, False]])]
        args[3][0] = torch.tensor([0.25, -0.4, 0.6], dtype=dtype)
        args[4][0] = torch.tensor([0.1, 0.2, -0.15], dtype=dtype)
    elif case == "active_cancellation":
        args[:3] = [torch.tensor([[0]]), torch.tensor([[1]]), torch.tensor([[True]])]
        args[4].fill_(-large)
    args[3].requires_grad_()
    args[4].requires_grad_()
    result = encode_binary_field_units(*args)
    expected = _numpy_forward(*_arrays(args))
    assert torch.isfinite(result).all()
    np.testing.assert_allclose(result.detach().numpy(), expected, atol=1e-5, rtol=1e-5)
    gradients = torch.autograd.grad(result.sum(), args[3:])
    expected_gradients = _numpy_scatter(*_arrays(args), np.ones_like(expected))
    for actual, oracle in zip(gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        np.testing.assert_array_equal(actual.numpy(), oracle)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_active_addition_overflow_is_rejected_without_promotion(dtype: torch.dtype) -> None:
    large = torch.finfo(dtype).max * 0.75
    args = [
        torch.tensor([[0]]), torch.tensor([[0]]), torch.tensor([[True]]),
        torch.full((1, 2), large, dtype=dtype), torch.full((2, 2), large, dtype=dtype),
    ]
    with pytest.raises(ValueError):
        encode_binary_field_units(*args)


def _snapshot(args: list[torch.Tensor]) -> list[tuple]:
    return [
        (arg.detach().clone(), arg.data_ptr(), arg.untyped_storage().data_ptr(),
         arg._version, arg.stride(), arg.storage_offset())
        for arg in args
    ]


def _assert_snapshot(args: list[torch.Tensor], before: list[tuple]) -> None:
    for arg, (value, pointer, storage, version, stride, offset) in zip(args, before):
        assert torch.equal(arg.detach(), value)
        assert (arg.data_ptr(), arg.untyped_storage().data_ptr(), arg._version,
                arg.stride(), arg.storage_offset()) == (pointer, storage, version, stride, offset)


def test_function_preserves_input_values_storage_versions_rng_and_returns_new_storage() -> None:
    args = _inputs()
    args[3].requires_grad_()
    args[4].requires_grad_()
    before = _snapshot(args)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    result = encode_binary_field_units(*args)
    torch.autograd.grad((result * _upstream(tuple(result.shape), result.dtype)).sum(), args[3:])
    assert result.untyped_storage().data_ptr() not in {arg.untyped_storage().data_ptr() for arg in args}
    _assert_snapshot(args, before)
    with torch.no_grad():
        result.fill_(123)
    _assert_snapshot(args, before)
    assert random.getstate() == python_state
    after_numpy_state = np.random.get_state()
    assert numpy_state[0] == after_numpy_state[0]
    np.testing.assert_array_equal(numpy_state[1], after_numpy_state[1])
    assert numpy_state[2:] == after_numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_noncontiguous_nonleaf_tables_and_payloads_keep_upstream_gradients() -> None:
    args = _inputs()
    field_leaf = args[3].T.contiguous().requires_grad_()
    value_leaf = args[4].T.contiguous().requires_grad_()
    args[3], args[4] = (field_leaf * 1.7).T, (value_leaf * -0.6).T
    for index in range(3):
        args[index] = torch.stack((args[index], args[index]), dim=-1)[..., 0]
    assert all(not arg.is_contiguous() for arg in args)
    before = _snapshot(args)
    result = encode_binary_field_units(*args)
    upstream = _upstream(tuple(result.shape), result.dtype)
    gradients = torch.autograd.grad((result * upstream).sum(), (field_leaf, value_leaf))
    expected = _numpy_forward(*_arrays(args))
    expected_gradients = _numpy_scatter(*_arrays(args), upstream.numpy())
    np.testing.assert_allclose(result.detach().numpy(), expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(gradients[0].numpy(), expected_gradients[0].T * 1.7, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(gradients[1].numpy(), expected_gradients[1].T * -0.6, atol=1e-12, rtol=1e-12)
    _assert_snapshot(args, before)


def test_caller_owned_parameter_tables_are_supported() -> None:
    args = _inputs()
    args[3], args[4] = torch.nn.Parameter(args[3]), torch.nn.Parameter(args[4])
    result = encode_binary_field_units(*args)
    upstream = _upstream(tuple(result.shape), result.dtype)
    result.backward(upstream)
    expected_gradients = _numpy_scatter(*_arrays(args), upstream.numpy())
    for parameter, expected in zip(args[3:], expected_gradients):
        assert parameter.grad is not None
        np.testing.assert_allclose(parameter.grad.numpy(), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_autocast_does_not_change_caller_dtype(dtype: torch.dtype) -> None:
    args = _inputs(dtype)
    args[3].requires_grad_()
    args[4].requires_grad_()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = encode_binary_field_units(*args)
        gradients = torch.autograd.grad(result.sum(), args[3:])
    assert result.dtype == dtype
    assert all(gradient.dtype == dtype for gradient in gradients)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    np.testing.assert_allclose(result.detach().numpy(), _numpy_forward(*_arrays(args)), atol=tolerance, rtol=tolerance)


def test_joint_field_id_and_table_relabeling_preserves_output_and_relabels_gradient() -> None:
    args = _inputs()
    args[3].requires_grad_()
    args[4].requires_grad_()
    result = encode_binary_field_units(*args)
    upstream = _upstream(tuple(result.shape), result.dtype)
    gradients = torch.autograd.grad((result * upstream).sum(), args[3:])
    old_to_new = torch.tensor([2, 0, 3, 1])
    renamed = [arg.detach().clone() for arg in args]
    known = renamed[0] >= 0
    renamed[0][known] = old_to_new[renamed[0][known]]
    renamed[3][old_to_new] = args[3].detach()
    renamed[3].requires_grad_()
    renamed[4].requires_grad_()
    renamed_result = encode_binary_field_units(*renamed)
    renamed_gradients = torch.autograd.grad((renamed_result * upstream).sum(), renamed[3:])
    torch.testing.assert_close(renamed_result, result, atol=0, rtol=0)
    torch.testing.assert_close(renamed_gradients[0][old_to_new], gradients[0], atol=0, rtol=0)
    torch.testing.assert_close(renamed_gradients[1], gradients[1], atol=0, rtol=0)


def test_unit_permutation_preserves_table_gradients_and_permutates_only_tokens() -> None:
    args = _inputs()
    args[3].requires_grad_()
    args[4].requires_grad_()
    upstream = _upstream((3, 4, 3), torch.float64)
    result = encode_binary_field_units(*args)
    gradients = torch.autograd.grad((result * upstream).sum(), args[3:])
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = [arg[:, permutation] for arg in args[:3]]
    permuted += [arg.detach().clone().requires_grad_() for arg in args[3:]]
    permuted_result = encode_binary_field_units(*permuted)
    permuted_gradients = torch.autograd.grad((permuted_result * upstream[:, permutation]).sum(), permuted[3:])
    torch.testing.assert_close(permuted_result, result[:, permutation], atol=0, rtol=0)
    for actual, expected in zip(permuted_gradients, gradients):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def _composition_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = [
        [[0.2, -0.4, 0.7], [-0.6, 0.3, 0.1], [0.8, -0.1, -0.5]],
        [[0.15, -0.05, 0.2], [-0.2, 0.4, -0.1]],
        [[0.6, -0.2], [-0.3, 0.8], [0.1, 0.5]],
        [[0.7, -0.4, 0.2], [-0.2, 0.6, 0.9]],
        [[0.4, 0.8, -0.3], [-0.5, 0.2, 0.7]],
    ]
    return tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in values)


def _composition_upstream() -> torch.Tensor:
    return torch.tensor([[[0.7, -0.4], [-0.2, 0.9], [0.6, 0.3]]], dtype=torch.float64)


def test_new_encoder_singleton_readout_has_zero_query_key_gradients_and_connected_table_paths(
    request: pytest.FixtureRequest,
) -> None:
    fields, values, queries, key, value = _composition_tables()
    args = [torch.tensor([[0]]), torch.tensor([[0]]), torch.tensor([[True]]), fields, values]
    encoded = encode_binary_field_units(*args)
    result, available, attention = compatible_readout(encoded, args[2], queries, key, value)
    upstream = _composition_upstream()
    gradients = torch.autograd.grad((result * upstream).sum(), (fields, values, queries, key, value))
    expected_tokens = _numpy_forward(*_arrays(args))
    projected = expected_tokens[0, 0] @ value.detach().numpy().T
    np.testing.assert_allclose(result.detach().numpy(), np.tile(projected, (1, 3, 1)), atol=1e-12, rtol=1e-12)
    assert available.tolist() == [True]
    assert torch.equal(attention, torch.ones_like(attention))
    assert torch.equal(result[:, 0], result[:, 1]) and torch.equal(result[:, 1], result[:, 2])
    assert torch.count_nonzero(gradients[2]) == 0
    assert torch.count_nonzero(gradients[3]) == 0
    request.node.user_properties.extend([
        ("query_gradient_norm", float(torch.linalg.vector_norm(gradients[2]))),
        ("key_gradient_norm", float(torch.linalg.vector_norm(gradients[3]))),
        ("slot0_slot1_output_norm", float(torch.linalg.vector_norm(result[:, 0] - result[:, 1]).detach())),
        ("attention", str(attention.detach().tolist())),
    ])
    dx = (upstream.numpy().sum(axis=1) @ value.detach().numpy()).reshape(1, 1, 3)
    expected_gradients = _numpy_scatter(*_arrays(args), dx)
    for actual, expected in zip(gradients[:2], expected_gradients):
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=1e-12)
        assert torch.count_nonzero(actual) > 0


def _new_composition_oracle(
    encoded: np.ndarray, mask: np.ndarray, queries: np.ndarray,
    key: np.ndarray, value: np.ndarray, upstream: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """New fixture only: scalar softmax and its analytic chain to input units."""
    entities, units, width = encoded.shape
    slots, compatible_width = queries.shape
    outputs = np.zeros((entities, slots, compatible_width), dtype=np.float64)
    weights = np.zeros((entities, slots, units), dtype=np.float64)
    token_grad = np.zeros((entities, units, width), dtype=np.float64)
    query_grad = np.zeros_like(queries)
    scale = sqrt(compatible_width)
    for e in range(entities):
        active = [p for p in range(units) if mask[e, p]]
        if not active:
            continue
        keys = {p: key @ encoded[e, p] for p in active}
        projected_values = {p: value @ encoded[e, p] for p in active}
        for k in range(slots):
            logits = [fsum(queries[k, h] * keys[p][h] for h in range(compatible_width)) / scale for p in active]
            numerators = [exp(logit - max(logits)) for logit in logits]
            for p, numerator in zip(active, numerators):
                weights[e, k, p] = numerator / fsum(numerators)
            for h in range(compatible_width):
                outputs[e, k, h] = fsum(weights[e, k, p] * projected_values[p][h] for p in active)
            for p in active:
                logit_grad = weights[e, k, p] * fsum(
                    upstream[e, k, h] * (projected_values[p][h] - outputs[e, k, h])
                    for h in range(compatible_width)
                )
                query_grad[k] += logit_grad * keys[p] / scale
                token_grad[e, p] += (
                    weights[e, k, p] * (upstream[e, k] @ value)
                    + logit_grad * (queries[k] @ key) / scale
                )
    return outputs, weights, token_grad, query_grad


def test_new_two_unit_readout_composition_matches_independent_chain_and_has_query_gradient(
    request: pytest.FixtureRequest,
) -> None:
    fields, values, queries, key, value = _composition_tables()
    args = [torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 0]]), torch.tensor([[True, True, False]]), fields, values]
    encoded = encode_binary_field_units(*args)
    result, available, attention = compatible_readout(encoded, args[2], queries, key, value)
    upstream = _composition_upstream()
    gradients = torch.autograd.grad((result * upstream).sum(), (fields, values, queries))
    expected = _new_composition_oracle(
        _numpy_forward(*_arrays(args)), args[2].numpy(), queries.detach().numpy(),
        key.detach().numpy(), value.detach().numpy(), upstream.numpy(),
    )
    np.testing.assert_allclose(result.detach().numpy(), expected[0], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(attention.detach().numpy(), expected[1], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(gradients[2].numpy(), expected[3], atol=1e-12, rtol=1e-12)
    expected_gradients = _numpy_scatter(*_arrays(args), expected[2])
    for actual, oracle in zip(gradients[:2], expected_gradients):
        np.testing.assert_allclose(actual.numpy(), oracle, atol=1e-12, rtol=1e-12)
        assert torch.count_nonzero(actual) > 0
    assert available.tolist() == [True]
    assert torch.linalg.vector_norm(gradients[2]) > 1e-4
    assert torch.linalg.vector_norm(result[:, 0] - result[:, 1]) > 1e-4
    assert torch.count_nonzero(gradients[0][2]) == 0
    request.node.user_properties.extend([
        ("query_gradient_norm", float(torch.linalg.vector_norm(gradients[2]))),
        ("field_gradient_norm", float(torch.linalg.vector_norm(gradients[0]))),
        ("value_gradient_norm", float(torch.linalg.vector_norm(gradients[1]))),
        ("slot0_slot1_output_norm", float(torch.linalg.vector_norm(result[:, 0] - result[:, 1]).detach())),
        ("attention", str(attention.detach().tolist())),
    ])


def test_new_encoder_p0_composition_preserves_absence_and_connected_zero_gradients() -> None:
    fields, values, queries, key, value = _composition_tables()
    ids = torch.empty((2, 0), dtype=torch.int64)
    codes = torch.empty((2, 0), dtype=torch.int64)
    mask = torch.empty((2, 0), dtype=torch.bool)
    encoded = encode_binary_field_units(ids, codes, mask, fields, values)
    result, available, attention = compatible_readout(encoded, mask, queries, key, value)
    assert result.shape == (2, 3, 2) and attention.shape == (2, 3, 0)
    assert available.tolist() == [False, False]
    assert torch.equal(result, torch.zeros_like(result))
    gradients = torch.autograd.grad(result.sum(), (fields, values, queries, key, value))
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)
