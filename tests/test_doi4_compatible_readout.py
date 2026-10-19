from __future__ import annotations

from itertools import product
from math import exp, fsum, sqrt

import numpy as np
import pytest
import torch

from src.methods.doi4 import compatible_readout


def _inputs(dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    # Rectangular maps and unequal d/h/P/K catch transposition/broadcast errors.
    return [
        torch.tensor(
            [
                [[0.2, -0.7, 1.1], [4.0, -2.0, 3.0], [1.3, 0.4, -0.2]],
                [[2.0, 3.0, 1.0], [-4.0, 2.0, 0.1], [0.5, -0.6, 1.0]],
                [[-0.3, 0.8, 0.6], [0.9, -0.2, 0.5], [2.0, 1.0, -1.0]],
            ],
            dtype=dtype,
        ),
        torch.tensor([[True, False, True], [False, False, False], [True, True, False]]),
        torch.tensor([[0.6, -0.2], [-0.5, 0.9]], dtype=dtype),
        torch.tensor([[0.7, -0.4, 0.2], [-0.1, 0.3, 0.8]], dtype=dtype),
        torch.tensor([[-0.2, 0.9, 0.5], [0.8, -0.6, 0.3]], dtype=dtype),
    ]


def _scalar_oracle(
    x: torch.Tensor,
    mask: torch.Tensor,
    b: torch.Tensor,
    key_map: torch.Tensor,
    value_map: torch.Tensor,
) -> tuple[list, list]:
    """Independent list/scalar calculation; no Torch attention/matrix operations."""
    xs, masks, queries = x.tolist(), mask.tolist(), b.tolist()
    keys, values = key_map.tolist(), value_map.tolist()
    entities, units, local_dim = x.shape
    slots, width = b.shape
    h = [[[0.0] * width for _ in range(slots)] for _ in range(entities)]
    a = [[[0.0] * units for _ in range(slots)] for _ in range(entities)]
    for e in range(entities):
        valid = [p for p in range(units) if masks[e][p]]
        if not valid:
            continue
        projected_keys = {
            p: [fsum(keys[j][q] * xs[e][p][q] for q in range(local_dim))
                for j in range(width)]
            for p in valid
        }
        projected_values = {
            p: [fsum(values[j][q] * xs[e][p][q] for q in range(local_dim))
                for j in range(width)]
            for p in valid
        }
        for k in range(slots):
            logits = [
                fsum(queries[k][j] * projected_keys[p][j] for j in range(width))
                / sqrt(width)
                for p in valid
            ]
            numerators = [exp(logit - max(logits)) for logit in logits]
            denominator = fsum(numerators)
            for p, numerator in zip(valid, numerators):
                a[e][k][p] = numerator / denominator
            for j in range(width):
                h[e][k][j] = fsum(a[e][k][p] * projected_values[p][j] for p in valid)
    return h, a


def _leaf_inputs(args: list[torch.Tensor]) -> list[torch.Tensor]:
    return [arg.clone().requires_grad_(arg.is_floating_point()) for arg in args]


def _floating(args: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return args[0], args[2], args[3], args[4]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_independent_scalar_oracle(dtype: torch.dtype) -> None:
    args = _inputs(dtype)
    h, available, a = compatible_readout(*args)
    expected_h, expected_a = _scalar_oracle(*args)
    torch.testing.assert_close(h, torch.tensor(expected_h, dtype=dtype))
    torch.testing.assert_close(a, torch.tensor(expected_a, dtype=dtype))
    assert torch.equal(available, torch.tensor([True, False, True]))
    assert h.shape == (3, 2, 2) and a.shape == (3, 2, 3)
    assert h.dtype == a.dtype == dtype and available.dtype == torch.bool
    assert h.device.type == a.device.type == available.device.type == "cpu"
    assert torch.count_nonzero(a.masked_select(~args[1].unsqueeze(1))) == 0
    torch.testing.assert_close(a.sum(-1), available[:, None].expand(3, 2).to(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_l7_analytic_outputs_and_query_gradients(dtype: torch.dtype) -> None:
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=dtype)
    mask = torch.tensor([[True, True]])
    b = torch.eye(2, dtype=dtype, requires_grad=True)
    w_key = torch.eye(2, dtype=dtype)
    w_value = torch.diag(torch.tensor([1.0, 2.0], dtype=dtype))
    h, available, a = compatible_readout(x, mask, b, w_key, w_value)
    alpha = 1 / (1 + exp(-1 / sqrt(2)))
    expected_h = torch.tensor(
        [[[alpha, 2 * (1 - alpha)], [1 - alpha, 2 * alpha]]], dtype=dtype
    )
    expected_a = torch.tensor(
        [[[alpha, 1 - alpha], [1 - alpha, alpha]]], dtype=dtype
    )
    torch.testing.assert_close(h, expected_h)
    torch.testing.assert_close(a, expected_a)
    assert available.tolist() == [True]
    assert not torch.allclose(h[:, 0], h[:, 1])
    h.backward(torch.tensor([[[1.0, -1.0], [1.0, -1.0]]], dtype=dtype))
    derivative = 3 * alpha * (1 - alpha) / sqrt(2)
    expected_gradient = torch.tensor([[derivative, -derivative]] * 2, dtype=dtype)
    torch.testing.assert_close(b.grad, expected_gradient)
    assert derivative > 0  # Nondegeneracy is claimed for this analytic fixture only.


def test_float64_gradcheck_for_tokens_queries_and_both_maps() -> None:
    args = _leaf_inputs(_inputs())

    def outputs(x, b, key, value):
        h, _, a = compatible_readout(x, args[1], b, key, value)
        return h, a

    assert torch.autograd.gradcheck(outputs, _floating(args), eps=1e-6, atol=1e-5)


def test_gradients_match_finite_differences_of_scalar_oracle() -> None:
    args = _leaf_inputs(_inputs())
    upstream = torch.tensor(
        [[[0.7, -1.2], [-0.3, 0.4]], [[0.2, 0.9], [0.6, -0.4]],
         [[-0.5, 0.8], [1.1, -0.6]]], dtype=torch.float64
    )
    h, _, _ = compatible_readout(*args)
    gradients = torch.autograd.grad((h * upstream).sum(), _floating(args))
    step = 1e-6
    for arg_index, gradient in zip((0, 2, 3, 4), gradients):
        expected = torch.empty_like(gradient)
        for index in product(*(range(size) for size in gradient.shape)):
            losses = []
            for sign in (1, -1):
                perturbed = [arg.detach().clone() for arg in args]
                perturbed[arg_index][index] += sign * step
                oracle_h, _ = _scalar_oracle(*perturbed)
                losses.append(fsum(
                    oracle_h[e][k][j] * upstream[e, k, j].item()
                    for e in range(3) for k in range(2) for j in range(2)
                ))
            expected[index] = (losses[0] - losses[1]) / (2 * step)
        torch.testing.assert_close(gradient, expected, atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("padded", [False, True])
def test_singleton_equal_slots_and_exact_zero_query_key_derivatives(
    dtype: torch.dtype, padded: bool
) -> None:
    tokens = [[[0.4, -0.8, 1.2]]]
    mask = [[True]]
    if padded:
        tokens = [[[9.0, 7.0, -2.0], tokens[0][0], [-4.0, 2.0, 8.0]]]
        mask = [[False, True, False]]
    args = _inputs(dtype)
    args[0] = torch.tensor(tokens, dtype=dtype)
    args[1] = torch.tensor(mask)
    args = _leaf_inputs(args)
    x, _, b, key, value = args
    h, available, a = compatible_readout(*args)
    position = 1 if padded else 0
    expected_value = value @ x[0, position]
    torch.testing.assert_close(h, expected_value.expand(1, 2, 2), rtol=0, atol=0)
    assert torch.equal(h[:, 0], h[:, 1]) and available.tolist() == [True]
    assert torch.equal(a[0, :, position], torch.ones(2, dtype=dtype))
    assert torch.count_nonzero(a) == 2
    upstream = torch.tensor([[[0.7, -1.2], [-0.3, 0.4]]], dtype=dtype)
    gx, gb, gkey, gvalue = torch.autograd.grad(h, (x, b, key, value), upstream)
    assert torch.equal(gb, torch.zeros_like(b))
    assert torch.equal(gkey, torch.zeros_like(key))
    expected_gx = torch.zeros_like(x)
    expected_gx[0, position] = value.T @ upstream.sum((0, 1))
    torch.testing.assert_close(gx, expected_gx)
    torch.testing.assert_close(gvalue, torch.outer(upstream.sum((0, 1)), x[0, position]))
    assert torch.isfinite(gx).all() and torch.isfinite(gvalue).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("entities,units", [(0, 3), (0, 0), (2, 0), (2, 3)])
def test_empty_and_absent_rows_never_enter_softmax(
    dtype: torch.dtype, entities: int, units: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(dtype)
    args[0] = torch.full((entities, units, 3), 4.0, dtype=dtype)
    args[1] = torch.zeros((entities, units), dtype=torch.bool)
    args = _leaf_inputs(args)

    def forbidden_softmax(*args, **kwargs):
        pytest.fail("softmax must not run for empty/absent inputs")

    monkeypatch.setattr(torch, "softmax", forbidden_softmax)
    h, available, a = compatible_readout(*args)
    assert h.shape == (entities, 2, 2) and a.shape == (entities, 2, units)
    assert available.shape == (entities,) and available.dtype == torch.bool
    assert not available.any()
    for output in (h, a):
        assert torch.isfinite(output).all() and torch.count_nonzero(output) == 0
    gradients = torch.autograd.grad(h.sum() + a.sum(), _floating(args))
    for arg, gradient in zip(_floating(args), gradients):
        assert torch.equal(gradient, torch.zeros_like(arg))


def test_mixed_rows_only_active_entities_enter_softmax(monkeypatch: pytest.MonkeyPatch) -> None:
    original = torch.softmax
    calls = []

    def recording_softmax(logits, dim):
        calls.append(logits.clone())
        assert torch.isfinite(logits).any(dim=dim).all()
        return original(logits, dim=dim)

    monkeypatch.setattr(torch, "softmax", recording_softmax)
    h, available, a = compatible_readout(*_inputs())
    assert len(calls) == 1 and calls[0].shape == (2, 2, 3)
    assert available.tolist() == [True, False, True]
    assert torch.equal(h[1], torch.zeros_like(h[1]))
    assert torch.equal(a[1], torch.zeros_like(a[1]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_joint_unit_mask_permutation_preserves_tokens_and_gradients(dtype: torch.dtype) -> None:
    args = _leaf_inputs(_inputs(dtype))
    permutation = torch.tensor([2, 0, 1])
    permuted = [arg.detach().clone() for arg in args]
    permuted[0] = permuted[0][:, permutation]
    permuted[1] = permuted[1][:, permutation]
    permuted = _leaf_inputs(permuted)
    h, available, a = compatible_readout(*args)
    other_h, other_available, other_a = compatible_readout(*permuted)
    torch.testing.assert_close(h, other_h)
    assert torch.equal(available, other_available)
    torch.testing.assert_close(a[:, :, permutation], other_a)
    upstream = torch.tensor([[[0.7, -1.2], [-0.3, 0.4]]] * 3, dtype=dtype)
    gradients = torch.autograd.grad(h, _floating(args), upstream)
    other_gradients = torch.autograd.grad(other_h, _floating(permuted), upstream)
    torch.testing.assert_close(gradients[0][:, permutation], other_gradients[0])
    for left, right in zip(gradients[1:], other_gradients[1:]):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_finite_payload_perturbation_preserves_outputs_and_gradients(dtype: torch.dtype) -> None:
    args = _inputs(dtype)
    perturbed = [arg.clone() for arg in args]
    # Projection of these values without masking first would overflow.
    perturbed[0][~args[1]] = torch.finfo(dtype).max
    reference, changed = _leaf_inputs(args), _leaf_inputs(perturbed)
    h, available, a = compatible_readout(*reference)
    other_h, other_available, other_a = compatible_readout(*changed)
    assert torch.equal(h, other_h) and torch.equal(a, other_a)
    assert torch.equal(available, other_available)
    upstream = torch.tensor([[[0.7, -1.2], [-0.3, 0.4]]] * 3, dtype=dtype)
    gradients = torch.autograd.grad(h, _floating(reference), upstream)
    changed_gradients = torch.autograd.grad(other_h, _floating(changed), upstream)
    for gradient, changed_gradient in zip(gradients, changed_gradients):
        assert torch.equal(gradient, changed_gradient)
    assert torch.count_nonzero(gradients[0][~args[1]]) == 0


def test_no_input_mutation_or_rng_advance_on_repeated_calls() -> None:
    args = _leaf_inputs(_inputs())
    args[2] = torch.nn.Parameter(args[2])  # Existing caller-owned parameter works directly.
    originals = [arg.detach().clone() for arg in args]
    versions = [arg._version for arg in args]
    identities = [id(arg) for arg in args]
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    import random
    python_state = random.getstate()
    first = compatible_readout(*args)
    second = compatible_readout(*args)
    for arg, original, version, identity in zip(args, originals, versions, identities):
        assert torch.equal(arg, original) and arg._version == version and id(arg) == identity
        assert arg.grad is None
    assert torch.equal(torch.get_rng_state(), torch_state)
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_two_calls_accumulate_shared_query_gradient_as_independent_sum() -> None:
    left = _inputs()
    right = _inputs()
    right[0] = right[0] * -0.7 + 0.2
    right[1] = torch.tensor([[True, True, True], [True, False, True], [False, True, True]])
    right[3] = right[3] * 0.6
    right[4] = right[4] - 0.3
    upstreams = (
        torch.tensor([[[0.7, -1.2], [-0.3, 0.4]]] * 3, dtype=torch.float64),
        torch.tensor([[[-0.2, 0.9], [1.1, -0.6]]] * 3, dtype=torch.float64),
    )
    contributions = []
    for args, upstream in zip((left, right), upstreams):
        independent_query = args[2].clone().requires_grad_()
        h, _, _ = compatible_readout(args[0], args[1], independent_query, args[3], args[4])
        contributions.append(torch.autograd.grad(h, independent_query, upstream)[0])
    shared_query = torch.nn.Parameter(left[2].clone())
    for args, upstream in zip((left, right), upstreams):
        h, _, _ = compatible_readout(args[0], args[1], shared_query, args[3], args[4])
        h.backward(upstream)
    torch.testing.assert_close(shared_query.grad, contributions[0] + contributions[1])


def test_noncontiguous_inputs_preserve_formula_and_autograd() -> None:
    args = _inputs()
    strided = []
    for arg in args:
        storage = torch.empty((*arg.shape, 2), dtype=arg.dtype)
        storage[..., 0] = arg
        view = storage[..., 0]
        assert not view.is_contiguous()
        strided.append(view.requires_grad_(view.is_floating_point()))
    expected = compatible_readout(*args)
    actual = compatible_readout(*strided)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)
    gradients = torch.autograd.grad(actual[0].sum(), _floating(strided))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_cpu_autocast_preserves_supplied_float32_dtype() -> None:
    args = _inputs(torch.float32)
    expected = compatible_readout(*args)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = compatible_readout(*args)
    for left, right in zip(expected, actual):
        assert torch.equal(left, right) and left.dtype == right.dtype


@pytest.mark.parametrize("arg_index", range(5))
def test_rejects_non_tensor_inputs_without_coercion(arg_index: int) -> None:
    args = _inputs()
    args[arg_index] = args[arg_index].tolist()
    with pytest.raises(TypeError, match="torch.Tensor"):
        compatible_readout(*args)


@pytest.mark.parametrize("arg_index", range(5))
def test_rejects_wrong_rank(arg_index: int) -> None:
    args = _inputs()
    args[arg_index] = args[arg_index].unsqueeze(0)
    with pytest.raises(ValueError, match="rank"):
        compatible_readout(*args)


@pytest.mark.parametrize("arg_index", range(5))
def test_rejects_non_cpu_device_without_gpu_work(arg_index: int) -> None:
    args = _inputs()
    args[arg_index] = torch.empty(args[arg_index].shape, dtype=args[arg_index].dtype, device="meta")
    with pytest.raises(ValueError, match="CPU"):
        compatible_readout(*args)


@pytest.mark.parametrize("arg_index", range(5))
def test_rejects_sparse_layout(arg_index: int) -> None:
    args = _inputs()
    args[arg_index] = args[arg_index].to_sparse()
    with pytest.raises(TypeError, match="dense strided"):
        compatible_readout(*args)


@pytest.mark.parametrize("arg_index", (0, 2, 3, 4))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.bool, torch.complex64])
def test_rejects_unsupported_floating_input_dtype(arg_index: int, dtype: torch.dtype) -> None:
    args = _inputs()
    args[arg_index] = args[arg_index].to(dtype)
    with pytest.raises(TypeError, match="float32 or float64"):
        compatible_readout(*args)


@pytest.mark.parametrize("arg_index", (0, 2, 3, 4))
def test_rejects_mixed_floating_dtypes(arg_index: int) -> None:
    args = _inputs()
    args[arg_index] = args[arg_index].float()
    with pytest.raises(TypeError, match="same dtype"):
        compatible_readout(*args)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_rejects_non_boolean_masks(dtype: torch.dtype) -> None:
    args = _inputs()
    args[1] = args[1].to(dtype)
    with pytest.raises(TypeError, match="Boolean"):
        compatible_readout(*args)


@pytest.mark.parametrize("arg_index", (0, 2, 3, 4))
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_supported_inputs(arg_index: int, bad_value: float) -> None:
    args = _inputs()
    args[arg_index].view(-1)[0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        compatible_readout(*args)


def test_rejects_nonfinite_masked_payload() -> None:
    args = _inputs()
    args[0][~args[1]] = torch.nan
    with pytest.raises(ValueError, match="encoded_tokens must be finite"):
        compatible_readout(*args)


@pytest.mark.parametrize(
    "arg_index,shape,message",
    [(1, (3, 2), r"\[E,P\]"), (1, (2, 3), r"\[E,P\]"),
     (3, (3, 2), r"\[h,d\]"), (4, (2, 2), r"\[h,d\]"),
     (0, (3, 3, 0), "positive"), (2, (0, 2), "positive"),
     (2, (2, 0), "positive")],
)
def test_rejects_incompatible_or_nonpositive_dimensions(
    arg_index: int, shape: tuple[int, ...], message: str
) -> None:
    args = _inputs()
    args[arg_index] = torch.zeros(shape, dtype=args[arg_index].dtype)
    with pytest.raises(ValueError, match=message):
        compatible_readout(*args)


@pytest.mark.parametrize("arg_index", (2, 3, 4))
def test_empty_entities_do_not_bypass_input_validation(arg_index: int) -> None:
    args = _inputs()
    args[0] = args[0][:0]
    args[1] = args[1][:0]
    args[arg_index].view(-1)[0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        compatible_readout(*args)


@pytest.mark.parametrize("stage", ["keys", "values", "logits"])
def test_arithmetic_overflow_is_explicit(stage: str) -> None:
    args = _inputs(torch.float32)
    if stage == "logits":
        args[0].fill_(1)
        args[2].fill_(torch.finfo(torch.float32).max)
        args[3].fill_(1)
    else:
        args[0].fill_(torch.finfo(torch.float32).max)
        args[3 if stage == "keys" else 4].fill_(2)
    with pytest.raises(ValueError, match=stage + " must be finite"):
        compatible_readout(*args)
