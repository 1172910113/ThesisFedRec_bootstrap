from __future__ import annotations

from itertools import product
from math import exp, sqrt
import random

import numpy as np
import pytest
import torch

from src.methods.doi4 import center_fusion, compatible_readout


_FLOAT_INDICES = (0, 2, 3, 4, 5, 6)
_FLOAT_NAMES = ("tokens", "items", "w_query", "w_key", "w_value", "w_output")


def _inputs(dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    # h=2, d=3 and nonsymmetric maps exercise projection orientation.
    return [
        torch.tensor([[[.2, -.7], [4., -2.], [1.3, .4]],
                      [[2., 3.], [-4., 2.], [.5, -.6]],
                      [[-.3, .8], [.9, -.2], [2., 1.]]], dtype=dtype),
        torch.tensor([[True, False, True], [False, False, False], [True, True, False]]),
        torch.tensor([[.6, -.2, .7], [1., -.3, .2], [-.5, .9, -.4]], dtype=dtype),
        torch.tensor([[.7, -.4, .2], [-.1, .3, .8]], dtype=dtype),
        torch.tensor([[.8, -.2], [.3, .6]], dtype=dtype),
        torch.tensor([[-.2, .9], [.8, -.6]], dtype=dtype),
        torch.tensor([[.7, -.3], [.2, .8]], dtype=dtype),
    ]


def _leaves(args: list[torch.Tensor]) -> list[torch.Tensor]:
    return [arg.detach().clone().requires_grad_(arg.is_floating_point()) for arg in args]


def _floating(args: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(args[i] for i in _FLOAT_INDICES)


def _oracle(*args: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Independent float64 NumPy calculation, without production/Torch algebra."""
    x, mask, items, query_map, key_map, value_map, output_map = [
        arg.detach().numpy().astype(np.float64 if arg.is_floating_point() else bool)
        for arg in args
    ]
    entities, tokens, width = x.shape
    fused = np.zeros((entities, width), dtype=np.float64)
    attention = np.zeros((entities, tokens), dtype=np.float64)
    for e in range(entities):
        selected = np.flatnonzero(mask[e])
        if selected.size == 0:
            continue
        query = query_map.dot(items[e])
        keys = x[e, selected].dot(key_map.T)
        values = x[e, selected].dot(value_map.T)
        scores = keys.dot(query) / sqrt(width)
        numerators = np.exp(scores - scores.max())
        weights = numerators / numerators.sum()
        attention[e, selected] = weights
        fused[e] = output_map.dot(weights.dot(values))
    return fused, attention


def _upstreams(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[.7, -1.2], [.2, .9], [-.5, .8]], dtype=dtype),
        torch.tensor([[.1, -.3, .2], [.4, -.5, .3], [-.2, .6, -.1]], dtype=dtype),
    )


def _gradient_evidence() -> dict[str, dict[str, float]]:
    args = _leaves(_inputs())
    upstream_f, upstream_a = _upstreams(torch.float64)
    fused, _, attention = center_fusion(*args)
    gradients = torch.autograd.grad(
        (fused * upstream_f).sum() + (attention * upstream_a).sum(), _floating(args)
    )
    step = 1e-6
    evidence = {}
    for arg_index, name, gradient in zip(_FLOAT_INDICES, _FLOAT_NAMES, gradients):
        expected = torch.empty_like(gradient)
        for index in product(*(range(size) for size in gradient.shape)):
            losses = []
            for sign in (1, -1):
                changed = [arg.detach().clone() for arg in args]
                changed[arg_index][index] += sign * step
                expected_f, expected_a = _oracle(*changed)
                losses.append(float(
                    (expected_f * upstream_f.numpy()).sum()
                    + (expected_a * upstream_a.numpy()).sum()
                ))
            expected[index] = (losses[0] - losses[1]) / (2 * step)
        # Smooth bounded fixtures have O(step^2) truncation and small cancellation
        # error; this absolute bound is much tighter than the requested ~1e-7.
        torch.testing.assert_close(gradient, expected, atol=2e-9, rtol=1e-7)
        evidence[name] = {
            "max_absolute_error": (gradient - expected).abs().max().item(),
            "gradient_norm": gradient.norm().item(),
        }
    return evidence


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_independent_numpy_oracle(dtype: torch.dtype) -> None:
    args = _inputs(dtype)
    fused, available, attention = center_fusion(*args)
    expected_f, expected_a = _oracle(*args)
    torch.testing.assert_close(fused, torch.tensor(expected_f, dtype=dtype))
    torch.testing.assert_close(attention, torch.tensor(expected_a, dtype=dtype))
    assert fused.shape == (3, 2) and attention.shape == (3, 3)
    assert available.shape == (3,) and available.tolist() == [True, False, True]
    assert available.dtype == torch.bool and fused.dtype == attention.dtype == dtype
    assert all(tensor.device.type == "cpu" for tensor in (fused, available, attention))
    assert torch.count_nonzero(attention[~args[1]]) == 0
    torch.testing.assert_close(attention.sum(-1), available.to(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_hand_example(dtype: torch.dtype) -> None:
    tokens = torch.tensor([[[1., 0.], [0., 2.]]], dtype=dtype)
    items = torch.tensor([[1., 0.]], dtype=dtype)
    identity = torch.eye(2, dtype=dtype)
    fused, available, attention = center_fusion(
        tokens, torch.tensor([[True, True]]), items,
        identity, identity, identity, identity,
    )
    a = exp(1 / sqrt(2)) / (exp(1 / sqrt(2)) + 1)
    torch.testing.assert_close(attention, torch.tensor([[a, 1 - a]], dtype=dtype))
    torch.testing.assert_close(fused, torch.tensor([[a, 2 * (1 - a)]], dtype=dtype))
    assert available.tolist() == [True]


def test_float64_central_differences_for_every_floating_input() -> None:
    evidence = _gradient_evidence()
    # This particular fixture intentionally exercises each nondegenerate path.
    assert all(record["gradient_norm"] > 0 for record in evidence.values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("padded", [False, True])
def test_singleton_connected_zero_query_key_and_correct_other_gradients(
    dtype: torch.dtype, padded: bool
) -> None:
    args = _inputs(dtype)
    args[0] = torch.tensor([[[.4, -.8]]], dtype=dtype)
    args[1] = torch.tensor([[True]])
    args[2] = args[2][:1]
    position = 0
    if padded:
        args[0] = torch.tensor([[[9., 7.], [.4, -.8], [-4., 2.]]], dtype=dtype)
        args[1] = torch.tensor([[False, True, False]])
        position = 1
    args = _leaves(args)
    x, _, items, query, key, value, output = args
    fused, available, attention = center_fusion(*args)
    mapped = value @ x[0, position]
    torch.testing.assert_close(fused[0], output @ mapped)
    assert available.tolist() == [True] and attention[0, position].item() == 1
    assert torch.count_nonzero(attention) == 1
    upstream = torch.tensor([[.7, -1.2]], dtype=dtype)
    gx, gi, gq, gk, gv, go = torch.autograd.grad(fused, _floating(args), upstream)
    for gradient, tensor in ((gi, items), (gq, query), (gk, key)):
        assert torch.equal(gradient, torch.zeros_like(tensor))
    value_upstream = output.T @ upstream[0]
    expected_gx = torch.zeros_like(x)
    expected_gx[0, position] = value.T @ value_upstream
    torch.testing.assert_close(gx, expected_gx)
    torch.testing.assert_close(gv, torch.outer(value_upstream, x[0, position]))
    torch.testing.assert_close(go, torch.outer(upstream[0], mapped))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("entities,tokens", [(0, 0), (0, 3), (2, 0), (2, 3)])
def test_empty_absent_outputs_are_connected_zeros_without_softmax(
    dtype: torch.dtype, entities: int, tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(dtype)
    args[0] = torch.full((entities, tokens, 2), 4., dtype=dtype)
    args[1] = torch.zeros((entities, tokens), dtype=torch.bool)
    args[2] = torch.full((entities, 3), .3, dtype=dtype)
    args = _leaves(args)

    def forbidden_softmax(*args, **kwargs):
        pytest.fail("empty/all-masked rows must not enter softmax")

    monkeypatch.setattr(torch, "softmax", forbidden_softmax)
    fused, available, attention = center_fusion(*args)
    assert fused.shape == (entities, 2) and attention.shape == (entities, tokens)
    assert available.shape == (entities,) and available.dtype == torch.bool
    assert not available.any()
    for result in (fused, attention):
        assert torch.isfinite(result).all() and torch.count_nonzero(result) == 0
    # Fused alone must retain every floating input, without allow_unused.
    gradients = torch.autograd.grad(fused.sum(), _floating(args))
    for tensor, gradient in zip(_floating(args), gradients):
        assert torch.equal(gradient, torch.zeros_like(tensor))


def test_mixed_rows_only_valid_entities_enter_softmax(monkeypatch: pytest.MonkeyPatch) -> None:
    original = torch.softmax
    calls = []

    def recording_softmax(logits, dim):
        calls.append(logits.clone())
        assert torch.isfinite(logits).any(dim=dim).all()
        return original(logits, dim=dim)

    monkeypatch.setattr(torch, "softmax", recording_softmax)
    args = _leaves(_inputs())
    fused, available, attention = center_fusion(*args)
    assert len(calls) == 1 and calls[0].shape == (2, 3)
    assert available.tolist() == [True, False, True]
    assert torch.equal(fused[1], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(attention[1], torch.zeros(3, dtype=torch.float64))
    gradients = torch.autograd.grad(fused[1].sum(), _floating(args))
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_joint_token_mask_permutation_preserves_outputs_and_gradients(dtype: torch.dtype) -> None:
    args = _leaves(_inputs(dtype))
    permutation = torch.tensor([2, 0, 1])
    permuted = [arg.detach().clone() for arg in args]
    permuted[0], permuted[1] = permuted[0][:, permutation], permuted[1][:, permutation]
    permuted = _leaves(permuted)
    fused, available, attention = center_fusion(*args)
    other_f, other_available, other_a = center_fusion(*permuted)
    torch.testing.assert_close(fused, other_f)
    assert torch.equal(available, other_available)
    torch.testing.assert_close(attention[:, permutation], other_a)
    upstream = _upstreams(dtype)[0]
    gradients = torch.autograd.grad(fused, _floating(args), upstream)
    other_gradients = torch.autograd.grad(other_f, _floating(permuted), upstream)
    torch.testing.assert_close(gradients[0][:, permutation], other_gradients[0])
    for left, right in zip(gradients[1:], other_gradients[1:]):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_finite_masked_perturbations_preserve_outputs_and_all_gradients(dtype: torch.dtype) -> None:
    args = _leaves(_inputs(dtype))
    changed = [arg.detach().clone() for arg in args]
    changed[0][~args[1]] = torch.finfo(dtype).max
    changed = _leaves(changed)
    fused, available, attention = center_fusion(*args)
    other_f, other_available, other_a = center_fusion(*changed)
    assert torch.equal(fused, other_f) and torch.equal(attention, other_a)
    assert torch.equal(available, other_available)
    upstream_f, upstream_a = _upstreams(dtype)
    gradients = torch.autograd.grad(
        (fused * upstream_f).sum() + (attention * upstream_a).sum(), _floating(args)
    )
    other_gradients = torch.autograd.grad(
        (other_f * upstream_f).sum() + (other_a * upstream_a).sum(), _floating(changed)
    )
    for left, right in zip(gradients, other_gradients):
        assert torch.equal(left, right)
    assert torch.count_nonzero(gradients[0][~args[1]]) == 0


def test_inputs_storage_versions_and_rng_unchanged() -> None:
    args = _leaves(_inputs())
    args[3] = torch.nn.Parameter(args[3])
    originals = [arg.detach().clone() for arg in args]
    identities = [id(arg) for arg in args]
    storage_pointers = [arg.untyped_storage().data_ptr() for arg in args]
    versions = [arg._version for arg in args]
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    first, second = center_fusion(*args), center_fusion(*args)
    for index, arg in enumerate(args):
        assert torch.equal(arg, originals[index]) and id(arg) == identities[index]
        assert arg._version == versions[index] and arg.grad is None
        assert arg.untyped_storage().data_ptr() == storage_pointers[index]
    assert random.getstate() == python_rng
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_noncontiguous_inputs_and_cpu_autocast() -> None:
    args = _inputs(torch.float32)
    strided = []
    for arg in args:
        storage = torch.zeros((*arg.shape, 2), dtype=arg.dtype)
        storage[..., 0] = arg
        view = storage[..., 0].requires_grad_(arg.is_floating_point())
        assert not view.is_contiguous()
        strided.append(view)
    expected = center_fusion(*args)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = center_fusion(*strided)
    for left, right in zip(expected, actual):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual[0].sum(), _floating(strided))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def _composition_leaves() -> list[torch.Tensor]:
    fusion = _inputs()
    return _leaves([
        torch.tensor([[[.2, -.7], [1.3, .4]], [[-.3, .8], [.9, -.2]]], dtype=torch.float64),
        torch.tensor([[.6, -.2], [-.5, .9]], dtype=torch.float64),
        torch.tensor([[.7, -.4], [-.1, .8]], dtype=torch.float64),
        torch.tensor([[-.2, .9], [.8, -.6]], dtype=torch.float64),
        fusion[2][:2], *fusion[3:],
    ])


def _composed_result(leaves: list[torch.Tensor], independent: bool) -> torch.Tensor:
    # Deliberate caller-owned non-leaf expressions for every kernel input.
    source, readout_b, readout_k, readout_v, items, query, key, value, output = [
        tensor * (1 + index * .03) + .02 for index, tensor in enumerate(leaves)
    ]
    if independent:
        # Unsplit monolithic reference, with a different algebraic layout.
        source_keys = torch.einsum("epd,hd->eph", source, readout_k)
        readout_scores = torch.einsum("kh,eph->ekp", readout_b, source_keys) / sqrt(2)
        readout_weights = torch.softmax(readout_scores, dim=-1)
        source_values = torch.einsum("epd,hd->eph", source, readout_v)
        messages = torch.einsum("ekp,eph->ekh", readout_weights, source_values)
        projected_query = torch.einsum("ed,hd->eh", items, query)
        fusion_keys = torch.einsum("elh,jh->elj", messages, key)
        fusion_values = torch.einsum("elh,jh->elj", messages, value)
        logits = torch.einsum("eh,elh->el", projected_query, fusion_keys) / sqrt(2)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("el,elh->eh", weights, fusion_values)
        fused = torch.einsum("eh,jh->ej", pooled, output)
    else:
        messages, availability, _ = compatible_readout(
            source, torch.ones((2, 2), dtype=torch.bool), readout_b, readout_k, readout_v
        )
        fused, available, _ = center_fusion(
            messages, availability[:, None].expand(2, 2), items,
            query, key, value, output,
        )
        assert available.all()
    return fused


def test_readout_fusion_nonleaf_composition_matches_unsplit_reference() -> None:
    leaves = _composition_leaves()
    expected_leaves = _leaves(leaves)
    actual = _composed_result(leaves, independent=False)
    expected = _composed_result(expected_leaves, independent=True)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    upstream = torch.tensor([[.7, -1.2], [-.5, .8]], dtype=torch.float64)
    gradients = torch.autograd.grad(actual, leaves, upstream)
    expected_gradients = torch.autograd.grad(expected, expected_leaves, upstream)
    for gradient, reference in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, reference, atol=1e-13, rtol=1e-11)
        # This fixture is chosen to exercise each path; not a general promise.
        assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_item_embedding_query_and_direct_score_gradient_matches_unsplit_oracle() -> None:
    args = _leaves(_inputs())
    fused, _, _ = center_fusion(*args)
    fused_weight = _upstreams(torch.float64)[0]
    direct_weight = torch.tensor([[.3, -.2, .7], [-.6, .1, .4], [.5, .8, -.3]], dtype=torch.float64)
    # The same item leaf has both query and direct-score paths; scoring is test-owned.
    loss = (fused * fused_weight).sum() + (args[2] * direct_weight).sum()
    combined_gradient = torch.autograd.grad(loss, args[2])[0]
    expected = torch.empty_like(args[2])
    for index in product(range(3), range(3)):
        losses = []
        for sign in (1, -1):
            changed = [arg.detach().clone() for arg in args]
            changed[2][index] += sign * 1e-6
            oracle_fused, _ = _oracle(*changed)
            losses.append(float(
                (oracle_fused * fused_weight.numpy()).sum()
                + (changed[2].numpy() * direct_weight.numpy()).sum()
            ))
        expected[index] = (losses[0] - losses[1]) / 2e-6
    torch.testing.assert_close(combined_gradient, expected, atol=2e-9, rtol=1e-7)
    # The absent entity has only the direct path, which must not disappear.
    torch.testing.assert_close(combined_gradient[1], direct_weight[1], rtol=0, atol=0)


def test_shared_fusion_graph_two_losses_matches_two_identical_calls() -> None:
    shared, repeated = _leaves(_inputs()), _leaves(_inputs())
    first_f, first_a = _upstreams(torch.float64)
    second_f = torch.tensor([[-.2, .9], [.1, -.6], [.8, -.4]], dtype=torch.float64)
    second_a = torch.tensor([[.3, -.1, .4], [-.2, .6, .1], [.5, -.3, .2]], dtype=torch.float64)

    def contribution(fused, attention, upstream_f, upstream_a):
        return (fused * upstream_f).sum() + (attention * upstream_a).sum()

    fused, _, attention = center_fusion(*shared)
    shared_loss = contribution(fused, attention, first_f, first_a) + contribution(fused, attention, second_f, second_a)
    gradients = torch.autograd.grad(shared_loss, _floating(shared))
    independent_contributions = []
    for upstream_f, upstream_a in ((first_f, first_a), (second_f, second_a)):
        other_f, _, other_a = center_fusion(*repeated)
        assert torch.equal(other_f, fused) and torch.equal(other_a, attention)
        independent_contributions.append(torch.autograd.grad(
            contribution(other_f, other_a, upstream_f, upstream_a), _floating(repeated)
        ))
    for actual, first, second in zip(gradients, *independent_contributions):
        torch.testing.assert_close(actual, first + second, atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("arg_index", range(7))
def test_rejects_non_tensor_without_coercion(arg_index: int) -> None:
    args = _inputs()
    args[arg_index] = args[arg_index].tolist()
    with pytest.raises(TypeError, match="torch.Tensor"):
        center_fusion(*args)


@pytest.mark.parametrize("arg_index", range(7))
def test_rejects_wrong_rank(arg_index: int) -> None:
    args = _inputs()
    args[arg_index] = args[arg_index].unsqueeze(0)
    with pytest.raises(ValueError, match="rank"):
        center_fusion(*args)


@pytest.mark.parametrize("arg_index", range(7))
def test_rejects_non_cpu_device_without_gpu_work(arg_index: int) -> None:
    args = _inputs()
    args[arg_index] = torch.empty(args[arg_index].shape, dtype=args[arg_index].dtype, device="meta")
    with pytest.raises(ValueError, match="CPU"):
        center_fusion(*args)


@pytest.mark.parametrize("arg_index", range(7))
def test_rejects_sparse_layout(arg_index: int) -> None:
    args = _inputs()
    args[arg_index] = args[arg_index].to_sparse()
    with pytest.raises(TypeError, match="dense strided"):
        center_fusion(*args)


@pytest.mark.parametrize("arg_index", _FLOAT_INDICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.bool, torch.complex64])
def test_rejects_unsupported_floating_input_dtype(arg_index: int, dtype: torch.dtype) -> None:
    args = _inputs()
    args[arg_index] = args[arg_index].to(dtype)
    with pytest.raises(TypeError, match="float32 or float64"):
        center_fusion(*args)


@pytest.mark.parametrize("arg_index", _FLOAT_INDICES)
def test_rejects_mixed_floating_dtypes(arg_index: int) -> None:
    args = _inputs()
    args[arg_index] = args[arg_index].float()
    with pytest.raises(TypeError, match="same dtype"):
        center_fusion(*args)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_rejects_non_boolean_mask(dtype: torch.dtype) -> None:
    args = _inputs()
    args[1] = args[1].to(dtype)
    with pytest.raises(TypeError, match="Boolean"):
        center_fusion(*args)


@pytest.mark.parametrize("arg_index", _FLOAT_INDICES)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_inputs(arg_index: int, bad_value: float) -> None:
    args = _inputs()
    args[arg_index].view(-1)[0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        center_fusion(*args)


def test_nonfinite_masked_values_are_rejected() -> None:
    args = _inputs()
    args[0][~args[1]] = torch.nan
    with pytest.raises(ValueError, match="compatible_tokens must be finite"):
        center_fusion(*args)


@pytest.mark.parametrize(
    "arg_index,shape,message",
    [(1, (3, 2), r"\[E,L\]"), (1, (2, 3), r"\[E,L\]"),
     (2, (2, 3), r"\[E,d\]"), (3, (3, 2), r"\[h,d\]"),
     (3, (2, 2), r"\[h,d\]"), (4, (3, 2), r"\[h,h\]"),
     (5, (2, 3), r"\[h,h\]"), (6, (1, 2), r"\[h,h\]"),
     (0, (3, 3, 0), "positive"), (2, (3, 0), "positive")],
)
def test_rejects_invalid_dimensions(arg_index: int, shape: tuple[int, ...], message: str) -> None:
    args = _inputs()
    args[arg_index] = torch.zeros(shape, dtype=args[arg_index].dtype)
    with pytest.raises(ValueError, match=message):
        center_fusion(*args)


@pytest.mark.parametrize("arg_index", (3, 4, 5, 6))
@pytest.mark.parametrize("empty_entities", [False, True])
def test_empty_axes_do_not_bypass_finite_validation(arg_index: int, empty_entities: bool) -> None:
    args = _inputs()
    if empty_entities:
        args[0], args[1], args[2] = args[0][:0], args[1][:0], args[2][:0]
    else:
        args[0], args[1] = args[0][:, :0], args[1][:, :0]
    args[arg_index].view(-1)[0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        center_fusion(*args)


@pytest.mark.parametrize("stage", ["query", "keys", "values", "logits", "fused"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_arithmetic_overflow_is_explicit(stage: str, dtype: torch.dtype) -> None:
    largest = torch.finfo(dtype).max
    args = [torch.ones((1, 1, 2), dtype=dtype), torch.ones((1, 1), dtype=torch.bool),
            torch.ones((1, 3), dtype=dtype), torch.full((2, 3), .1, dtype=dtype),
            torch.eye(2, dtype=dtype), torch.eye(2, dtype=dtype), torch.eye(2, dtype=dtype)]
    if stage == "query":
        args[3].fill_(largest)
    elif stage in ("keys", "values"):
        args[4 if stage == "keys" else 5].fill_(largest)
    elif stage == "logits":
        args[0].fill_(largest)
        args[3].fill_(1 / 3)
    else:
        args[0].fill_(largest / 4)
        args[6].fill_(4)
    with pytest.raises(ValueError, match=stage + " must be finite"):
        center_fusion(*args)
