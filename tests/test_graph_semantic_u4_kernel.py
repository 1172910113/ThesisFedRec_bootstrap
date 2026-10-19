"""Synthetic numerical evidence only; dense item-pair matrices stay in tiny oracles."""

import inspect
import json
import math
import resource
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from src.methods.graph_semantic import (
    apply_path,
    detached_operator_snapshot,
    freeze_source_snapshot,
    make_operator_sketch,
    prepare_path_factors,
    prepare_receiver,
    prepare_source_kernel,
    receiver_logits,
    receiver_loss,
    receiver_tokens,
    source_logits,
    source_loss,
)
from src.methods.graph_semantic import path_operator, u4_kernel


DTYPES = [torch.float64, torch.float32]
TRIPLES = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (2, 0, 1)]


def _edges(triples):
    return torch.tensor(triples, dtype=torch.long).reshape(-1, 3)


def _paths(triples=TRIPLES, local_ids=(0, 1, 2), n=3, dtype=torch.float64):
    return prepare_path_factors(torch.tensor(local_ids, dtype=torch.long), _edges(triples), num_items=n, dtype=dtype)


def _fixture(dtype=torch.float64, *, empty=False):
    paths = _paths([], (), dtype=dtype) if empty else _paths(dtype=dtype)
    e = torch.tensor([[2, 0], [0, 1], [0, 1]], dtype=dtype, requires_grad=True)
    w = torch.ones(2, dtype=dtype, requires_grad=True)
    b = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    basis = torch.tensor([[1, 0], [0, 1 / math.sqrt(2)], [0, 1 / math.sqrt(2)]], dtype=dtype, requires_grad=True)
    source = prepare_source_kernel(paths, e, w, b, basis)
    return source, (e, w, b, basis)


def _dense(m, triples):
    """Independent NumPy oracle: binary incidence, dense normalization and M."""
    unique = set(triples)
    keys = sorted({(relation, aux) for _, relation, aux in unique})
    r = np.zeros((m, len(keys)), dtype=np.float64)
    for row, relation, aux in unique:
        r[row, keys.index((relation, aux))] = 1
    di, da = r.sum(axis=1), r.sum(axis=0)
    inv_i, inv_a = np.zeros_like(di), np.zeros_like(da)
    inv_i[di > 0] = 1 / np.sqrt(di[di > 0])
    inv_a[da > 0] = 1 / np.sqrt(da[da > 0])
    s = inv_i[:, None] * r * inv_a[None, :]
    m2 = s @ s.T
    np.fill_diagonal(m2, 0)
    return r, s, m2


def _source_oracle(source, p, a, users, items, labels, triples=TRIPLES):
    e = source.item_embeddings.numpy()
    basis = source.basis.numpy()
    local_ids = source.paths.local_item_ids.numpy()
    _, _, m = _dense(len(local_ids), triples)
    f = np.stack([m @ e[local_ids], m @ m @ e[local_ids]])
    g = 1 / (1 + np.exp(-a))
    x = e.copy()
    x[local_ids] += (g[:, None, None] * f).sum(axis=0)
    z = (p[users] * x[items] * source.weight.numpy()).sum(axis=1) + source.bias.item()
    loss = np.mean(np.logaddexp(0, z) - labels * z)
    bl = basis[local_ids]
    sketch = np.stack([g[0] * bl.T @ m @ bl, g[1] * bl.T @ m @ m @ bl])
    return z, loss, sketch, f, x


def _tol(dtype):
    return 1e-10 if dtype == torch.float64 else 1e-5


def _assert(actual, expected, dtype=torch.float64):
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=_tol(dtype), rtol=_tol(dtype))


def _difference(function, value, step=1e-6):
    result = np.empty_like(value)
    for index in np.ndindex(value.shape):
        plus, minus = value.copy(), value.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2 * step)
    return result


@pytest.mark.parametrize("dtype", DTYPES)
def test_section_18_7_source_oracle_analytic_autograd_and_differences(dtype):
    source, public = _fixture(dtype)
    p = torch.tensor([[1, 0.5]], dtype=dtype, requires_grad=True)
    a = torch.zeros(2, dtype=dtype, requires_grad=True)
    users, items = torch.tensor([0]), torch.tensor([0])
    labels = torch.ones(1, dtype=dtype)
    z, expected_loss, sketch, f, x = _source_oracle(source, p.detach().numpy(), a.detach().numpy(), [0], [0], [1])
    _assert(source.f2, f[0], dtype)
    _assert(source.f4, f[1], dtype)
    _assert(source_logits(source, p, a, users, items), z, dtype)
    loss = source_loss(source, p, a, users, items, labels)
    _assert(loss, expected_loss, dtype)
    _assert(make_operator_sketch(source, a), sketch, dtype)
    assert z[0] == pytest.approx(2.2446383476, abs=1e-7)
    assert expected_loss == pytest.approx(0.1007190312, abs=1e-7)
    np.testing.assert_allclose(sketch, [
        [[0, 0.125], [0.125, math.sqrt(2) / 8]],
        [[1 / 16, math.sqrt(2) / 32], [math.sqrt(2) / 32, 3 / 32]],
    ], atol=1e-7)
    ga, gp = torch.autograd.grad(loss, (a, p), retain_graph=True)
    error = 1 / (1 + np.exp(-z[0])) - 1
    analytic_a = error * 0.25 * (f[:, 0] * p.detach().numpy()[0] * source.weight.numpy()).sum(axis=1)
    _assert(ga, analytic_a, dtype)
    _assert(gp, [error * source.weight.numpy() * x[0]], dtype)
    if dtype == torch.float64:
        fd = _difference(lambda v: _source_oracle(source, p.detach().numpy(), v, [0], [0], [1])[1], np.zeros(2))
        np.testing.assert_allclose(ga.numpy(), fd, atol=1e-7, rtol=0)
        fp = _difference(lambda v: _source_oracle(source, v, np.zeros(2), [0], [0], [1])[1], p.detach().numpy())
        np.testing.assert_allclose(gp.numpy(), fp, atol=1e-7, rtol=0)
    loss.backward()
    assert all(t.grad is None and t.requires_grad for t in public)
    assert source.item_embeddings.data_ptr() == public[0].data_ptr()
    assert source.basis.data_ptr() == public[3].data_ptr()


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_graph_receiver_oracle_analytic_autograd_and_differences(dtype):
    donor, _ = _fixture(dtype)
    a = torch.zeros(2, dtype=dtype, requires_grad=True)
    bank = detached_operator_snapshot(donor, a)
    source, public = _fixture(dtype, empty=True)
    p = torch.tensor([[0.5, 1]], dtype=dtype, requires_grad=True)
    receiver = prepare_receiver(freeze_source_snapshot(source, a), p, bank)
    beta = torch.zeros(2, dtype=dtype, requires_grad=True)
    items, users = torch.tensor([0]), torch.tensor([0])
    labels = torch.ones(1, dtype=dtype)
    # Receiver oracle uses only its own E/B and the explicit bank.
    e, basis = source.item_embeddings.numpy(), source.basis.numpy()
    _, _, oracle_bank, _, _ = _source_oracle(donor, np.array([[1, 0.5]]), np.zeros(2), [0], [0], [1])
    tokens = np.stack([basis[0] @ t @ basis.T @ e for t in oracle_bank])
    _assert(receiver_tokens(receiver, items), tokens[None], dtype)
    np.testing.assert_allclose(tokens, [[0, math.sqrt(2) / 8], [0.125, 0.0625]], atol=1e-7)
    _assert(receiver_logits(receiver, beta, users, items), [1], dtype)
    loss = receiver_loss(receiver, beta, users, items, labels)
    gradient, = torch.autograd.grad(loss, beta, retain_graph=True)
    analytic = (1 / (1 + math.exp(-1)) - 1) * (tokens * p.detach().numpy()[0]).sum(axis=1)
    _assert(gradient, analytic, dtype)
    if dtype == torch.float64:
        def objective(v):
            logit = 1 + ((v[:, None] * tokens).sum(axis=0) * np.array([0.5, 1])).sum()
            return np.logaddexp(0, -logit)
        np.testing.assert_allclose(gradient.numpy(), _difference(objective, np.zeros(2)), atol=1e-7, rtol=0)
    loss.backward()
    assert p.grad is None and a.grad is None and bank.grad is None
    assert all(t.grad is None and t.requires_grad for t in public)
    assert not receiver.z.requires_grad and not receiver.source.delta.requires_grad


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_factors_dense_oracle_diagonal_and_four_hop_returns(dtype):
    paths = _paths(dtype=dtype)
    r, s, m = _dense(3, TRIPLES)
    np.testing.assert_array_equal(r, [[1, 0], [1, 1], [0, 1]])
    _assert(paths.weights, s[paths.item_rows, paths.auxiliary_columns], dtype)
    _assert(paths.h, (s * s).sum(axis=1), dtype)
    # Dense identity and M exist only in this three-item test oracle.
    identity = torch.eye(3, dtype=dtype)
    two, four = apply_path(paths, identity), apply_path(paths, identity, steps=2)
    _assert(two, m, dtype)
    _assert(four, m @ m, dtype)
    _assert(torch.diag(two), [0, 0, 0], dtype)
    assert torch.all(torch.diag(four) > 0)
    assert four[0, 2] > 0  # four-hop reach and return diagonals coexist


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("triples,local_ids", [([], ()), ([], (0, 1, 2)), ([(0, 0, 7), (1, 0, 8)], (0, 1, 2))])
def test_empty_isolated_and_single_item_auxiliary_columns(dtype, triples, local_ids):
    paths = _paths(triples, local_ids, dtype=dtype)
    assert not paths.structurally_valid
    y = torch.arange(len(local_ids) * 2, dtype=dtype).reshape(-1, 2)
    _assert(apply_path(paths, y), np.zeros_like(y), dtype)
    _assert(apply_path(paths, y, steps=2), np.zeros_like(y), dtype)
    source, _ = _fixture(dtype)
    source = prepare_source_kernel(paths, source.item_embeddings, source.weight, source.bias, source.basis)
    a = torch.zeros(2, dtype=dtype, requires_grad=True)
    _assert(make_operator_sketch(source, a), np.zeros((2, 2, 2)), dtype)
    p = torch.ones((1, 2), dtype=dtype)
    _assert(source_logits(source, p, a, torch.tensor([0]), torch.tensor([0])), [2], dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mixed_isolated_rows_local_mapping_and_batch_mean(dtype):
    triples = [(0, 0, 8), (2, 0, 8), (2, 1, 8), (3, 1, 8)]
    local_ids = (5, 0, 3, 2)  # isolated public item 0; arbitrary one-to-one order
    paths = _paths(triples, local_ids, n=7, dtype=dtype)
    e = torch.arange(14, dtype=dtype).reshape(7, 2) / 7
    basis = torch.ones((7, 1), dtype=dtype) / math.sqrt(7)
    source = prepare_source_kernel(paths, e, torch.tensor([0.7, -0.2], dtype=dtype), torch.tensor(0.3, dtype=dtype), basis)
    p = torch.tensor([[1.0, 0.5], [-0.1, 0.8]], dtype=dtype, requires_grad=True)
    a = torch.tensor([-0.6, 0.8], dtype=dtype, requires_grad=True)
    users = torch.tensor([1, 0, 1, 0, 1])
    items = torch.tensor([6, 5, 0, 3, 5])  # unsupported, supported, isolated, duplicate
    labels = torch.tensor([0, 1, 0, 1, 1], dtype=dtype)
    z, loss, sketch, _, _ = _source_oracle(source, p.detach().numpy(), a.detach().numpy(), users.numpy(), items.numpy(), labels.numpy(), triples)
    _assert(source_logits(source, p, a, users, items), z, dtype)
    _assert(source_loss(source, p, a, users, items, labels), loss, dtype)
    _assert(make_operator_sketch(source, a), sketch, dtype)
    assert torch.equal(source.f2[1], torch.zeros(2, dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_deduplication_edge_order_and_auxiliary_renumbering(dtype):
    original = _paths(dtype=dtype)
    reordered = [(i, r + 4, 100 - a * 7) for i, r, a in reversed(TRIPLES + [TRIPLES[0], TRIPLES[2]])]
    transformed = _paths(reordered, dtype=dtype)
    values = torch.tensor([[1, -2], [3, 5], [0, 7]], dtype=dtype)
    assert original.weights.numel() == transformed.weights.numel() == 4
    assert transformed.auxiliary_count == 2
    _assert(transformed.h, original.h.numpy(), dtype)
    _assert(apply_path(transformed, values, steps=2), apply_path(original, values, steps=2).numpy(), dtype)
    # Caller-owned graph/mapping inputs are copied, never rewritten.
    ids, incidence = torch.tensor([2, 0, 1]), _edges(TRIPLES)
    paths = prepare_path_factors(ids, incidence, num_items=3, dtype=dtype)
    ids.fill_(0)
    incidence.fill_(0)
    assert paths.local_item_ids.tolist() == [2, 0, 1]
    assert paths.item_rows.numel() == 4


@pytest.mark.parametrize("dtype", DTYPES)
def test_relation_separation_and_cross_relation_four_hop(dtype):
    # Equal local auxiliary identity under different relations is two columns.
    separated = _paths([(0, 0, 8), (1, 1, 8)], dtype=dtype)
    assert separated.auxiliary_count == 2 and not separated.structurally_valid
    _assert(apply_path(separated, torch.ones((3, 1), dtype=dtype)), np.zeros((3, 1)), dtype)
    triples = [(0, 0, 8), (1, 0, 8), (1, 1, 8), (2, 1, 8)]
    paths = _paths(triples, dtype=dtype)
    two = apply_path(paths, torch.eye(3, dtype=dtype))
    four = apply_path(paths, torch.eye(3, dtype=dtype), steps=2)
    assert two[0, 2].abs() < _tol(dtype)
    assert four[0, 2] > 0
    _assert(four, _dense(3, triples)[2] @ _dense(3, triples)[2], dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_local_degrees_are_recalculated_after_deletion(dtype):
    full = _paths(dtype=dtype)
    reduced_triples = [TRIPLES[0], TRIPLES[1]]
    reduced = _paths(reduced_triples, dtype=dtype)
    values = torch.eye(3, dtype=dtype)
    full_m, reduced_m = apply_path(full, values), apply_path(reduced, values)
    assert full_m[0, 1].item() == pytest.approx(1 / (2 * math.sqrt(2)), abs=_tol(dtype))
    assert reduced_m[0, 1].item() == pytest.approx(0.5, abs=_tol(dtype))
    _assert(reduced_m, _dense(3, reduced_triples)[2], dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_independent_sketch_channels_and_no_direct_user_gradient(dtype):
    source, _ = _fixture(dtype)
    a = torch.tensor([-0.4, 1.2], dtype=dtype, requires_grad=True)
    p = torch.ones((1, 2), dtype=dtype, requires_grad=True)
    sketch = make_operator_sketch(source, a)
    jac = torch.autograd.functional.jacobian(lambda v: make_operator_sketch(source, v), a)
    gates = torch.sigmoid(a.detach())
    _assert(jac[0, :, :, 0], sketch[0].detach().numpy() * (1 - gates[0].item()), dtype)
    _assert(jac[1, :, :, 1], sketch[1].detach().numpy() * (1 - gates[1].item()), dtype)
    assert torch.count_nonzero(jac[0, :, :, 1]) == 0
    assert torch.count_nonzero(jac[1, :, :, 0]) == 0
    assert torch.autograd.grad(sketch.sum(), p, allow_unused=True)[0] is None
    # Four-hop is independently gated, not the square of the gated two-hop.
    f4 = source.f4
    incorrectly_squared = gates[0].square() * f4
    correct = gates[1] * f4
    assert not torch.allclose(correct, incorrectly_squared)
    changed = a.detach().clone()
    changed[0] += 1
    assert torch.equal(make_operator_sketch(source, changed)[1], sketch[1].detach())
    changed = a.detach().clone()
    changed[1] += 1
    assert torch.equal(make_operator_sketch(source, changed)[0], sketch[0].detach())
    if dtype == torch.float64:
        for channel in (0, 1):
            for row in range(2):
                for column in range(2):
                    fd = _difference(lambda v: make_operator_sketch(source, torch.tensor(v)).detach().numpy()[channel, row, column], a.detach().numpy())
                    np.testing.assert_allclose(jac[channel, row, column].numpy(), fd, atol=1e-7, rtol=0)


def test_rank_one_projection_loss_does_not_invalidate_structure():
    source, _ = _fixture()
    source = prepare_source_kernel(source.paths, source.item_embeddings, source.weight, source.bias, torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64))
    sketch = make_operator_sketch(source, torch.zeros(2, dtype=torch.float64))
    assert source.paths.structurally_valid
    assert sketch[0].item() == pytest.approx(0, abs=1e-15)
    assert torch.count_nonzero(source.f2) > 0
    # Valid graph can project to zero in BOTH channels (B on an isolated row).
    paths = _paths([(0, 0, 0), (1, 0, 0)])
    isolated_basis = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float64)
    source = prepare_source_kernel(paths, source.item_embeddings, source.weight, source.bias, isolated_basis)
    assert paths.structurally_valid
    assert torch.count_nonzero(make_operator_sketch(source, torch.zeros(2, dtype=torch.float64))) == 0
    assert torch.count_nonzero(source.f2) > 0


@pytest.mark.parametrize("fixed_a", [False, True])
def test_test_owned_source_adam_step_and_final_snapshot(fixed_a):
    source, public = _fixture()
    p = torch.tensor([[1, 0.5]], dtype=torch.float64, requires_grad=True)
    a = torch.zeros(2, dtype=torch.float64, requires_grad=not fixed_a)
    before_a = a.detach().clone()
    old_t = detached_operator_snapshot(source, a)
    optimizer = torch.optim.Adam([p] if fixed_a else [p, a], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    loss = source_loss(source, p, a, torch.tensor([0]), torch.tensor([0]), torch.ones(1, dtype=torch.float64))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    fresh = detached_operator_snapshot(source, a)
    assert not fresh.requires_grad and fresh.grad_fn is None
    assert torch.equal(before_a, a.detach()) == fixed_a
    assert torch.equal(old_t, fresh) == fixed_a
    _assert(fresh, _source_oracle(source, p.detach().numpy(), a.detach().numpy(), [0], [0], [1])[2])
    assert all(t.grad is None for t in public)


@pytest.mark.parametrize("dtype", DTYPES)
def test_receiver_snapshot_z_graph_local_residual_and_beta_only_step(dtype):
    source, public = _fixture(dtype)
    a = torch.tensor([-0.4, 0.7], dtype=dtype, requires_grad=True)
    p = torch.tensor([[0.5, 1]], dtype=dtype, requires_grad=True)
    snapshot = freeze_source_snapshot(source, a)
    # Even a caller-supplied differentiable D must not connect fusion to source.
    snapshot = replace(snapshot, delta=snapshot.delta.clone().requires_grad_())
    bank = detached_operator_snapshot(source, a)
    receiver = prepare_receiver(snapshot, p, bank)
    beta = torch.zeros(2, dtype=dtype, requires_grad=True)
    source_beta0 = source_logits(source, p, a, torch.tensor([0]), torch.tensor([0])).detach()
    assert torch.equal(receiver_logits(receiver, beta, torch.tensor([0]), torch.tensor([0])), source_beta0)
    full_x = source.item_embeddings.clone()
    full_x[source.paths.local_item_ids] += snapshot.delta.detach()
    _assert(receiver.z, (source.basis.T @ full_x).numpy(), dtype)
    expected_tokens = np.stack([source.basis.numpy()[0] @ t @ receiver.z.numpy() for t in bank.numpy()])
    _assert(receiver_tokens(receiver, torch.tensor([0]))[0], expected_tokens, dtype)
    tracked = [p, a, snapshot.delta, bank, receiver.z, receiver.bank, receiver.source.delta, *public]
    copies = [t.detach().clone() for t in tracked]
    flags = [t.requires_grad for t in tracked]
    optimizer = torch.optim.Adam([beta], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    optimizer.zero_grad()
    receiver_loss(receiver, beta, torch.tensor([0]), torch.tensor([0]), torch.ones(1, dtype=dtype)).backward()
    optimizer.step()
    assert torch.count_nonzero(beta) == 2
    assert all(torch.equal(t.detach(), old) for t, old in zip(tracked, copies))
    assert [t.requires_grad for t in tracked] == flags
    assert all(t.grad is None for t in tracked)
    # Receiver independently owns small bank and p buffers; public E stays shared.
    assert receiver.source.item_embeddings.data_ptr() == public[0].data_ptr()
    assert receiver.bank.data_ptr() != bank.data_ptr()
    assert receiver.user_embeddings.data_ptr() != p.data_ptr()


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_bank_exact_local_fallback_with_unconstrained_beta(dtype):
    source, _ = _fixture(dtype)
    a = torch.tensor([0.7, -0.2], dtype=dtype, requires_grad=True)
    p = torch.tensor([[0.5, 1]], dtype=dtype, requires_grad=True)
    receiver = prepare_receiver(freeze_source_snapshot(source, a), p, torch.zeros((2, 2, 2), dtype=dtype))
    beta = torch.tensor([-12, 30], dtype=dtype, requires_grad=True)
    users, items = torch.tensor([0, 0]), torch.tensor([0, 2])
    assert torch.equal(receiver_logits(receiver, beta, users, items), source_logits(source, p, a, users, items).detach())
    loss = receiver_loss(receiver, beta, users, items, torch.tensor([1, 0], dtype=dtype))
    loss.backward()
    assert torch.equal(beta.grad, torch.zeros_like(beta))


@pytest.mark.parametrize("bad_ids", [[0, 0], [-1], [3]])
def test_reject_noninjective_or_out_of_range_local_mapping(bad_ids):
    with pytest.raises(ValueError):
        _paths([], bad_ids)


@pytest.mark.parametrize("incidence", [torch.zeros((2, 2), dtype=torch.long), torch.tensor([[3, 0, 0]]), torch.tensor([[0, -1, 0]]), torch.tensor([[0, 0, -1]]), torch.zeros((1, 3))])
def test_reject_noncanonical_incidence(incidence):
    with pytest.raises(ValueError):
        prepare_path_factors(torch.arange(3), incidence, num_items=3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["nonorthogonal", "nonfinite", "wrong_rows", "zero_rank", "over_rank", "wrong_dtype"])
def test_reject_invalid_explicit_basis(dtype, kind):
    source, _ = _fixture(dtype)
    basis = source.basis.clone()
    if kind == "nonorthogonal":
        basis[:, 0] *= 2
    elif kind == "nonfinite":
        basis[0, 0] = float("nan")
    elif kind == "wrong_rows":
        basis = basis[:2]
    elif kind == "zero_rank":
        basis = basis[:, :0]
    elif kind == "over_rank":
        basis = torch.zeros((3, 4), dtype=dtype)
    else:
        basis = basis.to(torch.int64)
    with pytest.raises(ValueError):
        prepare_source_kernel(source.paths, source.item_embeddings, source.weight, source.bias, basis)


@pytest.mark.parametrize("dtype", DTYPES)
def test_basis_validation_tolerances(dtype):
    source, _ = _fixture(dtype)
    tolerance = 1e-10 if dtype == torch.float64 else 1e-5
    basis = source.basis.clone()
    basis[0, 0] += tolerance / 10
    prepare_source_kernel(source.paths, source.item_embeddings, source.weight, source.bias, basis)
    basis[0, 0] += tolerance * 10
    with pytest.raises(ValueError, match="orthonormal"):
        prepare_source_kernel(source.paths, source.item_embeddings, source.weight, source.bias, basis)


@pytest.mark.parametrize("field", ["e", "w", "b"])
@pytest.mark.parametrize("bad", ["shape", "nonfinite", "dtype"])
def test_public_input_validation(field, bad):
    source, _ = _fixture()
    values = {"e": source.item_embeddings.clone(), "w": source.weight.clone(), "b": source.bias.clone()}
    if bad == "shape":
        values[field] = values[field].reshape(-1, 1)
    elif bad == "nonfinite":
        values[field].fill_(float("inf"))
    else:
        values[field] = values[field].float()
    with pytest.raises(ValueError):
        prepare_source_kernel(source.paths, values["e"], values["w"], values["b"], source.basis)


@pytest.mark.parametrize("kind", ["empty", "unaligned", "user", "item", "negative", "ids_dtype", "label", "label_shape", "p_shape", "p_nan", "a_shape", "a_nan"])
def test_batch_private_and_label_validation(kind):
    source, _ = _fixture()
    p, a = torch.ones((1, 2), dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    users, items, labels = torch.tensor([0]), torch.tensor([0]), torch.ones(1, dtype=torch.float64)
    if kind == "empty":
        users, items, labels = users[:0], items[:0], labels[:0]
    elif kind == "unaligned":
        users = torch.tensor([0, 0])
    elif kind == "user":
        users = torch.tensor([1])
    elif kind == "item":
        items = torch.tensor([3])
    elif kind == "negative":
        items = torch.tensor([-1])
    elif kind == "ids_dtype":
        items = items.float()
    elif kind == "label":
        labels[0] = 0.5
    elif kind == "label_shape":
        labels = labels.reshape(1, 1)
    elif kind == "p_shape":
        p = p[:, :1]
    elif kind == "p_nan":
        p.fill_(float("nan"))
    elif kind == "a_shape":
        a = a[:1]
    else:
        a.fill_(float("nan"))
    with pytest.raises(ValueError):
        source_loss(source, p, a, users, items, labels)


@pytest.mark.parametrize("kind", ["attached_bank", "bank_shape", "bank_nan", "beta_shape", "beta_nan", "empty", "label", "p_shape", "delta_shape"])
def test_receiver_boundary_validation(kind):
    source, _ = _fixture()
    snapshot = freeze_source_snapshot(source, torch.zeros(2, dtype=torch.float64))
    p, bank = torch.ones((1, 2), dtype=torch.float64), torch.zeros((2, 2, 2), dtype=torch.float64)
    if kind == "attached_bank":
        bank.requires_grad_()
    elif kind == "bank_shape":
        bank = bank[:1]
    elif kind == "bank_nan":
        bank.fill_(float("nan"))
    elif kind == "p_shape":
        p = p[:, :1]
    elif kind == "delta_shape":
        snapshot = replace(snapshot, delta=snapshot.delta[:1])
    if kind in ("attached_bank", "bank_shape", "bank_nan", "p_shape", "delta_shape"):
        with pytest.raises(ValueError):
            prepare_receiver(snapshot, p, bank)
        return
    receiver = prepare_receiver(snapshot, p, bank)
    beta = torch.zeros(2, dtype=torch.float64)
    users, items, labels = torch.tensor([0]), torch.tensor([0]), torch.ones(1, dtype=torch.float64)
    if kind == "beta_shape":
        beta = beta[:1]
    elif kind == "beta_nan":
        beta.fill_(float("nan"))
    elif kind == "empty":
        users, items, labels = users[:0], items[:0], labels[:0]
    else:
        labels.fill_(float("inf"))
    with pytest.raises(ValueError):
        receiver_loss(receiver, beta, users, items, labels)


@pytest.mark.parametrize("step", [0, 3, True])
def test_reject_unsupported_path_steps(step):
    with pytest.raises(ValueError):
        apply_path(_paths(), torch.zeros((3, 1), dtype=torch.float64), steps=step)


def test_stable_bce_for_large_logits():
    paths = _paths([], (), n=1)
    source = prepare_source_kernel(paths, torch.tensor([[1000.0]], dtype=torch.float64), torch.ones(1, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64), torch.ones((1, 1), dtype=torch.float64))
    loss = source_loss(source, torch.ones((1, 1), dtype=torch.float64), torch.zeros(2, dtype=torch.float64), torch.tensor([0]), torch.tensor([0]), torch.zeros(1, dtype=torch.float64))
    assert torch.isfinite(loss) and loss.item() == 1000


def test_production_source_inspection():
    text = inspect.getsource(path_operator) + inspect.getsource(u4_kernel)
    for forbidden in (".to_dense(", "one_hot(", "DatasetBundle", "GraphPartition", "torch.optim", "from src.methods.", "from src.models."):
        assert forbidden not in text
    # Code review additionally checks each allocation/product; this guard is
    # only a regression sentinel, not a proof about every possible execution.
    assert "aux.index_add_(" in text and "propagated.index_add_(" in text


class _AllocationTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = set()
        self.max_tensor_bytes = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        output = function(*args, **(kwargs or {}))
        def visit(value):
            if isinstance(value, torch.Tensor):
                self.shapes.add(tuple(value.shape))
                self.max_tensor_bytes = max(self.max_tensor_bytes, value.numel() * value.element_size())
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)
        visit(output)
        return output


def _retained_buffers(*objects):
    buffers = {}
    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            if storage.nbytes():
                buffers[storage.data_ptr()] = storage.nbytes()
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
    for value in objects:
        visit(value)
    return sum(buffers.values())


def test_bounded_synthetic_resource_probe():
    n, m, q, d, p, batch = 8192, 2048, 512, 8, 4, 32
    rows = torch.arange(m)
    triples = torch.stack([rows.repeat_interleave(2), torch.zeros(m * 2, dtype=torch.long), torch.stack([rows % q, (rows + 1) % q], dim=1).reshape(-1)], dim=1)
    e = torch.sin(torch.arange(n * d, dtype=torch.float64).reshape(n, d) / 100).float()
    grid = torch.arange(n, dtype=torch.float64)[:, None] + 0.5
    basis = torch.cos(grid * torch.arange(p, dtype=torch.float64) * math.pi / n) * math.sqrt(2 / n)
    basis[:, 0] /= math.sqrt(2)
    basis = basis.float()
    weight, bias = torch.ones(d), torch.tensor(0.0)
    public_bytes = _retained_buffers(e, basis, weight, bias)
    trace = _AllocationTrace()
    with trace:
        paths = prepare_path_factors(rows * 3, triples, num_items=n)
        source = prepare_source_kernel(paths, e, weight, bias, basis)
        a = torch.zeros(2, requires_grad=True)
        bank = detached_operator_snapshot(source, a)
        private_p = torch.ones((2, d), requires_grad=True)
        users, items = torch.arange(batch) % 2, torch.arange(batch) * 7
        labels = (torch.arange(batch) % 2).float()
        source_loss(source, private_p, a, users, items, labels).backward()
        receiver = prepare_receiver(freeze_source_snapshot(source, a), private_p, bank)
        beta = torch.zeros(2, requires_grad=True)
        receiver_loss(receiver, beta, users, items, labels).backward()
    # These are shape/ownership checks, never performance thresholds.
    assert (n, n) not in trace.shapes and (m, m) not in trace.shapes
    assert (m, q) not in trace.shapes  # no dense incidence either
    assert source.f2.shape == source.f4.shape == (m, d)
    assert bank.shape == (2, p, p)
    assert source.item_embeddings.data_ptr() == receiver.source.item_embeddings.data_ptr() == e.data_ptr()
    retained = _retained_buffers(source, receiver, bank)
    print("U4_K0_RESOURCE_PROBE=" + json.dumps({
        "catalogue": n, "local_items": m, "relation_aux_columns": paths.auxiliary_count,
        "deduplicated_edges": paths.weights.numel(), "embedding_dimension": d,
        "basis_rank": p, "batch": batch, "dtype": "float32",
        "public_numeric_buffer_bytes": public_bytes,
        "retained_source_receiver_bank_bytes_including_shared_public": retained,
        "retained_bytes_excluding_shared_public": retained - public_bytes,
        "largest_observed_tensor_bytes": trace.max_tensor_bytes,
        "observed_tensor_shapes": sorted(trace.shapes, key=lambda shape: (len(shape), shape)),
        "process_peak_rss_kib_including_imports_and_other_tests": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "exclusions": "not a live allocator peak; excludes Python objects, allocator workspace, setup inputs, optimizer state, and other parties; no learning/resource qualification",
    }, sort_keys=True))
