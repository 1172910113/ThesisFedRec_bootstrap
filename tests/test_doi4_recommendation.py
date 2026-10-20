from __future__ import annotations

from itertools import product
from math import log
import random

import numpy as np
import pytest
import torch

from src.methods.doi4 import (
    center_fusion,
    compatible_readout,
    local_recommendation_logits,
    positive_group_bce,
)
from src.models.fedgmf import FedGMFLocalState, score_fedgmf

# Frozen before execution in api-before-implementation.txt and its fixture JSON.
STEP, ATOL64, RTOL64, TOL32 = 1e-6, 2e-7, 1e-6, 1e-5
NAMES = ("users", "items", "fused", "private", "w_fused", "w_private", "weight", "bias")


def _score_inputs(dtype=torch.float64, *, hand=False):
    if hand:
        values = ([[2, 1], [1, -1], [.5, 2]], [[1, 3], [2, 1], [-1, .5]],
                  [[1, 2], [3, 4], [-2, 1]], [[2], [-1], [3]],
                  [[1, 0], [0, .5]], [[.5], [-1]], [.5, -1], .25)
    else:
        values = ([[.2, -.7], [1.3, .4], [-.3, .8]], [[.6, -.2], [1., -.3], [-.5, .9]],
                  [[.4, .1, -.6], [-.2, .8, .3], [.7, -.1, .2]],
                  [[.3, -.5], [.8, .1], [-.2, .6]],
                  [[.7, -.4, .2], [-.1, .3, .8]], [[.8, -.2], [.3, .6]], [.7, -.3], .15)
    return [torch.tensor(value, dtype=dtype) for value in values], [
        torch.tensor([True, False, True]), torch.tensor([True, True, False])]


def _leaves(args):
    return [arg.detach().clone().requires_grad_(arg.is_floating_point()) for arg in args]


def _score(args, masks):
    return local_recommendation_logits(*args, fused_available=masks[0], private_available=masks[1])


def _score_oracle(args, masks):
    u, v, z, p, wf, wp, weight, bias = [arg.detach().numpy().astype(np.float64) for arg in args]
    fm, pm = [mask.numpy() for mask in masks]
    enhanced = v + np.where(fm[:, None], z, 0).dot(wf.T) + np.where(pm[:, None], p, 0).dot(wp.T)
    return np.einsum('ed,ed,d->e', u, enhanced, weight) + bias


def _loss_oracle(logits, labels, offsets):
    values = np.logaddexp(0., np.where(np.asarray(labels) == 1, -np.asarray(logits), np.asarray(logits)))
    return float(np.mean([values[start:end].mean() for start, end in zip(offsets[:-1], offsets[1:])]))


def _finite_difference_evidence(args, objective, oracle, names):
    leaves = _leaves(args)
    gradients = torch.autograd.grad(objective(leaves), leaves)
    evidence = {}
    for i, (name, gradient) in enumerate(zip(names, gradients)):
        expected = torch.empty_like(gradient)
        for index in product(*(range(size) for size in gradient.shape)):
            losses = []
            for sign in (1, -1):
                changed = [arg.detach().clone() for arg in args]
                changed[i][index] += sign * STEP
                losses.append(float(oracle(changed)))
            expected[index] = (losses[0] - losses[1]) / (2 * STEP)
        torch.testing.assert_close(gradient, expected, atol=ATOL64, rtol=RTOL64)
        evidence[name] = {'max_absolute_error': (gradient - expected).abs().max().item(),
                          'gradient_norm': gradient.norm().item()}
    return evidence


def _score_gradient_evidence():
    args, masks = _score_inputs()
    upstream = torch.tensor([.7, -1.2, .8], dtype=torch.float64)
    labels = torch.tensor([1., 0., 1.], dtype=torch.float64)
    offsets = torch.tensor([0, 2, 3])
    return {
        'weighted_logits': _finite_difference_evidence(args,
            lambda x: (_score(x, masks) * upstream).sum(),
            lambda x: (_score_oracle(x, masks) * upstream.numpy()).sum(), NAMES),
        'score_to_group_loss': _finite_difference_evidence(args,
            lambda x: positive_group_bce(_score(x, masks), labels, offsets),
            lambda x: _loss_oracle(_score_oracle(x, masks), labels.numpy(), offsets.tolist()), NAMES),
    }


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_hand_score_and_numpy_oracle(dtype):
    for hand in (False, True):
        args, masks = _score_inputs(dtype, hand=hand)
        actual = _score(args, masks)
        expected = torch.tensor(_score_oracle(args, masks), dtype=dtype)
        tolerance = TOL32 if dtype == torch.float32 else ATOL64
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        assert actual.shape == (3,) and actual.dtype == dtype and actual.device.type == 'cpu'
        if hand:
            assert torch.equal(actual, torch.tensor([1.25, 3., -2.5], dtype=dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_absent_branches_match_actual_public_gmf_on_matched_state_and_ids(dtype):
    args, _ = _score_inputs(dtype, hand=True)
    args[2].fill_(torch.finfo(dtype).max)
    args[3].fill_(torch.finfo(dtype).max)
    masks = [torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)]
    # Distinct party-local user IDs and catalogue item IDs; no ID inference.
    local = FedGMFLocalState(user_ids=torch.tensor([3, 17, 29]), user_embeddings=args[0])
    catalogue = torch.stack([args[1][2], args[1][0], args[1][1]])
    public = score_fedgmf({'item_embeddings': catalogue,
                          'prediction_weight': args[6].unsqueeze(0),
                          'prediction_bias': args[7].reshape(1)}, local,
                         [3, 17, 29], [1, 2, 0])
    actual = _score(args, masks)
    torch.testing.assert_close(actual, public, atol=TOL32 if dtype == torch.float32 else ATOL64,
                               rtol=TOL32 if dtype == torch.float32 else RTOL64)


def test_float64_all_score_inputs_and_loss_composition_finite_differences():
    evidence = _score_gradient_evidence()
    assert all(value['gradient_norm'] > 0 for section in evidence.values() for value in section.values())


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_unequal_positive_groups_stable_loss_and_analytic_gradient(dtype):
    logits = torch.tensor([0., log(3), -log(3), log(2)], dtype=dtype, requires_grad=True)
    labels = torch.tensor([1., 0., 0., 1.], dtype=dtype)
    offsets = torch.tensor([0, 3, 4])
    loss = positive_group_bce(logits, labels, offsets)
    expected = ((log(2) + log(4) + log(4 / 3)) / 3 + log(1.5)) / 2
    oracle = _loss_oracle(logits.detach().numpy(), labels.numpy(), offsets.tolist())
    tolerance = TOL32 if dtype == torch.float32 else ATOL64
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=dtype), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(loss, torch.tensor(oracle, dtype=dtype), atol=tolerance, rtol=tolerance)
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient, torch.tensor([-1 / 12, 1 / 8, 1 / 24, -1 / 6], dtype=dtype),
                               atol=tolerance, rtol=tolerance)
    plain_mean = np.logaddexp(0., np.where(labels.numpy() == 1, -logits.detach().numpy(), logits.detach().numpy())).mean()
    assert abs(loss.item() - plain_mean) > .01
    assert loss.ndim == 0 and loss.dtype == dtype and loss.device.type == 'cpu'


def test_loss_logits_finite_differences_and_caller_scaling_once():
    labels = torch.tensor([1., 0., 0., 1.], dtype=torch.float64)
    offsets = torch.tensor([0, 3, 4])
    args = [torch.tensor([0., log(3), -log(3), log(2)], dtype=torch.float64)]
    _finite_difference_evidence(args, lambda x: positive_group_bce(x[0], labels, offsets),
        lambda x: _loss_oracle(x[0].numpy(), labels.numpy(), offsets.tolist()), ['logits'])
    logits = args[0].requires_grad_()
    loss = positive_group_bce(logits, labels, offsets)
    gradient, = torch.autograd.grad(loss, logits, retain_graph=True)
    factor = 2 / 7
    scaled, = torch.autograd.grad(factor * loss, logits)
    torch.testing.assert_close(scaled, gradient * factor, atol=ATOL64, rtol=RTOL64)
    assert not torch.allclose(scaled, gradient * factor * factor)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('labels,offsets', [([1.], [0, 1]), ([1., 1., 1.], [0, 1, 2, 3]), ([1., 0.], [0, 2])])
def test_one_group_and_zero_negatives_are_valid(dtype, labels, offsets):
    logits = torch.zeros(len(labels), dtype=dtype, requires_grad=True)
    actual = positive_group_bce(logits, torch.tensor(labels, dtype=dtype), torch.tensor(offsets))
    torch.testing.assert_close(actual, torch.tensor(log(2), dtype=dtype))
    assert torch.isfinite(torch.autograd.grad(actual, logits)[0]).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_stable_loss_for_large_correct_and_incorrect_logits(dtype):
    labels = torch.tensor([1., 0.], dtype=dtype)
    for values in ([1000., -1000.], [-1000., 1000.]):
        logits = torch.tensor(values, dtype=dtype, requires_grad=True)
        actual = positive_group_bce(logits, labels, torch.tensor([0, 2]))
        expected = _loss_oracle(np.array(values), labels.numpy(), [0, 2])
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype), atol=TOL32, rtol=TOL32)
        assert torch.isfinite(torch.autograd.grad(actual, logits)[0]).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_empty_score_retains_all_eight_floating_gradient_paths(dtype):
    args, masks = _score_inputs(dtype)
    args[:4] = [arg[:0] for arg in args[:4]]
    masks = [mask[:0] for mask in masks]
    args = _leaves(args)
    result = _score(args, masks)
    assert result.shape == (0,) and result.dtype == dtype
    gradients = torch.autograd.grad(result.sum(), args)
    assert all(torch.equal(gradient, torch.zeros_like(arg)) for gradient, arg in zip(gradients, args))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('all_absent', [False, True])
def test_masked_large_finite_rows_and_exact_zero_gradients(dtype, all_absent):
    args, masks = _score_inputs(dtype)
    if all_absent:
        masks = [torch.zeros_like(mask) for mask in masks]
    changed = [arg.clone() for arg in args]
    for index, mask in zip((2, 3), masks):
        changed[index][~mask] = torch.finfo(dtype).max
    args, changed = _leaves(args), _leaves(changed)
    result, other = _score(args, masks), _score(changed, masks)
    assert torch.equal(result, other)
    gradients = torch.autograd.grad(result.sum(), args)
    other_gradients = torch.autograd.grad(other.sum(), changed)
    assert all(torch.equal(left, right) for left, right in zip(gradients, other_gradients))
    for index, mask in zip((2, 3), masks):
        assert torch.count_nonzero(gradients[index][~mask]) == 0
    if all_absent:
        for index in (2, 3, 4, 5):
            assert torch.count_nonzero(gradients[index]) == 0


def _composition_inputs():
    score, masks = _score_inputs(hand=True)
    values = [
        torch.tensor([[[.2, -.7], [1.3, .4]], [[-.3, .8], [.9, -.2]],
                      [[.5, -.1], [.2, .6]]], dtype=torch.float64),
        torch.tensor([[.6, -.2], [-.5, .9]], dtype=torch.float64),
        torch.tensor([[.7, -.4], [-.1, .8]], dtype=torch.float64),
        torch.tensor([[-.2, .9], [.8, -.6]], dtype=torch.float64),
        score[1],
        torch.tensor([[.7, -.4], [-.1, .8]], dtype=torch.float64),
        torch.tensor([[.8, -.2], [.3, .6]], dtype=torch.float64),
        torch.tensor([[-.2, .9], [.8, -.6]], dtype=torch.float64),
        torch.tensor([[.7, -.3], [.2, .8]], dtype=torch.float64),
        score[0], score[3], *score[4:],
    ]
    return values, masks[1]


COMPOSITION_NAMES = ('source', 'readout_queries', 'readout_key', 'readout_value', 'shared_items',
                     'fusion_query', 'fusion_key', 'fusion_value', 'fusion_output',
                     'users', 'private', 'w_fused', 'w_private', 'weight', 'bias')


def _reference_softmax(scores, dim):
    numerator = torch.exp(scores - scores.max(dim=dim, keepdim=True).values)
    return numerator / numerator.sum(dim=dim, keepdim=True)


def _compose(leaves, *, reference=False, split_items=False):
    # Every input is a caller-owned non-leaf. One item expression serves both paths.
    caller = [leaf * (1 + .02 * i) + .01 for i, leaf in enumerate(leaves)]
    source, queries, rk, rv, items, fq, fk, fv, fo, users, private, wf, wp, weight, bias = caller
    query_items, direct_items = items, items
    if split_items:
        assert reference
        query_items = items.detach().clone().requires_grad_()
        direct_items = items.detach().clone().requires_grad_()
    unit_mask = torch.tensor([[True, True], [True, False], [False, False]])
    available = unit_mask.any(1)
    private_mask = torch.tensor([True, True, False])
    if reference:
        messages, fused_rows = [], []
        for e in range(3):
            selected = unit_mask[e].nonzero(as_tuple=True)[0]
            if selected.numel():
                keys = torch.einsum('pd,hd->ph', source[e, selected], rk)
                values = torch.einsum('pd,hd->ph', source[e, selected], rv)
                readout_scores = torch.einsum('kh,ph->kp', queries, keys) / (2 ** .5)
                messages.append(torch.einsum('kp,ph->kh', _reference_softmax(readout_scores, 1), values))
            else:
                messages.append((source[e] @ rv.T).sum(0).expand(2, 2) * 0 + queries * 0)
        messages = torch.stack(messages)
        for e in range(3):
            if available[e]:
                query = torch.einsum('d,hd->h', query_items[e], fq)
                keys = torch.einsum('lh,jh->lj', messages[e], fk)
                values = torch.einsum('lh,jh->lj', messages[e], fv)
                scores = torch.einsum('h,lh->l', query, keys) / (2 ** .5)
                pooled = torch.einsum('l,lh->h', _reference_softmax(scores, 0), values)
                fused_rows.append(torch.einsum('h,jh->j', pooled, fo))
            else:
                fused_rows.append(messages[e].sum(0) * 0 + query_items[e] * 0)
        fused = torch.stack(fused_rows)
    else:
        messages, readout_available, _ = compatible_readout(source, unit_mask, queries, rk, rv)
        fused, available, _ = center_fusion(messages, readout_available[:, None].expand(3, 2),
                                            items, fq, fk, fv, fo)
    labels = torch.tensor([1., 0., 1.], dtype=torch.float64)
    offsets = torch.tensor([0, 2, 3])
    losses, logits_list = [], []
    for receiver_users, receiver_private in ((users, private), (users * .8 + .1, private * .6 - .2)):
        if reference:
            z = torch.where(available[:, None], fused, 0.)
            p = torch.where(private_mask[:, None], receiver_private, 0.)
            enhanced = direct_items + torch.einsum('eh,dh->ed', z, wf) + torch.einsum('eg,dg->ed', p, wp)
            logits = torch.einsum('ed,ed,d->e', receiver_users, enhanced, weight) + bias
            example_losses = torch.logaddexp(torch.zeros_like(logits), torch.where(labels == 1, -logits, logits))
            loss = (example_losses[:2].mean() + example_losses[2:].mean()) / 2
        else:
            logits = local_recommendation_logits(receiver_users, direct_items, fused, receiver_private,
                wf, wp, weight, bias, fused_available=available, private_available=private_mask)
            loss = positive_group_bce(logits, labels, offsets)
        losses.append(loss)
        logits_list.append(logits)
    return {'loss': losses[0] + losses[1], 'losses': losses, 'logits': logits_list,
            'caller': caller, 'split_items': (query_items, direct_items)}


def _composition_gradient_evidence():
    args, _ = _composition_inputs()
    leaves, expected_leaves = _leaves(args), _leaves(args)
    actual, expected = _compose(leaves), _compose(expected_leaves, reference=True)
    torch.testing.assert_close(actual['loss'], expected['loss'], atol=ATOL64, rtol=RTOL64)
    for left, right in zip(actual['logits'], expected['logits']):
        torch.testing.assert_close(left, right, atol=ATOL64, rtol=RTOL64)
    gradients = torch.autograd.grad(actual['loss'], [*leaves, *actual['caller']])
    expected_gradients = torch.autograd.grad(expected['loss'], [*expected_leaves, *expected['caller']])
    errors = {}
    for i, (gradient, reference) in enumerate(zip(gradients, expected_gradients)):
        torch.testing.assert_close(gradient, reference, atol=ATOL64, rtol=RTOL64)
        name = ('leaf:' if i < len(leaves) else 'nonleaf:') + COMPOSITION_NAMES[i % len(leaves)]
        errors[name] = (gradient - reference).abs().max().item()
    # Split only the test oracle's item input to isolate its two chain-rule paths.
    split = _compose(_leaves(args), reference=True, split_items=True)
    query_grad, direct_grad = torch.autograd.grad(split['loss'], split['split_items'])
    shared_grad = gradients[len(leaves) + 4]
    torch.testing.assert_close(shared_grad, query_grad + direct_grad, atol=ATOL64, rtol=RTOL64)
    assert query_grad.norm() > 0 and direct_grad.norm() > 0
    fd = _finite_difference_evidence(args, lambda x: _compose(x)['loss'],
        lambda x: _compose(x, reference=True)['loss'].item(), COMPOSITION_NAMES)
    return {'loss': actual['loss'].item(),
            'max_loss_error': abs(actual['loss'].item() - expected['loss'].item()),
            'gradient_errors': errors, 'central_differences': fd,
            'shared_item_sum_error': (shared_grad - query_grad - direct_grad).abs().max().item(),
            'query_item_gradient_norm': query_grad.norm().item(),
            'direct_item_gradient_norm': direct_grad.norm().item()}


def test_complete_T_F_score_group_loss_two_receivers_nonleaf_composition():
    evidence = _composition_gradient_evidence()
    assert all(record['gradient_norm'] > 0 for record in evidence['central_differences'].values())


def test_shared_parameters_accumulate_independent_receiver_contributions():
    args, masks = _score_inputs()
    args = _leaves(args)
    labels = torch.tensor([1., 0., 1.], dtype=torch.float64)
    offsets = torch.tensor([0, 2, 3])
    contributions = []
    for scale in (1., .8):
        local = [args[0] * scale, *args[1:]]
        loss = positive_group_bce(_score(local, masks), labels, offsets)
        contributions.append(torch.autograd.grad(loss, args))
    for scale in (1., .8):
        local = [args[0] * scale, *args[1:]]
        positive_group_bce(_score(local, masks), labels, offsets).backward()
    for leaf, first, second in zip(args, *contributions):
        torch.testing.assert_close(leaf.grad, first + second, atol=ATOL64, rtol=RTOL64)


def _strided(tensor):
    storage = torch.zeros((*tensor.shape, 2), dtype=tensor.dtype)
    storage[..., 0] = tensor
    return storage[..., 0].requires_grad_(tensor.is_floating_point())


def test_noncontiguous_inputs_and_cpu_autocast_preserve_dtype_and_gradients():
    args, masks = _score_inputs(torch.float32)
    strided, strided_masks = [_strided(arg) for arg in args], [_strided(mask) for mask in masks]
    expected = _score(args, masks)
    labels = _strided(torch.tensor([1., 0., 1.], dtype=torch.float32))
    offsets = _strided(torch.tensor([0, 2, 3]))
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        actual = _score(strided, strided_masks)
        loss = positive_group_bce(actual, labels, offsets)
    assert actual.dtype == loss.dtype == torch.float32
    assert torch.equal(actual, expected)
    expected_loss = positive_group_bce(expected, labels, offsets)
    assert torch.equal(loss, expected_loss)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(loss, strided))
    assert not strided[0].is_contiguous() and not offsets.is_contiguous()


def test_inputs_storage_versions_and_rng_are_preserved_by_both_kernels():
    args, masks = _score_inputs()
    args = _leaves(args)
    args[6] = torch.nn.Parameter(args[6])
    labels, offsets = torch.tensor([1., 0., 1.], dtype=torch.float64), torch.tensor([0, 2, 3])
    all_inputs = args + masks + [labels, offsets]
    copies = [arg.detach().clone() for arg in all_inputs]
    storage = [(id(arg), arg.untyped_storage().data_ptr(), arg._version) for arg in all_inputs]
    states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    result = _score(args, masks)
    result_copy, result_version, result_storage = result.clone(), result._version, result.untyped_storage().data_ptr()
    first, second = positive_group_bce(result, labels, offsets), positive_group_bce(result, labels, offsets)
    assert torch.equal(first, second)
    assert torch.equal(result, result_copy) and result._version == result_version
    assert result.untyped_storage().data_ptr() == result_storage
    for arg, copy, identity in zip(all_inputs, copies, storage):
        assert torch.equal(arg, copy) and (id(arg), arg.untyped_storage().data_ptr(), arg._version) == identity
        assert arg.grad is None
    assert random.getstate() == states[0]
    np.testing.assert_equal(np.random.get_state(), states[1])
    assert torch.equal(torch.get_rng_state(), states[2])


def _all_score_inputs():
    args, masks = _score_inputs()
    return args + masks


def _call_score(all_inputs):
    return _score(all_inputs[:8], all_inputs[8:])


@pytest.mark.parametrize('index', range(10))
def test_score_rejects_non_tensor_inputs(index):
    inputs = _all_score_inputs()
    inputs[index] = inputs[index].tolist()
    with pytest.raises(TypeError, match='torch.Tensor'):
        _call_score(inputs)


@pytest.mark.parametrize('index', range(10))
def test_score_rejects_wrong_rank(index):
    inputs = _all_score_inputs()
    inputs[index] = inputs[index].unsqueeze(0)
    with pytest.raises(ValueError, match='rank'):
        _call_score(inputs)


@pytest.mark.parametrize('index', range(10))
def test_score_rejects_non_cpu_without_gpu_execution(index):
    inputs = _all_score_inputs()
    inputs[index] = torch.empty(inputs[index].shape, dtype=inputs[index].dtype, device='meta')
    with pytest.raises(ValueError, match='CPU'):
        _call_score(inputs)


@pytest.mark.parametrize('index', (0, 2, 4, 8, 9))
def test_score_rejects_sparse_layout(index):
    inputs = _all_score_inputs()
    inputs[index] = inputs[index].to_sparse()
    with pytest.raises(TypeError, match='dense strided'):
        _call_score(inputs)


@pytest.mark.parametrize('index', range(8))
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.int64, torch.bool, torch.complex64])
def test_score_rejects_unsupported_floating_dtypes(index, dtype):
    inputs = _all_score_inputs()
    inputs[index] = inputs[index].to(dtype)
    with pytest.raises(TypeError, match='float32 or float64'):
        _call_score(inputs)


@pytest.mark.parametrize('index', range(8))
def test_score_rejects_mixed_floating_dtype(index):
    inputs = _all_score_inputs()
    inputs[index] = inputs[index].float()
    with pytest.raises(TypeError, match='same dtype'):
        _call_score(inputs)


@pytest.mark.parametrize('index', (8, 9))
@pytest.mark.parametrize('dtype', (torch.int64, torch.float64))
def test_score_rejects_non_boolean_masks(index, dtype):
    inputs = _all_score_inputs()
    inputs[index] = inputs[index].to(dtype)
    with pytest.raises(TypeError, match='Boolean'):
        _call_score(inputs)


@pytest.mark.parametrize('index', range(8))
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_score_rejects_nonfinite_inputs_even_when_empty(index, bad):
    args, masks = _score_inputs()
    args[index].reshape(-1)[0] = bad
    with pytest.raises(ValueError, match='finite'):
        _score(args, masks)
    if index >= 4:
        args[:4] = [arg[:0] for arg in args[:4]]
        with pytest.raises(ValueError, match='finite'):
            _score(args, [mask[:0] for mask in masks])


@pytest.mark.parametrize('index', (2, 3))
def test_score_rejects_nonfinite_masked_features(index):
    args, masks = _score_inputs()
    args[index][~masks[index - 2]] = torch.nan
    with pytest.raises(ValueError, match='finite'):
        _score(args, masks)


@pytest.mark.parametrize('index,shape,message', [
    (1, (2, 2), 'shape'), (1, (3, 3), 'shape'), (2, (2, 3), 'shape'),
    (3, (2, 2), 'shape'), (4, (3, 2), 'shape'), (5, (2, 3), 'shape'),
    (6, (3,), 'shape'), (8, (2,), 'shape'), (9, (2,), 'shape'),
    (0, (3, 0), 'positive'), (2, (3, 0), 'positive'), (3, (3, 0), 'positive')])
def test_score_rejects_malformed_dimensions(index, shape, message):
    inputs = _all_score_inputs()
    inputs[index] = torch.zeros(shape, dtype=inputs[index].dtype)
    with pytest.raises(ValueError, match=message):
        _call_score(inputs)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('stage', ['fused projection', 'private projection', 'item plus fused residual',
    'enhanced items', 'interactions', 'weighted interactions', 'summed scores', 'logits'])
def test_score_rejects_nonfinite_arithmetic(dtype, stage):
    largest = torch.finfo(dtype).max
    args = [torch.ones((1, 2), dtype=dtype), torch.ones((1, 2), dtype=dtype),
            torch.ones((1, 1), dtype=dtype), torch.ones((1, 1), dtype=dtype),
            torch.zeros((2, 1), dtype=dtype), torch.zeros((2, 1), dtype=dtype),
            torch.ones(2, dtype=dtype), torch.tensor(0., dtype=dtype)]
    masks = [torch.tensor([True]), torch.tensor([True])]
    if stage in ('fused projection', 'private projection'):
        index = 2 if stage == 'fused projection' else 3
        args[index].fill_(2)
        args[index + 2].fill_(largest)
    elif stage == 'item plus fused residual':
        args[1].fill_(largest)
        args[4].fill_(largest)
    elif stage == 'enhanced items':
        args[1].fill_(largest / 2)
        args[5].fill_(largest)
    elif stage == 'interactions':
        args[0].fill_(largest)
        args[1].fill_(2)
    elif stage == 'weighted interactions':
        args[1].fill_(largest)
        args[6].fill_(2)
    elif stage == 'summed scores':
        args[1].fill_(largest)
    else:
        args[1].fill_(largest / 4)
        args[7].fill_(largest)
    with pytest.raises(ValueError, match=stage + ' must be finite'):
        _score(args, masks)


def _loss_inputs():
    return [torch.tensor([.2, -.7, 1.1], dtype=torch.float64),
            torch.tensor([1., 0., 1.], dtype=torch.float64), torch.tensor([0, 2, 3])]


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('problem', ['list', 'rank', 'device', 'sparse'])
def test_loss_rejects_tensor_boundary_violations(index, problem):
    args = _loss_inputs()
    if problem == 'list':
        args[index] = args[index].tolist()
        error, message = TypeError, 'torch.Tensor'
    elif problem == 'rank':
        args[index] = args[index].unsqueeze(0)
        error, message = ValueError, 'rank'
    elif problem == 'device':
        args[index] = torch.empty(args[index].shape, dtype=args[index].dtype, device='meta')
        error, message = ValueError, 'CPU'
    else:
        args[index] = args[index].to_sparse()
        error, message = TypeError, 'dense strided'
    with pytest.raises(error, match=message):
        positive_group_bce(*args)


@pytest.mark.parametrize('index', (0, 1))
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.int64, torch.bool, torch.complex64])
def test_loss_rejects_unsupported_float_types(index, dtype):
    args = _loss_inputs()
    args[index] = args[index].to(dtype)
    with pytest.raises(TypeError, match='float32 or float64'):
        positive_group_bce(*args)


@pytest.mark.parametrize('index', (0, 1))
def test_loss_rejects_dtype_mismatch(index):
    args = _loss_inputs()
    args[index] = args[index].float()
    with pytest.raises(TypeError, match='same dtype'):
        positive_group_bce(*args)


@pytest.mark.parametrize('dtype', [torch.int32, torch.float32, torch.bool])
def test_loss_rejects_non_int64_offsets(dtype):
    args = _loss_inputs()
    args[2] = args[2].to(dtype)
    with pytest.raises(TypeError, match='int64'):
        positive_group_bce(*args)


@pytest.mark.parametrize('index', (0, 1))
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_loss_rejects_nonfinite_logits_or_labels(index, bad):
    args = _loss_inputs()
    args[index][0] = bad
    with pytest.raises(ValueError, match='finite'):
        positive_group_bce(*args)


@pytest.mark.parametrize('offsets', [[], [0], [1, 2, 3], [0, 2, 4], [0, 0, 3],
                                     [0, 2, 1, 3], [0, -1, 3], [0, 4, 3]])
def test_loss_rejects_malformed_offsets_and_empty_groups(offsets):
    args = _loss_inputs()
    args[2] = torch.tensor(offsets, dtype=torch.int64)
    with pytest.raises(ValueError):
        positive_group_bce(*args)


@pytest.mark.parametrize('labels', [[0., 0., 1.], [1., 1., 1.], [1., 0., 0.],
                                    [.5, 0., 1.], [1., -1., 1.], [1., 2., 1.]])
def test_loss_rejects_malformed_positive_first_groups_or_binary_labels(labels):
    args = _loss_inputs()
    args[1] = torch.tensor(labels, dtype=torch.float64)
    with pytest.raises(ValueError):
        positive_group_bce(*args)


def test_loss_rejects_mismatched_labels_and_empty_batch():
    args = _loss_inputs()
    with pytest.raises(ValueError, match='same shape'):
        positive_group_bce(args[0], args[1][:2], args[2])
    with pytest.raises(ValueError, match='nonempty'):
        positive_group_bce(args[0][:0], args[1][:0], torch.tensor([0, 0]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('stage', ['group losses', 'loss'])
def test_loss_rejects_finite_input_reduction_overflow(dtype, stage):
    largest = torch.finfo(dtype).max
    if stage == 'group losses':
        logits = torch.tensor([-largest, largest], dtype=dtype)
        labels, offsets = [1., 0.], [0, 2]
    else:
        logits = torch.tensor([-largest, -largest], dtype=dtype)
        labels, offsets = [1., 1.], [0, 1, 2]
    with pytest.raises(ValueError, match=stage + ' must be finite'):
        positive_group_bce(logits, torch.tensor(labels, dtype=dtype), torch.tensor(offsets))
