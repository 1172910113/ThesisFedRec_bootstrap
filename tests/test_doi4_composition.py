"""PW7 pure pri-none composition tests; fixed data, no training or optimizer run."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
import inspect
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

import src.methods.doi4.composition as composition
import src.methods.doi4.parameters as parameters
from src.methods.doi4.fusion import center_fusion
from src.methods.doi4.recommendation import local_recommendation_logits

# This exact literal is retained externally as FROZEN_FIXTURE.json before the
# first focused invocation. Expected values below never call production math.
FROZEN_FIXTURE_TEXT = r'''
{
  "fixture_id": "PW7-B-COMPOSITION-K0-explicit-v1",
  "dimensions": {"d": 2, "r": 2, "h": 3, "K": 2, "sources": 2, "E": 3},
  "tolerances": {"float64_atol": 1e-12, "float64_rtol": 1e-12, "float32_atol": 1e-5, "float32_rtol": 1e-5, "central_difference_step": 1e-6, "gradient_atol": 2e-7, "gradient_rtol": 2e-6, "nonzero_response_threshold": 1e-12},
  "floats": {
    "source0_field_embeddings": [[0.7, -0.4], [-0.6, 0.9], [1.1, 0.2]],
    "source0_value_embeddings": [[0.15, -0.25], [-0.3, 0.4]],
    "source0_w_key": [[0.8, -0.3], [0.2, 0.9], [-0.5, 0.6]],
    "source0_w_value": [[0.7, 0.2], [-0.4, 0.8], [0.3, -0.9]],
    "source1_field_embeddings": [[-0.8, 0.5], [0.4, 1.2], [0.9, -0.7]],
    "source1_value_embeddings": [[-0.2, 0.3], [0.35, -0.15]],
    "source1_w_key": [[-0.4, 0.7], [0.9, 0.1], [0.5, -0.6]],
    "source1_w_value": [[0.6, -0.8], [0.3, 0.7], [-0.5, 0.4]],
    "queries": [[0.6, -0.5, 0.8], [-0.7, 0.9, 0.2]],
    "fusion_w_query": [[0.7, -0.4], [-0.3, 0.8], [0.5, 0.6]],
    "fusion_w_key": [[0.8, -0.3, 0.2], [0.1, 0.7, -0.4], [-0.5, 0.2, 0.9]],
    "fusion_w_value": [[0.4, 0.6, -0.2], [-0.7, 0.3, 0.8], [0.5, -0.4, 0.7]],
    "fusion_w_output": [[0.9, 0.2, -0.3], [-0.4, 0.8, 0.1], [0.3, -0.6, 0.7]],
    "w_fused": [[0.6, -0.4, 0.8], [-0.5, 0.7, 0.3]],
    "user_embeddings": [[0.7, -0.6], [0.4, 0.9], [-0.8, 0.5]],
    "item_embeddings": [[0.6, -0.7], [-0.5, 0.8], [0.9, 0.3]],
    "score_weight": [0.8, -0.9],
    "score_bias": 0.12
  },
  "source0_ids": [[0, 1, 2], [2, 0, -1], [1, 2, 0]],
  "source0_codes": [[0, 1, 0], [1, 0, 1], [0, 1, 1]],
  "source0_mask": [[true, true, true], [true, true, false], [true, false, true]],
  "source1_ids": [[2, 0, 1], [1, 2, 0], [0, 1, 2]],
  "source1_codes": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
  "source1_mask": [[true, true, false], [true, true, true], [true, true, true]],
  "labels": [1.0, 0.0, 1.0],
  "group_offsets": [0, 2, 3],
  "selected_entries": [["source0_field_embeddings", [0, 1]], ["source0_value_embeddings", [0, 0]], ["source0_w_key", [1, 0]], ["source0_w_value", [2, 1]], ["source1_field_embeddings", [1, 0]], ["source1_value_embeddings", [1, 1]], ["source1_w_key", [2, 0]], ["source1_w_value", [0, 1]], ["queries", [0, 0]], ["fusion_w_query", [0, 0]], ["fusion_w_key", [1, 2]], ["fusion_w_value", [2, 1]], ["fusion_w_output", [0, 2]], ["w_fused", [1, 0]], ["user_embeddings", [0, 1]], ["item_embeddings", [0, 0]], ["score_weight", [1]], ["score_bias", []]],
  "oracle": "Independent NumPy scalar loops: F+V; stable masked softmax for T and center F; GMF with no private contribution; logaddexp(0,logit)-label*logit; mean within positive groups then across groups. No production kernel constructs expected values.",
  "other_cases": "Fixed literal or zero/one boundary tensors in the retained test source: E0/P0/J0, allmasked, P1, observed0, L0/L1/identicalL2; two receivers sharing one H; repeated gather [2,0,2,1] and weights [0.7,-0.2,1.1,0.5]; separate item direct/query identities; g1 versus gr zero adapter; no RNG or optimizer."
}
'''
FIXTURE = json.loads(FROZEN_FIXTURE_TEXT)
TOLERANCES = FIXTURE["tolerances"]


def _arrays():
    return {name: np.array(value, dtype=np.float64)
            for name, value in FIXTURE["floats"].items()}


def _tensors(dtype=torch.float64):
    return {name: torch.tensor(value, dtype=dtype, requires_grad=True)
            for name, value in FIXTURE["floats"].items()}


def _source_inputs(source):
    return tuple(torch.tensor(FIXTURE[f"source{source}_{suffix}"], dtype=dtype)
                 for suffix, dtype in (("ids", torch.int64), ("codes", torch.int64),
                                       ("mask", torch.bool)))


def _source(tensors, source):
    return composition.encode_readout_binary_units(
        *_source_inputs(source),
        field_embeddings=tensors[f"source{source}_field_embeddings"],
        value_embeddings=tensors[f"source{source}_value_embeddings"],
        queries=tensors["queries"], w_key=tensors[f"source{source}_w_key"],
        w_value=tensors[f"source{source}_w_value"],
    )


def _score_kwargs(tensors):
    return {
        "user_embeddings": tensors["user_embeddings"],
        "item_embeddings": tensors["item_embeddings"],
        "w_query": tensors["fusion_w_query"], "w_key": tensors["fusion_w_key"],
        "w_value": tensors["fusion_w_value"], "w_output": tensors["fusion_w_output"],
        "w_fused": tensors["w_fused"], "score_weight": tensors["score_weight"],
        "score_bias": tensors["score_bias"],
    }


def _graph(tensors):
    sources = [_source(tensors, source) for source in range(2)]
    packed = torch.cat([entry[0] for entry in sources], dim=1)
    mask = torch.cat([entry[1][:, None].expand(-1, 2) for entry in sources], dim=1)
    loss_output = composition.fuse_score_group_loss_pri_none(
        packed, mask, torch.tensor(FIXTURE["labels"], dtype=packed.dtype),
        torch.tensor(FIXTURE["group_offsets"], dtype=torch.int64),
        **_score_kwargs(tensors),
    )
    return sources, packed, mask, loss_output


def _mv(matrix, vector):
    """Scalar-loop matrix/vector multiply, independent of torch kernels."""
    return np.array([sum(float(row[j]) * float(vector[j]) for j in range(len(vector)))
                     for row in matrix], dtype=np.float64)


def _softmax(scores):
    maximum = max(scores)
    exp_values = [np.exp(float(value) - maximum) for value in scores]
    denominator = sum(exp_values)
    return [value / denominator for value in exp_values]


def _oracle(arrays):
    """Explicit masked attention and stable grouped BCE, including original masks."""
    h, k, entities = 3, 2, 3
    source_outputs = []
    for source in range(2):
        ids = FIXTURE[f"source{source}_ids"]
        codes = FIXTURE[f"source{source}_codes"]
        mask = FIXTURE[f"source{source}_mask"]
        field = arrays[f"source{source}_field_embeddings"]
        value = arrays[f"source{source}_value_embeddings"]
        keys = arrays[f"source{source}_w_key"]
        values = arrays[f"source{source}_w_value"]
        output, attention = np.zeros((entities, k, h)), np.zeros((entities, k, 3))
        for e in range(entities):
            valid = [p for p in range(3) if mask[e][p]]
            x = {p: [float(field[ids[e][p], j]) + float(value[codes[e][p], j])
                     for j in range(2)] for p in valid}
            for slot in range(k):
                scores = [sum(float(arrays["queries"][slot, j]) * _mv(keys, x[p])[j]
                              for j in range(h)) / np.sqrt(h) for p in valid]
                weights = _softmax(scores)
                for p, weight in zip(valid, weights):
                    attention[e, slot, p] = weight
                    output[e, slot] += weight * _mv(values, x[p])
        source_outputs.append((output, np.ones(entities, dtype=bool), attention))
    packed = np.concatenate([source[0] for source in source_outputs], axis=1)
    fused, attention = np.zeros((entities, h)), np.zeros((entities, 4))
    logits = np.zeros(entities)
    for e in range(entities):
        query = _mv(arrays["fusion_w_query"], arrays["item_embeddings"][e])
        keys = [_mv(arrays["fusion_w_key"], token) for token in packed[e]]
        values = [_mv(arrays["fusion_w_value"], token) for token in packed[e]]
        scores = [sum(query[j] * key[j] for j in range(h)) / np.sqrt(h) for key in keys]
        attention[e] = _softmax(scores)
        pooled = [sum(attention[e, p] * values[p][j] for p in range(4)) for j in range(h)]
        fused[e] = _mv(arrays["fusion_w_output"], pooled)
        residual = _mv(arrays["w_fused"], fused[e])
        logits[e] = float(arrays["score_bias"]) + sum(
            float(arrays["user_embeddings"][e, j])
            * (float(arrays["item_embeddings"][e, j]) + residual[j])
            * float(arrays["score_weight"][j]) for j in range(2)
        )
    labels = FIXTURE["labels"]
    offsets = FIXTURE["group_offsets"]
    example_losses = [np.logaddexp(0.0, float(logit)) - label * float(logit)
                      for logit, label in zip(logits, labels)]
    loss = sum(sum(example_losses[start:end]) / (end - start)
               for start, end in zip(offsets[:-1], offsets[1:])) / (len(offsets) - 1)
    return source_outputs, packed, logits, fused, attention, loss


def _assert_close(actual, expected, *, atol=1e-12, rtol=1e-12):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_full_chain_independent_oracle(dtype, record_property):
    expected = _oracle(_arrays())
    sources, packed, mask, actual = _graph(_tensors(dtype))
    atol = TOLERANCES[f"{str(dtype).split('.')[-1]}_atol"]
    rtol = TOLERANCES[f"{str(dtype).split('.')[-1]}_rtol"]
    errors = []
    for outputs, oracle_outputs in zip(sources, expected[0]):
        assert torch.equal(outputs[1], torch.tensor(oracle_outputs[1]))
        for index in (0, 2):
            _assert_close(outputs[index], oracle_outputs[index], atol=atol, rtol=rtol)
            errors.append(np.max(np.abs(outputs[index].detach().numpy() - oracle_outputs[index])))
    for tensor, oracle_value in zip((packed, actual[1], actual[2], actual[4], actual[0]),
                                    (expected[1], expected[2], expected[3], expected[4], expected[5])):
        _assert_close(tensor, oracle_value, atol=atol, rtol=rtol)
        errors.append(float(np.max(np.abs(tensor.detach().numpy() - oracle_value))))
    assert mask.shape == (3, 4) and bool(actual[3].all())
    assert not torch.equal(packed[:, 0], packed[:, 1])
    record_property("maximum_forward_absolute_error", float(max(errors)))


def test_selected_chain_central_differences(record_property):
    tensors, arrays = _tensors(), _arrays()
    loss = _graph(tensors)[3][0]
    gradients = dict(zip(tensors, torch.autograd.grad(loss, tuple(tensors.values()))))
    absolute_errors, relative_errors, derivative_records = [], [], []
    step = TOLERANCES["central_difference_step"]
    for name, raw_index in FIXTURE["selected_entries"]:
        index = tuple(raw_index)
        plus, minus = {k: v.copy() for k, v in arrays.items()}, {k: v.copy() for k, v in arrays.items()}
        plus[name][index] += step
        minus[name][index] -= step
        numerical = (_oracle(plus)[-1] - _oracle(minus)[-1]) / (2 * step)
        actual = float(gradients[name][index])
        np.testing.assert_allclose(actual, numerical,
                                   atol=TOLERANCES["gradient_atol"], rtol=TOLERANCES["gradient_rtol"])
        absolute_errors.append(abs(actual - numerical))
        derivative_records.append({"name": name, "index": raw_index, "autograd": actual, "central_difference": numerical, "absolute_error": abs(actual - numerical)})
        record_property(f"gradient_absolute_error__{name}", abs(actual - numerical))
        relative_errors.append(abs(actual - numerical) / max(abs(numerical), 1e-30))
    assert abs(float(gradients["queries"][0, 0])) > TOLERANCES["nonzero_response_threshold"]
    record_property("selected_entry_count", len(absolute_errors))
    record_property("selected_derivative_records", json.dumps(derivative_records, sort_keys=True))
    record_property("maximum_gradient_absolute_error", max(absolute_errors))
    record_property("maximum_gradient_relative_error", max(relative_errors))
    record_property("shared_query_selected_derivative", float(gradients["queries"][0, 0]))


def test_one_source_graph_reused_by_two_receivers_has_additive_gradients():
    tensors = _tensors()
    h, available, _ = _source(tensors, 0)
    mask = available[:, None].expand(-1, 2)
    first = composition.fuse_score_group_loss_pri_none(
        h, mask, torch.tensor(FIXTURE["labels"], dtype=h.dtype),
        torch.tensor(FIXTURE["group_offsets"]), **_score_kwargs(tensors))[0]
    second_kwargs = _score_kwargs(tensors)
    second_kwargs["user_embeddings"] = tensors["user_embeddings"] * -0.7 + 0.2
    second = composition.fuse_score_group_loss_pri_none(
        h, mask, torch.tensor(FIXTURE["labels"], dtype=h.dtype),
        torch.tensor(FIXTURE["group_offsets"]), **second_kwargs)[0]
    targets = (h, tensors["source0_field_embeddings"], tensors["source0_value_embeddings"],
               tensors["source0_w_key"], tensors["source0_w_value"], tensors["queries"])
    first_grad = torch.autograd.grad(first, targets, retain_graph=True)
    second_grad = torch.autograd.grad(second, targets, retain_graph=True)
    total_grad = torch.autograd.grad(first + second, targets)
    for total, left, right in zip(total_grad, first_grad, second_grad):
        torch.testing.assert_close(total, left + right, atol=1e-12, rtol=1e-12)


def test_outer_repeated_gathers_preserve_order_and_scatter_gradients():
    tensors = _tensors()
    sources = [_source(tensors, source) for source in range(2)]
    source_table = torch.cat([entry[0] for entry in sources], dim=1)
    indices = torch.tensor([2, 0, 2, 1], dtype=torch.int64)
    weights = torch.tensor([0.7, -0.2, 1.1, 0.5], dtype=torch.float64)
    kwargs = _score_kwargs(tensors)
    kwargs["item_embeddings"] = tensors["item_embeddings"][indices]
    kwargs["user_embeddings"] = tensors["user_embeddings"][indices]
    logits = composition.fuse_score_pri_none(
        source_table[indices], torch.ones((4, 4), dtype=torch.bool), **kwargs)[0]
    _assert_close(logits, _oracle(_arrays())[2][[2, 0, 2, 1]])
    targets = (source_table, tensors["item_embeddings"])
    separate = [torch.autograd.grad(logits[row] * weights[row], targets, retain_graph=True)
                for row in range(4)]
    together = torch.autograd.grad((logits * weights).sum(), targets)
    for index, actual in enumerate(together):
        torch.testing.assert_close(actual, sum(piece[index] for piece in separate), atol=1e-12, rtol=1e-12)
        assert torch.count_nonzero(separate[0][index][2]) > 0
        assert torch.count_nonzero(separate[2][index][2]) > 0
        torch.testing.assert_close(actual[2], separate[0][index][2] + separate[2][index][2],
                                   atol=1e-12, rtol=1e-12)


def test_same_item_direct_and_fusion_query_gradients_add(record_property):
    tensors = _tensors()
    packed = torch.tensor(_oracle(_arrays())[1], dtype=torch.float64)
    mask = torch.ones((3, 4), dtype=torch.bool)
    actual = composition.fuse_score_pri_none(packed, mask, **_score_kwargs(tensors))[0]
    total = torch.autograd.grad(actual.sum(), tensors["item_embeddings"])[0]
    direct = torch.tensor(FIXTURE["floats"]["item_embeddings"], dtype=torch.float64, requires_grad=True)
    query = torch.tensor(FIXTURE["floats"]["item_embeddings"], dtype=torch.float64, requires_grad=True)
    fused, available, _ = center_fusion(
        packed, mask, query, tensors["fusion_w_query"], tensors["fusion_w_key"],
        tensors["fusion_w_value"], tensors["fusion_w_output"])
    split = local_recommendation_logits(
        tensors["user_embeddings"], direct, fused, torch.zeros((3, 1), dtype=torch.float64),
        tensors["w_fused"], torch.zeros((2, 1), dtype=torch.float64),
        tensors["score_weight"], tensors["score_bias"], fused_available=available,
        private_available=torch.zeros(3, dtype=torch.bool))
    direct_gradient, query_gradient = torch.autograd.grad(split.sum(), (direct, query))
    torch.testing.assert_close(total, direct_gradient + query_gradient, atol=1e-12, rtol=1e-12)
    assert float(query_gradient.abs().max()) > TOLERANCES["nonzero_response_threshold"]
    record_property("item_query_maximum_absolute_derivative", float(query_gradient.abs().max()))


@pytest.mark.parametrize("entities,units,fields_count", [(0, 2, 3), (2, 0, 3), (2, 2, 0), (2, 2, 3)])
def test_source_empty_and_absent_paths_keep_connected_zero_gradients(entities, units, fields_count):
    field = torch.zeros((fields_count, 2), dtype=torch.float64, requires_grad=True)
    value = torch.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    tensors = _tensors()
    ids = torch.full((entities, units), -1, dtype=torch.int64)
    mask = torch.zeros((entities, units), dtype=torch.bool)
    output, available, attention = composition.encode_readout_binary_units(
        ids, torch.zeros_like(ids), mask, field_embeddings=field, value_embeddings=value,
        queries=tensors["queries"], w_key=tensors["source0_w_key"], w_value=tensors["source0_w_value"])
    assert output.shape == (entities, 2, 3) and attention.shape == (entities, 2, units)
    assert torch.count_nonzero(output) == 0 and not bool(available.any())
    targets = (field, value, tensors["queries"], tensors["source0_w_key"], tensors["source0_w_value"])
    for gradient in torch.autograd.grad(output.sum() + attention.sum(), targets):
        assert gradient is not None and torch.count_nonzero(gradient) == 0


def test_original_mask_observed_zero_and_singleton_zero_query_key(monkeypatch):
    original_mask = torch.tensor([[True, False]])
    seen = []
    original = composition.compatible_readout

    def observe(tokens, mask, *args):
        seen.append(mask)
        return original(tokens, mask, *args)

    monkeypatch.setattr(composition, "compatible_readout", observe)
    tensors = _tensors()
    field = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    value = torch.tensor([[0.0, 0.0], [0.3, -0.4]], dtype=torch.float64, requires_grad=True)
    h, available, attention = composition.encode_readout_binary_units(
        torch.tensor([[0, -1]]), torch.tensor([[0, 1]]), original_mask,
        field_embeddings=field, value_embeddings=value, queries=tensors["queries"],
        w_key=tensors["source0_w_key"], w_value=tensors["source0_w_value"])
    assert seen == [original_mask] and seen[0] is original_mask
    assert bool(available[0]) and torch.count_nonzero(h) == 0
    torch.testing.assert_close(attention, torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=h.dtype))
    gradients = torch.autograd.grad(h.sum(), (tensors["queries"], tensors["source0_w_key"], field, value))
    assert torch.count_nonzero(gradients[0]) == torch.count_nonzero(gradients[1]) == 0
    assert torch.count_nonzero(gradients[2]) > 0 and torch.count_nonzero(gradients[3][0]) > 0


@pytest.mark.parametrize("entities,length", [(0, 2), (2, 0), (2, 3)])
def test_receiver_empty_and_allmasked_keep_scoring_connected(entities, length):
    tensors = _tensors()
    tokens = torch.zeros((entities, length, 3), dtype=torch.float64, requires_grad=True)
    kwargs = _score_kwargs(tensors)
    kwargs["user_embeddings"] = tensors["user_embeddings"][:entities]
    kwargs["item_embeddings"] = tensors["item_embeddings"][:entities]
    logits, fused, available, attention = composition.fuse_score_pri_none(
        tokens, torch.zeros((entities, length), dtype=torch.bool), **kwargs)
    expected = (kwargs["user_embeddings"] * kwargs["item_embeddings"] * tensors["score_weight"]).sum(1)
    expected = expected + tensors["score_bias"]
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)
    assert fused.shape == (entities, 3) and attention.shape == (entities, length)
    assert torch.count_nonzero(fused) == 0 and not bool(available.any())
    targets = (tokens, tensors["fusion_w_query"], tensors["fusion_w_key"], tensors["fusion_w_value"],
               tensors["fusion_w_output"], tensors["w_fused"])
    for gradient in torch.autograd.grad(logits.sum(), targets):
        assert gradient is not None and torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("length", [1, 2])
def test_singleton_or_identical_slots_have_zero_query_key_derivatives(length):
    tensors = _tensors()
    tokens = torch.tensor([[[0.4, -0.7, 0.2]]], dtype=torch.float64).expand(3, length, 3)
    _, fused, available, attention = composition.fuse_score_pri_none(
        tokens, torch.ones((3, length), dtype=torch.bool), **_score_kwargs(tensors))
    assert bool(available.all())
    torch.testing.assert_close(attention, torch.full_like(attention, 1 / length), atol=0, rtol=0)
    for gradient in torch.autograd.grad(fused.sum(),
                                        (tensors["item_embeddings"], tensors["fusion_w_query"], tensors["fusion_w_key"])):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-15, rtol=0)


def test_g1_zero_adapter_is_plain_and_equivalent_to_gr(monkeypatch):
    seen = []
    original = composition.local_recommendation_logits

    def observe(*args, **kwargs):
        seen.append((args[3], args[5], kwargs["private_available"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(composition, "local_recommendation_logits", observe)
    tensors = _tensors()
    _, _, _, (_, logits, fused, available, _) = _graph(tensors)
    assert len(seen) == 1
    private_features, w_private, private_available = seen[0]
    assert private_features.shape == (3, 1) and w_private.shape == (2, 1)
    for tensor in (private_features, w_private, private_available):
        assert not tensor.requires_grad and not isinstance(tensor, torch.nn.Parameter)
        assert torch.count_nonzero(tensor) == 0 and tensor.device.type == "cpu"
    assert private_features.dtype == w_private.dtype == torch.float64
    gr_logits = local_recommendation_logits(
        tensors["user_embeddings"], tensors["item_embeddings"], fused,
        torch.zeros((3, 2), dtype=torch.float64), tensors["w_fused"],
        torch.zeros((2, 2), dtype=torch.float64), tensors["score_weight"], tensors["score_bias"],
        fused_available=available, private_available=torch.zeros(3, dtype=torch.bool))
    torch.testing.assert_close(logits, gr_logits, atol=0, rtol=0)
    targets = tuple(tensors.values())
    g1_gradients = torch.autograd.grad(logits.sum(), targets, retain_graph=True)
    gr_gradients = torch.autograd.grad(gr_logits.sum(), targets)
    for g1, gr in zip(g1_gradients, gr_gradients):
        torch.testing.assert_close(g1, gr, atol=0, rtol=0)


def test_caller_parameter_strided_nonleaf_values_grad_storage_versions_rng_unchanged():
    tensors = _tensors()
    tensors["source0_field_embeddings"] = torch.nn.Parameter(tensors["source0_field_embeddings"])
    backing = torch.empty((2, 6), dtype=torch.float64)
    backing[:, ::2] = tensors["queries"].detach()
    tensors["queries"] = backing[:, ::2].requires_grad_()
    item_leaf = tensors["item_embeddings"]
    tensors["item_embeddings"] = item_leaf * 1.25
    tensors["item_embeddings"].retain_grad()
    assert not tensors["queries"].is_contiguous() and not tensors["item_embeddings"].is_leaf
    all_inputs = list(tensors.values()) + [item_leaf]
    for tensor in all_inputs:
        tensor.grad = torch.full_like(tensor, 0.123)
    snapshots = [(tensor.detach().clone(), tensor.data_ptr(), tensor._version,
                  tensor.grad.clone(), tensor.grad.data_ptr()) for tensor in all_inputs]
    rng = torch.random.get_rng_state().clone()
    _, _, _, output = _graph(tensors)
    for tensor, (value, pointer, version, gradient, gradient_pointer) in zip(all_inputs, snapshots):
        assert torch.equal(tensor.detach(), value)
        assert tensor.data_ptr() == pointer and tensor._version == version
        assert torch.equal(tensor.grad, gradient) and tensor.grad.data_ptr() == gradient_pointer
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert output[0].requires_grad
    gradient = torch.autograd.grad(output[0], item_leaf)[0]
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("argument,value,error", [
    ("user_embeddings", [1.0], TypeError),
    ("user_embeddings", torch.tensor(1.0, dtype=torch.float64), ValueError),
    ("user_embeddings", torch.zeros((3, 2), dtype=torch.int64), TypeError),
    ("user_embeddings", torch.zeros((3, 1), dtype=torch.float64), ValueError),
    ("w_fused", torch.zeros((2, 2), dtype=torch.float64), ValueError),
    ("score_weight", torch.ones(3, dtype=torch.float64), ValueError),
    ("score_bias", torch.ones(1, dtype=torch.float64), ValueError),
])
def test_new_scoring_adapter_validates_before_placeholder_shape_use(argument, value, error, monkeypatch):
    tensors = _tensors()
    kwargs = _score_kwargs(tensors)
    kwargs[argument] = value
    tokens = torch.ones((3, 2, 3), dtype=torch.float64)
    mask = torch.ones((3, 2), dtype=torch.bool)

    def forbidden_zero_allocation(*args, **kwargs):
        raise AssertionError("invalid scoring inputs must fail before private placeholder allocation")

    monkeypatch.setattr(torch.Tensor, "new_zeros", forbidden_zero_allocation)
    monkeypatch.setattr(torch, "zeros", forbidden_zero_allocation)
    with pytest.raises(error):
        composition.fuse_score_pri_none(tokens, mask, **kwargs)


def test_source_wrapper_retains_full_finiteness_and_masked_code_validation():
    tensors = _tensors()
    ids, codes, mask = _source_inputs(0)
    codes[1, 2] = 2
    with pytest.raises(ValueError):
        composition.encode_readout_binary_units(
            ids, codes, mask, field_embeddings=tensors["source0_field_embeddings"],
            value_embeddings=tensors["source0_value_embeddings"], queries=tensors["queries"],
            w_key=tensors["source0_w_key"], w_value=tensors["source0_w_value"])
    field = torch.tensor([[float("nan"), 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        composition.encode_readout_binary_units(
            torch.tensor([[-1]]), torch.tensor([[0]]), torch.tensor([[False]]),
            field_embeddings=field, value_embeddings=tensors["source0_value_embeddings"],
            queries=tensors["queries"], w_key=tensors["source0_w_key"], w_value=tensors["source0_w_value"])


@pytest.mark.parametrize("labels,offsets,error", [
    ([1.0, 1.0, 1.0], [0, 2, 3], ValueError),
    ([0.0, 1.0, 1.0], [0, 2, 3], ValueError),
    ([1.0, 0.0, 1.0], [0, 0, 3], ValueError),
    ([1.0, 0.0, 1.0], [0, 2], ValueError),
    ([1.0, 0.0, 1.0], [], ValueError),
])
def test_loss_wrapper_rejects_bad_group_structure(labels, offsets, error):
    tensors = _tensors()
    with pytest.raises(error):
        composition.fuse_score_group_loss_pri_none(
            torch.ones((3, 2, 3), dtype=torch.float64), torch.ones((3, 2), dtype=torch.bool),
            torch.tensor(labels, dtype=torch.float64), torch.tensor(offsets, dtype=torch.int64),
            **_score_kwargs(tensors))


def test_loss_wrapper_positive_only_and_empty_and_wrong_dtypes():
    tensors = _tensors()
    tokens, mask = torch.ones((3, 2, 3), dtype=torch.float64), torch.ones((3, 2), dtype=torch.bool)
    output = composition.fuse_score_group_loss_pri_none(
        tokens, mask, torch.ones(3, dtype=torch.float64), torch.arange(4), **_score_kwargs(tensors))
    expected = np.mean([np.logaddexp(0.0, -float(x)) for x in output[1].detach()])
    _assert_close(output[0], expected)
    with pytest.raises(TypeError):
        composition.fuse_score_group_loss_pri_none(
            tokens, mask, torch.ones(3, dtype=torch.int64), torch.arange(4), **_score_kwargs(tensors))
    with pytest.raises(TypeError):
        composition.fuse_score_group_loss_pri_none(
            tokens, mask, torch.ones(3, dtype=torch.float64), torch.arange(4, dtype=torch.int32),
            **_score_kwargs(tensors))
    kwargs = _score_kwargs(tensors)
    kwargs["user_embeddings"], kwargs["item_embeddings"] = tensors["user_embeddings"][:0], tensors["item_embeddings"][:0]
    with pytest.raises(ValueError):
        composition.fuse_score_group_loss_pri_none(
            tokens[:0], mask[:0], torch.empty(0, dtype=torch.float64), torch.tensor([0]), **kwargs)


SPEC_ARGUMENTS = dict(party_id=7, public_item_count=5, local_user_count=4, local_field_count=3,
                      item_dim=2, unit_dim=2, compatible_dim=3, slot_count=2)
EXPECTED_SPECS = (
    ("item_embeddings", (5, 2), "center"), ("queries", (2, 3), "center"),
    ("fusion_w_query", (3, 2), "center"), ("fusion_w_key", (3, 3), "center"),
    ("fusion_w_value", (3, 3), "center"), ("fusion_w_output", (3, 3), "center"),
    ("w_fused", (2, 3), "center"), ("score_weight", (2,), "center"),
    ("score_bias", (), "center"), ("user_embeddings", (4, 2), "party"),
    ("field_embeddings", (3, 2), "party"), ("value_embeddings", (2, 2), "party"),
    ("readout_w_key", (3, 2), "party"), ("readout_w_value", (3, 2), "party"),
)


def test_exact_parameter_spec_order_owners_independent_user_domain_and_immutability():
    specs = parameters.doi4_parameter_specs(**SPEC_ARGUMENTS)
    assert type(specs) is tuple and len(specs) == 14
    assert tuple((record.name, record.shape, record.owner_scope) for record in specs) == EXPECTED_SPECS
    assert [field.name for field in fields(parameters.ParameterSpec)] == [
        "name", "shape", "owner_scope", "party_id", "trainable"]
    for record in specs:
        assert record.party_id == (None if record.owner_scope == "center" else 7)
        assert record.trainable is True and not hasattr(record, "__dict__")
        with pytest.raises(FrozenInstanceError):
            record.name = "other"
    other = parameters.doi4_parameter_specs(**{**SPEC_ARGUMENTS, "party_id": 99})
    assert specs[:9] == other[:9]
    assert all(left.party_id != right.party_id for left, right in zip(specs[9:], other[9:]))
    assert specs[9].shape[0] == 4 != specs[9].party_id
    assert not any("private" in record.name for record in specs)
    with pytest.raises(TypeError):
        parameters.ParameterSpec("score_bias", (), "center", None, True)
    with pytest.raises(TypeError):
        parameters.doi4_parameter_specs()


def test_parameter_shapes_zero_domains_and_capacity_without_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("parameter declarations must not allocate, initialize, sample or optimize")

    with monkeypatch.context() as patch:
        for name in ("tensor", "as_tensor", "empty", "zeros", "ones", "full", "rand", "randn", "manual_seed"):
            patch.setattr(torch, name, forbidden)
        patch.setattr(torch.nn, "Parameter", forbidden)
        patch.setattr(torch.optim, "Optimizer", forbidden)
        patch.setattr(torch.optim, "Adam", forbidden)
        patch.setattr(random, "random", forbidden)
        patch.setattr(np, "array", forbidden)
        patch.setattr(np.random, "default_rng", forbidden)
        zero = parameters.doi4_parameter_specs(**{
            **SPEC_ARGUMENTS, "public_item_count": 0, "local_user_count": 0, "local_field_count": 0})
        largest = 2**63 - 1
        boundary = parameters.doi4_parameter_specs(
            party_id=largest, public_item_count=largest, local_user_count=largest,
            local_field_count=largest, item_dim=1, unit_dim=1, compatible_dim=1, slot_count=largest)
    assert zero[0].shape == (0, 2) and zero[9].shape == (0, 2) and zero[10].shape == (0, 2)
    assert boundary[0].shape == boundary[9].shape == boundary[10].shape == (largest, 1)
    arithmetic = parameters.doi4_parameter_specs(
        party_id=7, public_item_count=5, local_user_count=1, local_field_count=6,
        item_dim=16, unit_dim=8, compatible_dim=8, slot_count=2)
    assert sum(np.prod(record.shape, dtype=object) for record in arithmetic[:9]) == 16 * 5 + 481
    assert sum(np.prod(record.shape, dtype=object) for record in arithmetic[9:]) == 208


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


@pytest.mark.parametrize("argument", tuple(SPEC_ARGUMENTS))
@pytest.mark.parametrize("value", [True, 1.0, np.int64(1), _IntSubclass(1)])
def test_parameter_factory_rejects_nonbuiltin_integers(argument, value):
    with pytest.raises(TypeError):
        parameters.doi4_parameter_specs(**{**SPEC_ARGUMENTS, argument: value})


@pytest.mark.parametrize("argument", tuple(SPEC_ARGUMENTS))
def test_parameter_factory_rejects_integer_range(argument):
    for value in (-1, 2**63):
        with pytest.raises(ValueError):
            parameters.doi4_parameter_specs(**{**SPEC_ARGUMENTS, argument: value})
    if argument in ("item_dim", "unit_dim", "compatible_dim", "slot_count"):
        with pytest.raises(ValueError):
            parameters.doi4_parameter_specs(**{**SPEC_ARGUMENTS, argument: 0})


@pytest.mark.parametrize("changes", [
    {"public_item_count": 2**63 - 1, "item_dim": 2},
    {"local_user_count": 2**63 - 1, "item_dim": 2},
    {"local_field_count": 2**63 - 1, "unit_dim": 2},
    {"slot_count": 2**63 - 1, "compatible_dim": 2},
    {"unit_dim": 2**63 - 1}, {"compatible_dim": 2**32},
])
def test_parameter_factory_rejects_python_integer_numel_overflow(changes):
    with pytest.raises(ValueError):
        parameters.doi4_parameter_specs(**{**SPEC_ARGUMENTS, **changes})


@pytest.mark.parametrize("changes,error", [
    ({"name": "unknown"}, ValueError), ({"name": _StrSubclass("item_embeddings")}, TypeError),
    ({"shape": [3, 2]}, TypeError), ({"shape": (True, 2)}, TypeError),
    ({"shape": (np.int64(3), 2)}, TypeError), ({"shape": (3,)}, ValueError),
    ({"shape": (-1, 2)}, ValueError), ({"shape": (3, 0)}, ValueError),
    ({"shape": (2**63 - 1, 2)}, ValueError), ({"shape": (2**63, 1)}, ValueError),
    ({"owner_scope": "party", "party_id": 7}, ValueError),
    ({"owner_scope": _StrSubclass("center")}, TypeError),
    ({"party_id": 7}, ValueError), ({"trainable": False}, ValueError), ({"trainable": 1}, TypeError),
])
def test_direct_parameter_record_rejects_invalid_immutable_contract(changes, error):
    kwargs = dict(name="item_embeddings", shape=(3, 2), owner_scope="center", party_id=None, trainable=True)
    with pytest.raises(error):
        parameters.ParameterSpec(**{**kwargs, **changes})


@pytest.mark.parametrize("name", ["fusion_w_key", "fusion_w_value", "fusion_w_output"])
def test_direct_parameter_record_rejects_nonsquare_fusion_maps(name):
    with pytest.raises(ValueError):
        parameters.ParameterSpec(name=name, shape=(2, 3), owner_scope="center", party_id=None, trainable=True)


def test_direct_parameter_record_party_and_fixed_value_table_boundaries():
    with pytest.raises(ValueError):
        parameters.ParameterSpec(name="value_embeddings", shape=(3, 2), owner_scope="party", party_id=7, trainable=True)
    for party_id, error in ((None, TypeError), (True, TypeError), (-1, ValueError), (2**63, ValueError)):
        with pytest.raises(error):
            parameters.ParameterSpec(name="user_embeddings", shape=(0, 2), owner_scope="party", party_id=party_id, trainable=True)


@pytest.mark.parametrize("module", [composition, parameters])
def test_new_modules_keep_method_math_and_declarations_free_of_execution_dependencies(module):
    tree = ast.parse(Path(inspect.getfile(module)).read_text())
    kernel_modules = {"binary_fields", "encoders", "fusion", "recommendation"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root in sys.stdlib_module_names or (module is composition and root == "torch")
        elif isinstance(node, ast.ImportFrom):
            imported = node.module or ""
            if node.level:
                assert module is composition and imported in kernel_modules
            else:
                root = imported.split(".")[0]
                assert (root in sys.stdlib_module_names or (module is composition and root == "torch")
                        or (module is composition and imported.removeprefix("src.methods.doi4.") in kernel_modules))
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in {"backward", "grad", "zero_grad", "step", "Adam", "Optimizer", "Parameter",
                                "manual_seed", "seed", "rand", "randn", "random", "open", "print",
                                "read_text", "write_text", "detach", "clone"}
    assert not any(isinstance(node, (ast.While, ast.AsyncFor)) for node in ast.walk(tree))


def test_nonzero_single_unit_readout_retains_formula_zero_query_and_key():
    tensors = _tensors()
    ids, codes, _ = _source_inputs(0)
    h, available, attention = composition.encode_readout_binary_units(
        ids[:, :1], codes[:, :1], torch.ones((3, 1), dtype=torch.bool),
        field_embeddings=tensors["source0_field_embeddings"],
        value_embeddings=tensors["source0_value_embeddings"], queries=tensors["queries"],
        w_key=tensors["source0_w_key"], w_value=tensors["source0_w_value"])
    assert bool(available.all()) and torch.count_nonzero(h) > 0
    torch.testing.assert_close(h[:, 0], h[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(attention, torch.ones_like(attention), atol=0, rtol=0)
    query_gradient, key_gradient, field_gradient = torch.autograd.grad(
        h.sum(), (tensors["queries"], tensors["source0_w_key"], tensors["source0_field_embeddings"]))
    assert query_gradient is not None and key_gradient is not None
    assert torch.count_nonzero(query_gradient) == torch.count_nonzero(key_gradient) == 0
    assert torch.count_nonzero(field_gradient) > 0


def test_receiver_masked_nonfinite_payload_is_still_rejected():
    tensors = _tensors()
    tokens = torch.zeros((3, 2, 3), dtype=torch.float64)
    tokens[0, 1, 2] = float("nan")
    with pytest.raises(ValueError):
        composition.fuse_score_pri_none(tokens, torch.zeros((3, 2), dtype=torch.bool), **_score_kwargs(tensors))


def test_wrappers_preserve_raw_unit_and_packed_token_values_storage_and_versions():
    tensors = _tensors()
    ids, codes, source_mask = _source_inputs(0)
    unit_inputs = (ids, codes, source_mask)
    before_source = [(tensor.clone(), tensor.data_ptr(), tensor._version) for tensor in unit_inputs]
    h, _, _ = composition.encode_readout_binary_units(
        ids, codes, source_mask, field_embeddings=tensors["source0_field_embeddings"],
        value_embeddings=tensors["source0_value_embeddings"], queries=tensors["queries"],
        w_key=tensors["source0_w_key"], w_value=tensors["source0_w_value"])
    for tensor, (value, pointer, version) in zip(unit_inputs, before_source):
        assert torch.equal(tensor, value)
        assert tensor.data_ptr() == pointer and tensor._version == version
    packed = torch.cat((h, h + 0.1), dim=1)
    packed_mask = torch.tensor([[True, False, True, True], [False, True, True, True],
                                [True, True, False, True]])
    packed_inputs = (packed, packed_mask)
    before_receiver = [(tensor.detach().clone(), tensor.data_ptr(), tensor._version)
                       for tensor in packed_inputs]
    result = composition.fuse_score_pri_none(packed, packed_mask, **_score_kwargs(tensors))
    assert result[0].requires_grad
    for tensor, (value, pointer, version) in zip(packed_inputs, before_receiver):
        assert torch.equal(tensor.detach(), value)
        assert tensor.data_ptr() == pointer and tensor._version == version
