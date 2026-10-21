"""Local schema adaptation: independent semantic oracle and bounded composition."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import gc
import inspect
import json
import math
import random
import resource
import tracemalloc
import weakref

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from src.data import (
    GraphAccess, GraphPartition, GraphView, PartyData, close_graph_runtime,
    graph_domain_fingerprint, graph_view_fingerprint,
    prepare_graph_runtime, project_party_graphs,
)
from src.data import graph as graph_module
from src.data import graph_runtime as runtime_module
from src.methods import graph_semantic
from src.methods.graph_semantic import (
    GraphOperatorSchema, GraphRelationRule, PathFactors,
    apply_path, detached_operator_snapshot, freeze_source_snapshot,
    make_operator_sketch, prepare_path_factors_from_graph, prepare_receiver,
    prepare_source_kernel, receiver_logits, receiver_loss,
    source_logits, source_loss,
)
from src.methods.graph_semantic import graph_adapter


NODE_NAMES = ("item", "author", "genre", "other")
RELATION_NAMES = ("written", "written_inverse", "genre_link", "excluded", "written_alias")
PUBLIC_IDS = (1, 3, 6, 7)
# Independent PUBLIC semantic triples, not reconstructed from adapter output.
SEMANTIC = ((1, 10, "A"), (3, 10, "A"), (3, 10, "B"), (6, 10, "B"), (1, 20, "A"), (6, 20, "A"))
DTYPES = (torch.float64, torch.float32)


def _ids(values=()):
    return np.array(values, dtype=np.int64)


def _rule(relation_id, canonical=10, direction="item_to_aux"):
    return GraphRelationRule(relation_id=relation_id, canonical_relation_id=canonical, direction=direction)


def _schema(**changes):
    arguments = dict(
        node_type_names=NODE_NAMES, relation_names=RELATION_NAMES,
        item_node_type_id=0, aux_node_type_ids=(1, 2),
        relation_rules=(_rule(0), _rule(1, direction="aux_to_item"), _rule(2, 20), _rule(4)),
        ignored_relation_ids=(3,),
    )
    arguments.update(changes)
    return GraphOperatorSchema(**arguments)


def _view(*, inverse=True, alias=True):
    edges = [(3, 0, 0), (5, 0, 0), (5, 0, 4), (1, 0, 4), (3, 2, 0), (1, 2, 0)]
    if inverse:
        edges += [(0, 1, 3), (0, 1, 5), (4, 1, 5), (4, 1, 1)]
    if alias:
        edges += [(3, 4, 0)]
    edges += [(2, 3, 3), (4, 3, 2)]  # explicit ignored endpoints include other type
    return GraphView(
        name="local", node_raw_ids=("private-A", "i6", "person", "i1", "private-B", "i3", "i7"),
        node_type_ids=_ids([1, 0, 3, 0, 2, 0, 0]), node_type_names=NODE_NAMES,
        edge_index=_ids([[edge[0] for edge in edges], [edge[2] for edge in edges]]),
        relation_ids=_ids([edge[1] for edge in edges]), relation_names=RELATION_NAMES,
        item_to_node=_ids([-1, 3, -1, 5, -1, -1, 1, 6]), user_to_node=_ids([2, 3]),
        metadata={"ignored_relation_ids": [0, 1, 2, 4], "private": ["not schema"]},
    )


def _adapt(view, dtype=torch.float64, schema=None, n=8):
    return prepare_path_factors_from_graph(view, _schema() if schema is None else schema, num_items=n, dtype=dtype)


def _oracle(public_ids=PUBLIC_IDS, triples=SEMANTIC):
    # Independent binary R from explicit symbolic semantic incidence.
    keys = sorted({(relation, aux) for _, relation, aux in triples})
    r = np.zeros((len(public_ids), len(keys)), dtype=np.float64)
    for item, relation, aux in set(triples):
        r[public_ids.index(item), keys.index((relation, aux))] = 1
    di, da = r.sum(axis=1), r.sum(axis=0)
    inv_i, inv_a = np.zeros_like(di), np.zeros_like(da)
    inv_i[di > 0] = 1 / np.sqrt(di[di > 0])
    inv_a[da > 0] = 1 / np.sqrt(da[da > 0])
    s = inv_i[:, None] * r * inv_a[None, :]
    m = s @ s.T
    np.fill_diagonal(m, 0)
    return r, s, m, m @ m


def _model(dtype):
    e = torch.tensor([[.1, -.2], [2, 0], [.5, .7], [0, 1], [1, -1], [-.2, .5], [1, .5], [.3, -.4]], dtype=dtype)
    basis = torch.zeros((8, 2), dtype=dtype)
    basis[1, 0] = 1
    basis[[3, 6], 1] = 1 / math.sqrt(2)
    return e, torch.tensor([1., -.2], dtype=dtype), torch.tensor(.1, dtype=dtype), basis


def _assert(actual, expected, dtype):
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-7 if dtype == torch.float64 else 1e-5, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_semantic_dense_R_S_M_M2_source_sketch_oracle(dtype):
    paths = _adapt(_view(), dtype)
    assert type(paths) is PathFactors and paths.local_item_ids.tolist() == list(PUBLIC_IDS)
    assert paths.weights.numel() == 6 and paths.auxiliary_count == 3 and paths.structurally_valid
    r, s, m, four = _oracle()
    np.testing.assert_array_equal(r, [[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0]])
    np.testing.assert_allclose(s, r / 2, atol=1e-15)
    expected_m = np.full((4, 4), .25)
    expected_m[3, :] = expected_m[:, 3] = 0
    np.fill_diagonal(expected_m, 0)
    np.testing.assert_allclose(m, expected_m, atol=1e-15)
    identity = torch.eye(4, dtype=dtype)  # tiny dense oracle only
    two_actual, four_actual = apply_path(paths, identity), apply_path(paths, identity, steps=2)
    _assert(two_actual, m, dtype)
    _assert(four_actual, four, dtype)
    _assert(torch.diag(two_actual), np.zeros(4), dtype)
    assert torch.all(torch.diag(four_actual)[:3] > 0) and four_actual[3, 3] == 0
    e, w, bias, basis = _model(dtype)
    source = prepare_source_kernel(paths, e, w, bias, basis)
    a = torch.tensor([-.2, .8], dtype=dtype, requires_grad=True)
    p = torch.tensor([[1., .5], [-.3, .8]], dtype=dtype, requires_grad=True)
    users, items = torch.tensor([0, 1, 0, 1, 0]), torch.tensor([1, 7, 3, 0, 6])
    labels = torch.tensor([1, 0, 0, 1, 1], dtype=dtype)
    gates = 1 / (1 + np.exp(-a.detach().numpy().astype(np.float64)))
    e64, b64 = e.numpy().astype(np.float64), basis.numpy().astype(np.float64)
    f2, f4 = m @ e64[list(PUBLIC_IDS)], four @ e64[list(PUBLIC_IDS)]
    x = e64.copy()
    x[list(PUBLIC_IDS)] += gates[0] * f2 + gates[1] * f4
    expected_logits = (p.detach().numpy()[users] * x[items] * w.numpy()).sum(axis=1) + bias.item()
    expected_loss = np.mean(np.logaddexp(0, expected_logits) - labels.numpy() * expected_logits)
    bl = b64[list(PUBLIC_IDS)]
    expected_sketch = np.stack([gates[0] * bl.T @ m @ bl, gates[1] * bl.T @ four @ bl])
    logits = source_logits(source, p, a, users, items)
    loss = source_loss(source, p, a, users, items, labels)
    sketch = make_operator_sketch(source, a)
    _assert(source.f2, f2, dtype)
    _assert(source.f4, f4, dtype)
    _assert(logits, expected_logits, dtype)
    _assert(loss, expected_loss, dtype)
    _assert(sketch, expected_sketch, dtype)
    loss.backward()
    assert a.grad is not None and p.grad is not None
    print("ADAPTER_ORACLE=" + json.dumps({"dtype": str(dtype), "R": r.tolist(), "S": s.tolist(), "M": m.tolist(), "M2": four.tolist(), "expected_logits": expected_logits.tolist(), "expected_sketch": expected_sketch.tolist(), "max_apply_error": float(np.max(np.abs(two_actual.numpy()-m))), "max_four_error": float(np.max(np.abs(four_actual.numpy()-four))), "max_logit_error": float(np.max(np.abs(logits.detach().numpy()-expected_logits))), "max_sketch_error": float(np.max(np.abs(sketch.detach().numpy()-expected_sketch)))}))


def test_canonical_deduplication_precedes_existing_factory(monkeypatch):
    original = graph_adapter.prepare_path_factors
    calls = []
    def factory(ids, triples, **kwargs):
        calls.append((ids.clone(), triples.clone(), kwargs))
        return original(ids, triples, **kwargs)
    monkeypatch.setattr(graph_adapter, "prepare_path_factors", factory)
    paths = _adapt(_view())
    assert len(calls) == 1
    assert calls[0][0].tolist() == list(PUBLIC_IDS)
    assert calls[0][1].tolist() == [[0, 10, 0], [0, 20, 0], [1, 10, 0], [1, 10, 4], [2, 10, 4], [2, 20, 0]]
    assert paths.weights.numel() == 6


def _renumber(view):
    # New local row -> old local row, mixing item/aux/other rows.
    new_to_old = np.array([6, 4, 2, 1, 0, 5, 3])
    old_to_new = np.argsort(new_to_old)
    return replace(view,
        node_raw_ids=tuple(f"opaque-{i}" for i in range(7)),
        node_type_ids=view.node_type_ids[new_to_old],
        edge_index=old_to_new[view.edge_index],
        item_to_node=np.where(view.item_to_node < 0, -1, old_to_new[view.item_to_node]),
        user_to_node=old_to_new[view.user_to_node], metadata={"different": object()},
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("variation", ["forward_only", "edge_order", "node_renumber", "raw_IDs", "strides", "vocabulary_names"])
def test_operator_invariance_under_declared_storage_changes(dtype, variation):
    view, schema = _view(), _schema()
    baseline = _adapt(view, dtype)
    if variation == "forward_only":
        view = _view(inverse=False, alias=False)
    elif variation == "edge_order":
        view = replace(view, edge_index=view.edge_index[:, ::-1], relation_ids=view.relation_ids[::-1])
    elif variation == "node_renumber":
        view = _renumber(view)
    elif variation == "raw_IDs":
        view.node_raw_ids = tuple(100 + i for i in range(7))
    elif variation == "strides":
        for field in ("node_type_ids", "relation_ids", "item_to_node", "user_to_node"):
            value = getattr(view, field)
            setattr(view, field, np.repeat(value, 2)[::2])
    else:
        schema = replace(schema, node_type_names=("x", "y", "z", "w"), relation_names=("r4", "r3", "r2", "r1", "r0"))
        view = replace(view, node_type_names=schema.node_type_names, relation_names=schema.relation_names)
    actual = _adapt(view, dtype, schema)
    assert actual.local_item_ids.tolist() == baseline.local_item_ids.tolist()
    _assert(apply_path(actual, torch.eye(4, dtype=dtype), steps=2), apply_path(baseline, torch.eye(4, dtype=dtype), steps=2).numpy(), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_relation_separation_and_local_degree_change_are_not_erased(dtype):
    original = _adapt(_view(), dtype)
    # Intentionally different declared semantics merge relation 20 into 10.
    schema = _schema(relation_rules=(_rule(0), _rule(1, direction="aux_to_item"), _rule(2), _rule(4)))
    merged = _adapt(_view(), dtype, schema)
    assert merged.auxiliary_count == 2 and original.auxiliary_count == 3
    identity = torch.eye(4, dtype=dtype)
    assert not torch.allclose(apply_path(original, identity), apply_path(merged, identity))
    # Delete the (public item 3, relation 10, A) edge, using forward-only storage.
    view = _view(inverse=False, alias=False)
    keep = np.arange(view.relation_ids.size) != 1
    view = replace(view, edge_index=view.edge_index[:, keep], relation_ids=view.relation_ids[keep])
    reduced = _adapt(view, dtype)
    semantic = tuple(row for row in SEMANTIC if row != (3, 10, "A"))
    _assert(apply_path(reduced, identity), _oracle(triples=semantic)[2], dtype)
    assert not torch.allclose(original.h, reduced.h)
    assert not torch.allclose(apply_path(original, identity), apply_path(reduced, identity))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["None", "zero_nodes", "zero_edges", "no_mapped_items", "mapped_isolates", "singleton_aux"])
def test_absent_empty_and_zero_action_cases(dtype, case):
    schema = _schema()
    if case == "None":
        view = None
    elif case in ("zero_nodes", "no_mapped_items"):
        view = GraphView(name="empty", node_raw_ids=() if case == "zero_nodes" else ("a",), node_type_ids=_ids() if case == "zero_nodes" else _ids([1]), node_type_names=NODE_NAMES, edge_index=np.empty((2, 0), dtype=np.int64), relation_ids=_ids(), relation_names=RELATION_NAMES, item_to_node=np.full(8, -1, dtype=np.int64), metadata={})
    elif case in ("zero_edges", "mapped_isolates"):
        view = _view()
        view.edge_index = np.empty((2, 0), dtype=np.int64)
        view.relation_ids = _ids()
    else:
        view = _view(inverse=False, alias=False)
        # One item per canonical relation-aux column: both columns are singletons.
        select = np.array([0, 2])
        view.edge_index, view.relation_ids = view.edge_index[:, select], view.relation_ids[select]
    paths = _adapt(view, dtype, schema)
    expected_rows = 0 if case in ("None", "zero_nodes", "no_mapped_items") else 4
    assert paths.local_item_ids.numel() == expected_rows and not paths.structurally_valid
    y = torch.arange(expected_rows * 2, dtype=dtype).reshape(expected_rows, 2)
    _assert(apply_path(paths, y), np.zeros((expected_rows, 2)), dtype)
    _assert(apply_path(paths, y, steps=2), np.zeros((expected_rows, 2)), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_structural_validity_survives_zero_projected_T(dtype):
    paths = _adapt(_view(), dtype)
    e, w, bias, _ = _model(dtype)
    basis = torch.zeros((8, 2), dtype=dtype)
    basis[0, 0], basis[2, 1] = 1, 1  # entirely outside local support
    source = prepare_source_kernel(paths, e, w, bias, basis)
    assert paths.structurally_valid and torch.count_nonzero(source.f2) > 0
    assert torch.count_nonzero(make_operator_sketch(source, torch.zeros(2, dtype=dtype))) == 0


class _String(str):
    pass


class _Integer(int):
    pass


class _Array(np.ndarray):
    pass


@pytest.mark.parametrize("field,value", [
    ("node_type_names", ["item", "aux"]), ("node_type_names", ("item", "item")),
    ("node_type_names", ("item", " ")), ("node_type_names", (_String("item"), "aux")),
    ("relation_names", ["link"]), ("relation_names", ("x", "x")), ("relation_names", ("",)),
    ("item_node_type_id", True), ("item_node_type_id", np.int64(0)), ("item_node_type_id", _Integer(0)),
    ("item_node_type_id", -1), ("item_node_type_id", 99),
    ("aux_node_type_ids", ()), ("aux_node_type_ids", [1]), ("aux_node_type_ids", (0,)),
    ("aux_node_type_ids", (1, 1)), ("aux_node_type_ids", (np.int64(1),)), ("aux_node_type_ids", (True,)), ("aux_node_type_ids", (99,)),
    ("relation_rules", []), ("relation_rules", ()), ("relation_rules", (object(),)),
    ("ignored_relation_ids", [3]), ("ignored_relation_ids", (3, 3)), ("ignored_relation_ids", (True,)),
    ("ignored_relation_ids", (99,)), ("ignored_relation_ids", (0, 3)), ("ignored_relation_ids", ()),
])
def test_schema_declaration_rejection(field, value):
    with pytest.raises((TypeError, ValueError)):
        _schema(**{field: value})


@pytest.mark.parametrize("field,value", [("relation_id", True), ("relation_id", np.int64(0)), ("relation_id", -1), ("canonical_relation_id", False), ("canonical_relation_id", np.int64(1)), ("canonical_relation_id", -1), ("direction", "inverse"), ("direction", _String("item_to_aux"))])
def test_rule_declaration_rejection(field, value):
    arguments = dict(relation_id=0, canonical_relation_id=10, direction="item_to_aux")
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        GraphRelationRule(**arguments)


def test_duplicate_conflicting_and_out_of_range_rule_ids():
    for rules in ((_rule(0), _rule(0, 20)), (_rule(0), _rule(1, direction="aux_to_item"), _rule(2, 20), _rule(99))):
        with pytest.raises(ValueError):
            _schema(relation_rules=rules)


@pytest.mark.parametrize("view", [None, "zero_edges"])
@pytest.mark.parametrize("field,value", [("relation_rules", ()), ("aux_node_type_ids", ()), ("ignored_relation_ids", (3, 3))])
def test_schema_revalidated_even_for_absent_or_empty_input(view, field, value):
    schema = _schema()
    object.__setattr__(schema, field, value)  # test-owned malformed frozen record
    actual = None
    if view is not None:
        actual = _view()
        actual.edge_index, actual.relation_ids = np.empty((2, 0), dtype=np.int64), _ids()
    with pytest.raises((TypeError, ValueError)):
        _adapt(actual, schema=schema)


def test_frozen_keyword_only_exports_and_no_new_operator_type():
    assert graph_semantic.prepare_path_factors_from_graph is graph_adapter.prepare_path_factors_from_graph
    for record in (GraphRelationRule, GraphOperatorSchema):
        assert all(parameter.kind == inspect.Parameter.KEYWORD_ONLY for parameter in inspect.signature(record).parameters.values())
    rule, schema = _rule(0), _schema()
    assert not hasattr(schema, "__dict__") and not hasattr(rule, "__dict__")
    with pytest.raises(FrozenInstanceError):
        schema.item_node_type_id = 1
    with pytest.raises(FrozenInstanceError):
        rule.direction = "aux_to_item"
    assert type(_adapt(None)) is PathFactors


@pytest.mark.parametrize("kind", ["missing", "wrong_length", "many_to_one", "wrong_type", "unanchored", "no_mapping_but_users", "negative", "out_of_range", "rank"])
def test_mapping_rejection(kind):
    view = _view()
    if kind == "missing":
        view.item_to_node = None
    elif kind == "wrong_length":
        view.item_to_node = view.item_to_node[:-1]
    elif kind == "many_to_one":
        view.item_to_node[3] = view.item_to_node[1]
    elif kind == "wrong_type":
        view.item_to_node[0] = 0
    elif kind == "unanchored":
        view.item_to_node[7] = -1
    elif kind == "no_mapping_but_users":
        view.item_to_node[:] = -1
        view.user_to_node = _ids([3, 5, 1, 6])
    elif kind == "negative":
        view.item_to_node[0] = -2
    elif kind == "out_of_range":
        view.item_to_node[0] = 99
    else:
        view.item_to_node = view.item_to_node[:, None]
    with pytest.raises((TypeError, ValueError)):
        _adapt(view)


@pytest.mark.parametrize("field", ["node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node"])
@pytest.mark.parametrize("kind", ["subclass", "masked", "float", "metadata", "list"])
def test_numeric_array_type_rejection_without_laundering(field, kind):
    view = _view()
    original = getattr(view, field)
    if kind == "subclass":
        invalid = original.view(_Array)
    elif kind == "masked":
        invalid = np.ma.array(original, mask=False)
    elif kind == "float":
        invalid = original.astype(np.float64)
    elif kind == "metadata":
        invalid = original.view(np.dtype(np.int64, metadata={}))
    else:
        invalid = original.tolist()
    setattr(view, field, invalid)
    with pytest.raises(TypeError):
        _adapt(view)
    assert getattr(view, field) is invalid


@pytest.mark.parametrize("field", ["node_type_names", "relation_names"])
@pytest.mark.parametrize("kind", ["order", "content", "subclass"])
def test_exact_vocabulary_rejection(field, kind):
    view = _view()
    original = getattr(view, field)
    changed = original[::-1] if kind == "order" else ("changed",) + original[1:]
    if kind == "subclass":
        changed = tuple(_String(value) for value in original)
    setattr(view, field, changed)
    with pytest.raises((TypeError, ValueError)):
        _adapt(view)


@pytest.mark.parametrize("kind", ["wrong_item", "wrong_aux", "wrong_direction", "undeclared_inverse", "invalid_relation", "duplicate_storage", "bad_metadata"])
def test_actual_view_structure_and_selected_endpoint_rejection(kind):
    view, schema = _view(), _schema()
    if kind == "wrong_item":
        view.edge_index[0, 0] = 2
    elif kind == "wrong_aux":
        view.edge_index[1, 0] = 2
    elif kind == "wrong_direction":
        schema = _schema(relation_rules=(_rule(0, direction="aux_to_item"), _rule(1, direction="aux_to_item"), _rule(2, 20), _rule(4)))
    elif kind == "undeclared_inverse":
        view.relation_names += ("undeclared_inverse",)
        view.edge_index = np.column_stack((view.edge_index, [0, 3]))
        view.relation_ids = np.append(view.relation_ids, 5)
    elif kind == "invalid_relation":
        view.relation_ids[0] = 99
    elif kind == "duplicate_storage":
        view.edge_index = np.column_stack((view.edge_index, view.edge_index[:, 0]))
        view.relation_ids = np.append(view.relation_ids, view.relation_ids[0])
    else:
        view.metadata = []
    with pytest.raises((TypeError, ValueError)):
        _adapt(view, schema=schema)


@pytest.mark.parametrize("num_items", [True, np.int64(8), _Integer(8), 0, -1, 8.0])
def test_explicit_catalogue_count_rejection(num_items):
    with pytest.raises((TypeError, ValueError)):
        _adapt(None, n=num_items)


@pytest.mark.parametrize("dtype", [None, "float32", np.float32, torch.float16, torch.int64])
def test_model_dtype_rejection(dtype):
    with pytest.raises(ValueError):
        _adapt(None, dtype=dtype)


def test_no_schema_or_view_subclass_bypass():
    with pytest.raises(TypeError):
        prepare_path_factors_from_graph(None, object(), num_items=8, dtype=torch.float32)
    class ViewSubclass(GraphView):
        pass
    original = _view()
    changed = ViewSubclass(**{field.name: getattr(original, field.name) for field in fields(original)})
    with pytest.raises(TypeError):
        _adapt(changed)


def test_unused_relations_must_be_explicit_and_canonical_ids_need_not_be_contiguous():
    schema = _schema(relation_rules=tuple(replace(rule, canonical_relation_id=rule.canonical_relation_id + 2**40) for rule in _schema().relation_rules))
    original, actual = _adapt(_view()), _adapt(_view(), schema=schema)
    assert torch.equal(apply_path(original, torch.eye(4, dtype=torch.float64)), apply_path(actual, torch.eye(4, dtype=torch.float64)))
    empty_relations = GraphOperatorSchema(node_type_names=("item", "aux"), relation_names=(), item_node_type_id=0, aux_node_type_ids=(1,), relation_rules=(), ignored_relation_ids=())
    assert _adapt(None, schema=empty_relations).weights.numel() == 0


def test_explicit_ignore_precedes_degrees_and_metadata_never_supplies_rules():
    view = _view()
    original = _adapt(view)
    selected = view.relation_ids != 3
    no_ignored = replace(view, edge_index=view.edge_index[:, selected], relation_ids=view.relation_ids[selected], metadata={})
    assert torch.equal(original.h, _adapt(no_ignored).h)
    assert torch.equal(original.weights, _adapt(no_ignored).weights)
    # All relations may be deliberately ignored, without inferring an alternative.
    schema = _schema(relation_rules=(), ignored_relation_ids=tuple(range(5)))
    paths = _adapt(view, schema=schema)
    assert paths.local_item_ids.tolist() == list(PUBLIC_IDS)
    assert not paths.structurally_valid and paths.weights.numel() == 0


def test_inputs_not_mutated_and_factor_buffers_are_detached():
    view = _view()
    arrays = {name: getattr(view, name).copy() for name in ("node_type_ids", "edge_index", "relation_ids", "item_to_node", "user_to_node")}
    metadata = deepcopy(view.metadata)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    factors = _adapt(view)
    copies = {field.name: getattr(factors, field.name).clone() for field in fields(factors) if isinstance(getattr(factors, field.name), torch.Tensor)}
    for name, saved in arrays.items():
        np.testing.assert_array_equal(getattr(view, name), saved)
    assert view.metadata == metadata
    assert random.getstate() == py_state and torch.equal(torch.random.get_rng_state(), torch_state)
    later = np.random.get_state()
    assert later[0] == np_state[0] and later[2:] == np_state[2:]
    np.testing.assert_array_equal(later[1], np_state[1])
    for field in fields(factors):
        value = getattr(factors, field.name)
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad and value.ndim == 1
            assert all(not np.shares_memory(value.numpy(), getattr(view, name)) for name in arrays)
    for name in arrays:
        getattr(view, name).fill(0)
    view.metadata.clear()
    for name, saved in copies.items():
        assert torch.equal(getattr(factors, name), saved)


def test_no_retained_graph_payload_or_metadata():
    class Payload:
        pass
    view = _view()
    view.metadata["payload"] = Payload()
    array_refs = [weakref.ref(getattr(view, name)) for name in ("node_type_ids", "edge_index", "relation_ids", "item_to_node", "user_to_node")]
    payload_ref = weakref.ref(view.metadata["payload"])
    paths = _adapt(view)
    del view
    gc.collect()
    assert all(reference() is None for reference in array_refs) and payload_ref() is None
    assert all(isinstance(getattr(paths, field.name), (torch.Tensor, int, bool)) for field in fields(paths))


def _small_schema():
    return GraphOperatorSchema(node_type_names=("item", "aux"), relation_names=("link",), item_node_type_id=0, aux_node_type_ids=(1,), relation_rules=(_rule(0),), ignored_relation_ids=())


def _forbid_global_graph_work(monkeypatch):
    calls = {}
    names = ("graph_view_fingerprint", "graph_domain_fingerprint", "graph_partition_checksum", "validate_graph_partition", "_validate_graph_sources", "_build_graph_projection_index", "_build_graph_mapping_index")
    for module in (graph_module, runtime_module):
        for name in names:
            if not hasattr(module, name):
                continue
            key = module.__name__ + "." + name
            calls[key] = 0
            def forbidden(*args, _key=key, **kwargs):
                calls[_key] += 1
                raise AssertionError("global graph work during adaptation/batch: " + _key)
            monkeypatch.setattr(module, name, forbidden)
    return calls


@pytest.mark.parametrize("dtype", DTYPES)
def test_runtime_projection_adapter_source_sketch_receiver_composition(processed_bundle, monkeypatch, dtype):
    bundle = processed_bundle
    n = len(bundle.item_id_map)
    mapping = np.full(n, -1, dtype=np.int64)
    mapping[:3] = [0, 1, 2]
    view = GraphView(name="local", node_raw_ids=("i0", "i1", "i2", "private-A", "private-B"), node_type_ids=_ids([0, 0, 0, 1, 1]), node_type_names=("item", "aux"), edge_index=_ids([[0, 1, 0, 1, 2], [3, 3, 4, 4, 4]]), relation_ids=_ids([0] * 5), relation_names=("link",), item_to_node=mapping, metadata={})
    bundle.graph_views = {"local": view}
    access = {
        7: (GraphAccess(view_name="local", node_ids=_ids([0, 1, 3]), edge_ids=_ids([0, 1])),),
        901: (GraphAccess(view_name="local", node_ids=_ids([0, 1, 2, 4]), edge_ids=_ids([2, 3, 4])),),
    }
    partition = GraphPartition(name="explicit", party_access=access, source_graph_fingerprints={"local": graph_view_fingerprint(view)}, source_domain_fingerprint=graph_domain_fingerprint(bundle), metadata={})
    parties = tuple(PartyData(party_id=party_id, train_interaction_ids=_ids(), valid_interaction_ids=_ids(), test_interaction_ids=_ids(), user_ids=_ids(), item_ids=_ids(), feature_view_names=(), graph_view_names=("local",), owns_labels=False, metadata={}) for party_id in access)
    runtime = prepare_graph_runtime(bundle, partition, parties, source_id="synthetic", split_id="fixed", recipe_id="adapter-test")
    try:
        binding = runtime.binding_id
        global_calls = _forbid_global_graph_work(monkeypatch)
        delivered = [project_party_graphs(runtime, party_id=party_id, binding_id=binding)["local"] for party_id in (7, 901)]
        original_validate = GraphView.validate
        local_calls = []
        def validate(local_view):
            local_calls.append(id(local_view))
            return original_validate(local_view)
        monkeypatch.setattr(GraphView, "validate", validate)
        factors = [prepare_path_factors_from_graph(local_view, _small_schema(), num_items=n, dtype=dtype) for local_view in delivered]
        assert local_calls == [id(local_view) for local_view in delivered]
        assert [factor.local_item_ids.tolist() for factor in factors] == [[0, 1], [0, 1, 2]]
        for factor, public in zip(factors, ((0, 1), (0, 1, 2))):
            expected = _oracle(public, tuple((item, 10, "local-only") for item in public))[2]
            _assert(apply_path(factor, torch.eye(len(public), dtype=dtype)), expected, dtype)
        assert not set(access[7][0].edge_ids) & set(access[901][0].edge_ids)
        assert set(factors[0].local_item_ids.tolist()) & set(factors[1].local_item_ids.tolist()) == {0, 1}
        e = torch.arange(n * 2, dtype=dtype).reshape(n, 2) / 10
        basis = torch.zeros((n, 2), dtype=dtype)
        basis[0, 0], basis[1, 1] = 1, 1
        source = [prepare_source_kernel(factor, e, torch.ones(2, dtype=dtype), torch.tensor(0., dtype=dtype), basis) for factor in factors]
        a = torch.zeros(2, dtype=dtype, requires_grad=True)
        bank = detached_operator_snapshot(source[1], a)
        p = torch.tensor([[.5, 1.]], dtype=dtype, requires_grad=True)
        receiver = prepare_receiver(freeze_source_snapshot(source[0], a), p, bank)
        beta = torch.zeros(2, dtype=dtype, requires_grad=True)
        users, items = torch.tensor([0]), torch.tensor([0])
        for _ in range(3):  # test-owned batches; no production executor
            assert torch.equal(receiver_logits(receiver, beta, users, items), source_logits(source[0], p, a, users, items).detach())
        receiver_loss(receiver, beta, users, items, torch.ones(1, dtype=dtype)).backward()
        assert torch.count_nonzero(beta.grad) > 0 and a.grad is None and p.grad is None
        assert len(local_calls) == 2 and all(value == 0 for value in global_calls.values())
        print("ADAPTER_RUNTIME_COUNTS=" + json.dumps({"dtype": str(dtype), "parties": [7, 901], "projections": 2, "adapter_preparations": 2, "local_validations_inside_adapter": len(local_calls), "test_receiver_batches": 3, "global_calls_after_runtime_prepare": global_calls}))
    finally:
        close_graph_runtime(runtime)
    # The local factors remain independent numerical data after runtime close.
    assert torch.isfinite(apply_path(factors[0], torch.ones((2, 1), dtype=dtype))).all()


class _TensorTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes, self.max_bytes = set(), 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        output = function(*args, **(kwargs or {}))
        def record(value):
            if isinstance(value, torch.Tensor):
                self.shapes.add(tuple(value.shape))
                self.max_bytes = max(self.max_bytes, value.numel() * value.element_size())
            elif isinstance(value, (tuple, list)):
                for child in value:
                    record(child)
        record(output)
        return output


def test_bounded_compact_preparation_resource_and_call_observation(monkeypatch):
    n, m, q = 4096, 256, 64
    public = np.arange(m, dtype=np.int64) * 7
    mapping = np.full(n, -1, dtype=np.int64)
    mapping[public] = np.arange(m)
    items = np.repeat(np.arange(m, dtype=np.int64), 2)
    aux = m + np.column_stack((np.arange(m) % q, (np.arange(m) + 1) % q)).reshape(-1)
    edges = np.concatenate((np.stack((items, aux)), np.stack((aux, items))), axis=1)
    view = GraphView(name="bounded", node_raw_ids=tuple(range(m + q)), node_type_ids=np.concatenate((np.zeros(m, dtype=np.int64), np.ones(q, dtype=np.int64))), node_type_names=("item", "aux"), edge_index=edges, relation_ids=np.repeat(np.array([0, 1], dtype=np.int64), items.size), relation_names=("forward", "inverse"), item_to_node=mapping, metadata={})
    schema = GraphOperatorSchema(node_type_names=view.node_type_names, relation_names=view.relation_names, item_node_type_id=0, aux_node_type_ids=(1,), relation_rules=(_rule(0), _rule(1, direction="aux_to_item")), ignored_relation_ids=())
    global_calls = _forbid_global_graph_work(monkeypatch)
    original_validate, validations = GraphView.validate, []
    def validate(local):
        validations.append(id(local))
        return original_validate(local)
    monkeypatch.setattr(GraphView, "validate", validate)
    original_factory, factory_inputs = graph_adapter.prepare_path_factors, []
    def factory(ids, incidence, **kwargs):
        factory_inputs.append(tuple(incidence.shape))
        return original_factory(ids, incidence, **kwargs)
    monkeypatch.setattr(graph_adapter, "prepare_path_factors", factory)
    trace = _TensorTrace()
    tracemalloc.start()
    with trace:
        paths = prepare_path_factors_from_graph(view, schema, num_items=n, dtype=torch.float32)
        apply_path(paths, torch.ones((m, 3)), steps=2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert validations == [id(view)] and factory_inputs == [(512, 3)]
    assert paths.local_item_ids.numel() == m and paths.weights.numel() == 512 and paths.auxiliary_count == q
    assert all(value == 0 for value in global_calls.values())
    assert not any(shape in trace.shapes for shape in ((n, n), (m, m), (m, q)))
    buffers = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes() for field in fields(paths) if isinstance(value := getattr(paths, field.name), torch.Tensor)}
    assert all(getattr(paths, field.name).ndim == 1 for field in fields(paths) if isinstance(getattr(paths, field.name), torch.Tensor))
    text = inspect.getsource(graph_adapter)
    for forbidden in ("DatasetBundle", "GraphPartition", "graph_runtime", ".to_dense(", "from src.methods.doi4", "np.eye("):
        assert forbidden not in text
    print("ADAPTER_RESOURCE_OBSERVATION=" + json.dumps({"catalogue": n, "mapped_local_items": m, "local_nodes": m+q, "stored_edges": edges.shape[1], "deduplicated_incidence": 512, "relation_aux_columns": q, "dtype": "float32", "delivered_mapping_bytes": mapping.nbytes, "deduplicated_numpy_incidence_bytes": 512*3*8, "retained_factor_numeric_bytes": sum(buffers.values()), "retained_tensor_shapes": {field.name: list(value.shape) for field in fields(paths) if isinstance(value := getattr(paths, field.name), torch.Tensor)}, "observed_torch_shapes": sorted(trace.shapes, key=lambda shape:(len(shape),shape)), "largest_observed_torch_tensor_bytes": trace.max_bytes, "tracemalloc_peak_bytes": peak, "process_peak_rss_kib_including_imports_and_prior_tests": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "local_validation_calls": len(validations), "factory_calls": len(factory_inputs), "global_calls": global_calls, "limits": "O(n) mapping read and local structure validation; per-rule edge scans, sorting/dedup workspace and compact coordinate arrays; Torch trace misses NumPy/native workspace, tracemalloc misses untracked native/BLAS/PyTorch work; no graph-square matrix found by exercised shape trace and source inspection; no constant-memory/performance/learning qualification"}, sort_keys=True))


# PW3-C-P2-int64-r1: canonical labels use the accepted CPU long incidence.
_CANONICAL_CONTEXTS = ("None", "zero_nodes", "zero_edges", "mapped_isolates", "selected_edges", "rule_absent_locally")


def _canonical_boundary_case(context, canonical):
    other = 1 if canonical == 0 else 0
    schema = _schema(relation_rules=(
        _rule(0, canonical), _rule(1, canonical, "aux_to_item"),
        _rule(2, other), _rule(4, canonical),
    ))
    if context == "None":
        return None, schema
    view = _view()
    if context in ("zero_nodes", "zero_edges"):
        view = replace(
            view, node_raw_ids=() if context == "zero_nodes" else ("aux",),
            node_type_ids=_ids() if context == "zero_nodes" else _ids([1]),
            edge_index=np.empty((2, 0), dtype=np.int64), relation_ids=_ids(),
            item_to_node=np.full(8, -1, dtype=np.int64), user_to_node=_ids(),
        )
    elif context == "mapped_isolates":
        view = replace(view, edge_index=np.empty((2, 0), dtype=np.int64), relation_ids=_ids())
    elif context == "rule_absent_locally":
        keep = np.isin(view.relation_ids, [2, 3])
        view = replace(view, edge_index=view.edge_index[:, keep], relation_ids=view.relation_ids[keep])
    return view, schema


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("context", _CANONICAL_CONTEXTS)
@pytest.mark.parametrize("canonical", [0, 2**40, 2**63 - 2, 2**63 - 1])
def test_p2_canonical_range_supported_labels(dtype, context, canonical, monkeypatch):
    view, schema = _canonical_boundary_case(context, canonical)
    original, calls = graph_adapter.prepare_path_factors, []
    def factory(ids, triples, **kwargs):
        calls.append(triples.clone())
        return original(ids, triples, **kwargs)
    monkeypatch.setattr(graph_adapter, "prepare_path_factors", factory)
    paths = _adapt(view, dtype, schema)
    assert len(calls) == 1 and calls[0].dtype == torch.long
    labels = set(calls[0][:, 1].tolist())
    other = 1 if canonical == 0 else 0
    expected = {canonical, other} if context == "selected_edges" else {other} if context == "rule_absent_locally" else set()
    assert labels == expected  # exact labels, without wrapping/remapping
    assert paths.local_item_ids.tolist() == ([] if context in ("None", "zero_nodes", "zero_edges") else list(PUBLIC_IDS))
    assert paths.weights.numel() == (6 if context == "selected_edges" else 2 if context == "rule_absent_locally" else 0)
    assert paths.structurally_valid == (context in ("selected_edges", "rule_absent_locally"))


@pytest.mark.parametrize("canonical", [2**63, 2**63 + 1, 2**200])
def test_p2_canonical_range_declaration_rejects_overflow(canonical):
    with pytest.raises(ValueError, match="canonical_relation_id.*range"):
        _rule(0, canonical)


@pytest.mark.parametrize("canonical", [2**63, 2**63 + 1, 2**200])
def test_p2_canonical_range_schema_rejects_overflow(canonical):
    rules = _schema().relation_rules
    object.__setattr__(rules[0], "canonical_relation_id", canonical)
    with pytest.raises(ValueError, match="canonical_relation_id.*range"):
        _schema(relation_rules=rules)


@pytest.mark.parametrize("context", _CANONICAL_CONTEXTS)
@pytest.mark.parametrize("canonical", [2**63, 2**63 + 1, 2**200])
def test_p2_canonical_range_revalidation_precedes_graph_work(context, canonical, monkeypatch):
    view, schema = _canonical_boundary_case(context, 0)
    # Test-owned corruption exercises factory revalidation of frozen declarations.
    object.__setattr__(schema.relation_rules[0], "canonical_relation_id", canonical)
    def forbidden(*args, **kwargs):
        raise AssertionError("canonical range validation must precede graph work/conversion/factory")
    monkeypatch.setattr(GraphView, "validate", forbidden)
    for name in ("_validate_local_view", "_canonical_incidence", "prepare_path_factors"):
        monkeypatch.setattr(graph_adapter, name, forbidden)
    monkeypatch.setattr(torch, "from_numpy", forbidden)
    with pytest.raises(ValueError, match="canonical_relation_id.*range"):
        _adapt(view, schema=schema)


@pytest.mark.parametrize("canonical,error", [
    (-1, ValueError), (True, TypeError), (False, TypeError), (1.0, TypeError),
    (np.int64(1), TypeError), (np.uint64(1), TypeError), (_Integer(1), TypeError),
])
def test_p2_canonical_range_preserves_exact_integer_policy(canonical, error):
    with pytest.raises(error, match="canonical_relation_id"):
        _rule(0, canonical)
