"""G2 snapshot lifecycle, independent projection oracle and scan accounting."""

from copy import deepcopy
from dataclasses import replace
from itertools import combinations
import weakref

import numpy as np
import pytest

from src.data import (
    GraphRuntime, check_graph_runtime_binding, close_graph_runtime,
    prepare_graph_runtime, project_party_graphs, resolve_party_graph_views,
)
from src.data import graph as graph_module
from src.data import graph_runtime as runtime_module
from test_graph_resolution import (
    ARRAY_FIELDS, _access, _assert_projection, _assignment, _graph, _ids, _party,
    _reference_projection, bundle,
)


CONTEXT = dict(source_id="synthetic/source-v1", split_id="split-v1", recipe_id="recipe-v1")


def _prepare(bundle, accesses=None, **context):
    if accesses is None:
        accesses = {7: (_access(),), 41: (_access([1, 2], [1]),), 901: ()}
    partition, parties = _assignment(bundle, accesses)
    runtime = prepare_graph_runtime(bundle, partition, parties, **(CONTEXT | context))
    return runtime, partition, parties


def _project(runtime, party_id=7):
    return project_party_graphs(runtime, party_id=party_id, binding_id=runtime.binding_id)


def _check_both(runtime, bundle, partition, parties, party_id):
    actual = _project(runtime, party_id)
    strict = resolve_party_graph_views(bundle, partition, parties, party_id=party_id)
    assert list(actual) == list(strict) == sorted(actual)
    for access in partition.party_access.get(party_id, ()):
        source = bundle.graph_views[access.view_name]
        expected = _reference_projection(source, access)  # Python-only oracle.
        _assert_projection(actual[access.view_name], expected, source, access)
        _assert_projection(strict[access.view_name], expected, source, access)
    return actual


def test_exhaustive_reference_and_strict_equivalence(bundle):
    passed = rejected = 0
    source = bundle.graph_views["relations"]
    node_cases = [None] + [c for size in range(5) for c in combinations(range(4), size)]
    edge_cases = [None] + [c for size in range(4) for c in combinations(range(3), size)]
    for nodes in node_cases:
        for edges in edge_cases:
            access = _access(nodes, edges)
            partition, parties = _assignment(bundle, {7: (access,)})
            try:
                _reference_projection(source, access)
            except ValueError:
                for operation in (
                    lambda: prepare_graph_runtime(bundle, partition, parties, **CONTEXT),
                    lambda: resolve_party_graph_views(bundle, partition, parties, party_id=7),
                ):
                    with pytest.raises(ValueError, match="endpoints outside"):
                        operation()
                rejected += 1
            else:
                runtime = prepare_graph_runtime(bundle, partition, parties, **CONTEXT)
                _check_both(runtime, bundle, partition, parties, 7)
                close_graph_runtime(runtime)
                passed += 1
    assert passed + rejected == 153
    assert passed > 34 and rejected > 94


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("nodes", [None, ()])
@pytest.mark.parametrize("edges", [None, ()])
def test_empty_and_edgeless_graphs(bundle, empty, nodes, edges):
    source = _graph(edge_index=np.empty((2, 0), dtype=np.int64), relation_ids=_ids())
    if empty:
        source = replace(source, node_raw_ids=(), node_type_ids=_ids(),
                         user_to_node=_ids([-1, -1]), item_to_node=_ids([-1, -1, -1]))
    bundle.graph_views = {source.name: source}
    runtime, partition, parties = _prepare(bundle, {7: (_access(nodes, edges),), 41: ()})
    _check_both(runtime, bundle, partition, parties, 7)
    assert _check_both(runtime, bundle, partition, parties, 41) == {}
    assert _project(runtime, 7)[source.name].edge_index.shape == (2, 0)


@pytest.mark.parametrize("users", [None, (-1, -1), (2, 2), (0, 3)])
@pytest.mark.parametrize("items", [None, (-1, -1, -1), (2, 2, 2), (3, 0, -1)])
@pytest.mark.parametrize("nodes,edges", [(None, None), ((), ()), ((1, 2), (1,))])
def test_mapping_forms(bundle, users, items, nodes, edges):
    view = bundle.graph_views["relations"]
    view.user_to_node = None if users is None else _ids(users)
    view.item_to_node = None if items is None else _ids(items)
    runtime, partition, parties = _prepare(bundle, {7: (_access(nodes, edges),)})
    _check_both(runtime, bundle, partition, parties, 7)


def test_multiple_views_parties_and_roster_order(bundle):
    bundle.graph_views["aaa"] = _graph(name="aaa")
    accesses = {901: (), 41: (_access(name="relations"), _access(name="aaa")),
                7: (_access([1, 2], [1], name="aaa"),)}
    runtime, partition, parties = _prepare(bundle, accesses)
    for party_id in accesses:
        _check_both(runtime, bundle, partition, parties, party_id)
    reordered = prepare_graph_runtime(bundle, partition, tuple(reversed(parties)), **CONTEXT)
    assert runtime.binding_id == reordered.binding_id
    assert list(_project(runtime, 41)) == ["aaa", "relations"]


@pytest.mark.parametrize("party_id", [-1, True, np.int64(7), "7", None, 8])
def test_invalid_or_unknown_party_requests(bundle, party_id):
    runtime, _, _ = _prepare(bundle)
    with pytest.raises(ValueError):
        _project(runtime, party_id)
    assert _project(runtime, 901) == {}


def test_snapshot_survives_original_mutation_and_does_not_retain_inputs(bundle):
    original = deepcopy(bundle.graph_views["relations"])
    runtime, partition, parties = _prepare(bundle)
    identity = runtime.binding_id
    refs = [weakref.ref(getattr(bundle.graph_views["relations"], name)) for name in ARRAY_FIELDS]
    # Corrupt everything the original strict validator depended on, including
    # domain counts, graph keys, public strings and roster/access declarations.
    for name in ARRAY_FIELDS:
        getattr(bundle.graph_views["relations"], name).fill(-10)
    bundle.graph_views["relations"].name = "changed"
    bundle.graph_views["relations"].metadata["private"]["source"].append("changed")
    bundle.graph_views.clear()
    bundle.user_id_map.clear()
    bundle.item_id_map["new"] = 100
    parties[0].party_id = 555
    parties[1].graph_view_names = ()
    partition.metadata["changed"] = ["not retained"]
    # Ordinary replacement of caller permission variables does not revoke a
    # prepared handle. Descriptors/mappings themselves are immutable G1a data.
    partition = replace(partition, party_access={7: (_access([], []),)}, metadata={})
    parties = ()
    assert all(ref() is None for ref in refs)
    assert runtime.binding_id == identity
    _assert_projection(_project(runtime, 7)["relations"],
                       _reference_projection(original, _access()), original, _access())
    partial = _access([1, 2], [1])
    _assert_projection(_project(runtime, 41)["relations"],
                       _reference_projection(original, partial), original, partial)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("field", ARRAY_FIELDS)
@pytest.mark.parametrize("mutation", ["contents", "shape", "dtype", "flags"])
def test_returned_array_mutation_isolation(bundle, empty, field, mutation):
    access = _access([], []) if empty else _access()
    runtime, partition, parties = _prepare(bundle, {7: (access,), 41: (access,)})
    first = _project(runtime)["relations"]
    second = _project(runtime, 41)["relations"]
    values = getattr(first, field)
    assert not np.shares_memory(values, getattr(second, field))
    if mutation == "contents":
        values.fill(-123)
    elif mutation == "shape":
        values.shape = (values.size, 1)
    elif mutation == "dtype":
        values.dtype = np.uint64
    else:
        values.setflags(write=False)
    first.metadata["private"] = {"mutable": [1]}
    first.name = "changed"
    expected = _reference_projection(bundle.graph_views["relations"], access)
    _assert_projection(second, expected, bundle.graph_views["relations"], access)
    for party_id in (7, 41):
        _check_both(runtime, bundle, partition, parties, party_id)


def test_binding_and_close_lifecycle(bundle):
    runtime, partition, parties = _prepare(bundle)
    binding_id = runtime.binding_id
    output = _project(runtime)
    check_graph_runtime_binding(runtime, binding_id=binding_id)
    with pytest.raises(AttributeError):
        runtime.binding_id = "changed"
    for context_name in CONTEXT:
        changed = prepare_graph_runtime(bundle, partition, parties,
                                        **(CONTEXT | {context_name: "changed"}))
        assert changed.binding_id != binding_id
        with pytest.raises(ValueError, match="binding mismatch"):
            project_party_graphs(changed, party_id=7, binding_id=binding_id)
        close_graph_runtime(changed)
    for operation in (
        lambda: check_graph_runtime_binding(runtime, binding_id="wrong"),
        lambda: project_party_graphs(runtime, party_id=7, binding_id="wrong"),
    ):
        with pytest.raises(ValueError, match="binding mismatch"):
            operation()
    owned_ref = weakref.ref(runtime._views["relations"].edge_index)
    close_graph_runtime(runtime)
    close_graph_runtime(runtime)
    assert owned_ref() is None
    assert runtime._views is runtime._indexes is runtime._access_by_party is None
    for operation in (
        lambda: runtime.binding_id,
        lambda: check_graph_runtime_binding(runtime, binding_id=binding_id),
        lambda: project_party_graphs(runtime, party_id=7, binding_id=binding_id),
    ):
        with pytest.raises(ValueError, match="closed"):
            operation()
    output["relations"].validate()
    _assert_projection(output["relations"], _reference_projection(
        bundle.graph_views["relations"], _access()), bundle.graph_views["relations"], _access())


@pytest.mark.parametrize("field", ["source_id", "split_id", "recipe_id"])
@pytest.mark.parametrize("invalid", [None, 7, "", "  "])
def test_context_is_explicit_and_nonblank(bundle, field, invalid):
    partition, parties = _assignment(bundle, {7: (_access(),)})
    with pytest.raises((TypeError, ValueError), match=field):
        prepare_graph_runtime(bundle, partition, parties, **(CONTEXT | {field: invalid}))


def test_type_errors_and_empty_partition(bundle):
    with pytest.raises(TypeError, match="prepare_graph_runtime"):
        GraphRuntime()
    for operation in (close_graph_runtime, lambda r: check_graph_runtime_binding(r, binding_id=""),
                      lambda r: project_party_graphs(r, party_id=7, binding_id="")):
        with pytest.raises(TypeError, match="GraphRuntime"):
            operation(None)
    runtime, _, _ = _prepare(bundle, {7: ()})
    with pytest.raises(TypeError, match="binding_id"):
        check_graph_runtime_binding(runtime, binding_id=None)
    assert _project(runtime) == {}
    partition, parties = _assignment(bundle, {})
    empty = prepare_graph_runtime(bundle, partition, parties, **CONTEXT)
    with pytest.raises(ValueError, match="unknown"):
        _project(empty)


class _StringSubclass(str):
    def __lt__(self, other):
        raise AssertionError("must reject before comparing")


class _ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize("field", ARRAY_FIELDS)
@pytest.mark.parametrize("restriction", ["subclass", "metadata", "empty_metadata"])
def test_array_delivery_restrictions_before_copy(bundle, field, restriction):
    source = bundle.graph_views["relations"]
    values = getattr(source, field)
    if restriction == "subclass":
        values = values.view(_ArraySubclass)
        # Bind plain content first; the subclass must not be sanitized to match.
        partition, parties = _assignment(bundle, {7: (_access(),)})
        setattr(source, field, values)
    else:
        values = values.view(np.dtype(np.int64, metadata={} if restriction == "empty_metadata"
                                    else {"private": [1]}))
        setattr(source, field, values)
        partition, parties = _assignment(bundle, {7: (_access(),)})
    with pytest.raises(TypeError, match="subclass|dtype metadata"):
        prepare_graph_runtime(bundle, partition, parties, **CONTEXT)
    assert getattr(source, field) is values


@pytest.mark.parametrize("field", ["name", "node_type_names", "relation_names", "access", "binding"])
def test_string_delivery_restrictions_before_copy(bundle, field):
    source = bundle.graph_views["relations"]
    access = _access()
    if field in ("name", "node_type_names", "relation_names"):
        value = getattr(source, field)
        setattr(source, field, _StringSubclass(value) if field == "name"
                else tuple(_StringSubclass(name) for name in value))
    elif field == "access":
        access = _access(name=_StringSubclass("relations"))
    partition, parties = _assignment(bundle, {7: (access,)})
    if field == "binding":
        partition = replace(partition, source_graph_fingerprints={
            _StringSubclass(name): digest
            for name, digest in partition.source_graph_fingerprints.items()
        })
    with pytest.raises(TypeError, match="plain built-in strings"):
        prepare_graph_runtime(bundle, partition, parties, **CONTEXT)


def test_input_nonmutation_and_failed_snapshot_validation(bundle, monkeypatch):
    runtime, partition, parties = _prepare(bundle)
    source = bundle.graph_views["relations"]
    before = deepcopy(source)
    digest = graph_module.graph_view_fingerprint(source)
    checksum = graph_module.graph_partition_checksum(partition)
    # Detect an invalid copied snapshot independently of original validation.
    original_copy = runtime_module._copy_graph

    def bad_copy(view):
        copied = original_copy(view)
        copied.item_to_node = _ids([0])
        return copied

    monkeypatch.setattr(runtime_module, "_copy_graph", bad_copy)
    with pytest.raises(ValueError, match="complete processed namespace"):
        prepare_graph_runtime(bundle, partition, parties, **CONTEXT)
    with pytest.raises(ValueError, match="unknown"):
        _project(runtime, 9000)
    assert graph_module.graph_view_fingerprint(source) == digest
    assert graph_module.graph_partition_checksum(partition) == checksum
    assert source.metadata == before.metadata
    assert [party.party_id for party in parties] == [7, 41, 901]
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(source, name), getattr(before, name))
        assert getattr(source, name).flags.writeable


def test_successful_preparation_preserves_inputs_and_rejects_copied_content_change(bundle, monkeypatch):
    partition, parties = _assignment(bundle, {7: (_access([1, 2], [1]),), 901: ()})
    source = bundle.graph_views["relations"]
    original_arrays = {name: getattr(source, name) for name in ARRAY_FIELDS}
    before_hash = graph_module.graph_view_fingerprint(source)
    before_partition = graph_module.graph_partition_checksum(partition)
    before_metadata = deepcopy(source.metadata)
    original_domain = (dict(bundle.user_id_map), dict(bundle.item_id_map))
    runtime = prepare_graph_runtime(bundle, partition, parties, **CONTEXT)
    _check_both(runtime, bundle, partition, parties, 7)
    assert graph_module.graph_view_fingerprint(source) == before_hash
    assert graph_module.graph_partition_checksum(partition) == before_partition
    assert source.metadata == before_metadata
    assert (bundle.user_id_map, bundle.item_id_map) == original_domain
    for name, array in original_arrays.items():
        assert getattr(source, name) is array and array.flags.writeable

    original_copy = runtime_module._copy_graph

    def changed_content(view):
        copied = original_copy(view)
        copied.relation_ids = _ids([1, 0, 0])  # Valid structure, wrong bound content.
        return copied

    monkeypatch.setattr(runtime_module, "_copy_graph", changed_content)
    with pytest.raises(ValueError, match="stale source graph fingerprint"):
        prepare_graph_runtime(bundle, partition, parties, **CONTEXT)
    assert graph_module.graph_view_fingerprint(source) == before_hash


def test_binding_tracks_content_domain_permissions_roster_not_metadata(bundle):
    runtime, partition, parties = _prepare(bundle)
    old_id = runtime.binding_id
    source = bundle.graph_views["relations"]
    source.metadata["ignored"] = [1]
    partition.metadata["ignored"] = [2]
    assert prepare_graph_runtime(bundle, partition, parties, **CONTEXT).binding_id == old_id
    with_extra_party = prepare_graph_runtime(bundle, partition, (*parties, _party(1234)), **CONTEXT)
    assert with_extra_party.binding_id != old_id
    changed_permissions, _, _ = _prepare(bundle, {7: (_access([], []),), 41: (), 901: ()})
    assert changed_permissions.binding_id != old_id
    bundle.item_id_map = {"different": 0, "item-b": 1, "item-c": 2}
    with pytest.raises(ValueError, match="stale source_domain"):
        prepare_graph_runtime(bundle, partition, parties, **CONTEXT)
    changed_domain, _, _ = _prepare(bundle)
    assert changed_domain.binding_id != old_id
    source.relation_ids[0] = 1
    changed_graph, _, _ = _prepare(bundle)
    assert changed_graph.binding_id != changed_domain.binding_id
    assert runtime.binding_id == old_id


def test_projection_does_not_repeat_global_work_or_touch_unrelated_views(bundle, monkeypatch):
    bundle.graph_views["aaa"] = _graph(name="aaa")
    partition, parties = _assignment(bundle, {7: (_access(),), 41: (_access(name="aaa"),), 901: ()})
    counters = {}

    def instrument(module, name):
        original = getattr(module, name)
        key = (module.__name__, name)
        counters[key] = 0

        def wrapper(*args, **kwargs):
            counters[key] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    for module, names in (
        (runtime_module, ("validate_graph_partition", "graph_domain_fingerprint",
                          "graph_partition_checksum", "_validate_graph_sources",
                          "_build_graph_projection_index", "_copy_graph")),
        (graph_module, ("graph_view_fingerprint", "graph_domain_fingerprint",
                        "_validate_graph_sources", "_build_graph_projection_index",
                        "_build_graph_mapping_index")),
    ):
        for name in names:
            instrument(module, name)
    runtime = prepare_graph_runtime(bundle, partition, parties, **CONTEXT)
    before = dict(counters)
    assert before[(runtime_module.__name__, "_build_graph_projection_index")] == 2
    assert before[(graph_module.__name__, "graph_view_fingerprint")] == 4
    visited = []
    original_projection = runtime_module._project_graph_view

    def selected_only(view, *args):
        visited.append(view.name)
        return original_projection(view, *args)

    monkeypatch.setattr(runtime_module, "_project_graph_view", selected_only)
    for _ in range(25):
        _project(runtime)
        _project(runtime, 901)
    assert counters == before
    assert visited == ["relations"] * 25
    # Strict G1b still performs global validation and constructs indexes afresh.
    resolve_party_graph_views(bundle, partition, parties, party_id=7)
    assert counters[(graph_module.__name__, "graph_view_fingerprint")] == 6
    assert counters[(graph_module.__name__, "_build_graph_projection_index")] == 1


def test_only_referenced_payload_is_copied_and_all_selectors_stay_none(bundle):
    bundle.graph_views["unrelated"] = object()  # Not a bound source; must not be inspected.
    runtime, _, _ = _prepare(bundle)
    assert tuple(runtime._views) == ("relations",)
    access = runtime._access_by_party[7][0]
    assert access.node_ids is None and access.edge_ids is None
    for field in ARRAY_FIELDS:
        values = getattr(runtime._views["relations"], field)
        with pytest.raises(ValueError):
            values.setflags(write=True)
    assert _project(runtime, 901) == {}
