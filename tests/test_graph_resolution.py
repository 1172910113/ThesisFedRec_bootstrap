"""Strict G1b delivery: independent Python projection and isolation checks."""

from copy import deepcopy
from dataclasses import asdict, fields, replace
from itertools import combinations

import numpy as np
import pytest

from src.data import (
    DatasetBundle,
    GraphAccess,
    GraphPartition,
    GraphView,
    InteractionSplits,
    InteractionTable,
    PartyData,
    graph_domain_fingerprint,
    graph_partition_checksum,
    graph_view_fingerprint,
    resolve_party_graph_views,
    validate_graph_partition,
)
from src.data import graph as graph_module


ARRAY_FIELDS = (
    "node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node",
)


def _ids(values=()):
    return np.array(values, dtype=np.int64)


def _graph(**overrides):
    arguments = dict(
        name="relations", node_raw_ids=("private-z", 17, "private-a", -9),
        node_type_ids=_ids([1, 0, 1, 0]),
        node_type_names=("entity", "aux", "unused_type"),
        edge_index=_ids([[0, 2, 2], [1, 1, 3]]), relation_ids=_ids([0, 0, 0]),
        relation_names=("link", "other", "unused_relation"),
        user_to_node=_ids([3, 2]), item_to_node=_ids([0, 2, -1]),
        metadata={"private": {"source": ["never deliver"]}},
    )
    arguments.update(overrides)
    return GraphView(**arguments)


@pytest.fixture
def bundle():
    view = _graph()
    return DatasetBundle(
        name="synthetic", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=_ids(range(6)), user_ids=_ids([0, 0, 0, 1, 1, 1]),
            item_ids=_ids([0, 1, 2, 0, 1, 2]), timestamps=_ids(range(6)),
            values=np.ones(6, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=_ids([0, 3]), valid_interaction_ids=_ids([1, 4]),
            test_interaction_ids=_ids([2, 5]),
        ),
        user_id_map={"user-a": 0, "user-b": 1},
        item_id_map={"item-a": 0, "item-b": 1, "item-c": 2},
        graph_views={view.name: view}, metadata={"private": ["base"]},
    )


def _party(party_id, names=()):
    return PartyData(
        party_id=party_id, train_interaction_ids=_ids(), valid_interaction_ids=_ids(),
        test_interaction_ids=_ids(), user_ids=_ids(), item_ids=_ids(),
        feature_view_names=(), graph_view_names=names, owns_labels=False,
        metadata={"private": [party_id]},
    )


def _assignment(bundle, accesses):
    names = {entry.view_name for entries in accesses.values() for entry in entries}
    partition = GraphPartition(
        name="explicit", party_access=accesses,
        source_graph_fingerprints={
            name: graph_view_fingerprint(bundle.graph_views[name]) for name in names
        },
        source_domain_fingerprint=graph_domain_fingerprint(bundle), metadata={},
    )
    parties = tuple(
        _party(party_id, tuple(access.view_name for access in entries))
        for party_id, entries in accesses.items()
    )
    return partition, parties


def _access(nodes=None, edges=None, name="relations"):
    return GraphAccess(
        view_name=name, node_ids=None if nodes is None else _ids(nodes),
        edge_ids=None if edges is None else _ids(edges),
    )


def _reference_projection(view, access):
    """Python-only selection/remapping; no production projection helpers."""
    nodes = (
        list(range(len(view.node_raw_ids))) if access.node_ids is None
        else access.node_ids.tolist()
    )
    edges = (
        list(range(len(view.relation_ids))) if access.edge_ids is None
        else access.edge_ids.tolist()
    )
    positions = {canonical: local for local, canonical in enumerate(nodes)}
    endpoints = [[], []]
    for edge in edges:
        for side in (0, 1):
            canonical = int(view.edge_index[side, edge])
            if canonical not in positions:
                raise ValueError("unauthorized endpoint")
            endpoints[side].append(positions[canonical])

    def mapping(values):
        return None if values is None else [positions.get(int(value), -1) for value in values]

    return dict(
        name=view.name, node_raw_ids=tuple(f"n{i}" for i in range(len(nodes))),
        node_type_ids=[int(view.node_type_ids[node]) for node in nodes],
        node_type_names=tuple(view.node_type_names), edge_index=endpoints,
        relation_ids=[int(view.relation_ids[edge]) for edge in edges],
        relation_names=tuple(view.relation_names),
        user_to_node=mapping(view.user_to_node), item_to_node=mapping(view.item_to_node),
        metadata={},
    )


def _assert_projection(actual, expected, source, access):
    assert type(actual) is GraphView and actual is not source
    assert {field.name for field in fields(actual)} == set(expected)
    for name, value in expected.items():
        received = getattr(actual, name)
        if name in ARRAY_FIELDS and value is not None:
            np.testing.assert_array_equal(received, _ids(value))
            assert type(received) is np.ndarray
            assert received.dtype == np.dtype(np.int64)
            assert received.dtype.metadata is None
            assert received.flags.owndata and received.flags.writeable
            assert received.base is None
            for source_name in ARRAY_FIELDS:
                original = getattr(source, source_name)
                if original is not None:
                    assert not np.shares_memory(received, original)
            for selector in (access.node_ids, access.edge_ids):
                if selector is not None:
                    assert not np.shares_memory(received, selector)
        else:
            assert received == value
    assert type(actual.metadata) is dict and actual.metadata == {}
    assert actual.metadata is not source.metadata
    assert type(actual.name) is str
    for name in ("node_raw_ids", "node_type_names", "relation_names"):
        assert type(getattr(actual, name)) is tuple
        assert all(type(value) is str for value in getattr(actual, name))
    actual.validate()


def _resolve(bundle, access):
    partition, parties = _assignment(bundle, {7: (access,)})
    return resolve_party_graph_views(bundle, partition, parties, party_id=7)[access.view_name]


def test_hand_computable_projection_and_no_induced_edges(bundle):
    access = _access([1, 2], [1])
    result = _resolve(bundle, access)
    assert result.node_raw_ids == ("n0", "n1")
    np.testing.assert_array_equal(result.edge_index, [[1], [0]])
    np.testing.assert_array_equal(result.item_to_node, [-1, 1, -1])
    _assert_projection(result, _reference_projection(bundle.graph_views["relations"], access),
                       bundle.graph_views["relations"], access)
    with pytest.raises(ValueError, match="endpoints outside"):
        _resolve(bundle, _access([1, 2], [0, 1]))
    # Even authorizing every endpoint must not infer either withheld edge.
    one_edge = _resolve(bundle, _access([0, 1, 2, 3], [1]))
    np.testing.assert_array_equal(one_edge.edge_index, [[2], [1]])
    assert one_edge.node_raw_ids == ("n0", "n1", "n2", "n3")


@pytest.mark.parametrize("nodes", [None, (), (1, 2), (0, 1, 2, 3)])
@pytest.mark.parametrize("edges", [None, (), (1,), (0, 1)])
def test_none_empty_partial_and_explicit_full_selections(bundle, nodes, edges):
    source, access = bundle.graph_views["relations"], _access(nodes, edges)
    try:
        expected = _reference_projection(source, access)
    except ValueError:
        with pytest.raises(ValueError, match="endpoints outside"):
            _resolve(bundle, access)
    else:
        _assert_projection(_resolve(bundle, access), expected, source, access)


def test_exhaustive_small_reference_comparison(bundle):
    source = bundle.graph_views["relations"]
    passed = rejected = 0
    for node_count in range(5):
        for nodes in combinations(range(4), node_count):
            for edge_count in range(4):
                for edges in combinations(range(3), edge_count):
                    access = _access(nodes, edges)
                    try:
                        expected = _reference_projection(source, access)
                    except ValueError:
                        with pytest.raises(ValueError, match="endpoints outside"):
                            _resolve(bundle, access)
                        rejected += 1
                    else:
                        _assert_projection(_resolve(bundle, access), expected, source, access)
                        passed += 1
    assert (passed, rejected) == (34, 94)  # 128 independent paired comparisons.


@pytest.mark.parametrize("empty_nodes", [False, True])
@pytest.mark.parametrize("nodes", [None, ()])
@pytest.mark.parametrize("edges", [None, ()])
def test_edgeless_and_empty_graphs_remain_visible(bundle, empty_nodes, nodes, edges):
    source = _graph(edge_index=np.empty((2, 0), dtype=np.int64), relation_ids=_ids())
    if empty_nodes:
        source = replace(source, node_raw_ids=(), node_type_ids=_ids(),
                         user_to_node=_ids([-1, -1]), item_to_node=_ids([-1, -1, -1]))
    bundle.graph_views = {source.name: source}
    access = _access(nodes, edges)
    _assert_projection(_resolve(bundle, access), _reference_projection(source, access), source, access)


def test_isolated_nodes_and_explicit_edge_forms_preserve_order(bundle):
    source = _graph(
        node_raw_ids=("z", "b", 7, "a", -10), node_type_ids=_ids([1, 0, 1, 0, 2]),
        edge_index=_ids([[2, 0, 1, 0, 0], [0, 1, 0, 0, 1]]),
        relation_ids=_ids([1, 0, 0, 1, 1]),
    )
    bundle.graph_views = {source.name: source}
    access = _access([0, 1, 2, 4], None)
    result = _resolve(bundle, access)
    _assert_projection(result, _reference_projection(source, access), source, access)
    assert result.node_raw_ids == ("n0", "n1", "n2", "n3")
    assert result.node_type_names == source.node_type_names
    assert result.relation_names == source.relation_names
    assert result.edge_index.shape == (2, 5)  # No generated loops or reverse arcs.


@pytest.mark.parametrize("user_mode", ["none", "unmapped", "many_to_one", "boundaries"])
@pytest.mark.parametrize("item_mode", ["none", "unmapped", "many_to_one", "boundaries"])
def test_independent_optional_and_many_to_one_mappings(bundle, user_mode, item_mode):
    source = bundle.graph_views["relations"]
    for field, mode, boundary in (
        ("user_to_node", user_mode, [0, 3]), ("item_to_node", item_mode, [0, 3, -1]),
    ):
        values = {"none": None, "unmapped": [-1] * len(boundary),
                  "many_to_one": [2] * len(boundary), "boundaries": boundary}[mode]
        setattr(source, field, None if values is None else _ids(values))
    access = _access([1, 2], [1])
    _assert_projection(_resolve(bundle, access), _reference_projection(source, access), source, access)


def test_empty_processed_domains_keep_owned_empty_mappings(bundle):
    source = _graph(user_to_node=_ids(), item_to_node=_ids())
    empty = DatasetBundle(
        name="empty", schema_version="1.0", user_id_map={}, item_id_map={}, metadata={},
        interactions=InteractionTable(
            interaction_ids=_ids(), user_ids=_ids(), item_ids=_ids(), timestamps=_ids(),
            values=np.empty(0, dtype=np.float32),
        ),
        splits=InteractionSplits(train_interaction_ids=_ids(), valid_interaction_ids=_ids(),
                                 test_interaction_ids=_ids()),
        graph_views={source.name: source},
    )
    access = _access([], [])
    _assert_projection(_resolve(empty, access), _reference_projection(source, access), source, access)


@pytest.mark.parametrize("explicit_entry", [False, True])
def test_known_absent_access_is_fresh_empty_result(bundle, explicit_entry):
    partition, _ = _assignment(bundle, {41: ()} if explicit_entry else {})
    parties = (_party(41),)
    first = resolve_party_graph_views(bundle, partition, parties, party_id=41)
    second = resolve_party_graph_views(bundle, partition, parties, party_id=41)
    assert first == second == {} and first is not second
    with pytest.raises(ValueError, match="unknown requested party_id"):
        resolve_party_graph_views(bundle, partition, parties, party_id=7)


def test_multiple_views_parties_and_lexical_order(bundle):
    bundle.graph_views["alpha"] = _graph(name="alpha")
    entries = (_access([1, 2], [1]), _access([], [], name="alpha"))
    partition, parties = _assignment(bundle, {41: entries, 7: entries})
    first = resolve_party_graph_views(bundle, partition, parties, party_id=41)
    reordered = replace(partition, party_access={7: tuple(reversed(entries)), 41: tuple(reversed(entries))})
    second = resolve_party_graph_views(bundle, reordered, tuple(reversed(parties)), party_id=7)
    assert type(first) is dict and list(first) == list(second) == ["alpha", "relations"]
    for access in entries:
        source = bundle.graph_views[access.view_name]
        expected = _reference_projection(source, access)
        for result in (first, second):
            _assert_projection(result[access.view_name], expected, source, access)
        assert first[access.view_name] is not second[access.view_name]
        for field in ARRAY_FIELDS:
            left, right = getattr(first[access.view_name], field), getattr(second[access.view_name], field)
            assert left is not right and not np.shares_memory(left, right)
    assert bundle.parties == ()  # Explicit graph access is not sample ownership.


class _IntSubclass(int):
    pass


class _StringSubclass(str):
    pass


class _ArraySubclass(np.ndarray):
    pass


class _TupleSubclass(tuple):
    pass


@pytest.mark.parametrize("party_id", [True, False, -1, np.int64(7), _IntSubclass(7), 7.0, "7", None])
def test_invalid_requested_party_ids(bundle, party_id):
    partition, parties = _assignment(bundle, {7: (_access(),)})
    with pytest.raises(ValueError, match="exact built-in nonnegative"):
        resolve_party_graph_views(bundle, partition, parties, party_id=party_id)


@pytest.mark.parametrize("invalid", ["bundle", "partition", "roster", "party", "duplicate", "unknown"])
def test_public_input_and_roster_validation(bundle, invalid):
    partition, parties = _assignment(bundle, {7: (_access(),)})
    if invalid == "bundle":
        bundle = object()
    elif invalid == "partition":
        partition = object()
    elif invalid == "roster":
        parties = "parties"
    elif invalid == "party":
        parties = (object(),)
    elif invalid == "duplicate":
        parties = parties + parties
    else:
        parties = ()
    with pytest.raises((TypeError, ValueError)):
        resolve_party_graph_views(bundle, partition, parties, party_id=7)


@pytest.mark.parametrize("empty_result", [False, True])
@pytest.mark.parametrize("invalid", ["endpoint", "stale_graph", "stale_domain", "mapping", "summary", "removed"])
def test_complete_validation_before_any_projection(bundle, monkeypatch, empty_result, invalid):
    bundle.graph_views["other"] = _graph(name="other")
    other_access = _access([1, 2], [0], name="other") if invalid == "endpoint" else _access(name="other")
    partition, parties = _assignment(bundle, {7: () if empty_result else (_access(),), 41: (other_access,)})
    source = bundle.graph_views["other"]
    if invalid == "stale_graph":
        source.edge_index[0, 0] = 3
    elif invalid == "stale_domain":
        bundle.item_id_map["item-a"], bundle.item_id_map["item-b"] = 1, 0
    elif invalid == "mapping":
        source.item_to_node[0] = 99
    elif invalid == "summary":
        parties[1].graph_view_names = ()
    elif invalid == "removed":
        del bundle.graph_views["other"]
    before = deepcopy(asdict(bundle))

    def forbid_projection(*args, **kwargs):
        pytest.fail("no projection may precede complete source/access validation")

    monkeypatch.setattr(graph_module, "_project_graph_view", forbid_projection)
    with pytest.raises(ValueError):
        resolve_party_graph_views(bundle, partition, parties, party_id=7)
    np.testing.assert_equal(asdict(bundle), before)


@pytest.mark.parametrize("field", ARRAY_FIELDS)
@pytest.mark.parametrize("metadata", [{}, {"private": ["hidden"]}])
@pytest.mark.parametrize("mode", ["full", "empty", "other_party"])
def test_delivery_dtype_metadata_rejected_on_all_references(bundle, field, metadata, mode):
    source = bundle.graph_views["relations"]
    original = getattr(source, field)
    # np.array(existing_array, dtype=...) can discard equivalent dtype metadata
    # in NumPy 1.26. A dtype view explicitly retains the unsupported metadata.
    supplied = original.view(np.dtype(np.int64, metadata=metadata))
    assert supplied.dtype.metadata == metadata
    setattr(source, field, supplied)
    access = _access([], []) if mode == "empty" else _access()
    partition, parties = _assignment(bundle, {7: (), 41: (access,)} if mode == "other_party" else {7: (access,)})
    validate_graph_partition(partition, bundle, parties)  # Accepted G1a remains unchanged.
    with pytest.raises(TypeError, match="dtype metadata"):
        resolve_party_graph_views(bundle, partition, parties, party_id=7)
    assert getattr(source, field) is supplied and supplied.dtype.metadata == metadata
    np.testing.assert_array_equal(supplied, original)


@pytest.mark.parametrize("field", ["name", "node_type_names", "relation_names"])
@pytest.mark.parametrize("string_type", [_StringSubclass, np.str_])
@pytest.mark.parametrize("mode", ["full", "empty", "other_party"])
def test_public_names_require_builtin_strings_everywhere(bundle, field, string_type, mode):
    source = bundle.graph_views["relations"]
    if field == "name":
        source.name = string_type(source.name)
    else:
        names = getattr(source, field)
        setattr(source, field, (*names[:-1], string_type(names[-1])))  # Unused entry too.
    access = _access([], []) if mode == "empty" else _access()
    partition, parties = _assignment(bundle, {7: (), 41: (access,)} if mode == "other_party" else {7: (access,)})
    validate_graph_partition(partition, bundle, parties)
    before = getattr(source, field)
    with pytest.raises(TypeError, match="plain built-in strings"):
        resolve_party_graph_views(bundle, partition, parties, party_id=7)
    assert getattr(source, field) is before


@pytest.mark.parametrize("field", ARRAY_FIELDS)
@pytest.mark.parametrize("representation", ["masked", "unmasked", "subclass"])
def test_g1a_subclass_rejection_is_not_weakened(bundle, field, representation):
    partition, parties = _assignment(bundle, {7: (_access([], []),)})
    source = bundle.graph_views["relations"]
    original = getattr(source, field)
    supplied = (
        original.view(_ArraySubclass) if representation == "subclass"
        else np.ma.array(original, mask=representation == "masked")
    )
    setattr(source, field, supplied)
    with pytest.raises(TypeError, match="plain NumPy ndarray"):
        resolve_party_graph_views(bundle, partition, parties, party_id=7)
    assert getattr(source, field) is supplied
    np.testing.assert_array_equal(np.asarray(supplied), original)


@pytest.mark.parametrize("layout", ["strided", "negative_stride", "fortran", "read_only"])
def test_plain_layouts_and_public_container_subclasses_are_detached(bundle, layout):
    source = bundle.graph_views["relations"]
    for field in ARRAY_FIELDS:
        original = getattr(source, field)
        if layout == "strided":
            storage = np.empty((*original.shape[:-1], original.shape[-1] * 2), dtype=np.int64)
            array = storage[..., ::2]
            array[...] = original
        elif layout == "negative_stride":
            array = original[..., ::-1].copy()[..., ::-1]
        elif layout == "fortran":
            array = np.asfortranarray(original)
        else:
            array = original.copy()
            array.setflags(write=False)
        setattr(source, field, array)
    for field in ("node_type_names", "relation_names"):
        names = _TupleSubclass(getattr(source, field))
        names.private = source
        setattr(source, field, names)
    access = _access()
    _assert_projection(_resolve(bundle, access), _reference_projection(source, access), source, access)


@pytest.mark.parametrize("mode", ["full", "partial", "empty"])
def test_every_output_array_header_and_metadata_are_isolated(bundle, mode):
    source = bundle.graph_views["relations"]
    access = {"full": _access(), "partial": _access([1, 2], [1]), "empty": _access([], [])}[mode]
    partition, parties = _assignment(bundle, {7: (access,), 41: (access,)})
    expected = _reference_projection(source, access)
    before = deepcopy(asdict(bundle))
    checksum = graph_partition_checksum(partition)
    for field in ARRAY_FIELDS:
        for change in ("elements", "shape", "dtype", "write_flag"):
            first = resolve_party_graph_views(bundle, partition, parties, party_id=7)[source.name]
            other = resolve_party_graph_views(bundle, partition, parties, party_id=41)[source.name]
            for left_field in ARRAY_FIELDS:
                left = getattr(first, left_field)
                for right_field in ARRAY_FIELDS:
                    right = getattr(other, right_field)
                    assert left is not right and not np.shares_memory(left, right)
            array = getattr(first, field)
            if change == "elements":
                array[...] = 99
            elif change == "shape":
                array.shape = (1, *array.shape)
            elif change == "dtype":
                array.dtype = np.int32
            else:
                array.setflags(write=False)
            first.metadata["new"] = ["local only"]
            first.node_raw_ids = ("changed",)
            _assert_projection(other, expected, source, access)
            later = resolve_party_graph_views(bundle, partition, parties, party_id=7)[source.name]
            _assert_projection(later, expected, source, access)
            assert first.metadata is not other.metadata and first.metadata is not later.metadata
            np.testing.assert_equal(asdict(bundle), before)
    assert graph_partition_checksum(partition) == checksum


@pytest.mark.parametrize("field", [*ARRAY_FIELDS, "node_raw_ids"])
def test_source_changes_cannot_change_existing_output_and_require_rebinding(bundle, field):
    source, access = bundle.graph_views["relations"], _access()
    partition, parties = _assignment(bundle, {7: (access,)})
    original_expected = _reference_projection(source, access)
    result = resolve_party_graph_views(bundle, partition, parties, party_id=7)[source.name]
    if field == "node_raw_ids":
        source.node_raw_ids = ("replacement-secret", *source.node_raw_ids[1:])
    elif field == "node_type_ids":
        source.node_type_ids[0] = 2
    elif field == "edge_index":
        source.edge_index[0, 0] = 3
    elif field == "relation_ids":
        source.relation_ids[0] = 1
    else:
        getattr(source, field)[0] = 1
    source.validate()
    _assert_projection(result, original_expected, source, access)
    with pytest.raises(ValueError, match="stale source graph"):
        resolve_party_graph_views(bundle, partition, parties, party_id=7)
    _assert_projection(_resolve(bundle, access), _reference_projection(source, access), source, access)


def test_success_preserves_inputs_and_ignores_unreferenced_payloads(bundle, monkeypatch):
    source, access = bundle.graph_views["relations"], _access([1, 2], [1])
    partition, parties = _assignment(bundle, {7: (access,)})
    source.metadata["hidden_edges"] = source.edge_index
    source.metadata["source_fingerprint"] = graph_view_fingerprint(source)
    before = deepcopy(asdict(bundle))
    before_parties = deepcopy([asdict(party) for party in parties])
    references = {field.name: getattr(source, field.name) for field in fields(source)}
    original_registry = bundle.graph_views
    checksum = graph_partition_checksum(partition)

    def forbid_bundle_reconstruction(*args, **kwargs):
        pytest.fail("resolution must not reconstruct or revalidate unrelated bundle content")

    monkeypatch.setattr(DatasetBundle, "__post_init__", forbid_bundle_reconstruction)
    result = resolve_party_graph_views(bundle, partition, parties, party_id=7)[source.name]
    _assert_projection(result, _reference_projection(source, access), source, access)
    np.testing.assert_equal(asdict(bundle), before)
    np.testing.assert_equal([asdict(party) for party in parties], before_parties)
    assert bundle.graph_views is original_registry and bundle.parties == ()
    for field, reference in references.items():
        assert getattr(source, field) is reference
    assert graph_partition_checksum(partition) == checksum
    assert not set(result.node_raw_ids).intersection(source.node_raw_ids)
    # Unsupported unreferenced sources and unrelated features grant no access
    # and are not materialized or required to pass this graph-only validator.
    bundle.graph_views["unreferenced"] = object()
    bundle.feature_views["unrelated"] = object()
    assert list(resolve_party_graph_views(bundle, partition, parties, party_id=7)) == ["relations"]
