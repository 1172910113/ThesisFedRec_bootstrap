"""Hand-declared public genre grants; no real data or method computations."""

from dataclasses import fields, replace
import hashlib
import json
import struct

import numpy as np
import pytest

from src.data.contracts import DatasetBundle, GraphAccess, GraphView, InteractionSplits, InteractionTable, PartyData
from src.data.graph import (
    graph_domain_fingerprint, graph_partition_checksum, graph_view_fingerprint,
    resolve_party_graph_views, validate_graph_partition,
)
from src.data.graph_runtime import (
    check_graph_runtime_binding, close_graph_runtime, prepare_graph_runtime, project_party_graphs,
)
from src.data import graph_scenarios as scenario_module
from src.data.graph_scenarios import build_public_genre_block_partition


NAME = "item_genres_graph"
GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
PUBLIC_IDS = (7, 41, 90)
EXPECTED = {
    90: dict(edges=(0, 2), nodes=(0, 1, 4, 7, 10, 13, 16, 19),
             mapping=(0, 1, -1, -1), local_edges=((0, 1), (2, 2))),
    7: dict(edges=(1, 4), nodes=(0, 2, 5, 8, 11, 14, 17, 20),
            mapping=(0, -1, 1, -1), local_edges=((0, 1), (2, 3))),
    41: dict(edges=(3, 5, 6), nodes=(1, 2, 3, 6, 9, 12, 15, 18, 21),
             mapping=(-1, 0, 1, 2), local_edges=((0, 1, 2), (3, 4, 8))),
}
CONTEXT = dict(source_id="literal/genre-source", split_id="literal/split", recipe_id="ml1m.public_genre_blocks3.v1")


def _ids(values=()):
    return np.array(values, dtype=np.int64)


def _canonical_graph(count=4, edges=None):
    if edges is None:
        edges = ((0, 0, 1, 1, 2, 2, 3), (4, 5, 4, 6, 8, 9, 21)) if count else ((), ())
    return GraphView(
        name=NAME, node_raw_ids=(10, 20, 30, 40)[:count] + GENRES,
        node_type_ids=_ids([0] * count + [1] * 18), node_type_names=("item", "genre"),
        edge_index=_ids(edges), relation_ids=_ids([0] * len(edges[0])),
        relation_names=("has_genre",), item_to_node=_ids(range(count)), user_to_node=None,
        metadata={"arbitrary": object(), "descriptive": {"never_select": [1]}},
    )


def _party(party_id, user, start):
    return PartyData(
        party_id=party_id, train_interaction_ids=_ids([start]),
        valid_interaction_ids=_ids([start + 1]), test_interaction_ids=_ids([start + 2]),
        user_ids=_ids([user]), item_ids=_ids([0, 1, 2, 3]),
        feature_view_names=("unchanged-feature",), graph_view_names=(), owns_labels=True,
        metadata={"nested": [party_id]},
    )


@pytest.fixture
def fixture():
    # Every item and user in the maps has explicit support interactions before
    # construction. External party order/IDs deliberately differ from user IDs.
    bundle = DatasetBundle(
        name="literal-ml1m", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=_ids(range(9)), user_ids=_ids([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            item_ids=_ids([0, 1, 2, 1, 2, 3, 0, 2, 3]), timestamps=_ids(range(9)),
            values=np.ones(9, dtype=np.float32),
        ),
        splits=InteractionSplits(train_interaction_ids=_ids([0, 3, 6]),
                                 valid_interaction_ids=_ids([1, 4, 7]),
                                 test_interaction_ids=_ids([2, 5, 8])),
        user_id_map={"u-a": 0, "u-b": 1, "u-c": 2},
        item_id_map={10: 0, 20: 1, 30: 2, 40: 3}, metadata={},
        graph_views={NAME: _canonical_graph()},
    )
    return bundle, (_party(41, 2, 6), _party(7, 0, 0), _party(90, 1, 3))


def _reference_array(array):
    if array is None:
        return None
    # Independent byte encoder rather than production hashing/array helpers.
    values = array.reshape(-1).tolist()
    encoded = b"".join(struct.pack("<q", int(value)) for value in values)
    return {"shape": list(array.shape), "dtype": "<i8", "sha256": hashlib.sha256(encoded).hexdigest()}


def _reference_graph_digest(view):
    payload = {
        "format": "thesisfedrec.graph_view", "version": 1, "name": view.name,
        "node_raw_ids": [{"type": "int" if type(raw) is int else "str", "value": raw}
                         for raw in view.node_raw_ids],
        "node_type_names": list(view.node_type_names), "relation_names": list(view.relation_names),
    }
    for name in ("node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node"):
        payload[name] = _reference_array(getattr(view, name))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _build(bundle, parties, *, expected=None, public_ids=PUBLIC_IDS, **kwargs):
    if expected is None:
        expected = _reference_graph_digest(bundle.graph_views[NAME])
    return build_public_genre_block_partition(
        bundle, parties, graph_name=kwargs.pop("graph_name", NAME),
        expected_graph_fingerprint=expected, public_party_ids=public_ids, **kwargs,
    )


def test_literal_three_block_grants_and_shallow_party_records(fixture):
    bundle, parties = fixture
    view = bundle.graph_views[NAME]
    expected = _reference_graph_digest(view)
    assert graph_view_fingerprint(view) == expected
    partition, updated = _build(bundle, parties, expected=expected)
    assert partition.name == "ml1m_public_genre_blocks3_catalogue_domain.r1"
    assert dict(partition.source_graph_fingerprints) == {NAME: expected}
    assert partition.source_domain_fingerprint == graph_domain_fingerprint(bundle)
    assert partition.metadata == {
        "recipe_id": "ml1m.public_genre_blocks3.v1", "group_count": 3,
        "assignment": "party_id_mod3", "full_public_party_ids": [7, 41, 90],
        "party_groups": [{"party_id": 7, "group_id": 1}, {"party_id": 41, "group_id": 2},
                         {"party_id": 90, "group_id": 0}],
    }
    assert [party.party_id for party in updated] == [41, 7, 90]
    for original, new in zip(parties, updated):
        assert new is not original and original.graph_view_names == ()
        assert new.graph_view_names == (NAME,)
        for name in ("train_interaction_ids", "valid_interaction_ids", "test_interaction_ids",
                     "user_ids", "item_ids", "feature_view_names", "owns_labels"):
            assert getattr(new, name) is getattr(original, name)
        assert new.metadata is not original.metadata
        assert new.metadata["nested"] is original.metadata["nested"]
        new.metadata["new-key"] = True
        assert "new-key" not in original.metadata
        access, = partition.party_access[original.party_id]
        assert access.view_name == NAME
        np.testing.assert_array_equal(access.node_ids, EXPECTED[original.party_id]["nodes"])
        np.testing.assert_array_equal(access.edge_ids, EXPECTED[original.party_id]["edges"])
        assert access.node_ids.dtype == access.edge_ids.dtype == np.int64
        assert not access.node_ids.flags.writeable and not access.edge_ids.flags.writeable
    assert bundle.parties == ()
    validate_graph_partition(partition, bundle, updated)


def _assert_projection(view, party_id):
    expected = EXPECTED[party_id]
    np.testing.assert_array_equal(view.item_to_node, expected["mapping"])
    np.testing.assert_array_equal(view.edge_index, expected["local_edges"])
    assert view.node_raw_ids == tuple(f"n{i}" for i in range(len(expected["nodes"])))
    assert view.metadata == {} and view.user_to_node is None
    assert view.node_type_names == ("item", "genre") and view.relation_names == ("has_genre",)
    for name in ("node_type_ids", "edge_index", "relation_ids", "item_to_node"):
        array = getattr(view, name)
        assert type(array) is np.ndarray and array.dtype == np.int64
        assert array.flags.owndata and array.flags.c_contiguous


def test_runtime_and_strict_resolver_match_hand_projections(fixture):
    bundle, parties = fixture
    partition, updated = _build(bundle, parties)
    runtime = prepare_graph_runtime(bundle, partition, updated, **CONTEXT)
    binding = runtime.binding_id
    for party_id in (90, 7, 41):
        runtime_view = project_party_graphs(runtime, party_id=party_id, binding_id=binding)[NAME]
        strict_view = resolve_party_graph_views(bundle, partition, updated, party_id=party_id)[NAME]
        _assert_projection(runtime_view, party_id)
        _assert_projection(strict_view, party_id)
        assert not np.shares_memory(runtime_view.edge_index, strict_view.edge_index)
    close_graph_runtime(runtime)
    close_graph_runtime(runtime)
    _assert_projection(runtime_view, 41)
    with pytest.raises(ValueError, match="closed"):
        project_party_graphs(runtime, party_id=7, binding_id=binding)


def test_output_descriptor_and_runtime_isolation_from_mutation(fixture):
    bundle, parties = fixture
    partition, updated = _build(bundle, parties)
    access = partition.party_access[7][0]
    header = access.node_ids
    with pytest.raises(ValueError):
        header[0] = 99
    with pytest.raises(ValueError):
        header.setflags(write=True)
    header.shape = (1, header.size)
    assert access.node_ids.ndim == 1
    runtime = prepare_graph_runtime(bundle, partition, updated, **CONTEXT)
    binding = runtime.binding_id
    first = project_party_graphs(runtime, party_id=7, binding_id=binding)[NAME]
    second = project_party_graphs(runtime, party_id=7, binding_id=binding)[NAME]
    first.edge_index.fill(-5)
    first.item_to_node.fill(-5)
    first.metadata["private"] = "isolated"
    _assert_projection(second, 7)
    bundle.graph_views[NAME].edge_index[1, 0] = 7
    _assert_projection(project_party_graphs(runtime, party_id=7, binding_id=binding)[NAME], 7)
    with pytest.raises(ValueError):
        validate_graph_partition(partition, bundle, updated)
    close_graph_runtime(runtime)
    _assert_projection(second, 7)


def test_grants_independent_of_sample_splits_summaries_and_metadata(fixture):
    bundle, parties = fixture
    original, _ = _build(bundle, parties)
    changed_bundle = replace(bundle, splits=InteractionSplits(
        train_interaction_ids=_ids([0, 1, 3, 4, 6, 7]),
        valid_interaction_ids=_ids([2, 5, 8]), test_interaction_ids=_ids(),
    ))
    changed = tuple(replace(
        party, train_interaction_ids=_ids([start, start + 1]),
        valid_interaction_ids=_ids([start + 2]), test_interaction_ids=_ids(),
        item_ids=_ids([int(changed_bundle.interactions.item_ids[start])]),
        metadata={"train_count": 999, "validation_gain": object()},
    ) for party, start in zip(parties, (6, 0, 3)))
    altered, _ = _build(changed_bundle, changed)
    assert graph_partition_checksum(original) == graph_partition_checksum(altered)
    bundle.graph_views[NAME].metadata = {"mask": object(), "train_count": -100}
    # The consumed-source contract does not inspect interaction/split objects.
    bundle.interactions = bundle.splits = object()
    unread, _ = _build(bundle, parties)
    assert graph_partition_checksum(original) == graph_partition_checksum(unread)


def test_party_order_map_insertion_and_declared_domain_boundaries(fixture):
    bundle, parties = fixture
    first, _ = _build(bundle, parties)
    bundle.item_id_map = dict(reversed(list(bundle.item_id_map.items())))
    second, reordered = _build(bundle, tuple(reversed(parties)))
    assert graph_partition_checksum(first) == graph_partition_checksum(second)
    assert tuple(p.party_id for p in reordered) == (90, 7, 41)
    with pytest.raises(ValueError, match="full public"):
        _build(bundle, parties[1:])
    subset, _ = _build(bundle, parties[1:], public_ids=(7, 90))
    for party_id in (7, 90):
        np.testing.assert_array_equal(subset.party_access[party_id][0].node_ids,
                                      first.party_access[party_id][0].node_ids)
    assert subset.metadata["full_public_party_ids"] == [7, 90]


def test_domain_graph_access_bindings_and_runtime_retirement(fixture):
    bundle, parties = fixture
    partition, updated = _build(bundle, parties)
    runtime = prepare_graph_runtime(bundle, partition, updated, **CONTEXT)
    original_binding = runtime.binding_id
    bundle.user_id_map = {"changed-user": 0, "u-b": 1, "u-c": 2}
    with pytest.raises(ValueError, match="domain"):
        validate_graph_partition(partition, bundle, updated)
    new_partition, new_parties = _build(bundle, parties)
    new_runtime = prepare_graph_runtime(bundle, new_partition, new_parties, **CONTEXT)
    with pytest.raises(ValueError, match="mismatch"):
        check_graph_runtime_binding(new_runtime, binding_id=original_binding)
    assert runtime.binding_id == original_binding
    changed_accesses = dict(new_partition.party_access)
    changed_accesses[7] = (GraphAccess(view_name=NAME, node_ids=_ids([0, 4]), edge_ids=_ids([0])),)
    changed_partition = replace(new_partition, party_access=changed_accesses)
    access_runtime = prepare_graph_runtime(bundle, changed_partition, new_parties, **CONTEXT)
    assert access_runtime.binding_id != new_runtime.binding_id
    for handle in (runtime, new_runtime, access_runtime):
        close_graph_runtime(handle)
        with pytest.raises(ValueError, match="closed"):
            _ = handle.binding_id


def test_zero_edge_grants_keep_six_aux_and_no_phantom_items(fixture):
    bundle, parties = fixture
    bundle.graph_views[NAME] = _canonical_graph(edges=((0, 1, 2, 3), (4, 4, 4, 4)))
    partition, updated = _build(bundle, parties)
    access = partition.party_access[7][0]
    np.testing.assert_array_equal(access.node_ids, [5, 8, 11, 14, 17, 20])
    assert access.edge_ids.shape == (0,) and access.edge_ids is not None
    runtime = prepare_graph_runtime(bundle, partition, updated, **CONTEXT)
    local = project_party_graphs(runtime, party_id=7, binding_id=runtime.binding_id)[NAME]
    assert len(local.node_raw_ids) == 6
    assert local.edge_index.shape == (2, 0)
    np.testing.assert_array_equal(local.item_to_node, [-1, -1, -1, -1])
    close_graph_runtime(runtime)


def test_zero_catalogue_and_empty_public_domain_are_explicit(fixture):
    empty = DatasetBundle(
        name="empty-domain", schema_version="1.0",
        interactions=InteractionTable(interaction_ids=_ids(), user_ids=_ids(), item_ids=_ids(),
                                      timestamps=_ids(), values=np.empty(0, dtype=np.float32)),
        splits=InteractionSplits(train_interaction_ids=_ids(), valid_interaction_ids=_ids(),
                                 test_interaction_ids=_ids()),
        user_id_map={}, item_id_map={}, metadata={}, graph_views={NAME: _canonical_graph(0)},
    )
    parties = tuple(PartyData(
        party_id=party_id, train_interaction_ids=_ids(), valid_interaction_ids=_ids(),
        test_interaction_ids=_ids(), user_ids=_ids(), item_ids=_ids(),
        feature_view_names=(), graph_view_names=(), owns_labels=False, metadata={},
    ) for party_id in (41, 7, 90))
    partition, updated = _build(empty, parties)
    runtime = prepare_graph_runtime(empty, partition, updated, **CONTEXT)
    for party_id in PUBLIC_IDS:
        local = project_party_graphs(runtime, party_id=party_id, binding_id=runtime.binding_id)[NAME]
        assert len(local.node_raw_ids) == 6 and local.item_to_node.shape == (0,)
        assert local.edge_index.shape == (2, 0)
    close_graph_runtime(runtime)
    bundle, _ = fixture
    partition, updated = _build(bundle, (), public_ids=())
    assert updated == () and dict(partition.party_access) == {}
    assert dict(partition.source_graph_fingerprints) == {}
    assert partition.source_domain_fingerprint == graph_domain_fingerprint(bundle)


class _Int(int):
    pass


class _Str(str):
    pass


class _Array(np.ndarray):
    pass


@pytest.mark.parametrize("bad", [[7, 41, 90], None, (True, 41, 90), (np.int64(7), 41, 90), (_Int(7), 41, 90)])
def test_public_domain_exact_container_and_element_types(fixture, bad):
    with pytest.raises(TypeError):
        _build(*fixture, public_ids=bad)


@pytest.mark.parametrize("bad", [(41, 7, 90), (7, 7, 90), (-1, 7, 41), (7, 41, 2**63), (7, 41), (7, 41, 90, 91)])
def test_public_domain_sort_uniqueness_range_and_full_membership(fixture, bad):
    with pytest.raises(ValueError):
        _build(*fixture, public_ids=bad)


@pytest.mark.parametrize("bad", [None, list, [], "parties"])
def test_party_container_must_be_exact_tuple(fixture, bad):
    bundle, _ = fixture
    with pytest.raises(TypeError):
        _build(bundle, bad)


@pytest.mark.parametrize("bad", [True, np.int64(7), _Int(7), "7"])
def test_party_id_exact_types(fixture, bad):
    bundle, parties = fixture
    parties[1].party_id = bad
    with pytest.raises(TypeError):
        _build(bundle, parties)


@pytest.mark.parametrize("bad", [-1, 2**63])
def test_party_id_range(fixture, bad):
    bundle, parties = fixture
    parties[1].party_id = bad
    with pytest.raises(ValueError):
        _build(bundle, parties)


def test_party_id_maximum_and_same_group_descriptor_sharing(fixture):
    bundle, parties = fixture
    maximum = 2**63 - 1
    parties = (replace(parties[0], party_id=maximum), replace(parties[1], party_id=7))
    partition, _ = _build(bundle, parties, public_ids=(7, maximum))
    assert maximum % 3 == 7 % 3
    assert partition.party_access[7][0] is partition.party_access[maximum][0]
    assert not partition.party_access[maximum][0].node_ids.flags.writeable


@pytest.mark.parametrize("names,error", [([NAME], TypeError), ((_Str(NAME),), TypeError),
                                        ((NAME, "other"), ValueError), (("other",), ValueError),
                                        ((NAME, NAME), ValueError)])
def test_existing_other_graph_summaries_are_not_silently_dropped(fixture, names, error):
    bundle, parties = fixture
    parties[0].graph_view_names = names
    with pytest.raises(error):
        _build(bundle, parties)


def test_existing_matching_graph_summary_and_duplicate_records(fixture):
    bundle, parties = fixture
    matching = tuple(replace(party, graph_view_names=(NAME,)) for party in parties)
    _build(bundle, matching)
    with pytest.raises(ValueError, match="unique"):
        _build(bundle, (parties[0], parties[0], parties[2]))
    with pytest.raises(TypeError):
        _build(bundle, (object(), parties[1], parties[2]))


@pytest.mark.parametrize("name,error", [(None, TypeError), (_Str(NAME), TypeError), ("other", ValueError)])
def test_exact_supported_graph_name(fixture, name, error):
    with pytest.raises(error):
        _build(*fixture, graph_name=name)


@pytest.mark.parametrize("expected,error", [(None, TypeError), (7, TypeError), (_Str("a" * 64), TypeError),
                                           ("A" * 64, ValueError), ("a" * 63, ValueError),
                                           ("g" * 64, ValueError), ("0" * 64, ValueError)])
@pytest.mark.parametrize("empty", [False, True])
def test_expected_graph_digest_checked_even_for_empty_roster(fixture, expected, error, empty):
    bundle, parties = fixture
    with pytest.raises(error):
        build_public_genre_block_partition(bundle, () if empty else parties, graph_name=NAME,
                                          expected_graph_fingerprint=expected,
                                          public_party_ids=() if empty else PUBLIC_IDS)


@pytest.mark.parametrize("field", ["node_type_ids", "edge_index", "relation_ids", "item_to_node", "user_to_node"])
@pytest.mark.parametrize("representation", ["subclass", "masked", "int32", "object", "non-native", "metadata"])
def test_graph_array_representation_precedes_semantics(fixture, field, representation):
    bundle, parties = fixture
    graph = bundle.graph_views[NAME]
    value = _ids([-1, -1, -1]) if field == "user_to_node" else getattr(graph, field)
    if representation == "subclass":
        invalid = value.view(_Array)
    elif representation == "masked":
        invalid = np.ma.array(value, mask=False)
    elif representation == "int32":
        invalid = value.astype(np.int32)
    elif representation == "object":
        invalid = value.astype(object)
    elif representation == "non-native":
        invalid = value.astype(np.dtype("int64").newbyteorder("S"))
    else:
        invalid = value.astype(np.dtype("int64", metadata={}))
    setattr(graph, field, invalid)
    with pytest.raises(TypeError):
        _build(bundle, parties, expected="0" * 64)


@pytest.mark.parametrize("field,value", [
    ("name", _Str(NAME)), ("node_type_names", ["item", "genre"]),
    ("node_type_names", (_Str("item"), "genre")), ("relation_names", (_Str("has_genre"),)),
    ("node_raw_ids", list((10, 20, 30, 40) + GENRES)),
    ("node_raw_ids", (_Int(10), 20, 30, 40) + GENRES),
])
def test_graph_container_and_scalar_types(fixture, field, value):
    bundle, parties = fixture
    setattr(bundle.graph_views[NAME], field, value)
    with pytest.raises(TypeError):
        _build(bundle, parties, expected="0" * 64)


@pytest.mark.parametrize("mutation", ["raw-item-order", "genre-order", "node-types", "node-schema", "relation-schema",
                                       "user-anchor", "item-map", "map-length", "edge-order", "reverse-edge",
                                       "aux-edge", "missing-item", "duplicate-edge", "out-of-range", "shape"])
@pytest.mark.parametrize("empty_roster", [False, True])
def test_current_graph_semantic_validation_not_only_supplied_digest(fixture, mutation, empty_roster):
    bundle, parties = fixture
    graph = bundle.graph_views[NAME]
    if mutation == "raw-item-order":
        graph.node_raw_ids = (20, 10, 30, 40) + GENRES
    elif mutation == "genre-order":
        graph.node_raw_ids = (10, 20, 30, 40) + (GENRES[1], GENRES[0]) + GENRES[2:]
    elif mutation == "node-types":
        graph.node_type_ids[4] = 0
    elif mutation == "node-schema":
        graph.node_type_names = ("genre", "item")
    elif mutation == "relation-schema":
        graph.relation_names = ("has_genre", "extra")
    elif mutation == "user-anchor":
        graph.user_to_node = _ids([-1, -1, -1])
    elif mutation == "item-map":
        graph.item_to_node = _ids([1, 0, 2, 3])
    elif mutation == "map-length":
        graph.item_to_node = _ids([0, 1, 2])
    elif mutation == "edge-order":
        graph.edge_index = graph.edge_index[:, ::-1].copy()
    elif mutation == "reverse-edge":
        graph.edge_index[:, 0] = [4, 0]
    elif mutation == "aux-edge":
        graph.edge_index[:, 0] = [4, 5]
    elif mutation == "missing-item":
        graph.edge_index = graph.edge_index[:, :-1].copy()
        graph.relation_ids = graph.relation_ids[:-1].copy()
    elif mutation == "duplicate-edge":
        graph.edge_index[:, 1] = graph.edge_index[:, 0]
    elif mutation == "out-of-range":
        graph.edge_index[1, 0] = 22
    else:
        graph.edge_index = graph.edge_index.reshape(-1)
    expected = _reference_graph_digest(graph)
    with pytest.raises(ValueError):
        _build(bundle, () if empty_roster else parties, expected=expected,
               public_ids=() if empty_roster else PUBLIC_IDS)


def test_strided_read_only_native_arrays_and_no_input_mutation(fixture):
    bundle, parties = fixture
    graph = bundle.graph_views[NAME]
    before = {name: getattr(graph, name).copy() for name in
              ("node_type_ids", "edge_index", "relation_ids", "item_to_node")}
    for name, value in before.items():
        storage = np.empty(value.shape[:-1] + (value.shape[-1] * 2,), dtype=np.int64)
        storage[..., ::2] = value
        strided = storage[..., ::2]
        strided.setflags(write=False)
        setattr(graph, name, strided)
    _build(bundle, parties)
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(graph, name), expected)
        assert not getattr(graph, name).flags.writeable


@pytest.mark.parametrize("mapping,error", [
    ({True: 0, 20: 1, 30: 2, 40: 3}, TypeError),
    ({np.int64(10): 0, 20: 1, 30: 2, 40: 3}, TypeError),
    ({10: np.int64(0), 20: 1, 30: 2, 40: 3}, TypeError),
    ({10: False, 20: 1, 30: 2, 40: 3}, TypeError),
    ({-10: 0, 20: 1, 30: 2, 40: 3}, ValueError),
    ({10: 0, 20: 0, 30: 2, 40: 3}, ValueError),
    ({10: 1, 20: 0, 30: 2, 40: 3}, ValueError),
])
def test_current_item_domain_and_raw_row_binding(fixture, mapping, error):
    bundle, parties = fixture
    bundle.item_id_map = mapping
    with pytest.raises(error):
        _build(bundle, parties)


def test_exact_records_missing_graph_and_capacity_without_large_allocation(fixture):
    bundle, parties = fixture
    class BundleSubclass(DatasetBundle):
        pass
    class GraphSubclass(GraphView):
        pass
    class PartySubclass(PartyData):
        pass
    bundle_args = {field.name: getattr(bundle, field.name) for field in fields(bundle)}
    with pytest.raises(TypeError):
        _build(BundleSubclass(**bundle_args), parties)
    graph = bundle.graph_views[NAME]
    bundle.graph_views[NAME] = GraphSubclass(**{field.name: getattr(graph, field.name) for field in fields(graph)})
    with pytest.raises(TypeError):
        _build(bundle, parties)
    bundle.graph_views[NAME] = graph
    subclass_party = PartySubclass(**{field.name: getattr(parties[0], field.name) for field in fields(parties[0])})
    with pytest.raises(TypeError):
        _build(bundle, (subclass_party,) + parties[1:])
    bundle.graph_views.clear()
    with pytest.raises(ValueError):
        _build(bundle, parties, expected="0" * 64)
    scenario_module._capacity(2**63 - 19, 2**63 - 1)
    for items, edges in ((2**63 - 18, 0), (0, 2**63), (-1, 0), (0, -1)):
        with pytest.raises(ValueError, match="capacity"):
            scenario_module._capacity(items, edges)


def test_plain_map_containers_and_required_numeric_arrays(fixture):
    bundle, parties = fixture
    class DictSubclass(dict):
        pass
    bundle.graph_views = DictSubclass(bundle.graph_views)
    with pytest.raises(TypeError):
        _build(bundle, parties)
    bundle.graph_views = dict(bundle.graph_views)
    bundle.item_id_map = DictSubclass(bundle.item_id_map)
    with pytest.raises(TypeError):
        _build(bundle, parties)
    bundle.item_id_map = dict(bundle.item_id_map)
    graph = bundle.graph_views[NAME]
    for name in ("node_type_ids", "edge_index", "relation_ids"):
        original = getattr(graph, name)
        setattr(graph, name, None)
        with pytest.raises(TypeError):
            _build(bundle, parties, expected="0" * 64)
        setattr(graph, name, original)
    graph.item_to_node = None
    with pytest.raises(ValueError):
        _build(bundle, parties)


@pytest.mark.parametrize("metadata", [None, [], "not-a-dictionary"])
def test_party_metadata_wrong_container_type_fails_before_output(fixture, metadata):
    bundle, parties = fixture
    parties[0].metadata = metadata
    with pytest.raises(TypeError):
        _build(bundle, parties)
