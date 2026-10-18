"""Synthetic G1a contract tests; reference encoding does not call production hashes."""

import hashlib
import json
import struct
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, replace

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
    validate_graph_partition,
)


def _ids(values=()):
    return np.array(values, dtype=np.int64)


def _graph(**overrides):
    arguments = dict(
        name="rélations", node_raw_ids=(7, "7", 7, -5),
        node_type_ids=_ids([1, 1, 0, 1]),
        node_type_names=("aux", "entity", "unused_type"),
        edge_index=_ids([[2, 0, 1, 0, 0], [0, 1, 0, 0, 1]]),
        relation_ids=_ids([1, 0, 0, 1, 1]),
        relation_names=("z", "a", "unused_relation"),
        user_to_node=_ids([-1, 2]), item_to_node=_ids([0, 1, 2]),
        metadata={"not_serializable": object()},
    )
    arguments.update(overrides)
    return GraphView(**arguments)


@pytest.fixture
def bundle():
    graph = _graph()
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
        user_id_map={10: 0, "10": 1}, item_id_map={"b": 1, "a": 0, 20: 2},
        graph_views={graph.name: graph}, metadata={"descriptive": []},
    )


def _party(party_id=7, names=("rélations",)):
    return PartyData(
        party_id=party_id, train_interaction_ids=_ids(), valid_interaction_ids=_ids(),
        test_interaction_ids=_ids(), user_ids=_ids(), item_ids=_ids(),
        feature_view_names=(), graph_view_names=names, owns_labels=False, metadata={},
    )


def _partition(bundle, accesses=None, **overrides):
    if accesses is None:
        accesses = {7: (GraphAccess(view_name="rélations"),)}
    names = {access.view_name for entries in accesses.values() for access in entries}
    arguments = dict(
        name="explicit", party_access=accesses,
        source_graph_fingerprints={
            name: graph_view_fingerprint(bundle.graph_views[name]) for name in names
        },
        source_domain_fingerprint=graph_domain_fingerprint(bundle), metadata={},
    )
    arguments.update(overrides)
    return GraphPartition(**arguments)


def _reference_hash(payload):
    # Independent reference uses standard-library JSON and struct, not any
    # production helper, NumPy array serialization, or dataclass traversal.
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _reference_array(shape, flattened):
    binary = b"".join(struct.pack("<q", value) for value in flattened)
    return {"shape": shape, "dtype": "<i8", "sha256": hashlib.sha256(binary).hexdigest()}


def _reference_payloads():
    graph = {
        "format": "thesisfedrec.graph_view", "version": 1, "name": "rélations",
        "node_raw_ids": [
            {"type": "int", "value": 7}, {"type": "str", "value": "7"},
            {"type": "int", "value": 7}, {"type": "int", "value": -5},
        ],
        "node_type_ids": _reference_array([4], [1, 1, 0, 1]),
        "node_type_names": ["aux", "entity", "unused_type"],
        "edge_index": _reference_array([2, 5], [2, 0, 1, 0, 0, 0, 1, 0, 0, 1]),
        "relation_ids": _reference_array([5], [1, 0, 0, 1, 1]),
        "relation_names": ["z", "a", "unused_relation"],
        "user_to_node": _reference_array([2], [-1, 2]),
        "item_to_node": _reference_array([3], [0, 1, 2]),
    }
    domain = {
        "format": "thesisfedrec.graph_domain", "version": 1,
        "name": "synthetic", "schema_version": "1.0",
        "user_raw_ids": [{"type": "int", "value": 10}, {"type": "str", "value": "10"}],
        "item_raw_ids": [
            {"type": "str", "value": "a"}, {"type": "str", "value": "b"},
            {"type": "int", "value": 20},
        ],
    }
    partition = {
        "format": "thesisfedrec.graph_partition", "version": 1,
        "source_domain_fingerprint": _reference_hash(domain),
        "source_graph_fingerprints": [
            {"view_name": "rélations", "fingerprint": _reference_hash(graph)},
        ],
        "parties": [
            {"party_id": 7, "views": [{
                "view_name": "rélations", "node_ids": _reference_array([3], [0, 1, 2]),
                "edge_ids": _reference_array([2], [0, 3]),
            }]},
            {"party_id": 41, "views": []},
        ],
    }
    return graph, domain, partition


# Filled from the independent reference encoding above, not production output.
_EXPECTED_HASHES = (
    "c40f4b42cb0eaf6b3aefe06a5cef8ead557216dfddcf6abd494ce1d1ab7f4290",
    "6a63805491e8c59cf975d58c8edbf9a4c5d54287aabb16693d6985b3c2e661ec",
    "cc1267584294164217f843d1adb4ccba8dd57b762a1a2734b015003937a73e85",
)


def test_fixed_reference_hashes(bundle):
    expected = tuple(_reference_hash(payload) for payload in _reference_payloads())
    assert expected == _EXPECTED_HASHES
    access = GraphAccess(view_name="rélations", node_ids=_ids([0, 1, 2]), edge_ids=_ids([0, 3]))
    partition = _partition(bundle, {41: (), 7: (access,)})
    validate_graph_partition(partition, bundle, (_party(41, ()), _party()))
    assert graph_view_fingerprint(bundle.graph_views["rélations"]) == expected[0]
    assert graph_domain_fingerprint(bundle) == expected[1]
    assert graph_partition_checksum(partition) == expected[2]


def _check_current_array(bundle, field, public_path):
    view = bundle.graph_views["rélations"]
    if public_path == "fingerprint":
        return graph_view_fingerprint(view)
    # Only this array differs from the fixed fixture. Bind its underlying
    # current content independently, so a stale hash cannot mask the defect.
    payload = _reference_payloads()[0]
    array = np.asarray(getattr(view, field))
    payload[field] = _reference_array(list(array.shape), array.ravel().tolist())
    partition = GraphPartition(
        name="current-content", party_access={7: (GraphAccess(view_name=view.name),)},
        source_graph_fingerprints={view.name: _reference_hash(payload)},
        source_domain_fingerprint=_EXPECTED_HASHES[1], metadata={},
    )
    return validate_graph_partition(partition, bundle, (_party(),))


@pytest.mark.parametrize("public_path", ["fingerprint", "partition"])
@pytest.mark.parametrize("field", [
    "node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node",
])
@pytest.mark.parametrize("invalid", [99, -2])
def test_g1a_rejects_masked_invalid_canonical_arrays(bundle, public_path, field, invalid):
    view = bundle.graph_views["rélations"]
    supplied = np.ma.array(getattr(view, field), mask=False, copy=True)
    supplied.data.flat[0] = invalid
    supplied.mask.flat[0] = True
    # Canonical views are mutable: check a replacement after construction.
    setattr(view, field, supplied)
    before_values = supplied.data.copy()
    before_mask = supplied.mask.copy()
    metadata = view.metadata
    metadata_values = dict(metadata)
    with pytest.raises(TypeError, match=f"{field}.*plain NumPy ndarray"):
        _check_current_array(bundle, field, public_path)
    assert getattr(view, field) is supplied
    assert view.metadata is metadata and view.metadata == metadata_values
    np.testing.assert_array_equal(supplied.data, before_values)
    np.testing.assert_array_equal(supplied.mask, before_mask)


class _ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize("public_path", ["fingerprint", "partition"])
@pytest.mark.parametrize("field", [
    "node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node",
])
@pytest.mark.parametrize("representation", ["masked", "all_false", "unmasked", "subclass"])
def test_g1a_rejects_otherwise_valid_array_subclasses(bundle, public_path, field, representation):
    view = bundle.graph_views["rélations"]
    original = getattr(view, field)
    if representation == "subclass":
        supplied = original.view(_ArraySubclass)
    else:
        mask = {
            "masked": np.ones(original.shape, dtype=bool),
            "all_false": np.zeros(original.shape, dtype=bool),
            "unmasked": np.ma.nomask,
        }[representation]
        supplied = np.ma.array(
            original, mask=mask,
        )
    setattr(view, field, supplied)
    before_mask = np.ma.getmaskarray(supplied).copy()
    with pytest.raises(TypeError, match=f"{field}.*plain NumPy ndarray"):
        _check_current_array(bundle, field, public_path)
    assert getattr(view, field) is supplied
    np.testing.assert_array_equal(np.asarray(supplied), original)
    np.testing.assert_array_equal(np.ma.getmaskarray(supplied), before_mask)


@pytest.mark.parametrize("field", ["user_to_node", "item_to_node"])
@pytest.mark.parametrize("value", [-1, 0, 3])
def test_g1a_plain_mapping_boundaries_remain_valid(bundle, field, value):
    view = bundle.graph_views["rélations"]
    mapping = np.full_like(getattr(view, field), value)
    setattr(view, field, mapping)
    before = mapping.copy()
    assert len(_check_current_array(bundle, field, "fingerprint")) == 64
    assert _check_current_array(bundle, field, "partition") is None
    assert getattr(view, field) is mapping
    np.testing.assert_array_equal(mapping, before)


@pytest.mark.parametrize("field", ["node_ids", "edge_ids"])
@pytest.mark.parametrize("invalid, error", [
    ([0], TypeError), ((0,), TypeError), (1, TypeError),
    (np.array(0, dtype=np.int64), ValueError), (_ids([[0]]), ValueError),
    (np.array([0], dtype=np.int32), TypeError), (np.array([0], dtype=np.uint64), TypeError),
    (np.array([], dtype=float), TypeError),
    (_ids([-1, 0]), ValueError), (_ids([1, 0]), ValueError), (_ids([0, 0]), ValueError),
    (np.ma.array([-1], mask=[True], dtype=np.int64), ValueError),
    (np.ma.array([0, 0], mask=[False, True], dtype=np.int64), ValueError),
])
def test_invalid_selections_are_rejected_without_repair(field, invalid, error):
    before = deepcopy(invalid)
    with pytest.raises(error):
        GraphAccess(view_name="rélations", **{field: invalid})
    np.testing.assert_equal(invalid, before)


@pytest.mark.parametrize("field", ["node_ids", "edge_ids"])
def test_array_subclass_masks_are_not_permission_selectors(field):
    supplied = np.ma.array([0, 1], mask=[True, False], dtype=np.int64)
    access = GraphAccess(view_name="rélations", **{field: supplied})
    np.testing.assert_array_equal(getattr(access, field), [0, 1])
    np.testing.assert_array_equal(supplied.mask, [True, False])
    np.testing.assert_array_equal(supplied.data, [0, 1])


@pytest.mark.parametrize("invalid, error", [(None, TypeError), (7, TypeError), ("", ValueError), (" \t", ValueError)])
def test_descriptor_and_partition_names(bundle, invalid, error):
    with pytest.raises(error):
        GraphAccess(view_name=invalid)
    with pytest.raises(error):
        _partition(bundle, name=invalid)
    with pytest.raises(error):
        _partition(bundle, source_graph_fingerprints={invalid: "a" * 64})


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


@pytest.mark.parametrize("invalid", [True, False, np.int64(7), _IntSubclass(7), -1, 7.0, "7"])
def test_party_ids_require_exact_builtin_nonnegative_ints(bundle, invalid):
    with pytest.raises(ValueError, match="exact built-in"):
        _partition(bundle, {invalid: (GraphAccess(view_name="rélations"),)})
    party = _party()
    party.party_id = invalid  # PartyData is mutable and has intentionally looser rules.
    with pytest.raises(ValueError, match="exact built-in"):
        validate_graph_partition(_partition(bundle), bundle, (party,))


@pytest.mark.parametrize("invalid", [None, 7, "", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 63 + "\n"])
@pytest.mark.parametrize("binding", ["graph", "domain"])
def test_invalid_fingerprint_strings(bundle, invalid, binding):
    arguments = (
        {"source_graph_fingerprints": {"rélations": invalid}}
        if binding == "graph" else {"source_domain_fingerprint": invalid}
    )
    with pytest.raises((TypeError, ValueError), match="fingerprint"):
        _partition(bundle, **arguments)


@pytest.mark.parametrize("overrides, message", [
    ({"party_access": []}, "mapping"),
    ({"party_access": {7: []}}, "tuples of GraphAccess"),
    ({"party_access": {7: (object(),)}}, "tuples of GraphAccess"),
    ({"source_graph_fingerprints": []}, "mapping"),
    ({"metadata": []}, "dictionary"),
])
def test_partition_container_types(bundle, overrides, message):
    partition = _partition(bundle)
    with pytest.raises(TypeError, match=message):
        replace(partition, **overrides)


def test_duplicate_descriptors_and_exact_bindings(bundle):
    access = GraphAccess(view_name="rélations")
    with pytest.raises(ValueError, match="duplicate"):
        _partition(bundle, {7: (access, access)})
    with pytest.raises(ValueError, match="exactly"):
        _partition(bundle, source_graph_fingerprints={})
    with pytest.raises(ValueError, match="exactly"):
        _partition(bundle, source_graph_fingerprints={"rélations": "a" * 64, "extra": "b" * 64})
    with pytest.raises(ValueError, match="exactly"):
        _partition(bundle, {}, source_graph_fingerprints={"rélations": "a" * 64})
    # Even a descriptor with zero rows/edges must bind its source.
    with pytest.raises(ValueError, match="exactly"):
        _partition(bundle, {7: (GraphAccess(view_name="rélations", node_ids=_ids(), edge_ids=_ids()),)},
                   source_graph_fingerprints={})


@pytest.mark.parametrize("field", ["node_ids", "edge_ids"])
@pytest.mark.parametrize("change", ["shape", "dtype"])
@pytest.mark.parametrize("empty", [False, True])
def test_selection_input_and_header_isolation(bundle, field, change, empty):
    supplied = _ids() if empty else _ids([0, 1])
    original = supplied.copy()
    arguments = dict(node_ids=_ids(range(4)), edge_ids=_ids())
    arguments[field] = supplied
    access = GraphAccess(view_name="rélations", **arguments)
    partition = _partition(bundle, {7: (access,)})
    validate_graph_partition(partition, bundle, (_party(),))
    checksum = graph_partition_checksum(partition)
    supplied[:] = 0
    first, second = getattr(access, field), getattr(access, field)
    with pytest.raises(ValueError, match="read-only"):
        first[:] = 0
    with pytest.raises(ValueError, match="WRITEABLE"):
        first.setflags(write=True)
    if change == "shape":
        first.shape = (1, first.size)
    else:
        first.dtype = np.int32
    later = getattr(access, field)
    for array in (second, later):
        assert array.ndim == 1 and array.dtype == np.int64
        assert not array.flags.writeable
        np.testing.assert_array_equal(array, original)
    assert first is not second and second is not later
    assert second.base is later.base  # Shared immutable storage, independent headers.
    assert graph_partition_checksum(partition) == checksum
    validate_graph_partition(partition, bundle, (_party(),))


def test_mapping_copies_readonly_collections_and_shallow_metadata(bundle):
    access = GraphAccess(view_name="rélations")
    requested = {7: (access,), 41: ()}
    bindings = {"rélations": graph_view_fingerprint(bundle.graph_views["rélations"])}
    metadata = {"nested": [], "descriptive": "original"}
    partition = _partition(bundle, requested, source_graph_fingerprints=bindings, metadata=metadata)
    checksum = graph_partition_checksum(partition)
    requested[7] = ()
    requested[99] = (access,)
    del requested[41]
    bindings.clear()
    metadata["descriptive"] = "changed input"
    metadata["nested"].append(object())
    partition.metadata["permission"] = "all"  # Never grants access or enters a hash.
    assert partition.metadata["descriptive"] == "original"
    assert partition.metadata["nested"] is metadata["nested"]
    assert partition.party_access == {7: (access,), 41: ()}
    for mapping, key, replacement, new_key in (
        (partition.party_access, 7, (), 99),
        (partition.source_graph_fingerprints, "rélations", "b" * 64, "new"),
    ):
        with pytest.raises(TypeError):
            mapping[key] = replacement
        with pytest.raises(TypeError):
            mapping[new_key] = replacement
        with pytest.raises(TypeError):
            del mapping[key]
    with pytest.raises(TypeError):
        partition.party_access[7][0] = access
    with pytest.raises(FrozenInstanceError):
        partition.name = "changed"
    with pytest.raises(FrozenInstanceError):
        access.view_name = "changed"
    assert not hasattr(access, "__dict__") and not hasattr(partition, "__dict__")
    assert all(field.kw_only for field in fields(access) + fields(partition))
    with pytest.raises(TypeError):
        GraphAccess("rélations")
    with pytest.raises(TypeError):
        GraphPartition("name", {}, {}, "a" * 64, {})
    validate_graph_partition(partition, bundle, (_party(), _party(41, ())))
    assert graph_partition_checksum(partition) == checksum


@pytest.mark.parametrize("nodes, edges, valid", [
    (None, None, True), (None, [], True), ([], [], True), ([], None, False),
    ([0, 1, 2, 3], None, True), ([3], [], True), ([0], [3], True),
    ([0, 1], [1, 2, 4], True), ([0], [1], False), ([1], [1], False),
    ([0, 1, 2], [0], True), ([0], None, False), ([4], [], False), (None, [5], False),
])
def test_literal_selection_and_endpoint_rules(bundle, nodes, edges, valid):
    access = GraphAccess(
        view_name="rélations", node_ids=None if nodes is None else _ids(nodes),
        edge_ids=None if edges is None else _ids(edges),
    )
    partition = _partition(bundle, {7: (access,)})
    if valid:
        assert validate_graph_partition(partition, bundle, (_party(),)) is None
    else:
        with pytest.raises(ValueError, match="out-of-range|endpoints outside"):
            validate_graph_partition(partition, bundle, (_party(),))


@pytest.mark.parametrize("explicit_entry", [False, True])
def test_absent_access_known_party_and_empty_roster(bundle, explicit_entry):
    partition = _partition(bundle, {41: ()} if explicit_entry else {})
    assert validate_graph_partition(partition, bundle, (_party(41, ()),)) is None
    assert validate_graph_partition(_partition(bundle, {}), bundle, ()) is None


def test_none_empty_and_explicit_full_have_distinct_checksums(bundle):
    selections = [(None, None), (_ids(), _ids()), (_ids(range(4)), _ids(range(5)))]
    checksums = {
        graph_partition_checksum(_partition(bundle, {7: (GraphAccess(
            view_name="rélations", node_ids=nodes, edge_ids=edges,
        ),)})) for nodes, edges in selections
    }
    checksums.add(graph_partition_checksum(_partition(bundle, {})))
    checksums.add(graph_partition_checksum(_partition(bundle, {7: ()})))
    assert len(checksums) == 5


@pytest.mark.parametrize("roster, error", [
    (None, TypeError), ("parties", TypeError), ({7: _party()}, TypeError),
    ((object(),), TypeError), ((_party(), _party()), ValueError),
    ((_party(41, ()),), ValueError),
])
def test_invalid_rosters_and_unknown_parties(bundle, roster, error):
    with pytest.raises(error):
        validate_graph_partition(_partition(bundle), bundle, roster)


@pytest.mark.parametrize("names", [(), ("other",), ("rélations", "rélations"), (None,), (" ",), ["rélations"]])
def test_coarse_summary_is_exact_and_revalidated(bundle, names):
    party = _party()
    party.graph_view_names = names
    with pytest.raises((ValueError, TypeError), match="graph_view_names"):
        validate_graph_partition(_partition(bundle), bundle, (party,))


def test_empty_descriptors_are_visible_and_all_parties_are_checked(bundle):
    empty = GraphAccess(view_name="rélations", node_ids=_ids(), edge_ids=_ids())
    partition = _partition(bundle, {7: (empty,)})
    with pytest.raises(ValueError, match="graph_view_names"):
        validate_graph_partition(partition, bundle, (_party(7, ()),))
    invalid = GraphAccess(view_name="rélations", node_ids=_ids([0]), edge_ids=_ids([1]))
    partition = _partition(bundle, {7: (GraphAccess(view_name="rélations"),), 41: (invalid,)})
    with pytest.raises(ValueError, match="endpoints outside"):
        validate_graph_partition(partition, bundle, (_party(), _party(41)))


@pytest.mark.parametrize("mutation, error, message", [
    ("registry_type", TypeError, "dictionary"),
    ("unknown", ValueError, "unknown GraphView"),
    ("value_type", TypeError, "GraphView objects"),
    ("key_name", ValueError, "keys must match"),
    ("bad_shape", ValueError, "one-dimensional"),
    ("bad_dtype", TypeError, "int64"),
    ("duplicate_edge", ValueError, "duplicate"),
    ("bad_raw_id", TypeError, "exact built-in"),
])
def test_current_referenced_graph_is_revalidated(bundle, mutation, error, message):
    partition = _partition(bundle)
    view = bundle.graph_views["rélations"]
    if mutation == "registry_type":
        bundle.graph_views = []
    elif mutation == "unknown":
        bundle.graph_views.clear()
    elif mutation == "value_type":
        bundle.graph_views["rélations"] = object()
    elif mutation == "key_name":
        view.name = "other"
    elif mutation == "bad_shape":
        view.node_type_ids.shape = (2, 2)
    elif mutation == "bad_dtype":
        view.edge_index.dtype = np.int32
    elif mutation == "duplicate_edge":
        view.edge_index[:, 0] = view.edge_index[:, 3]
    else:
        view.node_raw_ids = (_IntSubclass(7), "7", 7, -5)
    with pytest.raises(error, match=message):
        validate_graph_partition(partition, bundle, (_party(),))


@pytest.mark.parametrize("field", [
    "node_raw_ids", "node_type_ids", "node_type_names", "edge_index", "relation_ids",
    "relation_names", "user_to_node", "item_to_node",
])
def test_valid_source_content_changes_are_stale(bundle, field):
    partition = _partition(bundle)
    view = bundle.graph_views["rélations"]
    replacements = {
        "node_raw_ids": (8, "7", 7, -5), "node_type_names": ("aux", "entity", "changed_unused"),
        "relation_names": ("z", "a", "changed_unused"),
    }
    if field in replacements:
        setattr(view, field, replacements[field])
    elif field == "node_type_ids":
        view.node_type_ids[3] = 0
    elif field == "edge_index":
        view.edge_index[0, 0] = 3
    elif field == "relation_ids":
        view.relation_ids[0] = 0
    elif field == "user_to_node":
        view.user_to_node[0] = 0
    else:
        view.item_to_node[0] = 3
    view.validate()  # All changes are legal; source identity, not structure, rejects them.
    with pytest.raises(ValueError, match="stale source graph"):
        validate_graph_partition(partition, bundle, (_party(),))


@pytest.mark.parametrize("field", ["user_to_node", "item_to_node"])
@pytest.mark.parametrize("length", [0, 1, 4])
def test_complete_processed_mapping_lengths_rechecked(bundle, field, length):
    view = bundle.graph_views["rélations"]
    setattr(view, field, np.full(length, -1, dtype=np.int64))
    partition = _partition(bundle)  # Fresh source binding cannot excuse a short mapping.
    with pytest.raises(ValueError, match="complete processed namespace"):
        validate_graph_partition(partition, bundle, (_party(),))


@pytest.mark.parametrize("field", ["user_id_map", "item_id_map"])
def test_same_size_processed_namespace_remapping_is_stale(bundle, field):
    partition = _partition(bundle)
    mapping = getattr(bundle, field)
    first, second = list(mapping)[:2]
    mapping[first], mapping[second] = mapping[second], mapping[first]
    with pytest.raises(ValueError, match="stale source_domain"):
        validate_graph_partition(partition, bundle, (_party(),))


@pytest.mark.parametrize("field", ["name", "schema_version"])
def test_domain_name_and_schema_are_bound(bundle, field):
    partition = _partition(bundle)
    setattr(bundle, field, "different")
    with pytest.raises(ValueError, match="stale source_domain"):
        validate_graph_partition(partition, bundle, (_party(),))


@pytest.mark.parametrize("mapping", [{True: 0}, {np.int64(7): 0}, {7.0: 0}, {None: 0}, {7: 1}, {7: 0, 8: 0}, []])
def test_invalid_current_domain_maps_fail(bundle, mapping):
    partition = _partition(bundle)
    bundle.user_id_map = mapping
    with pytest.raises((TypeError, ValueError)):
        graph_domain_fingerprint(bundle)
    with pytest.raises((TypeError, ValueError)):
        validate_graph_partition(partition, bundle, (_party(),))


def test_existing_domain_subclass_and_processed_id_semantics_are_not_strengthened(bundle):
    original = graph_domain_fingerprint(bundle)
    # These values are already valid under DatasetBundle's existing map validator.
    equivalent = replace(
        bundle, user_id_map={_IntSubclass(10): False, _StrSubclass("10"): 1.0},
        item_id_map={np.str_("a"): np.int64(0), "b": 1, _IntSubclass(20): 2},
    )
    assert graph_domain_fingerprint(equivalent) == original
    validate_graph_partition(_partition(bundle), equivalent, (_party(),))


@pytest.mark.parametrize("mapping_mode", ["none", "unmapped", "many_to_one"])
def test_legal_graph_cases_and_optional_alignment(bundle, mapping_mode):
    view = bundle.graph_views["rélations"]
    if mapping_mode == "none":
        view.user_to_node = view.item_to_node = None
    else:
        value = -1 if mapping_mode == "unmapped" else 2
        view.user_to_node = np.full(2, value, dtype=np.int64)
        view.item_to_node = np.full(3, value, dtype=np.int64)
    partition = _partition(bundle)
    validate_graph_partition(partition, bundle, (_party(),))
    # Fixture retains a self-loop, reverse arcs, relation-distinct edges, an
    # isolated node, unused vocabularies, and a type-scoped repeated raw ID.
    np.testing.assert_array_equal(view.edge_index, [[2, 0, 1, 0, 0], [0, 1, 0, 0, 1]])


@pytest.mark.parametrize("unused_vocab", [False, True])
@pytest.mark.parametrize("selectors", [None, ()])
def test_empty_graphs_and_none_empty_selection_combinations(bundle, unused_vocab, selectors):
    view = _graph(
        node_raw_ids=(), node_type_ids=_ids(), edge_index=np.empty((2, 0), dtype=np.int64),
        relation_ids=_ids(), node_type_names=("unused",) if unused_vocab else (),
        relation_names=("unused",) if unused_vocab else (),
        user_to_node=_ids([-1, -1]), item_to_node=_ids([-1, -1, -1]),
    )
    bundle.graph_views = {view.name: view}
    for edges in (None, _ids()):
        access = GraphAccess(view_name=view.name, node_ids=None if selectors is None else _ids(), edge_ids=edges)
        validate_graph_partition(_partition(bundle, {7: (access,)}), bundle, (_party(),))
    empty_domain = DatasetBundle(
        name="empty", schema_version="1.0",
        interactions=InteractionTable(interaction_ids=_ids(), user_ids=_ids(), item_ids=_ids(),
                                      timestamps=_ids(), values=np.empty(0, dtype=np.float32)),
        splits=InteractionSplits(train_interaction_ids=_ids(), valid_interaction_ids=_ids(), test_interaction_ids=_ids()),
        user_id_map={}, item_id_map={}, metadata={},
        graph_views={view.name: replace(view, user_to_node=_ids(), item_to_node=_ids())},
    )
    validate_graph_partition(_partition(empty_domain), empty_domain, (_party(),))


def test_edgeless_isolated_nodes_and_none_vs_empty_mapping_identity(bundle):
    view = _graph(edge_index=np.empty((2, 0), dtype=np.int64), relation_ids=_ids(), relation_names=())
    bundle.graph_views = {view.name: view}
    validate_graph_partition(_partition(bundle), bundle, (_party(),))
    assert graph_view_fingerprint(replace(view, user_to_node=None)) != graph_view_fingerprint(
        replace(view, user_to_node=_ids())
    )


def test_metadata_names_and_insertion_order_identity_rules(bundle):
    second = _graph(name="alpha")
    bundle.graph_views[second.name] = second
    access_a, access_b = GraphAccess(view_name=second.name), GraphAccess(view_name="rélations")
    partition = _partition(bundle, {41: (), 7: (access_b, access_a)})
    checksum = graph_partition_checksum(partition)
    domain = graph_domain_fingerprint(bundle)
    source = graph_view_fingerprint(second)
    reordered = replace(
        partition, name="renamed", metadata={"opaque": object()},
        party_access={7: (access_a, access_b), 41: ()},
        source_graph_fingerprints=dict(reversed(list(partition.source_graph_fingerprints.items()))),
    )
    bundle.user_id_map = dict(reversed(list(bundle.user_id_map.items())))
    bundle.item_id_map = dict(reversed(list(bundle.item_id_map.items())))
    bundle.metadata["no_identity"] = object()
    second.metadata["no_identity"] = bundle
    assert graph_domain_fingerprint(bundle) == domain
    assert graph_view_fingerprint(second) == source
    assert graph_partition_checksum(reordered) == checksum
    validate_graph_partition(reordered, bundle, (_party(41, ()), _party(7, ("rélations", "alpha"))))
    assert graph_view_fingerprint(replace(second, name="renamed")) != source


@pytest.mark.parametrize("layout", ["strided", "fortran", "negative_stride", "read_only"])
def test_array_storage_layout_and_fingerprinting_do_not_change_sources(bundle, layout):
    original = bundle.graph_views["rélations"]
    view = deepcopy(original)
    names = ("node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node")
    for name in names:
        array = getattr(view, name)
        if layout == "fortran":
            changed = np.asfortranarray(array)
        elif layout == "read_only":
            changed = array.copy()
            changed.setflags(write=False)
        elif layout == "negative_stride":
            changed = array[..., ::-1].copy()[..., ::-1]
        else:
            storage = np.empty((*array.shape[:-1], array.shape[-1] * 2), dtype=np.int64)
            changed = storage[..., ::2]
            changed[...] = array
        setattr(view, name, changed)
    before = {name: (getattr(view, name), getattr(view, name).copy(), getattr(view, name).strides,
                     getattr(view, name).__array_interface__["data"]) for name in names}
    assert graph_view_fingerprint(view) == graph_view_fingerprint(original)
    bundle.graph_views = {view.name: view}
    validate_graph_partition(_partition(bundle), bundle, (_party(),))
    for name, (array, values, strides, pointer) in before.items():
        assert getattr(view, name) is array
        assert array.strides == strides and array.__array_interface__["data"] == pointer
        assert array.dtype == np.int64
        if layout == "read_only":
            assert not array.flags.writeable
        np.testing.assert_array_equal(array, values)


@pytest.mark.parametrize("ordering", ["node", "edge", "node_vocabulary", "relation_vocabulary"])
def test_isomorphic_reordering_changes_identity(ordering):
    view = _graph()
    before = graph_view_fingerprint(view)
    if ordering == "node":
        order = _ids([2, 0, 1, 3])
        inverse = np.argsort(order)
        view.node_raw_ids = tuple(view.node_raw_ids[i] for i in order)
        view.node_type_ids = view.node_type_ids[order]
        view.edge_index = inverse[view.edge_index]
        for name in ("user_to_node", "item_to_node"):
            mapping = getattr(view, name)
            mapping[mapping >= 0] = inverse[mapping[mapping >= 0]]
    elif ordering == "edge":
        view.edge_index = view.edge_index[:, ::-1]
        view.relation_ids = view.relation_ids[::-1]
    elif ordering == "node_vocabulary":
        view.node_type_names = tuple(reversed(view.node_type_names))
        view.node_type_ids = 2 - view.node_type_ids
    else:
        view.relation_names = tuple(reversed(view.relation_names))
        view.relation_ids = 2 - view.relation_ids
    assert graph_view_fingerprint(view) != before


def test_typed_raw_ids_and_optional_bindings_affect_identity(bundle):
    view = _graph(node_raw_ids=(7,), node_type_ids=_ids([0]),
                  edge_index=np.empty((2, 0), dtype=np.int64), relation_ids=_ids(),
                  user_to_node=None, item_to_node=None)
    assert graph_view_fingerprint(view) != graph_view_fingerprint(replace(view, node_raw_ids=("7",)))
    domain = graph_domain_fingerprint(bundle)
    bundle.item_id_map = {"b": 1, "a": 0, "20": 2}
    assert graph_domain_fingerprint(bundle) != domain
    partition = _partition(bundle)
    checksum = graph_partition_checksum(partition)
    assert graph_partition_checksum(replace(partition, source_domain_fingerprint="a" * 64)) != checksum
    assert graph_partition_checksum(replace(partition, source_graph_fingerprints={"rélations": "b" * 64})) != checksum


def test_validation_preserves_inputs_and_does_not_reconstruct_bundle(bundle, monkeypatch):
    partition = _partition(bundle)
    party = _party(41)
    partition = replace(partition, party_access={41: partition.party_access[7]})
    # Roster IDs need not be contiguous and sample ownership does not grant or
    # limit graph access. Leave all sample/entity arrays empty deliberately.
    before_party = deepcopy(asdict(party))
    before_interactions = deepcopy(asdict(bundle.interactions))
    before_splits = deepcopy(asdict(bundle.splits))
    before_maps = (dict(bundle.user_id_map), dict(bundle.item_id_map))
    before_checksum = graph_partition_checksum(partition)
    original_registry = bundle.graph_views
    original_arrays = {name: getattr(bundle.graph_views["rélations"], name) for name in (
        "node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node",
    )}

    def forbid_reconstruction(*args, **kwargs):
        pytest.fail("graph validation must not reconstruct or validate the entire bundle")

    monkeypatch.setattr(DatasetBundle, "__post_init__", forbid_reconstruction)
    bundle.graph_views["unreferenced"] = object()
    bundle.feature_views["unrelated"] = object()
    validate_graph_partition(partition, bundle, (party,))
    assert bundle.graph_views is original_registry
    for name, array in original_arrays.items():
        assert getattr(bundle.graph_views["rélations"], name) is array
    np.testing.assert_equal(asdict(party), before_party)
    np.testing.assert_equal(asdict(bundle.interactions), before_interactions)
    np.testing.assert_equal(asdict(bundle.splits), before_splits)
    assert (bundle.user_id_map, bundle.item_id_map) == before_maps
    assert graph_partition_checksum(partition) == before_checksum
    assert bundle.parties == ()


def test_embedded_party_rule_remains_unchanged(bundle):
    with pytest.raises(ValueError, match="contiguous party_id"):
        replace(bundle, partition_kind="hfl", partition_name="not_contiguous", parties=(_party(41),))


@pytest.mark.parametrize("function, args", [
    (graph_view_fingerprint, (object(),)),
    (graph_domain_fingerprint, (object(),)),
    (graph_partition_checksum, (object(),)),
])
def test_fingerprint_public_input_types(function, args):
    with pytest.raises(TypeError):
        function(*args)
