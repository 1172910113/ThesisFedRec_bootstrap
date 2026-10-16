from dataclasses import replace

import numpy as np
import pytest

from src.data import GraphView
from src.data.contracts import (
    DatasetBundle,
    FeatureView,
    InteractionSplits,
    InteractionTable,
)
from src.data.partition import build_hfl_partition


def _graph_fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "name": "relations",
        "node_raw_ids": (7, "7", 7, -5),
        "node_type_ids": np.array([1, 1, 0, 1], dtype=np.int64),
        "node_type_names": ("auxiliary", "entity", "unused_type"),
        "edge_index": np.array([[2, 0, 1, 0, 0], [0, 1, 0, 0, 1]], dtype=np.int64),
        "relation_ids": np.array([1, 0, 0, 1, 1], dtype=np.int64),
        "relation_names": ("z_relation", "a_relation", "unused_relation"),
        "metadata": {"source": "synthetic"},
    }
    fields.update(overrides)
    return fields


def _graph(**overrides: object) -> GraphView:
    return GraphView(**_graph_fields(**overrides))  # type: ignore[arg-type]


@pytest.fixture
def base_bundle() -> DatasetBundle:
    return DatasetBundle(
        name="synthetic",
        schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=np.arange(9, dtype=np.int64),
            user_ids=np.repeat(np.arange(3, dtype=np.int64), 3),
            item_ids=np.tile(np.arange(3, dtype=np.int64), 3),
            timestamps=np.tile(np.arange(3, dtype=np.int64), 3),
            values=np.ones(9, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=np.array([0, 3, 6], dtype=np.int64),
            valid_interaction_ids=np.array([1, 4, 7], dtype=np.int64),
            test_interaction_ids=np.array([2, 5, 8], dtype=np.int64),
        ),
        user_id_map={10: 0, 20: 1, 30: 2},
        item_id_map={100: 0, 200: 1, 300: 2},
        metadata={},
    )


def test_graph_preserves_identity_order_and_explicit_edges() -> None:
    fields = _graph_fields()
    view = GraphView(**fields)  # type: ignore[arg-type]

    assert view.validate() is None
    for name, supplied in fields.items():
        assert getattr(view, name) is supplied
    assert view.user_to_node is None
    assert view.item_to_node is None
    # 7 and "7" differ, and integer 7 is also valid under another node type.
    assert view.node_raw_ids == (7, "7", 7, -5)
    # Explicit reverse arcs, a self-loop, and two relations on 0 -> 1 survive.
    assert list(zip(view.edge_index[0], view.relation_ids, view.edge_index[1])) == [
        (2, 1, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 1, 1)
    ]
    # Node 3 is isolated; unused vocabulary entries are retained in input order.
    assert 3 not in view.edge_index
    assert view.relation_names == ("z_relation", "a_relation", "unused_relation")


def test_opaque_raw_ids_are_not_normalized_or_restricted() -> None:
    raw_ids = (-2, "", "  ", 2**90)
    view = _graph(node_raw_ids=raw_ids, node_type_ids=np.ones(4, dtype=np.int64))

    assert view.node_raw_ids is raw_ids
    assert view.validate() is None


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


@pytest.mark.parametrize(
    "raw_id",
    [True, np.int64(7), np.str_("7"), _IntSubclass(7), _StrSubclass("7"), 7.0, None],
)
@pytest.mark.parametrize("validation", ["constructor", "validate", "bundle"])
def test_raw_ids_require_exact_builtin_types(
    raw_id: object, validation: str, base_bundle: DatasetBundle
) -> None:
    raw_ids = (raw_id, "7", 7, -5)
    with pytest.raises(TypeError, match="exact built-in str or int"):
        if validation == "constructor":
            _graph(node_raw_ids=raw_ids)
        else:
            view = _graph()
            view.node_raw_ids = raw_ids
            if validation == "validate":
                view.validate()
            else:
                replace(base_bundle, graph_views={view.name: view})


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        ("name", None, TypeError, "name must be a string"),
        ("name", "  ", ValueError, "name must not be empty"),
        ("node_raw_ids", [7, "7", 7, -5], TypeError, "node_raw_ids"),
        ("node_raw_ids", (7, "7", -5, 7), ValueError, "node identities"),
        ("node_type_names", ["entity"], TypeError, "tuple of strings"),
        ("node_type_names", ("entity", 7), TypeError, "tuple of strings"),
        ("node_type_names", ("entity", "  "), ValueError, "nonblank"),
        ("node_type_names", ("entity", "entity"), ValueError, "unique"),
        ("node_type_names", (), ValueError, "out-of-range"),
        ("relation_names", ["relation"], TypeError, "tuple of strings"),
        ("relation_names", ("relation", None), TypeError, "tuple of strings"),
        ("relation_names", ("relation", ""), ValueError, "nonblank"),
        ("relation_names", ("relation", "relation"), ValueError, "unique"),
        ("relation_names", (), ValueError, "out-of-range"),
        ("node_type_ids", [1, 1, 0, 1], TypeError, "NumPy array"),
        ("node_type_ids", np.array([[1, 1, 0, 1]]), ValueError, "one-dimensional"),
        ("node_type_ids", np.array([1, 1, 0, 1], dtype=np.int32), TypeError, "int64"),
        ("node_type_ids", np.array([1, 1, 0], dtype=np.int64), ValueError, "per node"),
        ("node_type_ids", np.array([1, -1, 0, 1]), ValueError, "out-of-range"),
        ("node_type_ids", np.array([1, 3, 0, 1]), ValueError, "out-of-range"),
        ("edge_index", [[0], [1]], TypeError, "NumPy array"),
        ("edge_index", np.array([0, 1]), ValueError, "shape"),
        ("edge_index", np.zeros((3, 2), dtype=np.int64), ValueError, "shape"),
        ("edge_index", np.zeros((2, 1, 1), dtype=np.int64), ValueError, "shape"),
        ("edge_index", np.zeros((2, 5), dtype=np.int32), TypeError, "int64"),
        ("edge_index", np.array([[-1], [0]]), ValueError, "out-of-range"),
        ("edge_index", np.array([[0], [4]]), ValueError, "out-of-range"),
        (
            "edge_index",
            np.array([[2, 0, 1, 0, 2], [0, 1, 0, 0, 0]], dtype=np.int64),
            ValueError,
            "duplicate",
        ),
        ("relation_ids", [1, 0, 0, 1, 1], TypeError, "NumPy array"),
        ("relation_ids", np.array([[1, 0, 0, 1, 1]]), ValueError, "one-dimensional"),
        ("relation_ids", np.array([1, 0, 0, 1, 1], dtype=np.float64), TypeError, "int64"),
        ("relation_ids", np.array([1, 0, 0, 1]), ValueError, "per edge"),
        ("relation_ids", np.array([-1, 0, 0, 1, 1]), ValueError, "out-of-range"),
        ("relation_ids", np.array([3, 0, 0, 1, 1]), ValueError, "out-of-range"),
        ("metadata", [], TypeError, "metadata must be a dictionary"),
    ],
)
@pytest.mark.parametrize("validation", ["constructor", "validate", "bundle"])
def test_structural_errors_are_checked_at_construction_and_revalidation(
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
    validation: str,
    base_bundle: DatasetBundle,
) -> None:
    with pytest.raises(error_type, match=message):
        if validation == "constructor":
            _graph(**{field_name: value})
        else:
            view = _graph()
            setattr(view, field_name, value)
            if validation == "validate":
                view.validate()
            else:
                replace(base_bundle, graph_views={"relations": view})


@pytest.mark.parametrize("unused_vocabularies", [False, True])
def test_empty_graphs_are_valid(unused_vocabularies: bool) -> None:
    view = _graph(
        node_raw_ids=(),
        node_type_ids=np.empty(0, dtype=np.int64),
        node_type_names=("unused",) if unused_vocabularies else (),
        edge_index=np.empty((2, 0), dtype=np.int64),
        relation_ids=np.empty(0, dtype=np.int64),
        relation_names=("unused",) if unused_vocabularies else (),
        user_to_node=np.array([-1, -1], dtype=np.int64),
        item_to_node=np.empty(0, dtype=np.int64),
    )
    assert view.validate() is None
    assert view.edge_index.shape == (2, 0)


def test_edgeless_graph_with_isolated_nodes_and_no_relations() -> None:
    view = _graph(
        edge_index=np.empty((2, 0), dtype=np.int64),
        relation_ids=np.empty(0, dtype=np.int64),
        relation_names=(),
    )
    assert len(view.node_raw_ids) == 4
    assert view.validate() is None


def test_observed_vocabulary_ids_need_not_be_contiguous() -> None:
    view = _graph(
        node_type_ids=np.array([2, 2, 0, 2], dtype=np.int64),
        relation_ids=np.array([2, 0, 0, 2, 2], dtype=np.int64),
    )
    assert view.validate() is None
    np.testing.assert_array_equal(view.node_type_ids, [2, 2, 0, 2])
    np.testing.assert_array_equal(view.relation_ids, [2, 0, 0, 2, 2])


def test_empty_graph_attachment_keeps_unmapped_processed_entities(
    base_bundle: DatasetBundle,
) -> None:
    view = _graph(
        node_raw_ids=(),
        node_type_ids=np.empty(0, dtype=np.int64),
        edge_index=np.empty((2, 0), dtype=np.int64),
        relation_ids=np.empty(0, dtype=np.int64),
        user_to_node=np.full(3, -1, dtype=np.int64),
        item_to_node=np.full(3, -1, dtype=np.int64),
    )
    derived = replace(base_bundle, graph_views={view.name: view})
    assert derived.graph_views[view.name] is view
    assert derived.user_id_map is base_bundle.user_id_map
    assert derived.item_id_map is base_bundle.item_id_map


@pytest.mark.parametrize("reference", ["edge_index", "user_to_node", "item_to_node"])
def test_empty_graph_cannot_be_referenced_by_a_node_index(reference: str) -> None:
    fields = _graph_fields(
        node_raw_ids=(),
        node_type_ids=np.empty(0, dtype=np.int64),
        edge_index=np.empty((2, 0), dtype=np.int64),
        relation_ids=np.empty(0, dtype=np.int64),
    )
    if reference == "edge_index":
        fields["edge_index"] = np.array([[0], [0]], dtype=np.int64)
        fields["relation_ids"] = np.array([0], dtype=np.int64)
    else:
        fields[reference] = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="out-of-range|valid node IDs"):
        GraphView(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["user_to_node", "item_to_node"])
@pytest.mark.parametrize(
    ("mapping", "error_type", "message"),
    [
        ([0, 1, 2], TypeError, "NumPy array"),
        (np.array([[0, 1, 2]]), ValueError, "one-dimensional"),
        (np.array([0, 1, 2], dtype=np.int32), TypeError, "int64"),
        (np.empty(0, dtype=np.float64), TypeError, "int64"),
        (np.array([-2, 0, 1]), ValueError, "-1 or valid node IDs"),
        (np.array([0, 4, 1]), ValueError, "-1 or valid node IDs"),
    ],
)
@pytest.mark.parametrize("validation", ["constructor", "validate", "bundle"])
def test_mapping_structure_is_revalidated(
    field_name: str,
    mapping: object,
    error_type: type[Exception],
    message: str,
    validation: str,
    base_bundle: DatasetBundle,
) -> None:
    with pytest.raises(error_type, match=message):
        if validation == "constructor":
            _graph(**{field_name: mapping})
        else:
            view = _graph()
            setattr(view, field_name, mapping)
            if validation == "validate":
                view.validate()
            else:
                replace(base_bundle, graph_views={view.name: view})


@pytest.mark.parametrize("field_name", ["user_to_node", "item_to_node"])
@pytest.mark.parametrize("length", [0, 2, 4])
def test_bundle_requires_complete_mapping_namespace(
    base_bundle: DatasetBundle, field_name: str, length: int
) -> None:
    view = _graph(**{field_name: np.full(length, -1, dtype=np.int64)})
    assert view.validate() is None  # Standalone validation has no bundle context.
    with pytest.raises(ValueError, match=f"{field_name}.*complete processed namespace"):
        replace(base_bundle, graph_views={view.name: view})


@pytest.mark.parametrize("mapping", [None, [-1, -1, -1], [2, 0, 1], [2, 2, -1]])
def test_optional_mappings_allow_partial_repeated_and_unrelated_raw_ids(
    base_bundle: DatasetBundle, mapping: list[int] | None
) -> None:
    array = None if mapping is None else np.array(mapping, dtype=np.int64)
    view = _graph(user_to_node=array, item_to_node=array)
    derived = replace(base_bundle, graph_views={view.name: view})

    assert derived.graph_views[view.name] is view
    assert view.user_to_node is array
    assert view.item_to_node is array
    assert derived.user_id_map is base_bundle.user_id_map
    assert derived.item_id_map is base_bundle.item_id_map
    # No reserved user/item node type, raw-ID join, or injectivity is required.
    assert "user" not in view.node_type_names and "item" not in view.node_type_names
    assert base_bundle.graph_views == {}


def test_mapping_lengths_use_the_corresponding_processed_namespace(
    base_bundle: DatasetBundle,
) -> None:
    base_bundle = replace(
        base_bundle,
        interactions=replace(
            base_bundle.interactions,
            item_ids=np.tile(np.array([0, 1, 0], dtype=np.int64), 3),
        ),
        item_id_map={100: 0, 200: 1},
    )
    view = _graph(
        user_to_node=np.array([3, 3, -1], dtype=np.int64),
        item_to_node=np.array([2, 2], dtype=np.int64),
    )
    # Mapping values are graph indices, even when outside processed ID ranges.
    derived = replace(base_bundle, graph_views={view.name: view})
    assert derived.graph_views[view.name] is view
    for field_name, wrong_length in (("user_to_node", 2), ("item_to_node", 3)):
        wrong = replace(view, **{field_name: np.zeros(wrong_length, dtype=np.int64)})
        with pytest.raises(ValueError, match=f"{field_name}.*complete processed namespace"):
            replace(base_bundle, graph_views={wrong.name: wrong})


def test_empty_mappings_require_empty_processed_namespaces(
    base_bundle: DatasetBundle,
) -> None:
    empty = np.empty(0, dtype=np.int64)
    empty_base = replace(
        base_bundle,
        interactions=InteractionTable(
            interaction_ids=empty,
            user_ids=empty,
            item_ids=empty,
            timestamps=empty,
            values=np.empty(0, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=empty,
            valid_interaction_ids=empty,
            test_interaction_ids=empty,
        ),
        user_id_map={},
        item_id_map={},
    )
    view = _graph(user_to_node=empty, item_to_node=empty)
    derived = replace(empty_base, graph_views={view.name: view})
    assert derived.graph_views[view.name].user_to_node is empty
    assert derived.graph_views[view.name].item_to_node is empty


@pytest.mark.parametrize(
    ("graph_views", "error_type", "message"),
    [
        ([], TypeError, "graph_views must be a dictionary"),
        ({"relations": object()}, TypeError, "GraphView objects"),
        ({1: _graph()}, TypeError, "keys must be strings"),
        ({"other": _graph()}, ValueError, "keys must match"),
    ],
)
def test_bundle_rejects_invalid_graph_registry(
    base_bundle: DatasetBundle,
    graph_views: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        replace(base_bundle, graph_views=graph_views)


def test_graph_is_keyword_only_slotted_and_mutable() -> None:
    with pytest.raises(TypeError):
        GraphView(*_graph_fields().values())
    view = _graph()
    with pytest.raises(AttributeError):
        view.edge_weights = np.ones(5)
    view.name = "renamed"
    assert view.validate() is None


def test_arrays_and_metadata_are_shared_and_invalid_inplace_edits_are_detected(
    base_bundle: DatasetBundle,
) -> None:
    user_to_node = np.array([2, 2, -1], dtype=np.int64)
    metadata = {"source": "synthetic"}
    fields = _graph_fields(
        user_to_node=user_to_node,
        item_to_node=np.array([-1, 1, 0], dtype=np.int64),
        metadata=metadata,
    )
    view = GraphView(**fields)  # type: ignore[arg-type]
    for name, supplied in fields.items():
        assert getattr(view, name) is supplied
    attached = replace(base_bundle, graph_views={view.name: view})
    metadata["note"] = "current mutable source"
    user_to_node[0] = 1
    assert view.validate() is None
    assert attached.graph_views[view.name].user_to_node[0] == 1
    assert view.metadata["note"] == "current mutable source"

    view.edge_index[0, 0] = 99
    with pytest.raises(ValueError, match="out-of-range node ID"):
        view.validate()
    with pytest.raises(ValueError, match="out-of-range node ID"):
        replace(attached)


@pytest.mark.parametrize("header_mutation", ["shape", "dtype"])
def test_array_header_mutations_require_revalidation(
    base_bundle: DatasetBundle, header_mutation: str
) -> None:
    view = _graph()
    if header_mutation == "shape":
        view.node_type_ids.shape = (2, 2)
        error_type, message = ValueError, "one-dimensional"
    else:
        view.edge_index.dtype = np.int32
        error_type, message = TypeError, "int64"
    with pytest.raises(error_type, match=message):
        view.validate()
    with pytest.raises(error_type, match=message):
        replace(base_bundle, graph_views={view.name: view})


@pytest.mark.parametrize("mode", ["per_user", "grouped_round_robin"])
def test_graph_attachment_preserves_base_features_and_sample_ownership(
    base_bundle: DatasetBundle, mode: str
) -> None:
    feature = FeatureView(
        name="item_text",
        entity_kind="item",
        entity_ids=np.array([0, 2], dtype=np.int64),
        modality="text",
        values=("first", "last"),
        feature_names=(),
        missing_mask=None,
        metadata={},
    )
    base_bundle = replace(base_bundle, feature_views={feature.name: feature})
    first = _graph(name="z_graph")
    second = _graph(name="a_graph")
    derived = replace(
        base_bundle, graph_views={first.name: first, second.name: second}
    )
    assert tuple(derived.graph_views) == ("z_graph", "a_graph")
    for name in (
        "interactions", "splits", "user_id_map", "item_id_map", "feature_views"
    ):
        assert getattr(derived, name) is getattr(base_bundle, name)
    assert base_bundle.graph_views == {}
    options = {"mode": mode, "seed": 2026}
    if mode == "grouped_round_robin":
        options["num_parties"] = 2
    before = build_hfl_partition(base_bundle, **options)
    after = build_hfl_partition(derived, **options)
    assert len(before) == len(after)
    for original, current in zip(before, after):
        assert original.party_id == current.party_id
        assert original.owns_labels == current.owns_labels
        assert original.metadata == current.metadata
        assert original.feature_view_names == current.feature_view_names
        assert original.graph_view_names == ()
        assert current.graph_view_names == ("a_graph", "z_graph")
        for name in (
            "user_ids", "item_ids", "train_interaction_ids",
            "valid_interaction_ids", "test_interaction_ids",
        ):
            np.testing.assert_array_equal(
                getattr(original, name), getattr(current, name)
            )
    # Existing coarse summaries can be overridden; C1 adds no access enforcement.
    summaries = build_hfl_partition(
        derived, **options, party_graph_views={0: ()}
    )
    assert summaries[0].graph_view_names == ()
    assert derived.graph_views[first.name] is first
