"""Independent tiny canonical bytes/graph oracle; no raw data or method calls."""

import copy
import gc
import hashlib
import json
import struct
import weakref

from types import MappingProxyType

import numpy as np
import pytest

from src.data.contracts import DatasetBundle, FeatureView, InteractionSplits, InteractionTable
from src.data.graph import graph_view_fingerprint
from src.data.ml1m_graph import (
    build_ml1m_item_genre_graph,
    ml1m_item_genre_feature_binding,
)
from src.data import ml1m_graph


GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
CELLS = ((0, 0), (0, 1), (1, 0), (1, 2), (2, 4), (2, 5), (3, 17))
EDGES = ((0, 4), (0, 5), (1, 4), (1, 6), (2, 8), (2, 9), (3, 21))
FIXED_TEXT = {
    "builder": "ml1m_item_genres", "source_encoding": "latin-1",
    "source_format": "MovieID::Title::Genres",
    "vocabulary_source": "MovieLens-1M README: MOVIES FILE DESCRIPTION",
    "vocabulary_policy": "fixed", "encoding": "multi_hot",
    "row_order": "processed_item_id", "column_order": "lexical_genre_name",
    "raw_movie_id_policy": "positive_ascii_decimal",
    "genre_token_policy": "exact_known_unique_nonempty",
    "missing_metadata_policy": "error", "duplicate_movie_id_policy": "error",
    "metadata_only_policy": "validate_and_exclude",
}
COUNT_KEYS = (
    "source_records", "unique_metadata_items", "processed_items", "matched_items",
    "missing_items", "metadata_only_items", "genre_count", "positive_entries",
)
SOURCE_KEYS = (*FIXED_TEXT, "builder_version", "source_relative_path",
               "source_size_bytes", "source_sha256", "genre_vocabulary",
               "item_mapping_sha256", "counts")


class IntSubclass(int):
    pass


class StrSubclass(str):
    pass


class DictSubclass(dict):
    pass


class ArraySubclass(np.ndarray):
    pass


class FeatureSubclass(FeatureView):
    pass


class BundleSubclass(DatasetBundle):
    pass


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _mapping_hash(mapping):
    lines = [json.dumps({"raw_id": raw, "processed_id": row},
                        ensure_ascii=False, sort_keys=True) + "\n"
             for raw, row in sorted(mapping.items(), key=lambda pair: pair[1])]
    return _sha("".join(lines).encode("utf-8"))


def _bundle(empty=False, cls=DatasetBundle):
    # Frozen support row3 is part of the declared fixture from the start.
    size = 0 if empty else 4
    return cls(
        name="ml1m", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=np.arange(size, dtype=np.int64),
            user_ids=np.array([] if empty else [0, 1, 2, 0], dtype=np.int64),
            item_ids=np.arange(size, dtype=np.int64),
            timestamps=np.arange(size, dtype=np.int64),
            values=np.ones(size, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=np.array([] if empty else [0, 1], dtype=np.int64),
            valid_interaction_ids=np.array([] if empty else [2], dtype=np.int64),
            test_interaction_ids=np.array([] if empty else [3], dtype=np.int64),
        ),
        user_id_map={} if empty else {100: 0, 200: 1, 300: 2},
        item_id_map={} if empty else {10: 0, 20: 1, 30: 2, 40: 3}, metadata={},
    )


def _view(bundle, cls=FeatureView):
    n = len(bundle.item_id_map)
    values = np.zeros((n, 18), dtype=np.float32)
    if n:
        for i, j in CELLS:
            values[i, j] = 1
    return cls(
        name="item_genres", entity_kind="item", modality="tabular",
        entity_ids=np.arange(n, dtype=np.int64), values=values,
        feature_names=GENRES, missing_mask=np.zeros((n, 18), dtype=bool),
        metadata={
            **FIXED_TEXT, "builder_version": 1, "source_relative_path": "synthetic/movies.dat",
            "source_size_bytes": 113, "source_sha256": "a" * 64,
            "genre_vocabulary": list(GENRES),
            "item_mapping_sha256": _mapping_hash(bundle.item_id_map),
            "counts": {
                "source_records": n + 1, "unique_metadata_items": n + 1,
                "processed_items": n, "matched_items": n, "missing_items": 0,
                "metadata_only_items": 1, "genre_count": 18,
                "positive_entries": 7 if n else 0,
            },
        },
    )


def _oracle(bundle, view):
    # Explicit published envelope and scalar struct encoder, independent of
    # production graph/hash helpers and NumPy's production byte-casting route.
    def typed(raw):
        return {"type": "int" if isinstance(raw, int) else "str", "value": raw}
    domain = {
        "format": "thesisfedrec.graph_domain", "version": 1,
        "name": bundle.name, "schema_version": bundle.schema_version,
        "user_raw_ids": [typed(raw) for raw, _ in sorted(bundle.user_id_map.items(), key=lambda x: x[1])],
        "item_raw_ids": [typed(raw) for raw, _ in sorted(bundle.item_id_map.items(), key=lambda x: x[1])],
    }
    ids = b"".join(struct.pack("<q", int(v)) for v in view.entity_ids)
    floats = b"".join(struct.pack("<f", float(v)) for row in view.values for v in row)
    mask = None if view.missing_mask is None else {
        "shape": list(view.missing_mask.shape), "dtype": "bool",
        "sha256": _sha(bytes(int(v) for row in view.missing_mask for v in row)),
    }
    return _sha(_canonical({
        "format": "thesisfedrec.ml1m_item_genre_feature_binding", "version": 1,
        "domain_fingerprint": _sha(_canonical(domain)),
        "item_mapping_sha256": _mapping_hash(bundle.item_id_map),
        "view": {
            "name": view.name, "entity_kind": view.entity_kind, "modality": view.modality,
            "feature_names": list(view.feature_names),
            "entity_ids": {"shape": list(view.entity_ids.shape), "dtype": "int64", "sha256": _sha(ids)},
            "values": {"shape": list(view.values.shape), "dtype": "float32", "sha256": _sha(floats)},
            "missing_mask": mask,
        },
        "source_metadata": {key: copy.deepcopy(view.metadata[key]) for key in SOURCE_KEYS},
    }))


@pytest.fixture
def pair():
    bundle = _bundle()
    return bundle, _view(bundle)


_UNSET = object()


def _build(bundle, view, expected=_UNSET):
    return build_ml1m_item_genre_graph(
        bundle, genre_view=view,
        expected_feature_binding=_oracle(bundle, view) if expected is _UNSET else expected,
    )


def test_independent_envelope_and_literal_graph(pair):
    bundle, view = pair
    expected = _oracle(bundle, view)
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == expected
    graph = _build(bundle, view, expected)
    assert graph.name == "item_genres_graph"
    assert graph.node_raw_ids == (10, 20, 30, 40, *GENRES)
    assert graph.node_type_names == ("item", "genre")
    assert graph.relation_names == ("has_genre",)
    assert graph.user_to_node is None
    np.testing.assert_array_equal(graph.node_type_ids, [0] * 4 + [1] * 18)
    np.testing.assert_array_equal(graph.edge_index, np.array(EDGES, dtype=np.int64).T)
    np.testing.assert_array_equal(graph.relation_ids, [0] * 7)
    np.testing.assert_array_equal(graph.item_to_node, [0, 1, 2, 3])
    assert set(graph.metadata) == {
        "builder", "builder_version", "schema_revision", "source_feature_binding",
        "source_domain_fingerprint", "source_item_mapping_sha256", "source_sha256", "counts",
    }
    assert graph.metadata["source_feature_binding"] == expected
    assert graph.metadata["source_item_mapping_sha256"] == _mapping_hash(bundle.item_id_map)
    assert graph.metadata["source_sha256"] == "a" * 64
    assert graph.metadata["counts"] == {"item_count": 4, "genre_count": 18, "edge_count": 7}
    assert bundle.feature_views == {} and bundle.graph_views == {}
    print("ML1M_GRAPH_ORACLE", json.dumps({"expected_binding": expected, "actual_binding": ml1m_item_genre_feature_binding(bundle, genre_view=view), "literal_edges": EDGES}))


def test_owning_mutable_outputs_and_source_isolation(pair):
    bundle, view = pair
    before = (view.values.copy(), view.entity_ids.copy(), view.missing_mask.copy(), copy.deepcopy(view.metadata))
    first, second = _build(bundle, view), _build(bundle, view)
    names = ("node_type_ids", "edge_index", "relation_ids", "item_to_node")
    for name in names:
        array = getattr(first, name)
        assert type(array) is np.ndarray and array.dtype == np.dtype(np.int64)
        assert array.dtype.isnative and array.dtype.metadata is None
        assert array.flags.owndata and array.flags.c_contiguous and array.base is None
        assert not np.shares_memory(array, getattr(second, name))
        for source in (view.values, view.entity_ids, view.missing_mask):
            assert not np.shares_memory(array, source)
    first.edge_index[0, 0] = 3
    first.metadata["counts"]["edge_count"] = 99
    np.testing.assert_array_equal(second.edge_index, np.array(EDGES).T)
    assert second.metadata["counts"]["edge_count"] == 7
    view.values[:] = 0
    view.metadata["genre_vocabulary"][:] = []
    np.testing.assert_array_equal(second.edge_index, np.array(EDGES).T)
    assert second.node_raw_ids[-18:] == GENRES
    np.testing.assert_array_equal(before[1], view.entity_ids)
    np.testing.assert_array_equal(before[2], view.missing_mask)


def test_ignored_metadata_not_traversed_or_retained(pair):
    class Unvisited:
        def __iter__(self):
            raise AssertionError("unknown metadata traversed")
        def __deepcopy__(self, memo):
            raise AssertionError("unknown metadata copied")
        def __repr__(self):
            raise AssertionError("unknown metadata represented")
    bundle, view = pair
    expected = _oracle(bundle, view)
    sentinel = Unvisited()
    reference = weakref.ref(sentinel)
    view.metadata["GT"] = sentinel
    view.metadata["arbitrary"] = {"not_json": object()}
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == expected
    graph = _build(bundle, view, expected)
    assert "GT" not in graph.metadata and "arbitrary" not in graph.metadata
    del view.metadata["GT"], sentinel
    gc.collect()
    assert reference() is None


def test_strides_readonly_and_insertion_order_preserve_binding(pair):
    bundle, view = pair
    expected = _oracle(bundle, view)
    values = np.zeros((4, 36), dtype=np.float32)
    values[:, ::2] = view.values
    view.values = values[:, ::2]
    ids = np.zeros(8, dtype=np.int64)
    ids[::2] = view.entity_ids
    view.entity_ids = ids[::2]
    view.missing_mask = np.zeros((4, 36), dtype=bool)[:, ::2]
    for array in (view.values, view.entity_ids, view.missing_mask):
        array.setflags(write=False)
    bundle.item_id_map = dict(reversed(list(bundle.item_id_map.items())))
    bundle.user_id_map = dict(reversed(list(bundle.user_id_map.items())))
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == expected
    np.testing.assert_array_equal(_build(bundle, view).edge_index, np.array(EDGES).T)


def test_none_and_false_masks_bind_differently_same_graph(pair):
    bundle, view = pair
    first_binding = _oracle(bundle, view)
    first = _build(bundle, view, first_binding)
    view.missing_mask = None
    second_binding = _oracle(bundle, view)
    assert second_binding != first_binding
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == second_binding
    second = _build(bundle, view, second_binding)
    assert graph_view_fingerprint(first) == graph_view_fingerprint(second)
    assert first.metadata["source_feature_binding"] != second.metadata["source_feature_binding"]


def test_mapping_and_user_domain_change_bindings(pair):
    bundle, view = pair
    old = _oracle(bundle, view)
    bundle.item_id_map = {10: 1, 20: 0, 30: 2, 40: 3}
    view.values = view.values[[1, 0, 2, 3]].copy()
    view.metadata["item_mapping_sha256"] = _mapping_hash(bundle.item_id_map)
    remapped = _oracle(bundle, view)
    assert remapped != old
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == remapped
    assert _build(bundle, view).node_raw_ids[:4] == (20, 10, 30, 40)
    bundle.user_id_map = {999: 0, 200: 1, 300: 2}
    changed_domain = _oracle(bundle, view)
    assert changed_domain != remapped
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == changed_domain
    with pytest.raises(ValueError, match="binding"):
        _build(bundle, view, remapped)


def test_raw_movie_id_has_no_int64_label_bound(pair):
    bundle, view = pair
    bundle.item_id_map = {2**70: 0, 20: 1, 30: 2, 40: 3}
    view.metadata["item_mapping_sha256"] = _mapping_hash(bundle.item_id_map)
    assert _build(bundle, view).node_raw_ids[0] == 2**70


def test_signed_zero_bytes_are_bound(pair):
    bundle, view = pair
    old = _oracle(bundle, view)
    view.values[0, 17] = np.float32(-0.0)
    current = _oracle(bundle, view)
    assert current != old
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == current
    np.testing.assert_array_equal(_build(bundle, view).edge_index, np.array(EDGES).T)


def test_empty_catalogue_has_18_aux_and_nonempty_asserted_source():
    bundle = _bundle(empty=True)
    view = _view(bundle)
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == _oracle(bundle, view)
    graph = _build(bundle, view)
    assert graph.node_raw_ids == GENRES
    assert graph.edge_index.shape == (2, 0) and graph.relation_ids.shape == (0,)
    assert graph.item_to_node.shape == (0,)
    for array in (graph.edge_index, graph.relation_ids, graph.item_to_node):
        assert array.flags.owndata and array.base is None
    view.metadata["counts"].update(source_records=0, unique_metadata_items=0, metadata_only_items=0)
    with pytest.raises(ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("field", ("name", "entity_kind", "modality"))
@pytest.mark.parametrize("bad", (None, 1, StrSubclass("item")))
def test_view_text_element_types(pair, field, bad):
    bundle, view = pair
    setattr(view, field, bad)
    with pytest.raises(TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("field", ("name", "entity_kind", "modality"))
def test_view_text_values(pair, field):
    bundle, view = pair
    setattr(view, field, "unsupported")
    with pytest.raises(ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("field", ("entity_ids", "values", "missing_mask"))
@pytest.mark.parametrize("kind", ("list", "subclass", "masked", "metadata", "wrong_dtype", "structured", "object"))
def test_array_representation_rejection(pair, field, kind):
    bundle, view = pair
    array = getattr(view, field)
    if kind == "list":
        bad = array.tolist()
    elif kind == "subclass":
        bad = array.view(ArraySubclass)
    elif kind == "masked":
        bad = np.ma.array(array, mask=False)
    elif kind == "metadata":
        bad = array.view(np.dtype(array.dtype, metadata={}))
        assert bad.dtype.metadata == {}
    elif kind == "wrong_dtype":
        bad = array.astype(np.float64 if field == "values" else np.float32)
    elif kind == "structured":
        bad = np.zeros(array.shape, dtype=[("x", "i8")])
    else:
        bad = array.astype(object)
    setattr(view, field, bad)
    with pytest.raises(TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("field", ("entity_ids", "values"))
def test_non_native_arrays_rejected(pair, field):
    bundle, view = pair
    array = getattr(view, field)
    setattr(view, field, array.astype(array.dtype.newbyteorder("S")))
    with pytest.raises(TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf, -1.0, 0.5, 2.0))
def test_nonbinary_nonfinite_values(pair, bad):
    bundle, view = pair
    view.values[0, 17] = bad
    with pytest.raises(ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("field", ("entity_ids", "values", "missing_mask"))
def test_wrong_shapes(pair, field):
    bundle, view = pair
    setattr(view, field, getattr(view, field).reshape(-1)[:3])
    with pytest.raises(ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("ids", ([0, 1, 1, 3], [1, 0, 2, 3], [-1, 1, 2, 3], [0, 1, 2, 4]))
def test_catalogue_ids_exact(pair, ids):
    bundle, view = pair
    view.entity_ids = np.array(ids, dtype=np.int64)
    with pytest.raises(ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


def test_zero_genre_row_and_source_missingness_rejected(pair):
    bundle, view = pair
    view.values[3] = 0
    view.metadata["counts"]["positive_entries"] = 6
    with pytest.raises(ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)
    view.values[3, 17] = 1
    view.metadata["counts"]["positive_entries"] = 7
    view.missing_mask[0, 0] = True
    with pytest.raises(ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("field", ("feature_names", "genre_vocabulary"))
@pytest.mark.parametrize("kind", ("wrong_container", "subclass_member", "wrong_order", "missing", "wrong_name"))
def test_fixed_vocabulary(pair, field, kind):
    bundle, view = pair
    values = list(GENRES)
    if kind == "wrong_container":
        bad = values if field == "feature_names" else tuple(values)
    else:
        if kind == "subclass_member": values[0] = StrSubclass(values[0])
        if kind == "wrong_order": values.reverse()
        if kind == "missing": values.pop()
        if kind == "wrong_name": values[0] = "Unknown"
        bad = tuple(values) if field == "feature_names" else values
    if field == "feature_names": view.feature_names = bad
    else: view.metadata[field] = bad
    with pytest.raises(TypeError if kind in ("wrong_container", "subclass_member") else ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("field", tuple(FIXED_TEXT))
@pytest.mark.parametrize("kind", ("type", "value"))
def test_every_fixed_metadata_string(pair, field, kind):
    bundle, view = pair
    view.metadata[field] = StrSubclass(FIXED_TEXT[field]) if kind == "type" else "wrong"
    with pytest.raises(TypeError if kind == "type" else ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("key", SOURCE_KEYS)
def test_each_required_metadata_key_missing(pair, key):
    bundle, view = pair
    del view.metadata[key]
    with pytest.raises(ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("field", ("source_sha256", "item_mapping_sha256"))
@pytest.mark.parametrize("bad", (None, StrSubclass("a" * 64), "A" * 64, "a" * 63, "g" * 64))
def test_metadata_digest_types_values(pair, field, bad):
    bundle, view = pair
    view.metadata[field] = bad
    with pytest.raises(TypeError if bad is None or type(bad) is StrSubclass else ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("bad", ("", " movies.dat", "movies.dat ", "a\nb", "a\x00b", "a\u0085b", 1, StrSubclass("movies.dat")))
def test_source_path_is_only_strict_identity_text(pair, bad):
    bundle, view = pair
    view.metadata["source_relative_path"] = bad
    with pytest.raises(TypeError if type(bad) is not str else ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


def test_source_assertions_can_change_binding_without_file_reads(pair):
    bundle, view = pair
    old = _oracle(bundle, view)
    view.metadata["source_relative_path"] = "/not/opened/../source"
    view.metadata["source_sha256"] = "b" * 64
    view.metadata["source_size_bytes"] = 2**63 - 1
    current = _oracle(bundle, view)
    assert current != old
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == current
    assert _build(bundle, view).metadata["source_sha256"] == "b" * 64


@pytest.mark.parametrize("key", COUNT_KEYS)
@pytest.mark.parametrize("bad", (True, np.int64(1), IntSubclass(1), 1.0, -1, 2**63))
def test_exact_count_types_and_ranges(pair, key, bad):
    bundle, view = pair
    view.metadata["counts"][key] = bad
    with pytest.raises(ValueError if type(bad) is int else TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("key", COUNT_KEYS)
def test_each_count_consistency(pair, key):
    bundle, view = pair
    view.metadata["counts"][key] += 1
    with pytest.raises(ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("field", ("builder_version", "source_size_bytes"))
@pytest.mark.parametrize("bad", (True, np.int64(1), IntSubclass(1), "1", 0, 2**63))
def test_metadata_scalar_types_ranges(pair, field, bad):
    bundle, view = pair
    view.metadata[field] = bad
    with pytest.raises(ValueError if type(bad) is int else TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("field", ("metadata", "counts"))
@pytest.mark.parametrize("kind", ("proxy", "subclass", "list"))
def test_plain_metadata_containers(pair, field, kind):
    bundle, view = pair
    old = view.metadata if field == "metadata" else view.metadata["counts"]
    bad = MappingProxyType(old) if kind == "proxy" else DictSubclass(old) if kind == "subclass" else list(old.items())
    if field == "metadata": view.metadata = bad
    else: view.metadata["counts"] = bad
    with pytest.raises(TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("kind", ("extra", "missing", "wrong_key_type"))
def test_counts_exact_key_set(pair, kind):
    bundle, view = pair
    counts = view.metadata["counts"]
    if kind == "extra": counts["extra"] = 0
    elif kind == "missing": del counts["positive_entries"]
    else:
        value = counts.pop("positive_entries")
        counts[StrSubclass("positive_entries")] = value
    with pytest.raises(TypeError if kind == "wrong_key_type" else ValueError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("bad", (True, np.int64(10), IntSubclass(10), "10", 0, -1))
def test_raw_item_mapping_types_range(pair, bad):
    bundle, view = pair
    bundle.item_id_map = {bad: 0, 20: 1, 30: 2, 40: 3}
    with pytest.raises(ValueError if type(bad) is int else TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


@pytest.mark.parametrize("bad", (False, np.int64(0), IntSubclass(0), "0", -1, 1))
def test_processed_mapping_exact_bijection(pair, bad):
    bundle, view = pair
    bundle.item_id_map = {10: bad, 20: 1, 30: 2, 40: 3}
    with pytest.raises(ValueError if type(bad) is int else TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


def test_plain_item_mapping_container(pair):
    bundle, view = pair
    bundle.item_id_map = DictSubclass(bundle.item_id_map)
    with pytest.raises(TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=view)


def test_exact_source_records():
    bundle = _bundle()
    with pytest.raises(TypeError):
        ml1m_item_genre_feature_binding(object(), genre_view=_view(bundle))
    with pytest.raises(TypeError):
        ml1m_item_genre_feature_binding(_bundle(cls=BundleSubclass), genre_view=_view(bundle))
    with pytest.raises(TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=object())
    with pytest.raises(TypeError):
        ml1m_item_genre_feature_binding(bundle, genre_view=_view(bundle, cls=FeatureSubclass))


@pytest.mark.parametrize("bad", (None, StrSubclass("a" * 64), "A" * 64, "a" * 63, "g" * 64, "0" * 64))
def test_expected_binding_before_graph_output(pair, monkeypatch, bad):
    bundle, view = pair
    def forbidden(**kwargs):
        raise AssertionError("GraphView constructed before expected-binding rejection")
    monkeypatch.setattr(ml1m_graph, "GraphView", forbidden)
    with pytest.raises(TypeError if bad is None or type(bad) is StrSubclass else ValueError):
        _build(bundle, view, bad)


def test_current_source_revalidation_and_failure_nonmutation(pair):
    bundle, view = pair
    expected = _oracle(bundle, view)
    graph = _build(bundle, view, expected)
    view.values[0, 0] = 0
    view.metadata["counts"]["positive_entries"] = 6
    before = (view.values.tobytes(), copy.deepcopy(view.metadata), graph.edge_index.tobytes())
    with pytest.raises(ValueError, match="binding"):
        _build(bundle, view, expected)
    assert before == (view.values.tobytes(), view.metadata, graph.edge_index.tobytes())
    current = _oracle(bundle, view)
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == current != expected


def test_capacity_checks_without_large_allocations():
    maximum = 2**63 - 1
    ml1m_graph._capacity(maximum - 18, maximum)
    with pytest.raises(ValueError): ml1m_graph._capacity(maximum - 17, 0)
    with pytest.raises(ValueError): ml1m_graph._capacity(0, maximum + 1)


def test_no_interaction_split_feature_payload_consumed(pair):
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError("unrelated sample/feature payload consumed")
    bundle, view = pair
    expected = _oracle(bundle, view)
    bundle.interactions = bundle.splits = Forbidden()
    bundle.feature_views = {"GT": Forbidden()}
    assert ml1m_item_genre_feature_binding(bundle, genre_view=view) == expected
    np.testing.assert_array_equal(_build(bundle, view, expected).edge_index, np.array(EDGES).T)
