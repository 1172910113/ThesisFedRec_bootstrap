"""Public genre-field grants over an explicitly supplied party-ID domain.

This recipe varies availability of public metadata. It neither infers graph
permissions from sample ownership nor establishes privacy or graph utility.
Callers keep the consumed graph, maps and party records stable during a call.
"""

from dataclasses import replace

import numpy as np

from src.data.contracts import DatasetBundle, GraphAccess, GraphPartition, GraphView, PartyData
from src.data.graph import (
    graph_domain_fingerprint, graph_view_fingerprint, validate_graph_partition,
)


_GRAPH_NAME = "item_genres_graph"
_GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
_MAX_ID = 2**63 - 1


def _capacity(item_count: int, edge_count: int) -> None:
    if item_count < 0 or edge_count < 0 or item_count + 18 > _MAX_ID or edge_count > _MAX_ID:
        raise ValueError("canonical graph exceeds signed int64 capacity")


def _catalogue(bundle: DatasetBundle) -> tuple[int, ...]:
    if type(bundle) is not DatasetBundle:
        raise TypeError("bundle must be an exact DatasetBundle")
    mapping = bundle.item_id_map
    if type(mapping) is not dict:
        raise TypeError("item_id_map must be a plain dictionary")
    for raw_id, processed_id in mapping.items():
        if type(raw_id) is not int or type(processed_id) is not int:
            raise TypeError("MovieID and processed item IDs must be exact built-in integers")
        if raw_id <= 0:
            raise ValueError("MovieID must be positive")
    _capacity(len(mapping), 0)
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("processed item IDs must be a bijection onto the catalogue")
    return tuple(raw for raw, _ in sorted(mapping.items(), key=lambda pair: pair[1]))


def _int_array(value: np.ndarray, name: str) -> None:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a plain NumPy array")
    if value.dtype != np.dtype("int64") or not value.dtype.isnative or value.dtype.metadata is not None:
        raise TypeError(f"{name} must have native metadata-free int64 dtype")


def _names(value: tuple[str, ...], name: str) -> None:
    if type(value) is not tuple or any(type(entry) is not str for entry in value):
        raise TypeError(f"{name} must be an exact tuple of built-in strings")


def _fingerprint(value: str) -> None:
    if type(value) is not str:
        raise TypeError("expected_graph_fingerprint must be a built-in string")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("expected_graph_fingerprint must be lowercase SHA-256 hex")


def _graph(bundle: DatasetBundle, raw_items: tuple[int, ...], expected: str) -> GraphView:
    if type(bundle.graph_views) is not dict:
        raise TypeError("graph_views must be a dictionary")
    if _GRAPH_NAME not in bundle.graph_views:
        raise ValueError("the canonical item genre graph must already be attached")
    view = bundle.graph_views[_GRAPH_NAME]
    if type(view) is not GraphView:
        raise TypeError("the canonical graph must be an exact GraphView")
    if type(view.name) is not str:
        raise TypeError("graph name must be a built-in string")
    if type(view.node_raw_ids) is not tuple or any(
        type(raw_id) not in (int, str) for raw_id in view.node_raw_ids
    ):
        raise TypeError("node_raw_ids must be an exact tuple of built-in integers/strings")
    _names(view.node_type_names, "node_type_names")
    _names(view.relation_names, "relation_names")
    for name in ("node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node"):
        array = getattr(view, name)
        if array is not None:
            _int_array(array, name)
    for name in ("node_type_ids", "edge_index", "relation_ids"):
        if getattr(view, name) is None:
            raise TypeError(f"{name} must be a plain NumPy array")

    # The public function revalidates structure and rejects duplicates/bounds.
    actual = graph_view_fingerprint(view)
    count = len(raw_items)
    _capacity(count, view.edge_index.shape[1])
    if view.name != _GRAPH_NAME or view.node_type_names != ("item", "genre"):
        raise ValueError("unsupported canonical item-genre node schema")
    if view.relation_names != ("has_genre",) or np.any(view.relation_ids != 0):
        raise ValueError("unsupported canonical item-genre relation schema")
    if view.node_raw_ids != raw_items + _GENRES:
        raise ValueError("canonical node rows must match the current item domain and fixed genres")
    if np.any(view.node_type_ids[:count] != 0) or np.any(view.node_type_ids[count:] != 1):
        raise ValueError("canonical item and genre node types do not match their rows")
    if view.user_to_node is not None:
        raise ValueError("canonical genre graph must not have user anchors")
    if view.item_to_node is None or view.item_to_node.shape != (count,) or not np.array_equal(
        view.item_to_node, np.arange(count, dtype=np.int64),
    ):
        raise ValueError("item_to_node must be the complete identity mapping")
    sources, destinations = view.edge_index
    if np.any(sources >= count) or np.any(destinations < count) or np.any(destinations >= count + 18):
        raise ValueError("only forward item-to-genre edges are supported")
    if np.unique(sources).size != count:
        raise ValueError("every catalogue item must have at least one genre edge")
    if sources.size > 1 and not np.all(
        (sources[1:] > sources[:-1])
        | ((sources[1:] == sources[:-1]) & (destinations[1:] > destinations[:-1]))
    ):
        raise ValueError("canonical edges must be ordered by item then genre column")
    if actual != expected:
        raise ValueError("stale expected graph fingerprint")
    return view


def _parties(parties: tuple[PartyData, ...], public_ids: tuple[int, ...]) -> None:
    if type(public_ids) is not tuple:
        raise TypeError("public_party_ids must be an exact tuple")
    for party_id in public_ids:
        if type(party_id) is not int:
            raise TypeError("public party IDs must be exact built-in integers")
        if not 0 <= party_id <= _MAX_ID:
            raise ValueError("public party IDs must be in the signed int64 range")
    if any(left >= right for left, right in zip(public_ids, public_ids[1:])):
        raise ValueError("public_party_ids must be sorted and unique")
    if type(parties) is not tuple:
        raise TypeError("parties must be an exact tuple")
    seen = set()
    for party in parties:
        if type(party) is not PartyData:
            raise TypeError("parties must contain exact PartyData records")
        if type(party.party_id) is not int:
            raise TypeError("party_id must be an exact built-in integer")
        if not 0 <= party.party_id <= _MAX_ID:
            raise ValueError("party_id must be in the signed int64 range")
        if party.party_id in seen:
            raise ValueError("party IDs must be unique")
        seen.add(party.party_id)
        if not isinstance(party.metadata, dict):
            raise TypeError("party metadata must be a dictionary")
        _names(party.graph_view_names, "party graph_view_names")
        if party.graph_view_names not in ((), (_GRAPH_NAME,)):
            raise ValueError("party graph summaries may name only the canonical genre graph")
    if seen != set(public_ids):
        raise ValueError("parties must match exactly the declared full public party domain")


def build_public_genre_block_partition(
    bundle: DatasetBundle,
    parties: tuple[PartyData, ...],
    *,
    graph_name: str,
    expected_graph_fingerprint: str,
    public_party_ids: tuple[int, ...],
) -> tuple[GraphPartition, tuple[PartyData, ...]]:
    """Grant six lexical genre columns by full supplied ``party_id % 3``.

    The canonical graph is already attached by the caller. Expected content is
    checked on every call, including empty rosters. Graph metadata and sample
    content never determine grants. New party records share sample arrays and
    nested metadata; only their metadata dictionaries and graph summaries are
    replaced. The explicit full-domain packet is a caller assertion, not proof
    of population completeness. Returned immutable descriptors are suitable for
    the existing GraphRuntime; projected item mappings still cost O(catalogue).
    """
    raw_items = _catalogue(bundle)
    if type(graph_name) is not str:
        raise TypeError("graph_name must be a built-in string")
    if graph_name != _GRAPH_NAME:
        raise ValueError("only item_genres_graph is supported")
    _fingerprint(expected_graph_fingerprint)
    _parties(parties, public_party_ids)
    view = _graph(bundle, raw_items, expected_graph_fingerprint)
    domain = graph_domain_fingerprint(bundle)
    count = len(raw_items)
    by_group = {}
    for group in sorted({party_id % 3 for party_id in public_party_ids}):
        edge_ids = np.flatnonzero((view.edge_index[1] - count) % 3 == group).astype(np.int64)
        item_nodes = np.unique(view.edge_index[0, edge_ids])
        genre_nodes = count + np.arange(group, 18, 3, dtype=np.int64)
        node_ids = np.concatenate((item_nodes, genre_nodes)).astype(np.int64)
        by_group[group] = GraphAccess(view_name=graph_name, node_ids=node_ids, edge_ids=edge_ids)
    new_parties = tuple(
        replace(party, graph_view_names=(graph_name,), metadata=dict(party.metadata))
        for party in parties
    )
    partition = GraphPartition(
        name="ml1m_public_genre_blocks3_catalogue_domain.r1",
        party_access={party_id: (by_group[party_id % 3],) for party_id in public_party_ids},
        source_graph_fingerprints={graph_name: expected_graph_fingerprint} if public_party_ids else {},
        source_domain_fingerprint=domain,
        metadata={
            "recipe_id": "ml1m.public_genre_blocks3.v1", "group_count": 3,
            "assignment": "party_id_mod3", "full_public_party_ids": list(public_party_ids),
            "party_groups": [{"party_id": party_id, "group_id": party_id % 3} for party_id in public_party_ids],
        },
    )
    validate_graph_partition(partition, bundle, new_parties)
    return partition, new_parties
