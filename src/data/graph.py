"""Graph identities, strict access validation, and detached local projection."""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.data.contracts import (
    DatasetBundle,
    GraphAccess,
    GraphPartition,
    GraphView,
    PartyData,
    RawId,
    _validate_graph_name,
    _validate_graph_party_id,
    _validate_id_map,
)


def _json_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_identity(array: np.ndarray | None) -> dict[str, Any] | None:
    if array is None:
        return None
    canonical = np.asarray(array, dtype="<i8", order="C")
    return {
        "shape": list(canonical.shape),
        "dtype": "<i8",
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _typed_raw_id(raw_id: RawId) -> dict[str, Any]:
    # Callers validate with their own existing raw-ID rules first. JSON encodes
    # accepted str/int subclasses by value, without imposing GraphView rules on
    # DatasetBundle maps or relying on repr/stringification.
    return {"type": "int" if isinstance(raw_id, int) else "str", "value": raw_id}


def graph_view_fingerprint(view: GraphView) -> str:
    """Revalidate and hash plain-array graph content, excluding metadata."""
    if not isinstance(view, GraphView):
        raise TypeError("view must be a GraphView")
    # Subclass reductions/indexing can disagree with the underlying content
    # hashed by np.asarray (notably masked arrays). Reject them before either
    # structural validation or downstream partition endpoint/mapping checks.
    for field_name in (
        "node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node",
    ):
        array = getattr(view, field_name)
        if isinstance(array, np.ndarray) and type(array) is not np.ndarray:
            raise TypeError(
                f"GraphView {field_name} must be a plain NumPy ndarray for G1a; "
                "array subclasses (including masked arrays) are unsupported"
            )
    view.validate()
    return _json_fingerprint({
        "format": "thesisfedrec.graph_view",
        "version": 1,
        "name": view.name,
        "node_raw_ids": [_typed_raw_id(raw_id) for raw_id in view.node_raw_ids],
        "node_type_ids": _array_identity(view.node_type_ids),
        "node_type_names": list(view.node_type_names),
        "edge_index": _array_identity(view.edge_index),
        "relation_ids": _array_identity(view.relation_ids),
        "relation_names": list(view.relation_names),
        "user_to_node": _array_identity(view.user_to_node),
        "item_to_node": _array_identity(view.item_to_node),
    })


def graph_domain_fingerprint(bundle: DatasetBundle) -> str:
    """Hash processed namespaces, not interactions, splits, or semantic joins."""
    if not isinstance(bundle, DatasetBundle):
        raise TypeError("bundle must be a DatasetBundle")
    if not bundle.name:
        raise ValueError("dataset name must not be empty")
    if not bundle.schema_version:
        raise ValueError("schema_version must not be empty")
    _validate_id_map("user_id_map", bundle.user_id_map)
    _validate_id_map("item_id_map", bundle.item_id_map)
    return _json_fingerprint({
        "format": "thesisfedrec.graph_domain",
        "version": 1,
        "name": bundle.name,
        "schema_version": bundle.schema_version,
        "user_raw_ids": [
            _typed_raw_id(raw_id)
            for raw_id, _ in sorted(bundle.user_id_map.items(), key=lambda pair: pair[1])
        ],
        "item_raw_ids": [
            _typed_raw_id(raw_id)
            for raw_id, _ in sorted(bundle.item_id_map.items(), key=lambda pair: pair[1])
        ],
    })


def graph_partition_checksum(partition: GraphPartition) -> str:
    """Hash access and source bindings, excluding scenario name and metadata."""
    if not isinstance(partition, GraphPartition):
        raise TypeError("partition must be a GraphPartition")
    return _json_fingerprint({
        "format": "thesisfedrec.graph_partition",
        "version": 1,
        "source_domain_fingerprint": partition.source_domain_fingerprint,
        "source_graph_fingerprints": [
            {"view_name": name, "fingerprint": partition.source_graph_fingerprints[name]}
            for name in sorted(partition.source_graph_fingerprints)
        ],
        "parties": [
            {
                "party_id": party_id,
                "views": [
                    {
                        "view_name": access.view_name,
                        "node_ids": _array_identity(access.node_ids),
                        "edge_ids": _array_identity(access.edge_ids),
                    }
                    for access in sorted(
                        partition.party_access[party_id], key=lambda entry: entry.view_name
                    )
                ],
            }
            for party_id in sorted(partition.party_access)
        ],
    })


def validate_graph_partition(
    partition: GraphPartition,
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
) -> None:
    """Check every assignment against the current roster, graphs, and domain.

    Callers must keep mutable sources/roster stable during this call and use.
    This is neither a payload resolver nor a snapshot, leakage/privacy proof,
    or authorization boundary for code already holding the canonical bundle.
    """
    if not isinstance(partition, GraphPartition):
        raise TypeError("partition must be a GraphPartition")
    if not isinstance(bundle, DatasetBundle):
        raise TypeError("bundle must be a DatasetBundle")
    if not isinstance(parties, Sequence) or isinstance(parties, (str, bytes)):
        raise TypeError("parties must be a sequence of PartyData objects")
    parties_by_id: dict[int, PartyData] = {}
    for party in parties:
        if not isinstance(party, PartyData):
            raise TypeError("parties must contain PartyData objects")
        _validate_graph_party_id(party.party_id)
        if party.party_id in parties_by_id:
            raise ValueError("supplied parties must have unique party IDs")
        names = party.graph_view_names
        if not isinstance(names, tuple):
            raise TypeError("PartyData graph_view_names must be a tuple of strings")
        for name in names:
            _validate_graph_name("PartyData graph_view_names entry", name)
        if len(set(names)) != len(names):
            raise ValueError("PartyData graph_view_names must be unique")
        parties_by_id[party.party_id] = party

    for party_id in partition.party_access:
        if party_id not in parties_by_id:
            raise ValueError(f"GraphPartition references unknown party_id {party_id}")
    for party_id, party in parties_by_id.items():
        accesses = partition.party_access.get(party_id, ())
        if set(party.graph_view_names) != {access.view_name for access in accesses}:
            raise ValueError(
                "PartyData graph_view_names do not match GraphPartition "
                f"access for party_id {party_id}"
            )

    if graph_domain_fingerprint(bundle) != partition.source_domain_fingerprint:
        raise ValueError("stale source_domain_fingerprint")
    if not isinstance(bundle.graph_views, dict):
        raise TypeError("graph_views must be a dictionary")
    for name, expected in sorted(partition.source_graph_fingerprints.items()):
        if name not in bundle.graph_views:
            raise ValueError(f"GraphPartition references unknown GraphView {name!r}")
        view = bundle.graph_views[name]
        if not isinstance(view, GraphView):
            raise TypeError("referenced graph_views must be GraphView objects")
        if name != view.name:
            raise ValueError("graph_views keys must match their GraphView names")
        actual = graph_view_fingerprint(view)  # Revalidates current source structure.
        for field_name, size in (
            ("user_to_node", len(bundle.user_id_map)),
            ("item_to_node", len(bundle.item_id_map)),
        ):
            mapping = getattr(view, field_name)
            if mapping is not None and len(mapping) != size:
                raise ValueError(f"{field_name} must cover the complete processed namespace")
        if actual != expected:
            raise ValueError(f"stale source graph fingerprint for {name!r}")

    for accesses in partition.party_access.values():
        for access in accesses:
            view = bundle.graph_views[access.view_name]
            nodes, edges = access.node_ids, access.edge_ids
            if nodes is not None and nodes.size and nodes[-1] >= len(view.node_raw_ids):
                raise ValueError(f"GraphAccess {access.view_name!r} has out-of-range nodes")
            if edges is not None and edges.size and edges[-1] >= view.edge_index.shape[1]:
                raise ValueError(f"GraphAccess {access.view_name!r} has out-of-range edges")
            if nodes is not None:
                endpoints = view.edge_index if edges is None else view.edge_index[:, edges]
                if not np.all(np.isin(endpoints, nodes)):
                    raise ValueError(
                        f"GraphAccess {access.view_name!r} has endpoints outside selected nodes"
                    )


def _validate_graph_delivery(view: GraphView) -> None:
    # G1a has already checked current plain-array structure and source binding.
    # Apply delivery restrictions even to empty or other parties' selections.
    if any(type(name) is not str for name in (
        view.name, *view.node_type_names, *view.relation_names,
    )):
        raise TypeError("graph delivery view/vocabulary names must be plain built-in strings")
    for field_name in (
        "node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node",
    ):
        array = getattr(view, field_name)
        if array is not None and array.dtype.metadata is not None:
            raise TypeError(f"GraphView {view.name!r} {field_name} has unsupported dtype metadata")


def _project_graph_mapping(
    mapping: np.ndarray | None, local_ids: np.ndarray,
) -> np.ndarray | None:
    if mapping is None:
        return None
    projected = np.full(mapping.shape, -1, dtype=np.int64)
    mapped = mapping >= 0
    projected[mapped] = local_ids[mapping[mapped]]
    return projected


def _project_graph_view(view: GraphView, access: GraphAccess) -> GraphView:
    nodes, edges = access.node_ids, access.edge_ids
    if nodes is None:
        nodes = np.arange(len(view.node_raw_ids), dtype=np.int64)
    edge_selection = slice(None) if edges is None else edges
    # Temporary canonical-to-local lookup, never exposed or retained. All
    # selected endpoints have already been authorized by strict validation.
    local_ids = np.full(len(view.node_raw_ids), -1, dtype=np.int64)
    local_ids[nodes] = np.arange(len(nodes), dtype=np.int64)
    return GraphView(
        name=view.name,
        node_raw_ids=tuple(f"n{index}" for index in range(len(nodes))),
        node_type_ids=np.array(view.node_type_ids[nodes], dtype=np.int64, copy=True),
        node_type_names=tuple(name for name in view.node_type_names),
        edge_index=np.array(
            local_ids[view.edge_index[:, edge_selection]], dtype=np.int64, copy=True,
            order="C",
        ),
        relation_ids=np.array(view.relation_ids[edge_selection], dtype=np.int64, copy=True),
        relation_names=tuple(name for name in view.relation_names),
        user_to_node=_project_graph_mapping(view.user_to_node, local_ids),
        item_to_node=_project_graph_mapping(view.item_to_node, local_ids),
        metadata={},
    )


def resolve_party_graph_views(
    bundle: DatasetBundle,
    partition: GraphPartition,
    parties: Sequence[PartyData],
    *,
    party_id: int,
) -> dict[str, GraphView]:
    """Return detached, locally relabeled projections for one explicit party.

    Validate the complete partition and all bound sources on every call, even
    for an empty result. Delivery rejects dtype metadata and non-built-in
    public names on every referenced source. Arrays own their storage; metadata
    is empty. Public vocabularies and full recommendation-anchor domains remain
    visible, so local node labels do not provide cryptographic anonymity.

    Sources/roster must stay stable during the call. No cache, atomic snapshot,
    concurrent-write guarantee, or training integration is provided. Validation
    scans all bound sources/domains; projection additionally allocates a lookup
    per requested graph and complete non-None user/item mapping outputs.
    """
    _validate_graph_party_id(party_id)
    validate_graph_partition(partition, bundle, parties)
    if not any(party.party_id == party_id for party in parties):
        raise ValueError(f"unknown requested party_id {party_id}")
    for name in sorted(partition.source_graph_fingerprints):
        _validate_graph_delivery(bundle.graph_views[name])
    resolved: dict[str, GraphView] = {}
    for access in sorted(
        partition.party_access.get(party_id, ()), key=lambda entry: entry.view_name,
    ):
        view = _project_graph_view(bundle.graph_views[access.view_name], access)
        resolved[view.name] = view
    return resolved
