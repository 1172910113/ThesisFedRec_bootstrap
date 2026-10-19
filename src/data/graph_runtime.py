"""Trusted, in-memory graph snapshots; no method or persistence integration."""

from collections.abc import Sequence
from types import MappingProxyType

from src.data.contracts import (
    DatasetBundle, GraphAccess, GraphPartition, GraphView, PartyData,
    _validate_graph_party_id,
)
from src.data.graph import (
    _build_graph_projection_index,
    _immutable_int_array,
    _json_fingerprint,
    _project_graph_view,
    _validate_graph_delivery,
    _validate_graph_delivery_names,
    _validate_graph_sources,
    graph_domain_fingerprint,
    graph_partition_checksum,
    validate_graph_partition,
)


class GraphRuntime:
    """Opaque coordinator handle, created only by prepare_graph_runtime.

    The sole public data property is an immutable binding ID. Private storage
    is not a supported interface. This is not hostile-Python isolation.
    """

    __slots__ = ("_binding_id", "_views", "_indexes", "_access_by_party")

    def __init__(self) -> None:
        raise TypeError("use prepare_graph_runtime to create a GraphRuntime")

    @property
    def binding_id(self) -> str:
        """Trusted-local identity; reading a retired handle is an error."""
        _require_open(self)
        return self._binding_id


def _require_open(runtime: GraphRuntime) -> None:
    if not isinstance(runtime, GraphRuntime):
        raise TypeError("runtime must be a GraphRuntime")
    if runtime._binding_id is None:
        raise ValueError("GraphRuntime is closed")


def check_graph_runtime_binding(runtime: GraphRuntime, *, binding_id: str) -> None:
    """Check an expected binding without rescanning sources or caller metadata."""
    _require_open(runtime)
    if type(binding_id) is not str:
        raise TypeError("binding_id must be a plain built-in string")
    if binding_id != runtime._binding_id:
        raise ValueError("GraphRuntime binding mismatch")


def _copy_graph(view: GraphView) -> GraphView:
    def array(value):
        return None if value is None else _immutable_int_array(value)

    return GraphView(
        name=view.name, node_raw_ids=tuple(raw for raw in view.node_raw_ids),
        node_type_ids=array(view.node_type_ids),
        node_type_names=tuple(name for name in view.node_type_names),
        edge_index=array(view.edge_index), relation_ids=array(view.relation_ids),
        relation_names=tuple(name for name in view.relation_names),
        user_to_node=array(view.user_to_node), item_to_node=array(view.item_to_node),
        metadata={},
    )


def prepare_graph_runtime(
    bundle: DatasetBundle,
    partition: GraphPartition,
    parties: Sequence[PartyData],
    *,
    source_id: str,
    split_id: str,
    recipe_id: str,
) -> GraphRuntime:
    """Validate, own and index referenced data with stable caller-held inputs.

    source_id/split_id/recipe_id are required nonblank built-in strings supplied
    by the trusted coordinator, NOT verified provenance. Content fingerprints,
    partition checksum and roster are verified. No interaction/split scan is
    performed. Keep inputs stable throughout this call using coordination or
    an external lock. Later changes require a new handle and explicit close.
    """
    for name, value in (("source_id", source_id), ("split_id", split_id),
                        ("recipe_id", recipe_id)):
        if type(value) is not str:
            raise TypeError(f"{name} must be a plain built-in string")
        if not value.strip():
            raise ValueError(f"{name} must be nonblank")

    # Original-type restrictions must precede copies: otherwise ndarray/name
    # subclasses or dtype metadata could be silently sanitized.
    _validate_graph_delivery_names(partition)
    validate_graph_partition(partition, bundle, parties)
    for name in partition.source_graph_fingerprints:
        _validate_graph_delivery(bundle.graph_views[name])

    # Domain raw IDs are needed only for validation/hash, not delivery. Retain
    # their verified digest and cardinalities, never the mutable namespace maps.
    domain_id = graph_domain_fingerprint(bundle)
    user_count, item_count = len(bundle.user_id_map), len(bundle.item_id_map)
    views = {
        name: _copy_graph(bundle.graph_views[name])
        for name in sorted(partition.source_graph_fingerprints)
    }
    owned_partition = GraphPartition(
        name="runtime", metadata={}, source_domain_fingerprint=domain_id,
        source_graph_fingerprints={
            name: str.__str__(digest)
            for name, digest in partition.source_graph_fingerprints.items()
        },
        party_access={
            party_id: tuple(
                GraphAccess(view_name=entry.view_name, node_ids=entry.node_ids,
                            edge_ids=entry.edge_ids)
                for entry in sorted(entries, key=lambda entry: entry.view_name)
            )
            for party_id, entries in partition.party_access.items()
        },
    )
    # All copied structure, anchors, content bindings and endpoints are checked
    # again on OWNED data. The original roster was validated above; only its
    # numeric IDs and matched canonical names are relevant to this runtime.
    _validate_graph_sources(owned_partition, views, domain_id, user_count, item_count)
    access_by_party = {
        party.party_id: owned_partition.party_access.get(party.party_id, ())
        for party in parties
    }
    roster = [
        {"party_id": party_id, "view_names": [entry.view_name for entry in entries]}
        for party_id, entries in sorted(access_by_party.items())
    ]
    binding_id = _json_fingerprint({
        "format": "thesisfedrec.graph_runtime", "version": 1,
        "delivery_rules": "strict_g1b_v1",
        "source_id": source_id, "split_id": split_id, "recipe_id": recipe_id,
        "domain_fingerprint": domain_id,
        "graph_fingerprints": dict(owned_partition.source_graph_fingerprints),
        "partition_checksum": graph_partition_checksum(owned_partition),
        "roster": roster,
    })
    indexes = {name: _build_graph_projection_index(view) for name, view in views.items()}
    # Publish no usable handle until all validation and index construction pass.
    runtime = object.__new__(GraphRuntime)
    runtime._binding_id = binding_id
    runtime._views = MappingProxyType(views)
    runtime._indexes = MappingProxyType(indexes)
    runtime._access_by_party = MappingProxyType(access_by_party)
    return runtime


def project_party_graphs(
    runtime: GraphRuntime, *, party_id: int, binding_id: str,
) -> dict[str, GraphView]:
    """Deliver detached graphs for one party using an explicitly expected ID.

    No source hashes, global validation, unrelated-view/party scans, or index
    construction occur here. GraphView validates selected output structure.
    Full processed-domain mapping outputs still cost O(users + items) memory.
    """
    check_graph_runtime_binding(runtime, binding_id=binding_id)
    _validate_graph_party_id(party_id)
    if party_id not in runtime._access_by_party:
        raise ValueError(f"unknown requested party_id {party_id}")
    return {
        access.view_name: _project_graph_view(
            runtime._views[access.view_name], access, runtime._indexes[access.view_name],
        )
        for access in runtime._access_by_party[party_id]
    }


def close_graph_runtime(runtime: GraphRuntime) -> None:
    """Idempotently retire a handle and release its owned references."""
    if not isinstance(runtime, GraphRuntime):
        raise TypeError("runtime must be a GraphRuntime")
    runtime._binding_id = None
    runtime._views = None
    runtime._indexes = None
    runtime._access_by_party = None
