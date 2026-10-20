"""Explicit local GraphView schema adaptation to the existing U4 path factory.

The caller supplies an already-authorized local view and keeps it stable for
this call. Local structural validation cannot establish that provenance.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from src.data.contracts import GraphView
from .path_operator import PathFactors, prepare_path_factors


def _id(value: int, name: str, upper: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value < 0 or (upper is not None and value >= upper):
        raise ValueError(f"{name} is out of range")


def _vocabulary(names: tuple[str, ...], name: str) -> None:
    if type(names) is not tuple or any(type(value) is not str for value in names):
        raise TypeError(f"{name} must be a tuple of plain strings")
    if any(not value.strip() for value in names) or len(set(names)) != len(names):
        raise ValueError(f"{name} must contain distinct nonblank names")


def _id_tuple(values: tuple[int, ...], name: str, upper: int) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    for value in values:
        _id(value, name, upper)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique IDs")


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphRelationRule:
    relation_id: int
    canonical_relation_id: int
    direction: Literal["item_to_aux", "aux_to_item"]

    def __post_init__(self) -> None:
        _validate_rule(self)


def _validate_rule(rule: GraphRelationRule) -> None:
    if type(rule) is not GraphRelationRule:
        raise TypeError("relation_rules must contain exact GraphRelationRule records")
    _id(rule.relation_id, "relation_id")
    _id(rule.canonical_relation_id, "canonical_relation_id", 2**63)
    if type(rule.direction) is not str or rule.direction not in ("item_to_aux", "aux_to_item"):
        raise ValueError("direction must be item_to_aux or aux_to_item")


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphOperatorSchema:
    node_type_names: tuple[str, ...]
    relation_names: tuple[str, ...]
    item_node_type_id: int
    aux_node_type_ids: tuple[int, ...]
    relation_rules: tuple[GraphRelationRule, ...]
    ignored_relation_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_schema(self)


def _validate_schema(schema: GraphOperatorSchema) -> None:
    if type(schema) is not GraphOperatorSchema:
        raise TypeError("schema must be an exact GraphOperatorSchema")
    _vocabulary(schema.node_type_names, "node_type_names")
    _vocabulary(schema.relation_names, "relation_names")
    _id(schema.item_node_type_id, "item_node_type_id", len(schema.node_type_names))
    _id_tuple(schema.aux_node_type_ids, "aux_node_type_ids", len(schema.node_type_names))
    if not schema.aux_node_type_ids or schema.item_node_type_id in schema.aux_node_type_ids:
        raise ValueError("auxiliary types must be nonempty and distinct from item type")
    _id_tuple(schema.ignored_relation_ids, "ignored_relation_ids", len(schema.relation_names))
    if type(schema.relation_rules) is not tuple:
        raise TypeError("relation_rules must be a tuple")
    selected = set()
    for rule in schema.relation_rules:
        _validate_rule(rule)
        _id(rule.relation_id, "relation_id", len(schema.relation_names))
        if rule.relation_id in selected:
            raise ValueError("duplicate relation rule")
        selected.add(rule.relation_id)
    ignored = set(schema.ignored_relation_ids)
    if selected & ignored or selected | ignored != set(range(len(schema.relation_names))):
        raise ValueError("rules and ignored IDs must disjointly cover every declared relation")


def _validate_local_view(view: GraphView, schema: GraphOperatorSchema) -> None:
    if type(view) is not GraphView:
        raise TypeError("view must be an exact GraphView or None")
    for name in ("node_type_ids", "edge_index", "relation_ids", "user_to_node", "item_to_node"):
        values = getattr(view, name)
        if values is None and name in ("user_to_node", "item_to_node"):
            continue
        if type(values) is not np.ndarray or values.dtype != np.dtype(np.int64):
            raise TypeError(f"{name} must be a plain NumPy int64 array")
        if values.dtype.metadata is not None:
            raise TypeError(f"{name} must not have dtype metadata")
    _vocabulary(view.node_type_names, "view.node_type_names")
    _vocabulary(view.relation_names, "view.relation_names")
    view.validate()
    if view.node_type_names != schema.node_type_names or view.relation_names != schema.relation_names:
        raise ValueError("view vocabularies must match schema exactly in order")


def _anchors(view: GraphView, schema: GraphOperatorSchema, num_items: int) -> tuple[np.ndarray, np.ndarray]:
    mapping = view.item_to_node
    if mapping is None or mapping.shape != (num_items,):
        raise ValueError("item_to_node is required with exact length num_items")
    public_ids = np.flatnonzero(mapping >= 0)
    nodes = mapping[public_ids]
    if np.unique(nodes).size != nodes.size:
        raise ValueError("item_to_node mapped anchors must be injective")
    if np.any(view.node_type_ids[nodes] != schema.item_node_type_id):
        raise ValueError("mapped anchors must land on the declared item node type")
    if nodes.size != np.count_nonzero(view.node_type_ids == schema.item_node_type_id):
        raise ValueError("every item-type node must have one public anchor")
    return public_ids, nodes


def _canonical_incidence(
    view: GraphView, schema: GraphOperatorSchema, nodes: np.ndarray,
) -> np.ndarray:
    order = np.argsort(nodes)
    sorted_nodes = nodes[order]
    parts = []
    for rule in schema.relation_rules:
        positions = np.flatnonzero(view.relation_ids == rule.relation_id)
        if not positions.size:
            continue
        endpoints = view.edge_index[:, positions]
        item_nodes, aux_nodes = endpoints if rule.direction == "item_to_aux" else endpoints[::-1]
        if (
            np.any(view.node_type_ids[item_nodes] != schema.item_node_type_id)
            or not np.isin(view.node_type_ids[aux_nodes], schema.aux_node_type_ids).all()
        ):
            raise ValueError("selected edge endpoints do not match explicit direction/types")
        # Every item-type node was checked to have one anchor; order maps it
        # back to its compact row in sorted PUBLIC item order.
        local_rows = order[np.searchsorted(sorted_nodes, item_nodes)]
        parts.append(np.column_stack((
            local_rows,
            np.full(positions.size, rule.canonical_relation_id, dtype=np.int64),
            aux_nodes,
        )))
    if not parts:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.concatenate(parts, axis=0), axis=0)


def prepare_path_factors_from_graph(
    view: GraphView | None, schema: GraphOperatorSchema, *,
    num_items: int, dtype: torch.dtype,
) -> PathFactors:
    """Validate a local view, canonicalize/deduplicate, call the existing factory.

    Mapping reads cost O(num_items) at preparation. Factor outputs retain only
    owned numeric buffers, never GraphView, raw IDs, metadata or source arrays.
    None denotes absence explicitly; it does not perform an access lookup.
    """
    _id(num_items, "num_items")
    if num_items == 0:
        raise ValueError("num_items must be positive")
    if dtype is not torch.float32 and dtype is not torch.float64:
        raise ValueError("dtype must be exactly torch.float32 or torch.float64")
    _validate_schema(schema)
    if view is None:
        public_ids = np.empty(0, dtype=np.int64)
        incidence = np.empty((0, 3), dtype=np.int64)
    else:
        _validate_local_view(view, schema)
        public_ids, nodes = _anchors(view, schema, num_items)
        incidence = _canonical_incidence(view, schema, nodes)
    return prepare_path_factors(
        torch.from_numpy(public_ids), torch.from_numpy(incidence),
        num_items=num_items, dtype=dtype,
    )
