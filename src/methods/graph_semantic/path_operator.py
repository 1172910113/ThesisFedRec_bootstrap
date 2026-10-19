"""U4-K0 paths on synthetic, already canonical local binary incidence."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PathFactors:
    """Coordinate S and its row-square sum; no item-pair matrix is retained.

    Tensor fields are method-local intermediates, not an immutable delivery or
    permission boundary. The caller must keep them fixed during kernel use.
    """

    num_items: int
    local_item_ids: torch.Tensor
    sorted_item_ids: torch.Tensor
    sorted_local_rows: torch.Tensor
    item_rows: torch.Tensor
    auxiliary_columns: torch.Tensor
    weights: torch.Tensor
    h: torch.Tensor
    auxiliary_count: int
    structurally_valid: bool


def _ids(value: torch.Tensor, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or value.dtype != torch.long
        or value.ndim != 1
        or torch.any(value < 0).item()
    ):
        raise ValueError(f"{name} must be a nonnegative CPU torch.long vector")


def _floating(value: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or value.dtype != dtype
        or dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError(f"{name} must be finite CPU {dtype} with strided layout")


def prepare_path_factors(
    local_item_ids: torch.Tensor,
    incidence: torch.Tensor,
    *,
    num_items: int,
    dtype: torch.dtype = torch.float32,
) -> PathFactors:
    """Build S from (local item row, relation ID, local aux ID) triples.

    IDs are nonnegative integers. An auxiliary column is keyed by the pair
    (relation ID, aux ID); triples are deduplicated before local degrees.
    local_item_ids is an injective vector into the public catalogue, in any
    order, and can include isolated rows. This is not a GraphView adapter.
    """
    if type(num_items) is not int or num_items < 1:
        raise ValueError("num_items must be a positive integer")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    _ids(local_item_ids, "local_item_ids")
    if (
        torch.any(local_item_ids >= num_items).item()
        or torch.unique(local_item_ids).numel() != local_item_ids.numel()
    ):
        raise ValueError("local_item_ids must be one-to-one and in catalogue range")
    if (
        not isinstance(incidence, torch.Tensor)
        or incidence.layout != torch.strided
        or incidence.device.type != "cpu"
        or incidence.dtype != torch.long
        or incidence.ndim != 2
        or incidence.shape[1] != 3
        or torch.any(incidence < 0).item()
        or torch.any(incidence[:, 0] >= local_item_ids.numel()).item()
    ):
        raise ValueError("incidence must contain valid CPU long (item, relation, aux) triples")

    triples = torch.unique(incidence.detach(), dim=0, sorted=True)
    auxiliary_keys, columns = torch.unique(
        triples[:, 1:], dim=0, sorted=True, return_inverse=True
    )
    rows = triples[:, 0].clone()
    m, q = local_item_ids.numel(), auxiliary_keys.shape[0]
    item_degree = torch.bincount(rows, minlength=m).to(torch.float64)
    aux_degree = torch.bincount(columns, minlength=q).to(torch.float64)
    item_inv = torch.zeros_like(item_degree)
    aux_inv = torch.zeros_like(aux_degree)
    item_inv[item_degree > 0] = item_degree[item_degree > 0].rsqrt()
    aux_inv[aux_degree > 0] = aux_degree[aux_degree > 0].rsqrt()
    weights = (item_inv[rows] * aux_inv[columns]).to(dtype)
    # Derive h from the very same computation-dtype S used by apply_path.
    h = torch.zeros(m, dtype=dtype).index_add_(0, rows, weights.square())
    local_ids = local_item_ids.detach().clone()
    sorted_ids, sorted_rows = torch.sort(local_ids)
    return PathFactors(
        num_items=num_items,
        local_item_ids=local_ids,
        sorted_item_ids=sorted_ids,
        sorted_local_rows=sorted_rows,
        item_rows=rows,
        auxiliary_columns=columns,
        weights=weights,
        h=h,
        auxiliary_count=q,
        structurally_valid=bool(torch.any(aux_degree >= 2).item()),
    )


def apply_path(paths: PathFactors, values: torch.Tensor, *, steps: int = 1) -> torch.Tensor:
    """Apply S(S^T Y) - h Y once or twice, keeping four-hop returns."""
    if type(steps) is not int or steps not in (1, 2):
        raise ValueError("steps must be 1 or 2")
    _floating(values, "values", paths.weights.dtype)
    if values.ndim != 2 or values.shape[0] != paths.local_item_ids.numel():
        raise ValueError("values must have shape (local item count, channels)")
    result = values
    for _ in range(steps):
        aux = values.new_zeros((paths.auxiliary_count, values.shape[1]))
        aux.index_add_(
            0, paths.auxiliary_columns,
            paths.weights[:, None] * result[paths.item_rows],
        )
        propagated = torch.zeros_like(result)
        propagated.index_add_(
            0, paths.item_rows,
            paths.weights[:, None] * aux[paths.auxiliary_columns],
        )
        result = propagated - paths.h[:, None] * result
    return result
