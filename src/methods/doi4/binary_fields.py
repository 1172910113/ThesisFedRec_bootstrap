"""Pure additive encoding of caller-provided local binary-field units."""

from __future__ import annotations

import torch


def encode_binary_field_units(
    local_field_ids: torch.Tensor,
    binary_values: torch.Tensor,
    valid_unit_mask: torch.Tensor,
    field_embeddings: torch.Tensor,
    value_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Return new encoded tokens [E,P,r], with exact zeros at masked units.

    IDs/codes are dense strided CPU int64 [E,P]; the mask is bool [E,P].
    Caller-owned tables F[J,r] and V[2,r] are finite CPU float32/64 of one
    dtype, with J>=0 and r>=1. E/P may be zero. Codes are 0/1 everywhere.
    IDs are in [0,J) or masked padding -1; valid padding is rejected.
    J=0 therefore requires all-masked units with IDs=-1.

    Only valid cells gather/add X_ep = F[id_ep] + V[code_ep]. Empty gathers
    preserve connected zero gradients to both tables even for J/E/P=0 or
    all-masked batches, without summing large unused tables. Repeated IDs
    accumulate gradients. Non-leaf/strided caller paths are preserved.

    No coercion, clone/detach, mutation, initialization, RNG, hidden parameter
    or ownership state is introduced. CPU autocast is locally disabled;
    nonfinite active addition raises ValueError without clipping/promotion.
    Type/dtype/device/layout errors raise TypeError; rank/shape/range and
    nonfinite errors raise ValueError, including for empty inputs.

    Whole-table finiteness validation costs O((J+2)r), integer/mask checks
    cost O(EP), and output materialization costs O(EPr); gathering/addition
    costs O(N_valid*r). This is not an O(active units) complete-call claim.
    The same original mask must be supplied separately to compatible_readout;
    this function neither returns a mask nor infers availability from values.
    """
    _validate_inputs(
        local_field_ids, binary_values, valid_unit_mask,
        field_embeddings, value_embeddings,
    )
    with torch.autocast(device_type="cpu", enabled=False):
        # Empty gathers remain differentiable to BOTH caller tables. Padding
        # never reaches indexing, and unused finite rows never enter addition.
        active = (
            field_embeddings[local_field_ids[valid_unit_mask]]
            + value_embeddings[binary_values[valid_unit_mask]]
        )
        if not bool(torch.isfinite(active).all()):
            raise ValueError("active field/value addition must be finite")
        shape = (*local_field_ids.shape, field_embeddings.shape[1])
        output = field_embeddings.new_zeros(shape)
        return output.masked_scatter(valid_unit_mask.unsqueeze(-1), active)


def _validate_inputs(
    local_field_ids: torch.Tensor,
    binary_values: torch.Tensor,
    valid_unit_mask: torch.Tensor,
    field_embeddings: torch.Tensor,
    value_embeddings: torch.Tensor,
) -> None:
    inputs = (
        ("local_field_ids", local_field_ids),
        ("binary_values", binary_values),
        ("valid_unit_mask", valid_unit_mask),
        ("field_embeddings", field_embeddings),
        ("value_embeddings", value_embeddings),
    )
    for name, tensor in inputs:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.is_nested or tensor.layout != torch.strided:
            raise TypeError(f"{name} must be a dense strided tensor")
        if tensor.device.type != "cpu":
            raise TypeError(f"{name} must be on CPU")
    for name, tensor in inputs[:2]:
        if tensor.dtype != torch.int64:
            raise TypeError(f"{name} must have int64 dtype")
    if valid_unit_mask.dtype != torch.bool:
        raise TypeError("valid_unit_mask must have Boolean dtype")
    for name, tensor in inputs[3:]:
        if tensor.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have float32 or float64 dtype")
    if field_embeddings.dtype != value_embeddings.dtype:
        raise TypeError("embedding tables must have the same floating dtype")
    for name, tensor in inputs:
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have rank 2")
    if binary_values.shape != local_field_ids.shape or valid_unit_mask.shape != local_field_ids.shape:
        raise ValueError("IDs, binary values and mask must have the same shape [E,P]")
    fields, width = field_embeddings.shape
    if width < 1:
        raise ValueError("embedding width r must be positive")
    if value_embeddings.shape != (2, width):
        raise ValueError("value_embeddings must have shape [2,r]")

    if bool(((local_field_ids < -1) | (local_field_ids >= fields)).any()):
        raise ValueError("field IDs must be -1 padding or in [0,J), even when masked")
    if bool((local_field_ids[valid_unit_mask] < 0).any()):
        raise ValueError("valid units must have field IDs in [0,J); padding must be masked")
    if not bool(((binary_values == 0) | (binary_values == 1)).all()):
        raise ValueError("binary values must be 0/1, even when masked")
    for name, tensor in inputs[3:]:
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite, including unused rows")
