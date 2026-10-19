"""DOI4 numerical readout kernels for already encoded local units."""

from __future__ import annotations

from math import sqrt

import torch


def compatible_readout(
    encoded_tokens: torch.Tensor,
    valid_unit_mask: torch.Tensor,
    queries: torch.Tensor,
    w_key: torch.Tensor,
    w_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compatible tokens [E,K,h], availability [E], attention [E,K,P].

    Inputs are encoded_tokens [E,P,d], a Boolean valid_unit_mask [E,P],
    caller-owned queries [K,h] and key/value maps [h,d]. True means valid.
    E/P may be zero; d/h/K must be positive. Floating inputs must be finite,
    dense strided CPU float32/float64 tensors of one dtype. The mask must be
    a dense strided CPU Boolean tensor. No inputs are coerced or modified.

    For valid p, A_ekp = softmax_p(B_k dot (W_key X_ep) / sqrt(h)) and
    H_ek = sum_p A_ekp (W_value X_ep). Masked weights are exactly zero.
    Absent entities (including P=0) return zero tokens/weights without an
    empty/all-masked softmax. Singleton attention remains connected to
    autograd, yielding the formula's zero query/key derivatives.

    No parameters, initialization, random operations, residual, bias,
    normalization or dropout are added. CPU autocast is locally disabled to
    preserve the supplied dtype. Nonfinite inputs (even masked entries) and
    nonfinite arithmetic intermediates/results raise ValueError; overflow
    is not repaired by clipping or dtype promotion. GPU/other dtypes are
    outside this narrow numerical boundary.
    """
    _validate_readout_inputs(
        encoded_tokens, valid_unit_mask, queries, w_key, w_value
    )
    available = valid_unit_mask.any(dim=1)
    with torch.autocast(device_type="cpu", enabled=False):
        # Remove masked payloads before projections, so even large finite
        # missing values cannot affect outputs, gradients or arithmetic range.
        tokens = encoded_tokens.masked_fill(~valid_unit_mask.unsqueeze(-1), 0)
        keys = tokens @ w_key.T
        values = tokens @ w_value.T
        logits = (queries @ keys.transpose(1, 2)) / sqrt(queries.shape[1])
        for name, tensor in (("keys", keys), ("values", values), ("logits", logits)):
            _require_finite(name, tensor)

        # Keeping the zero branch connected also gives defined zero gradients
        # for absent/empty inputs, without an artificial query residual.
        attention = logits * 0
        active_indices = available.nonzero(as_tuple=True)[0]
        if active_indices.numel():
            active_logits = logits[active_indices].masked_fill(
                ~valid_unit_mask[active_indices].unsqueeze(1), -torch.inf
            )
            active_attention = torch.softmax(active_logits, dim=-1)
            attention = attention.index_copy(0, active_indices, active_attention)
        compatible_tokens = attention @ values
        _require_finite("compatible_tokens", compatible_tokens)
    return compatible_tokens, available, attention


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite (check inputs/arithmetic overflow)")


def _validate_readout_inputs(
    encoded_tokens: torch.Tensor,
    valid_unit_mask: torch.Tensor,
    queries: torch.Tensor,
    w_key: torch.Tensor,
    w_value: torch.Tensor,
) -> None:
    floating_inputs = (
        ("encoded_tokens", encoded_tokens, 3),
        ("queries", queries, 2),
        ("w_key", w_key, 2),
        ("w_value", w_value, 2),
    )
    for name, tensor, rank in (*floating_inputs, ("valid_unit_mask", valid_unit_mask, 2)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.is_nested or tensor.layout != torch.strided:
            raise TypeError(f"{name} must be a dense strided tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if tensor.ndim != rank:
            raise ValueError(f"{name} must have rank {rank}")
    for name, tensor, _ in floating_inputs:
        if tensor.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have dtype float32 or float64")
        if tensor.dtype != encoded_tokens.dtype:
            raise TypeError("floating inputs must have the same dtype")
        _require_finite(name, tensor)
    if valid_unit_mask.dtype != torch.bool:
        raise TypeError("valid_unit_mask must have Boolean dtype")

    entities, units, local_dim = encoded_tokens.shape
    slots, compatible_dim = queries.shape
    if min(local_dim, slots, compatible_dim) <= 0:
        raise ValueError("d, K and h must be positive")
    if valid_unit_mask.shape != (entities, units):
        raise ValueError("valid_unit_mask must have shape [E,P]")
    for name, tensor in (("w_key", w_key), ("w_value", w_value)):
        if tensor.shape != (compatible_dim, local_dim):
            raise ValueError(f"{name} must have shape [h,d]")
