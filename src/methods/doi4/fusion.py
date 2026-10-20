"""DOI4 center-fusion numerical kernel for caller-authorized token sets."""

from __future__ import annotations

from math import sqrt

import torch


def center_fusion(
    compatible_tokens: torch.Tensor,
    valid_token_mask: torch.Tensor,
    item_embeddings: torch.Tensor,
    w_query: torch.Tensor,
    w_key: torch.Tensor,
    w_value: torch.Tensor,
    w_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fused [E,h], Boolean availability [E], attention [E,L].

    Inputs are tokens [E,L,h], a Boolean valid mask [E,L], items [E,d],
    w_query [h,d] and key/value/output maps [h,h]. L is an already packed
    authorized source-slot set; the caller owns packing and permissions.
    For available entities, query-to-set attention computes
    q = item_embeddings @ w_query.T, keys/values = tokens @ their_map.T,
    A = softmax_valid(dot(q, keys) / sqrt(h)), and
    fused = sum_l(A_l * values_l) @ w_output.T.

    E/L may be zero; d/h must be positive. Dense strided CPU float32/float64
    inputs of one dtype and a Boolean CPU mask are supported. All values,
    including masked entries, must be finite. Nonfinite arithmetic raises
    ValueError. Inputs are never coerced, mutated, cloned or detached; CPU
    autocast is locally disabled to preserve their dtype.

    Masked payloads are removed before projection. Absent/empty entities
    return exact zeros without empty/all-masked softmax, retaining connected
    zero derivatives to all floating inputs. Singleton query/key derivatives
    are connected zeros. Caller-owned non-leaf gradient paths are preserved.
    No parameters, initialization, RNG, residual, bias, normalization,
    dropout, source self-attention, routing or reuse mechanism is added.
    """
    _validate_fusion_inputs(
        compatible_tokens, valid_token_mask, item_embeddings,
        w_query, w_key, w_value, w_output,
    )
    available = valid_token_mask.any(dim=1)
    with torch.autocast(device_type="cpu", enabled=False):
        tokens = compatible_tokens.masked_fill(~valid_token_mask.unsqueeze(-1), 0)
        query = item_embeddings @ w_query.T
        keys = tokens @ w_key.T
        values = tokens @ w_value.T
        for name, tensor in (("query", query), ("keys", keys), ("values", values)):
            _require_finite(name, tensor)
        logits = (keys @ query.unsqueeze(-1)).squeeze(-1) / sqrt(tokens.shape[2])
        _require_finite("logits", logits)

        # This zero branch preserves autograd even for E/L=0 or no valid rows.
        attention = logits * 0
        active_indices = available.nonzero(as_tuple=True)[0]
        if active_indices.numel():
            active_logits = logits[active_indices].masked_fill(
                ~valid_token_mask[active_indices], -torch.inf
            )
            active_attention = torch.softmax(active_logits, dim=-1)
            attention = attention.index_copy(0, active_indices, active_attention)
        pooled = (attention.unsqueeze(1) @ values).squeeze(1)
        _require_finite("pooled", pooled)
        fused = pooled @ w_output.T
        _require_finite("fused", fused)
    return fused, available, attention


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite (check inputs/arithmetic overflow)")


def _validate_fusion_inputs(
    compatible_tokens: torch.Tensor,
    valid_token_mask: torch.Tensor,
    item_embeddings: torch.Tensor,
    w_query: torch.Tensor,
    w_key: torch.Tensor,
    w_value: torch.Tensor,
    w_output: torch.Tensor,
) -> None:
    floating_inputs = (
        ("compatible_tokens", compatible_tokens, 3),
        ("item_embeddings", item_embeddings, 2),
        ("w_query", w_query, 2),
        ("w_key", w_key, 2),
        ("w_value", w_value, 2),
        ("w_output", w_output, 2),
    )
    for name, tensor, rank in (*floating_inputs, ("valid_token_mask", valid_token_mask, 2)):
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
        if tensor.dtype != compatible_tokens.dtype:
            raise TypeError("floating inputs must have the same dtype")
        _require_finite(name, tensor)
    if valid_token_mask.dtype != torch.bool:
        raise TypeError("valid_token_mask must have Boolean dtype")

    entities, tokens, width = compatible_tokens.shape
    local_dim = item_embeddings.shape[1]
    if min(width, local_dim) <= 0:
        raise ValueError("h and d must be positive")
    if valid_token_mask.shape != (entities, tokens):
        raise ValueError("valid_token_mask must have shape [E,L]")
    if item_embeddings.shape[0] != entities:
        raise ValueError("item_embeddings must have shape [E,d]")
    if w_query.shape != (width, local_dim):
        raise ValueError("w_query must have shape [h,d]")
    for name, tensor in (("w_key", w_key), ("w_value", w_value), ("w_output", w_output)):
        if tensor.shape != (width, width):
            raise ValueError(f"{name} must have shape [h,h]")
