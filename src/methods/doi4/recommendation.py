"""Pure DOI4 local GMF-form scoring and caller-packed positive-group loss."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def local_recommendation_logits(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    fused_features: torch.Tensor,
    private_features: torch.Tensor,
    w_fused: torch.Tensor,
    w_private: torch.Tensor,
    score_weight: torch.Tensor,
    score_bias: torch.Tensor,
    *,
    fused_available: torch.Tensor,
    private_available: torch.Tensor,
) -> torch.Tensor:
    """Return raw logits [E] for caller-owned representations and parameters.

    Floating shapes are [E,d], [E,d], [E,h], [E,g], [d,h], [d,g], [d],
    and scalar []. Availability masks are Boolean [E]; True means available.
    Unavailable feature rows are zeroed BEFORE projection. With these masked
    features z/q, v_tilde = items + z @ w_fused.T + q @ w_private.T, then
    logits = sum(users * v_tilde * score_weight, dim=1) + score_bias.

    Dense strided CPU float32/64 inputs of one dtype, finite even when masked,
    are supported. E may be zero; d/h/g must be positive. Empty/absent paths
    retain connected zero derivatives. Non-contiguous/non-leaf caller paths
    are preserved without mutation, coercion, clone/detach or initialization.
    CPU autocast is locally disabled; nonfinite arithmetic raises ValueError.
    No sigmoid, encoder, private pooling, source routing or execution is added.
    """
    floats = (
        ("user_embeddings", user_embeddings, 2),
        ("item_embeddings", item_embeddings, 2),
        ("fused_features", fused_features, 2),
        ("private_features", private_features, 2),
        ("w_fused", w_fused, 2),
        ("w_private", w_private, 2),
        ("score_weight", score_weight, 1),
        ("score_bias", score_bias, 0),
    )
    _validate_floats(floats)
    entities, width = user_embeddings.shape
    fused_width, private_width = fused_features.shape[1], private_features.shape[1]
    if min(width, fused_width, private_width) <= 0:
        raise ValueError("d, h and g must be positive")
    expected = (
        ("item_embeddings", item_embeddings, (entities, width)),
        ("fused_features", fused_features, (entities, fused_width)),
        ("private_features", private_features, (entities, private_width)),
        ("w_fused", w_fused, (width, fused_width)),
        ("w_private", w_private, (width, private_width)),
        ("score_weight", score_weight, (width,)),
    )
    for name, tensor, shape in expected:
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name, mask in (("fused_available", fused_available), ("private_available", private_available)):
        _validate_tensor(name, mask, 1)
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must have Boolean dtype")
        if mask.shape != (entities,):
            raise ValueError(f"{name} must have shape [E]")

    with torch.autocast(device_type="cpu", enabled=False):
        fused = fused_features.masked_fill(~fused_available[:, None], 0) @ w_fused.T
        private = private_features.masked_fill(~private_available[:, None], 0) @ w_private.T
        _require_finite("fused projection", fused)
        _require_finite("private projection", private)
        enhanced = item_embeddings + fused
        _require_finite("item plus fused residual", enhanced)
        enhanced = enhanced + private
        _require_finite("enhanced items", enhanced)
        interactions = user_embeddings * enhanced
        _require_finite("interactions", interactions)
        weighted = interactions * score_weight
        _require_finite("weighted interactions", weighted)
        scores = weighted.sum(dim=1)
        _require_finite("summed scores", scores)
        logits = scores + score_bias
        _require_finite("logits", logits)
    return logits


def positive_group_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_offsets: torch.Tensor,
) -> torch.Tensor:
    """Mean stable BCE within each positive group, then equally across groups.

    logits/labels are same-dtype finite CPU float32/64 [N]. Binary labels
    are caller-packed positive first, then zero or more negatives per group.
    CPU dense strided int64 offsets [M+1] start at 0, end at N and strictly
    increase; M>=1 and N>0. Inputs are neither sampled nor reordered.

    Preserves logits' differentiable/non-leaf path, with no input mutation,
    dtype promotion or RNG. Labels/offsets are data; label gradients are not
    part of the contract. CPU autocast is locally disabled; nonfinite loss
    arithmetic raises ValueError. Future m_c,t/N_r scaling belongs ONCE to
    the caller; no client/source weights, renormalization or updates live here.
    """
    _validate_floats((("logits", logits, 1), ("labels", labels, 1)))
    _validate_tensor("group_offsets", group_offsets, 1)
    if group_offsets.dtype != torch.int64:
        raise TypeError("group_offsets must have int64 dtype")
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same shape [N]")
    if not logits.numel() or group_offsets.numel() < 2:
        raise ValueError("loss requires nonempty examples and at least one group")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must contain exact binary 0/1 values")
    if group_offsets[0].item() != 0 or group_offsets[-1].item() != logits.numel():
        raise ValueError("group_offsets must start at 0 and end at N")
    if not bool((group_offsets[1:] > group_offsets[:-1]).all()):
        raise ValueError("group_offsets must strictly increase (nonempty groups)")
    starts = group_offsets[:-1]
    if not bool((labels[starts] == 1).all()) or torch.count_nonzero(labels).item() != starts.numel():
        raise ValueError("each group must contain a first positive and remaining negatives")
    boundaries = group_offsets.tolist()
    with torch.autocast(device_type="cpu", enabled=False):
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        _require_finite("example losses", losses)
        group_losses = torch.stack([
            losses[start:end].mean()
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ])
        _require_finite("group losses", group_losses)
        loss = group_losses.mean()
        _require_finite("loss", loss)
    return loss


def _validate_tensor(name: str, tensor: torch.Tensor, rank: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.is_nested or tensor.layout != torch.strided:
        raise TypeError(f"{name} must be a dense strided tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")


def _validate_floats(inputs: tuple[tuple[str, torch.Tensor, int], ...]) -> None:
    for name, tensor, rank in inputs:
        _validate_tensor(name, tensor, rank)
    for name, tensor, _ in inputs:
        if tensor.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have dtype float32 or float64")
        if tensor.dtype != inputs[0][1].dtype:
            raise TypeError("floating inputs must have the same dtype")
        _require_finite(name, tensor)


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite (check inputs/arithmetic overflow)")
