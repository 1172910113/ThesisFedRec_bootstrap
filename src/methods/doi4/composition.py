"""Pure tensor composition of the accepted DOI4 kernels, with pri-none only."""

from __future__ import annotations

import torch

from .binary_fields import encode_binary_field_units
from .encoders import compatible_readout
from .fusion import center_fusion
from .recommendation import local_recommendation_logits, positive_group_bce


def encode_readout_binary_units(
    local_field_ids: torch.Tensor,
    binary_values: torch.Tensor,
    valid_unit_mask: torch.Tensor,
    *,
    field_embeddings: torch.Tensor,
    value_embeddings: torch.Tensor,
    queries: torch.Tensor,
    w_key: torch.Tensor,
    w_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode F[J,r]+V[2,r], then read out H[E,K,h] with the original mask.

    Returns (H, Boolean availability[E], attention[E,K,P]). The accepted
    kernels validate CPU tensor types, shapes, finite inputs/arithmetic and
    preserve empty, strided and non-leaf caller paths. Widths r and h may
    differ. Observed zero is valid; neither codes nor token norms imply a
    mask. The caller owns all parameters, local IDs and source permissions.
    """
    encoded = encode_binary_field_units(
        local_field_ids, binary_values, valid_unit_mask,
        field_embeddings, value_embeddings,
    )
    return compatible_readout(encoded, valid_unit_mask, queries, w_key, w_value)


def fuse_score_pri_none(
    compatible_tokens: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    w_query: torch.Tensor,
    w_key: torch.Tensor,
    w_value: torch.Tensor,
    w_output: torch.Tensor,
    w_fused: torch.Tensor,
    score_weight: torch.Tensor,
    score_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits[E], fused[E,h], availability[E] and attention[E,L].

    Tokens/mask are an already authorized packed set. The SAME item tensor
    enters both center F's query and the direct score path. No routing,
    repacking, source selection, detach, coercion or input mutation occurs.
    E/L=0 and all-masked paths retain the kernels' connected zero gradients.

    Pri-none uses temporary ordinary non-grad zeros of private width one,
    allocated only after scoring-input validation. They are not parameters,
    returned state or a private model. All trainable state remains caller
    owned. No RNG, backward, optimizer, schedule or graph lifetime is owned
    by this function; the caller may apply standard autograd to its results.
    """
    fused, available, attention = center_fusion(
        compatible_tokens, valid_token_mask, item_embeddings,
        w_query, w_key, w_value, w_output,
    )
    _validate_score_inputs(
        user_embeddings, item_embeddings, fused, w_fused,
        score_weight, score_bias,
    )
    entities, item_dim = item_embeddings.shape
    private_features = item_embeddings.new_zeros((entities, 1))
    w_private = item_embeddings.new_zeros((item_dim, 1))
    private_available = torch.zeros(
        (entities,), dtype=torch.bool, device=item_embeddings.device,
    )
    logits = local_recommendation_logits(
        user_embeddings, item_embeddings, fused, private_features,
        w_fused, w_private, score_weight, score_bias,
        fused_available=available, private_available=private_available,
    )
    return logits, fused, available, attention


def fuse_score_group_loss_pri_none(
    compatible_tokens: torch.Tensor,
    valid_token_mask: torch.Tensor,
    labels: torch.Tensor,
    group_offsets: torch.Tensor,
    *,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    w_query: torch.Tensor,
    w_key: torch.Tensor,
    w_value: torch.Tensor,
    w_output: torch.Tensor,
    w_fused: torch.Tensor,
    score_weight: torch.Tensor,
    score_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unweighted group-mean BCE followed by the four scoring results.

    Labels and int64 offsets retain the accepted positive-first group
    contract, including groups with no negatives. Empty loss/zero groups
    reject. A future public executor applies m/N once OUTSIDE this scalar;
    this module neither samples/reorders examples nor reweights gradients.
    """
    logits, fused, available, attention = fuse_score_pri_none(
        compatible_tokens, valid_token_mask,
        user_embeddings=user_embeddings, item_embeddings=item_embeddings,
        w_query=w_query, w_key=w_key, w_value=w_value, w_output=w_output,
        w_fused=w_fused, score_weight=score_weight, score_bias=score_bias,
    )
    loss = positive_group_bce(logits, labels, group_offsets)
    return loss, logits, fused, available, attention


def _validate_score_inputs(
    users: torch.Tensor,
    items: torch.Tensor,
    fused: torch.Tensor,
    w_fused: torch.Tensor,
    score_weight: torch.Tensor,
    score_bias: torch.Tensor,
) -> None:
    """Validate caller scoring inputs before allocating private zero tensors.

    Fusion has already validated items and produced finite fused[E,h].
    The existing scorer remains the mathematical/validation authority; this
    small precheck makes malformed new-wrapper inputs fail before allocation.
    """
    for name, tensor, rank in (
        ("user_embeddings", users, 2),
        ("item_embeddings", items, 2),
        ("w_fused", w_fused, 2),
        ("score_weight", score_weight, 1),
        ("score_bias", score_bias, 0),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.is_nested or tensor.layout != torch.strided:
            raise TypeError(f"{name} must be a dense strided tensor")
        if tensor.device.type != "cpu":
            raise TypeError(f"{name} must be on CPU")
        if tensor.ndim != rank:
            raise ValueError(f"{name} must have rank {rank}")
        if tensor.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have dtype float32 or float64")
        if tensor.dtype != fused.dtype:
            raise TypeError("floating inputs must have the same dtype")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
    entities, item_dim = items.shape
    for name, tensor, shape in (
        ("user_embeddings", users, (entities, item_dim)),
        ("w_fused", w_fused, (item_dim, fused.shape[1])),
        ("score_weight", score_weight, (item_dim,)),
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
