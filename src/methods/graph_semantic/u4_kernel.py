"""Differentiable synthetic U4-K0 source/sketch and frozen K1 receiver.

No optimizer, sampling, graph delivery, basis construction or message service.
All inputs are CPU float32/float64. Public tensors are detached references;
callers must keep their contents fixed throughout use (no full E/X clone).
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .path_operator import PathFactors, _floating, _ids, apply_path


@dataclass(frozen=True, slots=True)
class SourceKernel:
    paths: PathFactors
    item_embeddings: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    basis: torch.Tensor
    f2: torch.Tensor
    f4: torch.Tensor


@dataclass(frozen=True, slots=True)
class FrozenSourceSnapshot:
    paths: PathFactors
    item_embeddings: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    basis: torch.Tensor
    delta: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReceiverKernel:
    source: FrozenSourceSnapshot
    user_embeddings: torch.Tensor
    bank: torch.Tensor
    z: torch.Tensor


def prepare_source_kernel(
    paths: PathFactors,
    item_embeddings: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    basis: torch.Tensor,
) -> SourceKernel:
    """Freeze E/w/b/B and compute F only on supported local rows.

    E: [n,d], w: [d], b: scalar, B: [n,p], 1 <= p <= n. B is supplied,
    not constructed: B.T @ B must equal I using atol=rtol=1e-10 (float64)
    or 1e-5 (float32). p=2 in tests is a fixture, not a formal profile.
    """
    dtype = paths.weights.dtype
    for name, tensor in (
        ("item_embeddings", item_embeddings), ("weight", weight),
        ("bias", bias), ("basis", basis),
    ):
        _floating(tensor, name, dtype)
    if (
        item_embeddings.ndim != 2
        or item_embeddings.shape[0] != paths.num_items
        or item_embeddings.shape[1] < 1
    ):
        raise ValueError("item_embeddings must have shape (catalogue size, positive dimension)")
    if weight.shape != (item_embeddings.shape[1],) or bias.shape != ():
        raise ValueError("weight must have shape (embedding dimension,) and bias must be scalar")
    if basis.ndim != 2 or basis.shape[0] != paths.num_items or not 1 <= basis.shape[1] <= paths.num_items:
        raise ValueError("basis must have shape (catalogue size, p), 1 <= p <= catalogue size")
    basis = basis.detach()
    tolerance = 1e-10 if dtype == torch.float64 else 1e-5
    if not torch.allclose(
        basis.T @ basis, torch.eye(basis.shape[1], dtype=dtype),
        atol=tolerance, rtol=tolerance,
    ):
        raise ValueError("basis columns must be orthonormal")
    items = item_embeddings.detach()
    f2 = apply_path(paths, items[paths.local_item_ids])
    f4 = apply_path(paths, f2)
    return SourceKernel(paths, items, weight.detach(), bias.detach(), basis, f2, f4)


def _scalars(value: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    _floating(value, name, dtype)
    if value.shape != (2,):
        raise ValueError(f"{name} must have shape (2,) in channel order [2,4]")


def _delta(source: SourceKernel, a: torch.Tensor) -> torch.Tensor:
    _scalars(a, "a", source.item_embeddings.dtype)
    gates = torch.sigmoid(a)
    return gates[0] * source.f2 + gates[1] * source.f4


def _batch(
    source: SourceKernel | FrozenSourceSnapshot,
    users: torch.Tensor,
    user_rows: torch.Tensor,
    item_ids: torch.Tensor,
) -> None:
    _floating(users, "user_embeddings", source.item_embeddings.dtype)
    if users.ndim != 2 or users.shape[1] != source.item_embeddings.shape[1]:
        raise ValueError("user_embeddings must have shape (local users, embedding dimension)")
    _ids(user_rows, "user_rows")
    _ids(item_ids, "item_ids")
    if not item_ids.numel() or item_ids.shape != user_rows.shape:
        raise ValueError("batch must be nonempty with aligned user_rows and item_ids")
    if torch.any(user_rows >= users.shape[0]).item() or torch.any(item_ids >= source.paths.num_items).item():
        raise ValueError("batch user row or public item ID is out of range")


def _item_vectors(
    source: SourceKernel | FrozenSourceSnapshot,
    delta: torch.Tensor,
    item_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather E and supported residuals for a batch; never materialize X."""
    result = source.item_embeddings.detach()[item_ids]
    paths = source.paths
    if not paths.local_item_ids.numel():
        return result
    positions = torch.searchsorted(paths.sorted_item_ids, item_ids)
    safe_positions = positions.clamp(max=paths.local_item_ids.numel() - 1)
    present = (positions < paths.local_item_ids.numel()) & (
        paths.sorted_item_ids[safe_positions] == item_ids
    )
    batch_rows = torch.nonzero(present, as_tuple=True)[0]
    local_rows = paths.sorted_local_rows[positions[present]]
    return result.index_add(0, batch_rows, delta[local_rows])


def _score(users: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    return (users * items * weight.detach()).sum(dim=1) + bias.detach()


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _floating(labels, "labels", logits.dtype)
    if labels.shape != logits.shape or not torch.all((labels == 0) | (labels == 1)).item():
        raise ValueError("labels must be an aligned binary vector")
    return F.binary_cross_entropy_with_logits(logits, labels.detach(), reduction="mean")


def source_logits(
    source: SourceKernel,
    user_embeddings: torch.Tensor,
    a: torch.Tensor,
    user_rows: torch.Tensor,
    item_ids: torch.Tensor,
) -> torch.Tensor:
    """Score explicit public items; only private user embeddings and a train."""
    _batch(source, user_embeddings, user_rows, item_ids)
    items = _item_vectors(source, _delta(source, a), item_ids)
    return _score(user_embeddings[user_rows], source.weight, source.bias, items)


def source_loss(
    source: SourceKernel,
    user_embeddings: torch.Tensor,
    a: torch.Tensor,
    user_rows: torch.Tensor,
    item_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Stable BCE-with-logits, mean over an explicit nonempty labeled batch."""
    return _loss(source_logits(source, user_embeddings, a, user_rows, item_ids), labels)


def make_operator_sketch(source: SourceKernel, a: torch.Tensor) -> torch.Tensor:
    """Return differentiable [2,p,p] using the same caller-owned source a."""
    _scalars(a, "a", source.item_embeddings.dtype)
    local_basis = source.basis[source.paths.local_item_ids]
    two = apply_path(source.paths, local_basis)
    four = apply_path(source.paths, two)
    gates = torch.sigmoid(a)
    return torch.stack((gates[0] * (local_basis.T @ two), gates[1] * (local_basis.T @ four)))


def detached_operator_snapshot(source: SourceKernel, final_a: torch.Tensor) -> torch.Tensor:
    """Freshly recompute final T, then detach; no initialization-time T cache."""
    return make_operator_sketch(source, final_a).detach()


def freeze_source_snapshot(source: SourceKernel, final_a: torch.Tensor) -> FrozenSourceSnapshot:
    """Freeze the local residual at final a; keep shared public storage."""
    return FrozenSourceSnapshot(
        source.paths, source.item_embeddings.detach(), source.weight.detach(),
        source.bias.detach(), source.basis.detach(), _delta(source, final_a).detach().clone(),
    )


def prepare_receiver(
    source: FrozenSourceSnapshot,
    user_embeddings: torch.Tensor,
    bank: torch.Tensor,
) -> ReceiverKernel:
    """Freeze local p/D/Z and an explicit detached K1 bank [2,p,p].

    Bank provenance/aggregation and snapshot versions are coordinator concerns.
    The receiver has no implicit access to any other source.
    """
    dtype = source.item_embeddings.dtype
    _floating(user_embeddings, "user_embeddings", dtype)
    if user_embeddings.ndim != 2 or user_embeddings.shape[1] != source.item_embeddings.shape[1]:
        raise ValueError("user_embeddings must have shape (local users, embedding dimension)")
    _floating(bank, "bank", dtype)
    p = source.basis.shape[1]
    if bank.shape != (2, p, p) or bank.requires_grad or bank.grad_fn is not None:
        raise ValueError("bank must be an explicit detached K1 tensor of shape (2,p,p)")
    _floating(source.delta, "source.delta", dtype)
    if source.delta.shape != (source.paths.local_item_ids.numel(), source.item_embeddings.shape[1]):
        raise ValueError("source.delta must contain only supported local rows")
    frozen = FrozenSourceSnapshot(
        source.paths, source.item_embeddings.detach(), source.weight.detach(),
        source.bias.detach(), source.basis.detach(), source.delta.detach().clone(),
    )
    z = frozen.basis.T @ frozen.item_embeddings
    z = z + frozen.basis[frozen.paths.local_item_ids].T @ frozen.delta
    return ReceiverKernel(frozen, user_embeddings.detach().clone(), bank.detach().clone(), z.detach())


def receiver_tokens(receiver: ReceiverKernel, item_ids: torch.Tensor) -> torch.Tensor:
    """Return frozen batch tokens [batch,2,embedding dimension]."""
    _ids(item_ids, "item_ids")
    if torch.any(item_ids >= receiver.source.paths.num_items).item():
        raise ValueError("item_ids are out of catalogue range")
    basis_rows = receiver.source.basis.detach()[item_ids]
    return torch.stack([
        basis_rows @ receiver.bank.detach()[channel] @ receiver.z.detach()
        for channel in (0, 1)
    ], dim=1)


def receiver_logits(
    receiver: ReceiverKernel,
    beta: torch.Tensor,
    user_rows: torch.Tensor,
    item_ids: torch.Tensor,
) -> torch.Tensor:
    """Score with two unconstrained caller-owned beta scalars as sole gradients."""
    source = receiver.source
    _scalars(beta, "beta", source.item_embeddings.dtype)
    _batch(source, receiver.user_embeddings, user_rows, item_ids)
    local = _item_vectors(source, source.delta.detach(), item_ids)
    items = local + (beta[None, :, None] * receiver_tokens(receiver, item_ids)).sum(dim=1)
    return _score(receiver.user_embeddings.detach()[user_rows], source.weight, source.bias, items)


def receiver_loss(
    receiver: ReceiverKernel,
    beta: torch.Tensor,
    user_rows: torch.Tensor,
    item_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Mean labeled BCE for beta only; empty labeled batches raise ValueError."""
    return _loss(receiver_logits(receiver, beta, user_rows, item_ids), labels)
