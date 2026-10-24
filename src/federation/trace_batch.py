"""Pack a supplied positive-group descriptor; no sampling or loss weighting."""

from dataclasses import dataclass

import torch

from src.federation.round_drain import PartyGroupBatch


_MAX_INDEX = 2**63 - 1


@dataclass(frozen=True, slots=True, kw_only=True)
class PackedPartyGroupBatch:
    """Trusted immutable sidecars and independently owned, mutable CPU tensors.

    Freezing the record does not freeze its tensors. Output mutations cannot
    change the supplied descriptor or another call's output.
    """

    party_id: int
    positive_ids: tuple[int, ...]
    loss_coefficient: tuple[int, int]
    user_ids: torch.Tensor
    item_ids: torch.Tensor
    labels: torch.Tensor
    group_offsets: torch.Tensor


def _integer(value: int, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in int")
    if value < 0 or value > _MAX_INDEX:
        raise ValueError(f"{name} must be in 0..2**63-1")


def _ids(value: tuple[int, ...], name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    for item in value:
        _integer(item, f"{name} element")


def _expanded_size(group_count: int, negatives_per_positive: int) -> int:
    """Python-integer capacity check, before any tensor/output allocation."""
    length = group_count * (negatives_per_positive + 1)
    if length > _MAX_INDEX or group_count + 1 > _MAX_INDEX:
        raise ValueError("expanded batch or group-offset capacity exceeds int64")
    return length


def _validate_batch(batch: PartyGroupBatch) -> tuple[int, int, int]:
    if type(batch) is not PartyGroupBatch:
        raise TypeError("batch must be an exact PartyGroupBatch")
    for name in ("party_id", "start", "stop"):
        _integer(getattr(batch, name), name)
    for name in ("positive_ids", "user_ids", "positive_item_ids"):
        _ids(getattr(batch, name), name)
    if type(batch.negative_item_ids) is not tuple:
        raise TypeError("negative_item_ids must be an exact tuple")
    for row in batch.negative_item_ids:
        _ids(row, "negative_item_ids row")
    _ids(batch.loss_coefficient, "loss_coefficient")

    groups = len(batch.positive_ids)
    if groups == 0:
        raise ValueError("batch must contain at least one positive group")
    if any(len(values) != groups for values in (
        batch.user_ids, batch.positive_item_ids, batch.negative_item_ids,
    )):
        raise ValueError("positive, user, item and negative rows must align")
    if batch.start >= batch.stop or batch.stop - batch.start != groups:
        raise ValueError("start/stop must delimit exactly the positive groups")
    if len(set(batch.positive_ids)) != groups:
        raise ValueError("positive_ids must be unique within the batch")
    negatives = len(batch.negative_item_ids[0])
    if any(len(row) != negatives for row in batch.negative_item_ids):
        raise ValueError("every negative row must have the same length")
    for positive, row in zip(batch.positive_item_ids, batch.negative_item_ids):
        if positive in row:
            raise ValueError("a current positive item cannot be its own negative")
    if (len(batch.loss_coefficient) != 2
            or batch.loss_coefficient[0] != groups
            or batch.loss_coefficient[1] < groups):
        raise ValueError("loss_coefficient must be the unreduced pair (m, N), N >= m")
    length = _expanded_size(groups, negatives)
    return groups, negatives, length


def pack_party_group_batch(
    batch: PartyGroupBatch, *, dtype: torch.dtype,
) -> PackedPartyGroupBatch:
    """Create group-major positive-first tensors, preserving supplied order.

    Only float32/float64 labels are supported. All descriptor types, domains,
    lengths, current-target exclusions and int64 capacities are checked before
    allocation. This has no bundle and does not certify known-positive
    exclusion or access permissions. The caller, not this function, applies
    the unchanged integer loss coefficient exactly once.
    """
    if dtype is not torch.float32 and dtype is not torch.float64:
        raise TypeError("dtype must be torch.float32 or torch.float64")
    groups, negatives, length = _validate_batch(batch)
    width = negatives + 1
    users = [user for user in batch.user_ids for _ in range(width)]
    items = [item for positive, row in zip(
        batch.positive_item_ids, batch.negative_item_ids,
    ) for item in (positive, *row)]
    user_ids = torch.tensor(users, dtype=torch.int64, device="cpu")
    item_ids = torch.tensor(items, dtype=torch.int64, device="cpu")
    labels = torch.zeros(length, dtype=dtype, device="cpu")
    labels[::width] = 1
    group_offsets = torch.tensor(
        [group * width for group in range(groups + 1)],
        dtype=torch.int64, device="cpu",
    )
    return PackedPartyGroupBatch(
        party_id=batch.party_id,
        positive_ids=batch.positive_ids,
        loss_coefficient=batch.loss_coefficient,
        user_ids=user_ids,
        item_ids=item_ids,
        labels=labels,
        group_offsets=group_offsets,
    )
