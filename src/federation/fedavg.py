"""Deterministic FedAvg over shared tensor state only."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class ClientUpdate:
    """One party's shared-state upload and positive-interaction weight."""

    party_id: int
    shared_state: Mapping[str, torch.Tensor]
    num_positive_interactions: int


def aggregate_fedavg(
    reference_shared_state: Mapping[str, torch.Tensor],
    updates: Sequence[ClientUpdate],
) -> dict[str, torch.Tensor]:
    """Compute positive-interaction-weighted FedAvg on CPU."""
    reference_keys = _validate_reference_state(reference_shared_state)
    ordered_updates = _validate_updates(
        updates,
        reference_shared_state,
        reference_keys,
    )
    total_weight = sum(
        update.num_positive_interactions for update in ordered_updates
    )

    result: dict[str, torch.Tensor] = {}
    for key in reference_keys:
        reference = reference_shared_state[key].detach().cpu()
        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for update in ordered_updates:
                value = update.shared_state[key].detach().cpu()
                accumulator.add_(
                    value.to(dtype=torch.float64),
                    alpha=float(update.num_positive_interactions),
                )
            result[key] = (accumulator / total_weight).to(dtype=reference.dtype)
        else:
            for update in ordered_updates:
                value = update.shared_state[key].detach().cpu()
                if not torch.equal(value, reference):
                    raise ValueError(
                        f"non-floating shared tensor {key!r} differs "
                        "from the reference state"
                    )
            result[key] = reference.clone()
    return result


def _validate_reference_state(
    reference_shared_state: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    if not isinstance(reference_shared_state, Mapping):
        raise TypeError("reference_shared_state must be a mapping")
    if not reference_shared_state:
        raise ValueError("reference_shared_state must not be empty")
    if any(not isinstance(key, str) for key in reference_shared_state):
        raise TypeError("shared-state keys must be strings")

    for key, value in reference_shared_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"reference shared-state value {key!r} must be a tensor")
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError(
                f"reference floating shared tensor {key!r} must be finite"
            )
    return tuple(reference_shared_state)


def _validate_updates(
    updates: Sequence[ClientUpdate],
    reference_shared_state: Mapping[str, torch.Tensor],
    reference_keys: tuple[str, ...],
) -> tuple[ClientUpdate, ...]:
    ordered_updates = tuple(updates)
    if not ordered_updates:
        raise ValueError("at least one client update is required")

    party_ids: list[int] = []
    expected_keys = set(reference_keys)
    for update in ordered_updates:
        if not isinstance(update, ClientUpdate):
            raise TypeError("updates must contain ClientUpdate values")
        if (
            isinstance(update.party_id, bool)
            or not isinstance(update.party_id, int)
            or update.party_id < 0
        ):
            raise ValueError("party_id must be a non-negative integer")
        party_ids.append(update.party_id)
        if (
            isinstance(update.num_positive_interactions, bool)
            or not isinstance(update.num_positive_interactions, int)
            or update.num_positive_interactions <= 0
        ):
            raise ValueError("num_positive_interactions must be a positive integer")
        if not isinstance(update.shared_state, Mapping):
            raise TypeError("ClientUpdate.shared_state must be a mapping")
        if set(update.shared_state) != expected_keys:
            raise ValueError(
                f"party {update.party_id} shared-state keys do not match "
                "the reference state"
            )
        _validate_update_tensors(update, reference_shared_state, reference_keys)

    if len(set(party_ids)) != len(party_ids):
        raise ValueError("client updates must have unique party_id values")
    return tuple(sorted(ordered_updates, key=lambda update: update.party_id))


def _validate_update_tensors(
    update: ClientUpdate,
    reference_shared_state: Mapping[str, torch.Tensor],
    reference_keys: tuple[str, ...],
) -> None:
    for key in reference_keys:
        value = update.shared_state[key]
        reference = reference_shared_state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"party {update.party_id} shared-state value {key!r} "
                "must be a tensor"
            )
        if value.shape != reference.shape:
            raise ValueError(
                f"party {update.party_id} shared tensor {key!r} has "
                "a different shape"
            )
        if value.dtype != reference.dtype:
            raise TypeError(
                f"party {update.party_id} shared tensor {key!r} has "
                "a different dtype"
            )
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError(
                f"party {update.party_id} floating shared tensor {key!r} "
                "must be finite"
            )
