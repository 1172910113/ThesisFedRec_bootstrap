"""Deterministic selection and synchronous horizontal-federation rounds."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Generic, TypeVar

import numpy as np
import torch

from src.data.contracts import PartyData
from src.federation.fedavg import ClientUpdate, aggregate_fedavg

LocalStateT = TypeVar("LocalStateT")


@dataclass(frozen=True, slots=True, kw_only=True)
class HFLRoundResult(Generic[LocalStateT]):
    """The state transition and minimal bookkeeping from one HFL round."""

    next_shared_state: dict[str, torch.Tensor]
    local_states: dict[int, LocalStateT]
    selected_party_ids: tuple[int, ...]
    total_positive_interactions: int


def run_synchronous_hfl_round(
    shared_state: Mapping[str, torch.Tensor],
    parties: Sequence[PartyData],
    local_states: Mapping[int, LocalStateT],
    *,
    client_fraction: float,
    run_seed: int,
    round_index: int,
    local_update: Callable[
        [Mapping[str, torch.Tensor], LocalStateT, PartyData],
        tuple[LocalStateT, ClientUpdate],
    ],
) -> HFLRoundResult[LocalStateT]:
    """Run one synchronous HFL selection, local-update, and FedAvg step."""
    parties_by_id = _parties_by_id(parties)
    _validate_local_state_keys(local_states, parties_by_id)
    selected_party_ids = select_party_ids(
        tuple(parties_by_id),
        client_fraction=client_fraction,
        run_seed=run_seed,
        round_index=round_index,
    )

    next_local_states = dict(local_states)
    client_updates: list[ClientUpdate] = []
    for party_id in selected_party_ids:
        updated_local_state, client_update = local_update(
            shared_state,
            local_states[party_id],
            parties_by_id[party_id],
        )
        if not isinstance(client_update, ClientUpdate):
            raise TypeError("local_update must return a ClientUpdate")
        if client_update.party_id != party_id:
            raise ValueError(
                f"party {party_id} local update returned ClientUpdate for "
                f"party {client_update.party_id}"
            )
        next_local_states[party_id] = updated_local_state
        client_updates.append(client_update)

    next_shared_state = aggregate_fedavg(shared_state, client_updates)
    return HFLRoundResult(
        next_shared_state=next_shared_state,
        local_states=next_local_states,
        selected_party_ids=selected_party_ids,
        total_positive_interactions=sum(
            update.num_positive_interactions for update in client_updates
        ),
    )


def select_party_ids(
    available_party_ids: Sequence[int],
    *,
    client_fraction: float,
    run_seed: int,
    round_index: int,
) -> tuple[int, ...]:
    """Select sorted party IDs without replacement for one HFL round."""
    party_ids = _party_ids(available_party_ids)
    fraction = _client_fraction(client_fraction)
    _seed_component("run_seed", run_seed, upper_bound=2**32)
    _seed_component("round_index", round_index)

    if fraction == 1.0:
        return party_ids

    selected_count = max(1, ceil(fraction * len(party_ids)))
    seed_sequence = np.random.SeedSequence([run_seed, round_index])
    rng = np.random.default_rng(seed_sequence)
    selected_indices = rng.choice(
        len(party_ids),
        size=selected_count,
        replace=False,
    )
    return tuple(sorted(party_ids[int(index)] for index in selected_indices))


def _parties_by_id(parties: Sequence[PartyData]) -> dict[int, PartyData]:
    result: dict[int, PartyData] = {}
    for party in parties:
        if not isinstance(party, PartyData):
            raise TypeError("parties must contain PartyData values")
        if party.party_id in result:
            raise ValueError("parties must not contain duplicate party IDs")
        result[party.party_id] = party
    if not result:
        raise ValueError("parties must not be empty")
    return result


def _validate_local_state_keys(
    local_states: Mapping[int, LocalStateT],
    parties_by_id: Mapping[int, PartyData],
) -> None:
    if not isinstance(local_states, Mapping):
        raise TypeError("local_states must be a mapping")
    if any(
        isinstance(party_id, bool)
        or not isinstance(party_id, (int, np.integer))
        or party_id < 0
        for party_id in local_states
    ):
        raise ValueError("local_states keys must be non-negative party IDs")

    expected = set(parties_by_id)
    provided = {int(party_id) for party_id in local_states}
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        raise ValueError(
            "local_states keys must exactly match available party IDs; "
            f"missing={missing}, extra={extra}"
        )


def _party_ids(available_party_ids: Sequence[int]) -> tuple[int, ...]:
    values = tuple(available_party_ids)
    if not values:
        raise ValueError("available_party_ids must not be empty")
    if any(
        isinstance(party_id, bool)
        or not isinstance(party_id, (int, np.integer))
        or party_id < 0
        for party_id in values
    ):
        raise ValueError(
            "available_party_ids must contain non-negative integers"
        )
    party_ids = tuple(int(party_id) for party_id in values)
    if len(set(party_ids)) != len(party_ids):
        raise ValueError("available_party_ids must not contain duplicates")
    return tuple(sorted(party_ids))


def _client_fraction(client_fraction: float) -> float:
    if isinstance(client_fraction, bool) or not isinstance(
        client_fraction,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("client_fraction must be a finite number in (0, 1]")
    fraction = float(client_fraction)
    if not isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("client_fraction must be a finite number in (0, 1]")
    return fraction


def _seed_component(
    name: str,
    value: int,
    *,
    upper_bound: int | None = None,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < 0
        or (upper_bound is not None and value >= upper_bound)
    ):
        suffix = f" below {upper_bound}" if upper_bound is not None else ""
        raise ValueError(f"{name} must be a non-negative integer{suffix}")
