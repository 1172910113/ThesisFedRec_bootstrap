"""Small deterministic helpers for horizontal federation."""

from collections.abc import Sequence
from math import ceil, isfinite

import numpy as np


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
