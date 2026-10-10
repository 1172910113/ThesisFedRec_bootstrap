import numpy as np
import pytest

from src.federation import select_party_ids


def test_full_participation_returns_every_sorted_party() -> None:
    selected = select_party_ids(
        [40, 3, 17, 9],
        client_fraction=1.0,
        run_seed=2026,
        round_index=0,
    )

    assert selected == (3, 9, 17, 40)


def test_full_participation_does_not_create_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("full participation must not create an RNG")

    monkeypatch.setattr(np.random, "default_rng", fail_if_called)

    assert select_party_ids(
        [2, 0, 1],
        client_fraction=1.0,
        run_seed=8,
        round_index=3,
    ) == (0, 1, 2)


def test_partial_participation_uses_ceil_without_replacement_and_sorts() -> None:
    selected = select_party_ids(
        [50, 10, 30, 20, 40],
        client_fraction=0.41,
        run_seed=19,
        round_index=2,
    )

    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert selected == tuple(sorted(selected))
    assert set(selected) <= {10, 20, 30, 40, 50}


def test_same_seed_round_and_available_ids_reproduce_selection() -> None:
    first = select_party_ids(
        [101, 7, 44, 900, 12, 63],
        client_fraction=0.5,
        run_seed=2027,
        round_index=4,
    )
    second = select_party_ids(
        [63, 12, 900, 44, 7, 101],
        client_fraction=0.5,
        run_seed=2027,
        round_index=4,
    )

    assert first == second


def test_different_rounds_use_distinct_deterministic_streams() -> None:
    party_ids = list(range(20))

    first = select_party_ids(
        party_ids,
        client_fraction=0.25,
        run_seed=2026,
        round_index=0,
    )
    second = select_party_ids(
        party_ids,
        client_fraction=0.25,
        run_seed=2026,
        round_index=1,
    )

    assert first != second
    assert first == select_party_ids(
        party_ids,
        client_fraction=0.25,
        run_seed=2026,
        round_index=0,
    )
    assert second == select_party_ids(
        party_ids,
        client_fraction=0.25,
        run_seed=2026,
        round_index=1,
    )


def test_non_contiguous_party_ids_are_not_treated_as_user_ids() -> None:
    party_ids = [51, 205, 999, 4001]
    recommendation_user_ids = {0, 1, 2, 3}

    selected = select_party_ids(
        party_ids,
        client_fraction=0.5,
        run_seed=33,
        round_index=7,
    )

    assert set(selected) <= set(party_ids)
    assert set(selected).isdisjoint(recommendation_user_ids)


@pytest.mark.parametrize(
    "client_fraction",
    [0, -0.1, 1.1, float("nan"), float("inf"), True],
)
def test_invalid_client_fraction_fails(client_fraction: float) -> None:
    with pytest.raises(ValueError, match="client_fraction"):
        select_party_ids(
            [0, 1],
            client_fraction=client_fraction,
            run_seed=2,
            round_index=0,
        )


def test_duplicate_available_party_ids_fail() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        select_party_ids(
            [4, 9, 4],
            client_fraction=0.5,
            run_seed=2,
            round_index=0,
        )
