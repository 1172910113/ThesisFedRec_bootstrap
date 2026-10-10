from dataclasses import fields

import pytest
import torch

from src.federation import ClientUpdate, aggregate_fedavg


def _update(
    party_id: int,
    value: torch.Tensor,
    weight: int,
    *,
    extra_state: dict[str, torch.Tensor] | None = None,
) -> ClientUpdate:
    shared_state = {"weight": value}
    if extra_state:
        shared_state.update(extra_state)
    return ClientUpdate(
        party_id=party_id,
        shared_state=shared_state,
        num_positive_interactions=weight,
    )


def test_client_update_contains_only_shared_aggregation_inputs() -> None:
    assert [field.name for field in fields(ClientUpdate)] == [
        "party_id",
        "shared_state",
        "num_positive_interactions",
    ]

    update = _update(37, torch.tensor([1.0]), 4)
    assert not hasattr(update, "user_embeddings")
    assert not hasattr(update, "optimizer_state")


def test_equal_weight_fedavg() -> None:
    reference = {"weight": torch.zeros(2)}
    result = aggregate_fedavg(
        reference,
        [
            _update(3, torch.tensor([1.0, 5.0]), 2),
            _update(8, torch.tensor([3.0, 1.0]), 2),
        ],
    )

    torch.testing.assert_close(result["weight"], torch.tensor([2.0, 3.0]))


def test_fedavg_uses_only_positive_interaction_weight() -> None:
    reference = {"weight": torch.zeros(1)}
    result = aggregate_fedavg(
        reference,
        [
            _update(4, torch.tensor([0.0]), 1),
            _update(9, torch.tensor([10.0]), 3),
        ],
    )

    torch.testing.assert_close(result["weight"], torch.tensor([7.5]))


def test_party_order_does_not_change_fedavg_result() -> None:
    reference = {"weight": torch.zeros(3, dtype=torch.float32)}
    updates = [
        _update(22, torch.tensor([0.1, 0.2, 0.3]), 1),
        _update(5, torch.tensor([1.5, 2.5, 3.5]), 2),
        _update(101, torch.tensor([8.0, 7.0, 6.0]), 4),
    ]

    first = aggregate_fedavg(reference, updates)
    second = aggregate_fedavg(reference, list(reversed(updates)))

    torch.testing.assert_close(first["weight"], second["weight"], rtol=0, atol=0)


def test_fedavg_requires_at_least_one_update() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_fedavg({"weight": torch.zeros(1)}, [])


@pytest.mark.parametrize(
    "shared_state",
    [
        {},
        {"weight": torch.ones(1), "user_embeddings": torch.ones(1)},
    ],
)
def test_missing_or_extra_shared_state_key_fails(
    shared_state: dict[str, torch.Tensor],
) -> None:
    update = ClientUpdate(
        party_id=1,
        shared_state=shared_state,
        num_positive_interactions=1,
    )

    with pytest.raises(ValueError, match="keys do not match"):
        aggregate_fedavg({"weight": torch.zeros(1)}, [update])


def test_shared_tensor_shape_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="different shape"):
        aggregate_fedavg(
            {"weight": torch.zeros(2)},
            [_update(1, torch.zeros(3), 1)],
        )


def test_shared_tensor_dtype_mismatch_fails() -> None:
    with pytest.raises(TypeError, match="different dtype"):
        aggregate_fedavg(
            {"weight": torch.zeros(2, dtype=torch.float32)},
            [_update(1, torch.zeros(2, dtype=torch.float64), 1)],
        )


@pytest.mark.parametrize("weight", [0, -1])
def test_non_positive_aggregation_weight_fails(weight: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        aggregate_fedavg(
            {"weight": torch.zeros(1)},
            [_update(1, torch.ones(1), weight)],
        )


def test_duplicate_party_id_fails() -> None:
    with pytest.raises(ValueError, match="unique party_id"):
        aggregate_fedavg(
            {"weight": torch.zeros(1)},
            [
                _update(7, torch.ones(1), 1),
                _update(7, torch.ones(1), 1),
            ],
        )


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_non_finite_floating_update_fails(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        aggregate_fedavg(
            {"weight": torch.zeros(1)},
            [_update(1, torch.tensor([invalid_value]), 1)],
        )


def test_identical_non_floating_state_is_preserved() -> None:
    reference = {
        "weight": torch.zeros(1),
        "step": torch.tensor([4], dtype=torch.int64),
    }
    updates = [
        _update(
            2,
            torch.tensor([1.0]),
            1,
            extra_state={"step": torch.tensor([4], dtype=torch.int64)},
        ),
        _update(
            9,
            torch.tensor([3.0]),
            1,
            extra_state={"step": torch.tensor([4], dtype=torch.int64)},
        ),
    ]

    result = aggregate_fedavg(reference, updates)

    torch.testing.assert_close(result["weight"], torch.tensor([2.0]))
    assert torch.equal(result["step"], reference["step"])
    assert result["step"] is not reference["step"]


def test_differing_non_floating_state_fails() -> None:
    reference = {
        "weight": torch.zeros(1),
        "step": torch.tensor([4], dtype=torch.int64),
    }
    update = _update(
        2,
        torch.tensor([1.0]),
        1,
        extra_state={"step": torch.tensor([5], dtype=torch.int64)},
    )

    with pytest.raises(ValueError, match="differs from the reference"):
        aggregate_fedavg(reference, [update])
