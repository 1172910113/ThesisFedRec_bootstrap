from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch

import src.federation.hfl as hfl_module
from src.data.contracts import DatasetBundle, PartyData
from src.data.partition import build_hfl_partition
from src.federation import (
    ClientUpdate,
    aggregate_fedavg,
    run_synchronous_hfl_round,
    select_party_ids,
)
from src.models import (
    FEDGMF_SHARED_KEYS,
    FedGMFLocalState,
    initialize_fedgmf_local_state,
    initialize_fedgmf_shared_state,
    train_local_fedgmf,
)


def _party(party_id: int, user_id: int) -> PartyData:
    return PartyData(
        party_id=party_id,
        train_interaction_ids=np.asarray([user_id], dtype=np.int64),
        valid_interaction_ids=np.empty(0, dtype=np.int64),
        test_interaction_ids=np.empty(0, dtype=np.int64),
        user_ids=np.asarray([user_id], dtype=np.int64),
        item_ids=np.asarray([0], dtype=np.int64),
        feature_view_names=(),
        graph_view_names=(),
        owns_labels=True,
        metadata={},
    )


def _fake_update(
    party: PartyData,
    local_state: str,
    *,
    shared_value: float,
    weight: int = 1,
) -> tuple[str, ClientUpdate]:
    return (
        f"{local_state}:updated-by-{party.party_id}",
        ClientUpdate(
            party_id=party.party_id,
            shared_state={"weight": torch.tensor([shared_value])},
            num_positive_interactions=weight,
        ),
    )


def test_full_round_is_synchronous_ordered_weighted_and_immutable() -> None:
    parties = (_party(40, 2), _party(3, 0), _party(17, 1))
    shared_state = {"weight": torch.tensor([2.0])}
    original_shared = shared_state["weight"].clone()
    local_states = {3: "local-3", 17: "local-17", 40: "local-40"}
    original_local_states = dict(local_states)
    call_order: list[int] = []
    received_shared: list[torch.Tensor] = []
    values = {3: 1.0, 17: 5.0, 40: 9.0}
    weights = {3: 1, 17: 2, 40: 3}

    def local_update(
        current_shared: Mapping[str, torch.Tensor],
        local_state: str,
        party: PartyData,
    ) -> tuple[str, ClientUpdate]:
        call_order.append(party.party_id)
        received_shared.append(current_shared["weight"].clone())
        return _fake_update(
            party,
            local_state,
            shared_value=values[party.party_id],
            weight=weights[party.party_id],
        )

    result = run_synchronous_hfl_round(
        shared_state,
        parties,
        local_states,
        client_fraction=1.0,
        run_seed=2026,
        round_index=0,
        local_update=local_update,
    )

    assert result.selected_party_ids == (3, 17, 40)
    assert call_order == [3, 17, 40]
    assert all(torch.equal(value, original_shared) for value in received_shared)
    expected = torch.tensor([(1.0 * 1 + 5.0 * 2 + 9.0 * 3) / 6])
    torch.testing.assert_close(result.next_shared_state["weight"], expected)
    assert result.total_positive_interactions == 6
    assert result.local_states == {
        3: "local-3:updated-by-3",
        17: "local-17:updated-by-17",
        40: "local-40:updated-by-40",
    }
    assert result.local_states is not local_states
    assert local_states == original_local_states
    assert torch.equal(shared_state["weight"], original_shared)


def test_partial_round_replaces_only_selected_states_and_reproduces() -> None:
    party_ids = (101, 7, 44, 900, 12)
    parties = tuple(
        _party(party_id, user_id)
        for user_id, party_id in enumerate(party_ids)
    )
    shared_state = {"weight": torch.tensor([0.0])}
    local_states = {
        party_id: f"local-{party_id}" for party_id in party_ids
    }
    expected_selected = select_party_ids(
        party_ids,
        client_fraction=0.4,
        run_seed=91,
        round_index=6,
    )
    calls: list[int] = []

    def local_update(
        current_shared: Mapping[str, torch.Tensor],
        local_state: str,
        party: PartyData,
    ) -> tuple[str, ClientUpdate]:
        calls.append(party.party_id)
        return _fake_update(
            party,
            local_state,
            shared_value=float(party.party_id),
        )

    first = run_synchronous_hfl_round(
        shared_state,
        parties,
        local_states,
        client_fraction=0.4,
        run_seed=91,
        round_index=6,
        local_update=local_update,
    )
    first_calls = tuple(calls)
    calls.clear()
    second = run_synchronous_hfl_round(
        shared_state,
        tuple(reversed(parties)),
        local_states,
        client_fraction=0.4,
        run_seed=91,
        round_index=6,
        local_update=local_update,
    )

    assert first.selected_party_ids == expected_selected
    assert first_calls == expected_selected
    assert tuple(calls) == expected_selected
    assert first.selected_party_ids == second.selected_party_ids
    assert first.local_states == second.local_states
    assert torch.equal(
        first.next_shared_state["weight"], second.next_shared_state["weight"]
    )
    for party_id in party_ids:
        if party_id in expected_selected:
            assert first.local_states[party_id] != local_states[party_id]
        else:
            assert first.local_states[party_id] is local_states[party_id]


def test_duplicate_party_ids_fail() -> None:
    parties = (_party(7, 0), _party(7, 1))

    with pytest.raises(ValueError, match="duplicate party IDs"):
        run_synchronous_hfl_round(
            {"weight": torch.zeros(1)},
            parties,
            {7: "local"},
            client_fraction=1.0,
            run_seed=1,
            round_index=0,
            local_update=lambda *_: pytest.fail("local update must not run"),
        )


@pytest.mark.parametrize(
    "local_states",
    [
        {3: "local-3"},
        {3: "local-3", 17: "local-17", 99: "extra"},
    ],
)
def test_missing_or_extra_local_state_keys_fail(
    local_states: dict[int, str],
) -> None:
    parties = (_party(3, 0), _party(17, 1))

    with pytest.raises(ValueError, match="exactly match available party IDs"):
        run_synchronous_hfl_round(
            {"weight": torch.zeros(1)},
            parties,
            local_states,
            client_fraction=1.0,
            run_seed=1,
            round_index=0,
            local_update=lambda *_: pytest.fail("local update must not run"),
        )


def test_client_failure_aborts_before_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parties = (_party(40, 2), _party(3, 0), _party(17, 1))
    shared_state = {"weight": torch.tensor([2.0])}
    local_states = {3: "local-3", 17: "local-17", 40: "local-40"}
    calls: list[int] = []

    def local_update(
        current_shared: Mapping[str, torch.Tensor],
        local_state: str,
        party: PartyData,
    ) -> tuple[str, ClientUpdate]:
        calls.append(party.party_id)
        if party.party_id == 17:
            raise RuntimeError("client failed")
        return _fake_update(party, local_state, shared_value=1.0)

    def forbidden_aggregation(*args: object, **kwargs: object) -> None:
        raise AssertionError("aggregation must not run after a client failure")

    monkeypatch.setattr(hfl_module, "aggregate_fedavg", forbidden_aggregation)

    with pytest.raises(RuntimeError, match="client failed"):
        run_synchronous_hfl_round(
            shared_state,
            parties,
            local_states,
            client_fraction=1.0,
            run_seed=1,
            round_index=0,
            local_update=local_update,
        )

    assert calls == [3, 17]
    assert local_states == {3: "local-3", 17: "local-17", 40: "local-40"}
    assert torch.equal(shared_state["weight"], torch.tensor([2.0]))


def test_client_update_party_id_must_match_selected_party() -> None:
    party = _party(17, 0)

    def mismatched_update(
        current_shared: Mapping[str, torch.Tensor],
        local_state: str,
        selected_party: PartyData,
    ) -> tuple[str, ClientUpdate]:
        return (
            "updated",
            ClientUpdate(
                party_id=999,
                shared_state={"weight": torch.ones(1)},
                num_positive_interactions=1,
            ),
        )

    with pytest.raises(ValueError, match="ClientUpdate for party 999"):
        run_synchronous_hfl_round(
            {"weight": torch.zeros(1)},
            (party,),
            {17: "local"},
            client_fraction=1.0,
            run_seed=1,
            round_index=0,
            local_update=mismatched_update,
        )


def test_real_fedgmf_local_updates_integrate_with_one_hfl_round(
    processed_bundle: DatasetBundle,
) -> None:
    parties = build_hfl_partition(
        processed_bundle,
        mode="grouped_round_robin",
        num_parties=2,
        seed=17,
    )
    shared_state = initialize_fedgmf_shared_state(
        num_items=len(processed_bundle.item_id_map),
        embedding_dim=4,
        run_seed=2026,
    )
    original_shared = {
        key: value.clone() for key, value in shared_state.items()
    }
    local_states = {
        party.party_id: initialize_fedgmf_local_state(
            party.user_ids,
            embedding_dim=4,
            run_seed=2026,
            party_id=party.party_id,
        )
        for party in parties
    }
    original_local_embeddings = {
        party_id: state.user_embeddings.clone()
        for party_id, state in local_states.items()
    }
    updates: list[ClientUpdate] = []

    def local_update(
        current_shared: Mapping[str, torch.Tensor],
        local_state: FedGMFLocalState,
        party: PartyData,
    ) -> tuple[FedGMFLocalState, ClientUpdate]:
        updated_local, update = train_local_fedgmf(
            shared_state=current_shared,
            local_state=local_state,
            bundle=processed_bundle,
            party=party,
            round_index=0,
            local_epochs=1,
            learning_rate=0.03,
            batch_size=5,
            weight_decay=0.0,
            negatives_per_positive=4,
            negative_strategy="uniform",
            excluded_splits=("train", "valid", "test"),
            run_seed=2026,
            device="cpu",
        )
        updates.append(update)
        return updated_local, update

    result = run_synchronous_hfl_round(
        shared_state,
        parties,
        local_states,
        client_fraction=1.0,
        run_seed=2026,
        round_index=0,
        local_update=local_update,
    )

    assert result.selected_party_ids == (0, 1)
    assert len(updates) == 2
    assert result.total_positive_interactions == sum(
        party.train_interaction_ids.size for party in parties
    )
    expected_shared = aggregate_fedavg(shared_state, updates)
    for key in FEDGMF_SHARED_KEYS:
        assert torch.equal(result.next_shared_state[key], expected_shared[key])
        assert torch.isfinite(result.next_shared_state[key]).all().item()
        assert torch.equal(shared_state[key], original_shared[key])
    for update in updates:
        assert tuple(update.shared_state) == FEDGMF_SHARED_KEYS
        assert all("user" not in key for key in update.shared_state)
    for party_id, state in result.local_states.items():
        assert state is not local_states[party_id]
        assert not torch.equal(
            state.user_embeddings,
            original_local_embeddings[party_id],
        )
        assert torch.equal(
            local_states[party_id].user_embeddings,
            original_local_embeddings[party_id],
        )
