from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch

import src.models.fedgmf as fedgmf
from src.data.contracts import DatasetBundle, PartyData
from src.data.negative_sampling import sample_negative_items as real_sampler
from src.federation.fedavg import ClientUpdate
from src.models import (
    FEDGMF_SHARED_KEYS,
    FedGMFLocalState,
    initialize_fedgmf_local_state,
    initialize_fedgmf_shared_state,
    score_fedgmf,
    train_local_fedgmf,
)


def _grouped_party(bundle: DatasetBundle) -> PartyData:
    user_ids = np.asarray([0, 2], dtype=np.int64)
    owned = np.isin(bundle.interactions.user_ids, user_ids)
    all_interaction_ids = np.flatnonzero(owned)

    def split_ids(name: str) -> np.ndarray:
        split = getattr(bundle.splits, f"{name}_interaction_ids")
        return split[np.isin(split, all_interaction_ids)]

    item_ids = np.unique(bundle.interactions.item_ids[all_interaction_ids])
    return PartyData(
        party_id=17,
        user_ids=user_ids,
        item_ids=item_ids,
        train_interaction_ids=split_ids("train"),
        valid_interaction_ids=split_ids("valid"),
        test_interaction_ids=split_ids("test"),
        feature_view_names=(),
        graph_view_names=(),
        owns_labels=True,
        metadata={"partition_mode": "grouped_round_robin"},
    )


def _initial_states(
    bundle: DatasetBundle,
    party: PartyData,
    *,
    embedding_dim: int = 4,
    run_seed: int = 2026,
) -> tuple[dict[str, torch.Tensor], FedGMFLocalState]:
    shared_state = initialize_fedgmf_shared_state(
        num_items=len(bundle.item_id_map),
        embedding_dim=embedding_dim,
        run_seed=run_seed,
    )
    local_state = initialize_fedgmf_local_state(
        user_ids=party.user_ids,
        embedding_dim=embedding_dim,
        run_seed=run_seed,
        party_id=party.party_id,
    )
    return shared_state, local_state


def _train(
    bundle: DatasetBundle,
    party: PartyData,
    shared_state: dict[str, torch.Tensor],
    local_state: FedGMFLocalState,
    *,
    round_index: int = 0,
    local_epochs: int = 1,
    run_seed: int = 2026,
) -> tuple[FedGMFLocalState, ClientUpdate]:
    return train_local_fedgmf(
        shared_state=shared_state,
        local_state=local_state,
        bundle=bundle,
        party=party,
        round_index=round_index,
        local_epochs=local_epochs,
        learning_rate=0.03,
        batch_size=3,
        weight_decay=0.0,
        negatives_per_positive=4,
        negative_strategy="uniform",
        excluded_splits=("train", "valid", "test"),
        run_seed=run_seed,
        device="cpu",
    )


def _known_items(bundle: DatasetBundle, user_id: int) -> set[int]:
    mask = bundle.interactions.user_ids == user_id
    return set(bundle.interactions.item_ids[mask].tolist())


def test_score_fedgmf_matches_manual_logits() -> None:
    shared_state = {
        "item_embeddings": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32
        ),
        "prediction_weight": torch.tensor([[0.5, -1.0]], dtype=torch.float32),
        "prediction_bias": torch.tensor([0.25], dtype=torch.float32),
    }
    local_state = FedGMFLocalState(
        user_ids=torch.tensor([3, 17], dtype=torch.long),
        user_embeddings=torch.tensor(
            [[2.0, 1.0], [1.0, 3.0]], dtype=torch.float32
        ),
    )

    logits = score_fedgmf(
        shared_state=shared_state,
        local_state=local_state,
        user_ids=[17, 3],
        item_ids=[1, 2],
    )

    expected = torch.tensor([-10.25, -0.75], dtype=torch.float32)
    assert logits.shape == (2,)
    assert torch.equal(logits, expected)
    assert logits.device.type == "cpu"


def test_initialization_is_deterministic_and_separates_state() -> None:
    first_shared = initialize_fedgmf_shared_state(11, 5, run_seed=9)
    second_shared = initialize_fedgmf_shared_state(11, 5, run_seed=9)
    first_local = initialize_fedgmf_local_state(
        [8, 2], 5, run_seed=9, party_id=41
    )
    second_local = initialize_fedgmf_local_state(
        [2, 8], 5, run_seed=9, party_id=41
    )

    assert tuple(first_shared) == FEDGMF_SHARED_KEYS
    assert all("user" not in key for key in first_shared)
    for key in FEDGMF_SHARED_KEYS:
        assert torch.equal(first_shared[key], second_shared[key])
    assert torch.equal(first_local.user_ids, torch.tensor([2, 8]))
    assert torch.equal(first_local.user_ids, second_local.user_ids)
    assert torch.equal(first_local.user_embeddings, second_local.user_embeddings)
    assert not torch.equal(
        first_local.user_embeddings[0], first_local.user_embeddings[1]
    )
    assert not hasattr(first_local, "item_embeddings")


def test_score_rejects_user_not_owned_by_party() -> None:
    shared_state = initialize_fedgmf_shared_state(4, 3, run_seed=5)
    local_state = initialize_fedgmf_local_state(
        [1, 2], 3, run_seed=5, party_id=99
    )

    with pytest.raises(ValueError, match="does not own user IDs"):
        score_fedgmf(shared_state, local_state, [99], [0])


def test_grouped_party_training_updates_only_returned_states(
    processed_bundle: DatasetBundle,
) -> None:
    party = _grouped_party(processed_bundle)
    shared_state, local_state = _initial_states(processed_bundle, party)
    original_shared = {key: value.clone() for key, value in shared_state.items()}
    original_local = local_state.user_embeddings.clone()

    updated_local, update = _train(
        processed_bundle, party, shared_state, local_state
    )

    for key in FEDGMF_SHARED_KEYS:
        assert torch.equal(shared_state[key], original_shared[key])
    assert torch.equal(local_state.user_embeddings, original_local)
    assert torch.equal(updated_local.user_ids, torch.tensor([0, 2]))
    assert updated_local.user_embeddings.device.type == "cpu"
    assert not torch.equal(updated_local.user_embeddings, original_local)
    assert not torch.equal(
        updated_local.user_embeddings[0], updated_local.user_embeddings[1]
    )
    assert update.party_id == 17
    assert update.num_positive_interactions == party.train_interaction_ids.size
    assert tuple(update.shared_state) == FEDGMF_SHARED_KEYS
    assert all("user" not in key for key in update.shared_state)
    assert all(value.device.type == "cpu" for value in update.shared_state.values())
    assert not hasattr(update, "optimizer_state")
    assert not hasattr(update, "local_state")
    assert any(
        not torch.equal(update.shared_state[key], original_shared[key])
        for key in FEDGMF_SHARED_KEYS
    )


def test_local_training_is_exactly_reproducible(
    processed_bundle: DatasetBundle,
) -> None:
    party = _grouped_party(processed_bundle)
    shared_state, local_state = _initial_states(processed_bundle, party)

    first_local, first_update = _train(
        processed_bundle,
        party,
        shared_state,
        local_state,
        round_index=3,
        local_epochs=2,
    )
    second_local, second_update = _train(
        processed_bundle,
        party,
        shared_state,
        local_state,
        round_index=3,
        local_epochs=2,
    )

    assert torch.equal(first_local.user_embeddings, second_local.user_embeddings)
    for key in FEDGMF_SHARED_KEYS:
        assert torch.equal(
            first_update.shared_state[key], second_update.shared_state[key]
        )


def test_training_uses_shared_negative_sampler_with_all_known_exclusions(
    processed_bundle: DatasetBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    party = _grouped_party(processed_bundle)
    shared_state, local_state = _initial_states(processed_bundle, party)
    calls: list[
        tuple[np.ndarray, dict[str, object], dict[int, tuple[int, ...]]]
    ] = []

    def recording_sampler(
        bundle: DatasetBundle,
        interaction_ids: Sequence[int] | np.ndarray,
        **kwargs: object,
    ) -> dict[int, tuple[int, ...]]:
        sampled = real_sampler(bundle, interaction_ids, **kwargs)
        calls.append((np.asarray(interaction_ids).copy(), dict(kwargs), sampled))
        return sampled

    monkeypatch.setattr(fedgmf, "sample_negative_items", recording_sampler)

    _train(processed_bundle, party, shared_state, local_state)

    assert len(calls) == 1
    positive_ids, kwargs, sampled = calls[0]
    assert np.array_equal(positive_ids, party.train_interaction_ids)
    assert not np.intersect1d(positive_ids, party.valid_interaction_ids).size
    assert not np.intersect1d(positive_ids, party.test_interaction_ids).size
    assert kwargs["negatives_per_positive"] == 4
    assert kwargs["strategy"] == "uniform"
    assert kwargs["excluded_splits"] == ("train", "valid", "test")
    assert kwargs["replace"] is False
    for interaction_id in positive_ids:
        negatives = sampled[int(interaction_id)]
        user_id = int(processed_bundle.interactions.user_ids[interaction_id])
        positive_item = int(processed_bundle.interactions.item_ids[interaction_id])
        assert len(negatives) == 4
        assert np.unique(negatives).size == 4
        assert positive_item not in negatives
        assert _known_items(processed_bundle, user_id).isdisjoint(negatives)


def test_negative_seed_changes_by_round_and_local_epoch(
    processed_bundle: DatasetBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    party = _grouped_party(processed_bundle)
    shared_state, local_state = _initial_states(processed_bundle, party)
    recorded_seeds: list[int] = []

    def recording_sampler(
        bundle: DatasetBundle,
        interaction_ids: Sequence[int] | np.ndarray,
        **kwargs: object,
    ) -> dict[int, tuple[int, ...]]:
        recorded_seeds.append(int(kwargs["seed"]))
        return real_sampler(bundle, interaction_ids, **kwargs)

    monkeypatch.setattr(fedgmf, "sample_negative_items", recording_sampler)

    _train(
        processed_bundle,
        party,
        shared_state,
        local_state,
        round_index=0,
        local_epochs=2,
    )
    _train(
        processed_bundle,
        party,
        shared_state,
        local_state,
        round_index=1,
        local_epochs=2,
    )

    expected = [
        int(
            np.random.SeedSequence([2026, round_index, 17, local_epoch])
            .generate_state(1, dtype=np.uint64)[0]
        )
        for round_index in (0, 1)
        for local_epoch in (0, 1)
    ]
    assert recorded_seeds == expected
    assert len(set(recorded_seeds)) == 4


def test_updated_local_state_can_be_reused_next_round(
    processed_bundle: DatasetBundle,
) -> None:
    party = _grouped_party(processed_bundle)
    shared_state, local_state = _initial_states(processed_bundle, party)

    round_zero_local, round_zero_update = _train(
        processed_bundle, party, shared_state, local_state, round_index=0
    )
    round_one_local, _ = _train(
        processed_bundle,
        party,
        round_zero_update.shared_state,
        round_zero_local,
        round_index=1,
    )

    assert torch.equal(round_one_local.user_ids, round_zero_local.user_ids)
    assert not torch.equal(
        round_one_local.user_embeddings, round_zero_local.user_embeddings
    )
