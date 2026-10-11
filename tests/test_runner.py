from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import src.federation.runner as runner_module
from src.data.contracts import DatasetBundle, PartyData
from src.data.partition import build_hfl_partition
from src.evaluation.ranking import EvaluationCandidateSet
from src.federation.runner import run_fedgmf_hfl_experiment
from src.models import (
    FEDGMF_SHARED_KEYS,
    FedGMFLocalState,
    initialize_fedgmf_local_state,
    initialize_fedgmf_shared_state,
)
from src.utils.config import load_yaml, validate_phase3_experiment_config

_REFERENCE_CONFIG = (
    Path(__file__).parents[1]
    / "configs"
    / "experiments"
    / "ml1m_fedgmf_fedavg.yaml"
)


def _config(
    *,
    rounds: int = 2,
    client_fraction: float = 1.0,
    grouped_parties: int | None = 2,
) -> dict[str, Any]:
    config = deepcopy(load_yaml(_REFERENCE_CONFIG))
    config["federation"].update(
        rounds=rounds,
        client_fraction=client_fraction,
        local_epochs=1,
    )
    if grouped_parties is None:
        config["federation"]["partition"] = "per_user"
        config["federation"].pop("num_parties", None)
    else:
        config["federation"]["partition"] = "grouped_round_robin"
        config["federation"]["num_parties"] = grouped_parties
    config["model"]["embedding_dim"] = 3
    config["optimization"].update(
        learning_rate=0.02,
        batch_size=4,
        train_negatives_per_positive=1,
    )
    config["evaluation"].update(
        k=[1, 3],
        negatives_per_positive=2,
    )
    config["runtime"].update(seed=37, device="cpu")
    validate_phase3_experiment_config(config)
    return config


def _grouped_parties(bundle: DatasetBundle) -> tuple[PartyData, ...]:
    base_parties = build_hfl_partition(
        bundle,
        mode="grouped_round_robin",
        num_parties=2,
        seed=11,
    )
    arbitrary_party_ids = (40, 7)
    return tuple(
        replace(party, party_id=party_id)
        for party, party_id in zip(
            base_parties,
            arbitrary_party_ids,
            strict=True,
        )
    )


def _owner(parties: tuple[PartyData, ...], user_id: int) -> PartyData:
    return next(party for party in parties if user_id in party.user_ids)


def _assert_metrics(metrics: dict[str, float | int]) -> None:
    assert metrics["num_targets"] > 0
    for name, value in metrics.items():
        if name != "num_targets":
            assert isfinite(value)
            assert 0.0 <= value <= 1.0


def test_two_round_grouped_experiment_wires_complete_lifecycle(
    processed_bundle: DatasetBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parties = _grouped_parties(processed_bundle)
    config = _config()
    built_candidates: list[EvaluationCandidateSet] = []
    evaluation_calls: list[EvaluationCandidateSet] = []
    score_snapshots: dict[
        int,
        tuple[dict[str, torch.Tensor], FedGMFLocalState],
    ] = {}
    active_evaluation: list[int | None] = [None]
    real_build = runner_module.build_evaluation_candidates
    real_evaluate = runner_module.evaluate_candidates
    real_score = runner_module.score_fedgmf

    def tracked_build(*args: Any, **kwargs: Any) -> EvaluationCandidateSet:
        candidates = real_build(*args, **kwargs)
        built_candidates.append(candidates)
        return candidates

    def tracked_evaluate(
        candidate_set: EvaluationCandidateSet,
        score_function: Any,
        *,
        k_values: list[int],
    ) -> dict[str, float | int]:
        evaluation_index = len(evaluation_calls)
        evaluation_calls.append(candidate_set)
        active_evaluation[0] = evaluation_index
        try:
            return real_evaluate(
                candidate_set,
                score_function,
                k_values=k_values,
            )
        finally:
            active_evaluation[0] = None

    def tracked_score(
        shared_state: dict[str, torch.Tensor],
        local_state: FedGMFLocalState,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        *,
        device: str,
    ) -> torch.Tensor:
        evaluation_index = active_evaluation[0]
        if evaluation_index is not None and evaluation_index not in score_snapshots:
            score_snapshots[evaluation_index] = (
                {
                    key: tensor.detach().clone()
                    for key, tensor in shared_state.items()
                },
                FedGMFLocalState(
                    user_ids=local_state.user_ids.clone(),
                    user_embeddings=local_state.user_embeddings.clone(),
                ),
            )
        return real_score(
            shared_state,
            local_state,
            user_ids,
            item_ids,
            device=device,
        )

    monkeypatch.setattr(
        runner_module,
        "build_evaluation_candidates",
        tracked_build,
    )
    monkeypatch.setattr(runner_module, "evaluate_candidates", tracked_evaluate)
    monkeypatch.setattr(runner_module, "score_fedgmf", tracked_score)

    result = run_fedgmf_hfl_experiment(
        processed_bundle,
        tuple(reversed(parties)),
        config,
    )

    assert [candidate_set.split for candidate_set in built_candidates] == [
        "valid",
        "test",
    ]
    assert evaluation_calls == [
        built_candidates[0],
        built_candidates[0],
        built_candidates[1],
    ]
    assert result.validation_candidates is built_candidates[0]
    assert result.test_candidates is built_candidates[1]
    assert len(result.round_summaries) == 2
    assert [summary.round_index for summary in result.round_summaries] == [0, 1]
    assert all(
        summary.selected_party_ids == (7, 40)
        for summary in result.round_summaries
    )

    initial_shared = initialize_fedgmf_shared_state(
        len(processed_bundle.item_id_map),
        config["model"]["embedding_dim"],
        run_seed=config["runtime"]["seed"],
    )
    first_validation_shared, first_validation_local = score_snapshots[0]
    assert any(
        not torch.equal(first_validation_shared[key], initial_shared[key])
        for key in FEDGMF_SHARED_KEYS
    )
    first_user_id = result.validation_candidates.candidates[0].user_id
    first_owner = _owner(parties, first_user_id)
    initial_local = initialize_fedgmf_local_state(
        first_owner.user_ids,
        config["model"]["embedding_dim"],
        run_seed=config["runtime"]["seed"],
        party_id=first_owner.party_id,
    )
    assert torch.equal(first_validation_local.user_ids, initial_local.user_ids)
    assert not torch.equal(
        first_validation_local.user_embeddings,
        initial_local.user_embeddings,
    )

    assert any(len(party.user_ids) > 1 for party in parties)
    assert set(result.final_local_states) == {7, 40}
    assert all(
        party.party_id not in party.user_ids for party in parties
    )
    assert tuple(result.final_shared_state) == FEDGMF_SHARED_KEYS
    assert all("user" not in key for key in result.final_shared_state)
    for summary in result.round_summaries:
        _assert_metrics(summary.validation_metrics)
    _assert_metrics(result.final_test_metrics)


def test_duplicate_user_ownership_fails(
    processed_bundle: DatasetBundle,
) -> None:
    first, second = _grouped_parties(processed_bundle)
    duplicated_user = int(first.user_ids[0])
    duplicate_users = np.unique(
        np.append(second.user_ids, duplicated_user)
    ).astype(np.int64)
    invalid_second = replace(second, user_ids=duplicate_users)

    with pytest.raises(ValueError, match="belongs to multiple HFL parties"):
        run_fedgmf_hfl_experiment(
            processed_bundle,
            (first, invalid_second),
            _config(),
        )


def test_missing_evaluation_target_owner_fails(
    processed_bundle: DatasetBundle,
) -> None:
    parties = build_hfl_partition(processed_bundle, mode="per_user")
    target_user = int(
        processed_bundle.interactions.user_ids[
            processed_bundle.splits.valid_interaction_ids[0]
        ]
    )
    incomplete_parties = tuple(
        party for party in parties if target_user not in party.user_ids
    )

    with pytest.raises(ValueError, match="valid target users.*no owning"):
        run_fedgmf_hfl_experiment(
            processed_bundle,
            incomplete_parties,
            _config(grouped_parties=None),
        )


def test_partial_participation_preserves_unselected_local_state(
    processed_bundle: DatasetBundle,
) -> None:
    parties = _grouped_parties(processed_bundle)
    config = _config(rounds=1, client_fraction=0.5)

    result = run_fedgmf_hfl_experiment(processed_bundle, parties, config)

    selected_party_id = result.round_summaries[0].selected_party_ids[0]
    unselected_party = next(
        party for party in parties if party.party_id != selected_party_id
    )
    initial_unselected = initialize_fedgmf_local_state(
        unselected_party.user_ids,
        config["model"]["embedding_dim"],
        run_seed=config["runtime"]["seed"],
        party_id=unselected_party.party_id,
    )
    final_unselected = result.final_local_states[unselected_party.party_id]
    assert torch.equal(final_unselected.user_ids, initial_unselected.user_ids)
    assert torch.equal(
        final_unselected.user_embeddings,
        initial_unselected.user_embeddings,
    )


def test_same_seed_reproduces_final_states_and_metrics(
    processed_bundle: DatasetBundle,
) -> None:
    parties = _grouped_parties(processed_bundle)
    config = _config()

    first = run_fedgmf_hfl_experiment(processed_bundle, parties, config)
    second = run_fedgmf_hfl_experiment(
        processed_bundle,
        tuple(reversed(parties)),
        config,
    )

    assert first.validation_candidates == second.validation_candidates
    assert first.test_candidates == second.test_candidates
    assert first.round_summaries == second.round_summaries
    assert first.final_test_metrics == second.final_test_metrics
    for key in FEDGMF_SHARED_KEYS:
        assert torch.equal(
            first.final_shared_state[key],
            second.final_shared_state[key],
        )
    assert set(first.final_local_states) == set(second.final_local_states)
    for party_id, first_state in first.final_local_states.items():
        second_state = second.final_local_states[party_id]
        assert torch.equal(first_state.user_ids, second_state.user_ids)
        assert torch.equal(
            first_state.user_embeddings,
            second_state.user_embeddings,
        )


def test_validation_failure_propagates(
    processed_bundle: DatasetBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_evaluation(*args: Any, **kwargs: Any) -> dict[str, float | int]:
        raise RuntimeError("validation failed")

    monkeypatch.setattr(runner_module, "evaluate_candidates", fail_evaluation)

    with pytest.raises(RuntimeError, match="validation failed"):
        run_fedgmf_hfl_experiment(
            processed_bundle,
            _grouped_parties(processed_bundle),
            _config(rounds=1),
        )
