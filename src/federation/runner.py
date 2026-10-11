"""In-memory multi-round FedGMF experiment execution for HFL."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.data.contracts import DatasetBundle, PartyData
from src.evaluation.ranking import (
    EvaluationCandidateSet,
    build_evaluation_candidates,
    evaluate_candidates,
)
from src.federation.fedavg import ClientUpdate
from src.federation.hfl import run_synchronous_hfl_round
from src.models.fedgmf import (
    FedGMFLocalState,
    initialize_fedgmf_local_state,
    initialize_fedgmf_shared_state,
    score_fedgmf,
    train_local_fedgmf,
)
from src.utils.seed import set_random_seed

MetricValue = float | int


@dataclass(frozen=True, slots=True, kw_only=True)
class FedGMFHFLRoundSummary:
    """In-memory monitoring values from one completed communication round."""

    round_index: int
    selected_party_ids: tuple[int, ...]
    total_positive_interactions: int
    validation_metrics: dict[str, MetricValue]


@dataclass(frozen=True, slots=True, kw_only=True)
class FedGMFHFLRunResult:
    """Final state and fixed evaluation inputs from one in-memory run."""

    final_shared_state: dict[str, torch.Tensor]
    final_local_states: dict[int, FedGMFLocalState]
    validation_candidates: EvaluationCandidateSet
    test_candidates: EvaluationCandidateSet
    round_summaries: tuple[FedGMFHFLRoundSummary, ...]
    final_test_metrics: dict[str, MetricValue]


def run_fedgmf_hfl_experiment(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
    config: Mapping[str, Any],
) -> FedGMFHFLRunResult:
    """Run the validated Phase 3 FedGMF/HFL configuration in memory."""
    ordered_parties, user_to_party = _hfl_user_ownership(bundle, parties)
    federation = config["federation"]
    model = config["model"]
    optimization = config["optimization"]
    evaluation = config["evaluation"]
    runtime = config["runtime"]

    run_seed = runtime["seed"]
    embedding_dim = model["embedding_dim"]
    device = runtime["device"]
    set_random_seed(run_seed)

    shared_state = initialize_fedgmf_shared_state(
        num_items=len(bundle.item_id_map),
        embedding_dim=embedding_dim,
        run_seed=run_seed,
    )
    local_states = {
        party.party_id: initialize_fedgmf_local_state(
            party.user_ids,
            embedding_dim,
            run_seed=run_seed,
            party_id=party.party_id,
        )
        for party in ordered_parties
    }

    candidate_arguments = {
        "negatives_per_positive": evaluation["negatives_per_positive"],
        "strategy": "uniform",
        "base_seed": run_seed,
        "excluded_splits": tuple(evaluation["excluded_splits"]),
    }
    validation_candidates = build_evaluation_candidates(
        bundle,
        bundle.splits.valid_interaction_ids,
        split="valid",
        **candidate_arguments,
    )
    test_candidates = build_evaluation_candidates(
        bundle,
        bundle.splits.test_interaction_ids,
        split="test",
        **candidate_arguments,
    )

    round_summaries: list[FedGMFHFLRoundSummary] = []
    for round_index in range(federation["rounds"]):

        def local_update(
            round_shared_state: Mapping[str, torch.Tensor],
            local_state: FedGMFLocalState,
            party: PartyData,
        ) -> tuple[FedGMFLocalState, ClientUpdate]:
            return train_local_fedgmf(
                round_shared_state,
                local_state,
                bundle,
                party,
                round_index=round_index,
                local_epochs=federation["local_epochs"],
                learning_rate=optimization["learning_rate"],
                batch_size=optimization["batch_size"],
                weight_decay=optimization["weight_decay"],
                negatives_per_positive=optimization[
                    "train_negatives_per_positive"
                ],
                negative_strategy=optimization["train_negative_strategy"],
                excluded_splits=tuple(
                    optimization["train_negative_excluded_splits"]
                ),
                run_seed=run_seed,
                device=device,
            )

        round_result = run_synchronous_hfl_round(
            shared_state,
            ordered_parties,
            local_states,
            client_fraction=federation["client_fraction"],
            run_seed=run_seed,
            round_index=round_index,
            local_update=local_update,
        )
        shared_state = round_result.next_shared_state
        local_states = round_result.local_states
        validation_metrics = evaluate_candidates(
            validation_candidates,
            _fedgmf_score_function(
                shared_state,
                local_states,
                user_to_party,
                device=device,
            ),
            k_values=evaluation["k"],
        )
        round_summaries.append(
            FedGMFHFLRoundSummary(
                round_index=round_index,
                selected_party_ids=round_result.selected_party_ids,
                total_positive_interactions=(
                    round_result.total_positive_interactions
                ),
                validation_metrics=validation_metrics,
            )
        )

    final_test_metrics = evaluate_candidates(
        test_candidates,
        _fedgmf_score_function(
            shared_state,
            local_states,
            user_to_party,
            device=device,
        ),
        k_values=evaluation["k"],
    )
    return FedGMFHFLRunResult(
        final_shared_state=shared_state,
        final_local_states=local_states,
        validation_candidates=validation_candidates,
        test_candidates=test_candidates,
        round_summaries=tuple(round_summaries),
        final_test_metrics=final_test_metrics,
    )


def _hfl_user_ownership(
    bundle: DatasetBundle,
    parties: Sequence[PartyData],
) -> tuple[tuple[PartyData, ...], dict[int, int]]:
    parties_by_id: dict[int, PartyData] = {}
    user_to_party: dict[int, int] = {}
    for party in parties:
        if not isinstance(party, PartyData):
            raise TypeError("parties must contain PartyData values")
        if party.party_id in parties_by_id:
            raise ValueError("parties must not contain duplicate party IDs")
        parties_by_id[party.party_id] = party
        for user_id in party.user_ids.tolist():
            if not 0 <= user_id < len(bundle.user_id_map):
                raise ValueError(
                    f"party {party.party_id} contains out-of-range user ID "
                    f"{user_id}"
                )
            if user_id in user_to_party:
                raise ValueError(
                    f"user {user_id} belongs to multiple HFL parties"
                )
            user_to_party[user_id] = party.party_id

    if not parties_by_id:
        raise ValueError("parties must not be empty")
    for split in ("valid", "test"):
        interaction_ids = getattr(
            bundle.splits,
            f"{split}_interaction_ids",
        )
        target_users = np.unique(bundle.interactions.user_ids[interaction_ids])
        missing_users = [
            int(user_id)
            for user_id in target_users
            if int(user_id) not in user_to_party
        ]
        if missing_users:
            raise ValueError(
                f"{split} target users have no owning HFL party: "
                f"{missing_users}"
            )

    ordered_parties = tuple(
        parties_by_id[party_id] for party_id in sorted(parties_by_id)
    )
    return ordered_parties, user_to_party


def _fedgmf_score_function(
    shared_state: Mapping[str, torch.Tensor],
    local_states: Mapping[int, FedGMFLocalState],
    user_to_party: Mapping[int, int],
    *,
    device: str | torch.device,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def score(user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        users = np.asarray(user_ids)
        items = np.asarray(item_ids)
        if users.ndim != 1 or items.ndim != 1 or users.shape != items.shape:
            raise ValueError("user_ids and item_ids must be aligned 1D arrays")

        party_ids = np.empty(users.shape, dtype=np.int64)
        for index, user_id in enumerate(users.tolist()):
            owner = user_to_party.get(int(user_id))
            if owner is None:
                raise ValueError(
                    f"user {int(user_id)} has no owning HFL party"
                )
            party_ids[index] = owner

        scores = np.empty(users.shape, dtype=np.float64)
        for party_id in np.unique(party_ids):
            rows = np.flatnonzero(party_ids == party_id)
            local_state = local_states.get(int(party_id))
            if local_state is None:
                raise ValueError(f"party {int(party_id)} has no local state")
            logits = score_fedgmf(
                shared_state,
                local_state,
                users[rows],
                items[rows],
                device=device,
            )
            scores[rows] = logits.numpy()
        return scores

    return score
