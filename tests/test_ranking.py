from math import log2

import numpy as np
import pytest

from src.data.contracts import DatasetBundle
from src.evaluation import (
    EvaluationCandidate,
    EvaluationCandidateSet,
    build_evaluation_candidates,
    evaluate_candidates,
    hit_rate_at_k,
    ndcg_at_k,
    rank_item_ids,
)


def test_hr_and_ndcg_at_rank_one() -> None:
    assert hit_rate_at_k(1, 10) == 1.0
    assert ndcg_at_k(1, 10) == 1.0


def test_hr_and_ndcg_inside_k_but_not_first() -> None:
    assert hit_rate_at_k(3, 5) == 1.0
    assert ndcg_at_k(3, 5) == pytest.approx(1.0 / log2(4))


def test_hr_and_ndcg_outside_k() -> None:
    assert hit_rate_at_k(6, 5) == 0.0
    assert ndcg_at_k(6, 5) == 0.0


def test_ranking_uses_descending_scores() -> None:
    ranked = rank_item_ids([4, 2, 8], [0.25, 0.9, -0.1])

    assert ranked == (2, 4, 8)


def test_equal_scores_use_ascending_item_id_tie_break() -> None:
    ranked = rank_item_ids([9, 3, 7], [0.5, 0.5, 0.5])

    assert ranked == (3, 7, 9)


def test_candidate_input_order_does_not_change_ranking() -> None:
    first = rank_item_ids([7, 2, 5, 3], [0.2, 0.8, 0.8, 0.1])
    second = rank_item_ids([3, 5, 7, 2], [0.1, 0.8, 0.2, 0.8])

    assert first == second == (2, 5, 7, 3)


def test_candidate_construction_includes_positive_once_and_excludes_known(
    processed_bundle: DatasetBundle,
) -> None:
    candidate_set = build_evaluation_candidates(
        processed_bundle,
        processed_bundle.splits.valid_interaction_ids,
        split="valid",
        negatives_per_positive=3,
        base_seed=37,
        excluded_splits=("train", "valid", "test"),
    )

    assert len(candidate_set.candidates) == len(
        processed_bundle.splits.valid_interaction_ids
    )
    for candidate in candidate_set.candidates:
        target_id = candidate.target_interaction_id
        assert candidate.user_id == int(
            processed_bundle.interactions.user_ids[target_id]
        )
        assert candidate.positive_item_id == int(
            processed_bundle.interactions.item_ids[target_id]
        )
        assert candidate.item_ids.count(candidate.positive_item_id) == 1
        assert len(candidate.negative_item_ids) == 3
        assert len(set(candidate.negative_item_ids)) == 3
        known_items = set(
            processed_bundle.interactions.item_ids[
                processed_bundle.interactions.user_ids == candidate.user_id
            ].tolist()
        )
        assert known_items.isdisjoint(candidate.negative_item_ids)


def test_same_seed_and_target_set_reproduce_candidates(
    processed_bundle: DatasetBundle,
) -> None:
    target_ids = processed_bundle.splits.test_interaction_ids
    first = build_evaluation_candidates(
        processed_bundle,
        target_ids,
        split="test",
        negatives_per_positive=2,
        base_seed=91,
    )
    second = build_evaluation_candidates(
        processed_bundle,
        target_ids[::-1],
        split="test",
        negatives_per_positive=2,
        base_seed=91,
    )

    assert first == second


def test_validation_and_test_use_distinct_seed_streams(
    processed_bundle: DatasetBundle,
) -> None:
    validation = build_evaluation_candidates(
        processed_bundle,
        processed_bundle.splits.valid_interaction_ids,
        split="valid",
        negatives_per_positive=2,
        base_seed=2026,
    )
    test = build_evaluation_candidates(
        processed_bundle,
        processed_bundle.splits.test_interaction_ids,
        split="test",
        negatives_per_positive=2,
        base_seed=2026,
    )

    assert validation.sampling_seed == 4052
    assert test.sampling_seed == 4053


def test_validation_and_test_targets_cannot_be_mixed(
    processed_bundle: DatasetBundle,
) -> None:
    with pytest.raises(ValueError, match="must belong to valid"):
        build_evaluation_candidates(
            processed_bundle,
            processed_bundle.splits.test_interaction_ids,
            split="valid",
            negatives_per_positive=1,
        )


def test_evaluation_averages_across_targets_for_multiple_k_values() -> None:
    candidates = EvaluationCandidateSet(
        split="valid",
        base_seed=5,
        sampling_seed=10,
        candidates=(
            EvaluationCandidate(
                target_interaction_id=17,
                user_id=5,
                positive_item_id=0,
                negative_item_ids=(1, 2),
            ),
            EvaluationCandidate(
                target_interaction_id=29,
                user_id=8,
                positive_item_id=3,
                negative_item_ids=(4, 5),
            ),
        ),
    )
    scores = {
        (5, 0): 0.9,
        (5, 1): 0.2,
        (5, 2): 0.1,
        (8, 3): 0.2,
        (8, 4): 0.9,
        (8, 5): 0.5,
    }

    result = evaluate_candidates(
        candidates,
        lambda users, items: np.asarray(
            [
                scores[(int(user), int(item))]
                for user, item in zip(users, items, strict=True)
            ]
        ),
        k_values=[1, 3],
    )

    assert result == {
        "num_targets": 2,
        "hr@1": 0.5,
        "ndcg@1": 0.5,
        "hr@3": 1.0,
        "ndcg@3": pytest.approx(0.75),
    }


def test_grouped_party_shape_does_not_introduce_party_level_weighting() -> None:
    candidates = EvaluationCandidateSet(
        split="test",
        base_seed=7,
        sampling_seed=15,
        candidates=tuple(
            EvaluationCandidate(
                target_interaction_id=user_id,
                user_id=user_id,
                positive_item_id=2 * user_id,
                negative_item_ids=(2 * user_id + 1,),
            )
            for user_id in range(3)
        ),
    )

    def score(users: np.ndarray, items: np.ndarray) -> np.ndarray:
        user_id = int(users[0])
        positive_item = 2 * user_id
        positive_score = 1.0 if user_id == 0 else 0.0
        return np.asarray(
            [positive_score if int(item) == positive_item else 0.5 for item in items]
        )

    result = evaluate_candidates(candidates, score, k_values=[1])
    target_hits = {0: 1.0, 1: 0.0, 2: 0.0}
    grouped_party_targets = ((0,), (1, 2))
    party_means = [
        sum(target_hits[target_id] for target_id in target_ids) / len(target_ids)
        for target_ids in grouped_party_targets
    ]

    assert result["num_targets"] == 3
    assert result["hr@1"] == pytest.approx(1.0 / 3.0)
    assert sum(party_means) / len(party_means) == 0.5
    assert result["hr@1"] != sum(party_means) / len(party_means)
