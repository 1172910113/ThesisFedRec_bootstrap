"""Shared recommendation evaluation infrastructure."""

from src.evaluation.ranking import (
    EvaluationCandidate,
    EvaluationCandidateSet,
    build_evaluation_candidates,
    derive_evaluation_seed,
    evaluate_candidates,
    hit_rate_at_k,
    ndcg_at_k,
    rank_item_ids,
)

__all__ = [
    "EvaluationCandidate",
    "EvaluationCandidateSet",
    "build_evaluation_candidates",
    "derive_evaluation_seed",
    "evaluate_candidates",
    "hit_rate_at_k",
    "ndcg_at_k",
    "rank_item_ids",
]
