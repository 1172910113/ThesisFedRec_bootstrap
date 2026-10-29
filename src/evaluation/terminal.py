"""One in-memory terminal TEST opportunity over a privately bound scorer.

This is trusted cooperative framework bookkeeping, not a privacy boundary or
durable exactly-once protocol. Candidate provenance belongs to the outer
coordinator; method-specific route/input checks belong to the bound validator.
"""

from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
import json

from src.evaluation.ranking import (
    EvaluationCandidate,
    EvaluationCandidateSet,
    ScoreFunction,
    evaluate_candidates,
)


_PREPARED = object()


class TerminalEvaluationHandle:
    """Prepare-only handle; successful fit finalization privately binds it.

    Observations never expose the scorer, validator, or mutable receipt.
    Copying a handle is not a way to obtain another evaluation opportunity.
    Concurrent calls and deliberate private-attribute bypasses are unsupported.
    """

    __slots__ = (
        "_prepared", "_score_function", "_validate_candidates", "_k_values",
        "_receipt", "_in_call",
    )

    def __new__(cls, *args: object, **kwargs: object) -> "TerminalEvaluationHandle":
        raise TypeError("TerminalEvaluationHandle is prepared by a successful fit")

    def __copy__(self) -> "TerminalEvaluationHandle":
        raise TypeError("terminal evaluation handles cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> "TerminalEvaluationHandle":
        raise TypeError("terminal evaluation handles cannot be copied")


def _validated_k_values(k_values: tuple[int, ...]) -> tuple[int, ...]:
    if type(k_values) is not tuple:
        raise TypeError("k_values must be an exact tuple")
    if any(type(k) is not int for k in k_values):
        raise TypeError("k_values must contain exact built-in integers")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must be nonempty and positive")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must be unique")
    return tuple(sorted(k_values))


def _nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _copy_candidates(
    candidate_set: EvaluationCandidateSet, *, split: str,
) -> EvaluationCandidateSet:
    """Revalidate and own immutable candidate DTO contents, without scoring."""
    if type(candidate_set) is not EvaluationCandidateSet:
        raise TypeError("candidate_set must be an EvaluationCandidateSet")
    if type(split) is not str or type(candidate_set.split) is not str:
        raise TypeError("candidate split must be an exact built-in string")
    if split not in ("valid", "test") or candidate_set.split != split:
        raise ValueError(f"candidate split must be {split!r}")
    _nonnegative_int(candidate_set.base_seed, "base_seed")
    _nonnegative_int(candidate_set.sampling_seed, "sampling_seed")
    if type(candidate_set.candidates) is not tuple:
        raise TypeError("candidates must be an exact tuple")
    if not candidate_set.candidates:
        raise ValueError("at least one evaluation candidate is required")
    copied = []
    for candidate in candidate_set.candidates:
        if type(candidate) is not EvaluationCandidate:
            raise TypeError("each candidate must be an EvaluationCandidate")
        for name in ("target_interaction_id", "user_id", "positive_item_id"):
            _nonnegative_int(getattr(candidate, name), name)
        if type(candidate.negative_item_ids) is not tuple:
            raise TypeError("negative_item_ids must be an exact tuple")
        for item_id in candidate.negative_item_ids:
            _nonnegative_int(item_id, "negative item ID")
        # The shared evaluator creates native int64 arrays for these two IDs.
        # Reject unsupported values before consuming the terminal opportunity.
        if candidate.user_id >= 2**63 or any(
            item_id >= 2**63 for item_id in candidate.item_ids
        ):
            raise ValueError("scored user/item IDs must fit nonnegative int64")
        copied.append(EvaluationCandidate(
            target_interaction_id=candidate.target_interaction_id,
            user_id=candidate.user_id,
            positive_item_id=candidate.positive_item_id,
            negative_item_ids=tuple(item_id for item_id in candidate.negative_item_ids),
        ))
    return EvaluationCandidateSet(
        split=split,
        base_seed=candidate_set.base_seed,
        sampling_seed=candidate_set.sampling_seed,
        candidates=tuple(copied),
    )


def _candidate_digest(candidate_set: EvaluationCandidateSet) -> str:
    """Bind ordered, previously validated candidate contents and seed fields."""
    payload = {
        "format": "thesisfedrec.terminal_candidates", "version": 1,
        "split": candidate_set.split,
        "base_seed": candidate_set.base_seed,
        "sampling_seed": candidate_set.sampling_seed,
        "candidates": [
            {"target_interaction_id": candidate.target_interaction_id,
             "user_id": candidate.user_id,
             "positive_item_id": candidate.positive_item_id,
             "negative_item_ids": candidate.negative_item_ids}
            for candidate in candidate_set.candidates
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _terminal_id(value: str) -> None:
    if type(value) is not str:
        raise TypeError("terminal_id must be an exact built-in string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("terminal_id must be a lowercase SHA-256 hex identity")


def _bind_terminal_evaluation(
    *, terminal_id: str,
    score_function: ScoreFunction,
    validate_candidates: Callable[[EvaluationCandidateSet], None],
    k_values: tuple[int, ...],
) -> TerminalEvaluationHandle:
    """Internal fit-finalization seam, not a public arbitrary-scorer API."""
    _terminal_id(terminal_id)
    if not callable(score_function) or not callable(validate_candidates):
        raise TypeError("the private scorer and validator must be callable")
    ks = _validated_k_values(k_values)
    handle = object.__new__(TerminalEvaluationHandle)
    handle._prepared = _PREPARED
    handle._score_function = score_function
    handle._validate_candidates = validate_candidates
    handle._k_values = ks
    handle._in_call = False
    handle._receipt = {
        "terminal_id": terminal_id, "state": "ready", "attempt_count": 0,
        "candidate_digest": None, "metrics": None,
        "error_type": None, "error_message": None,
    }
    return handle


def _prepared_handle(handle: TerminalEvaluationHandle) -> TerminalEvaluationHandle:
    if type(handle) is not TerminalEvaluationHandle:
        raise TypeError("handle must be a prepared TerminalEvaluationHandle")
    if getattr(handle, "_prepared", None) is not _PREPARED:
        raise ValueError("terminal evaluation handle is not prepared")
    return handle


def observe_terminal_evaluation(handle: TerminalEvaluationHandle) -> dict[str, object]:
    """Return an independent receipt observation, never mutable authority."""
    return deepcopy(_prepared_handle(handle)._receipt)


def evaluate_terminal_once(
    handle: TerminalEvaluationHandle,
    candidate_set: EvaluationCandidateSet,
    *, expected_terminal_id: str,
) -> dict[str, float | int]:
    """Validate TEST without scoring, then consume one bound evaluation call.

    Pre-start failures (including validator interrupts) leave ready/attempt 0.
    The validation guard also rejects reentry while the receipt is still ready.
    Every BaseException after started consumes the opportunity and propagates.
    """
    handle = _prepared_handle(handle)
    receipt = handle._receipt
    if handle._in_call:
        raise ValueError("terminal evaluation is already in progress")
    if receipt["state"] != "ready":
        raise ValueError("terminal evaluation opportunity has already been consumed")
    _terminal_id(expected_terminal_id)
    if expected_terminal_id != receipt["terminal_id"]:
        raise ValueError("expected_terminal_id does not match the bound terminal")
    handle._in_call = True
    try:
        candidates = _copy_candidates(candidate_set, split="test")
        handle._validate_candidates(candidates)
        digest = _candidate_digest(candidates)
        receipt.update(state="started", attempt_count=1, candidate_digest=digest)
        try:
            metrics = evaluate_candidates(
                candidates, handle._score_function, k_values=handle._k_values,
            )
            receipt.update(state="complete", metrics=dict(metrics))
            return dict(metrics)
        except BaseException as error:
            # Error rendering must not replace the original scorer/evaluator error.
            try:
                message = str(error)
            except BaseException:
                message = "<error message unavailable>"
            receipt.update(
                state="failed", error_type=type(error).__name__, error_message=message,
            )
            raise
    finally:
        handle._in_call = False
