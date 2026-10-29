"""Frozen synthetic lifecycle expectations for the shared terminal TEST gate.

No method/runtime objects are required here. Real Unit2-fit integration lives
in test_joint_execution; these tests bind finite numeric scorers privately.
"""

from copy import copy, deepcopy
from hashlib import sha256
import json
from math import log2

import numpy as np
import pytest

from src.evaluation import terminal
from src.evaluation.ranking import EvaluationCandidate, EvaluationCandidateSet
from src.evaluation.terminal import (
    TerminalEvaluationHandle,
    _bind_terminal_evaluation,
    _candidate_digest,
    _copy_candidates,
    _validated_k_values,
    evaluate_terminal_once,
    observe_terminal_evaluation,
)


TERMINAL_ID = "a" * 64


def _candidates(split="test"):
    return EvaluationCandidateSet(
        split=split, base_seed=17, sampling_seed=35,
        candidates=(
            EvaluationCandidate(100, 7, 3, (1, 5)),
            EvaluationCandidate(101, 23, 1, (2, 4)),
        ),
    )


def _handle(*, scorer=None, validator=None, k_values=(3, 1)):
    return _bind_terminal_evaluation(
        terminal_id=TERMINAL_ID,
        score_function=(lambda users, items: items.astype(np.float64))
        if scorer is None else scorer,
        validate_candidates=(lambda candidates: None) if validator is None else validator,
        k_values=k_values,
    )


def _evaluate(handle, candidates=None, expected_terminal_id=TERMINAL_ID):
    return evaluate_terminal_once(
        handle, _candidates() if candidates is None else candidates,
        expected_terminal_id=expected_terminal_id,
    )


def _ready(handle):
    assert observe_terminal_evaluation(handle) == {
        "terminal_id": TERMINAL_ID, "state": "ready", "attempt_count": 0,
        "candidate_digest": None, "metrics": None,
        "error_type": None, "error_message": None,
    }


def _altered_candidate(field, value):
    candidates = _candidates()
    object.__setattr__(candidates.candidates[0], field, value)
    return candidates


def _altered_set(field, value):
    candidates = _candidates()
    object.__setattr__(candidates, field, value)
    return candidates


def test_handle_has_no_public_construction_or_copy_reset():
    with pytest.raises(TypeError, match="prepared"):
        TerminalEvaluationHandle()
    with pytest.raises(TypeError):
        TerminalEvaluationHandle(terminal_id=TERMINAL_ID, score_function=lambda *args: 0)
    handle = _handle()
    for copier in (copy, deepcopy):
        with pytest.raises(TypeError, match="cannot be copied"):
            copier(handle)
    with pytest.raises(ValueError, match="not prepared"):
        observe_terminal_evaluation(object.__new__(TerminalEvaluationHandle))


@pytest.mark.parametrize("invalid", [None, {}, object(), 1, True])
def test_only_prepared_handle_is_accepted(invalid):
    with pytest.raises(TypeError):
        _evaluate(invalid)
    with pytest.raises(TypeError):
        observe_terminal_evaluation(invalid)


@pytest.mark.parametrize("invalid,error", [
    (None, TypeError), (1, TypeError), ("", ValueError), ("a" * 63, ValueError),
    ("A" * 64, ValueError), ("g" * 64, ValueError),
])
def test_private_binding_rejects_invalid_identity(invalid, error):
    with pytest.raises(error):
        _bind_terminal_evaluation(
            terminal_id=invalid, score_function=lambda *args: [],
            validate_candidates=lambda candidates: None, k_values=(1,),
        )


@pytest.mark.parametrize("name", ["score_function", "validate_candidates"])
def test_private_binding_requires_callable_seams(name):
    kwargs = dict(
        terminal_id=TERMINAL_ID, score_function=lambda *args: [],
        validate_candidates=lambda candidates: None, k_values=(1,),
    )
    kwargs[name] = None
    with pytest.raises(TypeError):
        _bind_terminal_evaluation(**kwargs)


@pytest.mark.parametrize("invalid,error", [
    ([], TypeError), ([1], TypeError), (None, TypeError),
    ((True,), TypeError), ((np.int64(1),), TypeError), ((1.0,), TypeError),
    ((), ValueError), ((0,), ValueError), ((-1,), ValueError), ((1, 1), ValueError),
])
def test_k_validation_is_exact_and_duplicate_sensitive(invalid, error):
    with pytest.raises(error):
        _validated_k_values(invalid)
    with pytest.raises(error):
        _handle(k_values=invalid)


def test_success_uses_shared_ranking_fixed_sorted_k_and_independent_observations():
    calls = []
    validations = []

    def score(users, items):
        calls.append((users.copy(), items.copy()))
        return items.astype(np.float64)

    candidates = _candidates()
    handle = _handle(scorer=score, validator=validations.append)
    _ready(handle)
    before = observe_terminal_evaluation(handle)
    before["state"] = "complete"
    before["attempt_count"] = 10
    _ready(handle)
    result = _evaluate(handle, candidates)
    assert list(result) == ["num_targets", "hr@1", "ndcg@1", "hr@3", "ndcg@3"]
    assert result == {
        "num_targets": 2, "hr@1": 0.0, "ndcg@1": 0.0, "hr@3": 1.0,
        "ndcg@3": (1 / log2(3) + 1 / log2(4)) / 2,
    }
    assert len(validations) == 1
    assert validations[0] == candidates and validations[0] is not candidates
    assert validations[0].candidates[0] is not candidates.candidates[0]
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], np.array([7, 7, 7], dtype=np.int64))
    np.testing.assert_array_equal(calls[0][1], np.array([3, 1, 5], dtype=np.int64))
    np.testing.assert_array_equal(calls[1][0], np.array([23, 23, 23], dtype=np.int64))
    np.testing.assert_array_equal(calls[1][1], np.array([1, 2, 4], dtype=np.int64))
    receipt = observe_terminal_evaluation(handle)
    assert receipt["state"] == "complete" and receipt["attempt_count"] == 1
    assert receipt["candidate_digest"] == _candidate_digest(candidates)
    expected = dict(result)
    result["hr@1"] = -10
    receipt["metrics"]["hr@1"] = -20
    receipt["state"] = "ready"
    receipt["attempt_count"] = 0
    assert observe_terminal_evaluation(handle)["metrics"] == expected
    with pytest.raises(ValueError, match="consumed"):
        _evaluate(handle)
    assert len(calls) == 2


def test_k_above_candidate_count_keeps_existing_evaluator_semantics():
    candidates = EvaluationCandidateSet(
        "test", 0, 1, (EvaluationCandidate(0, 0, 0, ()),),
    )
    assert _evaluate(_handle(k_values=(10,)), candidates) == {
        "num_targets": 1, "hr@10": 1.0, "ndcg@10": 1.0,
    }


@pytest.mark.parametrize("identity,error", [
    ("b" * 64, ValueError), ("A" * 64, ValueError), (17, TypeError),
])
def test_bad_expected_identity_does_not_validate_score_or_consume(identity, error):
    calls = []
    handle = _handle(scorer=lambda *args: calls.append("score"), validator=calls.append)
    with pytest.raises(error):
        _evaluate(handle, expected_terminal_id=identity)
    assert calls == []
    _ready(handle)


@pytest.mark.parametrize("factory,error", [
    (lambda: {}, TypeError),
    (lambda: _candidates("valid"), ValueError),
    (lambda: _altered_set("candidates", ()), ValueError),
    (lambda: _altered_set("candidates", list(_candidates().candidates)), TypeError),
    (lambda: _altered_set("candidates", (object(),)), TypeError),
    (lambda: _altered_set("candidates", (_candidates().candidates[0],) * 2), ValueError),
    (lambda: _altered_set("split", 1), TypeError),
    (lambda: _altered_set("base_seed", True), TypeError),
    (lambda: _altered_set("sampling_seed", np.int64(1)), TypeError),
    (lambda: _altered_set("base_seed", -1), ValueError),
    (lambda: _altered_candidate("target_interaction_id", True), TypeError),
    (lambda: _altered_candidate("target_interaction_id", -1), ValueError),
    (lambda: _altered_candidate("user_id", np.int64(7)), TypeError),
    (lambda: _altered_candidate("user_id", -1), ValueError),
    (lambda: _altered_candidate("user_id", 2**63), ValueError),
    (lambda: _altered_candidate("positive_item_id", 2**63), ValueError),
    (lambda: _altered_candidate("positive_item_id", 3.0), TypeError),
    (lambda: _altered_candidate("negative_item_ids", [1, 5]), TypeError),
    (lambda: _altered_candidate("negative_item_ids", (1, 1)), ValueError),
    (lambda: _altered_candidate("negative_item_ids", (3, 5)), ValueError),
    (lambda: _altered_candidate("negative_item_ids", (-1,)), ValueError),
    (lambda: _altered_candidate("negative_item_ids", (True,)), TypeError),
    (lambda: _altered_candidate("negative_item_ids", (2**63,)), ValueError),
])
def test_structural_candidate_failure_is_prestart_and_correctable(factory, error):
    calls = []
    handle = _handle(validator=calls.append)
    with pytest.raises(error):
        _evaluate(handle, factory())
    assert calls == []
    _ready(handle)
    assert _evaluate(handle)["num_targets"] == 2


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_method_preflight_failure_releases_ready_guard_without_scoring(error_type):
    attempts = []
    scores = []
    original = error_type("preflight unavailable")

    def validator(candidates):
        attempts.append(candidates)
        if len(attempts) == 1:
            raise original

    def score(users, items):
        scores.append(items.copy())
        return items.astype(np.float64)

    handle = _handle(scorer=score, validator=validator)
    with pytest.raises(error_type) as caught:
        _evaluate(handle)
    assert caught.value is original and scores == []
    _ready(handle)
    _evaluate(handle)
    assert len(attempts) == 2 and len(scores) == 2


def test_validator_reentry_is_rejected_while_ready_and_allows_later_correction():
    calls = []
    first = True

    def validator(candidates):
        nonlocal first
        if first:
            first = False
            assert observe_terminal_evaluation(handle)["state"] == "ready"
            _evaluate(handle, candidates)

    def score(users, items):
        calls.append(items.copy())
        return items.astype(np.float64)

    handle = _handle(scorer=score, validator=validator)
    with pytest.raises(ValueError, match="in progress"):
        _evaluate(handle)
    assert calls == []
    _ready(handle)
    _evaluate(handle)
    assert len(calls) == 2


def test_caught_validator_reentry_does_not_create_a_second_opportunity():
    attempts = []

    def validator(candidates):
        with pytest.raises(ValueError, match="in progress"):
            _evaluate(handle, candidates)
        attempts.append("inner rejected")

    handle = _handle(validator=validator)
    _evaluate(handle)
    assert attempts == ["inner rejected"]
    assert observe_terminal_evaluation(handle)["attempt_count"] == 1


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_started_scorer_error_or_interrupt_consumes_and_preserves_primary(error_type):
    calls = []
    original = error_type("actual scoring failure")

    def score(users, items):
        calls.append((users, items))
        during = observe_terminal_evaluation(handle)
        assert during["state"] == "started" and during["attempt_count"] == 1
        raise original

    handle = _handle(scorer=score)
    with pytest.raises(error_type) as caught:
        _evaluate(handle)
    assert caught.value is original
    receipt = observe_terminal_evaluation(handle)
    assert receipt["state"] == "failed" and receipt["attempt_count"] == 1
    assert receipt["error_type"] == error_type.__name__
    assert receipt["error_message"] == "actual scoring failure"
    assert receipt["candidate_digest"] == _candidate_digest(_candidates())
    assert receipt["metrics"] is None
    with pytest.raises(ValueError, match="consumed"):
        _evaluate(handle)
    assert len(calls) == 1


def test_started_reentry_fails_outer_call_and_consumes():
    def score(users, items):
        _evaluate(handle)

    handle = _handle(scorer=score)
    with pytest.raises(ValueError, match="in progress"):
        _evaluate(handle)
    assert observe_terminal_evaluation(handle)["state"] == "failed"
    with pytest.raises(ValueError, match="consumed"):
        _evaluate(handle)


def test_surviving_started_state_rejects_unknown_completion_without_scoring():
    calls = []
    handle = _handle(scorer=lambda *args: calls.append(args))
    # Simulate a surviving interrupted private state; no production reset API.
    handle._receipt.update(state="started", attempt_count=1)
    with pytest.raises(ValueError, match="consumed"):
        _evaluate(handle)
    assert calls == []
    assert observe_terminal_evaluation(handle)["state"] == "started"


@pytest.mark.parametrize("scores", [np.array([np.nan] * 3), np.array([1.0]), np.ones((3, 1))])
def test_shared_evaluator_validation_failure_consumes(scores):
    handle = _handle(scorer=lambda users, items: scores)
    with pytest.raises(ValueError):
        _evaluate(handle)
    assert observe_terminal_evaluation(handle)["state"] == "failed"
    assert observe_terminal_evaluation(handle)["attempt_count"] == 1


def test_evaluator_interrupt_consumes_even_before_first_score(monkeypatch):
    calls = []

    def evaluator(*args, **kwargs):
        raise KeyboardInterrupt("evaluator interrupted")

    handle = _handle(scorer=lambda *args: calls.append(args))
    monkeypatch.setattr(terminal, "evaluate_candidates", evaluator)
    with pytest.raises(KeyboardInterrupt, match="evaluator interrupted"):
        _evaluate(handle)
    assert calls == []
    assert observe_terminal_evaluation(handle)["state"] == "failed"


def test_error_message_failure_does_not_replace_primary_exception():
    class UnrenderableError(Exception):
        def __str__(self):
            raise RuntimeError("cannot render")

    original = UnrenderableError()

    def score(users, items):
        raise original

    handle = _handle(scorer=score)
    with pytest.raises(UnrenderableError) as caught:
        _evaluate(handle)
    assert caught.value is original
    assert observe_terminal_evaluation(handle)["error_message"] == "<error message unavailable>"


def test_candidate_digest_has_explicit_ordered_versioned_record():
    record = {
        "format": "thesisfedrec.terminal_candidates", "version": 1,
        "split": "test", "base_seed": 17, "sampling_seed": 35,
        "candidates": [
            {"target_interaction_id": 100, "user_id": 7,
             "positive_item_id": 3, "negative_item_ids": [1, 5]},
            {"target_interaction_id": 101, "user_id": 23,
             "positive_item_id": 1, "negative_item_ids": [2, 4]},
        ],
    }
    expected = sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    candidates = _candidates()
    assert _candidate_digest(candidates) == expected
    assert _candidate_digest(_copy_candidates(candidates, split="test")) == expected
    for changed in (
        _candidates("valid"),
        _altered_set("sampling_seed", 36),
        _altered_set("candidates", tuple(reversed(candidates.candidates))),
        _altered_candidate("negative_item_ids", (5, 1)),
    ):
        assert _candidate_digest(changed) != expected


def test_candidate_copy_is_structural_without_route_or_runtime_assumptions():
    candidates = _candidates()
    copied = _copy_candidates(candidates, split="test")
    assert copied == candidates and copied is not candidates
    assert copied.candidates is not candidates.candidates
    # Deliberate mutation only simulates a stale external DTO, not a supported
    # public mutation interface. The already-owned snapshot remains independent.
    object.__setattr__(candidates.candidates[0], "user_id", 1000)
    assert copied.candidates[0].user_id == 7
    assert _copy_candidates(_candidates("valid"), split="valid").split == "valid"
