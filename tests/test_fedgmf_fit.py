from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import src.federation.runner as runner
from src.data.contracts import DatasetBundle, InteractionSplits, PartyData
from src.data.partition import build_hfl_partition
from src.utils.config import load_yaml, validate_phase3_experiment_config

_REFERENCE_CONFIG = (
    Path(__file__).parents[1] / "configs/experiments/ml1m_fedgmf_fedavg.yaml"
)
_CASES = (
    ("grouped_full", 37, 2, 1.0, 1, "grouped_round_robin"),
    ("grouped_partial", 2026, 3, 0.5, 2, "grouped_round_robin"),
    ("per_user_partial", 37, 3, 0.5, 1, "per_user"),
)

# Captured through the actual accepted runner BEFORE production edits, at
# 652347431d10f001bb69dd5a5db39294cb9ad197 (Python 3.11.16, NumPy 1.26.4,
# PyTorch 2.7.1+cu126, CPU). These are retained outputs, not a second engine.
_BASELINE = {
    "grouped_full": {
        "shared": "5c444e5d25f990c00f25634d547ffca86e25342adbee28b934bceb5e350e58d4",
        "local": "b567b8ce12e410a7807c2c13ab5e0cb43416b33121547152a8aaceb3c8b8578e",
        "validation_candidates": "2e824aba2b98bf0554f89bb8d1f19bd1a90d0728e030a36a52a581d975b277bd",
        "test_candidates": "764a1492b274315b982039064f6fd13fe47070b5592d7058b01a11c1cd413bda",
        "round_summaries": "ed2228c3154c02ccfcd75e72b4c144441a8de78808a81772ee081dfcbe1e0d4b",
        "final_test_metrics": "aa507e4d3afae1249ff2b39d23356cceee9b47282995894378bdbc284db3024c",
    },
    "grouped_partial": {
        "shared": "5c8ec6087d0e8ced033434a31e6286294dae2beb3f7e4a4a67af3c7c393d90bc",
        "local": "c55b681f09cba78fe702d8538172b4151f9eb5b015e514da1975b9c0e3600dc7",
        "validation_candidates": "c0a20c47039c28909830ab1161ea2a1e3eb5009240c0fd7ff04467749a417705",
        "test_candidates": "e960c6a1c7e128fd033d181b9d23362f97844a03d2d708b1e66cb7d92141bacf",
        "round_summaries": "0fec331afedbe32b96ae13f676008f99efb7e60c2b6092d8b807aca00f9d2fc9",
        "final_test_metrics": "c6382fa1803a543d437b08390cd441cea22368cfa100162fbf0993a644fa4640",
    },
    "per_user_partial": {
        "shared": "e3c9cb5d2bd84ce978fea8d893b6fdf9de4b8d57ed551492add5d026fb62176f",
        "local": "febe113876f924badfc2074663f0021bd2bb0d0162d8584222dd9edf334c2572",
        "validation_candidates": "2e824aba2b98bf0554f89bb8d1f19bd1a90d0728e030a36a52a581d975b277bd",
        "test_candidates": "764a1492b274315b982039064f6fd13fe47070b5592d7058b01a11c1cd413bda",
        "round_summaries": "02ea52b206e9958889a10bb05dd8c1641683a8714d66c589598557754d250744",
        "final_test_metrics": "f2a2ec60aced2c785d5de4c52d65feee4d2d8b45da82b768a92603d0ff6b7d17",
    },
}


def _inputs(bundle: DatasetBundle, case: tuple) -> tuple:
    _, seed, rounds, fraction, epochs, mode = case
    config = load_yaml(_REFERENCE_CONFIG)
    config["federation"].update(
        partition=mode, rounds=rounds, client_fraction=fraction,
        local_epochs=epochs,
    )
    if mode == "grouped_round_robin":
        config["federation"]["num_parties"] = 2
    else:
        config["federation"].pop("num_parties", None)
    config["model"]["embedding_dim"] = 3
    config["optimization"].update(
        learning_rate=0.02, batch_size=4, train_negatives_per_positive=1,
    )
    config["evaluation"].update(k=[1, 3], negatives_per_positive=2)
    config["runtime"].update(seed=seed, device="cpu")
    validate_phase3_experiment_config(config)
    parties = build_hfl_partition(
        bundle, mode=mode,
        num_parties=2 if mode == "grouped_round_robin" else None, seed=11,
    )
    ids = (40, 7) if mode == "grouped_round_robin" else (90, 7, 40, 11)
    return config, tuple(
        replace(party, party_id=party_id)
        for party, party_id in zip(parties, ids, strict=True)
    )


def _tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes": tensor.detach().cpu().contiguous().numpy().tobytes().hex(),
    }


def _snapshot(result: runner.FedGMFHFLRunResult) -> dict[str, str]:
    # Hash exact tensor bytes AND names/order/IDs/shapes/dtypes, not rounded
    # floats or a tolerance. Separate digests identify which contract drifted.
    sections = {
        "shared": [(key, _tensor_identity(value))
                   for key, value in result.final_shared_state.items()],
        "local": [(party_id, _tensor_identity(state.user_ids),
                   _tensor_identity(state.user_embeddings))
                  for party_id, state in result.final_local_states.items()],
        "validation_candidates": asdict(result.validation_candidates),
        "test_candidates": asdict(result.test_candidates),
        "round_summaries": [asdict(summary) for summary in result.round_summaries],
        "final_test_metrics": result.final_test_metrics,
    }
    return {
        name: hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        for name, value in sections.items()
    }


@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
def test_fit_and_legacy_runner_match_accepted_baseline(processed_bundle, case) -> None:
    config, parties = _inputs(processed_bundle, case)
    before = deepcopy((processed_bundle, parties, config))
    result = runner.run_fedgmf_hfl_experiment(
        processed_bundle, tuple(reversed(parties)), config
    )
    assert _snapshot(result) == _BASELINE[case[0]]
    _assert_inputs_unchanged(processed_bundle, parties, config, before)

    fitted = runner.fit_fedgmf_hfl(processed_bundle, parties, config)
    assert isinstance(fitted, runner.FedGMFHFLFitResult)
    assert list(fitted.final_shared_state) == list(result.final_shared_state)
    for key, tensor in result.final_shared_state.items():
        assert torch.equal(tensor, fitted.final_shared_state[key])
    assert list(fitted.final_local_states) == list(result.final_local_states)
    for party_id, local in result.final_local_states.items():
        assert torch.equal(local.user_ids, fitted.final_local_states[party_id].user_ids)
        assert torch.equal(
            local.user_embeddings,
            fitted.final_local_states[party_id].user_embeddings,
        )
    assert fitted.validation_candidates == result.validation_candidates
    assert fitted.round_summaries == result.round_summaries
    _assert_inputs_unchanged(processed_bundle, parties, config, before)


def _assert_inputs_unchanged(bundle, parties, config, before) -> None:
    old_bundle, old_parties, old_config = before
    np.testing.assert_equal(asdict(bundle), asdict(old_bundle))
    assert len(parties) == len(old_parties)
    for party, old_party in zip(parties, old_parties, strict=True):
        np.testing.assert_equal(asdict(party), asdict(old_party))
    assert config == old_config


def test_fit_only_builds_and_scores_validation_candidates(
    processed_bundle, monkeypatch
) -> None:
    config, parties = _inputs(processed_bundle, _CASES[1])
    real_build = runner.build_evaluation_candidates
    real_evaluate = runner.evaluate_candidates
    real_score = runner.score_fedgmf
    real_train = runner.train_local_fedgmf
    built, evaluations, scores, local_updates = [], [], [], []
    active_candidates = []

    def guarded_build(bundle, targets, *, split, **kwargs):
        assert split == "valid", "fit must not construct test candidates"
        np.testing.assert_array_equal(targets, bundle.splits.valid_interaction_ids)
        assert kwargs["excluded_splits"] == ("train", "valid", "test")
        candidates = real_build(bundle, targets, split=split, **kwargs)
        built.append(candidates)
        return candidates

    def guarded_evaluate(candidates, score_function, *, k_values):
        assert candidates.split == "valid", "fit must not evaluate test targets"
        assert candidates is built[0]
        evaluations.append(candidates)
        active_candidates.append(candidates)
        try:
            return real_evaluate(candidates, score_function, k_values=k_values)
        finally:
            active_candidates.pop()

    def guarded_score(shared, local, users, items, **kwargs):
        assert active_candidates, "scoring must be inside validation evaluation"
        candidates = active_candidates[-1]
        assert candidates.split == "valid"
        candidate = next(c for c in candidates.candidates if c.user_id == users[0])
        assert users.tolist() == [candidate.user_id] * len(candidate.item_ids)
        assert tuple(items.tolist()) == candidate.item_ids
        scores.append(candidate.target_interaction_id)
        return real_score(shared, local, users, items, **kwargs)

    def tracked_train(*args, **kwargs):
        assert kwargs["excluded_splits"] == ("train", "valid", "test")
        local_updates.append(kwargs["round_index"])
        return real_train(*args, **kwargs)

    def forbid_full_execution(*args, **kwargs):
        pytest.fail("fit must not call full execution or artifact/data resolution")

    monkeypatch.setattr(runner, "build_evaluation_candidates", guarded_build)
    monkeypatch.setattr(runner, "evaluate_candidates", guarded_evaluate)
    monkeypatch.setattr(runner, "score_fedgmf", guarded_score)
    monkeypatch.setattr(runner, "train_local_fedgmf", tracked_train)
    for name in (
        "run_fedgmf_hfl_experiment", "execute_phase3_experiment",
        "_resolve_ml1m_dataset", "create_run_directory", "write_result_artifacts",
    ):
        monkeypatch.setattr(runner, name, forbid_full_execution)

    result = runner.fit_fedgmf_hfl(processed_bundle, parties, config)
    assert len(built) == 1
    assert result.validation_candidates is built[0]
    assert evaluations == [built[0]] * 3
    assert scores == list(processed_bundle.splits.valid_interaction_ids) * 3
    assert local_updates == [0, 1, 2]
    assert tuple(field.name for field in fields(result)) == (
        "final_shared_state", "final_local_states", "validation_candidates",
        "round_summaries",
    )
    assert all(field.kw_only for field in fields(result))
    assert not hasattr(result, "test_candidates")
    assert not hasattr(result, "final_test_metrics")


@pytest.mark.parametrize("entry_point", ["fit_fedgmf_hfl", "run_fedgmf_hfl_experiment"])
@pytest.mark.parametrize(
    "fault, error_type, message",
    [
        ("empty", ValueError, "parties must not be empty"),
        ("not_party", TypeError, "parties must contain PartyData values"),
        ("duplicate_party", ValueError, "duplicate party IDs"),
        ("duplicate_user", ValueError, "belongs to multiple HFL parties"),
        ("invalid_user", ValueError, "out-of-range user ID"),
        ("missing_valid_owner", ValueError, "valid target users have no owning"),
        ("missing_test_owner", ValueError, "test target users have no owning"),
    ],
)
def test_fit_retains_structural_ownership_checks(
    processed_bundle, monkeypatch, entry_point, fault, error_type, message
) -> None:
    bundle = processed_bundle
    config, parties = _inputs(bundle, _CASES[2])
    if fault == "empty":
        parties = ()
    elif fault == "not_party":
        parties = (object(),)
    elif fault == "duplicate_party":
        parties = (parties[0], parties[0])
    elif fault == "duplicate_user":
        parties = (parties[0], replace(parties[1], user_ids=parties[0].user_ids))
    elif fault == "invalid_user":
        parties = (replace(
            parties[0], user_ids=np.array([len(bundle.user_id_map)], dtype=np.int64)
        ),)
    else:
        parties = parties[1:]
        if fault == "missing_test_owner":
            # Valid structural split with user 0 absent only from validation;
            # train-only execution must still reject that user's missing owner.
            valid = bundle.splits.valid_interaction_ids
            missing = bundle.interactions.user_ids[valid] == 0
            bundle = replace(bundle, splits=InteractionSplits(
                train_interaction_ids=np.sort(np.concatenate((
                    bundle.splits.train_interaction_ids, valid[missing],
                ))),
                valid_interaction_ids=valid[~missing],
                test_interaction_ids=bundle.splits.test_interaction_ids,
            ))

    def forbid_initialization(*args, **kwargs):
        pytest.fail("ownership failures must precede initialization")

    before = deepcopy(asdict(bundle))
    monkeypatch.setattr(runner, "initialize_fedgmf_shared_state", forbid_initialization)
    with pytest.raises(error_type, match=message):
        getattr(runner, entry_point)(bundle, parties, config)
    np.testing.assert_equal(asdict(bundle), before)


def test_fit_propagates_validation_failure_without_later_rounds(
    processed_bundle, monkeypatch
) -> None:
    config, parties = _inputs(processed_bundle, _CASES[1])
    before = deepcopy((processed_bundle, parties, config))
    real_train = runner.train_local_fedgmf
    trained_rounds = []

    def track_train(*args, **kwargs):
        trained_rounds.append(kwargs["round_index"])
        return real_train(*args, **kwargs)

    def fail_validation(candidates, *args, **kwargs):
        assert candidates.split == "valid"
        raise RuntimeError("validation failed")

    monkeypatch.setattr(runner, "train_local_fedgmf", track_train)
    monkeypatch.setattr(runner, "evaluate_candidates", fail_validation)
    with pytest.raises(RuntimeError, match="validation failed"):
        runner.fit_fedgmf_hfl(processed_bundle, parties, config)
    assert trained_rounds == [0]
    _assert_inputs_unchanged(processed_bundle, parties, config, before)
