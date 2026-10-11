from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

import src.federation.runner as runner_module
import src.utils.results as results_module
from src.data.contracts import DatasetBundle, PartyData
from src.evaluation.ranking import EvaluationCandidateSet
from src.federation.runner import (
    FedGMFHFLRoundSummary,
    FedGMFHFLRunResult,
    execute_phase3_experiment,
)
from src.models import FedGMFLocalState
from src.utils.config import load_yaml, validate_phase3_experiment_config

_REFERENCE_CONFIG = (
    Path(__file__).parents[1]
    / "configs"
    / "experiments"
    / "ml1m_fedgmf_fedavg.yaml"
)


def _config(
    results_root: Path,
    *,
    grouped_parties: int | None,
) -> dict[str, Any]:
    config = deepcopy(load_yaml(_REFERENCE_CONFIG))
    config["runtime"].update(
        seed=37,
        device="cpu",
        results_root=str(results_root),
        save_checkpoint=False,
    )
    config["federation"]["rounds"] = 2
    if grouped_parties is None:
        config["federation"]["partition"] = "per_user"
        config["federation"].pop("num_parties", None)
    else:
        config["federation"]["partition"] = "grouped_round_robin"
        config["federation"]["num_parties"] = grouped_parties
    validate_phase3_experiment_config(config)
    return config


def _fake_result(
    parties: tuple[PartyData, ...],
    rounds: int,
) -> FedGMFHFLRunResult:
    shared_state = {
        "item_embeddings": torch.zeros(11, 2),
        "prediction_weight": torch.ones(1, 2),
        "prediction_bias": torch.zeros(1),
    }
    local_states = {
        party.party_id: FedGMFLocalState(
            user_ids=torch.from_numpy(party.user_ids.copy()),
            user_embeddings=torch.zeros(len(party.user_ids), 2),
        )
        for party in parties
    }
    validation_candidates = EvaluationCandidateSet(
        split="valid",
        base_seed=37,
        sampling_seed=74,
        candidates=(),
    )
    test_candidates = EvaluationCandidateSet(
        split="test",
        base_seed=37,
        sampling_seed=75,
        candidates=(),
    )
    party_ids = tuple(sorted(party.party_id for party in parties))
    summaries = tuple(
        FedGMFHFLRoundSummary(
            round_index=round_index,
            selected_party_ids=party_ids,
            total_positive_interactions=sum(
                len(party.train_interaction_ids) for party in parties
            ),
            validation_metrics={
                "num_targets": 4,
                "hr@10": 0.5,
                "ndcg@10": 0.25,
            },
        )
        for round_index in range(rounds)
    )
    return FedGMFHFLRunResult(
        final_shared_state=shared_state,
        final_local_states=local_states,
        validation_candidates=validation_candidates,
        test_candidates=test_candidates,
        round_summaries=summaries,
        final_test_metrics={
            "num_targets": 4,
            "hr@10": 0.75,
            "ndcg@10": 0.5,
        },
    )


def test_high_level_executor_builds_grouped_partition_and_artifacts(
    processed_bundle: DatasetBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, grouped_parties=2)
    captured_parties: list[tuple[PartyData, ...]] = []

    monkeypatch.setattr(
        runner_module,
        "_resolve_ml1m_dataset",
        lambda _: (
            processed_bundle,
            {
                "dataset_name": "ml1m",
                "raw_ratings_sha256": "synthetic",
            },
        ),
    )

    def fake_run(
        bundle: DatasetBundle,
        parties: tuple[PartyData, ...],
        executed_config: dict[str, Any],
    ) -> FedGMFHFLRunResult:
        assert bundle is processed_bundle
        assert executed_config is config
        captured_parties.append(parties)
        return _fake_result(parties, executed_config["federation"]["rounds"])

    monkeypatch.setattr(
        runner_module,
        "run_fedgmf_hfl_experiment",
        fake_run,
    )
    def unavailable_cuda() -> bool:
        raise RuntimeError("CUDA runtime unavailable")

    monkeypatch.setattr(
        results_module.torch.cuda,
        "is_available",
        unavailable_cuda,
    )

    execution = execute_phase3_experiment(config)

    assert list(inspect.signature(execute_phase3_experiment).parameters) == [
        "config"
    ]
    assert len(captured_parties) == 1
    assert len(captured_parties[0]) == 2
    assert all(
        party.metadata["hfl_mode"] == "grouped_round_robin"
        for party in captured_parties[0]
    )
    required_artifacts = {
        "config.yaml",
        "metrics.json",
        "round_metrics.csv",
        "runtime.json",
        "partition.json",
        "run.log",
    }
    assert required_artifacts <= {
        path.name for path in execution.run_directory.iterdir()
    }
    assert not (execution.run_directory / "checkpoint.pt").exists()
    assert yaml.safe_load(
        (execution.run_directory / "config.yaml").read_text(encoding="utf-8")
    ) == config
    assert json.loads(
        (execution.run_directory / "metrics.json").read_text(encoding="utf-8")
    ) == {
        "validation": (
            execution.experiment_result.round_summaries[-1].validation_metrics
        ),
        "test": execution.experiment_result.final_test_metrics,
    }

    partition = json.loads(
        (execution.run_directory / "partition.json").read_text(
            encoding="utf-8"
        )
    )
    assert partition["partition_mode"] == "grouped_round_robin"
    assert partition["partition_seed"] == 37
    assert partition["actual_party_count"] == 2
    runtime = json.loads(
        (execution.run_directory / "runtime.json").read_text(encoding="utf-8")
    )
    assert runtime["configured_device"] == "cpu"
    assert runtime["cuda_available"] is False
    assert runtime["selected_device_name"] is None
    assert runtime["run_seed"] == 37
    assert runtime["partition_checksum"] == partition["assignment_checksum"]
    assert runtime["python_version"]
    assert runtime["pytorch_version"]
    assert runtime["start_utc"]
    assert runtime["end_utc"]
    assert runtime["duration_seconds"] >= 0
    log_text = (execution.run_directory / "run.log").read_text(
        encoding="utf-8"
    )
    for message in (
        "configuration loaded",
        "dataset resolved",
        "partition created",
        "run started",
        "round 0 complete",
        "final validation",
        "artifact directory",
    ):
        assert message in log_text


def test_high_level_executor_uses_per_user_partition(
    processed_bundle: DatasetBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, grouped_parties=None)
    captured_parties: list[tuple[PartyData, ...]] = []
    monkeypatch.setattr(
        runner_module,
        "_resolve_ml1m_dataset",
        lambda _: (processed_bundle, {"dataset_name": "ml1m"}),
    )

    def fake_run(
        bundle: DatasetBundle,
        parties: tuple[PartyData, ...],
        executed_config: dict[str, Any],
    ) -> FedGMFHFLRunResult:
        captured_parties.append(parties)
        return _fake_result(parties, executed_config["federation"]["rounds"])

    monkeypatch.setattr(
        runner_module,
        "run_fedgmf_hfl_experiment",
        fake_run,
    )

    execute_phase3_experiment(config)

    assert len(captured_parties[0]) == len(processed_bundle.user_id_map)
    assert all(party.user_ids.size == 1 for party in captured_parties[0])
    assert all(
        party.metadata["hfl_mode"] == "per_user"
        for party in captured_parties[0]
    )


def test_git_provenance_is_captured_before_run_artifacts(
    processed_bundle: DatasetBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "results"
    config = _config(results_root, grouped_parties=None)
    events: list[str] = []
    git_state = {"dirty": False, "calls": 0}
    real_create_run_directory = runner_module.create_run_directory

    def fake_capture_git_identity() -> tuple[str, bool]:
        assert not results_root.exists()
        git_state["calls"] += 1
        if git_state["calls"] != 1:
            pytest.fail("Git identity must only be captured at launch")
        events.append("git_identity")
        return "launch-commit", git_state["dirty"]

    def tracked_create_run_directory(
        results_root: str | Path,
        experiment_name: str,
        *,
        run_seed: int,
    ) -> Path:
        assert events == ["git_identity"]
        run_directory = real_create_run_directory(
            results_root,
            experiment_name,
            run_seed=run_seed,
        )
        (run_directory / "experiment-created.txt").write_text(
            "created after provenance capture\n",
            encoding="utf-8",
        )
        git_state["dirty"] = True
        events.append("run_directory")
        return run_directory

    monkeypatch.setattr(
        runner_module,
        "capture_git_identity",
        fake_capture_git_identity,
    )
    monkeypatch.setattr(
        runner_module,
        "create_run_directory",
        tracked_create_run_directory,
    )
    monkeypatch.setattr(
        runner_module,
        "_resolve_ml1m_dataset",
        lambda _: (processed_bundle, {"dataset_name": "ml1m"}),
    )
    monkeypatch.setattr(
        runner_module,
        "run_fedgmf_hfl_experiment",
        lambda _, parties, executed_config: _fake_result(
            parties,
            executed_config["federation"]["rounds"],
        ),
    )

    def unavailable_cuda() -> bool:
        raise RuntimeError("CUDA runtime unavailable")

    monkeypatch.setattr(
        results_module.torch.cuda,
        "is_available",
        unavailable_cuda,
    )

    execution = execute_phase3_experiment(config)
    stored_runtime = json.loads(
        (execution.run_directory / "runtime.json").read_text(
            encoding="utf-8"
        )
    )

    assert events == ["git_identity", "run_directory"]
    assert git_state == {"dirty": True, "calls": 1}
    assert stored_runtime["git_commit"] == "launch-commit"
    assert stored_runtime["git_dirty"] is False
    assert stored_runtime["git_state_captured_at"] == "experiment_launch"
    assert stored_runtime["configured_device"] == "cpu"
    assert stored_runtime["cuda_available"] is False
    assert stored_runtime["selected_device_name"] is None


def test_cache_mode_loads_only_the_exact_configured_cache(
    processed_bundle: DatasetBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, grouped_parties=None)
    config["dataset"].update(
        input="cache",
        cache_path="chosen/cache",
        expected_raw_sha256=processed_bundle.metadata[
            "raw_ratings_sha256"
        ],
    )
    validate_phase3_experiment_config(config)
    loaded_paths: list[Path] = []

    def fake_load(path: Path) -> DatasetBundle:
        loaded_paths.append(path)
        return processed_bundle

    monkeypatch.setattr(runner_module, "load_base_cache", fake_load)
    monkeypatch.setattr(
        runner_module,
        "preprocess_ml1m_from_root",
        lambda *args, **kwargs: pytest.fail("cache mode must not preprocess"),
    )
    monkeypatch.setattr(
        runner_module,
        "_sha256_file",
        lambda _: "cache-manifest-checksum",
    )

    bundle, identity = runner_module._resolve_ml1m_dataset(config)

    assert bundle is processed_bundle
    assert loaded_paths == [Path("chosen/cache")]
    assert identity["cache_path"] == "chosen/cache"
    assert identity["cache_manifest_sha256"] == "cache-manifest-checksum"


def test_raw_mode_preprocesses_only_the_configured_relative_ratings(
    processed_bundle: DatasetBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, grouped_parties=None)
    config["dataset"].update(
        ratings_path="chosen/ratings.dat",
        expected_raw_sha256=processed_bundle.metadata[
            "raw_ratings_sha256"
        ],
    )
    validate_phase3_experiment_config(config)
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_preprocess(path: str, **kwargs: Any) -> DatasetBundle:
        calls.append((path, kwargs))
        return processed_bundle

    monkeypatch.setattr(
        runner_module,
        "preprocess_ml1m_from_root",
        fake_preprocess,
    )
    monkeypatch.setattr(
        runner_module,
        "load_base_cache",
        lambda *args, **kwargs: pytest.fail("raw mode must not load a cache"),
    )

    bundle, identity = runner_module._resolve_ml1m_dataset(config)

    assert bundle is processed_bundle
    assert calls == [
        (
            "chosen/ratings.dat",
            {
                "min_interactions_per_user": 3,
                "duplicate_policy": "keep_latest",
                "implicit_feedback": "all_observed",
                "seed": 37,
            },
        )
    ]
    assert identity["input_mode"] == "raw"
    assert "cache_path" not in identity
