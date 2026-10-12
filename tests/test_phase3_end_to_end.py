from __future__ import annotations

import csv
import json
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

import src.federation.runner as runner_module
from src.data.contracts import DatasetBundle
from src.federation.runner import (
    Phase3ExperimentExecutionResult,
    execute_phase3_experiment,
)
from src.utils.config import load_yaml, validate_phase3_experiment_config

_REFERENCE_CONFIG = (
    Path(__file__).parents[1]
    / "configs"
    / "experiments"
    / "ml1m_fedgmf_fedavg.yaml"
)
_MANDATORY_ARTIFACTS = {
    "config.yaml",
    "metrics.json",
    "round_metrics.csv",
    "runtime.json",
    "partition.json",
    "run.log",
}


def _synthetic_config(
    bundle: DatasetBundle,
    results_root: Path,
) -> dict[str, Any]:
    config = deepcopy(load_yaml(_REFERENCE_CONFIG))
    config["experiment"]["name"] = "synthetic_phase3_end_to_end"
    config["dataset"]["expected_raw_sha256"] = bundle.metadata[
        "raw_ratings_sha256"
    ]
    config["federation"].update(
        partition="grouped_round_robin",
        num_parties=2,
        rounds=2,
        client_fraction=0.5,
        local_epochs=1,
    )
    config["model"]["embedding_dim"] = 3
    config["optimization"].update(
        batch_size=4,
        train_negatives_per_positive=1,
    )
    config["evaluation"].update(k=[3], negatives_per_positive=2)
    config["runtime"].update(
        seed=37,
        device="cpu",
        results_root=str(results_root),
        save_checkpoint=False,
    )
    validate_phase3_experiment_config(config)
    return config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_round_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _assert_ranking_metrics(metrics: dict[str, Any]) -> None:
    assert metrics["num_targets"] > 0
    for name, value in metrics.items():
        if name != "num_targets":
            assert isfinite(value)
            assert 0.0 <= value <= 1.0


def _assert_same_scientific_result(
    first: Phase3ExperimentExecutionResult,
    second: Phase3ExperimentExecutionResult,
) -> None:
    first_result = first.experiment_result
    second_result = second.experiment_result
    assert first_result.validation_candidates == (
        second_result.validation_candidates
    )
    assert first_result.test_candidates == second_result.test_candidates
    assert first_result.round_summaries == second_result.round_summaries
    assert first_result.final_test_metrics == second_result.final_test_metrics
    for key, tensor in first_result.final_shared_state.items():
        assert torch.equal(tensor, second_result.final_shared_state[key])
    assert set(first_result.final_local_states) == set(
        second_result.final_local_states
    )
    for party_id, first_state in first_result.final_local_states.items():
        second_state = second_result.final_local_states[party_id]
        assert torch.equal(first_state.user_ids, second_state.user_ids)
        assert torch.equal(
            first_state.user_embeddings,
            second_state.user_embeddings,
        )


def test_phase3_full_stack_is_reproducible(
    processed_bundle: DatasetBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _synthetic_config(processed_bundle, tmp_path / "results")
    dataset_identity = {
        "input_mode": "synthetic",
        "dataset_name": processed_bundle.name,
        "schema_version": processed_bundle.schema_version,
        "raw_ratings_sha256": processed_bundle.metadata[
            "raw_ratings_sha256"
        ],
    }
    monkeypatch.setattr(
        runner_module,
        "_resolve_ml1m_dataset",
        lambda _: (processed_bundle, dict(dataset_identity)),
    )

    first = execute_phase3_experiment(config)
    second = execute_phase3_experiment(config)

    for execution in (first, second):
        run_directory = execution.run_directory
        assert _MANDATORY_ARTIFACTS <= {
            path.name for path in run_directory.iterdir()
        }
        assert not (run_directory / "checkpoint.pt").exists()
        assert yaml.safe_load(
            (run_directory / "config.yaml").read_text(encoding="utf-8")
        ) == config

        rows = _read_round_rows(run_directory / "round_metrics.csv")
        assert len(rows) == 2
        assert [int(row["round_index"]) for row in rows] == [0, 1]
        for row in rows:
            assert "validation_hr@3" in row
            assert "validation_ndcg@3" in row
            download = int(row["download_bytes"])
            upload = int(row["upload_bytes"])
            total = int(row["total_communication_bytes"])
            assert download > 0
            assert upload > 0
            assert total == download + upload

        metrics = _read_json(run_directory / "metrics.json")
        _assert_ranking_metrics(metrics["validation"])
        _assert_ranking_metrics(metrics["test"])

        partition = _read_json(run_directory / "partition.json")
        assert partition["partition_mode"] == "grouped_round_robin"
        assert partition["actual_party_count"] == 2
        assert partition["assignment_checksum"]

        runtime = _read_json(run_directory / "runtime.json")
        assert runtime["configured_device"] == "cpu"
        assert runtime["dataset_identity"] == dataset_identity
        assert runtime["partition_checksum"] == (
            partition["assignment_checksum"]
        )

        result = execution.experiment_result
        assert all("user" not in key for key in result.final_shared_state)
        assert all(
            torch.isfinite(tensor).all().item()
            for tensor in result.final_shared_state.values()
        )
        assert result.final_local_states
        assert all(
            state.user_embeddings.shape[0] == state.user_ids.numel()
            and torch.isfinite(state.user_embeddings).all().item()
            for state in result.final_local_states.values()
        )
        for summary in result.round_summaries:
            _assert_ranking_metrics(summary.validation_metrics)
        _assert_ranking_metrics(result.final_test_metrics)

    _assert_same_scientific_result(first, second)
    assert _read_json(first.run_directory / "metrics.json") == _read_json(
        second.run_directory / "metrics.json"
    )
    assert _read_round_rows(
        first.run_directory / "round_metrics.csv"
    ) == _read_round_rows(second.run_directory / "round_metrics.csv")
