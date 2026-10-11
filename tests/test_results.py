from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.data.contracts import DatasetBundle
from src.data.partition import build_hfl_partition
from src.evaluation.ranking import EvaluationCandidateSet
from src.federation.runner import (
    FedGMFHFLRoundSummary,
    FedGMFHFLRunResult,
)
from src.models import FedGMFLocalState
from src.utils.results import (
    CHECKPOINT_NAME,
    build_partition_metadata,
    create_run_directory,
    partition_assignment_checksum,
    round_communication_bytes,
    shared_state_num_bytes,
    write_result_artifacts,
)


def _result() -> FedGMFHFLRunResult:
    shared_state = {
        "item_embeddings": torch.arange(10, dtype=torch.float32).reshape(5, 2),
        "prediction_weight": torch.ones(1, 2),
        "prediction_bias": torch.zeros(1),
    }
    local_states = {
        7: FedGMFLocalState(
            user_ids=torch.tensor([3], dtype=torch.long),
            user_embeddings=torch.tensor([[0.1, 0.2]]),
        ),
        40: FedGMFLocalState(
            user_ids=torch.tensor([8, 9], dtype=torch.long),
            user_embeddings=torch.tensor([[0.3, 0.4], [0.5, 0.6]]),
        ),
    }
    candidates = EvaluationCandidateSet(
        split="valid",
        base_seed=11,
        sampling_seed=22,
        candidates=(),
    )
    test_candidates = EvaluationCandidateSet(
        split="test",
        base_seed=11,
        sampling_seed=23,
        candidates=(),
    )
    summaries = (
        FedGMFHFLRoundSummary(
            round_index=0,
            selected_party_ids=(7, 40),
            total_positive_interactions=6,
            validation_metrics={
                "num_targets": 3,
                "hr@1": 1.0 / 3.0,
                "ndcg@1": 1.0 / 3.0,
            },
        ),
        FedGMFHFLRoundSummary(
            round_index=1,
            selected_party_ids=(40,),
            total_positive_interactions=4,
            validation_metrics={
                "num_targets": 3,
                "hr@1": 2.0 / 3.0,
                "ndcg@1": 0.5,
            },
        ),
    )
    return FedGMFHFLRunResult(
        final_shared_state=shared_state,
        final_local_states=local_states,
        validation_candidates=candidates,
        test_candidates=test_candidates,
        round_summaries=summaries,
        final_test_metrics={
            "num_targets": 3,
            "hr@1": 1.0,
            "ndcg@1": 0.75,
        },
    )


def _config(*, save_checkpoint: bool) -> dict:
    return {
        "experiment": {"name": "test"},
        "model": {"name": "fedgmf", "embedding_dim": 2},
        "runtime": {
            "seed": 11,
            "device": "cpu",
            "results_root": "results",
            "save_checkpoint": save_checkpoint,
        },
    }


def _partition_metadata() -> dict:
    return {
        "assignment_checksum": "abc123",
        "base_data_identity": {"dataset_name": "synthetic"},
    }


def test_run_directory_never_overwrites(tmp_path: Path) -> None:
    first = create_run_directory(
        tmp_path,
        "experiment",
        run_seed=11,
        run_id="fixed",
    )

    assert first == tmp_path / "experiment" / "fixed"
    with pytest.raises(FileExistsError):
        create_run_directory(
            tmp_path,
            "experiment",
            run_seed=11,
            run_id="fixed",
        )


def test_communication_counts_shared_tensor_payload_only() -> None:
    result = _result()

    assert shared_state_num_bytes(result.final_shared_state) == 52
    assert round_communication_bytes(result.final_shared_state, 3) == {
        "download_bytes": 156,
        "upload_bytes": 156,
        "total_communication_bytes": 312,
    }
    local_bytes = sum(
        state.user_embeddings.numel() * state.user_embeddings.element_size()
        for state in result.final_local_states.values()
    )
    assert local_bytes > 0
    assert shared_state_num_bytes(result.final_shared_state) == 52


def test_partition_metadata_is_canonical_and_reproducible(
    processed_bundle: DatasetBundle,
) -> None:
    parties = build_hfl_partition(processed_bundle, mode="per_user")

    first_checksum = partition_assignment_checksum(parties)
    second_checksum = partition_assignment_checksum(tuple(reversed(parties)))
    metadata = build_partition_metadata(
        tuple(reversed(parties)),
        federation_type="hfl",
        partition_mode="per_user",
        partition_seed=37,
        base_data_identity={"raw_ratings_sha256": "example"},
    )

    assert first_checksum == second_checksum
    assert metadata["assignment_checksum"] == first_checksum
    assert [party["party_id"] for party in metadata["parties"]] == [0, 1, 2, 3]
    assert all(
        party["user_ids"] == sorted(party["user_ids"])
        for party in metadata["parties"]
    )


def test_result_artifacts_match_in_memory_result_without_checkpoint(
    tmp_path: Path,
) -> None:
    result = _result()
    config = _config(save_checkpoint=False)

    write_result_artifacts(
        tmp_path,
        config=config,
        result=result,
        partition_metadata=_partition_metadata(),
        runtime_metadata={"configured_device": "cpu"},
    )

    assert yaml.safe_load(
        (tmp_path / "config.yaml").read_text(encoding="utf-8")
    ) == config
    metrics = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics == {
        "validation": result.round_summaries[-1].validation_metrics,
        "test": result.final_test_metrics,
    }
    with (tmp_path / "round_metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    assert rows[0]["selected_party_ids"] == "[7,40]"
    assert rows[0]["download_bytes"] == "104"
    assert rows[0]["upload_bytes"] == "104"
    assert rows[0]["total_communication_bytes"] == "208"
    assert not (tmp_path / CHECKPOINT_NAME).exists()


def test_checkpoint_round_trips_final_shared_and_local_state(
    tmp_path: Path,
) -> None:
    result = _result()
    config = _config(save_checkpoint=True)

    write_result_artifacts(
        tmp_path,
        config=config,
        result=result,
        partition_metadata=_partition_metadata(),
        runtime_metadata={"configured_device": "cpu"},
    )
    checkpoint = torch.load(
        tmp_path / CHECKPOINT_NAME,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["completed_round"] == 1
    assert checkpoint["run_seed"] == 11
    assert checkpoint["partition_checksum"] == "abc123"
    with (tmp_path / "round_metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as source:
        first_round = next(csv.DictReader(source))
    assert first_round["total_communication_bytes"] == "208"
    for key, tensor in result.final_shared_state.items():
        assert torch.equal(checkpoint["shared_state"][key], tensor)
    assert set(checkpoint["local_states"]) == {7, 40}
    for party_id, local_state in result.final_local_states.items():
        stored = checkpoint["local_states"][party_id]
        assert torch.equal(stored["user_ids"], local_state.user_ids)
        assert torch.equal(
            stored["user_embeddings"],
            local_state.user_embeddings,
        )
