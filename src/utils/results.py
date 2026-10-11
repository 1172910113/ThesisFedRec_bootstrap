"""Plain, reproducible Phase 3 experiment result artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

from src.data.contracts import PartyData

if TYPE_CHECKING:
    from src.federation.runner import FedGMFHFLRunResult

CHECKPOINT_NAME = "checkpoint.pt"


def create_run_directory(
    results_root: str | Path,
    experiment_name: str,
    *,
    run_seed: int,
    run_id: str | None = None,
) -> Path:
    """Create and return one non-overwriting experiment run directory."""
    if Path(experiment_name).name != experiment_name:
        raise ValueError("experiment_name must be a single path component")
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"seed_{run_seed}_{timestamp}"
    if Path(run_id).name != run_id:
        raise ValueError("run_id must be a single path component")

    run_directory = Path(results_root) / experiment_name / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def shared_state_num_bytes(
    shared_state: Mapping[str, torch.Tensor],
) -> int:
    """Return tensor payload bytes for a shared state only."""
    if not shared_state:
        raise ValueError("shared_state must not be empty")
    total = 0
    for key, tensor in shared_state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("shared_state must map string keys to tensors")
        total += tensor.numel() * tensor.element_size()
    return total


def round_communication_bytes(
    shared_state: Mapping[str, torch.Tensor],
    selected_party_count: int,
) -> dict[str, int]:
    """Count shared-state download and upload bytes for one HFL round."""
    if (
        isinstance(selected_party_count, bool)
        or not isinstance(selected_party_count, int)
        or selected_party_count <= 0
    ):
        raise ValueError("selected_party_count must be a positive integer")
    payload_bytes = shared_state_num_bytes(shared_state)
    one_direction = payload_bytes * selected_party_count
    return {
        "download_bytes": one_direction,
        "upload_bytes": one_direction,
        "total_communication_bytes": 2 * one_direction,
    }


def partition_assignment_checksum(parties: Sequence[PartyData]) -> str:
    """Hash the canonical party-to-user assignment."""
    assignment = [
        {
            "party_id": int(party.party_id),
            "user_ids": sorted(int(user_id) for user_id in party.user_ids),
        }
        for party in sorted(parties, key=lambda value: value.party_id)
    ]
    serialized = json.dumps(
        assignment,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_partition_metadata(
    parties: Sequence[PartyData],
    *,
    federation_type: str,
    partition_mode: str,
    partition_seed: int,
    base_data_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build canonical run-level HFL partition metadata."""
    ordered_parties = tuple(sorted(parties, key=lambda value: value.party_id))
    checksum = partition_assignment_checksum(ordered_parties)
    return {
        "federation_type": federation_type,
        "partition_mode": partition_mode,
        "partition_seed": partition_seed,
        "actual_party_count": len(ordered_parties),
        "assignment_checksum": checksum,
        "base_data_identity": dict(base_data_identity),
        "parties": [
            {
                "party_id": int(party.party_id),
                "user_ids": sorted(
                    int(user_id) for user_id in party.user_ids
                ),
                "train_interaction_count": int(
                    party.train_interaction_ids.size
                ),
                "validation_interaction_count": int(
                    party.valid_interaction_ids.size
                ),
                "test_interaction_count": int(
                    party.test_interaction_ids.size
                ),
                "distinct_item_count": int(party.item_ids.size),
            }
            for party in ordered_parties
        ],
    }


def capture_git_identity() -> tuple[str | None, bool | None]:
    """Return the current repository commit and dirty flag."""
    repository_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())

def collect_runtime_metadata(
    *,
    start_utc: datetime,
    end_utc: datetime,
    duration_seconds: float,
    configured_device: str,
    run_seed: int,
    dataset_identity: Mapping[str, Any],
    partition_checksum: str,
    git_commit: str | None,
    git_dirty: bool | None,
) -> dict[str, Any]:
    """Collect non-scientific runtime and reproducibility facts."""
    try:
        cuda_available = bool(torch.cuda.is_available())
    except (AssertionError, RuntimeError):
        cuda_available = False
    device_name: str | None = None
    if configured_device.startswith("cuda:") and cuda_available:
        try:
            device_name = torch.cuda.get_device_name(
                int(configured_device.split(":", maxsplit=1)[1])
            )
        except (AssertionError, RuntimeError, ValueError):
            device_name = None

    return {
        "start_utc": start_utc.astimezone(timezone.utc).isoformat(),
        "end_utc": end_utc.astimezone(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "configured_device": configured_device,
        "selected_device_name": device_name,
        "run_seed": run_seed,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_state_captured_at": "experiment_launch",
        "dataset_identity": dict(dataset_identity),
        "partition_checksum": partition_checksum,
        "reproducibility_note": (
            "Best-effort reproducibility; deterministic CUDA behavior is "
            "not guaranteed."
        ),
    }


def write_result_artifacts(
    run_directory: str | Path,
    *,
    config: Mapping[str, Any],
    result: FedGMFHFLRunResult,
    partition_metadata: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
) -> None:
    """Write the required Phase 3 result files except the active run log."""
    path = Path(run_directory)
    _write_text(
        path / "config.yaml",
        yaml.safe_dump(dict(config), sort_keys=False),
    )

    metrics = {
        "validation": dict(result.round_summaries[-1].validation_metrics),
        "test": dict(result.final_test_metrics),
    }
    _write_json(path / "metrics.json", metrics)
    _write_round_metrics(path / "round_metrics.csv", result)
    _write_json(path / "partition.json", dict(partition_metadata))
    _write_json(path / "runtime.json", dict(runtime_metadata))

    if config["runtime"]["save_checkpoint"]:
        _write_checkpoint(
            path / CHECKPOINT_NAME,
            config=config,
            result=result,
            partition_metadata=partition_metadata,
        )


def _write_round_metrics(
    path: Path,
    result: FedGMFHFLRunResult,
) -> None:
    metric_names = tuple(result.round_summaries[0].validation_metrics)
    fieldnames = [
        "round_index",
        "selected_party_count",
        "selected_party_ids",
        "total_positive_interactions",
        *(f"validation_{name}" for name in metric_names),
        "download_bytes",
        "upload_bytes",
        "total_communication_bytes",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for summary in result.round_summaries:
        communication = round_communication_bytes(
            result.final_shared_state,
            len(summary.selected_party_ids),
        )
        row: dict[str, Any] = {
            "round_index": summary.round_index,
            "selected_party_count": len(summary.selected_party_ids),
            "selected_party_ids": json.dumps(
                list(summary.selected_party_ids),
                separators=(",", ":"),
            ),
            "total_positive_interactions": (
                summary.total_positive_interactions
            ),
            **{
                f"validation_{name}": summary.validation_metrics[name]
                for name in metric_names
            },
            **communication,
        }
        writer.writerow(row)
    _write_text(path, output.getvalue())


def _write_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    result: FedGMFHFLRunResult,
    partition_metadata: Mapping[str, Any],
) -> None:
    checkpoint = {
        "schema_version": "1.0",
        "completed_round": len(result.round_summaries) - 1,
        "run_seed": config["runtime"]["seed"],
        "model": dict(config["model"]),
        "partition_checksum": partition_metadata["assignment_checksum"],
        "dataset_identity": dict(partition_metadata["base_data_identity"]),
        "shared_state": {
            key: tensor.detach().cpu().clone()
            for key, tensor in result.final_shared_state.items()
        },
        "local_states": {
            int(party_id): {
                "user_ids": local_state.user_ids.detach().cpu().clone(),
                "user_embeddings": (
                    local_state.user_embeddings.detach().cpu().clone()
                ),
            }
            for party_id, local_state in result.final_local_states.items()
        },
    }
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
