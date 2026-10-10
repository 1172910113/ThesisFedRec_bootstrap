"""YAML configuration loading."""

import math
import re
from pathlib import Path
from typing import Any

import yaml

_PHASE3_TOP_LEVEL_SECTIONS = {
    "experiment",
    "dataset",
    "federation",
    "model",
    "optimization",
    "evaluation",
    "runtime",
}
_KNOWN_POSITIVE_SPLITS = ("train", "valid", "test")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from the given path.

    Empty files are treated as empty mappings. A non-mapping document is
    rejected because experiment configuration is always key-value based.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        content = yaml.safe_load(config_file)

    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(f"YAML configuration must be a mapping: {config_path}")
    return content


def load_phase3_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one explicit Phase 3 experiment configuration."""
    config = load_yaml(path)
    validate_phase3_experiment_config(config)
    return config


def validate_phase3_experiment_config(config: dict[str, Any]) -> None:
    """Validate the currently supported Phase 3 configuration domain."""
    if not isinstance(config, dict):
        raise ValueError("Phase 3 experiment configuration must be a mapping")
    _validate_keys(
        config,
        "configuration",
        required=_PHASE3_TOP_LEVEL_SECTIONS,
    )

    _validate_experiment(_section(config, "experiment"))
    _validate_dataset(_section(config, "dataset"))
    _validate_federation(_section(config, "federation"))
    _validate_model(_section(config, "model"))
    _validate_optimization(_section(config, "optimization"))
    _validate_evaluation(_section(config, "evaluation"))
    _validate_runtime(_section(config, "runtime"))


def _validate_experiment(section: dict[str, Any]) -> None:
    _validate_keys(section, "experiment", required={"name"})
    _require_string(section, "name", "experiment")


def _validate_dataset(section: dict[str, Any]) -> None:
    required = {
        "name",
        "task",
        "input",
        "expected_raw_sha256",
        "min_interactions_per_user",
        "duplicate_policy",
        "implicit_feedback",
        "canonical_order",
        "split",
        "tie_breaker",
    }
    _validate_keys(
        section,
        "dataset",
        required=required,
        optional={"ratings_path", "cache_path"},
    )

    _require_enum(section, "name", "dataset", {"ml1m"})
    _require_enum(section, "task", "dataset", {"implicit"})
    input_mode = _require_enum(
        section,
        "input",
        "dataset",
        {"raw", "cache"},
    )
    if input_mode == "raw" and "ratings_path" not in section:
        raise ValueError("dataset.ratings_path is required for raw input")
    if input_mode == "cache" and "cache_path" not in section:
        raise ValueError("dataset.cache_path is required for cache input")
    for path_key in ("ratings_path", "cache_path"):
        if path_key in section:
            _require_relative_path(section, path_key, "dataset")

    checksum = _require_string(section, "expected_raw_sha256", "dataset")
    if re.fullmatch(r"[0-9a-fA-F]{64}", checksum) is None:
        raise ValueError(
            "dataset.expected_raw_sha256 must be a 64-character SHA-256"
        )
    minimum = _require_int(
        section,
        "min_interactions_per_user",
        "dataset",
        minimum=1,
    )
    if minimum < 3:
        raise ValueError(
            "dataset.min_interactions_per_user must be at least 3 "
            "for temporal leave-one-out"
        )
    _require_enum(
        section,
        "duplicate_policy",
        "dataset",
        {"keep_latest", "keep_all"},
    )
    _require_enum(
        section,
        "implicit_feedback",
        "dataset",
        {"all_observed"},
    )
    _require_enum(
        section,
        "canonical_order",
        "dataset",
        {"source_order"},
    )
    _require_enum(
        section,
        "split",
        "dataset",
        {"leave_one_out_temporal"},
    )
    _require_enum(
        section,
        "tie_breaker",
        "dataset",
        {"interaction_id"},
    )


def _validate_federation(section: dict[str, Any]) -> None:
    required = {
        "type",
        "partition",
        "aggregation",
        "rounds",
        "client_fraction",
        "local_epochs",
    }
    _validate_keys(
        section,
        "federation",
        required=required,
        optional={"num_parties"},
    )

    _require_enum(section, "type", "federation", {"hfl"})
    partition = _require_enum(
        section,
        "partition",
        "federation",
        {"per_user", "grouped_round_robin"},
    )
    _require_enum(section, "aggregation", "federation", {"fedavg"})
    _require_int(section, "rounds", "federation", minimum=1)
    _require_int(section, "local_epochs", "federation", minimum=1)
    client_fraction = _require_number(
        section,
        "client_fraction",
        "federation",
    )
    if not 0 < client_fraction <= 1:
        raise ValueError("federation.client_fraction must be in (0, 1]")

    if partition == "per_user" and "num_parties" in section:
        raise ValueError(
            "federation.num_parties must be omitted for per_user partition"
        )
    if partition == "grouped_round_robin":
        if "num_parties" not in section:
            raise ValueError(
                "federation.num_parties is required for "
                "grouped_round_robin partition"
            )
        _require_int(section, "num_parties", "federation", minimum=1)


def _validate_model(section: dict[str, Any]) -> None:
    required = {
        "name",
        "embedding_dim",
        "user_embedding_scope",
        "item_embedding_scope",
        "prediction_layer_scope",
    }
    _validate_keys(section, "model", required=required)
    _require_enum(section, "name", "model", {"fedgmf"})
    _require_int(section, "embedding_dim", "model", minimum=1)
    _require_enum(
        section,
        "user_embedding_scope",
        "model",
        {"party_local"},
    )
    _require_enum(
        section,
        "item_embedding_scope",
        "model",
        {"federated"},
    )
    _require_enum(
        section,
        "prediction_layer_scope",
        "model",
        {"federated"},
    )


def _validate_optimization(section: dict[str, Any]) -> None:
    required = {
        "optimizer",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "train_negatives_per_positive",
        "train_negative_strategy",
        "train_negative_replacement",
        "train_negative_excluded_splits",
        "train_negative_resampling",
    }
    _validate_keys(section, "optimization", required=required)

    _require_enum(section, "optimizer", "optimization", {"adam"})
    learning_rate = _require_number(
        section,
        "learning_rate",
        "optimization",
    )
    if learning_rate <= 0:
        raise ValueError("optimization.learning_rate must be greater than 0")
    _require_int(section, "batch_size", "optimization", minimum=1)
    weight_decay = _require_number(
        section,
        "weight_decay",
        "optimization",
    )
    if weight_decay < 0:
        raise ValueError("optimization.weight_decay must be non-negative")
    _require_int(
        section,
        "train_negatives_per_positive",
        "optimization",
        minimum=1,
    )
    _require_enum(
        section,
        "train_negative_strategy",
        "optimization",
        {"uniform"},
    )
    replacement = _require_bool(
        section,
        "train_negative_replacement",
        "optimization",
    )
    if replacement:
        raise ValueError(
            "optimization.train_negative_replacement must be false"
        )
    _require_known_positive_splits(
        section,
        "train_negative_excluded_splits",
        "optimization",
    )
    _require_enum(
        section,
        "train_negative_resampling",
        "optimization",
        {"per_local_epoch"},
    )


def _validate_evaluation(section: dict[str, Any]) -> None:
    required = {
        "k",
        "negatives_per_positive",
        "replacement",
        "excluded_splits",
        "candidate_reuse",
        "ranking_direction",
        "tie_breaker",
        "validation_policy",
        "early_stopping",
        "model_selection",
        "test_policy",
    }
    _validate_keys(section, "evaluation", required=required)

    k_values = section["k"]
    if not isinstance(k_values, list) or not k_values:
        raise ValueError("evaluation.k must be a non-empty list")
    for value in k_values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "evaluation.k must contain only positive integers"
            )
    if len(set(k_values)) != len(k_values):
        raise ValueError("evaluation.k must not contain duplicates")

    negative_count = _require_int(
        section,
        "negatives_per_positive",
        "evaluation",
        minimum=1,
    )
    if negative_count + 1 < max(k_values):
        raise ValueError(
            "evaluation candidate count must be at least the largest K"
        )
    replacement = _require_bool(section, "replacement", "evaluation")
    if replacement:
        raise ValueError("evaluation.replacement must be false")
    _require_known_positive_splits(
        section,
        "excluded_splits",
        "evaluation",
    )
    _require_enum(
        section,
        "candidate_reuse",
        "evaluation",
        {"fixed_per_run"},
    )
    _require_enum(
        section,
        "ranking_direction",
        "evaluation",
        {"descending_score"},
    )
    _require_enum(
        section,
        "tie_breaker",
        "evaluation",
        {"item_id_ascending"},
    )
    _require_enum(
        section,
        "validation_policy",
        "evaluation",
        {"every_round_monitoring_only"},
    )
    early_stopping = _require_bool(
        section,
        "early_stopping",
        "evaluation",
    )
    if early_stopping:
        raise ValueError("evaluation.early_stopping must be false")
    _require_enum(
        section,
        "model_selection",
        "evaluation",
        {"none"},
    )
    _require_enum(
        section,
        "test_policy",
        "evaluation",
        {"final_state_only"},
    )


def _validate_runtime(section: dict[str, Any]) -> None:
    required = {
        "seed",
        "device",
        "results_root",
        "save_checkpoint",
    }
    _validate_keys(section, "runtime", required=required)

    seed = _require_int(section, "seed", "runtime", minimum=0)
    if seed >= 2**32:
        raise ValueError("runtime.seed must be less than 2**32")
    device = _require_string(section, "device", "runtime")
    if device != "cpu" and re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError(
            "runtime.device must be 'cpu' or an explicit 'cuda:N' device"
        )
    _require_string(section, "results_root", "runtime")
    _require_bool(section, "save_checkpoint", "runtime")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config[name]
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_keys(
    values: dict[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    if any(not isinstance(key, str) for key in values):
        raise ValueError(f"{path} keys must be strings")
    allowed = required | (optional or set())
    missing = required - set(values)
    if missing:
        raise ValueError(
            f"{path} is missing required fields: {', '.join(sorted(missing))}"
        )
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"{path} has unsupported fields: {', '.join(sorted(unknown))}"
        )


def _require_string(
    values: dict[str, Any],
    key: str,
    path: str,
) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value


def _require_enum(
    values: dict[str, Any],
    key: str,
    path: str,
    allowed: set[str],
) -> str:
    value = _require_string(values, key, path)
    if value not in allowed:
        supported = ", ".join(sorted(allowed))
        raise ValueError(
            f"{path}.{key} has unsupported value {value!r}; "
            f"expected one of: {supported}"
        )
    return value


def _require_int(
    values: dict[str, Any],
    key: str,
    path: str,
    *,
    minimum: int,
) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}.{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{path}.{key} must be at least {minimum}")
    return value


def _require_number(
    values: dict[str, Any],
    key: str,
    path: str,
) -> float:
    value = values[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{path}.{key} must be a finite number")
    return float(value)


def _require_bool(
    values: dict[str, Any],
    key: str,
    path: str,
) -> bool:
    value = values[key]
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a boolean")
    return value


def _require_relative_path(
    values: dict[str, Any],
    key: str,
    path: str,
) -> None:
    value = _require_string(values, key, path)
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(
            f"{path}.{key} must be relative and remain within its data root"
        )


def _require_known_positive_splits(
    values: dict[str, Any],
    key: str,
    path: str,
) -> None:
    value = values[key]
    if (
        not isinstance(value, list)
        or len(value) != len(_KNOWN_POSITIVE_SPLITS)
        or any(not isinstance(name, str) for name in value)
        or set(value) != set(_KNOWN_POSITIVE_SPLITS)
    ):
        raise ValueError(
            f"{path}.{key} must contain train, valid, and test exactly once"
        )
