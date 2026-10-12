from pathlib import Path

import pytest

from src.utils import (
    load_phase3_experiment_config,
    load_yaml,
    validate_phase3_experiment_config,
)

REFERENCE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "ml1m_fedgmf_fedavg.yaml"
)
SMOKE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "ml1m_fedgmf_fedavg_smoke.yaml"
)


def _reference_config() -> dict:
    return load_yaml(REFERENCE_CONFIG)


def test_load_yaml_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "seed: 2026\nfederation:\n  type: hfl\n  rounds: 20\n",
        encoding="utf-8",
    )

    config = load_yaml(config_path)

    assert config == {
        "seed": 2026,
        "federation": {"type": "hfl", "rounds": 20},
    }


def test_reference_phase3_config_loads_and_validates() -> None:
    config = load_phase3_experiment_config(REFERENCE_CONFIG)

    assert config == _reference_config()
    assert config["experiment"]["name"] == "ml1m_fedgmf_fedavg"
    assert config["dataset"]["ratings_path"] == "ml-1m/ml-1m/ratings.dat"
    assert config["federation"] == {
        "type": "hfl",
        "partition": "per_user",
        "aggregation": "fedavg",
        "rounds": 20,
        "client_fraction": 1.0,
        "local_epochs": 1,
    }
    assert config["model"]["name"] == "fedgmf"
    assert config["model"]["embedding_dim"] == 8
    assert config["model"]["initialization"] == "ncf_gmf"
    assert config["optimization"]["train_negative_resampling"] == (
        "per_local_epoch"
    )
    assert config["evaluation"]["negatives_per_positive"] == 99


def test_ml1m_smoke_config_loads_and_preserves_frozen_semantics() -> None:
    config = load_phase3_experiment_config(SMOKE_CONFIG)

    assert config["experiment"]["name"] == "ml1m_fedgmf_fedavg_smoke"
    assert config["dataset"]["input"] == "raw"
    assert config["federation"] == {
        "type": "hfl",
        "partition": "grouped_round_robin",
        "num_parties": 32,
        "aggregation": "fedavg",
        "rounds": 1,
        "client_fraction": 0.03125,
        "local_epochs": 1,
    }
    assert 32 * config["federation"]["client_fraction"] == 1
    assert config["model"] == {
        "name": "fedgmf",
        "embedding_dim": 8,
        "initialization": "ncf_gmf",
        "user_embedding_scope": "party_local",
        "item_embedding_scope": "federated",
        "prediction_layer_scope": "federated",
    }
    assert config["optimization"] == {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 256,
        "weight_decay": 0,
        "train_negatives_per_positive": 4,
        "train_negative_strategy": "uniform",
        "train_negative_replacement": False,
        "train_negative_excluded_splits": ["train", "valid", "test"],
        "train_negative_resampling": "per_local_epoch",
    }
    assert config["evaluation"] == {
        "k": [10],
        "negatives_per_positive": 99,
        "replacement": False,
        "excluded_splits": ["train", "valid", "test"],
        "candidate_reuse": "fixed_per_run",
        "ranking_direction": "descending_score",
        "tie_breaker": "item_id_ascending",
        "validation_policy": "every_round_monitoring_only",
        "early_stopping": False,
        "model_selection": "none",
        "test_policy": "final_state_only",
    }
    assert config["runtime"] == {
        "seed": 2026,
        "device": "cuda:0",
        "results_root": "tmp/phase3_smoke_results",
        "save_checkpoint": False,
    }


def test_missing_required_scientific_field_fails() -> None:
    config = _reference_config()
    del config["optimization"]["learning_rate"]

    with pytest.raises(ValueError, match="optimization.*learning_rate"):
        validate_phase3_experiment_config(config)


def test_model_initialization_protocol_is_required_and_frozen() -> None:
    missing = _reference_config()
    del missing["model"]["initialization"]
    with pytest.raises(ValueError, match="model.*initialization"):
        validate_phase3_experiment_config(missing)

    unsupported = _reference_config()
    unsupported["model"]["initialization"] = "framework_default"
    with pytest.raises(ValueError, match="model.initialization"):
        validate_phase3_experiment_config(unsupported)


def test_unknown_top_level_section_fails() -> None:
    config = _reference_config()
    config["unknown"] = {}

    with pytest.raises(ValueError, match="unsupported fields: unknown"):
        validate_phase3_experiment_config(config)


@pytest.mark.parametrize("client_fraction", [0, -0.1, 1.1])
def test_invalid_client_fraction_fails(client_fraction: float) -> None:
    config = _reference_config()
    config["federation"]["client_fraction"] = client_fraction

    with pytest.raises(ValueError, match="client_fraction"):
        validate_phase3_experiment_config(config)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("federation", "rounds"),
        ("federation", "local_epochs"),
        ("optimization", "batch_size"),
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_integer_fields_fail(
    section: str,
    field: str,
    value: int,
) -> None:
    config = _reference_config()
    config[section][field] = value

    with pytest.raises(ValueError, match=field):
        validate_phase3_experiment_config(config)


@pytest.mark.parametrize("k_values", [[], [0], [-1], ["10"]])
def test_invalid_evaluation_k_fails(k_values: list) -> None:
    config = _reference_config()
    config["evaluation"]["k"] = k_values

    with pytest.raises(ValueError, match="evaluation.k"):
        validate_phase3_experiment_config(config)


def test_insufficient_evaluation_candidate_count_fails() -> None:
    config = _reference_config()
    config["evaluation"]["k"] = [10]
    config["evaluation"]["negatives_per_positive"] = 8

    with pytest.raises(ValueError, match="candidate count"):
        validate_phase3_experiment_config(config)


def test_invalid_input_mode_fails() -> None:
    config = _reference_config()
    config["dataset"]["input"] = "automatic"

    with pytest.raises(ValueError, match="dataset.input"):
        validate_phase3_experiment_config(config)


def test_invalid_partition_num_parties_combinations_fail() -> None:
    per_user = _reference_config()
    per_user["federation"]["num_parties"] = 10
    with pytest.raises(ValueError, match="omitted for per_user"):
        validate_phase3_experiment_config(per_user)

    grouped_missing_count = _reference_config()
    grouped_missing_count["federation"]["partition"] = "grouped_round_robin"
    with pytest.raises(ValueError, match="required for grouped_round_robin"):
        validate_phase3_experiment_config(grouped_missing_count)

    grouped_invalid_count = _reference_config()
    grouped_invalid_count["federation"]["partition"] = "grouped_round_robin"
    grouped_invalid_count["federation"]["num_parties"] = 0
    with pytest.raises(ValueError, match="num_parties"):
        validate_phase3_experiment_config(grouped_invalid_count)


def test_negative_resampling_policy_is_required() -> None:
    config = _reference_config()
    del config["optimization"]["train_negative_resampling"]

    with pytest.raises(
        ValueError,
        match="train_negative_resampling",
    ):
        validate_phase3_experiment_config(config)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("optimization", "learning_rate", 0),
        ("optimization", "learning_rate", -0.1),
        ("optimization", "weight_decay", -0.1),
    ],
)
def test_invalid_optimization_ranges_fail(
    section: str,
    field: str,
    value: float,
) -> None:
    config = _reference_config()
    config[section][field] = value

    with pytest.raises(ValueError, match=field):
        validate_phase3_experiment_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("validation_policy", "final_only"),
        ("early_stopping", True),
        ("model_selection", "best_validation"),
        ("test_policy", "every_round"),
    ],
)
def test_invalid_validation_or_test_policy_fails(
    field: str,
    value: object,
) -> None:
    config = _reference_config()
    config["evaluation"][field] = value

    with pytest.raises(ValueError, match=field):
        validate_phase3_experiment_config(config)


def test_cpu_and_supported_non_reference_values_validate() -> None:
    config = _reference_config()
    config["federation"].update(
        {
            "partition": "grouped_round_robin",
            "num_parties": 2,
            "rounds": 1,
            "client_fraction": 0.5,
            "local_epochs": 2,
        }
    )
    config["model"]["embedding_dim"] = 16
    config["evaluation"]["k"] = [5]
    config["evaluation"]["negatives_per_positive"] = 4
    config["runtime"]["device"] = "cpu"

    validate_phase3_experiment_config(config)
