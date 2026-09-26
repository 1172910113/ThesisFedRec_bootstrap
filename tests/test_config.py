from pathlib import Path

from src.utils import load_yaml


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
