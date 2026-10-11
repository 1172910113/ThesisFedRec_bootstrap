from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_experiment


def test_cli_loads_only_config_and_calls_high_level_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("experiment: {}\n", encoding="utf-8")
    config = {"validated": True}
    run_directory = tmp_path / "results" / "run"
    loaded_paths: list[Path] = []
    executed_configs: list[dict] = []

    def fake_load(path: Path) -> dict:
        loaded_paths.append(path)
        return config

    def fake_execute(executed_config: dict) -> SimpleNamespace:
        executed_configs.append(executed_config)
        return SimpleNamespace(run_directory=run_directory)

    monkeypatch.setattr(
        run_experiment,
        "load_phase3_experiment_config",
        fake_load,
    )
    monkeypatch.setattr(
        run_experiment,
        "execute_phase3_experiment",
        fake_execute,
    )

    exit_code = run_experiment.main(["--config", str(config_path)])

    assert exit_code == 0
    assert loaded_paths == [config_path]
    assert executed_configs == [config]
    assert capsys.readouterr().out.strip() == str(run_directory)


def test_cli_rejects_scientific_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"

    with pytest.raises(SystemExit) as error:
        run_experiment.main(
            ["--config", str(config_path), "--rounds", "2"]
        )

    assert error.value.code == 2
