from __future__ import annotations

import csv
import json
import os
from math import isfinite
from pathlib import Path

import pytest

from scripts import run_experiment

_REPOSITORY_ROOT = Path(__file__).parents[1]
_SMOKE_CONFIG = (
    _REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "ml1m_fedgmf_fedavg_smoke.yaml"
)
_EXPECTED_ML1M_SHA256 = (
    "506d64ca44484487c11dc2d9a28de5c"
    "54948213e6b96285e298afe28d6ea4e0f"
)


@pytest.mark.skipif(
    os.environ.get("THESISFEDREC_RUN_REAL_SMOKE") != "1",
    reason="set THESISFEDREC_RUN_REAL_SMOKE=1 for the real ML1M/GPU smoke",
)
def test_real_ml1m_smoke_through_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(_REPOSITORY_ROOT)

    assert run_experiment.main(["--config", str(_SMOKE_CONFIG)]) == 0

    run_directory = Path(capsys.readouterr().out.strip().splitlines()[-1])
    mandatory = {
        "config.yaml",
        "metrics.json",
        "round_metrics.csv",
        "runtime.json",
        "partition.json",
        "run.log",
    }
    assert mandatory <= {path.name for path in run_directory.iterdir()}
    assert not (run_directory / "checkpoint.pt").exists()

    metrics = json.loads(
        (run_directory / "metrics.json").read_text(encoding="utf-8")
    )
    for split in ("validation", "test"):
        for name in ("hr@10", "ndcg@10"):
            assert isfinite(metrics[split][name])
            assert 0.0 <= metrics[split][name] <= 1.0

    with (run_directory / "round_metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert int(rows[0]["selected_party_count"]) == 1

    runtime = json.loads(
        (run_directory / "runtime.json").read_text(encoding="utf-8")
    )
    assert runtime["configured_device"] == "cuda:0"
    assert runtime["cuda_available"] is True
    assert runtime["selected_device_name"]
    assert runtime["dataset_identity"]["raw_ratings_sha256"] == (
        _EXPECTED_ML1M_SHA256
    )
