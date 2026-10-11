"""Run one validated Phase 3 experiment configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.federation.runner import execute_phase3_experiment
from src.utils.config import load_phase3_experiment_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one ThesisFedRec Phase 3 experiment.",
    )
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)

    config = load_phase3_experiment_config(arguments.config)
    execution = execute_phase3_experiment(config)
    print(execution.run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
