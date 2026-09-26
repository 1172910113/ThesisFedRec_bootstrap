"""YAML configuration loading."""

from pathlib import Path
from typing import Any

import yaml


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
