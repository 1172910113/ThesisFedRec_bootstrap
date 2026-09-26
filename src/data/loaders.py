"""Explicit raw-data loading and data-root resolution."""

import os
from pathlib import Path

import numpy as np

DATA_ROOT_ENV = "THESISFEDREC_DATA_ROOT"


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve an explicit root, the environment root, or local data/raw."""
    if data_root is not None:
        return Path(data_root).expanduser().resolve()

    environment_root = os.environ.get(DATA_ROOT_ENV)
    if environment_root:
        return Path(environment_root).expanduser().resolve()

    repository_root = Path(__file__).resolve().parents[2]
    return (repository_root / "data" / "raw").resolve()


def resolve_data_path(
    relative_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Resolve a dataset-relative path without allowing root traversal."""
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("dataset paths must be relative to the configured data root")

    root = resolve_data_root(data_root)
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("dataset path must stay within the configured data root") from error
    return resolved


def load_ml1m_ratings(ratings_path: str | Path) -> dict[str, np.ndarray]:
    """Parse a standard MovieLens-1M ratings.dat file in source order."""
    path = Path(ratings_path)
    raw_user_ids: list[int] = []
    raw_item_ids: list[int] = []
    ratings: list[float] = []
    timestamps: list[int] = []

    with path.open(encoding="utf-8") as ratings_file:
        for source_index, line in enumerate(ratings_file):
            fields = line.rstrip("\n\r").split("::")
            if len(fields) != 4:
                raise ValueError(
                    f"invalid ratings.dat record at line {source_index + 1}"
                )
            try:
                raw_user_id = int(fields[0])
                raw_item_id = int(fields[1])
                rating = float(fields[2])
                timestamp = int(fields[3])
            except ValueError as error:
                raise ValueError(
                    f"non-numeric ratings.dat field at line {source_index + 1}"
                ) from error

            raw_user_ids.append(raw_user_id)
            raw_item_ids.append(raw_item_id)
            ratings.append(rating)
            timestamps.append(timestamp)

    if not raw_user_ids:
        raise ValueError(f"ratings.dat is empty: {path}")

    size = len(raw_user_ids)
    return {
        "raw_user_ids": np.asarray(raw_user_ids, dtype=np.int64),
        "raw_item_ids": np.asarray(raw_item_ids, dtype=np.int64),
        "ratings": np.asarray(ratings, dtype=np.float32),
        "timestamps": np.asarray(timestamps, dtype=np.int64),
        "source_indices": np.arange(size, dtype=np.int64),
    }
