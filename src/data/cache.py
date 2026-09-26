"""Minimal base-cache serialization for processed datasets."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.contracts import DatasetBundle, InteractionSplits, InteractionTable

_REQUIRED_METADATA_KEYS = (
    "raw_ratings_sha256",
    "preprocessing_configuration",
    "preprocessing_seed",
    "mapping_policy",
    "minimum_interaction_rule",
    "duplicate_policy",
    "implicit_feedback_conversion",
    "split_policy",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_mapping(path: Path, mapping: dict[str | int, int]) -> None:
    with path.open("w", encoding="utf-8") as mapping_file:
        for raw_id, processed_id in sorted(
            mapping.items(), key=lambda entry: entry[1]
        ):
            record = {"raw_id": raw_id, "processed_id": processed_id}
            mapping_file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _read_mapping(path: Path) -> dict[str | int, int]:
    mapping: dict[str | int, int] = {}
    with path.open(encoding="utf-8") as mapping_file:
        for line_number, line in enumerate(mapping_file, start=1):
            record = json.loads(line)
            raw_id = record["raw_id"]
            processed_id = record["processed_id"]
            if raw_id in mapping:
                raise ValueError(
                    f"duplicate raw ID in {path} at line {line_number}"
                )
            mapping[raw_id] = processed_id
    return mapping


def save_base_cache(
    bundle: DatasetBundle, cache_directory: str | Path
) -> dict[str, Any]:
    """Save only base interactions, splits, mappings, and a manifest."""
    missing = [
        key for key in _REQUIRED_METADATA_KEYS if key not in bundle.metadata
    ]
    if missing:
        raise ValueError(
            "dataset metadata is missing required manifest values: "
            + ", ".join(missing)
        )

    cache_path = Path(cache_directory)
    base_path = cache_path / "base"
    base_path.mkdir(parents=True, exist_ok=True)

    interactions_path = base_path / "interactions.npz"
    splits_path = base_path / "splits.npz"
    users_path = base_path / "users.jsonl"
    items_path = base_path / "items.jsonl"

    np.savez_compressed(
        interactions_path,
        interaction_ids=bundle.interactions.interaction_ids,
        user_ids=bundle.interactions.user_ids,
        item_ids=bundle.interactions.item_ids,
        timestamps=bundle.interactions.timestamps,
        values=bundle.interactions.values,
    )
    np.savez_compressed(
        splits_path,
        train_interaction_ids=bundle.splits.train_interaction_ids,
        valid_interaction_ids=bundle.splits.valid_interaction_ids,
        test_interaction_ids=bundle.splits.test_interaction_ids,
    )
    _write_mapping(users_path, bundle.user_id_map)
    _write_mapping(items_path, bundle.item_id_map)

    artifacts = {
        "base/interactions.npz": interactions_path,
        "base/splits.npz": splits_path,
        "base/users.jsonl": users_path,
        "base/items.jsonl": items_path,
    }
    artifact_checksums = {
        relative_path: _sha256_file(path)
        for relative_path, path in artifacts.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": bundle.schema_version,
        "dataset_name": bundle.name,
        **{
            key: bundle.metadata[key]
            for key in _REQUIRED_METADATA_KEYS
        },
        "counts": {
            "users": len(bundle.user_id_map),
            "items": len(bundle.item_id_map),
            "interactions": len(bundle.interactions),
            "train": len(bundle.splits.train_interaction_ids),
            "valid": len(bundle.splits.valid_interaction_ids),
            "test": len(bundle.splits.test_interaction_ids),
        },
        "artifact_checksums": artifact_checksums,
        "dataset_metadata": bundle.metadata,
    }
    manifest_path = cache_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(
            manifest,
            manifest_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        manifest_file.write("\n")
    return manifest


def load_base_cache(cache_directory: str | Path) -> DatasetBundle:
    """Load and validate a minimal base cache."""
    cache_path = Path(cache_directory)
    manifest_path = cache_path / "manifest.json"
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    for relative_path, expected_checksum in manifest[
        "artifact_checksums"
    ].items():
        artifact_path = cache_path / relative_path
        actual_checksum = _sha256_file(artifact_path)
        if actual_checksum != expected_checksum:
            raise ValueError(f"cache checksum mismatch: {relative_path}")

    with np.load(
        cache_path / "base" / "interactions.npz", allow_pickle=False
    ) as stored:
        interactions = InteractionTable(
            interaction_ids=stored["interaction_ids"].copy(),
            user_ids=stored["user_ids"].copy(),
            item_ids=stored["item_ids"].copy(),
            timestamps=stored["timestamps"].copy(),
            values=stored["values"].copy(),
        )
    with np.load(
        cache_path / "base" / "splits.npz", allow_pickle=False
    ) as stored:
        splits = InteractionSplits(
            train_interaction_ids=stored["train_interaction_ids"].copy(),
            valid_interaction_ids=stored["valid_interaction_ids"].copy(),
            test_interaction_ids=stored["test_interaction_ids"].copy(),
        )

    bundle = DatasetBundle(
        name=manifest["dataset_name"],
        schema_version=manifest["schema_version"],
        interactions=interactions,
        splits=splits,
        user_id_map=_read_mapping(cache_path / "base" / "users.jsonl"),
        item_id_map=_read_mapping(cache_path / "base" / "items.jsonl"),
        metadata=manifest["dataset_metadata"],
    )
    expected_counts = manifest["counts"]
    actual_counts = {
        "users": len(bundle.user_id_map),
        "items": len(bundle.item_id_map),
        "interactions": len(bundle.interactions),
        "train": len(bundle.splits.train_interaction_ids),
        "valid": len(bundle.splits.valid_interaction_ids),
        "test": len(bundle.splits.test_interaction_ids),
    }
    if actual_counts != expected_counts:
        raise ValueError("cache counts do not match the manifest")
    return bundle
