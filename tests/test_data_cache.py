import json
from pathlib import Path

import numpy as np

from src.data.cache import load_base_cache, save_base_cache
from src.data.contracts import DatasetBundle


def test_base_cache_round_trip(
    processed_bundle: DatasetBundle, tmp_path: Path
) -> None:
    cache_path = tmp_path / "processed"
    manifest = save_base_cache(processed_bundle, cache_path)
    loaded = load_base_cache(cache_path)

    assert manifest["dataset_name"] == "ml1m"
    assert manifest["counts"] == {
        "users": 4,
        "items": 11,
        "interactions": 12,
        "train": 4,
        "valid": 4,
        "test": 4,
    }
    assert set(manifest["artifact_checksums"]) == {
        "base/interactions.npz",
        "base/splits.npz",
        "base/users.jsonl",
        "base/items.jsonl",
    }
    assert loaded.user_id_map == processed_bundle.user_id_map
    assert loaded.item_id_map == processed_bundle.item_id_map
    assert loaded.metadata == processed_bundle.metadata

    for name in (
        "interaction_ids",
        "user_ids",
        "item_ids",
        "timestamps",
        "values",
    ):
        np.testing.assert_array_equal(
            getattr(loaded.interactions, name),
            getattr(processed_bundle.interactions, name),
        )
    for name in (
        "train_interaction_ids",
        "valid_interaction_ids",
        "test_interaction_ids",
    ):
        np.testing.assert_array_equal(
            getattr(loaded.splits, name),
            getattr(processed_bundle.splits, name),
        )

    stored_manifest = json.loads(
        (cache_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert stored_manifest == manifest
