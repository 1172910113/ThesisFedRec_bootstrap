import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data.cache import load_base_cache, save_base_cache
from src.data.contracts import DatasetBundle, FeatureView


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
    assert processed_bundle.feature_views == {}
    assert loaded.feature_views == {}

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


def test_base_cache_rejects_modified_artifact(
    processed_bundle: DatasetBundle, tmp_path: Path
) -> None:
    cache_path = tmp_path / "processed"
    save_base_cache(processed_bundle, cache_path)

    interactions_path = cache_path / "base" / "interactions.npz"
    interactions_path.write_bytes(interactions_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="cache checksum mismatch"):
        load_base_cache(cache_path)


def test_base_cache_rejects_unsupported_schema_version(
    processed_bundle: DatasetBundle, tmp_path: Path
) -> None:
    cache_path = tmp_path / "processed"
    save_base_cache(processed_bundle, cache_path)

    manifest_path = cache_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "999.0"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported cache schema version"):
        load_base_cache(cache_path)


def test_schema_1_base_cache_rejects_non_empty_feature_views(
    processed_bundle: DatasetBundle, tmp_path: Path
) -> None:
    view = FeatureView(
        name="item_numeric",
        entity_kind="item",
        entity_ids=np.array([0], dtype=np.int64),
        modality="tabular",
        values=np.array([[1.0]], dtype=np.float32),
        feature_names=("value",),
        missing_mask=None,
        metadata={},
    )
    bundle = replace(
        processed_bundle,
        feature_views={view.name: view},
    )
    cache_path = tmp_path / "processed"

    with pytest.raises(
        ValueError,
        match="schema 1.0 cannot persist non-empty FeatureView payloads",
    ):
        save_base_cache(bundle, cache_path)

    assert not cache_path.exists()
