from pathlib import Path

import numpy as np
import pytest

from src.data.loaders import (
    load_ml1m_ratings,
    resolve_data_path,
    resolve_data_root,
)
from src.data.preprocess import preprocess_ml1m


def test_ml1m_loader_preserves_raw_values_and_source_order(
    ml1m_ratings_file: Path,
) -> None:
    raw = load_ml1m_ratings(ml1m_ratings_file)

    assert raw["raw_user_ids"][:3].tolist() == [10, 10, 10]
    assert raw["raw_item_ids"][:3].tolist() == [100, 101, 102]
    assert raw["ratings"][:3].tolist() == [5.0, 4.0, 3.0]
    assert raw["timestamps"][:3].tolist() == [10, 20, 20]
    np.testing.assert_array_equal(
        raw["source_indices"],
        np.arange(15, dtype=np.int64),
    )


def test_data_root_prefers_environment_and_paths_remain_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THESISFEDREC_DATA_ROOT", str(tmp_path))

    assert resolve_data_root() == tmp_path.resolve()
    assert resolve_data_path("ml-1m/ml-1m/ratings.dat") == (
        tmp_path / "ml-1m" / "ml-1m" / "ratings.dat"
    )
    with pytest.raises(ValueError, match="must be relative"):
        resolve_data_path("/data/somewhere/ratings.dat")
    with pytest.raises(ValueError, match="stay within"):
        resolve_data_path("../ratings.dat")


def test_preprocessing_is_deterministic_and_ids_are_contiguous(
    ml1m_ratings_file: Path,
) -> None:
    first = preprocess_ml1m(ml1m_ratings_file)
    second = preprocess_ml1m(ml1m_ratings_file)

    assert first.user_id_map == second.user_id_map
    assert first.item_id_map == second.item_id_map
    assert first.user_id_map == {10: 0, 20: 1, 40: 2, 50: 3}
    assert sorted(first.user_id_map.values()) == list(range(4))
    assert sorted(first.item_id_map.values()) == list(range(11))
    np.testing.assert_array_equal(
        first.interactions.user_ids,
        second.interactions.user_ids,
    )
    np.testing.assert_array_equal(
        first.interactions.item_ids,
        second.interactions.item_ids,
    )
    np.testing.assert_array_equal(
        first.interactions.interaction_ids,
        np.arange(12, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        first.interactions.values,
        np.ones(12, dtype=np.float32),
    )


def test_minimum_filtering_and_duplicate_policy(
    ml1m_ratings_file: Path,
) -> None:
    bundle = preprocess_ml1m(ml1m_ratings_file)

    assert 30 not in bundle.user_id_map
    assert len(bundle.interactions) == 12
    user_40 = bundle.user_id_map[40]
    item_105 = bundle.item_id_map[105]
    rows = np.flatnonzero(
        (bundle.interactions.user_ids == user_40)
        & (bundle.interactions.item_ids == item_105)
    )
    assert rows.tolist() == [6]
    assert bundle.interactions.timestamps[rows[0]] == 2


def test_leave_one_out_uses_interaction_id_to_break_timestamp_ties(
    ml1m_ratings_file: Path,
) -> None:
    bundle = preprocess_ml1m(ml1m_ratings_file)

    assert 1 in bundle.splits.valid_interaction_ids
    assert 2 in bundle.splits.test_interaction_ids
    assert 0 in bundle.splits.train_interaction_ids

    train = set(bundle.splits.train_interaction_ids.tolist())
    valid = set(bundle.splits.valid_interaction_ids.tolist())
    test = set(bundle.splits.test_interaction_ids.tolist())
    assert train.isdisjoint(valid)
    assert train.isdisjoint(test)
    assert valid.isdisjoint(test)
    assert train | valid | test == set(range(len(bundle.interactions)))

    valid_users = bundle.interactions.user_ids[
        bundle.splits.valid_interaction_ids
    ]
    test_users = bundle.interactions.user_ids[
        bundle.splits.test_interaction_ids
    ]
    assert sorted(valid_users.tolist()) == list(range(4))
    assert sorted(test_users.tolist()) == list(range(4))
