import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_amazon_books import build_audit_report, main
from src.data import (
    audit_amazon_books_metadata,
    iter_amazon_books_reviews,
    preprocess_amazon_books,
)
from src.data.amazon_books import METADATA_PATH, REVIEWS_PATH
from src.data.loaders import resolve_data_path
from src.data.partition import build_hfl_partition
from src.data.preprocess import preprocess_ml1m
from src.utils.config import load_yaml


def _record(user: str = "U01", item: str = "000", rating: float = 5.0,
            timestamp: int = 10) -> dict:
    return {"reviewerID": user, "asin": item, "overall": rating,
            "unixReviewTime": timestamp, "reviewText": "never a training feature"}


def _write(path: Path, records: list[dict], *, literal: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for record in records:
            output.write((repr(record) if literal else json.dumps(record)) + "\n")


@pytest.fixture
def books_root(tmp_path: Path) -> Path:
    rows = [
        ("U02", "001", 5, 5), ("U01", "000", 1, 10),
        ("U02", "002", 2, 5), ("U01", "001", 4, 20),
        ("U02", "001", 3, 5), ("U02", "000", 5, 10),
        ("U01", "0", 2, 20), ("tiny", "hidden", 4, 2),
        ("short", "dup", 4, 1), ("short", "dup", 5, 2),
        ("short", "only", 4, 3),
    ]
    _write(tmp_path / REVIEWS_PATH, [_record(*row) for row in rows])
    _write(tmp_path / METADATA_PATH, [
        {"asin": "000", "title": "Zero", "price": 0, "categories": [["Books"]]},
        {"asin": "001", "title": "", "description": None, "related": {"also_bought": ["000"]}},
        {"asin": "000", "title": None, "description": "duplicate metadata, no feature merge"},
        {"asin": "0", "title": "Different", "imUrl": "image-reference-only"},
        {"asin": "metadata-only", "brand": "Publisher"},
    ], literal=True)
    return tmp_path


def test_books_loader_is_streaming_and_preserves_raw_fields(books_root: Path) -> None:
    reviews = iter_amazon_books_reviews(data_root=books_root)
    assert iter(reviews) is reviews
    assert next(reviews) == ("U02", "001", 5.0, 5)
    assert next(reviews) == ("U01", "000", 1.0, 10)
    assert len(list(reviews)) == 9


def test_books_exact_strings_source_order_and_temporal_split(books_root: Path) -> None:
    before = (books_root / REVIEWS_PATH).read_bytes()
    bundle = preprocess_amazon_books(data_root=books_root)
    again = preprocess_amazon_books(data_root=books_root, seed=2027)
    assert bundle.user_id_map == {"U01": 0, "U02": 1}
    assert bundle.item_id_map == {"000": 0, "002": 1, "001": 2, "0": 3}
    assert bundle.user_id_map == again.user_id_map
    assert bundle.item_id_map == again.item_id_map
    assert bundle.interactions.user_ids.tolist() == [0, 1, 0, 1, 1, 0]
    assert bundle.interactions.item_ids.tolist() == [0, 1, 2, 2, 0, 3]
    assert bundle.interactions.timestamps.tolist() == [10, 5, 20, 5, 10, 20]
    assert bundle.splits.train_interaction_ids.tolist() == [0, 1]
    assert bundle.splits.valid_interaction_ids.tolist() == [2, 3]
    assert bundle.splits.test_interaction_ids.tolist() == [4, 5]
    for field in ("interaction_ids", "user_ids", "item_ids", "timestamps", "values"):
        np.testing.assert_array_equal(getattr(bundle.interactions, field), getattr(again.interactions, field))
    for split in ("train", "valid", "test"):
        field = f"{split}_interaction_ids"
        np.testing.assert_array_equal(getattr(bundle.splits, field), getattr(again.splits, field))
    assert bundle.interactions.values.dtype == np.float32
    assert np.all(bundle.interactions.values == 1)
    assert bundle.feature_views == bundle.graph_views == {}
    assert bundle.parties == ()
    assert bundle.metadata["source"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert (books_root / REVIEWS_PATH).read_bytes() == before
    bundle.splits.validate_coverage(6)


def test_degree_audit_does_not_repair_the_five_core(books_root: Path) -> None:
    bundle = preprocess_amazon_books(data_root=books_root)
    assert bundle.metadata["counts"] == {
        "raw_interactions": 11, "raw_users": 4, "raw_items": 7,
        "after_initial_user_filter": 10, "after_duplicate_policy": 8,
        "duplicate_rows_removed": 2, "retained_interactions": 6,
        "users": 2, "items": 4, "train": 2, "valid": 2, "test": 2,
    }
    degrees = bundle.metadata["degree_audit"]
    assert degrees["raw_review_counts"]["users"]["minimum"] == 1
    assert degrees["after_deduplication"]["users"]["below_3"] == 1
    assert degrees["retained_unique_pair_counts"]["users"]["minimum"] == 3
    assert degrees["retained_unique_pair_counts"]["items"]["minimum"] == 1
    assert degrees["retained_unique_pair_counts"]["users"]["below_5"] == 2
    assert degrees["retained_unique_pair_counts"]["items"]["removed_original_entities"] == 3


def test_latest_duplicate_timestamp_wins_over_later_source_line(tmp_path: Path) -> None:
    _write(tmp_path / REVIEWS_PATH, [
        _record(item="A", timestamp=100), _record(item="B", timestamp=2),
        _record(item="C", timestamp=3), _record(item="A", timestamp=4),
    ])
    bundle = preprocess_amazon_books(data_root=tmp_path)
    assert bundle.item_id_map == {"A": 0, "B": 1, "C": 2}
    assert bundle.interactions.timestamps.tolist() == [100, 2, 3]
    assert bundle.splits.test_interaction_ids.tolist() == [0]


@pytest.mark.parametrize("literal", [False, True])
def test_json_and_safe_literal_review_formats(tmp_path: Path, literal: bool) -> None:
    records = [_record(item=item, timestamp=number) for number, item in enumerate(("000", "00A", "0"))]
    _write(tmp_path / REVIEWS_PATH, records, literal=literal)
    bundle = preprocess_amazon_books(data_root=tmp_path)
    assert bundle.item_id_map == {"000": 0, "00A": 1, "0": 2}


@pytest.mark.parametrize("field", ["reviewerID", "asin", "overall", "unixReviewTime"])
def test_missing_review_fields_fail(tmp_path: Path, field: str) -> None:
    record = _record()
    del record[field]
    _write(tmp_path / REVIEWS_PATH, [record])
    with pytest.raises(ValueError, match=f"line 1.*{field}"):
        preprocess_amazon_books(data_root=tmp_path)


@pytest.mark.parametrize(("field", "value"), [
    ("reviewerID", 123), ("reviewerID", "  "), ("asin", 123), ("asin", ""),
    ("overall", True), ("overall", "5"), ("overall", 0), ("overall", 6),
    ("overall", float("nan")), ("overall", float("inf")),
    ("unixReviewTime", True), ("unixReviewTime", "123"),
    ("unixReviewTime", 1.5), ("unixReviewTime", -1), ("unixReviewTime", 2**63),
])
def test_invalid_review_fields_are_not_silently_filtered(tmp_path: Path, field: str, value: object) -> None:
    record = _record()
    record[field] = value
    _write(tmp_path / REVIEWS_PATH, [record])
    with pytest.raises(ValueError, match="line 1"):
        preprocess_amazon_books(data_root=tmp_path)


@pytest.mark.parametrize("line", [
    "", "not a record", "[]", "{'asin': 'a', 'asin': 'b'}",
    '{"asin":"a","asin":"b"}', "{1: 'non-string-key'}",
    "__import__('os').system('false')", "dict(asin='a')",
])
def test_malformed_or_executable_records_fail_without_eval(tmp_path: Path, line: str) -> None:
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wt") as output:
        output.write(json.dumps(_record()) + "\n" + line + "\n")
    iterator = iter_amazon_books_reviews("bad.gz", data_root=tmp_path)
    assert next(iterator)[1] == "000"
    with pytest.raises(ValueError, match="line 2"):
        next(iterator)


def test_missing_empty_truncated_and_unsplittable_sources_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        preprocess_amazon_books(data_root=tmp_path)
    _write(tmp_path / REVIEWS_PATH, [])
    with pytest.raises(ValueError, match="empty source"):
        preprocess_amazon_books(data_root=tmp_path)
    _write(tmp_path / REVIEWS_PATH, [_record(), _record(), _record()])
    with pytest.raises(ValueError, match="no interactions remain"):
        preprocess_amazon_books(data_root=tmp_path)
    path = tmp_path / REVIEWS_PATH
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises((EOFError, gzip.BadGzipFile)):
        preprocess_amazon_books(data_root=tmp_path)


def test_metadata_coverage_duplicates_and_missingness_do_not_change_bundle(books_root: Path) -> None:
    bundle = preprocess_amazon_books(data_root=books_root)
    mapping = dict(bundle.item_id_map)
    original = (books_root / METADATA_PATH).read_bytes()
    audit = audit_amazon_books_metadata(bundle, data_root=books_root)
    assert audit["source_records"] == 5
    assert audit["unique_metadata_items"] == 4
    assert audit["matched_items"] == 3
    assert audit["missing_items"] == 1
    assert audit["missing_item_examples"] == ["002"]
    assert audit["excluded_metadata_only_items"] == 1
    assert audit["duplicate_records"] == audit["duplicate_metadata_ids"] == 1
    assert audit["duplicate_processed_item_ids"] == 1
    assert audit["field_coverage"]["title"] == {
        "source_records_present": 4, "source_records_nonempty": 2,
        "matched_items_present": 3, "matched_items_nonempty": 2,
    }
    assert audit["field_coverage"]["price"]["matched_items_nonempty"] == 1  # zero is observed
    assert audit["field_coverage"]["related.also_bought"]["matched_items_present"] == 1
    assert audit["field_coverage"]["brand"]["matched_items_present"] == 0
    assert bundle.item_id_map == mapping
    assert bundle.feature_views == bundle.graph_views == {}
    assert audit["source"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert (books_root / METADATA_PATH).read_bytes() == original


@pytest.mark.parametrize("record", [{}, {"asin": 1}, {"asin": " "}])
def test_invalid_metadata_ids_fail(books_root: Path, record: dict) -> None:
    bundle = preprocess_amazon_books(data_root=books_root)
    _write(books_root / METADATA_PATH, [record])
    with pytest.raises(ValueError, match="line 1.*asin"):
        audit_amazon_books_metadata(bundle, data_root=books_root)


def test_paths_use_environment_and_reject_escapes(books_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THESISFEDREC_DATA_ROOT", str(books_root))
    assert len(preprocess_amazon_books().interactions) == 6
    with pytest.raises(ValueError, match="relative"):
        preprocess_amazon_books(books_root / REVIEWS_PATH)
    with pytest.raises(ValueError, match="within"):
        preprocess_amazon_books("../outside.gz")


@pytest.mark.parametrize("seed", [True, -1, 2**32, "2026", 1.0])
def test_seed_validation_precedes_source_access(tmp_path: Path, seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        preprocess_amazon_books(data_root=tmp_path, seed=seed)


def test_hfl_ownership_and_ml1m_are_unchanged(books_root: Path, ml1m_ratings_file: Path) -> None:
    before = preprocess_ml1m(ml1m_ratings_file)
    bundle = preprocess_amazon_books(data_root=books_root)
    for mode, number in (("per_user", None), ("grouped_round_robin", 1)):
        parties = build_hfl_partition(bundle, mode=mode, num_parties=number, seed=2026)
        for split in ("train", "valid", "test"):
            combined = np.concatenate([getattr(p, f"{split}_interaction_ids") for p in parties])
            np.testing.assert_array_equal(np.sort(combined), getattr(bundle.splits, f"{split}_interaction_ids"))
        assert all(p.graph_view_names == p.feature_view_names == () for p in parties)
    after = preprocess_ml1m(ml1m_ratings_file)
    assert before.metadata == after.metadata
    assert before.user_id_map == after.user_id_map == {10: 0, 20: 1, 40: 2, 50: 3}
    np.testing.assert_array_equal(before.interactions.timestamps, after.interactions.timestamps)


def test_audit_command_reports_mismatches_and_writes_only_new_reports(books_root: Path, capsys: pytest.CaptureFixture) -> None:
    code = main(["--data-root", str(books_root), "--audit-output", "audits/test.json",
                 "--source-manifest", "manifests/test.json"])
    assert code == 1  # Synthetic inputs are not mislabeled as the complete official source.
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["input_archives_unchanged"]
    assert report["checks"]["chronological_validation"]
    assert report["checks"]["chronological_test"]
    assert not report["checks"]["reference_review_count"]
    assert not report["checks"]["reference_metadata_count"]
    assert json.loads((books_root / "audits/test.json").read_text()) == report
    assert not (books_root / "processed").exists()
    with pytest.raises(FileExistsError):
        main(["--data-root", str(books_root), "--audit-output", "audits/test.json"])
    with pytest.raises(ValueError, match="audits"):
        main(["--data-root", str(books_root), "--audit-output", "raw/report.json"])


def test_dataset_yaml_matches_standalone_defaults() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs/datasets/amazon_books_2014_5core.yaml")
    assert config["dataset"]["reviews_path"] == REVIEWS_PATH
    assert config["dataset"]["metadata_path"] == METADATA_PATH
    assert config["preprocessing"]["min_interactions_per_user"] == 3
    assert config["preprocessing"]["additional_k_core_pruning"] is False


@pytest.mark.skipif(os.environ.get("THESISFEDREC_RUN_REAL_BOOKS") != "1",
                    reason="set THESISFEDREC_RUN_REAL_BOOKS=1 for CPU-only official Books acceptance")
def test_real_amazon_books_data_acceptance() -> None:
    # Missing files raise; this opt-in never downloads or silently skips.
    report = build_audit_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    assert report["acceptance_passed"], report["checks"]
    assert report["processing"]["counts"]["valid"] == report["processing"]["counts"]["users"]
    assert report["processing"]["counts"]["test"] == report["processing"]["counts"]["users"]
    assert resolve_data_path(REVIEWS_PATH).is_file()
