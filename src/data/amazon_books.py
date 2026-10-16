"""Streaming Amazon Reviews 2014 Books 5-core data support; no persistence."""

import ast
import gzip
import json
import logging
import math
import os
from array import array
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from src.data.contracts import SCHEMA_VERSION, DatasetBundle, InteractionTable
from src.data.loaders import DATA_ROOT_ENV, resolve_data_path, resolve_data_root
from src.data.preprocess import (
    _apply_duplicate_policy,
    _filter_minimum_interactions,
    _select,
    _sha256_file,
)
from src.data.split import leave_one_out_split

SOURCE_PAGE = "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html"
SOURCE_BASE = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
REVIEWS_PATH = "raw/amazon_books_2014/reviews_Books_5.json.gz"
METADATA_PATH = "raw/amazon_books_2014/meta_Books.json.gz"
REFERENCE_REVIEW_COUNT = 8_898_041
REFERENCE_METADATA_COUNT = 2_370_585


def _resolve_books_data_root(data_root: str | Path | None = None) -> Path:
    """Use a project root; only Books falls back to repository-local data/."""
    if data_root is None:
        data_root = os.environ.get(DATA_ROOT_ENV) or (
            Path(__file__).resolve().parents[2] / "data"
        )
    return resolve_data_root(data_root)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("record dictionaries require unique string keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_record(line: str) -> dict[str, Any]:
    try:
        record = json.loads(
            line, object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError:
        # The original 2014 metadata may use Python dictionary literals.
        expression = ast.parse(line.strip(), mode="eval")
        for node in ast.walk(expression):
            if isinstance(node, ast.Dict):
                _unique_object([(ast.literal_eval(key), None) for key in node.keys])
        record = ast.literal_eval(expression)
    if type(record) is not dict:
        raise ValueError("record must be a dictionary")
    return record


def _iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Keep only one decoded record at a time, consuming gzip through its CRC."""
    opener = gzip.open if path.suffix == ".gz" else Path.open
    count = 0
    with opener(path, mode="rt", encoding="utf-8") as source:
        for count, line in enumerate(source, start=1):
            try:
                record = _parse_record(line)
            except (ValueError, SyntaxError, TypeError, RecursionError) as error:
                raise ValueError(f"{path.name}: malformed record at line {count}: {error}") from error
            yield count, record
    if count == 0:
        raise ValueError(f"empty source: {path}")


def _required_id(record: dict[str, Any], field: str, line: int) -> str:
    value = record.get(field)
    if type(value) is not str or not value.strip():
        raise ValueError(f"line {line}: {field} must be a nonblank string")
    return value  # Never strip, normalize, or cast an ASIN/reviewer ID.


def iter_amazon_books_reviews(
    reviews_relative_path: str | Path = REVIEWS_PATH,
    *,
    data_root: str | Path | None = None,
) -> Iterator[tuple[str, str, float, int]]:
    """Yield (reviewerID, asin, overall, unixReviewTime) in source-line order."""
    data_root = _resolve_books_data_root(data_root)
    path = resolve_data_path(reviews_relative_path, data_root=data_root)
    for line, record in _iter_records(path):
        user = _required_id(record, "reviewerID", line)
        item = _required_id(record, "asin", line)
        rating = record.get("overall")
        if type(rating) not in (int, float) or not 1 <= rating <= 5:
            raise ValueError(f"line {line}: overall must be a finite number in [1, 5]")
        if not math.isfinite(rating):
            raise ValueError(f"line {line}: overall must be finite")
        timestamp = record.get("unixReviewTime")
        if type(timestamp) is not int or not 0 <= timestamp <= np.iinfo(np.int64).max:
            raise ValueError(f"line {line}: unixReviewTime must be a non-negative int64 integer")
        yield user, item, float(rating), timestamp


def _file_state(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _source_identity(path: Path, relative_path: str | Path, *, reviews: bool) -> dict[str, Any]:
    return {
        "source_page": SOURCE_PAGE,
        "source_url": SOURCE_BASE + ("reviews_Books_5.json.gz" if reviews else "meta_Books.json.gz"),
        "source_variant": "2014_books_official_5core" if reviews else "2014_books_full_metadata",
        "relative_path": Path(relative_path).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "checksum_authority": "locally_computed_not_independently_published",
        "encoding": "utf-8",
        "compression": "gzip" if path.suffix == ".gz" else "none",
        "parser": "json_then_ast.literal_eval",
    }


def _degree_summary(columns: dict[str, np.ndarray], users: int, items: int) -> dict[str, Any]:
    summary = {}
    for name, size in (("users", users), ("items", items)):
        column = "raw_user_ids" if name == "users" else "raw_item_ids"
        degrees = np.bincount(columns[column], minlength=size)
        positive = degrees[degrees > 0]
        summary[name] = {
            "entities": int(positive.size),
            "minimum": int(positive.min()) if positive.size else 0,
            "maximum": int(positive.max()) if positive.size else 0,
            "below_3": int(np.count_nonzero((degrees > 0) & (degrees < 3))),
            "below_5": int(np.count_nonzero((degrees > 0) & (degrees < 5))),
            "removed_original_entities": int(np.count_nonzero(degrees == 0)),
        }
    return summary


def _map_strings(codes: np.ndarray, interned: dict[str, int]) -> tuple[dict[str, int], np.ndarray]:
    """Finalize IDs only after filtering; temporary codes are not public IDs."""
    unique, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    processed = np.empty(len(unique), dtype=np.int64)
    processed[order] = np.arange(len(unique), dtype=np.int64)
    raw_ids = tuple(interned)
    mapping = {raw_ids[int(unique[index])]: int(processed[index]) for index in order}
    return mapping, processed[inverse]


def preprocess_amazon_books(
    reviews_relative_path: str | Path = REVIEWS_PATH,
    *,
    data_root: str | Path | None = None,
    seed: int = 2026,
) -> DatasetBundle:
    """Build ID-only Books interactions using fixed, documented thesis defaults.

    Parsing is incremental. Compact integer columns and unique string tables
    are retained, never a list of review dictionaries or review-text features.
    No download, cache write, extra k-core pruning, or rating threshold occurs.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a Python integer in [0, 2**32)")
    data_root = _resolve_books_data_root(data_root)
    path = resolve_data_path(reviews_relative_path, data_root=data_root)
    original_state = _file_state(path)
    source = _source_identity(path, reviews_relative_path, reviews=True)
    users: dict[str, int] = {}
    items: dict[str, int] = {}
    user_codes, item_codes, timestamps = array("q"), array("q"), array("q")
    for user, item, _, timestamp in iter_amazon_books_reviews(
        reviews_relative_path, data_root=data_root
    ):
        user_codes.append(users.setdefault(user, len(users)))
        item_codes.append(items.setdefault(item, len(items)))
        timestamps.append(timestamp)
        if len(timestamps) % 1_000_000 == 0:
            logging.getLogger(__name__).info("Validated %s review records", len(timestamps))
    if _file_state(path) != original_state:
        raise ValueError("review source changed during processing")

    # Reuse only numeric-column helpers; ML1M's final mapper casts raw IDs to
    # int and is deliberately NOT used for Amazon's exact string identifiers.
    columns = {
        "raw_user_ids": np.asarray(user_codes, dtype=np.int64),
        "raw_item_ids": np.asarray(item_codes, dtype=np.int64),
        "timestamps": np.asarray(timestamps, dtype=np.int64),
        "source_indices": np.arange(len(timestamps), dtype=np.int64),
    }
    counts = {"raw_interactions": len(timestamps), "raw_users": len(users), "raw_items": len(items)}
    degrees = {"raw_review_counts": _degree_summary(columns, len(users), len(items))}
    columns = _filter_minimum_interactions(columns, 3)
    counts["after_initial_user_filter"] = len(columns["timestamps"])
    columns = _apply_duplicate_policy(columns, "keep_latest")
    counts["after_duplicate_policy"] = len(columns["timestamps"])
    counts["duplicate_rows_removed"] = counts["after_initial_user_filter"] - counts["after_duplicate_policy"]
    degrees["after_deduplication"] = _degree_summary(columns, len(users), len(items))
    columns = _filter_minimum_interactions(columns, 3)
    if not len(columns["timestamps"]):
        raise ValueError("no interactions remain after the split-minimum user filter")
    degrees["retained_unique_pair_counts"] = _degree_summary(columns, len(users), len(items))
    columns = _select(columns, np.argsort(columns["source_indices"], kind="stable"))
    user_map, user_ids = _map_strings(columns["raw_user_ids"], users)
    item_map, item_ids = _map_strings(columns["raw_item_ids"], items)
    interactions = InteractionTable(
        interaction_ids=np.arange(len(user_ids), dtype=np.int64),
        user_ids=user_ids,
        item_ids=item_ids,
        timestamps=columns["timestamps"],
        values=np.ones(len(user_ids), dtype=np.float32),
    )
    splits = leave_one_out_split(interactions)
    counts.update(
        retained_interactions=len(interactions), users=len(user_map), items=len(item_map),
        train=len(splits.train_interaction_ids), valid=len(splits.valid_interaction_ids),
        test=len(splits.test_interaction_ids),
    )
    return DatasetBundle(
        name="amazon_books_2014_5core", schema_version=SCHEMA_VERSION,
        interactions=interactions, splits=splits,
        user_id_map=user_map, item_id_map=item_map,
        metadata={
            "source": source, "preprocessing_version": 1, "preprocessing_seed": seed,
            "preprocessing_configuration": {
                "min_interactions_per_user": 3, "duplicate_policy": "keep_latest",
                "implicit_feedback": "all_observed", "canonical_order": "source_order",
                "additional_k_core_pruning": False,
            },
            "mapping_policy": "first_appearance_in_retained_canonical_source_order_exact_strings",
            "duplicate_policy": "keep_latest_timestamp_then_latest_source_line",
            "minimum_interaction_rule": "user_minimum_3_before_and_after_deduplication",
            "timestamp_unit": "unix_seconds", "timestamp_timezone": "UTC",
            "implicit_feedback_conversion": {"strategy": "all_observed", "positive_value": 1.0},
            "split_policy": {"strategy": "leave_one_out_temporal", "order": ["timestamp", "interaction_id"]},
            "counts": counts, "degree_audit": degrees,
        },
    )


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set, bytes)):
        return len(value) > 0
    return True  # Numeric zero / False are not implicitly missing.


def audit_amazon_books_metadata(
    bundle: DatasetBundle,
    metadata_relative_path: str | Path = METADATA_PATH,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Count metadata coverage without expanding IDs or constructing views.

    Duplicate ASINs are reported, not merged into feature payloads. Matched
    field coverage is a per-item union across records, explicitly audit-only.
    """
    if bundle.name != "amazon_books_2014_5core":
        raise ValueError("metadata audit requires an amazon_books_2014_5core bundle")
    mapping = bundle.item_id_map
    if not isinstance(mapping, dict) or any(
        type(raw) is not str or not raw.strip() or type(processed) is not int
        for raw, processed in mapping.items()
    ) or sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("item mapping must preserve string IDs and contiguous processed integers")
    data_root = _resolve_books_data_root(data_root)
    path = resolve_data_path(metadata_relative_path, data_root=data_root)
    original_state = _file_state(path)
    source = _source_identity(path, metadata_relative_path, reviews=False)
    seen: set[str] = set()
    duplicates: set[str] = set()
    matched = np.zeros(len(mapping), dtype=np.bool_)
    present: Counter[str] = Counter()
    nonempty: Counter[str] = Counter()
    matched_present: dict[str, np.ndarray] = {}
    matched_nonempty: dict[str, np.ndarray] = {}
    records = 0
    for records, record in _iter_records(path):
        if records % 1_000_000 == 0:
            logging.getLogger(__name__).info("Auditing metadata record %s", records)
        asin = _required_id(record, "asin", records)
        if asin in seen:
            duplicates.add(asin)
        seen.add(asin)
        item_id = mapping.get(asin)
        if item_id is not None:
            matched[item_id] = True
        fields = dict(record)
        if isinstance(record.get("related"), dict):
            for name, value in record["related"].items():
                fields[f"related.{name}"] = value
        for field, value in fields.items():
            if field not in matched_present:
                matched_present[field] = np.zeros(len(mapping), dtype=np.bool_)
                matched_nonempty[field] = np.zeros(len(mapping), dtype=np.bool_)
            present[field] += 1
            available = _has_value(value)
            nonempty[field] += int(available)
            if item_id is not None:
                matched_present[field][item_id] = True
                matched_nonempty[field][item_id] |= available
    if _file_state(path) != original_state:
        raise ValueError("metadata source changed during audit")
    matched_count = int(matched.sum())
    missing_examples = [raw for raw, index in sorted(mapping.items(), key=lambda pair: pair[1])
                        if not matched[index]][:10]
    return {
        "source": source, "source_records": records, "unique_metadata_items": len(seen),
        "processed_items": len(mapping), "matched_items": matched_count,
        "missing_items": len(mapping) - matched_count, "missing_item_examples": missing_examples,
        "duplicate_records": records - len(seen), "duplicate_metadata_ids": len(duplicates),
        "duplicate_processed_item_ids": sum(asin in mapping for asin in duplicates),
        "duplicate_id_examples": sorted(duplicates)[:10],
        "excluded_metadata_only_items": len(seen) - matched_count,
        "field_coverage": {
            field: {
                "source_records_present": present[field],
                "source_records_nonempty": nonempty[field],
                "matched_items_present": int(matched_present[field].sum()),
                "matched_items_nonempty": int(matched_nonempty[field].sum()),
            } for field in sorted(present)
        },
        "coverage_rule": "record_counts_and_unique_matched_item_union_not_feature_resolution",
        "use_restriction": "audit_only; relation_and_snapshot_time_policy_not_approved",
    }
