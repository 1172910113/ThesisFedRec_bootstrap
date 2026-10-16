"""CPU-only Books data acceptance and optional source/audit JSON reports."""

import argparse
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.amazon_books import (
    METADATA_PATH,
    REFERENCE_METADATA_COUNT,
    REFERENCE_REVIEW_COUNT,
    REVIEWS_PATH,
    _resolve_books_data_root,
    audit_amazon_books_metadata,
    preprocess_amazon_books,
)
from src.data.loaders import resolve_data_path
from src.data.preprocess import _sha256_file


def _mapping_checksum(mapping: dict[str, int]) -> str:
    digest = hashlib.sha256()
    for raw, processed in sorted(mapping.items(), key=lambda pair: pair[1]):
        record = json.dumps({"raw_id": raw, "processed_id": processed},
                            ensure_ascii=False, sort_keys=True) + "\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def build_audit_report(
    reviews_path: str | Path = REVIEWS_PATH,
    metadata_path: str | Path = METADATA_PATH,
    *,
    data_root: str | Path | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    """Validate locally available sources; never download or persist a dataset."""
    data_root = _resolve_books_data_root(data_root)
    logging.info("Processing Books reviews from %s", reviews_path)
    bundle = preprocess_amazon_books(reviews_path, data_root=data_root, seed=seed)
    logging.info("Auditing Books metadata from %s", metadata_path)
    coverage = audit_amazon_books_metadata(bundle, metadata_path, data_root=data_root)
    table = bundle.interactions
    chronological = np.lexsort((table.interaction_ids, table.timestamps, table.user_ids))
    ends = np.cumsum(np.bincount(table.user_ids))
    pairs = np.lexsort((table.item_ids, table.user_ids))
    unique_pairs = not np.any(
        (table.user_ids[pairs[1:]] == table.user_ids[pairs[:-1]])
        & (table.item_ids[pairs[1:]] == table.item_ids[pairs[:-1]])
    )
    sources = {"reviews": bundle.metadata["source"], "metadata": coverage["source"]}
    source_integrity = all(
        _sha256_file(resolve_data_path(source["relative_path"], data_root=data_root))
        == source["sha256"] for source in sources.values()
    )
    raw_degrees = bundle.metadata["degree_audit"]["raw_review_counts"]
    checks = {
        "reference_review_count": bundle.metadata["counts"]["raw_interactions"] == REFERENCE_REVIEW_COUNT,
        "reference_metadata_count": coverage["source_records"] == REFERENCE_METADATA_COUNT,
        "raw_review_degrees_at_least_5": all(raw_degrees[k]["minimum"] >= 5 for k in ("users", "items")),
        "unique_retained_user_item_pairs": bool(unique_pairs),
        "chronological_validation": np.array_equal(
            bundle.splits.valid_interaction_ids, np.sort(chronological[ends - 2])),
        "chronological_test": np.array_equal(
            bundle.splits.test_interaction_ids, np.sort(chronological[ends - 1])),
        "input_archives_unchanged": source_integrity,
        "empty_information_views": not bundle.feature_views and not bundle.graph_views,
    }
    bundle.splits.validate_coverage(len(table))
    checks["disjoint_complete_splits"] = True
    for kind, ids in (("users", table.user_ids), ("items", table.item_ids)):
        unique, first = np.unique(ids, return_index=True)
        checks[f"{kind}_first_appearance_mapping"] = np.array_equal(
            unique[np.argsort(first)], np.arange(len(unique)))
    arrays = {name: getattr(table, name) for name in (
        "interaction_ids", "user_ids", "item_ids", "timestamps", "values")}
    arrays.update({f"{name}_split": getattr(bundle.splits, f"{name}_interaction_ids")
                   for name in ("train", "valid", "test")})
    return {
        "dataset": bundle.name, "acceptance_passed": all(checks.values()),
        "checks": {key: bool(value) for key, value in checks.items()},
        "reference_counts": {"reviews": REFERENCE_REVIEW_COUNT, "metadata": REFERENCE_METADATA_COUNT},
        "processing": bundle.metadata, "metadata_coverage": coverage,
        "sources": sources,
        "mapping_sha256": {"users": _mapping_checksum(bundle.user_id_map),
                           "items": _mapping_checksum(bundle.item_id_map)},
        "array_sha256": {name: hashlib.sha256(memoryview(values)).hexdigest()
                         for name, values in arrays.items()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "implementation_sha256": {
            name: _sha256_file(REPOSITORY_ROOT / name)
            for name in ("src/data/amazon_books.py", "src/data/preprocess.py", "src/data/split.py",
                         "src/data/contracts.py", "scripts/audit_amazon_books.py")
        },
        "scope": "CPU data acceptance only; no processed cache, features, graph, or training",
    }


def _report_path(relative: str | None, directory: str, data_root: str | Path | None) -> Path | None:
    if relative is None:
        return None
    data_root = _resolve_books_data_root(data_root)
    path = resolve_data_path(relative, data_root=data_root)
    parts = path.relative_to(data_root).parts
    if len(parts) < 2 or parts[0] != directory or path.suffix != ".json":
        raise ValueError(f"report must be a .json path under {directory}/")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-path", default=REVIEWS_PATH)
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    parser.add_argument("--data-root")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--audit-output", help="new data-root-relative audits/*.json")
    parser.add_argument("--source-manifest", help="new data-root-relative manifests/*.json")
    arguments = parser.parse_args(argv)
    data_root = _resolve_books_data_root(arguments.data_root)
    audit_path = _report_path(arguments.audit_output, "audits", data_root)
    manifest_path = _report_path(arguments.source_manifest, "manifests", data_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = build_audit_report(arguments.reviews_path, arguments.metadata_path,
                               data_root=data_root, seed=arguments.seed)
    for path, payload in ((audit_path, report), (manifest_path, report["sources"])):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as output:
                json.dump(payload, output, indent=2, sort_keys=True, ensure_ascii=False)
                output.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
