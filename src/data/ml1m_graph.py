"""Narrow identity and canonical graph conversion for complete ML1M genres.

The caller keeps inputs stable during each call. Source metadata is checked
as an assertion; this module does not read or authenticate a raw source file.
"""

import hashlib
import json
import unicodedata

import numpy as np

from src.data.contracts import DatasetBundle, FeatureView, GraphView
from src.data.graph import graph_domain_fingerprint


_GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
_INT64_MAX = 2**63 - 1
_FIXED_METADATA = {
    "builder": "ml1m_item_genres",
    "source_encoding": "latin-1",
    "source_format": "MovieID::Title::Genres",
    "vocabulary_source": "MovieLens-1M README: MOVIES FILE DESCRIPTION",
    "vocabulary_policy": "fixed",
    "encoding": "multi_hot",
    "row_order": "processed_item_id",
    "column_order": "lexical_genre_name",
    "raw_movie_id_policy": "positive_ascii_decimal",
    "genre_token_policy": "exact_known_unique_nonempty",
    "missing_metadata_policy": "error",
    "duplicate_movie_id_policy": "error",
    "metadata_only_policy": "validate_and_exclude",
}
_COUNT_KEYS = (
    "source_records", "unique_metadata_items", "processed_items",
    "matched_items", "missing_items", "metadata_only_items", "genre_count",
    "positive_entries",
)


def _capacity(num_items: int, edge_count: int) -> None:
    if num_items + 18 > _INT64_MAX or edge_count > _INT64_MAX:
        raise ValueError("graph node/edge counts exceed int64 capacity")


def _integer(value: int, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if not minimum <= value <= _INT64_MAX:
        raise ValueError(f"{name} is out of range")


def _digest(value: str, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in string")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


def _fixed_text(value: str, expected: str, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in string")
    if value != expected:
        raise ValueError(f"unsupported {name}")


def _array(value: np.ndarray, dtype: str, name: str) -> None:
    if (
        type(value) is not np.ndarray or value.dtype.metadata is not None
        or not value.dtype.isnative or value.dtype != np.dtype(dtype)
    ):
        raise TypeError(f"{name} must be a plain native metadata-free {dtype} array")


def _item_mapping(bundle: DatasetBundle) -> tuple[tuple[int, ...], str]:
    if type(bundle) is not DatasetBundle:
        raise TypeError("bundle must be an exact DatasetBundle")
    mapping = bundle.item_id_map
    if type(mapping) is not dict:
        raise TypeError("item_id_map must be a plain dictionary")
    _capacity(len(mapping), 0)
    for raw_id, processed_id in mapping.items():
        if type(raw_id) is not int or type(processed_id) is not int:
            raise TypeError("item mapping IDs must be exact built-in integers")
        if raw_id <= 0:
            raise ValueError("raw MovieIDs must be positive")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("item mapping must be a bijection onto range(I)")
    ordered = sorted(mapping.items(), key=lambda pair: pair[1])
    data = "".join(
        json.dumps({"raw_id": raw, "processed_id": row},
                   ensure_ascii=False, sort_keys=True) + "\n"
        for raw, row in ordered
    ).encode("utf-8")
    return tuple(raw for raw, _ in ordered), hashlib.sha256(data).hexdigest()


def _metadata(view: FeatureView, n: int, edges: int, mapping_sha: str) -> dict:
    metadata = view.metadata
    if type(metadata) is not dict:
        raise TypeError("metadata must be a plain dictionary")
    # Look up only this whitelist. Unknown keys/values are never traversed.
    keys = (*_FIXED_METADATA, "builder_version", "source_relative_path",
            "source_size_bytes", "source_sha256", "genre_vocabulary",
            "item_mapping_sha256", "counts")
    if any(key not in metadata for key in keys):
        raise ValueError("missing required source metadata")
    result = {}
    for name, expected in _FIXED_METADATA.items():
        _fixed_text(metadata[name], expected, name)
        result[name] = metadata[name]
    _integer(metadata["builder_version"], "builder_version", minimum=1)
    if metadata["builder_version"] != 1:
        raise ValueError("unsupported builder_version")
    result["builder_version"] = metadata["builder_version"]
    path = metadata["source_relative_path"]
    if type(path) is not str:
        raise TypeError("source_relative_path must be an exact built-in string")
    if not path or path.strip() != path or any(unicodedata.category(c) == "Cc" for c in path):
        raise ValueError("source_relative_path must be nonblank identity text")
    result["source_relative_path"] = path
    _integer(metadata["source_size_bytes"], "source_size_bytes", minimum=1)
    result["source_size_bytes"] = metadata["source_size_bytes"]
    for name in ("source_sha256", "item_mapping_sha256"):
        _digest(metadata[name], name)
        result[name] = metadata[name]
    if result["item_mapping_sha256"] != mapping_sha:
        raise ValueError("item_mapping_sha256 does not match current item mapping")
    vocabulary = metadata["genre_vocabulary"]
    if type(vocabulary) is not list or any(type(v) is not str for v in vocabulary):
        raise TypeError("genre_vocabulary must be a list of exact built-in strings")
    if tuple(vocabulary) != _GENRES:
        raise ValueError("genre_vocabulary must match fixed lexical genres")
    result["genre_vocabulary"] = list(vocabulary)
    counts = metadata["counts"]
    if type(counts) is not dict or any(type(key) is not str for key in counts):
        raise TypeError("counts must be a plain dictionary with plain string keys")
    if set(counts) != set(_COUNT_KEYS):
        raise ValueError("counts must contain exactly the eight declared keys")
    for name in _COUNT_KEYS:
        _integer(counts[name], f"counts.{name}")
    expected_counts = {
        "source_records": n + counts["metadata_only_items"],
        "unique_metadata_items": n + counts["metadata_only_items"],
        "processed_items": n, "matched_items": n, "missing_items": 0,
        "metadata_only_items": counts["metadata_only_items"],
        "genre_count": 18, "positive_entries": edges,
    }
    if counts != expected_counts or counts["source_records"] == 0:
        raise ValueError("source counts disagree with the complete genre view")
    result["counts"] = {name: counts[name] for name in _COUNT_KEYS}
    return result


def _validated_inputs(bundle: DatasetBundle, view: FeatureView) -> tuple:
    raw_ids, mapping_sha = _item_mapping(bundle)
    n = len(raw_ids)
    if type(view) is not FeatureView:
        raise TypeError("genre_view must be an exact FeatureView")
    for name, expected in (("name", "item_genres"), ("entity_kind", "item"),
                           ("modality", "tabular")):
        _fixed_text(getattr(view, name), expected, name)
    if type(view.feature_names) is not tuple or any(type(v) is not str for v in view.feature_names):
        raise TypeError("feature_names must be a tuple of exact built-in strings")
    if view.feature_names != _GENRES:
        raise ValueError("feature_names must match fixed lexical genres")
    _array(view.entity_ids, "int64", "entity_ids")
    _array(view.values, "float32", "values")
    if view.missing_mask is not None:
        _array(view.missing_mask, "bool", "missing_mask")
    if view.entity_ids.shape != (n,) or not np.array_equal(view.entity_ids, np.arange(n, dtype=np.int64)):
        raise ValueError("entity_ids must be exactly the complete processed catalogue")
    if view.values.shape != (n, 18):
        raise ValueError("values must have shape (I,18)")
    if not np.isfinite(view.values).all() or not ((view.values == 0) | (view.values == 1)).all():
        raise ValueError("values must be finite binary cells")
    if n and not np.any(view.values == 1, axis=1).all():
        raise ValueError("every item must have at least one observed genre")
    if view.missing_mask is not None and (
        view.missing_mask.shape != (n, 18) or np.any(view.missing_mask)
    ):
        raise ValueError("missing_mask must have shape (I,18) and be all False")
    edges = int(np.count_nonzero(view.values))
    _capacity(n, edges)
    metadata = _metadata(view, n, edges, mapping_sha)
    domain = graph_domain_fingerprint(bundle)
    return raw_ids, mapping_sha, domain, metadata


def _array_identity(array: np.ndarray, dtype: str, canonical_dtype: str) -> dict:
    return {
        "shape": list(array.shape), "dtype": dtype,
        "sha256": hashlib.sha256(array.astype(canonical_dtype, copy=False).tobytes(order="C")).hexdigest(),
    }


def _binding(view: FeatureView, mapping_sha: str, domain: str, metadata: dict) -> str:
    payload = {
        "format": "thesisfedrec.ml1m_item_genre_feature_binding", "version": 1,
        "domain_fingerprint": domain, "item_mapping_sha256": mapping_sha,
        "view": {
            "name": view.name, "entity_kind": view.entity_kind, "modality": view.modality,
            "feature_names": list(view.feature_names),
            "entity_ids": _array_identity(view.entity_ids, "int64", "<i8"),
            "values": _array_identity(view.values, "float32", "<f4"),
            "missing_mask": None if view.missing_mask is None else _array_identity(view.missing_mask, "bool", "?"),
        },
        "source_metadata": metadata,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ml1m_item_genre_feature_binding(
    bundle: DatasetBundle, *, genre_view: FeatureView,
) -> str:
    """Validate the complete narrow source and bind its logical bytes/context.

    This digest checks consistency, not raw-source provenance or authentication.
    No caller arrays, maps or metadata are retained or modified.
    """
    _, mapping_sha, domain, metadata = _validated_inputs(bundle, genre_view)
    return _binding(genre_view, mapping_sha, domain, metadata)


def build_ml1m_item_genre_graph(
    bundle: DatasetBundle, *, genre_view: FeatureView,
    expected_feature_binding: str,
) -> GraphView:
    """Return an unattached owning canonical graph after trusted expected binding.

    The caller obtains the expected digest from its frozen trusted bookkeeping,
    not merely from a newly received view. Returned arrays remain mutable.
    """
    _digest(expected_feature_binding, "expected_feature_binding")
    raw_ids, mapping_sha, domain, metadata = _validated_inputs(bundle, genre_view)
    actual = _binding(genre_view, mapping_sha, domain, metadata)
    if actual != expected_feature_binding:
        raise ValueError("feature binding mismatch")
    n = len(raw_ids)
    rows, columns = np.nonzero(genre_view.values)
    edge_count = int(rows.size)
    _capacity(n, edge_count)
    edges = np.empty((2, edge_count), dtype=np.int64)
    edges[0] = rows
    edges[1] = n + columns
    return GraphView(
        name="item_genres_graph", node_raw_ids=raw_ids + _GENRES,
        node_type_ids=np.concatenate((np.zeros(n, dtype=np.int64), np.ones(18, dtype=np.int64))),
        node_type_names=("item", "genre"), edge_index=edges,
        relation_ids=np.zeros(edge_count, dtype=np.int64), relation_names=("has_genre",),
        user_to_node=None, item_to_node=np.arange(n, dtype=np.int64),
        metadata={
            "builder": "ml1m_item_genre_graph", "builder_version": 1,
            "schema_revision": "ml1m.item-genre.v1", "source_feature_binding": actual,
            "source_domain_fingerprint": domain, "source_item_mapping_sha256": mapping_sha,
            "source_sha256": metadata["source_sha256"],
            "counts": {"item_count": n, "genre_count": 18, "edge_count": edge_count},
        },
    )
