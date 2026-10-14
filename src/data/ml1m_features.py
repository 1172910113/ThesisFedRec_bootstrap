"""Standalone canonical ML1M item genres; no preprocessing or cache writes."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from src.data.contracts import DatasetBundle, FeatureView
from src.data.loaders import resolve_data_path

# MovieLens-1M README, MOVIES FILE DESCRIPTION; fixed lexical column order.
_GENRE_NAMES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)


def _ordered_item_mapping(bundle: DatasetBundle) -> list[tuple[int, int]]:
    mapping = bundle.item_id_map
    if not isinstance(mapping, dict):
        raise TypeError("item_id_map must be a dictionary")
    for raw_id, processed_id in mapping.items():
        if type(raw_id) is not int or type(processed_id) is not int:
            raise TypeError(
                "ML1M raw and processed item IDs must be built-in integers, "
                "excluding bool"
            )
        if raw_id <= 0:
            raise ValueError("ML1M raw item IDs must be positive")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("processed item IDs must form a bijection onto range(N)")
    return sorted(mapping.items(), key=lambda entry: entry[1])


def _parse_movies(text: str) -> dict[int, tuple[str, ...]]:
    movies: dict[int, tuple[str, ...]] = {}
    # Split on LF only: Latin-1 characters such as U+0085 are not newlines.
    for line_number, line in enumerate(io.StringIO(text, newline="\n"), start=1):
        if line.endswith("\n"):
            line = line[:-1]
            if line.endswith("\r"):
                line = line[:-1]
        fields = line.split("::")
        if len(fields) != 3:
            raise ValueError(f"movies.dat line {line_number}: expected three fields")
        raw_id_text, title, genres_text = fields
        if not raw_id_text.isascii() or not raw_id_text.isdecimal():
            raise ValueError(
                f"movies.dat line {line_number}: MovieID must be positive ASCII decimal"
            )
        try:
            raw_id = int(raw_id_text)
        except ValueError as error:
            raise ValueError(
                f"movies.dat line {line_number}: invalid MovieID"
            ) from error
        if raw_id <= 0:
            raise ValueError(f"movies.dat line {line_number}: MovieID must be positive")
        if raw_id in movies:
            raise ValueError(
                f"movies.dat line {line_number}: duplicate MovieID {raw_id}"
            )
        if not title.strip():
            raise ValueError(f"movies.dat line {line_number}: title must not be blank")
        genres = tuple(genres_text.split("|"))
        if any(not genre for genre in genres):
            raise ValueError(f"movies.dat line {line_number}: empty genre token")
        if len(set(genres)) != len(genres):
            raise ValueError(f"movies.dat line {line_number}: repeated genre token")
        for genre in genres:
            if genre not in _GENRE_NAMES:
                raise ValueError(
                    f"movies.dat line {line_number}: unknown genre token {genre!r}"
                )
        movies[raw_id] = genres
    if not movies:
        raise ValueError("movies.dat source must not be empty")
    return movies


def build_ml1m_item_genre_view(
    bundle: DatasetBundle,
    *,
    movies_relative_path: str | Path,
    data_root: str | Path | None = None,
) -> FeatureView:
    """Build fresh multi-hot genres in the existing processed item namespace.

    Read/hash the same source bytes and decode as Latin-1. Validate every
    metadata record, including unmatched movies; missing matched metadata is
    an error, not an observed-zero row. No randomness or input mutation occurs.
    The returned view remains mutable and unattached. Callers may explicitly
    attach it through a collision-checked new feature_views dictionary and
    dataclasses.replace; unchanged bundle objects then remain shallow-shared.
    """
    ordered_mapping = _ordered_item_mapping(bundle)
    source_path = resolve_data_path(movies_relative_path, data_root=data_root)
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    movies = _parse_movies(source_bytes.decode("latin-1"))

    missing = sorted(raw_id for raw_id, _ in ordered_mapping if raw_id not in movies)
    if missing:
        raise ValueError(
            f"missing metadata for {len(missing)} processed items; "
            f"representative raw MovieIDs: {missing[:10]}"
        )

    num_items = len(ordered_mapping)
    values = np.zeros((num_items, len(_GENRE_NAMES)), dtype=np.float32)
    genre_columns = {genre: column for column, genre in enumerate(_GENRE_NAMES)}
    for raw_id, processed_id in ordered_mapping:
        for genre in movies[raw_id]:
            values[processed_id, genre_columns[genre]] = 1.0

    # Match base/items.jsonl exactly, but never invoke the cache writer.
    mapping_jsonl = "".join(
        json.dumps(
            {"raw_id": raw_id, "processed_id": processed_id},
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n"
        for raw_id, processed_id in ordered_mapping
    )
    mapping_sha256 = hashlib.sha256(mapping_jsonl.encode("utf-8")).hexdigest()
    metadata = {
        "builder": "ml1m_item_genres",
        "builder_version": 1,
        "source_relative_path": Path(movies_relative_path).as_posix(),
        "source_encoding": "latin-1",
        "source_format": "MovieID::Title::Genres",
        "source_size_bytes": len(source_bytes),
        "source_sha256": source_sha256,
        "vocabulary_source": "MovieLens-1M README: MOVIES FILE DESCRIPTION",
        "vocabulary_policy": "fixed",
        "genre_vocabulary": list(_GENRE_NAMES),
        "encoding": "multi_hot",
        "row_order": "processed_item_id",
        "column_order": "lexical_genre_name",
        "raw_movie_id_policy": "positive_ascii_decimal",
        "genre_token_policy": "exact_known_unique_nonempty",
        "missing_metadata_policy": "error",
        "duplicate_movie_id_policy": "error",
        "metadata_only_policy": "validate_and_exclude",
        "item_mapping_sha256": mapping_sha256,
        "counts": {
            "source_records": len(movies),
            "unique_metadata_items": len(movies),
            "processed_items": num_items,
            "matched_items": num_items,
            "missing_items": 0,
            "metadata_only_items": len(movies) - num_items,
            "genre_count": len(_GENRE_NAMES),
            "positive_entries": int(np.count_nonzero(values)),
        },
    }
    return FeatureView(
        name="item_genres",
        entity_kind="item",
        entity_ids=np.arange(num_items, dtype=np.int64),
        modality="tabular",
        values=values,
        feature_names=_GENRE_NAMES,
        missing_mask=np.zeros(values.shape, dtype=np.bool_),
        metadata=metadata,
    )
