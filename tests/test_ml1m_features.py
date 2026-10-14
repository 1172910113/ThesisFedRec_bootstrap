import hashlib
import json
import os
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pytest

from src.data import (
    DatasetBundle,
    FeatureView,
    InteractionSplits,
    InteractionTable,
    build_complete_feature_scenario,
    build_hfl_partition,
    build_ml1m_item_genre_view,
    build_single_view_absence_scenario,
    loaders,
    preprocess_ml1m_from_root,
    validate_feature_partition,
)
from src.data.cache import save_base_cache

_EXPECTED_GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
_MOVIE_RECORDS = (
    "7::Amélie (2001)::Drama|Comedy",
    "42::Third::Action",
    "999::Metadata only::Western",
    "90::First::Sci-Fi|Action",
)
_MAPPING_JSONL = (
    b'{"processed_id": 0, "raw_id": 90}\n'
    b'{"processed_id": 1, "raw_id": 7}\n'
    b'{"processed_id": 2, "raw_id": 42}\n'
)


@pytest.fixture
def item_bundle() -> DatasetBundle:
    return DatasetBundle(
        name="ml1m", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=np.arange(3, dtype=np.int64),
            user_ids=np.zeros(3, dtype=np.int64),
            item_ids=np.arange(3, dtype=np.int64),
            timestamps=np.arange(3, dtype=np.int64),
            values=np.ones(3, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=np.array([0], dtype=np.int64),
            valid_interaction_ids=np.array([1], dtype=np.int64),
            test_interaction_ids=np.array([2], dtype=np.int64),
        ),
        user_id_map={11: 0}, item_id_map={42: 2, 7: 1, 90: 0},
        metadata={"untouched": {"nested": [1, 2]}},
    )


@pytest.fixture
def movies_file(tmp_path: Path) -> Path:
    path = tmp_path / "movies.dat"
    path.write_bytes(("\n".join(_MOVIE_RECORDS) + "\n").encode("latin-1"))
    return path


def _build(bundle: DatasetBundle, source: Path) -> FeatureView:
    return build_ml1m_item_genre_view(
        bundle, movies_relative_path=source.name, data_root=source.parent,
    )


def test_encoding_and_exact_provenance(item_bundle, movies_file):
    view = _build(item_bundle, movies_file)

    assert (view.name, view.entity_kind, view.modality) == (
        "item_genres", "item", "tabular",
    )
    np.testing.assert_array_equal(view.entity_ids, [0, 1, 2])
    assert view.entity_ids.dtype == np.int64
    assert view.entity_ids.ndim == 1
    assert view.feature_names == _EXPECTED_GENRES == tuple(sorted(_EXPECTED_GENRES))
    expected = np.zeros((3, 18), dtype=np.float32)
    expected[0, [0, 14]] = 1  # raw 90: Action, Sci-Fi
    expected[1, [4, 7]] = 1  # raw 7: Comedy, Drama
    expected[2, 0] = 1  # raw 42: Action
    np.testing.assert_array_equal(view.values, expected)
    assert view.values.dtype == np.float32
    assert view.missing_mask.dtype == np.bool_
    np.testing.assert_array_equal(view.missing_mask, np.zeros((3, 18), dtype=bool))
    # Western occurs only in unmatched metadata; its fixed column is observed zero.
    assert not view.values[:, 17].any()
    assert view.metadata == {
        "builder": "ml1m_item_genres", "builder_version": 1,
        "source_relative_path": "movies.dat", "source_encoding": "latin-1",
        "source_format": "MovieID::Title::Genres",
        "source_size_bytes": len(movies_file.read_bytes()),
        "source_sha256": hashlib.sha256(movies_file.read_bytes()).hexdigest(),
        "vocabulary_source": "MovieLens-1M README: MOVIES FILE DESCRIPTION",
        "vocabulary_policy": "fixed", "genre_vocabulary": list(_EXPECTED_GENRES),
        "encoding": "multi_hot", "row_order": "processed_item_id",
        "column_order": "lexical_genre_name",
        "raw_movie_id_policy": "positive_ascii_decimal",
        "genre_token_policy": "exact_known_unique_nonempty",
        "missing_metadata_policy": "error", "duplicate_movie_id_policy": "error",
        "metadata_only_policy": "validate_and_exclude",
        "item_mapping_sha256": hashlib.sha256(_MAPPING_JSONL).hexdigest(),
        "counts": {
            "source_records": 4, "unique_metadata_items": 4,
            "processed_items": 3, "matched_items": 3, "missing_items": 0,
            "metadata_only_items": 1, "genre_count": 18, "positive_entries": 5,
        },
    }
    assert json.loads(json.dumps(view.metadata)) == view.metadata


def test_mapping_file_and_token_order(item_bundle, movies_file):
    original = _build(item_bundle, movies_file)
    reordered = replace(
        item_bundle, item_id_map=dict(reversed(list(item_bundle.item_id_map.items()))),
    )
    same_source = _build(reordered, movies_file)
    np.testing.assert_equal(asdict(same_source), asdict(original))

    records = [record.rsplit("::", 1) for record in reversed(_MOVIE_RECORDS)]
    movies_file.write_bytes(("\n".join(
        prefix + "::" + "|".join(reversed(genres.split("|")))
        for prefix, genres in records
    ) + "\n").encode("latin-1"))
    reordered_source = _build(reordered, movies_file)
    for name in ("entity_ids", "values", "missing_mask"):
        np.testing.assert_array_equal(
            getattr(original, name), getattr(reordered_source, name),
        )
    assert reordered_source.feature_names == original.feature_names
    assert reordered_source.metadata["counts"] == original.metadata["counts"]
    assert (
        reordered_source.metadata["item_mapping_sha256"]
        == original.metadata["item_mapping_sha256"]
    )
    assert (
        reordered_source.metadata["source_sha256"]
        != original.metadata["source_sha256"]
    )


def test_columns_do_not_depend_on_source_subset_or_split(item_bundle, movies_file):
    before = _build(item_bundle, movies_file)
    movies_file.write_bytes(("\n".join(
        record for record in _MOVIE_RECORDS if not record.startswith("999::")
    ) + "\n").encode("latin-1"))
    swapped_splits = replace(
        item_bundle,
        splits=InteractionSplits(
            train_interaction_ids=item_bundle.splits.test_interaction_ids,
            valid_interaction_ids=item_bundle.splits.valid_interaction_ids,
            test_interaction_ids=item_bundle.splits.train_interaction_ids,
        ),
    )
    after = _build(swapped_splits, movies_file)
    assert after.feature_names == before.feature_names
    np.testing.assert_array_equal(after.values, before.values)
    assert after.metadata["counts"]["metadata_only_items"] == 0


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("final_newline", [False, True])
def test_latin1_and_record_terminators(
    item_bundle, movies_file, newline, final_newline,
):
    before = _build(item_bundle, movies_file)
    content = newline.join(_MOVIE_RECORDS) + (newline if final_newline else "")
    movies_file.write_bytes(content.encode("latin-1"))
    after = _build(item_bundle, movies_file)
    np.testing.assert_array_equal(after.values, before.values)
    assert after.metadata["source_size_bytes"] == len(content.encode("latin-1"))
    assert after.metadata["source_sha256"] == hashlib.sha256(
        content.encode("latin-1")
    ).hexdigest()
    if newline != "\n" or not final_newline:
        assert after.metadata["source_sha256"] != before.metadata["source_sha256"]


def test_latin1_title_characters_are_not_record_boundaries(item_bundle, movies_file):
    content = movies_file.read_bytes().replace(b"Third", b"Third\x85part\x0btitle")
    movies_file.write_bytes(content)
    assert _build(item_bundle, movies_file).metadata["counts"]["source_records"] == 4


def test_source_is_read_once_and_hashes_the_parsed_bytes(
    item_bundle, movies_file, monkeypatch,
):
    content = movies_file.read_bytes()
    reads = []

    def read_once(path):
        assert path == movies_file
        reads.append(path)
        assert len(reads) == 1
        return content

    monkeypatch.setattr(Path, "read_bytes", read_once)
    view = _build(item_bundle, movies_file)
    assert len(reads) == 1
    assert view.metadata["source_sha256"] == hashlib.sha256(content).hexdigest()
    assert view.metadata["counts"]["positive_entries"] == 5


@pytest.mark.parametrize(
    "record, message",
    [
        (b"", "expected three fields"),
        (b"999::Title", "expected three fields"),
        (b"999::Title::Action::extra", "expected three fields"),
        (b"::Title::Action", "MovieID"),
        (b"0::Title::Action", "MovieID"),
        (b"-1::Title::Action", "MovieID"),
        (b"+1::Title::Action", "MovieID"),
        (b" 999::Title::Action", "MovieID"),
        (b"999 ::Title::Action", "MovieID"),
        (b"999.0::Title::Action", "MovieID"),
        (b"\xb2::Title::Action", "MovieID"),
        (b"999::::Action", "title must not be blank"),
        (b"999:: \t ::Action", "title must not be blank"),
        (b"999::Title::", "empty genre token"),
        (b"999::Title::|Action", "empty genre token"),
        (b"999::Title::Action|", "empty genre token"),
        (b"999::Title::Action||Drama", "empty genre token"),
        (b"999::Title::Action|Action", "repeated genre token"),
        (b"999::Title::action", "unknown genre token"),
        (b"999::Title:: Action", "unknown genre token"),
        (b"999::Title::Action ", "unknown genre token"),
        (b"999::Title::IMAX", "unknown genre token"),
        (b"90::First::Action", "duplicate MovieID 90"),
        (b"90::Different::Drama", "duplicate MovieID 90"),
        (b"090::Leading zero::Drama", "duplicate MovieID 90"),
    ],
)
def test_strict_failures_include_unmatched_records_and_line_numbers(
    item_bundle, movies_file, record, message,
):
    # Every record must be checked, even when its ID is not in the item map.
    movies_file.write_bytes(b"90::First::Action\n" + record + b"\n")
    with pytest.raises(ValueError, match=f"line 2: .*{message}"):
        _build(item_bundle, movies_file)


def test_duplicate_unmatched_movies_are_rejected(item_bundle, movies_file):
    with movies_file.open("ab") as source:
        source.write(b"999::Metadata only::Western\n")
    with pytest.raises(ValueError, match="line 5: duplicate MovieID 999"):
        _build(item_bundle, movies_file)


@pytest.mark.parametrize("content", [b"", b"\n", b"\r\n", b"   \n"])
def test_empty_or_blank_sources_fail(item_bundle, movies_file, content):
    movies_file.write_bytes(content)
    with pytest.raises(ValueError, match="empty|line 1"):
        _build(item_bundle, movies_file)


def test_unpaired_cr_is_not_silently_stripped(item_bundle, movies_file):
    movies_file.write_bytes(b"90::First::Action\r")
    with pytest.raises(ValueError, match="line 1: unknown genre token"):
        _build(item_bundle, movies_file)


def test_missing_metadata_reports_count_and_raw_ids(item_bundle, movies_file):
    movies_file.write_bytes(b"7::Only retained::Drama\n999::Extra::Western\n")
    with pytest.raises(
        ValueError, match=r"missing metadata for 2 processed items; .*\[42, 90\]",
    ):
        _build(item_bundle, movies_file)


@pytest.mark.parametrize("raw_id", [True, "90", 90.0, np.int64(90), None])
def test_revalidates_raw_id_types_without_coercion(item_bundle, movies_file, raw_id):
    item_bundle.item_id_map = {raw_id: 0, 7: 1, 42: 2}
    with pytest.raises(TypeError, match="built-in integers"):
        _build(item_bundle, movies_file)
    assert next(iter(item_bundle.item_id_map)) is raw_id


@pytest.mark.parametrize("processed_id", [False, "0", 0.0, np.int64(0), None])
def test_revalidates_processed_id_types_without_coercion(
    item_bundle, movies_file, processed_id,
):
    item_bundle.item_id_map[90] = processed_id
    with pytest.raises(TypeError, match="built-in integers"):
        _build(item_bundle, movies_file)
    assert item_bundle.item_id_map[90] is processed_id


@pytest.mark.parametrize("raw_id", [0, -90])
def test_rejects_nonpositive_mapping_keys(item_bundle, movies_file, raw_id):
    item_bundle.item_id_map = {raw_id: 0, 7: 1, 42: 2}
    with pytest.raises(ValueError, match="raw item IDs must be positive"):
        _build(item_bundle, movies_file)


@pytest.mark.parametrize("processed_id", [-1, 1, 3])
def test_rejects_negative_duplicate_or_gapped_processed_ids(
    item_bundle, movies_file, processed_id,
):
    item_bundle.item_id_map[90] = processed_id
    with pytest.raises(ValueError, match="bijection onto range"):
        _build(item_bundle, movies_file)


def test_rejects_non_dictionary_mapping(item_bundle, movies_file):
    item_bundle.item_id_map = [(90, 0)]
    with pytest.raises(TypeError, match="item_id_map must be a dictionary"):
        _build(item_bundle, movies_file)


def test_empty_namespace_has_empty_arrays_and_mapping_hash(item_bundle, movies_file):
    empty = replace(
        item_bundle,
        interactions=replace(item_bundle.interactions, **{
            field.name: getattr(item_bundle.interactions, field.name)[:0].copy()
            for field in fields(item_bundle.interactions)
        }),
        splits=replace(item_bundle.splits, **{
            field.name: getattr(item_bundle.splits, field.name)[:0].copy()
            for field in fields(item_bundle.splits)
        }),
        user_id_map={}, item_id_map={},
    )
    view = _build(empty, movies_file)
    assert view.entity_ids.shape == (0,)
    assert view.values.shape == view.missing_mask.shape == (0, 18)
    assert view.metadata["item_mapping_sha256"] == hashlib.sha256(b"").hexdigest()
    assert view.metadata["counts"]["metadata_only_items"] == 4
    assert view.metadata["counts"]["matched_items"] == 0
    movies_file.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        _build(empty, movies_file)


def test_mapping_checksum_matches_actual_synthetic_cache(processed_bundle, tmp_path):
    source = tmp_path / "movies.dat"
    source.write_bytes("".join(
        f"{raw_id}::Synthetic title::Comedy\n"
        for raw_id in reversed(processed_bundle.item_id_map)
    ).encode("latin-1"))
    view = _build(processed_bundle, source)
    cache_directory = tmp_path / "synthetic_base_cache"
    manifest = save_base_cache(processed_bundle, cache_directory)
    stored_mapping = (cache_directory / "base" / "items.jsonl").read_bytes()
    assert view.metadata["item_mapping_sha256"] == hashlib.sha256(
        stored_mapping
    ).hexdigest()
    assert (
        view.metadata["item_mapping_sha256"]
        == manifest["artifact_checksums"]["base/items.jsonl"]
    )


def test_fresh_mutable_arrays_and_metadata(item_bundle, movies_file):
    first = _build(item_bundle, movies_file)
    second = _build(item_bundle, movies_file)
    np.testing.assert_equal(asdict(first), asdict(second))
    for name in ("entity_ids", "values", "missing_mask"):
        assert getattr(first, name) is not getattr(second, name)
        assert not np.shares_memory(getattr(first, name), getattr(second, name))
        assert getattr(first, name).flags.writeable
    first.entity_ids[0] = 100
    first.values[:] = 0
    first.missing_mask[:] = True
    first.metadata["counts"]["matched_items"] = -1
    first.metadata["genre_vocabulary"][0] = "changed"
    assert second.entity_ids[0] == 0
    assert second.values.sum() == 5
    assert not second.missing_mask.any()
    assert second.metadata["counts"]["matched_items"] == 3
    assert second.metadata["genre_vocabulary"][0] == "Action"


@pytest.mark.parametrize("fail", [False, True])
def test_input_objects_unchanged_on_success_and_failure(
    item_bundle, movies_file, fail,
):
    existing_view = FeatureView(
        name="existing", entity_kind="item",
        entity_ids=np.array([1], dtype=np.int64), modality="tabular",
        values=np.array([[2.0]], dtype=np.float32), feature_names=("field",),
        missing_mask=np.array([[True]]), metadata={"nested": [1]},
    )
    bundle = replace(item_bundle, feature_views={existing_view.name: existing_view})
    bundle = replace(
        bundle, partition_kind="hfl", partition_name="existing",
        parties=build_hfl_partition(bundle),
    )
    before = asdict(bundle)
    records = [
        bundle, bundle.interactions, bundle.splits, existing_view, *bundle.parties,
    ]
    references = [
        (record, field.name, getattr(record, field.name))
        for record in records for field in fields(record)
    ]
    if fail:
        movies_file.write_bytes(b"999::Unmatched bad row::Unknown\n")
        with pytest.raises(ValueError, match="unknown genre"):
            _build(bundle, movies_file)
    else:
        _build(bundle, movies_file)
    np.testing.assert_equal(asdict(bundle), before)
    for record, name, reference in references:
        assert getattr(record, name) is reference


def test_explicit_collision_checked_attachment_and_cache_rejection(
    item_bundle, movies_file, tmp_path,
):
    existing = FeatureView(
        name="existing", entity_kind="item", entity_ids=np.array([0], dtype=np.int64),
        modality="text", values=("unchanged",), feature_names=(),
        missing_mask=None, metadata={},
    )
    base = replace(item_bundle, feature_views={existing.name: existing})
    view = _build(base, movies_file)
    # Caller-owned attachment, matching the documented recipe (not a public helper).
    feature_views = dict(base.feature_views)
    if view.name in feature_views:
        raise ValueError(f"feature view already exists: {view.name}")
    feature_views[view.name] = view
    derived = replace(base, feature_views=feature_views)
    assert base.feature_views == {"existing": existing}
    assert derived.feature_views is not base.feature_views
    assert derived.feature_views["existing"] is existing
    assert derived.feature_views[view.name] is view
    for field in fields(base):
        if field.name != "feature_views":
            assert getattr(derived, field.name) is getattr(base, field.name)
    with pytest.raises(ValueError, match="already exists"):
        feature_views = dict(derived.feature_views)
        if view.name in feature_views:
            raise ValueError(f"feature view already exists: {view.name}")
        feature_views[view.name] = view
        replace(derived, feature_views=feature_views)
    cache_directory = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="schema 1.0 cannot persist"):
        save_base_cache(derived, cache_directory)
    assert not cache_directory.exists()


def test_root_resolution_environment_and_explicit_override(
    item_bundle, movies_file, tmp_path, monkeypatch,
):
    monkeypatch.setenv("THESISFEDREC_DATA_ROOT", str(tmp_path))
    environment_view = build_ml1m_item_genre_view(
        item_bundle, movies_relative_path="movies.dat",
    )
    monkeypatch.setenv("THESISFEDREC_DATA_ROOT", str(tmp_path / "not_used"))
    explicit_view = _build(item_bundle, movies_file)
    np.testing.assert_equal(asdict(explicit_view), asdict(environment_view))


def test_repository_local_fallback(item_bundle, tmp_path, monkeypatch):
    # Emulate another repository root; never write into the real data/raw tree.
    monkeypatch.delenv("THESISFEDREC_DATA_ROOT", raising=False)
    monkeypatch.setattr(loaders, "__file__", str(tmp_path / "src/data/loaders.py"))
    local_raw = tmp_path / "data" / "raw"
    local_raw.mkdir(parents=True)
    (local_raw / "movies.dat").write_bytes(
        ("\n".join(_MOVIE_RECORDS) + "\n").encode("latin-1")
    )
    view = build_ml1m_item_genre_view(item_bundle, movies_relative_path="movies.dat")
    assert view.metadata["source_relative_path"] == "movies.dat"
    assert view.values.shape == (3, 18)


def test_absolute_and_escaping_paths_are_rejected(item_bundle, movies_file, tmp_path):
    for relative_path, message in (
        (movies_file, "must be relative"),
        ("../movies.dat", "must stay within"),
    ):
        with pytest.raises(ValueError, match=message):
            build_ml1m_item_genre_view(
                item_bundle, movies_relative_path=relative_path, data_root=tmp_path,
            )
    subdirectory = tmp_path / "subdirectory"
    subdirectory.mkdir()
    (subdirectory / "link.dat").symlink_to(movies_file)
    with pytest.raises(ValueError, match="must stay within"):
        build_ml1m_item_genre_view(
            item_bundle, movies_relative_path="link.dat", data_root=subdirectory,
        )


def test_missing_source_fails_without_fallback(item_bundle, tmp_path):
    with pytest.raises(FileNotFoundError):
        build_ml1m_item_genre_view(
            item_bundle, movies_relative_path="missing.dat", data_root=tmp_path,
        )


@pytest.mark.skipif(
    os.environ.get("THESISFEDREC_RUN_REAL_FEATURES") != "1",
    reason="set THESISFEDREC_RUN_REAL_FEATURES=1 for CPU-only real ML1M features",
)
def test_real_ml1m_item_genres():
    preprocessing = {
        "min_interactions_per_user": 3,
        "duplicate_policy": "keep_latest",
        "implicit_feedback": "all_observed",
        "seed": 2026,
    }
    partition_parameters = {
        "mode": "grouped_round_robin", "num_parties": 3, "seed": 2026,
    }
    base = preprocess_ml1m_from_root(
        "ml-1m/ml-1m/ratings.dat", **preprocessing,
    )
    before = asdict(base)
    parties = build_hfl_partition(base, **partition_parameters)
    parties_before = [asdict(party) for party in parties]
    view = build_ml1m_item_genre_view(
        base, movies_relative_path="ml-1m/ml-1m/movies.dat",
    )
    repeated = build_ml1m_item_genre_view(
        base, movies_relative_path="ml-1m/ml-1m/movies.dat",
    )
    np.testing.assert_equal(asdict(view), asdict(repeated))
    assert view.metadata["source_sha256"] == (
        "0140fc2356357c1a851d0f52e893a1e4d3696df632c4141cea8d5bc3d621f0b9"
    )
    assert view.metadata["source_size_bytes"] == 171308
    assert view.metadata["counts"] == {
        "source_records": 3883, "unique_metadata_items": 3883,
        "processed_items": 3706, "matched_items": 3706, "missing_items": 0,
        "metadata_only_items": 177, "genre_count": 18, "positive_entries": 6192,
    }
    np.testing.assert_array_equal(view.entity_ids, np.arange(3706, dtype=np.int64))
    assert view.feature_names == _EXPECTED_GENRES
    assert view.values.shape == (3706, 18)
    assert view.values.dtype == np.float32
    assert set(np.unique(view.values)) == {0.0, 1.0}
    assert view.values.sum() == 6192
    assert view.missing_mask.shape == (3706, 18)
    assert view.missing_mask.dtype == np.bool_
    assert not view.missing_mask.any()

    feature_views = dict(base.feature_views)
    if view.name in feature_views:
        raise ValueError(f"feature view already exists: {view.name}")
    feature_views[view.name] = view
    derived = replace(base, feature_views=feature_views)
    complete, complete_parties, complete_stats = build_complete_feature_scenario(
        derived, parties, view_names=(view.name,), name="real_complete",
    )
    repeated_complete, _, repeated_complete_stats = build_complete_feature_scenario(
        derived, tuple(reversed(parties)), view_names=(view.name,),
        name="repeated_complete",
    )
    assert repeated_complete.party_access == complete.party_access
    assert repeated_complete_stats == complete_stats
    absence_parameters = {
        "missing_view": view.name, "missing_party_count": 1, "seed": 2026,
    }
    absence, absence_parties, absence_stats = build_single_view_absence_scenario(
        derived, parties, view_names=(view.name,), name="real_absence",
        **absence_parameters,
    )
    repeated_absence, _, repeated_stats = build_single_view_absence_scenario(
        derived, tuple(reversed(parties)), view_names=(view.name,),
        name="repeated_absence", **absence_parameters,
    )
    assert repeated_absence.party_access == absence.party_access
    assert repeated_stats == absence_stats
    assert complete_stats["available_party_view_pairs"] == 3
    assert absence_stats["available_party_view_pairs"] == 2
    assert absence_stats["featureless_party_count"] == 1
    for partition, scenario_parties in (
        (complete, complete_parties), (absence, absence_parties),
    ):
        validate_feature_partition(partition, derived, scenario_parties)
        for original, scenario_party in zip(parties, scenario_parties):
            for field in fields(original):
                if field.name != "feature_view_names":
                    assert getattr(original, field.name) is getattr(
                        scenario_party, field.name,
                    )
    np.testing.assert_equal(asdict(base), before)
    np.testing.assert_equal([asdict(party) for party in parties], parties_before)
    np.testing.assert_equal(asdict(view), asdict(repeated))
    print(json.dumps({
        "preprocessing": preprocessing, "partition": partition_parameters,
        "absence": absence_parameters, "source_sha256": view.metadata["source_sha256"],
        "source_relative_path": view.metadata["source_relative_path"],
        "source_size_bytes": view.metadata["source_size_bytes"],
        "item_mapping_sha256": view.metadata["item_mapping_sha256"],
        "counts": view.metadata["counts"],
        "complete": complete_stats, "single_view_absence": absence_stats,
    }, sort_keys=True))
