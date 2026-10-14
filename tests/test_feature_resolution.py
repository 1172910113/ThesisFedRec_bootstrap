from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from src.data import (
    DatasetBundle,
    FeatureAccess,
    FeaturePartition,
    FeatureView,
    InteractionSplits,
    InteractionTable,
    PartyData,
    build_complete_feature_scenario,
    build_hfl_partition,
    build_single_view_absence_scenario,
    feature_partition_checksum,
    resolve_party_feature_views,
    validate_feature_partition,
)
from src.data import features


@pytest.fixture
def feature_bundle() -> DatasetBundle:
    views = {
        "item_signals": FeatureView(
            name="item_signals", entity_kind="item", modality="tabular",
            entity_ids=np.array([2, 7, 11], dtype=np.int64),
            values=np.array([[10, 100, 1], [20, 200, 0], [30, 300, 1]], np.float32),
            feature_names=("score", "count", "flag"),
            missing_mask=np.array([
                [False, True, False], [False, False, False], [True, False, False],
            ]),
            metadata={"source": {"private": ["not local metadata"]}},
        ),
        "item_text": FeatureView(
            name="item_text", entity_kind="item", modality="text",
            entity_ids=np.array([2, 7, 11], dtype=np.int64),
            values=("two", "seven", "eleven"), feature_names=(),
            missing_mask=np.array([False, True, False]), metadata={},
        ),
        "user_tensor": FeatureView(
            name="user_tensor", entity_kind="user", modality="tensor",
            entity_ids=np.array([0, 2], dtype=np.int64),
            values=np.arange(12, dtype=np.int16).reshape(2, 2, 3),
            feature_names=(), missing_mask=None, metadata={},
        ),
        "context": FeatureView(
            name="context", entity_kind="interaction", modality="numeric",
            entity_ids=np.array([1, 8, 23], dtype=np.int64),
            values=np.array([0, np.nan, 9], dtype=np.float64), feature_names=(),
            missing_mask=np.array([False, False, True]), metadata={},
        ),
    }
    return DatasetBundle(
        name="synthetic", schema_version="1.0",
        interactions=InteractionTable(
            interaction_ids=np.arange(24, dtype=np.int64),
            user_ids=np.repeat(np.arange(4, dtype=np.int64), 6),
            item_ids=np.tile(np.arange(12, dtype=np.int64), 2),
            timestamps=np.arange(24, dtype=np.int64),
            values=np.ones(24, dtype=np.float32),
        ),
        splits=InteractionSplits(
            train_interaction_ids=np.array([
                user * 6 + offset for user in range(4) for offset in range(4)
            ], dtype=np.int64),
            valid_interaction_ids=np.arange(4, 24, 6, dtype=np.int64),
            test_interaction_ids=np.arange(5, 24, 6, dtype=np.int64),
        ),
        user_id_map={f"user-{i}": i for i in range(4)},
        item_id_map={f"item-{i}": i for i in range(12)},
        feature_views=views, metadata={"untouched": [1, 2]},
    )


def _party(party_id: int, view_names: tuple[str, ...] = ()) -> PartyData:
    return PartyData(
        party_id=party_id,
        train_interaction_ids=np.array([0, 1], dtype=np.int64),
        valid_interaction_ids=np.array([4], dtype=np.int64),
        test_interaction_ids=np.array([5], dtype=np.int64),
        user_ids=np.array([0], dtype=np.int64),
        item_ids=np.array([0], dtype=np.int64),
        feature_view_names=view_names, graph_view_names=(), owns_labels=True,
        metadata={"private": [party_id]},
    )


def _assignment(party_access):
    partition = FeaturePartition(
        name="explicit", party_access=party_access, metadata={},
    )
    parties = tuple(
        _party(party_id, tuple(access.view_name for access in accesses))
        for party_id, accesses in party_access.items()
    )
    return partition, parties


def _resolve(bundle, *, name="item_signals", rows=None, selected_fields=None):
    access = FeatureAccess(
        view_name=name,
        entity_ids=None if rows is None else np.array(rows, dtype=np.int64),
        feature_names=selected_fields,
    )
    partition, parties = _assignment({101: (access,)})
    return resolve_party_feature_views(bundle, partition, parties, party_id=101)[name]


def _assert_owned_arrays(view: FeatureView, source: FeatureView) -> None:
    for name in ("entity_ids", "values", "missing_mask"):
        array = getattr(view, name)
        original = getattr(source, name)
        if isinstance(array, np.ndarray):
            assert type(array) is np.ndarray
            assert array.flags.owndata
            assert array.flags.writeable
            assert array.base is None
            assert not np.shares_memory(array, original)
            assert array.dtype == original.dtype
    assert type(view.metadata) is dict
    assert view.metadata == {}


def test_sparse_ids_fields_and_mask_hand_calculation(feature_bundle):
    view = _resolve(feature_bundle, rows=(2, 11), selected_fields=("flag", "score"))
    assert (view.name, view.entity_kind, view.modality) == (
        "item_signals", "item", "tabular",
    )
    np.testing.assert_array_equal(view.entity_ids, [2, 11])
    assert view.entity_ids.dtype == np.int64
    assert view.feature_names == ("score", "flag")
    np.testing.assert_array_equal(view.values, [[10, 1], [30, 1]])
    np.testing.assert_array_equal(view.missing_mask, [[False, False], [True, False]])
    _assert_owned_arrays(view, feature_bundle.feature_views[view.name])


@pytest.mark.parametrize(
    "rows, positions", [(None, [0, 1, 2]), ((), []), ((2, 11), [0, 2])],
)
@pytest.mark.parametrize(
    "selected_fields, columns",
    [(None, [0, 1, 2]), ((), []), (("flag", "score"), [0, 2])],
)
def test_all_none_empty_and_subset_combinations(
    feature_bundle, rows, positions, selected_fields, columns,
):
    source = feature_bundle.feature_views["item_signals"]
    view = _resolve(feature_bundle, rows=rows, selected_fields=selected_fields)
    row_indices = np.array(positions, dtype=np.intp)
    column_indices = np.array(columns, dtype=np.intp)
    np.testing.assert_array_equal(view.entity_ids, source.entity_ids[row_indices])
    np.testing.assert_array_equal(
        view.values, source.values[np.ix_(row_indices, column_indices)],
    )
    np.testing.assert_array_equal(
        view.missing_mask, source.missing_mask[np.ix_(row_indices, column_indices)],
    )
    expected_shape = (len(positions), len(columns))
    assert view.values.shape == view.missing_mask.shape == expected_shape
    assert view.feature_names == tuple(source.feature_names[i] for i in columns)
    _assert_owned_arrays(view, source)


@pytest.mark.parametrize("explicit_entry", [False, True])
def test_known_featureless_party_has_fresh_empty_dictionary(
    feature_bundle, explicit_entry,
):
    partition = FeaturePartition(
        name="hidden", party_access={101: ()} if explicit_entry else {}, metadata={},
    )
    parties = (_party(101),)
    first = resolve_party_feature_views(
        feature_bundle, partition, parties, party_id=101,
    )
    second = resolve_party_feature_views(
        feature_bundle, partition, parties, party_id=101,
    )
    assert first == second == {}
    assert first is not second
    with pytest.raises(ValueError, match="unknown requested party_id 102"):
        resolve_party_feature_views(feature_bundle, partition, parties, party_id=102)


def test_dictionary_and_column_order_ignore_assignment_input_order(feature_bundle):
    first, parties = _assignment({
        101: (
            FeatureAccess(view_name="item_text"),
            FeatureAccess(view_name="item_signals", feature_names=("flag", "score")),
        ),
        7: (),
    })
    second = FeaturePartition(name="reordered", party_access={
        7: (),
        101: (
            FeatureAccess(view_name="item_signals", feature_names=("score", "flag")),
            FeatureAccess(view_name="item_text"),
        ),
    }, metadata={})
    before = asdict(feature_bundle)
    left = resolve_party_feature_views(feature_bundle, first, parties, party_id=101)
    right = resolve_party_feature_views(
        feature_bundle, second, tuple(reversed(parties)), party_id=101,
    )
    assert tuple(left) == tuple(right) == ("item_signals", "item_text")
    for name in left:
        np.testing.assert_equal(asdict(left[name]), asdict(right[name]))
    assert feature_partition_checksum(first) == feature_partition_checksum(second)
    np.testing.assert_equal(asdict(feature_bundle), before)


@pytest.mark.parametrize("name, rows, positions", [
    ("user_tensor", (2,), [1]), ("user_tensor", (), []),
    ("context", (8, 23), [1, 2]), ("context", (), []),
    ("item_text", (7, 11), [1, 2]), ("item_text", (), []),
])
def test_unnamed_arrays_text_and_entity_kinds(feature_bundle, name, rows, positions):
    source = feature_bundle.feature_views[name]
    view = _resolve(feature_bundle, name=name, rows=rows)
    assert view.entity_kind == source.entity_kind
    assert view.feature_names == ()
    np.testing.assert_array_equal(view.entity_ids, rows)
    indices = np.array(positions, dtype=np.intp)
    if isinstance(source.values, np.ndarray):
        assert view.values.shape == (len(rows), *source.values.shape[1:])
        np.testing.assert_array_equal(view.values, source.values[indices])
    else:
        assert type(view.values) is tuple
        assert view.values == tuple(source.values[i] for i in positions)
        assert all(type(value) is str for value in view.values)
    if source.missing_mask is None:
        assert view.missing_mask is None
    else:
        np.testing.assert_array_equal(view.missing_mask, source.missing_mask[indices])
    _assert_owned_arrays(view, source)


@pytest.mark.parametrize("name", ["item_text", "context", "user_tensor"])
@pytest.mark.parametrize("selected_fields", [(), ("unknown",)])
def test_unnamed_sources_reject_any_field_selector_even_without_rows(
    feature_bundle, name, selected_fields,
):
    with pytest.raises(ValueError, match="does not support field-level access"):
        _resolve(feature_bundle, name=name, rows=(), selected_fields=selected_fields)


@pytest.mark.parametrize("mask_present", [False, True])
def test_preserves_zero_nan_and_masked_values(feature_bundle, mask_present):
    source = feature_bundle.feature_views["item_signals"]
    source.values[0] = [np.nan, 0, 42]
    source.missing_mask = None if not mask_present else np.array([
        [False, False, True], [False, False, False], [True, False, False],
    ])
    view = _resolve(feature_bundle)
    np.testing.assert_array_equal(view.values, source.values)
    assert np.isnan(view.values[0, 0])
    assert view.values[0, 1] == 0
    assert view.values[0, 2] == 42
    if mask_present:
        np.testing.assert_array_equal(view.missing_mask, source.missing_mask)
    else:
        assert view.missing_mask is None


@pytest.mark.parametrize("empty_axis", ["rows", "columns"])
def test_empty_canonical_views_remain_present(feature_bundle, empty_axis):
    source = feature_bundle.feature_views["item_signals"]
    if empty_axis == "rows":
        source.entity_ids = source.entity_ids[:0]
        source.values = source.values[:0]
        source.missing_mask = source.missing_mask[:0]
    else:
        source.values = source.values[:, :0]
        source.missing_mask = source.missing_mask[:, :0]
        source.feature_names = ()
    view = _resolve(feature_bundle)
    assert view.values.shape == source.values.shape
    _assert_owned_arrays(view, source)


@pytest.mark.parametrize(
    "party_id", [True, False, -1, 101.0, "101", np.int64(101), None],
)
@pytest.mark.parametrize("location", ["requested", "roster"])
def test_rejects_invalid_party_ids(feature_bundle, party_id, location):
    partition, parties = _assignment({101: (FeatureAccess(view_name="item_signals"),)})
    requested = party_id if location == "requested" else 101
    if location == "roster":
        parties[0].party_id = party_id
    with pytest.raises(ValueError, match="Python integer"):
        resolve_party_feature_views(
            feature_bundle, partition, parties, party_id=requested,
        )


@pytest.mark.parametrize("parties", [None, "101", b"101", {101: "party"}, (object(),)])
def test_rejects_invalid_roster(feature_bundle, parties):
    partition = FeaturePartition(name="empty", party_access={}, metadata={})
    with pytest.raises(TypeError, match="parties"):
        resolve_party_feature_views(feature_bundle, partition, parties, party_id=101)


def test_unknown_duplicate_and_missing_roster_members(feature_bundle):
    partition, parties = _assignment({101: ()})
    with pytest.raises(ValueError, match="unique party IDs"):
        resolve_party_feature_views(
            feature_bundle, partition, parties * 2, party_id=101,
        )
    with pytest.raises(ValueError, match="unknown requested"):
        resolve_party_feature_views(feature_bundle, partition, (), party_id=101)
    with pytest.raises(ValueError, match="unknown party_id 101"):
        resolve_party_feature_views(
            feature_bundle, partition, (_party(7),), party_id=7,
        )


def test_inconsistent_other_party_summary_is_rejected(feature_bundle):
    partition, parties = _assignment({
        101: (), 7: (FeatureAccess(view_name="item_text"),),
    })
    parties[1].feature_view_names = ()
    with pytest.raises(ValueError, match="do not match FeaturePartition"):
        resolve_party_feature_views(feature_bundle, partition, parties, party_id=101)


@pytest.mark.parametrize("field, value, error", [
    ("entity_ids", np.array([7, 2, 11], np.int64), "sorted"),
    ("entity_ids", np.array([2, 2, 11], np.int64), "unique"),
    ("entity_ids", np.array([-1, 7, 11], np.int64), "non-negative"),
    ("entity_ids", np.array([2, 7, 12], np.int64), "out-of-range"),
    ("entity_ids", np.array([[2, 7, 11]], np.int64), "one-dimensional"),
    ("entity_ids", np.array([2, 7, 11], np.int32), "dtype int64"),
    ("entity_ids", [2, 7, 11], "NumPy array"),
    ("values", np.zeros((2, 3)), "same row count"),
    ("values", np.zeros((3, 1, 3)), "two-dimensional"),
    ("values", np.array(1.0), "row dimension"),
    ("feature_names", ("score",), "match the values columns"),
    ("feature_names", ("score", "score", "flag"), "unique"),
    ("feature_names", ("score", "", "flag"), "empty names"),
    ("feature_names", ["score", "count", "flag"], "tuple of strings"),
    ("missing_mask", np.zeros((3, 3), np.int64), "boolean dtype"),
    ("missing_mask", np.zeros(3, np.bool_), "match the values shape"),
    ("missing_mask", [[False] * 3] * 3, "NumPy array"),
    ("name", "renamed", "keys must match"),
    ("entity_kind", "party", "unsupported FeatureView entity_kind"),
    ("modality", " ", "modality must not be empty"),
    ("metadata", None, "metadata must be a dictionary"),
])
def test_revalidates_mutated_source_contract(feature_bundle, field, value, error):
    source = feature_bundle.feature_views["item_signals"]
    setattr(source, field, value)
    with pytest.raises((TypeError, ValueError), match=error):
        _resolve(feature_bundle)
    assert getattr(source, field) is value


@pytest.mark.parametrize(
    "mutation", ["removed", "not_a_view", "bad_shape", "out_of_range"],
)
def test_malformed_other_party_view_blocks_even_featureless_party(
    feature_bundle, mutation,
):
    partition, parties = _assignment({
        101: (), 7: (FeatureAccess(view_name="item_text"),),
    })
    if mutation == "removed":
        del feature_bundle.feature_views["item_text"]
    elif mutation == "not_a_view":
        feature_bundle.feature_views["item_text"] = object()
    elif mutation == "bad_shape":
        feature_bundle.feature_views["item_text"].missing_mask = np.array([False])
    else:
        feature_bundle.feature_views["item_text"].entity_ids = np.array([2, 7, 99])
    with pytest.raises((TypeError, ValueError), match="FeatureView"):
        resolve_party_feature_views(feature_bundle, partition, parties, party_id=101)


def test_unreferenced_malformed_views_are_not_inspected(feature_bundle):
    feature_bundle.feature_views["item_text"] = object()
    feature_bundle.feature_views["user_tensor"].entity_ids = None
    assert _resolve(feature_bundle).values.shape == (3, 3)


@pytest.mark.parametrize("mutation", ["view", "rows", "fields"])
def test_stale_references_fail_after_previous_success(feature_bundle, mutation):
    access = FeatureAccess(
        view_name="item_signals", entity_ids=np.array([2, 11], np.int64),
        feature_names=("flag",),
    )
    partition, parties = _assignment({101: (access,)})
    resolve_party_feature_views(feature_bundle, partition, parties, party_id=101)
    source = feature_bundle.feature_views["item_signals"]
    if mutation == "view":
        del feature_bundle.feature_views["item_signals"]
    elif mutation == "rows":
        source.entity_ids = source.entity_ids[:2]
        source.values = source.values[:2]
        source.missing_mask = source.missing_mask[:2]
    else:
        source.feature_names = ("score", "count", "renamed")
    with pytest.raises(ValueError, match="unknown"):
        resolve_party_feature_views(feature_bundle, partition, parties, party_id=101)


def test_empty_axes_do_not_bypass_reference_validation(feature_bundle):
    with pytest.raises(ValueError, match="unknown feature field"):
        _resolve(feature_bundle, rows=(), selected_fields=("unknown",))
    with pytest.raises(ValueError, match="unknown entity ID"):
        _resolve(feature_bundle, rows=(3,), selected_fields=())


def test_existing_partition_validator_is_called_with_complete_roster(
    feature_bundle, monkeypatch,
):
    partition, parties = _assignment({
        101: (), 7: (FeatureAccess(view_name="item_text"),),
    })
    original = features.validate_feature_partition
    calls = []

    def record_call(current_partition, current_bundle, current_parties):
        calls.append((current_partition, current_bundle, current_parties))
        original(current_partition, current_bundle, current_parties)

    monkeypatch.setattr(features, "validate_feature_partition", record_call)
    assert resolve_party_feature_views(
        feature_bundle, partition, parties, party_id=101,
    ) == {}
    assert len(calls) == 1
    assert calls[0][0] is partition
    assert calls[0][1] is feature_bundle
    assert calls[0][2] is parties


class _ArrayWithState(np.ndarray):
    pass


class _StringWithState(str):
    pass


class _TupleWithState(tuple):
    pass


class _ReportedSubarray(np.ndarray):
    """Exercise exceptional dtype reporting without native-memory manipulation."""

    @property
    def dtype(self):
        return np.dtype((np.float32, (2,)))


class _ReportedFloat(np.ndarray):
    @property
    def dtype(self):
        return np.dtype(np.float32)


def test_subclass_cannot_hide_exceptional_backing_dtype(feature_bundle):
    source = feature_bundle.feature_views["item_signals"]
    source.values = source.values.astype(object).view(_ReportedFloat)
    replace(source)
    with pytest.raises(TypeError, match="unsupported delivery dtype"):
        _resolve(feature_bundle)


@pytest.mark.parametrize("kind", ["object", "structured", "record", "subarray"])
@pytest.mark.parametrize("selection", ["full", "empty_rows", "empty_fields"])
def test_rejects_exceptional_value_dtypes_even_for_empty_selections(
    feature_bundle, kind, selection,
):
    source = feature_bundle.feature_views["item_signals"]
    if kind == "object":
        source.values = source.values.astype(object)
    elif kind in ("structured", "record"):
        source.values = np.zeros((3, 3), dtype=[("field", "f4")])
        if kind == "record":
            source.values = source.values.view(np.recarray)
    else:
        source.values = source.values.view(_ReportedSubarray)
    # These remain structurally valid canonical views; only delivery rejects them.
    replace(source)
    with pytest.raises(TypeError, match="unsupported delivery dtype"):
        _resolve(
            feature_bundle,
            rows=() if selection == "empty_rows" else None,
            selected_fields=() if selection == "empty_fields" else None,
        )


@pytest.mark.parametrize("array_name", ["entity_ids", "values", "missing_mask"])
@pytest.mark.parametrize("selection", ["full", "empty_rows", "empty_fields"])
def test_rejects_dtype_metadata_on_every_delivered_array(
    feature_bundle, array_name, selection,
):
    source = feature_bundle.feature_views["item_signals"]
    array = getattr(source, array_name)
    dtype = np.dtype(array.dtype, metadata={"hidden": source.values})
    setattr(source, array_name, array.view(dtype))
    assert getattr(source, array_name).dtype.metadata is not None
    replace(source)
    with pytest.raises(TypeError, match=f"{array_name} has unsupported delivery dtype"):
        _resolve(
            feature_bundle,
            rows=() if selection == "empty_rows" else None,
            selected_fields=() if selection == "empty_fields" else None,
        )


def test_empty_dtype_metadata_is_also_rejected(feature_bundle):
    source = feature_bundle.feature_views["item_signals"]
    source.values = source.values.view(np.dtype("f4", metadata={}))
    assert source.values.dtype.metadata is not None
    with pytest.raises(TypeError, match="unsupported delivery dtype"):
        _resolve(feature_bundle)


@pytest.mark.parametrize("array_name", ["entity_ids", "values", "missing_mask"])
def test_delivery_dtype_restrictions_do_not_apply_to_other_parties(
    feature_bundle, array_name,
):
    other = replace(feature_bundle.feature_views["item_signals"], name="other")
    array = getattr(other, array_name)
    exceptional = (
        array.astype(object) if array_name == "values"
        else array.view(np.dtype(array.dtype, metadata={}))
    )
    setattr(other, array_name, exceptional)
    feature_bundle.feature_views[other.name] = other
    partition, parties = _assignment({
        101: (FeatureAccess(view_name="item_signals"),),
        7: (FeatureAccess(view_name="other"),),
        42: (),
    })
    # Preserve existing canonical construction and public reference validation.
    replace(feature_bundle)
    validate_feature_partition(partition, feature_bundle, parties)
    assert tuple(resolve_party_feature_views(
        feature_bundle, partition, parties, party_id=101,
    )) == ("item_signals",)
    assert resolve_party_feature_views(
        feature_bundle, partition, parties, party_id=42,
    ) == {}
    with pytest.raises(TypeError, match="unsupported delivery dtype"):
        resolve_party_feature_views(feature_bundle, partition, parties, party_id=7)


def test_array_and_tuple_subclass_state_is_not_exposed(feature_bundle):
    source = feature_bundle.feature_views["item_signals"]
    hidden = np.arange(100)
    for name in ("entity_ids", "values", "missing_mask"):
        array = getattr(source, name).view(_ArrayWithState)
        array.hidden = hidden
        setattr(source, name, array)
    source.feature_names = _TupleWithState(source.feature_names)
    source.feature_names.hidden = hidden
    access_name = _StringWithState(source.name)
    access_name.hidden = hidden
    partition, parties = _assignment({
        101: (FeatureAccess(view_name=access_name, feature_names=("score",)),),
    })
    result = resolve_party_feature_views(
        feature_bundle, partition, parties, party_id=101,
    )
    assert all(type(name) is str for name in result)
    view = result["item_signals"]
    assert type(view) is FeatureView
    assert type(view.feature_names) is tuple
    _assert_owned_arrays(view, source)
    for name in ("entity_ids", "values", "missing_mask", "feature_names"):
        assert not hasattr(getattr(view, name), "hidden")

    text_source = feature_bundle.feature_views["item_text"]
    text_source.values = _TupleWithState(text_source.values)
    text_source.values.hidden = hidden
    text = _resolve(feature_bundle, name="item_text", rows=(2, 11))
    assert type(text.values) is tuple
    assert text.values == ("two", "eleven")
    assert not hasattr(text.values, "hidden")


@pytest.mark.parametrize(
    "field", ["name", "entity_kind", "modality", "feature_names", "text"],
)
def test_rejects_emitted_string_subclasses(feature_bundle, field):
    source_name = "item_text" if field == "text" else "item_signals"
    source = feature_bundle.feature_views[source_name]
    if field == "feature_names":
        value = _StringWithState("count")
        source.feature_names = ("score", value, "flag")
    elif field == "text":
        value = _StringWithState("seven")
        source.values = ("two", value, "eleven")
    else:
        value = _StringWithState(getattr(source, field))
        setattr(source, field, value)
    value.hidden = feature_bundle
    replace(source)  # Existing FeatureView acceptance is unchanged.
    with pytest.raises(TypeError, match="plain built-in strings"):
        _resolve(feature_bundle, name=source_name)


def test_unselected_text_and_field_string_subclasses_are_not_forwarded(feature_bundle):
    hidden_text = _StringWithState("seven")
    hidden_text.hidden = feature_bundle
    feature_bundle.feature_views["item_text"].values = ("two", hidden_text, "eleven")
    text = _resolve(feature_bundle, name="item_text", rows=(2, 11))
    assert text.values == ("two", "eleven")
    hidden_field = _StringWithState("count")
    hidden_field.hidden = feature_bundle
    feature_bundle.feature_views["item_signals"].feature_names = (
        "score", hidden_field, "flag",
    )
    view = _resolve(feature_bundle, selected_fields=("flag",))
    assert view.feature_names == ("flag",)


def test_metadata_is_omitted_without_deepcopy(feature_bundle):
    class MustNotCopy:
        def __deepcopy__(self, memo):
            pytest.fail("canonical metadata must not be deep-copied into payloads")

    source = feature_bundle.feature_views["item_signals"]
    source.metadata = {"hidden_values": source.values, "opaque": MustNotCopy()}
    original_metadata = source.metadata
    first = _resolve(feature_bundle)
    second = _resolve(feature_bundle)
    assert first.metadata == second.metadata == {}
    assert first.metadata is not second.metadata
    first.metadata["local"] = [1]
    assert second.metadata == {}
    assert source.metadata is original_metadata


def test_repeated_calls_and_parties_have_isolated_mutable_outputs(feature_bundle):
    accesses = (
        FeatureAccess(view_name="item_signals"), FeatureAccess(view_name="item_text"),
    )
    partition, parties = _assignment({101: accesses, 7: accesses})
    source_before = asdict(feature_bundle)
    outputs = [
        resolve_party_feature_views(
            feature_bundle, partition, parties, party_id=party_id,
        )
        for party_id in (101, 101, 7)
    ]
    for index, result in enumerate(outputs):
        for name, view in result.items():
            _assert_owned_arrays(view, feature_bundle.feature_views[name])
            for other in outputs[index + 1:]:
                assert result is not other
                assert view is not other[name]
                assert view.metadata is not other[name].metadata
                for array_name in ("entity_ids", "values", "missing_mask"):
                    array = getattr(view, array_name)
                    if isinstance(array, np.ndarray):
                        assert array is not getattr(other[name], array_name)
                        assert not np.shares_memory(
                            array, getattr(other[name], array_name),
                        )
    unchanged = {name: asdict(view) for name, view in outputs[1].items()}
    first = outputs[0]["item_signals"]
    first.values[:] = -1
    first.entity_ids.dtype = np.int32
    first.missing_mask.shape = (9,)
    first.metadata["changed"] = [1]
    outputs[0].pop("item_text")
    for result in outputs[1:]:
        np.testing.assert_equal(
            {name: asdict(view) for name, view in result.items()}, unchanged,
        )
    np.testing.assert_equal(asdict(feature_bundle), source_before)


def test_current_valid_source_edits_are_observed_without_changing_prior_outputs(
    feature_bundle,
):
    partition, parties = _assignment({101: (FeatureAccess(view_name="item_signals"),)})
    checksum = feature_partition_checksum(partition)
    first = resolve_party_feature_views(
        feature_bundle, partition, parties, party_id=101,
    )
    first_before = asdict(first["item_signals"])
    source = feature_bundle.feature_views["item_signals"]
    source.entity_ids = np.array([2, 7, 9, 11], dtype=np.int64)
    source.values = np.insert(source.values, 2, [99, 999, 1], axis=0)
    source.missing_mask = np.insert(
        source.missing_mask, 2, [False, True, False], axis=0,
    )
    source.values[0, 0] = -10
    source.values = source.values[:, [2, 0, 1]]
    source.missing_mask = source.missing_mask[:, [2, 0, 1]]
    source.feature_names = ("flag", "score", "count")
    second = resolve_party_feature_views(
        feature_bundle, partition, parties, party_id=101,
    )
    np.testing.assert_array_equal(second["item_signals"].entity_ids, [2, 7, 9, 11])
    np.testing.assert_array_equal(second["item_signals"].values, source.values)
    np.testing.assert_array_equal(
        second["item_signals"].missing_mask, source.missing_mask,
    )
    assert second["item_signals"].feature_names == ("flag", "score", "count")
    np.testing.assert_equal(asdict(first["item_signals"]), first_before)
    assert feature_partition_checksum(partition) == checksum


@pytest.mark.parametrize("fail", [False, True])
def test_inputs_remain_unchanged_on_success_and_failure(feature_bundle, fail):
    partition, parties = _assignment({101: (FeatureAccess(
        view_name="item_signals", feature_names=("unknown",) if fail else ("flag",),
    ),)})
    before = asdict(feature_bundle)
    party_before = [asdict(party) for party in parties]
    checksum = feature_partition_checksum(partition)
    references = [
        (record, field.name, getattr(record, field.name))
        for record in (
            feature_bundle, feature_bundle.interactions, feature_bundle.splits,
            *feature_bundle.feature_views.values(), *parties,
        )
        for field in fields(record)
    ]
    if fail:
        with pytest.raises(ValueError, match="unknown feature field"):
            resolve_party_feature_views(
                feature_bundle, partition, parties, party_id=101,
            )
    else:
        resolve_party_feature_views(feature_bundle, partition, parties, party_id=101)
    np.testing.assert_equal(asdict(feature_bundle), before)
    np.testing.assert_equal([asdict(party) for party in parties], party_before)
    for record, name, reference in references:
        assert getattr(record, name) is reference
    assert feature_partition_checksum(partition) == checksum


@pytest.mark.parametrize(
    "missing_count", [None, 0, 1, 2], ids=["complete", "none", "one", "all"],
)
def test_accepted_scenarios_and_sample_ownership_are_unchanged(
    feature_bundle, missing_count,
):
    source_parties = build_hfl_partition(
        feature_bundle, mode="grouped_round_robin", num_parties=2, seed=2026,
    )
    bundle = replace(
        feature_bundle, partition_kind="hfl", partition_name="base",
        parties=source_parties,
    )
    roster = tuple(
        replace(party, party_id=party_id)
        for party, party_id in zip(source_parties, (101, 7))
    )
    if missing_count is None:
        partition, parties, stats = build_complete_feature_scenario(
            bundle, roster, view_names=("item_text", "item_signals"), name="complete",
        )
    else:
        partition, parties, stats = build_single_view_absence_scenario(
            bundle, roster, view_names=("item_text", "item_signals"),
            missing_view="item_text", missing_party_count=missing_count,
            seed=2026, name="absence",
        )
    before = asdict(bundle)
    party_before = [asdict(party) for party in parties]
    assert any(len(party.user_ids) > 1 for party in parties)
    for party in parties:
        result = resolve_party_feature_views(
            bundle, partition, parties, party_id=party.party_id,
        )
        assert tuple(result) == party.feature_view_names
        for name, view in result.items():
            source = bundle.feature_views[name]
            # Complete access is not intersected with party.item_ids/user_ids.
            np.testing.assert_array_equal(view.entity_ids, source.entity_ids)
            np.testing.assert_equal(view.values, source.values)
            np.testing.assert_equal(view.missing_mask, source.missing_mask)
            _assert_owned_arrays(view, source)
    np.testing.assert_equal(asdict(bundle), before)
    np.testing.assert_equal([asdict(party) for party in parties], party_before)
    assert feature_partition_checksum(partition) == stats["assignment_checksum"]
    validate_feature_partition(partition, bundle, parties)


@pytest.mark.parametrize("rows, selected_fields, expected_checksum", [
    (None, None, "99327dd04859e1232f8868683718c1b21327d3c46f45f78656c42c0e32aae9cd"),
    ((), (), "0d65e9d22fa64b1ef9f889dea50a9602ddeff245813def9f99541fe259d22aff"),
    (
        (0, 4), ("brand",),
        "31159190a1b5c2cbf3be50ae12644dd2a2a3704a9b14c587f893efd35df9e895",
    ),
    (
        (0, 2, 4), ("category", "brand"),
        "245aa8836a5e204264c5bcd07bebaf90aa685aa62f206f83b626ca5b952b09c5",
    ),
])
def test_resolution_preserves_pinned_assignment_checksums(
    feature_bundle, rows, selected_fields, expected_checksum,
):
    source = feature_bundle.feature_views["item_signals"]
    source = replace(
        source, name="item_tabular", entity_ids=np.array([0, 2, 4], dtype=np.int64),
        values=source.values[:, :2], feature_names=("category", "brand"),
        missing_mask=source.missing_mask[:, :2],
    )
    bundle = replace(feature_bundle, feature_views={source.name: source})
    partition, parties = _assignment({7: (FeatureAccess(
        view_name=source.name,
        entity_ids=None if rows is None else np.array(rows, dtype=np.int64),
        feature_names=selected_fields,
    ),)})
    assert feature_partition_checksum(partition) == expected_checksum
    resolve_party_feature_views(bundle, partition, parties, party_id=7)
    assert feature_partition_checksum(partition) == expected_checksum
