from dataclasses import replace

import numpy as np
import pytest

from src.data import (
    FeatureAccess,
    FeaturePartition,
    FeatureView,
    feature_partition_checksum,
    validate_feature_partition,
)
from src.data.contracts import DatasetBundle, PartyData


@pytest.fixture
def feature_bundle(processed_bundle: DatasetBundle) -> DatasetBundle:
    item_tabular = FeatureView(
        name="item_tabular",
        entity_kind="item",
        entity_ids=np.array([0, 2, 4], dtype=np.int64),
        modality="tabular",
        values=np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            dtype=np.float32,
        ),
        feature_names=("category", "brand"),
        missing_mask=np.array(
            [[False, True], [False, False], [True, False]],
            dtype=np.bool_,
        ),
        metadata={"source": "synthetic"},
    )
    item_text = FeatureView(
        name="item_text",
        entity_kind="item",
        entity_ids=np.array([0, 2, 4], dtype=np.int64),
        modality="text",
        values=("zero", "two", "four"),
        feature_names=(),
        missing_mask=np.array([False, True, False], dtype=np.bool_),
        metadata={"source": "synthetic"},
    )
    return replace(
        processed_bundle,
        feature_views={
            item_tabular.name: item_tabular,
            item_text.name: item_text,
        },
    )


def _party(
    party_id: int,
    *,
    feature_view_names: tuple[str, ...] = (),
    train_ids: tuple[int, ...] = (0,),
    valid_ids: tuple[int, ...] = (1,),
    test_ids: tuple[int, ...] = (2,),
    user_ids: tuple[int, ...] = (0,),
    item_ids: tuple[int, ...] = (0,),
) -> PartyData:
    return PartyData(
        party_id=party_id,
        train_interaction_ids=np.array(train_ids, dtype=np.int64),
        valid_interaction_ids=np.array(valid_ids, dtype=np.int64),
        test_interaction_ids=np.array(test_ids, dtype=np.int64),
        user_ids=np.array(user_ids, dtype=np.int64),
        item_ids=np.array(item_ids, dtype=np.int64),
        feature_view_names=feature_view_names,
        graph_view_names=(),
        owns_labels=True,
        metadata={},
    )


def _partition(
    party_id: int,
    access: FeatureAccess,
) -> FeaturePartition:
    return FeaturePartition(
        name="explicit_access",
        party_access={party_id: (access,)},
        metadata={"construction": "test"},
    )


def test_valid_whole_view_access(feature_bundle: DatasetBundle) -> None:
    party = _party(7, feature_view_names=("item_tabular",))
    access = FeatureAccess(
        view_name="item_tabular",
        entity_ids=None,
        feature_names=None,
    )
    partition = _partition(party.party_id, access)

    validate_feature_partition(partition, feature_bundle, (party,))

    assert partition.access_for_party(7) == (access,)


def test_valid_subset_of_rows_access(feature_bundle: DatasetBundle) -> None:
    party = _party(7, feature_view_names=("item_tabular",))
    requested_rows = np.array([0, 4], dtype=np.int64)
    access = FeatureAccess(
        view_name="item_tabular",
        entity_ids=requested_rows,
        feature_names=None,
    )

    validate_feature_partition(
        _partition(party.party_id, access),
        feature_bundle,
        (party,),
    )

    requested_rows[0] = 2
    np.testing.assert_array_equal(
        access.entity_ids,
        np.array([0, 4], dtype=np.int64),
    )
    assert access.entity_ids is not None
    assert not access.entity_ids.flags.writeable


def test_valid_subset_of_fields_access(feature_bundle: DatasetBundle) -> None:
    party = _party(7, feature_view_names=("item_tabular",))
    access = FeatureAccess(
        view_name="item_tabular",
        entity_ids=None,
        feature_names=("brand",),
    )

    validate_feature_partition(
        _partition(party.party_id, access),
        feature_bundle,
        (party,),
    )

    assert access.feature_names == ("brand",)


def test_feature_partition_rejects_assignment_mutations(
    feature_bundle: DatasetBundle,
) -> None:
    party = _party(7, feature_view_names=("item_tabular",))
    access = FeatureAccess(view_name="item_tabular", feature_names=("brand",))
    partition = _partition(party.party_id, access)
    validate_feature_partition(partition, feature_bundle, (party,))
    checksum_before = feature_partition_checksum(partition)

    with pytest.raises(TypeError):
        partition.party_access[99] = (access,)
    with pytest.raises(TypeError):
        partition.party_access[party.party_id] = ()
    with pytest.raises(TypeError):
        del partition.party_access[party.party_id]
    with pytest.raises(TypeError):
        partition.access_for_party(party.party_id)[0] = access
    with pytest.raises(TypeError):
        access.feature_names[0] = "category"

    validate_feature_partition(partition, feature_bundle, (party,))
    assert feature_partition_checksum(partition) == checksum_before


def test_feature_partition_copies_constructor_inputs(
    feature_bundle: DatasetBundle,
) -> None:
    parties = (_party(7, feature_view_names=("item_tabular",)), _party(11))
    requested_rows = np.array([0, 4], dtype=np.int64)
    access = FeatureAccess(view_name="item_tabular", entity_ids=requested_rows)
    requested_access = {7: (access,), 11: ()}
    partition = FeaturePartition(
        name="copied_inputs", party_access=requested_access, metadata={}
    )
    validate_feature_partition(partition, feature_bundle, parties)
    checksum_before = feature_partition_checksum(partition)

    requested_access[99] = (access,)
    requested_access[7] = ()
    del requested_access[11]
    requested_rows[:] = [1, 3]

    assert set(partition.party_access) == {7, 11}
    assert partition.access_for_party(7) == (access,)
    assert partition.access_for_party(11) == ()
    np.testing.assert_array_equal(access.entity_ids, [0, 4])
    validate_feature_partition(partition, feature_bundle, parties)
    assert feature_partition_checksum(partition) == checksum_before


@pytest.mark.parametrize("rows", [(), (0, 4)], ids=["empty", "subset"])
def test_feature_access_ids_cannot_be_made_writable(
    feature_bundle: DatasetBundle,
    rows: tuple[int, ...],
) -> None:
    party = _party(7, feature_view_names=("item_tabular",))
    access = FeatureAccess(
        view_name="item_tabular", entity_ids=np.array(rows, dtype=np.int64)
    )
    partition = _partition(party.party_id, access)
    validate_feature_partition(partition, feature_bundle, (party,))
    checksum_before = feature_partition_checksum(partition)
    stored_ids = access.entity_ids
    assert isinstance(stored_ids, np.ndarray)
    assert stored_ids.ndim == 1
    assert stored_ids.dtype == np.int64

    with pytest.raises(ValueError, match="read-only"):
        stored_ids[:] = 2
    with pytest.raises(ValueError, match="WRITEABLE"):
        stored_ids.setflags(write=True)

    np.testing.assert_array_equal(stored_ids, np.array(rows, dtype=np.int64))
    assert not stored_ids.flags.writeable
    validate_feature_partition(partition, feature_bundle, (party,))
    assert feature_partition_checksum(partition) == checksum_before


def test_missing_whole_view_is_no_access_descriptor(
    feature_bundle: DatasetBundle,
) -> None:
    party = _party(7)
    partition = FeaturePartition(
        name="no_features",
        party_access={},
        metadata={},
    )

    validate_feature_partition(partition, feature_bundle, (party,))

    assert partition.access_for_party(party.party_id) == ()


def test_unknown_feature_view_fails(feature_bundle: DatasetBundle) -> None:
    party = _party(7, feature_view_names=("unknown_view",))
    partition = _partition(
        party.party_id,
        FeatureAccess(view_name="unknown_view"),
    )

    with pytest.raises(ValueError, match="unknown FeatureView"):
        validate_feature_partition(partition, feature_bundle, (party,))


def test_unknown_party_fails(feature_bundle: DatasetBundle) -> None:
    known_party = _party(7)
    partition = _partition(
        99,
        FeatureAccess(view_name="item_tabular"),
    )

    with pytest.raises(ValueError, match="unknown party_id 99"):
        validate_feature_partition(partition, feature_bundle, (known_party,))


def test_unknown_feature_row_fails(feature_bundle: DatasetBundle) -> None:
    party = _party(7, feature_view_names=("item_tabular",))
    partition = _partition(
        party.party_id,
        FeatureAccess(
            view_name="item_tabular",
            entity_ids=np.array([1], dtype=np.int64),
        ),
    )

    with pytest.raises(ValueError, match="unknown entity ID 1"):
        validate_feature_partition(partition, feature_bundle, (party,))


def test_unknown_feature_field_fails(feature_bundle: DatasetBundle) -> None:
    party = _party(7, feature_view_names=("item_tabular",))
    partition = _partition(
        party.party_id,
        FeatureAccess(
            view_name="item_tabular",
            feature_names=("unknown_field",),
        ),
    )

    with pytest.raises(ValueError, match="unknown feature field"):
        validate_feature_partition(partition, feature_bundle, (party,))


def test_duplicate_party_view_descriptor_fails() -> None:
    first = FeatureAccess(view_name="item_tabular")
    second = FeatureAccess(
        view_name="item_tabular",
        feature_names=("brand",),
    )

    with pytest.raises(ValueError, match="duplicate party/view access"):
        FeaturePartition(
            name="duplicate",
            party_access={7: (first, second)},
            metadata={},
        )


def test_text_view_rejects_field_level_access(
    feature_bundle: DatasetBundle,
) -> None:
    party = _party(7, feature_view_names=("item_text",))
    partition = _partition(
        party.party_id,
        FeatureAccess(
            view_name="item_text",
            feature_names=("token",),
        ),
    )

    with pytest.raises(ValueError, match="does not support field-level access"):
        validate_feature_partition(partition, feature_bundle, (party,))


def test_partition_validation_does_not_mutate_canonical_feature_view(
    feature_bundle: DatasetBundle,
) -> None:
    view = feature_bundle.feature_views["item_tabular"]
    entity_ids_before = view.entity_ids.copy()
    values_before = view.values.copy()
    mask_before = view.missing_mask.copy()
    party = _party(7, feature_view_names=(view.name,))
    partition = _partition(
        party.party_id,
        FeatureAccess(
            view_name=view.name,
            entity_ids=np.array([0, 4], dtype=np.int64),
            feature_names=("brand",),
        ),
    )

    validate_feature_partition(partition, feature_bundle, (party,))

    np.testing.assert_array_equal(view.entity_ids, entity_ids_before)
    np.testing.assert_array_equal(view.values, values_before)
    np.testing.assert_array_equal(view.missing_mask, mask_before)


def test_canonical_missing_mask_is_independent_of_scenario_visibility(
    feature_bundle: DatasetBundle,
) -> None:
    view = feature_bundle.feature_views["item_tabular"]
    party = _party(7, feature_view_names=(view.name,))
    access = FeatureAccess(
        view_name=view.name,
        entity_ids=np.array([0], dtype=np.int64),
        feature_names=("category",),
    )
    canonical_mask_before = view.missing_mask.copy()

    validate_feature_partition(
        _partition(party.party_id, access),
        feature_bundle,
        (party,),
    )

    assert access.entity_ids is not view.entity_ids
    np.testing.assert_array_equal(view.missing_mask, canonical_mask_before)


def test_same_sample_ownership_can_have_different_feature_access(
    feature_bundle: DatasetBundle,
) -> None:
    tabular_party = _party(10, feature_view_names=("item_tabular",))
    text_party = _party(42, feature_view_names=("item_text",))
    partition = FeaturePartition(
        name="different_features",
        party_access={
            tabular_party.party_id: (
                FeatureAccess(
                    view_name="item_tabular",
                    feature_names=("brand",),
                ),
            ),
            text_party.party_id: (FeatureAccess(view_name="item_text"),),
        },
        metadata={},
    )

    validate_feature_partition(
        partition,
        feature_bundle,
        (tabular_party, text_party),
    )

    np.testing.assert_array_equal(
        tabular_party.train_interaction_ids,
        text_party.train_interaction_ids,
    )
    assert partition.access_for_party(10) != partition.access_for_party(42)


def test_different_sample_ownership_can_share_feature_access(
    feature_bundle: DatasetBundle,
) -> None:
    first_party = _party(10, feature_view_names=("item_tabular",))
    second_party = _party(
        42,
        feature_view_names=("item_tabular",),
        train_ids=(3,),
        valid_ids=(4,),
        test_ids=(5,),
        user_ids=(1,),
        item_ids=(2,),
    )
    shared_rows = np.array([0, 2], dtype=np.int64)
    first_access = FeatureAccess(
        view_name="item_tabular",
        entity_ids=shared_rows,
    )
    second_access = FeatureAccess(
        view_name="item_tabular",
        entity_ids=shared_rows,
    )
    partition = FeaturePartition(
        name="shared_features",
        party_access={
            first_party.party_id: (first_access,),
            second_party.party_id: (second_access,),
        },
        metadata={},
    )

    validate_feature_partition(
        partition,
        feature_bundle,
        (first_party, second_party),
    )

    assert not np.array_equal(
        first_party.train_interaction_ids,
        second_party.train_interaction_ids,
    )
    np.testing.assert_array_equal(first_access.entity_ids, second_access.entity_ids)


def test_non_contiguous_party_ids_are_valid_when_supplied(
    feature_bundle: DatasetBundle,
) -> None:
    first_party = _party(3, feature_view_names=("item_tabular",))
    second_party = _party(101, feature_view_names=("item_text",))
    partition = FeaturePartition(
        name="non_contiguous",
        party_access={
            101: (FeatureAccess(view_name="item_text"),),
            3: (FeatureAccess(view_name="item_tabular"),),
        },
        metadata={},
    )

    validate_feature_partition(
        partition,
        feature_bundle,
        (first_party, second_party),
    )

    assert set(partition.party_access) == {3, 101}


def test_party_feature_summary_must_match_detailed_access(
    feature_bundle: DatasetBundle,
) -> None:
    party = _party(7, feature_view_names=("item_text",))
    partition = _partition(
        party.party_id,
        FeatureAccess(view_name="item_tabular"),
    )

    with pytest.raises(ValueError, match="do not match FeaturePartition"):
        validate_feature_partition(partition, feature_bundle, (party,))


@pytest.mark.parametrize(
    "entity_ids",
    [
        np.array([-1], dtype=np.int64),
        np.array([2, 1], dtype=np.int64),
    ],
)
def test_feature_access_rejects_invalid_entity_ids(
    entity_ids: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        FeatureAccess(view_name="item_tabular", entity_ids=entity_ids)


def test_feature_partition_rejects_negative_party_id() -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        FeaturePartition(
            name="invalid_party",
            party_access={-1: ()},
            metadata={},
        )


@pytest.mark.parametrize(
    "rows, fields, expected_checksum",
    [
        (
            None,
            None,
            "99327dd04859e1232f8868683718c1b21327d3c46f45f78656c42c0e32aae9cd",
        ),
        (
            (),
            (),
            "0d65e9d22fa64b1ef9f889dea50a9602ddeff245813def9f99541fe259d22aff",
        ),
        (
            (0, 4),
            ("brand",),
            "31159190a1b5c2cbf3be50ae12644dd2a2a3704a9b14c587f893efd35df9e895",
        ),
        (
            (0, 2, 4),
            ("category", "brand"),
            "245aa8836a5e204264c5bcd07bebaf90aa685aa62f206f83b626ca5b952b09c5",
        ),
    ],
    ids=["all", "empty", "subset", "explicit_full"],
)
def test_feature_selection_checksums_match_reviewed_revision(
    feature_bundle: DatasetBundle,
    rows: tuple[int, ...] | None,
    fields: tuple[str, ...] | None,
    expected_checksum: str,
) -> None:
    # Recorded from reviewed commit 5fad8cc before changing storage.
    party = _party(7, feature_view_names=("item_tabular",))
    access = FeatureAccess(
        view_name="item_tabular",
        entity_ids=None if rows is None else np.array(rows, dtype=np.int64),
        feature_names=fields,
    )
    partition = _partition(party.party_id, access)

    validate_feature_partition(partition, feature_bundle, (party,))

    assert feature_partition_checksum(partition) == expected_checksum


def test_feature_partition_checksum_is_order_independent() -> None:
    rows = np.array([0, 2], dtype=np.int64)
    first = FeaturePartition(
        name="ordered_one_way",
        party_access={
            9: (
                FeatureAccess(view_name="item_text"),
                FeatureAccess(
                    view_name="item_tabular",
                    entity_ids=rows,
                    feature_names=("category", "brand"),
                ),
            ),
            3: (FeatureAccess(view_name="item_text"),),
        },
        metadata={"seed": 2026},
    )
    second = FeaturePartition(
        name="ordered_another_way",
        party_access={
            3: (FeatureAccess(view_name="item_text"),),
            9: (
                FeatureAccess(
                    view_name="item_tabular",
                    entity_ids=rows,
                    feature_names=("brand", "category"),
                ),
                FeatureAccess(view_name="item_text"),
            ),
        },
        metadata={"seed": 2026},
    )
    changed = FeaturePartition(
        name="changed",
        party_access={
            3: (FeatureAccess(view_name="item_text"),),
            9: (
                FeatureAccess(
                    view_name="item_tabular",
                    entity_ids=np.array([0], dtype=np.int64),
                    feature_names=("brand", "category"),
                ),
                FeatureAccess(view_name="item_text"),
            ),
        },
        metadata={"seed": 2026},
    )

    first_checksum = feature_partition_checksum(first)

    assert first_checksum == feature_partition_checksum(second)
    assert first_checksum != feature_partition_checksum(changed)
    assert len(first_checksum) == 64
