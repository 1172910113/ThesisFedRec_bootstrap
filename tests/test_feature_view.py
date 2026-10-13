from dataclasses import replace

import numpy as np
import pytest

from src.data.contracts import DatasetBundle, FeatureView


def _make_feature_view(**overrides: object) -> FeatureView:
    fields: dict[str, object] = {
        "name": "item_numeric",
        "entity_kind": "item",
        "entity_ids": np.array([0, 2], dtype=np.int64),
        "modality": "tabular",
        "values": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "feature_names": ("first", "second"),
        "missing_mask": np.array(
            [[False, True], [False, False]], dtype=np.bool_
        ),
        "metadata": {"source": "synthetic"},
    }
    fields.update(overrides)
    return FeatureView(**fields)  # type: ignore[arg-type]


def test_valid_feature_view_construction() -> None:
    view = _make_feature_view()

    assert view.name == "item_numeric"
    assert view.entity_kind == "item"
    assert view.feature_names == ("first", "second")
    np.testing.assert_array_equal(
        view.entity_ids,
        np.array([0, 2], dtype=np.int64),
    )

    text_view = _make_feature_view(
        name="user_text",
        entity_kind="user",
        entity_ids=np.array([0, 1], dtype=np.int64),
        modality="text",
        values=("first user", "second user"),
        feature_names=(),
        missing_mask=None,
    )
    assert text_view.values == ("first user", "second user")


@pytest.mark.parametrize(
    ("values", "feature_names", "error"),
    [
        (
            np.ones((1, 2), dtype=np.float32),
            ("first", "second"),
            "same row count",
        ),
        (
            np.ones((2, 2), dtype=np.float32),
            ("only_one",),
            "match the values columns",
        ),
        (
            np.ones(2, dtype=np.float32),
            ("scalar",),
            "must be two-dimensional",
        ),
    ],
)
def test_feature_view_rejects_malformed_value_alignment(
    values: np.ndarray,
    feature_names: tuple[str, ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _make_feature_view(values=values, feature_names=feature_names)


@pytest.mark.parametrize(
    ("entity_ids", "error_type", "error"),
    [
        (
            np.array([0, 0], dtype=np.int64),
            ValueError,
            "sorted and contain unique IDs",
        ),
        (
            np.array([-1, 1], dtype=np.int64),
            ValueError,
            "must be non-negative",
        ),
        (
            np.array([0, 1], dtype=np.int32),
            TypeError,
            "dtype int64",
        ),
    ],
)
def test_feature_view_rejects_duplicate_or_invalid_entity_ids(
    entity_ids: np.ndarray,
    error_type: type[Exception],
    error: str,
) -> None:
    with pytest.raises(error_type, match=error):
        _make_feature_view(entity_ids=entity_ids)


def test_feature_view_rejects_invalid_entity_kind() -> None:
    with pytest.raises(ValueError, match="unsupported FeatureView entity_kind"):
        _make_feature_view(entity_kind="party")


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"name": "  "}, "name must not be empty"),
        ({"modality": ""}, "modality must not be empty"),
        (
            {"feature_names": ("duplicate", "duplicate")},
            "feature_names must be unique",
        ),
    ],
)
def test_feature_view_rejects_invalid_identity_fields(
    overrides: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _make_feature_view(**overrides)


@pytest.mark.parametrize(
    ("missing_mask", "error_type", "error"),
    [
        (
            np.zeros((2, 1), dtype=np.bool_),
            ValueError,
            "must match the values shape",
        ),
        (
            np.zeros((2, 2), dtype=np.int64),
            TypeError,
            "must have boolean dtype",
        ),
    ],
)
def test_feature_view_rejects_invalid_missing_mask(
    missing_mask: np.ndarray,
    error_type: type[Exception],
    error: str,
) -> None:
    with pytest.raises(error_type, match=error):
        _make_feature_view(missing_mask=missing_mask)


def test_dataset_bundle_accepts_valid_canonical_feature_view(
    processed_bundle: DatasetBundle,
) -> None:
    view = _make_feature_view()

    bundle = replace(
        processed_bundle,
        feature_views={view.name: view},
    )

    assert bundle.feature_views == {"item_numeric": view}


@pytest.mark.parametrize("entity_kind", ["user", "item", "interaction"])
def test_dataset_bundle_rejects_out_of_range_feature_entity_ids(
    processed_bundle: DatasetBundle,
    entity_kind: str,
) -> None:
    limits = {
        "user": len(processed_bundle.user_id_map),
        "item": len(processed_bundle.item_id_map),
        "interaction": len(processed_bundle.interactions),
    }
    view = _make_feature_view(
        name=f"{entity_kind}_numeric",
        entity_kind=entity_kind,
        entity_ids=np.array([limits[entity_kind]], dtype=np.int64),
        values=np.ones((1, 1), dtype=np.float32),
        feature_names=("value",),
        missing_mask=None,
    )

    with pytest.raises(ValueError, match="out-of-range"):
        replace(processed_bundle, feature_views={view.name: view})


def test_dataset_bundle_requires_feature_view_key_to_match_name(
    processed_bundle: DatasetBundle,
) -> None:
    view = _make_feature_view()

    with pytest.raises(ValueError, match="keys must match"):
        replace(processed_bundle, feature_views={"different_name": view})


def test_multiple_named_feature_views_can_coexist(
    processed_bundle: DatasetBundle,
) -> None:
    item_view = _make_feature_view()
    user_view = _make_feature_view(
        name="user_text",
        entity_kind="user",
        entity_ids=np.array([0], dtype=np.int64),
        modality="text",
        values=("profile",),
        feature_names=(),
        missing_mask=None,
    )

    bundle = replace(
        processed_bundle,
        feature_views={item_view.name: item_view, user_view.name: user_view},
    )

    assert tuple(bundle.feature_views) == ("item_numeric", "user_text")


def test_existing_empty_view_bundle_remains_valid(
    processed_bundle: DatasetBundle,
) -> None:
    assert processed_bundle.feature_views == {}
