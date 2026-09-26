from src.data.contracts import DatasetBundle
from src.data.negative_sampling import sample_negative_items


def test_negative_samples_exclude_all_known_positives_and_are_deterministic(
    processed_bundle: DatasetBundle,
) -> None:
    positive_ids = processed_bundle.splits.test_interaction_ids

    first = sample_negative_items(
        processed_bundle,
        positive_ids,
        negatives_per_positive=3,
        seed=23,
    )
    second = sample_negative_items(
        processed_bundle,
        positive_ids[::-1],
        negatives_per_positive=3,
        seed=23,
    )

    assert first == second
    for interaction_id, negative_items in first.items():
        user_id = int(processed_bundle.interactions.user_ids[interaction_id])
        known_items = set(
            processed_bundle.interactions.item_ids[
                processed_bundle.interactions.user_ids == user_id
            ].tolist()
        )
        assert len(negative_items) == 3
        assert tuple(sorted(negative_items)) == negative_items
        assert known_items.isdisjoint(negative_items)
        positive_item = int(
            processed_bundle.interactions.item_ids[interaction_id]
        )
        assert positive_item not in negative_items
