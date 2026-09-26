"""Deterministic interaction splitting."""

import numpy as np

from src.data.contracts import InteractionSplits, InteractionTable


def leave_one_out_split(interactions: InteractionTable) -> InteractionSplits:
    """Create temporal leave-one-out validation and test splits per user."""
    if len(interactions) == 0:
        raise ValueError("cannot split an empty interaction table")

    order = np.lexsort(
        (
            interactions.interaction_ids,
            interactions.timestamps,
            interactions.user_ids,
        )
    )
    ordered_users = interactions.user_ids[order]
    _, counts = np.unique(ordered_users, return_counts=True)
    if np.any(counts < 3):
        raise ValueError("every retained user must have at least three interactions")

    group_ends = np.cumsum(counts)
    test_positions = group_ends - 1
    valid_positions = group_ends - 2

    evaluation_positions = np.zeros(len(interactions), dtype=bool)
    evaluation_positions[test_positions] = True
    evaluation_positions[valid_positions] = True

    ordered_interaction_ids = interactions.interaction_ids[order]
    splits = InteractionSplits(
        train_interaction_ids=np.sort(
            ordered_interaction_ids[~evaluation_positions]
        ).astype(np.int64, copy=False),
        valid_interaction_ids=np.sort(
            ordered_interaction_ids[valid_positions]
        ).astype(np.int64, copy=False),
        test_interaction_ids=np.sort(
            ordered_interaction_ids[test_positions]
        ).astype(np.int64, copy=False),
    )
    splits.validate_coverage(len(interactions))

    valid_users = interactions.user_ids[splits.valid_interaction_ids]
    test_users = interactions.user_ids[splits.test_interaction_ids]
    expected_users = np.unique(interactions.user_ids)
    if not np.array_equal(np.sort(valid_users), expected_users):
        raise ValueError("every user must have exactly one validation interaction")
    if not np.array_equal(np.sort(test_users), expected_users):
        raise ValueError("every user must have exactly one test interaction")
    return splits
