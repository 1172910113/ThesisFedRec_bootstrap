import random

import numpy as np
import torch

from src.utils import set_random_seed


def _random_values() -> tuple[float, np.ndarray, torch.Tensor]:
    return random.random(), np.random.rand(3), torch.rand(3)


def test_random_seed_is_reproducible() -> None:
    set_random_seed(2026)
    first = _random_values()

    set_random_seed(2026)
    second = _random_values()

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
