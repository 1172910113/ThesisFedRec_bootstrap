"""Shared random-seed handling."""

import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for best-effort reproducibility."""
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 2**32 - 1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These settings reduce cuDNN nondeterminism but cannot guarantee that
    # every PyTorch or CUDA operation is deterministic.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
