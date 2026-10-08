"""Deterministic seeding for reproducibility (FASER: fixed seeds, logged configs)."""

from __future__ import annotations

import hashlib
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def derive_seed(base_seed: int, *parts: str) -> int:
    """Stable per-item seed: same (base_seed, condition, item_id) -> same seed."""
    h = hashlib.sha256(("|".join([str(base_seed), *parts])).encode()).hexdigest()
    return int(h[:8], 16)
