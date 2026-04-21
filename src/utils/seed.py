"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

try:
    import pyro
    _HAS_PYRO = True
except Exception:
    _HAS_PYRO = False


def set_seed(seed: int) -> None:
    """Set seed across random, numpy, torch and pyro (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _HAS_PYRO:
        pyro.set_rng_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
