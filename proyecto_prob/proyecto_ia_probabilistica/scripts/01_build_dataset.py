"""Build the enriched dataset (IMDb + realistic deficiencies + OOD sources).

Usage:
    python scripts/01_build_dataset.py --imdb-size 20000 --ood-size-per-source 2000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

from src.data.preprocessing import main as preprocessing_main  # noqa: E402


if __name__ == "__main__":
    preprocessing_main()
