"""I/O helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
