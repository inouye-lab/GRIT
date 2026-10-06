"""JSON records shared by the UrbanCars search and report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def read_json(path: Path) -> dict[str, Any]:
    result: object = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return cast(dict[str, Any], result)


def write_json(path: Path, value: object) -> None:
    """Atomic publication; interrupted computations cannot appear complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)
