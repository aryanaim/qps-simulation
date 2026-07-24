"""Small file I/O helpers used by all workstreams."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def write_json(path: str | Path, data: object) -> Path:
    """Write JSON with stable formatting."""
    output = Path(path)
    ensure_dir(output.parent)
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return output


def read_json(path: str | Path) -> object:
    """Read JSON from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write a list of dictionaries as CSV."""
    output = Path(path)
    ensure_dir(output.parent)
    materialized = list(rows)
    if fieldnames is None:
        seen: list[str] = []
        for row in materialized:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fieldnames = seen

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output

