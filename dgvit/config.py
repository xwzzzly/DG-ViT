from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or base is None:
        return candidate
    return Path(base) / candidate
