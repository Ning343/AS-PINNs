"""JSON configuration loader for AS-PINNs cases."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import config_dir


def load_case_config(case_id: str) -> dict[str, object]:
    path = config_dir() / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing case configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_case_config_paths() -> tuple[Path, ...]:
    return tuple(sorted(config_dir().glob("*.json")))
