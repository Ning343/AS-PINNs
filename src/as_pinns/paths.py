"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return project_root() / "configs" / "cases"


def notebooks_dir() -> Path:
    return project_root() / "notebooks"


def notebook_ports_dir() -> Path:
    return project_root() / "scripts" / "notebook_ports"
