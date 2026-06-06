"""Lightweight structural checks for the AS-PINNs repository."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "environment.yml",
    "CITATION.cff",
    "src/as_pinns/__init__.py",
    "src/as_pinns/cases.py",
    "src/as_pinns/reference_solutions.py",
    "src/as_pinns/training_plan.py",
    "src/as_pinns/execution.py",
    "src/as_pinns/training/__init__.py",
    "src/as_pinns/training/runner.py",
    "scripts/clean_notebooks.py",
    "scripts/export_notebooks_to_py.py",
    "docs/method-overview.md",
    "docs/notebook-migration.md",
    "docs/reproducibility.md",
    "docs/development.md",
    "docs/notebook-inventory.md",
    ".github/workflows/checks.yml",
]

REQUIRED_CONFIG_FIELDS = {
    "case_id",
    "title",
    "problem_type",
    "domain",
    "expected_discontinuities",
    "notebook",
    "solution_notebook",
    "python_port",
    "solution_port",
    "training",
    "lightweight_checks",
}

REQUIRED_TRAINING_FIELDS = {
    "seed",
    "optimizer",
    "learning_rates",
    "iterations",
    "trainable_interfaces",
    "execute_requires",
}


def _check_path(path: str) -> None:
    if not (ROOT / path).exists():
        raise AssertionError(f"Missing required path: {path}")


def _check_notebook(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "cells" not in payload:
        raise AssertionError(f"Notebook has no cells: {path}")
    if not any(cell.get("cell_type") == "code" for cell in payload["cells"]):
        raise AssertionError(f"Notebook has no code cells: {path}")
    for index, cell in enumerate(payload["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            raise AssertionError(f"Notebook cell {index} has execution_count: {path}")
        if cell.get("outputs"):
            raise AssertionError(f"Notebook cell {index} has saved outputs: {path}")


def _check_config(path: Path) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_CONFIG_FIELDS - set(config)
    if missing:
        raise AssertionError(f"{path} is missing fields: {sorted(missing)}")

    training = config["training"]
    missing_training = REQUIRED_TRAINING_FIELDS - set(training)
    if missing_training:
        raise AssertionError(f"{path} training section is missing fields: {sorted(missing_training)}")

    if len(config["domain"]) != 2 or config["domain"][0] >= config["domain"][1]:
        raise AssertionError(f"{path} has an invalid domain")

    for key in ("notebook", "solution_notebook", "python_port", "solution_port"):
        referenced = ROOT / config[key]
        if not referenced.exists():
            raise AssertionError(f"{path} references a missing {key}: {config[key]}")

    _check_notebook(ROOT / config["notebook"])
    _check_notebook(ROOT / config["solution_notebook"])


def main() -> int:
    for path in REQUIRED_PATHS:
        _check_path(path)

    config_paths = sorted((ROOT / "configs" / "cases").glob("*.json"))
    if len(config_paths) != 3:
        raise AssertionError(f"Expected 3 case configs, found {len(config_paths)}")
    for path in config_paths:
        _check_config(path)

    port_paths = sorted((ROOT / "scripts" / "notebook_ports").rglob("*.py"))
    if len(port_paths) != 6:
        raise AssertionError(f"Expected 6 notebook Python ports, found {len(port_paths)}")

    print("AS-PINNs repository structure check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
