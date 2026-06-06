"""Execution helpers for reproducible AS-PINNs training runs."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .training_plan import TrainingPlan, build_training_plan


@dataclass(frozen=True)
class DependencyStatus:
    package: str
    available: bool


@dataclass(frozen=True)
class RunManifest:
    plan: TrainingPlan
    created_at_utc: str
    python_version: str
    platform: str
    dependency_status: tuple[DependencyStatus, ...]
    ready_for_training: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["plan"] = self.plan.to_dict()
        return payload


def check_dependencies(plan: TrainingPlan) -> tuple[DependencyStatus, ...]:
    return tuple(
        DependencyStatus(package=name, available=importlib.util.find_spec(name) is not None)
        for name in plan.execute_requires
    )


def build_run_manifest(case_id: str, output_directory: str = "outputs/_intermediate") -> RunManifest:
    plan = build_training_plan(case_id, output_directory=output_directory)
    dependency_status = check_dependencies(plan)
    missing = tuple(status.package for status in dependency_status if not status.available)
    notes = (
        "The CLI writes a manifest by default and does not launch full DeepXDE training.",
        "Run direct notebook ports or notebooks in a prepared GPU/Linux environment for full training.",
    )
    if missing:
        notes += (f"Missing training dependencies: {', '.join(missing)}.",)
    return RunManifest(
        plan=plan,
        created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        dependency_status=dependency_status,
        ready_for_training=not missing,
        notes=notes,
    )


def write_run_manifest(manifest: RunManifest, output_directory: str | Path | None = None) -> Path:
    target_root = Path(output_directory or manifest.plan.output_directory)
    target_root.mkdir(parents=True, exist_ok=True)
    path = target_root / f"{manifest.plan.case_id}_run_manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path
