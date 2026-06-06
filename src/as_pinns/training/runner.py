"""Controlled execution for AS-PINNs Python case scripts.

The case scripts under ``scripts/cases`` preserve the executable AS-PINNs
training workflows. This module wraps those scripts in reproducible run
directories so full training can be launched explicitly without making
DeepXDE/TensorFlow required for ordinary package imports.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..cases import get_case
from ..execution import build_run_manifest, check_dependencies, write_run_manifest
from ..paths import project_root


class TrainingDependencyError(RuntimeError):
    """Raised when full-training dependencies are not available."""


class TrainingRunError(RuntimeError):
    """Raised when a training subprocess exits unsuccessfully."""


@dataclass(frozen=True)
class PreparedTrainingRun:
    """Paths and command line for a full AS-PINNs training run."""

    case_id: str
    output_directory: Path
    work_directory: Path
    manifest_path: Path
    script_path: Path
    solution_path: Path
    copied_script_path: Path
    copied_solution_path: Path
    log_path: Path
    summary_path: Path
    command: tuple[str, ...]
    missing_dependencies: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["command"] = list(self.command)
        payload["missing_dependencies"] = list(self.missing_dependencies)
        return payload


@dataclass(frozen=True)
class TrainingRunSummary:
    """Machine-readable summary produced after a training subprocess ends."""

    case_id: str
    started_at_utc: str
    finished_at_utc: str
    elapsed_seconds: float
    return_code: int
    success: bool
    output_directory: str
    work_directory: str
    manifest_path: str
    log_path: str
    generated_files: tuple[str, ...]
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["generated_files"] = list(self.generated_files)
        payload["command"] = list(self.command)
        return payload


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _copy_training_sources(case_id: str, work_directory: Path) -> tuple[Path, Path]:
    case = get_case(case_id)
    work_directory.mkdir(parents=True, exist_ok=True)
    script_target = work_directory / case.script_path.name
    solution_target = work_directory / case.solution_script_path.name
    shutil.copy2(case.script_path, script_target)
    shutil.copy2(case.solution_script_path, solution_target)
    return script_target, solution_target


def _relative_generated_files(work_directory: Path) -> tuple[str, ...]:
    files: list[str] = []
    for path in sorted(work_directory.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(work_directory).as_posix())
    return tuple(files)


def prepare_training_run(
    case_id: str,
    output_directory: str | Path = "outputs/_intermediate",
    run_name: str | None = None,
    *,
    require_dependencies: bool = True,
) -> PreparedTrainingRun:
    """Prepare an isolated full-training run directory.

    Parameters
    ----------
    case_id:
        Case identifier from the registry.
    output_directory:
        Root directory for experiment artifacts. A case-specific run directory
        is created below it.
    run_name:
        Optional stable run name. If omitted, a UTC timestamp is used.
    require_dependencies:
        When true, raise ``TrainingDependencyError`` if the full training
        dependencies from the case plan are not importable.
    """

    case = get_case(case_id)
    root = Path(output_directory)
    if not root.is_absolute():
        root = project_root() / root
    run_slug = run_name or _utc_timestamp()
    run_directory = root / case.case_id / run_slug
    work_directory = run_directory / "work"
    log_path = run_directory / "training.log"
    summary_path = run_directory / "training_summary.json"

    manifest = build_run_manifest(case_id, output_directory=str(run_directory))
    dependency_status = check_dependencies(manifest.plan)
    missing = tuple(status.package for status in dependency_status if not status.available)
    if require_dependencies and missing:
        install_hint = 'python -m pip install -e ".[training]"'
        raise TrainingDependencyError(
            "Missing full-training dependencies for "
            f"{case_id}: {', '.join(missing)}. Install them with: {install_hint}"
        )

    run_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = write_run_manifest(manifest, output_directory=run_directory)
    copied_script, copied_solution = _copy_training_sources(case_id, work_directory)
    command = (sys.executable, str(copied_script))

    prepared = PreparedTrainingRun(
        case_id=case.case_id,
        output_directory=run_directory,
        work_directory=work_directory,
        manifest_path=manifest_path,
        script_path=case.script_path,
        solution_path=case.solution_script_path,
        copied_script_path=copied_script,
        copied_solution_path=copied_solution,
        log_path=log_path,
        summary_path=summary_path,
        command=command,
        missing_dependencies=missing,
    )
    (run_directory / "prepared_run.json").write_text(
        json.dumps(prepared.to_dict(), indent=2), encoding="utf-8"
    )
    return prepared


def execute_training_run(
    case_id: str,
    output_directory: str | Path = "outputs/_intermediate",
    run_name: str | None = None,
    *,
    require_dependencies: bool = True,
) -> TrainingRunSummary:
    """Execute a Python case script in an isolated run directory."""

    prepared = prepare_training_run(
        case_id,
        output_directory=output_directory,
        run_name=run_name,
        require_dependencies=require_dependencies,
    )
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    src_path = str(project_root() / "src")
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    )

    started = datetime.now(timezone.utc).replace(microsecond=0)
    t0 = time.perf_counter()
    with prepared.log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(prepared.command) + "\n")
        log.write("Working directory: " + str(prepared.work_directory) + "\n\n")
        log.flush()
        completed = subprocess.run(
            prepared.command,
            cwd=prepared.work_directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - t0
    finished = datetime.now(timezone.utc).replace(microsecond=0)

    summary = TrainingRunSummary(
        case_id=prepared.case_id,
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        elapsed_seconds=round(elapsed, 6),
        return_code=completed.returncode,
        success=completed.returncode == 0,
        output_directory=str(prepared.output_directory),
        work_directory=str(prepared.work_directory),
        manifest_path=str(prepared.manifest_path),
        log_path=str(prepared.log_path),
        generated_files=_relative_generated_files(prepared.work_directory),
        command=prepared.command,
    )
    prepared.summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    if completed.returncode != 0:
        raise TrainingRunError(
            f"Training run for {case_id} failed with exit code {completed.returncode}. "
            f"See log: {prepared.log_path}"
        )
    return summary
