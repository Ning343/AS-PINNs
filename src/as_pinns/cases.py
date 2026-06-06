"""Case metadata for the AS-PINNs benchmark Python scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import case_scripts_dir


@dataclass(frozen=True)
class TrainingSchedule:
    seed: int
    learning_rates: tuple[float, ...]
    iterations: tuple[int, ...]
    residual_points_per_round: int
    adaptive_anchor_points: int
    execute_requires: tuple[str, ...]


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    title: str
    problem_type: str
    domain: tuple[float, float]
    expected_discontinuities: tuple[float, ...]
    trainable_interfaces: int
    primary_script: str
    solution_script: str
    outputs: tuple[str, ...]
    training: TrainingSchedule

    @property
    def script_path(self) -> Path:
        return case_scripts_dir() / self.primary_script

    @property
    def solution_script_path(self) -> Path:
        return case_scripts_dir() / self.solution_script


CASES: dict[str, CaseDefinition] = {
    "function_fitting": CaseDefinition(
        case_id="function_fitting",
        title="Discontinuous function fitting",
        problem_type="one-dimensional discontinuous function reconstruction",
        domain=(0.0, 3.0),
        expected_discontinuities=(0.4, 0.9, 2.0),
        trainable_interfaces=3,
        primary_script="function_fitting/as_pinns_ex1.py",
        solution_script="function_fitting/solution_ex1.py",
        outputs=("piecewise field", "first derivative", "second derivative"),
        training=TrainingSchedule(
            seed=1234,
            learning_rates=(1e-3, 1e-4),
            iterations=(20000, 20000),
            residual_points_per_round=100,
            adaptive_anchor_points=0,
            execute_requires=("deepxde", "tensorflow", "matplotlib", "scipy", "xlrd", "xlwt"),
        ),
    ),
    "force_discontinuity": CaseDefinition(
        case_id="force_discontinuity",
        title="Euler-Bernoulli beam with discontinuous loading",
        problem_type="fourth-order beam equation with source-term discontinuity",
        domain=(0.0, 1.0),
        expected_discontinuities=(1.0 / 3.0, 0.75),
        trainable_interfaces=1,
        primary_script="force_discontinuity/as_pinns_ex2.py",
        solution_script="force_discontinuity/solution_ex2.py",
        outputs=("deflection", "rotation", "moment", "shear", "load"),
        training=TrainingSchedule(
            seed=1234,
            learning_rates=(1e-3, 1e-4),
            iterations=(10000, 90000),
            residual_points_per_round=0,
            adaptive_anchor_points=20,
            execute_requires=("deepxde", "tensorflow", "matplotlib", "scipy"),
        ),
    ),
    "force_material_discontinuity": CaseDefinition(
        case_id="force_material_discontinuity",
        title="Euler-Bernoulli beam with discontinuous loading and stiffness",
        problem_type="fourth-order multi-material beam equation",
        domain=(0.0, 1.0),
        expected_discontinuities=(0.35, 0.5, 0.65),
        trainable_interfaces=3,
        primary_script="force_material_discontinuity/as_pinns_ex3.py",
        solution_script="force_material_discontinuity/solution_ex3.py",
        outputs=("deflection", "rotation", "moment", "shear", "load", "stiffness"),
        training=TrainingSchedule(
            seed=1234,
            learning_rates=(1e-3, 1e-4),
            iterations=(40000, 80000),
            residual_points_per_round=0,
            adaptive_anchor_points=20,
            execute_requires=("deepxde", "tensorflow", "matplotlib", "scipy"),
        ),
    ),
}

CASE_IDS = tuple(CASES)


def list_cases() -> tuple[CaseDefinition, ...]:
    return tuple(CASES[case_id] for case_id in CASE_IDS)


def get_case(case_id: str) -> CaseDefinition:
    try:
        return CASES[case_id]
    except KeyError as exc:
        available = ", ".join(CASE_IDS)
        raise KeyError(f"Unknown case_id {case_id!r}. Available cases: {available}") from exc
