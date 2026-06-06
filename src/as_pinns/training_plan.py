"""Dry-run training plans for AS-PINNs Python case scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .cases import get_case


@dataclass(frozen=True)
class TrainingPlan:
    case_id: str
    title: str
    domain: tuple[float, float]
    expected_discontinuities: tuple[float, ...]
    trainable_interfaces: int
    python_script: str
    solution_script: str
    seed: int
    learning_rates: tuple[float, ...]
    iterations: tuple[int, ...]
    adaptive_anchor_points: int
    execute_requires: tuple[str, ...]
    output_directory: str
    default_mode: str = "dry-run"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_training_plan(case_id: str, output_directory: str = "outputs/_intermediate") -> TrainingPlan:
    case = get_case(case_id)
    return TrainingPlan(
        case_id=case.case_id,
        title=case.title,
        domain=case.domain,
        expected_discontinuities=case.expected_discontinuities,
        trainable_interfaces=case.trainable_interfaces,
        python_script=f"scripts/cases/{case.primary_script}",
        solution_script=f"scripts/cases/{case.solution_script}",
        seed=case.training.seed,
        learning_rates=case.training.learning_rates,
        iterations=case.training.iterations,
        adaptive_anchor_points=case.training.adaptive_anchor_points,
        execute_requires=case.training.execute_requires,
        output_directory=output_directory,
    )
