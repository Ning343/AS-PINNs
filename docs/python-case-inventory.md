# Python Case Inventory

The repository is Python-only. The AS-PINNs benchmark workflows are maintained as executable scripts under `scripts/cases/`, with reusable metadata and lightweight numerical references under `src/as_pinns/`.

| Case ID | Role | Python script | Package layer |
| --- | --- | --- | --- |
| `function_fitting` | AS-PINNs discontinuous function fitting experiment | `scripts/cases/function_fitting/as_pinns_ex1.py` | `as_pinns.cases`, `as_pinns.reference_solutions` |
| `function_fitting` | Analytical reference profile | `scripts/cases/function_fitting/solution_ex1.py` | `function_fitting_reference` |
| `force_discontinuity` | Beam experiment with load discontinuity | `scripts/cases/force_discontinuity/as_pinns_ex2.py` | `as_pinns.cases`, `as_pinns.training_plan` |
| `force_discontinuity` | Beam reference solution construction | `scripts/cases/force_discontinuity/solution_ex2.py` | `force_profile` |
| `force_material_discontinuity` | Beam experiment with load and stiffness discontinuities | `scripts/cases/force_material_discontinuity/as_pinns_ex3.py` | `as_pinns.cases`, `as_pinns.training_plan` |
| `force_material_discontinuity` | Beam reference profile construction | `scripts/cases/force_material_discontinuity/solution_ex3.py` | `force_material_profiles` |

## Script Role

The case scripts contain the full plotting, training loops, residual-adaptive point insertion, and case-specific visual diagnostics needed to reproduce the original AS-PINNs examples. The Python package records case identifiers, expected discontinuities, script paths, training schedules, reference profiles, dry-run plans, and controlled full-training execution wrappers.

## Maintenance Policy

The scripts under `scripts/cases/` are the canonical executable case sources. Shared code belongs in `src/as_pinns/` when it is deterministic, reusable, and testable without launching full neural-network training.
