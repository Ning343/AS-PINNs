# Notebook Inventory

The original notebooks are preserved under `notebooks/` with stable, space-free paths. Each notebook has a direct Python port under `scripts/notebook_ports/`. Reusable metadata and lightweight numerical references are maintained under `src/as_pinns/`.

| Case ID | Original role | Notebook | Python port | Package layer |
| --- | --- | --- | --- | --- |
| `function_fitting` | AS-PINNs discontinuous function fitting experiment | `notebooks/function_fitting/as_pinns_ex1.ipynb` | `scripts/notebook_ports/function_fitting/as_pinns_ex1.py` | `as_pinns.cases`, `as_pinns.reference_solutions` |
| `function_fitting` | Analytical reference profile | `notebooks/function_fitting/solution_ex1.ipynb` | `scripts/notebook_ports/function_fitting/solution_ex1.py` | `function_fitting_reference` |
| `force_discontinuity` | Beam experiment with load discontinuity | `notebooks/force_discontinuity/as_pinns_ex2.ipynb` | `scripts/notebook_ports/force_discontinuity/as_pinns_ex2.py` | `as_pinns.cases`, `as_pinns.training_plan` |
| `force_discontinuity` | Beam reference solution construction | `notebooks/force_discontinuity/solution_ex2.ipynb` | `scripts/notebook_ports/force_discontinuity/solution_ex2.py` | `force_profile` |
| `force_material_discontinuity` | Beam experiment with load and stiffness discontinuities | `notebooks/force_material_discontinuity/as_pinns_ex3.ipynb` | `scripts/notebook_ports/force_material_discontinuity/as_pinns_ex3.py` | `as_pinns.cases`, `as_pinns.training_plan` |
| `force_material_discontinuity` | Beam reference profile construction | `notebooks/force_material_discontinuity/solution_ex3.ipynb` | `scripts/notebook_ports/force_material_discontinuity/solution_ex3.py` | `force_material_profiles` |

## Notebook Role

The notebooks are research records. They contain full plotting, training loops, residual-adaptive point insertion, and case-specific visual diagnostics. They are useful for tracing the original experiment procedure.

The Python package is the maintained execution layer for reusable behavior. It records case identifiers, expected discontinuities, notebook paths, training schedules, reference profiles, dry-run plans, and controlled full-training execution wrappers.

## Porting Policy

Direct ports keep the original notebook execution order. They are not treated as the final maintainable API. Shared code should move into `src/as_pinns/` when it is deterministic, reusable, and testable without launching full neural-network training.
