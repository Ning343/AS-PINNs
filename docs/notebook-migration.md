# Notebook Migration Strategy

## Goal

The repository follows a layered migration strategy. Original notebooks remain available as research records, while reusable code is moved into Python modules, scripts, configurations, and tests.

## Layers

| Layer | Location | Responsibility |
| --- | --- | --- |
| Research notebooks | `notebooks/` | Preserve the original experiment flow, figures, exploratory cells, and training procedure |
| Direct Python ports | `scripts/notebook_ports/` | Provide a cell-order Python representation of each notebook for inspection and future refactoring |
| Reusable package | `src/as_pinns/` | Hold case metadata, reference profiles, training plans, controlled execution wrappers, and testable utilities |
| Case configuration | `configs/cases/` | Record domain, discontinuities, notebooks, ports, dependencies, and validation commands |
| Validation | `scripts/check_repository.py`, `tests/` | Verify repository structure and lightweight numerical behavior |

## Migration Rules

Direct ports can be regenerated with:

```bash
python scripts/export_notebooks_to_py.py
```

Reusable code belongs in `src/as_pinns/` when it is:

- independent of notebook display state
- deterministic enough for unit tests
- shared by multiple scripts or cases
- part of reproducibility metadata
- needed by command-line workflows

Notebook-only code can remain in notebooks or direct ports when it is:

- a one-off figure composition
- exploratory diagnostic plotting
- full training code that depends on DeepXDE/TensorFlow state
- a long-running experiment cell

## Current Migration Status

| Case | Direct `.py` port | Config metadata | Reference/profile function | Dry-run plan |
| --- | --- | --- | --- | --- |
| Function fitting | Available | Available | Available | Available |
| Force discontinuity | Available | Available | Available for load profile | Available |
| Force and material discontinuity | Available | Available | Available for load and stiffness profiles | Available |

The full DeepXDE training implementation remains traceable through the notebook ports. The maintained runner in `as_pinns.training.runner` launches those ports only when the user explicitly requests execution. Each run is isolated under `outputs/_intermediate/<case_id>/<run_name>/`, with a manifest, copied source files, a command log, and a machine-readable summary. This keeps the repository importable and testable without DeepXDE/TensorFlow while still providing a standard path for reproducible full training.
