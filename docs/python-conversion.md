# Python Conversion Strategy

## Goal

The repository uses a Python-only project structure. Benchmark workflows are maintained as executable Python scripts, while reusable code lives in package modules, configurations, and tests.

## Layers

| Layer | Location | Responsibility |
| --- | --- | --- |
| Case scripts | `scripts/cases/` | Provide executable AS-PINNs training and reference workflows |
| Reusable package | `src/as_pinns/` | Hold case metadata, reference profiles, training plans, controlled execution wrappers, and testable utilities |
| Case configuration | `configs/cases/` | Record domain, discontinuities, Python script paths, dependencies, and validation commands |
| Validation | `scripts/check_repository.py`, `tests/` | Verify repository structure, metadata, and lightweight numerical behavior |

## Maintenance Rules

Reusable code belongs in `src/as_pinns/` when it is:

- deterministic enough for unit tests
- shared by multiple scripts or cases
- part of reproducibility metadata
- needed by command-line workflows

Case-specific experiment code can remain in `scripts/cases/` when it is:

- a one-off figure composition
- a full training loop that depends on DeepXDE/TensorFlow state
- a long-running experiment section
- a direct reproduction path for a published benchmark case

## Current Status

| Case | Python script | Config metadata | Reference/profile function | Dry-run plan |
| --- | --- | --- | --- | --- |
| Function fitting | Available | Available | Available | Available |
| Force discontinuity | Available | Available | Available for load profile | Available |
| Force and material discontinuity | Available | Available | Available for load and stiffness profiles | Available |

The full DeepXDE training implementation remains traceable through `scripts/cases/`. The maintained runner in `as_pinns.training.runner` launches those scripts only when the user explicitly requests execution. Each run is isolated under `outputs/_intermediate/<case_id>/<run_name>/`, with a manifest, copied source files, a command log, and a machine-readable summary.
