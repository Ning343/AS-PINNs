# Development Guide

## Project Layout

The repository is organized around reproducible research use:

| Directory | Purpose |
| --- | --- |
| `scripts/cases/` | Executable AS-PINNs benchmark and reference scripts |
| `src/as_pinns/` | Reusable Python package layer |
| `configs/cases/` | Case metadata and validation commands |
| `docs/` | Method, reproduction, Python script, and maintenance documentation |
| `tests/` | Lightweight tests that avoid full neural-network training |

## Adding a Case

1. Place executable AS-PINNs scripts under `scripts/cases/<case_id>/`.
2. Add or update the corresponding reference script when the case needs one.
3. Add a case JSON file under `configs/cases/`.
4. Register the case in `src/as_pinns/cases.py`.
5. Add deterministic reference utilities when available.
6. Add lightweight tests for metadata and reference profiles.
7. Update `docs/python-case-inventory.md`.
8. Run repository checks before publishing.

## Code Policy

Full training code should not execute at import time. Reusable modules should expose functions, dataclasses, and command-line utilities that can be tested without DeepXDE/TensorFlow. Deep learning dependencies should be imported lazily in training execution paths.

Training entry points should produce or update a run manifest before expensive execution. The manifest is the minimum reproducibility record for case ID, seed, optimizer schedule, dependency status, runtime environment, and output directory.

## Validation Commands

```bash
python scripts/check_repository.py
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

Full training validation should state hardware, backend versions, runtime, case ID, random seed, and output location.
