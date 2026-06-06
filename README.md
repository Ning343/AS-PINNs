# AS-PINNs

AS-PINNs provides reproducible examples for adversarial and self-adaptive domain decomposition physics-informed neural networks. The method targets high-order differential equations with discontinuities, where fixed domain decomposition and conventional PINN losses can become difficult to tune.

The repository corresponds to:

> Mingsheng Peng, Hesheng Tang, Yingwei Kou. Adversarial and self-adaptive domain decomposition physics-informed neural networks for high-order differential equations with discontinuities. Engineering Applications of Artificial Intelligence, 145, 110156, 2025. DOI: 10.1016/j.engappai.2025.110156.

## Repository Structure

```text
.
|-- configs/
|   `-- cases/                         # Case-level reproduction metadata
|-- docs/                              # Method, migration, and reproduction notes
|-- scripts/
|   |-- check_repository.py            # Lightweight structural validation
|   |-- run_case.py                    # Case manifest and full-training wrapper
|   `-- cases/                         # Executable Python benchmark scripts
|-- src/
|   `-- as_pinns/                      # Reusable and testable Python package layer
|-- tests/                             # Lightweight tests
|-- pyproject.toml
|-- requirements.txt
|-- environment.yml
|-- LICENSE
`-- CITATION.cff
```

## Benchmark Cases

| Case ID | Problem | Python script |
| --- | --- | --- |
| `function_fitting` | Discontinuous function fitting | `scripts/cases/function_fitting/as_pinns_ex1.py` |
| `force_discontinuity` | Euler-Bernoulli beam with discontinuous loading | `scripts/cases/force_discontinuity/as_pinns_ex2.py` |
| `force_material_discontinuity` | Euler-Bernoulli beam with discontinuous loading and stiffness | `scripts/cases/force_material_discontinuity/as_pinns_ex3.py` |

The repository is Python-only. The Python package and scripts provide reusable case metadata, reference profiles, dry-run training plans, controlled training execution, and repository checks.

## Installation

For lightweight inspection and tests:

```bash
python -m pip install -e .
```

For full DeepXDE training, install the training extras in a suitable TensorFlow environment:

```bash
python -m pip install -e ".[training]"
```

GPU-capable Linux or WSL environments are recommended for full PINN training. The default command-line entry points are dry-run utilities and do not launch expensive training.

## Usage

List available cases:

```bash
PYTHONPATH=src python -m as_pinns.cli list-cases
```

Print a case summary:

```bash
PYTHONPATH=src python -m as_pinns.cli case-summary force_discontinuity
```

Print a reproducible dry-run training plan:

```bash
PYTHONPATH=src python scripts/run_case.py force_material_discontinuity
```

Write a run manifest before full training:

```bash
PYTHONPATH=src python scripts/run_case.py force_material_discontinuity --write-manifest
```

Launch a full Python case run after installing the training dependencies:

```bash
PYTHONPATH=src python scripts/run_case.py force_material_discontinuity --execute
```

The explicit execution path creates an isolated directory under `outputs/_intermediate/<case_id>/<run_name>/`, copies the case script and its solution helper into that directory, writes a manifest, captures command output in `training.log`, and writes `training_summary.json` after the subprocess exits. If DeepXDE, TensorFlow, or another required training dependency is unavailable, the command exits before starting a long training run.

Run lightweight validation:

```bash
python scripts/check_repository.py
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

## Documentation

- `docs/method-overview.md` describes the AS-PINNs method and the benchmark scope.
- `docs/python-case-inventory.md` maps every Python case script to its package-level counterpart.
- `docs/python-conversion.md` defines how Python case scripts are separated from reusable package modules and controlled execution wrappers.
- `docs/reproducibility.md` describes environment setup and validation levels.
- `docs/development.md` gives maintenance rules for adding or revising cases.

## Citation

Please cite the paper above when using the code or examples. A machine-readable citation entry is available in `CITATION.cff`.

## License

The repository is distributed under the MIT License. See `LICENSE` for details.
