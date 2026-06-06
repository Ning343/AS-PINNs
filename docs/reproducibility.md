# Reproducibility

## Environment

The lightweight project layer requires Python 3.9 or later with NumPy, Matplotlib, and SymPy. Full training additionally requires DeepXDE, TensorFlow, and SciPy.

Lightweight setup:

```bash
python -m pip install -e .
```

Full training setup:

```bash
python -m pip install -e ".[training]"
```

The Conda environment file provides an alternative environment specification:

```bash
conda env create -f environment.yml
conda activate as-pinns
```

## Case Metadata

Case-level metadata is stored in `configs/cases/`. Each JSON file records:

- case identifier and problem description
- computational domain
- expected discontinuity locations
- Python case script paths
- optimizer schedule and iteration counts
- full training dependencies
- lightweight validation commands

## Lightweight Checks

Routine validation uses commands that do not launch full PINN training:

```bash
python scripts/check_repository.py
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src scripts tests
PYTHONPATH=src python -m as_pinns.cli list-cases
```

These checks verify structure, JSON metadata, absence of notebook files, Python case scripts, package imports, and lightweight reference profiles.

## Full Training

Full training should be treated as an experiment run rather than a routine CI step. Recommended practice:

1. Use a GPU-capable Linux or WSL environment when available.
2. Install DeepXDE and a compatible TensorFlow backend.
3. Launch the controlled case runner with `--execute`, or run the relevant Python case script directly when reproducing a benchmark workflow.
4. Save generated histories, interface traces, figures, and result arrays under `outputs/_intermediate/`.
5. Record the case ID, seed, package versions, optimizer schedule, runtime, and final metrics.

The command-line runner prints dry-run plans by default:

```bash
PYTHONPATH=src python scripts/run_case.py force_discontinuity
```

The same runner can write a JSON manifest before full training:

```bash
PYTHONPATH=src python scripts/run_case.py force_discontinuity --write-manifest
```

The manifest records the case ID, Python case script paths, random seed, learning-rate schedule, iteration schedule, output directory, Python version, platform, and training dependency status. This behavior prevents accidental long CPU training runs while still preserving the information required to reproduce a future training run.

The explicit execution path is:

```bash
PYTHONPATH=src python scripts/run_case.py force_discontinuity --execute
```

Execution creates a case-specific experiment directory:

```text
outputs/_intermediate/<case_id>/<run_name>/
|-- <case_id>_run_manifest.json
|-- prepared_run.json
|-- training.log
|-- training_summary.json
`-- work/
    |-- as_pinns_ex*.py
    `-- solution_ex*.py
```

The `work/` directory is the subprocess working directory. The runner copies the Python case script and its solution helper there before execution, so generated `.dat`, `.txt`, figure, and DeepXDE output files are contained inside the experiment run instead of being written to the repository root. `training.log` captures standard output and errors. `training_summary.json` records the return code, elapsed time, command, and files produced in `work/`.

If a required training dependency is missing, the runner stops before launching the subprocess and prints an installation hint. Full AS-PINNs training is not run by CI because it is hardware-sensitive and can take a long time.
