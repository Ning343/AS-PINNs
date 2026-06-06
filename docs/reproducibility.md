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
- notebook and Python-port paths
- optimizer schedule and iteration counts
- full training dependencies
- lightweight validation commands

## Lightweight Checks

Routine validation uses commands that do not launch full PINN training:

```bash
python scripts/check_repository.py
python scripts/clean_notebooks.py --check
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src scripts tests
PYTHONPATH=src python -m as_pinns.cli list-cases
```

These checks verify structure, JSON metadata, notebook readability, notebook output hygiene, direct Python ports, package imports, and lightweight reference profiles.

## Full Training

Full training should be treated as an experiment run rather than a routine CI step. Recommended practice:

1. Use a GPU-capable Linux or WSL environment when available.
2. Install DeepXDE and a compatible TensorFlow backend.
3. Run the relevant notebook or direct Python port.
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

The manifest records the case ID, notebook path, Python port, random seed, learning-rate schedule, iteration schedule, output directory, Python version, platform, and training dependency status. This behavior prevents accidental long CPU training runs while still preserving the information required to reproduce a future training run.
