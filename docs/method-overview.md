# Method Overview

## Research Context

Physics-informed neural networks (PINNs) solve differential equations by embedding governing equations and boundary conditions into neural-network training. For high-order equations with discontinuous loads, material properties, or response derivatives, conventional smooth neural networks may accumulate large errors near discontinuities. Domain decomposition can reduce this difficulty, but fixed interfaces require prior knowledge and extra interface-condition losses.

The AS-PINNs paper proposes adversarial and self-adaptive domain decomposition physics-informed neural networks for these settings. The method combines reduced-order PINN ideas with trainable interface positions. Subnetworks compete through residual information so that the computational domain can adapt to discontinuity locations during training.

Reference paper:

Peng, M., Tang, H., and Kou, Y. Adversarial and self-adaptive domain decomposition physics-informed neural networks for high-order differential equations with discontinuities. Engineering Applications of Artificial Intelligence, 145, 110156, 2025. DOI: 10.1016/j.engappai.2025.110156.

## Core Method

The paper considers high-order differential equations with spatially varying system parameters and source terms. In reduced-order form, the primary response and its derivatives are represented by separate network outputs. This avoids repeatedly differentiating a single network to very high order and allows boundary constraints to be imposed more directly on selected variables.

AS-PINNs extend this structure by assigning subnetworks to subdomains. Interface positions are represented as trainable variables. Residual differences between adjacent subdomains guide adversarial adjustment of the interface locations. When the residual pattern indicates that a discontinuity is better represented by another decomposition, the trainable interfaces move accordingly.

## Repository Scope

The repository contains three benchmark groups from the original code release:

| Case | Purpose | Main discontinuity |
| --- | --- | --- |
| Function fitting | Demonstrates adaptive representation of a discontinuous one-dimensional function and derivatives | Piecewise function slopes and curvature |
| Force discontinuity | Solves an Euler-Bernoulli beam case with a discontinuous load profile | Source-term discontinuity |
| Force and material discontinuity | Solves a beam case with both load and stiffness changes | Source and material discontinuities |

The repository is organized as a Python-only reproducibility project. Executable benchmark scripts under `scripts/cases/` provide the full AS-PINNs workflows, while the package under `src/as_pinns/` contains maintained metadata, reference profiles, dry-run training plans, controlled execution, and validation utilities.

Training execution is intentionally manifest-first. The package can build a reproducibility manifest that records the selected benchmark case, training schedule, output directory, platform, Python version, and dependency status before any long DeepXDE run is started. When the user explicitly passes `--execute`, the runner launches the Python case script in an isolated experiment directory and captures logs, copied sources, and a summary JSON.

## Validation Levels

The repository separates validation into two levels:

| Level | Purpose | Typical command |
| --- | --- | --- |
| Lightweight repository validation | Checks project structure, case metadata, absence of notebook files, Python imports, and reference profiles | `python scripts/check_repository.py` and `python -m unittest discover -s tests` |
| Full training validation | Runs DeepXDE/TensorFlow training for each benchmark case | `PYTHONPATH=src python scripts/run_case.py <case_id> --execute` in a prepared GPU/Linux environment |

Full training is environment-sensitive and can be expensive. Lightweight checks are intended for routine repository health and CI.
