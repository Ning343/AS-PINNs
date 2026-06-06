"""Lightweight reference profiles extracted from the original notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    x: np.ndarray
    values: dict[str, np.ndarray]


def _column(x: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1, 1)


def function_fitting_reference(x: np.ndarray | list[float]) -> Profile:
    """Return the piecewise reference field used in example 1."""

    x_arr = _column(x)
    y = np.piecewise(
        x_arr,
        [
            x_arr < 0.4,
            (x_arr >= 0.4) & (x_arr < 0.9),
            (x_arr >= 0.9) & (x_arr < 2.0),
            x_arr >= 2.0,
        ],
        [
            lambda z: 4.0 * z,
            lambda z: -(z**2) + 4.8 * z - 0.16,
            lambda z: 3.0 * z + 0.65,
            lambda z: 2.0 * z + 2.65,
        ],
    )
    slope = np.piecewise(
        x_arr,
        [
            x_arr < 0.4,
            (x_arr >= 0.4) & (x_arr < 0.9),
            (x_arr >= 0.9) & (x_arr < 2.0),
            x_arr >= 2.0,
        ],
        [4.0, lambda z: -2.0 * z + 4.8, 3.0, 2.0],
    )
    curvature = np.piecewise(
        x_arr,
        [
            x_arr < 0.4,
            (x_arr >= 0.4) & (x_arr < 0.9),
            (x_arr >= 0.9) & (x_arr < 2.0),
            x_arr >= 2.0,
        ],
        [0.0, -2.0, 0.0, 0.0],
    )
    return Profile(x=x_arr, values={"u": y, "du": slope, "d2u": curvature})


def force_profile(x: np.ndarray | list[float]) -> np.ndarray:
    """Distributed load profile for example 2."""

    x_arr = _column(x)
    return np.where(x_arr < 1.0 / 3.0, 30.0 - 30.0 * x_arr, np.where(x_arr < 0.75, 20.0, 0.0))


def force_material_profiles(x: np.ndarray | list[float]) -> Profile:
    """Load and bending-stiffness profiles for example 3."""

    x_arr = _column(x)
    load = np.where((x_arr >= 0.35) & (x_arr < 0.65), 20.0, 0.0)
    stiffness = np.where(x_arr < 0.5, 10.0, 5.0)
    return Profile(x=x_arr, values={"q": load, "ei": stiffness})


def sample_reference(case_id: str, points: int = 101) -> Profile:
    if points < 2:
        raise ValueError("points must be at least 2")
    if case_id == "function_fitting":
        x = np.linspace(0.0, 3.0, points).reshape(-1, 1)
        return function_fitting_reference(x)
    if case_id == "force_discontinuity":
        x = np.linspace(0.0, 1.0, points).reshape(-1, 1)
        return Profile(x=x, values={"q": force_profile(x)})
    if case_id == "force_material_discontinuity":
        x = np.linspace(0.0, 1.0, points).reshape(-1, 1)
        return force_material_profiles(x)
    raise KeyError(f"Unknown reference case: {case_id}")
