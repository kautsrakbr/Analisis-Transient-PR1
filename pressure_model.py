"""Core calculations for an infinite-acting, homogeneous radial-flow well."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, log

import numpy as np
from scipy.special import kve


@dataclass(frozen=True)
class ReservoirInputs:
    permeability_md: float
    porosity: float
    total_compressibility_psi_inv: float
    viscosity_cp: float
    formation_volume_factor_rb_stb: float
    thickness_ft: float
    wellbore_radius_ft: float
    rate_stb_day: float
    initial_pressure_psi: float


def stehfest_weights(n_terms: int = 12) -> np.ndarray:
    """Return Gaver-Stehfest weights for an even number of terms."""
    if n_terms < 6 or n_terms > 18 or n_terms % 2:
        raise ValueError("n_terms must be an even integer between 6 and 18")

    half = n_terms // 2
    weights = np.zeros(n_terms, dtype=float)
    for k in range(1, n_terms + 1):
        total = 0.0
        for j in range((k + 1) // 2, min(k, half) + 1):
            numerator = (j**half) * factorial(2 * j)
            denominator = (
                factorial(half - j)
                * factorial(j)
                * factorial(j - 1)
                * factorial(k - j)
                * factorial(2 * j - k)
            )
            total += numerator / denominator
        weights[k - 1] = ((-1) ** (k + half)) * total
    return weights


def laplace_wellbore_pressure(u: np.ndarray | float) -> np.ndarray:
    r"""Finite-radius well pressure in Laplace space.

    The unit-flux inner boundary at r_D=1 gives
    pbar_D(u) = K0(sqrt(u)) / [u^(3/2) K1(sqrt(u))].
    Scaled Bessel functions avoid exponential underflow while preserving K0/K1.
    """
    u_array = np.asarray(u, dtype=float)
    if np.any(u_array <= 0):
        raise ValueError("Laplace variable u must be positive")
    root_u = np.sqrt(u_array)
    bessel_ratio = kve(0, root_u) / kve(1, root_u)
    return bessel_ratio / np.power(u_array, 1.5)


def invert_pressure_gaver_stehfest(
    dimensionless_time: np.ndarray | float, n_terms: int = 12
) -> np.ndarray:
    """Invert the Laplace pressure solution to real dimensionless time."""
    td = np.atleast_1d(np.asarray(dimensionless_time, dtype=float))
    if np.any(td <= 0) or np.any(~np.isfinite(td)):
        raise ValueError("Dimensionless time must contain finite positive values")

    weights = stehfest_weights(n_terms)
    indices = np.arange(1, n_terms + 1, dtype=float)
    u = np.log(2.0) * indices[None, :] / td[:, None]
    pbar = laplace_wellbore_pressure(u)
    pd = (np.log(2.0) / td) * np.sum(weights[None, :] * pbar, axis=1)
    return pd


def dimensionless_time(time_hours: np.ndarray, inputs: ReservoirInputs) -> np.ndarray:
    """Convert field-unit time to dimensionless radial-flow time."""
    return (
        0.0002637
        * inputs.permeability_md
        * np.asarray(time_hours, dtype=float)
        / (
            inputs.porosity
            * inputs.viscosity_cp
            * inputs.total_compressibility_psi_inv
            * inputs.wellbore_radius_ft**2
        )
    )


def dimensionless_wellbore_storage(
    storage_bbl_psi: float, inputs: ReservoirInputs
) -> float:
    """Convert wellbore storage C to C_D; C=0 for this model."""
    return (
        0.8936
        * storage_bbl_psi
        / (
            inputs.porosity
            * inputs.total_compressibility_psi_inv
            * inputs.thickness_ft
            * inputs.wellbore_radius_ft**2
        )
    )


def simulate_drawdown(
    time_hours: np.ndarray,
    inputs: ReservoirInputs,
    n_terms: int = 12,
) -> dict[str, np.ndarray | float]:
    """Simulate constant-rate drawdown and its logarithmic derivative."""
    time = np.asarray(time_hours, dtype=float)
    if time.ndim != 1 or len(time) < 3 or np.any(time <= 0):
        raise ValueError("time_hours must be a 1-D array with at least 3 positive values")

    positive_fields = (
        inputs.permeability_md,
        inputs.porosity,
        inputs.total_compressibility_psi_inv,
        inputs.viscosity_cp,
        inputs.formation_volume_factor_rb_stb,
        inputs.thickness_ft,
        inputs.wellbore_radius_ft,
        inputs.rate_stb_day,
        inputs.initial_pressure_psi,
    )
    if any(value <= 0 for value in positive_fields):
        raise ValueError("All reservoir and well inputs must be positive")

    td = dimensionless_time(time, inputs)
    pd = invert_pressure_gaver_stehfest(td, n_terms=n_terms)
    pressure_scale = (
        141.2
        * inputs.rate_stb_day
        * inputs.formation_volume_factor_rb_stb
        * inputs.viscosity_cp
        / (inputs.permeability_md * inputs.thickness_ft)
    )
    delta_pressure = pressure_scale * pd
    flowing_pressure = inputs.initial_pressure_psi - delta_pressure
    derivative = np.gradient(delta_pressure, np.log(time), edge_order=2)

    return {
        "time_hours": time,
        "dimensionless_time": td,
        "dimensionless_pressure": pd,
        "pressure_drop_psi": delta_pressure,
        "flowing_pressure_psi": flowing_pressure,
        "pressure_derivative_psi": derivative,
        "pressure_scale_psi": pressure_scale,
        "radial_derivative_plateau_psi": 0.5 * pressure_scale,
        "dimensionless_storage": dimensionless_wellbore_storage(0.0, inputs),
    }


def late_time_pressure_approximation(dimensionless_time_value: np.ndarray) -> np.ndarray:
    """Late-time infinite-acting radial-flow approximation for verification."""
    td = np.asarray(dimensionless_time_value, dtype=float)
    return 0.5 * (np.log(td) + 0.80907)
