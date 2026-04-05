"""
Standard and custom noise models for open-system QuTiP simulations.

Wraps Lindblad / collapse operators and project-specific noise channels.

Also provides **classical** SDE paths (Wiener, Ornstein–Uhlenbeck) used by ensemble
phase-coherence experiments (no QuTiP required for those paths).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import qutip as qt
from numpy.random import Generator


def standard_lindblad_ops(**kwargs: Any) -> list[qt.Qobj]:
    """
    Return collapse operators for a standard noise model (e.g. amplitude / phase damping).

    Parameters
    ----------
    **kwargs
        Rates and system size (stub).

    Returns
    -------
    list of qutip.Qobj
        Collapse operators (stub).
    """
    raise NotImplementedError


def custom_noise_ops(**kwargs: Any) -> list[qt.Qobj]:
    """
    Return collapse operators for the project's custom noise model.

    Parameters
    ----------
    **kwargs
        Custom rate tensors or correlation structure (stub).

    Returns
    -------
    list of qutip.Qobj
        Collapse operators (stub).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Classical paths (phase SDEs, ensemble observables)
# ---------------------------------------------------------------------------


def wiener_path(rng: Generator, n_times: int, dt: float) -> np.ndarray:
    """
    Standard scalar Wiener process on a uniform grid: ``W(0)=0``, i.i.d. increments
    ``N(0, dt)``.
    """
    w = np.zeros(n_times, dtype=float)
    if n_times < 2:
        return w
    dW = rng.normal(0.0, np.sqrt(dt), size=n_times - 1)
    w[1:] = np.cumsum(dW)
    return w


def ou_path(
    rng: Generator,
    n_times: int,
    dt: float,
    gamma: float,
    sigma: float,
    phi0: float = 0.0,
) -> np.ndarray:
    """
    Scalar Ornstein–Uhlenbeck (Itô): ``dφ = -γ φ dt + σ dW``, ``γ > 0``.

    Uses the exact discrete transition for one time step ``dt``:

    ``φ_k = e^{-γ dt} φ_{k-1} + ξ_k``, with
    ``ξ_k ~ N(0, σ²/(2γ) · (1 - e^{-2γ dt}))``.
    """
    if gamma <= 0.0:
        raise ValueError("ou_path requires gamma > 0")
    phi = np.zeros(n_times, dtype=float)
    phi[0] = float(phi0)
    if n_times < 2:
        return phi
    exp_m = float(np.exp(-gamma * dt))
    var_inc = (float(sigma) ** 2) / (2.0 * gamma) * (1.0 - np.exp(-2.0 * gamma * dt))
    std_inc = float(np.sqrt(max(var_inc, 0.0)))
    for k in range(1, n_times):
        phi[k] = phi[k - 1] * exp_m + rng.normal(0.0, std_inc)
    return phi
