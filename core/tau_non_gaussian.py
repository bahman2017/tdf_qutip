"""
Non-Gaussian cumulative **τ** paths for ensemble phase-noise experiments.

Used by falsification tests; keeps :mod:`core.tau_model` free of heavy rewrites.
All paths use ``τ(t) = ω t + σ · (\\mathrm{cumulative\\ increments})`` with i.i.d. increments
chosen so the **small-time** variance of increments matches a Gaussian Wiener with the
same ``σ`` (where applicable), for fair comparison to Gaussian τ.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy import stats

from core.tau_model import linear_tau


def cumulative_path_gaussian_wiener(
    rng: Generator,
    t: np.ndarray,
    omega: float,
    sigma: float,
) -> np.ndarray:
    """``ω t + σ W(t)`` with standard Wiener ``W`` (same scaling as ``noise_models.wiener_path``)."""
    n = int(t.size)
    if n < 2:
        return linear_tau(t, omega)
    dt = float(t[1] - t[0])
    dW = rng.normal(0.0, np.sqrt(dt), size=n - 1)
    w = np.concatenate([[0.0], np.cumsum(dW)])
    return linear_tau(t, omega) + sigma * w


def cumulative_path_student_t(
    rng: Generator,
    t: np.ndarray,
    omega: float,
    sigma: float,
    df: float,
) -> np.ndarray:
    """
    Cumulative heavy-tailed path: increments ``∝ t_{df} · √(dt)`` scaled to match
    Gaussian increment variance ``σ² dt`` for ``df > 2``.
    """
    if df <= 2.0:
        raise ValueError("student-t path requires df > 2 for finite variance")
    n = int(t.size)
    if n < 2:
        return linear_tau(t, omega)
    dt = float(t[1] - t[0])
    # Var(t_df) = df/(df-2); match to 1 for N(0,1): scale std of t draw
    std_t = float(np.sqrt(df / (df - 2.0)))
    inc = rng.standard_t(df, size=n - 1) / std_t * sigma * np.sqrt(dt)
    s = np.concatenate([[0.0], np.cumsum(inc)])
    return linear_tau(t, omega) + s


def cumulative_path_skew_normal(
    rng: Generator,
    t: np.ndarray,
    omega: float,
    sigma: float,
    alpha: float,
) -> np.ndarray:
    """
    Skewed increments via SciPy ``skewnorm``: per step, draw ``ξ`` with variance matched
    to ``σ² dt`` (scale chosen from ``alpha`` and ``σ``).
    """
    n = int(t.size)
    if n < 2:
        return linear_tau(t, omega)
    dt = float(t[1] - t[0])
    # skewnorm: Var depends on alpha and scale; set scale so Var = sigma^2 * dt
    a = float(alpha)
    var_sn = float(stats.skewnorm.var(a, loc=0.0, scale=1.0))
    if var_sn < 1e-18:
        var_sn = 1.0
    sc = float(sigma * np.sqrt(dt) / np.sqrt(var_sn))
    inc = stats.skewnorm.rvs(a, loc=0.0, scale=sc, size=n - 1, random_state=rng)
    s = np.concatenate([[0.0], np.cumsum(inc)])
    return linear_tau(t, omega) + s


def cumulative_path_bimodal_gaussian(
    rng: Generator,
    t: np.ndarray,
    omega: float,
    sigma: float,
    *,
    p: float = 0.5,
    delta: float = 0.6,
) -> np.ndarray:
    """
    Bimodal **increment** mixture: ``p · N(+m, s²) + (1-p) · N(-m, s²)`` with mean ``(2p-1)m``.
    For ``p = 0.5``, mean zero and ``Var = m² + s² = σ² dt`` with ``m = δ √(σ² dt)`` (clamped so ``s² ≥ 0``).
    """
    n = int(t.size)
    if n < 2:
        return linear_tau(t, omega)
    dt = float(t[1] - t[0])
    target_var = (sigma**2) * dt
    p = float(np.clip(p, 0.05, 0.95))
    m = float(delta) * np.sqrt(target_var)
    s2 = max(target_var - m * m, 1e-12)
    s = float(np.sqrt(s2))
    u = rng.random(n - 1)
    inc = np.where(u < p, rng.normal(m, s, size=n - 1), rng.normal(-m, s, size=n - 1))
    path = np.concatenate([[0.0], np.cumsum(inc)])
    return linear_tau(t, omega) + path
