"""
TDF τ-field definitions.

τ is the **phase field** accumulated along a trajectory; it enters the quantum phase as

    ψ ∝ exp(i τ),

so local dynamics and interference patterns are controlled by how τ(t) is chosen
(linear drift, oscillatory modulation, noise, etc.).
"""

from __future__ import annotations

import numpy as np


def _as_tau_array(t: np.ndarray | float | list[float]) -> np.ndarray:
    """Coerce time samples to a float ndarray for element-wise τ formulas."""
    return np.asarray(t, dtype=float)


def linear_tau(t: np.ndarray | float | list[float], omega: float) -> np.ndarray:
    """
    Linear phase accumulation τ(t) = ω t.

    Physical meaning: uniform rotation in the complex phase at rate ω, so
    ψ ∝ exp(i ω t) up to other factors.

    Parameters
    ----------
    t
        Time samples (any shape); returned array matches ``t``'s shape.
    omega
        Phase rate (rad / time).

    Returns
    -------
    ndarray
        τ values with the same shape as ``t``.
    """
    t_arr = _as_tau_array(t)
    return omega * t_arr


def oscillatory_tau(
    t: np.ndarray | float | list[float],
    omega: float,
    amplitude: float,
) -> np.ndarray:
    """
    Drift plus sinusoidal phase modulation: τ(t) = ω t + A sin(ω t).

    Physical meaning: the carrier phase ω t is **wobbled** by an internal oscillation
    at the same rate in the sine term, producing a structured phase for ψ ∝ exp(i τ)
    (e.g. nonlinear Ramsey fringes or effective multi-path phase).

    Parameters
    ----------
    t
        Time samples; output shape matches ``t``.
    omega
        Drift rate and argument of the modulation (rad / time).
    amplitude
        Strength A of the sinusoidal correction (radians).

    Returns
    -------
    ndarray
        τ values with the same shape as ``t``.
    """
    t_arr = _as_tau_array(t)
    return omega * t_arr + amplitude * np.sin(omega * t_arr)


def stochastic_tau(
    t: np.ndarray | float | list[float],
    omega: float,
    noise_strength: float,
) -> np.ndarray:
    """
    Linear drift plus additive Gaussian phase noise: τ(t) = ω t + ξ(t).

    Physical meaning: models **dephasing** or fluctuating path length in the phase
    field; ψ ∝ exp(i τ) then describes a random walk of phase across the ensemble.

    Parameters
    ----------
    t
        Time samples; output shape matches ``t``.
    omega
        Mean phase rate (rad / time).
    noise_strength
        Standard deviation of Gaussian noise ξ at each sample (radians).

    Returns
    -------
    ndarray
        τ values with the same shape as ``t``. A new noise realization each call.

    Notes
    -----
    Uses `numpy.random.normal`; set `numpy.random.seed` (or use your own RNG) for
    reproducible runs.
    """
    t_arr = _as_tau_array(t)
    noise = np.random.normal(0.0, noise_strength, size=t_arr.shape)
    return omega * t_arr + noise


def structured_tau(
    t: np.ndarray | float | list[float],
    omega: float,
    freq: float,
) -> np.ndarray:
    """
    Drift plus oscillation at independent frequency: τ(t) = ω t + sin(ν t).

    Physical meaning: separates **slow** linear framing (ω) from a **fast** structured
    modulation at ν, giving rich interference when ψ ∝ exp(i τ) is compared across
    paths or times.

    Parameters
    ----------
    t
        Time samples; output shape matches ``t``.
    omega
        Linear phase rate (rad / time).
    freq
        Modulation frequency ν (rad / time).

    Returns
    -------
    ndarray
        τ values with the same shape as ``t``.
    """
    t_arr = _as_tau_array(t)
    return omega * t_arr + np.sin(freq * t_arr)


def structured_stochastic_tau(
    t: np.ndarray | float | list[float],
    omega: float = 1.0,
    freq: float = 3.0,
    noise_strength: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Combined τ model: τ(t) = ω t + sin(ν t) + ξ(t).

    Structured part (drift + sinusoid) shapes spectrum and interference; additive
    Gaussian ξ(t) models stochastic dephasing in the phase field.

    Parameters
    ----------
    t
        Time samples; output shape matches ``t``.
    omega
        Linear phase rate (rad / time).
    freq
        Structure frequency ν in sin(ν t) (rad / time).
    noise_strength
        Standard deviation of Gaussian noise ξ at each sample (radians).
    seed
        If given, ``numpy.random.seed(seed)`` is called before drawing ξ.

    Returns
    -------
    ndarray
        τ values with the same shape as ``t``. A new noise realization each call
        unless ``seed`` fixes the RNG state.
    """
    t_arr = _as_tau_array(t)
    if seed is not None:
        np.random.seed(seed)
    structured = omega * t_arr + np.sin(freq * t_arr)
    noise = np.random.normal(0.0, noise_strength, size=t_arr.shape)
    return structured + noise


def correlated_stochastic_tau(
    t: np.ndarray | float | list[float],
    omega: float = 1.0,
    freq: float = 3.0,
    noise_strength: float = 0.5,
    correlation_time: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    τ(t) = ω t + sin(ν t) + Ornstein–Uhlenbeck noise.

    The additive noise follows a mean-reverting OU process (temporal correlation /
    memory). Discrete update with step ``Δt_i = t_i - t_{i-1}``:

        ξ_i = ξ_{i-1} - (ξ_{i-1} / τ_c) Δt_i + σ √(Δt_i) η_i,

    with η_i ~ 𝒩(0,1), ``τ_c`` = ``correlation_time``, and ``σ`` = ``noise_strength``.

    Parameters
    ----------
    t
        Strictly increasing time samples; output shape matches ``t``.
    omega
        Linear phase rate (rad / time).
    freq
        Structure frequency ν in sin(ν t) (rad / time).
    noise_strength
        Diffusion scale σ for the OU increment (radians · time^{-1/2} in this scheme).
    correlation_time
        Relaxation time τ_c > 0; larger values give slower decay of correlation.
    seed
        If given, ``numpy.random.seed(seed)`` is called before drawing noise.

    Returns
    -------
    ndarray
        τ values with the same shape as ``t``. ξ(0) = 0.
    """
    t_arr = _as_tau_array(t)
    if correlation_time <= 0.0:
        raise ValueError("correlation_time must be positive")
    if seed is not None:
        np.random.seed(seed)

    structured = omega * t_arr + np.sin(freq * t_arr)
    noise = np.zeros_like(t_arr, dtype=float)
    n = int(t_arr.size)
    if n <= 1:
        return structured + noise

    flat = t_arr.ravel()
    n_f = flat.size
    noise_flat = np.zeros(n_f, dtype=float)
    for i in range(1, n_f):
        dt_i = float(flat[i] - flat[i - 1])
        if dt_i <= 0.0:
            raise ValueError("time samples must be strictly increasing")
        noise_flat[i] = (
            noise_flat[i - 1]
            - (noise_flat[i - 1] / correlation_time) * dt_i
            + noise_strength * np.sqrt(dt_i) * np.random.standard_normal()
        )
    noise = noise_flat.reshape(t_arr.shape)
    return structured + noise


def multi_scale_tau(t: np.ndarray | float | list[float], omega: float) -> np.ndarray:
    """
    Multi-scale phase: τ(t) = ω t + ½ sin(2t) + 0.2 sin(5t).

    Physical meaning: superposes several temporal scales in the phase field so that
    ψ ∝ exp(i τ) encodes **beating** and non-monotonic phase curvature (useful for
    probing sensitivity to multiple clock or environment modes).

    Parameters
    ----------
    t
        Time samples; output shape matches ``t``.
    omega
        Overall linear phase rate (rad / time).

    Returns
    -------
    ndarray
        τ values with the same shape as ``t``.
    """
    t_arr = _as_tau_array(t)
    return omega * t_arr + 0.5 * np.sin(2.0 * t_arr) + 0.2 * np.sin(5.0 * t_arr)


# Non-Gaussian τ increment paths (Student-t, skew-normal, bimodal) live in
# :mod:`core.tau_non_gaussian` for falsification experiments; import that module directly.
