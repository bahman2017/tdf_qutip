"""
Ramsey-channel spectral metrics for bench τ models (⟨σ_x⟩ FFT).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from core.tau_model import (
    correlated_stochastic_tau,
    linear_tau,
    multi_scale_tau,
    oscillatory_tau,
    structured_stochastic_tau,
    structured_tau,
)
from experiments.ramsey import run_tdf_with_tau


def phase_magnitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """√(⟨σ_x⟩² + ⟨σ_y⟩²)."""
    sx = np.asarray(x, dtype=float)
    sy = np.asarray(y, dtype=float)
    return np.sqrt(sx**2 + sy**2)


def tau_models_specs() -> list[tuple[str, Callable[..., np.ndarray], dict[str, Any]]]:
    """Default parameters for spectral / discrimination benches."""
    return [
        ("linear_tau", linear_tau, {"omega": 1.0}),
        ("oscillatory_tau", oscillatory_tau, {"omega": 1.0, "amplitude": 0.5}),
        ("structured_tau", structured_tau, {"omega": 1.0, "freq": 3.0}),
        ("multi_scale_tau", multi_scale_tau, {"omega": 1.0}),
    ]


def tau_models_specs_v2() -> list[tuple[str, Callable[..., np.ndarray], dict[str, Any]]]:
    """V2 bench: adds structured + i.i.d. / OU stochastic τ with fixed seeds for reproducible runs."""
    return tau_models_specs() + [
        (
            "structured_stochastic_tau",
            structured_stochastic_tau,
            {"omega": 1.0, "freq": 3.0, "noise_strength": 0.5, "seed": 4242},
        ),
        (
            "correlated_stochastic_tau",
            correlated_stochastic_tau,
            {
                "omega": 1.0,
                "freq": 3.0,
                "noise_strength": 0.5,
                "correlation_time": 1.0,
                "seed": 4242,
            },
        ),
    ]


def top_peak_frequencies(
    freq_pos: np.ndarray, mag_half: np.ndarray, k: int = 3
) -> np.ndarray:
    mag_half = np.asarray(mag_half, dtype=float)
    n = len(mag_half)
    if n == 0:
        return np.array([])
    k = min(k, n)
    idx = np.argpartition(mag_half, -k)[-k:]
    idx = idx[np.argsort(mag_half[idx])][::-1]
    return np.asarray(freq_pos[idx], dtype=float)


def spectral_entropy_nats(mag_half: np.ndarray) -> float:
    p = np.asarray(mag_half, dtype=float) ** 2
    s = np.sum(p)
    if s <= 0:
        return 0.0
    p = p / s
    return float(-np.sum(p * np.log(p + 1e-30)))


def bandwidth_90_percent_power(freq_pos: np.ndarray, mag_half: np.ndarray) -> float:
    power = (np.asarray(mag_half, dtype=float) ** 2)
    target = 0.9 * np.sum(power)
    n = len(power)
    if target <= 0 or n == 0:
        return float("nan")
    best = np.inf
    for i in range(n):
        s = 0.0
        for j in range(i, n):
            s += power[j]
            if s >= target:
                w = float(freq_pos[j] - freq_pos[i])
                if w < best:
                    best = w
                break
    return float(best) if np.isfinite(best) else float("nan")


def analyze_tau_models(
    t: np.ndarray,
    models: list[tuple[str, Callable[..., np.ndarray], dict[str, Any]]] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Evolve each τ model with :func:`run_tdf_with_tau`, FFT ⟨σ_x⟩, positive-half metrics.

    Parameters
    ----------
    models
        Bench list ``(name, tau_fn, kwargs)``. Defaults to :func:`tau_models_specs`.

    Returns
    -------
    freq_pos
        Positive-frequency axis.
    rows
        Per-model dicts with ``name``, ``mag_half``, ``dominant_freq``, ``spectral_entropy``,
        ``bandwidth_90``, ``phase_mag_mean``, peaks, etc.
    """
    if models is None:
        models = tau_models_specs()
    freq = np.fft.fftfreq(len(t), d=(float(t[1]) - float(t[0])))
    n_half = len(freq) // 2
    freq_pos = np.asarray(freq[:n_half], dtype=float)

    rows: list[dict] = []
    for name, tau_fn, kw in models:
        tau_field = tau_fn(t, **kw)
        result = run_tdf_with_tau(t, tau_field)
        sig_x = np.asarray(result.expect[0], dtype=float)
        sig_y = np.asarray(result.expect[1], dtype=float)
        fft_x = np.fft.fft(sig_x)
        mag_half = np.abs(fft_x)[:n_half].astype(float)

        dominant = float(freq_pos[np.argmax(mag_half)])
        top3 = top_peak_frequencies(freq_pos, mag_half, k=3)
        while top3.size < 3:
            top3 = np.append(top3, np.nan)
        ent = spectral_entropy_nats(mag_half)
        bw = bandwidth_90_percent_power(freq_pos, mag_half)
        pm_mean = float(np.mean(phase_magnitude(sig_x, sig_y)))

        rows.append(
            {
                "name": name,
                "mag_half": mag_half,
                "dominant_freq": dominant,
                "peak1": float(top3[0]),
                "peak2": float(top3[1]),
                "peak3": float(top3[2]),
                "spectral_entropy": ent,
                "bandwidth_90": bw,
                "phase_mag_mean": pm_mean,
            }
        )

    return freq_pos, rows
