"""
Metrics for comparing two-qubit correlation trajectories (standard vs TDF).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square error between equal-length traces."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("a and b must have the same shape")
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((x - y) ** 2)))


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; NaN if variance is zero."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def dominant_frequency_hz(t: np.ndarray, signal: np.ndarray) -> float:
    """Peak frequency on the positive fftfreq half (detrended mean)."""
    y = np.asarray(signal, dtype=float) - np.mean(signal)
    t_arr = np.asarray(t, dtype=float)
    if t_arr.size < 2:
        return float("nan")
    fft_y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(t_arr), d=float(t_arr[1] - t_arr[0]))
    n_half = len(freq) // 2
    freq_pos = freq[:n_half]
    mag = np.abs(fft_y[:n_half])
    if mag.size == 0:
        return float("nan")
    return float(freq_pos[int(np.argmax(mag))])


def spectral_entropy_nats_fft(t: np.ndarray, signal: np.ndarray) -> float:
    """
    Shannon entropy (nats) of normalized |FFT|² on the positive half, signal mean removed.
    """
    y = np.asarray(signal, dtype=float) - np.mean(signal)
    t_arr = np.asarray(t, dtype=float)
    if t_arr.size < 2:
        return 0.0
    fft_y = np.fft.fft(y)
    n_half = len(t_arr) // 2
    mag = np.abs(fft_y[:n_half]).astype(float)
    p = mag**2
    s = np.sum(p)
    if s <= 0:
        return 0.0
    p = p / s
    return float(-np.sum(p * np.log(p + 1e-30)))


def summarize_cxx_comparison(
    t: np.ndarray,
    cxx_std: np.ndarray,
    cxx_tdf: np.ndarray,
) -> dict[str, Any]:
    """
    RMSE and overlap statistics between standard and TDF ``C_xx`` traces, plus spectral
    descriptors of the **TDF** ``C_xx`` (change keys if you prefer standard).
    """
    return {
        "rmse_cxx_std_vs_tdf": rmse(cxx_std, cxx_tdf),
        "pearson_r_cxx": pearson_r(cxx_std, cxx_tdf),
        "spectral_entropy_cxx_tdf_nats": spectral_entropy_nats_fft(t, cxx_tdf),
        "dominant_frequency_cxx_tdf": dominant_frequency_hz(t, cxx_tdf),
    }
