"""
Parameter-sweep predictivity checks for τ models (spectrum vs interference vs ensemble decoherence).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from analysis.tau_model_spectrum import spectral_entropy_nats
from core.evolution import create_superposition_state, run_evolution
from core.hamiltonians import tau_to_hamiltonian
from core.tau_model import correlated_stochastic_tau, linear_tau
from experiments.decoherence import fit_best_gamma, lindblad_coherence_curve
from experiments.interference import analyze_interference_from_tau_fields
from experiments.ramsey import run_tdf_with_tau

CORRELATION_TIMES_DEFAULT = (0.2, 0.5, 1.0, 2.0, 5.0)
TAU_SEED = 4242
N_ENSEMBLE_DEFAULT = 40
ENSEMBLE_BASE_SEED = 42_424
FIT_FRACTION = 0.2


def _ensemble_mixed_transverse_coherence(
    states_per_time: list[list[qt.Qobj]],
) -> np.ndarray:
    """ρ(t) = (1/N) Σ_i |ψ_i⟩⟨ψ_i|, then C(t) = √(⟨σ_x⟩_ρ² + ⟨σ_y⟩_ρ²)."""
    n_times = len(states_per_time)
    out = np.zeros(n_times, dtype=float)
    sx_op = qt.sigmax()
    sy_op = qt.sigmay()
    for k in range(n_times):
        kets = states_per_time[k]
        rho = kets[0] * kets[0].dag()
        for psi in kets[1:]:
            rho += psi * psi.dag()
        rho /= len(kets)
        sx = float(np.real(qt.expect(sx_op, rho)))
        sy = float(np.real(qt.expect(sy_op, rho)))
        out[k] = float(np.sqrt(sx**2 + sy**2))
    return out


def _spectrum_dominant_entropy(
    t: np.ndarray, tau_field: np.ndarray
) -> tuple[float, float]:
    """Dominant positive frequency and spectral entropy (nats) from FFT of ⟨σ_x⟩."""
    result = run_tdf_with_tau(t, tau_field)
    sig_x = np.asarray(result.expect[0], dtype=float)
    freq = np.fft.fftfreq(len(t), d=(float(t[1]) - float(t[0])))
    n_half = len(freq) // 2
    freq_pos = np.asarray(freq[:n_half], dtype=float)
    mag_half = np.abs(np.fft.fft(sig_x))[:n_half].astype(float)
    dominant = float(freq_pos[np.argmax(mag_half)])
    ent = spectral_entropy_nats(mag_half)
    return dominant, ent


def _correlated_ensemble_coherence(
    t: np.ndarray,
    omega: float,
    freq: float,
    noise_strength: float,
    correlation_time: float,
    n_ensemble: int,
    ensemble_base_seed: int,
) -> np.ndarray:
    """ρ-averaged transverse coherence for independent OU-τ realizations."""
    psi0 = create_superposition_state()
    bucket: list[list[qt.Qobj]] = [[] for _ in range(len(t))]
    for i in range(n_ensemble):
        np.random.seed(ensemble_base_seed + i)
        tau = correlated_stochastic_tau(
            t,
            omega=omega,
            freq=freq,
            noise_strength=noise_strength,
            correlation_time=correlation_time,
        )
        H = tau_to_hamiltonian(tau, t)
        res = run_evolution(H, psi0, t)
        for k, psi in enumerate(res.states):
            bucket[k].append(psi)
    return _ensemble_mixed_transverse_coherence(bucket)


def _decoherence_rmse_and_final(
    t: np.ndarray,
    omega: float,
    c_tdf: np.ndarray,
    *,
    fit_fraction: float = FIT_FRACTION,
    gamma_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Full-curve RMSE vs best-fit Lindblad (early window) and mean tail coherence of C_tdf."""
    if gamma_grid is None:
        gamma_grid = np.linspace(0.001, 1.0, 200)
    psi0 = create_superposition_state()
    n_initial = max(3, int(fit_fraction * len(t)))
    best_gamma, _ = fit_best_gamma(
        t, omega, psi0, c_tdf, n_initial, gamma_grid
    )
    c_lind = lindblad_coherence_curve(t, omega, best_gamma, psi0)
    residual = c_tdf - c_lind
    rmse_full = float(np.sqrt(np.mean(residual**2)))
    final_coh = float(np.mean(c_tdf[-10:]))
    return rmse_full, final_coh


def plot_correlated_tau_predictivity_trends(
    rows: list[dict[str, Any]],
    *,
    output_path: Path | str | None = None,
    show: bool = False,
) -> Path:
    """
    Plot ``correlation_time`` vs spectral entropy, overlap correlation, and Lindblad RMSE.

    Intended for visual checks that sweeps vary smoothly (no jagged jumps at fixed seeds).
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parent.parent
            / "outputs"
            / "correlated_tau_predictivity_trends.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.array([float(r["correlation_time"]) for r in rows], dtype=float)
    s_ent = np.array([float(r["spectral_entropy"]) for r in rows], dtype=float)
    r_ov = np.array([float(r["overlap_correlation"]) for r in rows], dtype=float)
    rmse = np.array([float(r["lindblad_rmse_full"]) for r in rows], dtype=float)

    order = np.argsort(x)
    x = x[order]
    s_ent = s_ent[order]
    r_ov = r_ov[order]
    rmse = rmse[order]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    xc = np.clip(x, 1e-6, None)

    ax0, ax1, ax2 = axes
    ax0.semilogx(xc, s_ent, "o-", color="C0", markersize=7, linewidth=1.5)
    ax0.set_xlabel(r"correlation time $\tau_c$")
    ax0.set_ylabel("spectral entropy (nats)")
    ax0.set_title(r"Spectrum richness vs $\tau_c$")
    ax0.grid(True, alpha=0.3)

    valid_r = np.isfinite(r_ov)
    ax1.semilogx(xc[valid_r], r_ov[valid_r], "o-", color="C1", markersize=7, linewidth=1.5)
    ax1.set_xlabel(r"correlation time $\tau_c$")
    ax1.set_ylabel(r"overlap correlation")
    ax1.set_title(r"Interference vs $\tau_c$")
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(xc, rmse, "o-", color="C2", markersize=7, linewidth=1.5)
    ax2.set_xlabel(r"correlation time $\tau_c$")
    ax2.set_ylabel(r"Lindblad RMSE (full curve)")
    ax2.set_title(r"Decoherence mismatch vs $\tau_c$")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "correlated_stochastic_tau predictivity (fixed seed single-τ spectrum/interference; "
        "ensemble for decoherence)",
        fontsize=10,
    )

    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Wrote {output_path}")
    return output_path


def run_correlated_tau_predictivity_test(
    t: np.ndarray | None = None,
    *,
    correlation_times: tuple[float, ...] = CORRELATION_TIMES_DEFAULT,
    omega: float = 1.0,
    freq: float = 3.0,
    noise_strength: float = 0.5,
    tau_seed: int = TAU_SEED,
    n_ensemble: int = N_ENSEMBLE_DEFAULT,
    ensemble_base_seed: int = ENSEMBLE_BASE_SEED,
    output_csv: Path | str | None = None,
    plot: bool = True,
    output_plot: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Sweep ``correlation_time`` for :func:`correlated_stochastic_tau` and record spectrum,
    interference (vs linear τ_A), and ensemble-ρ decoherence vs fitted Lindblad.

    Single-trajectory metrics use a fixed ``tau_seed`` per run so sweeps are reproducible;
    decoherence uses an independent ensemble with fresh seeds.

    Parameters
    ----------
    t
        Time grid; default ``linspace(0, 10, 300)``.
    output_csv
        If None, writes ``<tdf_qutip>/outputs/correlated_tau_predictivity.csv``.
    plot
        If True, save a 1×3 trend figure (see ``output_plot``).
    output_plot
        If None, writes ``<tdf_qutip>/outputs/correlated_tau_predictivity_trends.png``.

    Returns
    -------
    list of dict
        One row per ``correlation_time`` with scalar metrics.
    """
    if t is None:
        t = np.linspace(0.0, 10.0, 300)
    t = np.asarray(t, dtype=float)

    if output_csv is None:
        output_csv = Path(__file__).resolve().parent.parent / "outputs" / "correlated_tau_predictivity.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []

    for tau_c in correlation_times:
        tau_field = correlated_stochastic_tau(
            t,
            omega=omega,
            freq=freq,
            noise_strength=noise_strength,
            correlation_time=float(tau_c),
            seed=tau_seed,
        )

        dom_f, s_ent = _spectrum_dominant_entropy(t, tau_field)

        tau_a = linear_tau(t, omega)
        inter = analyze_interference_from_tau_fields(
            t,
            tau_a,
            tau_field,
            plot=False,
        )
        mae = float(inter["mean_abs_error"])
        r_ov = inter["overlap_correlation"]
        if isinstance(r_ov, float) and np.isnan(r_ov):
            r_ov_f = float("nan")
        else:
            r_ov_f = float(r_ov)

        c_tdf = _correlated_ensemble_coherence(
            t,
            omega,
            freq,
            noise_strength,
            float(tau_c),
            n_ensemble,
            ensemble_base_seed,
        )
        rmse_l, final_coh = _decoherence_rmse_and_final(t, omega, c_tdf)

        row = {
            "correlation_time": float(tau_c),
            "dominant_frequency": dom_f,
            "spectral_entropy": s_ent,
            "mean_abs_error": mae,
            "overlap_correlation": r_ov_f,
            "final_coherence": final_coh,
            "lindblad_rmse_full": rmse_l,
            "omega": omega,
            "freq": freq,
            "noise_strength": noise_strength,
            "tau_seed": tau_seed,
            "n_ensemble": n_ensemble,
        }
        rows_out.append(row)

    fieldnames = [
        "correlation_time",
        "dominant_frequency",
        "spectral_entropy",
        "mean_abs_error",
        "overlap_correlation",
        "final_coherence",
        "lindblad_rmse_full",
        "omega",
        "freq",
        "noise_strength",
        "tau_seed",
        "n_ensemble",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(f"Wrote {output_csv}")

    if plot:
        plot_correlated_tau_predictivity_trends(rows_out, output_path=output_plot)

    return rows_out
