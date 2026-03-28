"""
Δτ interference test.

Compares the direct phase signal cos(Δτ) with |⟨ψ_A(t)|ψ_B(t)⟩| from evolving the
same initial state under H_A and H_B built from τ_A and τ_B (TDF Hamiltonians).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from analysis.plots import plot_timeseries
from core.evolution import create_superposition_state, run_evolution
from core.hamiltonians import tau_to_hamiltonian
from core.tau_model import linear_tau, oscillatory_tau, structured_tau


def _quantum_overlap_amplitude(psi_a: qt.Qobj, psi_b: qt.Qobj) -> float:
    """|⟨ψ_a|ψ_b⟩| from ψ_a† ψ_b (scalar or 1×1 Qobj depending on QuTiP version)."""
    q = psi_a.dag() * psi_b
    if isinstance(q, qt.Qobj):
        val = complex(q.full().flat[0])
    else:
        val = complex(q)
    return float(abs(val))


def remove_mean(signal: np.ndarray | list[float]) -> np.ndarray:
    """Return ``signal - mean(signal)`` (oscillatory part; kills DC before FFT)."""
    x = np.asarray(signal, dtype=float)
    return x - np.mean(x)


def dominant_positive_frequency(
    t: np.ndarray,
    signal: np.ndarray,
    *,
    detrend: bool = True,
) -> float:
    """
    Dominant frequency on the positive ``fftfreq`` half.

    If ``detrend`` is True, subtract the mean before the FFT so the peak reflects
    oscillatory structure rather than DC offset.
    """
    y = np.asarray(signal, dtype=float)
    if detrend:
        y = remove_mean(y)
    t_arr = np.asarray(t, dtype=float)
    fft_y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(t_arr), d=float(t_arr[1] - t_arr[0]))
    n_half = len(freq) // 2
    freq_pos = freq[:n_half]
    mag = np.abs(fft_y[:n_half])
    return float(freq_pos[np.argmax(mag)])


def _positive_half_fft_mag(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frequency grid and |FFT| on positive half for plotting."""
    y = np.asarray(y, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    fft_y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(t_arr), d=float(t_arr[1] - t_arr[0]))
    n_half = len(freq) // 2
    return freq[:n_half], np.abs(fft_y[:n_half])


def _overlap_correlation(direct: np.ndarray, quantum: np.ndarray) -> float:
    """Pearson r between direct and quantum interference traces; NaN if degenerate."""
    a = np.asarray(direct, dtype=float).ravel()
    b = np.asarray(quantum, dtype=float).ravel()
    if a.size < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze_interference_from_tau_fields(
    t: np.ndarray,
    tau_A: np.ndarray,
    tau_B: np.ndarray,
    *,
    plot: bool = False,
    show: bool = False,
    plot_detrend_diagnostics: bool = True,
    tau_A_label: str = r"$\tau_A$",
    tau_B_label: str = r"$\tau_B$",
) -> dict[str, Any]:
    """
    Core physics: cos(Δτ) vs quantum overlap, metrics, optional τ overview + comparison plots.

    Parameters
    ----------
    plot_detrend_diagnostics
        When ``plot`` is True, add detrended time-domain comparison and detrended FFT plot.
    """
    tau_A = np.asarray(tau_A, dtype=float)
    tau_B = np.asarray(tau_B, dtype=float)
    delta_tau = tau_A - tau_B
    direct_interference = np.cos(delta_tau)
    visibility = float(np.max(direct_interference) - np.min(direct_interference))

    H_A = tau_to_hamiltonian(tau_A, t)
    H_B = tau_to_hamiltonian(tau_B, t)
    psi0 = create_superposition_state()

    result_A = run_evolution(H_A, psi0, t)
    result_B = run_evolution(H_B, psi0, t)

    states_a = result_A.states
    states_b = result_B.states
    n = len(t)
    if len(states_a) != n or len(states_b) != n:
        raise RuntimeError("sesolve state count must match time grid length")

    quantum_interference = np.empty(n, dtype=float)
    for i in range(n):
        quantum_interference[i] = _quantum_overlap_amplitude(states_a[i], states_b[i])

    diff = direct_interference - quantum_interference
    mean_abs_error = float(np.mean(np.abs(diff)))
    max_abs_error = float(np.max(np.abs(diff)))

    dominant_freq_direct_raw = dominant_positive_frequency(
        t, direct_interference, detrend=False
    )
    dominant_freq_quantum_raw = dominant_positive_frequency(
        t, quantum_interference, detrend=False
    )
    dominant_freq_direct_detrended = dominant_positive_frequency(
        t, direct_interference, detrend=True
    )
    dominant_freq_quantum_detrended = dominant_positive_frequency(
        t, quantum_interference, detrend=True
    )

    overlap_correlation = _overlap_correlation(direct_interference, quantum_interference)

    direct_det = remove_mean(direct_interference)
    quantum_det = remove_mean(quantum_interference)

    out: dict[str, Any] = {
        "t": t,
        "tau_A": tau_A,
        "tau_B": tau_B,
        "delta_tau": delta_tau,
        "direct_interference": direct_interference,
        "quantum_interference": quantum_interference,
        "direct_interference_detrended": direct_det,
        "quantum_interference_detrended": quantum_det,
        "interference": direct_interference,
        "visibility": visibility,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "dominant_freq_direct_raw": dominant_freq_direct_raw,
        "dominant_freq_quantum_raw": dominant_freq_quantum_raw,
        "dominant_freq_direct_detrended": dominant_freq_direct_detrended,
        "dominant_freq_quantum_detrended": dominant_freq_quantum_detrended,
        "dominant_freq_direct": dominant_freq_direct_raw,
        "dominant_freq_quantum": dominant_freq_quantum_raw,
        "overlap_correlation": overlap_correlation,
        "H_A": H_A,
        "H_B": H_B,
        "result_A": result_A,
        "result_B": result_B,
    }

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
        ax = axes.ravel()

        ax[0].plot(t, tau_A, color="C0")
        ax[0].set_ylabel(r"$\tau_A$")
        ax[0].set_title(tau_A_label)
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(t, tau_B, color="C1")
        ax[1].set_ylabel(r"$\tau_B$")
        ax[1].set_title(tau_B_label)
        ax[1].grid(True, alpha=0.3)

        ax[2].plot(t, delta_tau, color="C2")
        ax[2].set_ylabel(r"$\Delta\tau$")
        ax[2].set_title(r"$\Delta\tau = \tau_A - \tau_B$")
        ax[2].grid(True, alpha=0.3)

        ax[3].plot(t, direct_interference, color="C3", label=r"$\cos(\Delta\tau)$")
        ax[3].plot(t, quantum_interference, color="C4", alpha=0.85, label="overlap")
        ax[3].set_ylabel("signal")
        ax[3].set_xlabel("t")
        ax[3].set_title(f"Direct vs quantum (vis. cos Δτ = {visibility:.4f})")
        ax[3].legend(loc="upper right", fontsize=8)
        ax[3].grid(True, alpha=0.3)

        fig.suptitle("Δτ interference")
        fig.tight_layout()

        plot_timeseries(
            t,
            [direct_interference, quantum_interference],
            labels=[r"$\cos(\Delta\tau)$", "Quantum overlap"],
            title="TDF Interference vs Quantum Evolution",
            ylabel="Amplitude",
        )

        if plot_detrend_diagnostics:
            plot_timeseries(
                t,
                [direct_det, quantum_det],
                labels=[r"$\cos(\Delta\tau) - \langle\cdot\rangle$", "overlap − ⟨·⟩"],
                title="Detrended interference (DC removed)",
                ylabel="Amplitude",
            )

            fq, mag_d = _positive_half_fft_mag(t, direct_det)
            _, mag_q = _positive_half_fft_mag(t, quantum_det)
            fig_fft, axf = plt.subplots(figsize=(8, 4))
            axf.plot(fq, mag_d, label=r"FFT|$|$ detrended $\cos\Delta\tau$")
            axf.plot(fq, mag_q, label=r"FFT|$|$ detrended overlap", alpha=0.85)
            axf.set_xlabel("Frequency")
            axf.set_ylabel(r"|FFT| (detrended)")
            axf.set_title("Spectrum after mean removal (oscillatory structure)")
            axf.legend()
            axf.grid(True, alpha=0.3)
            fig_fft.tight_layout()
            out["figure_fft_detrended"] = fig_fft

        out["figure"] = fig
        out["axes"] = axes

        print("Mean |cos(Δτ) − quantum overlap|:", mean_abs_error)
        print("Overlap correlation r:", overlap_correlation)

        if show:
            plt.show()

    return out


def run_interference_experiment(
    t: np.ndarray | None = None,
    *,
    show: bool = False,
) -> dict[str, Any]:
    """
    Default demo: linear τ_A vs structured τ_B (freq=3), with plots.

    See :func:`analyze_interference_from_tau_fields` for the shared pipeline.
    """
    if t is None:
        t = np.linspace(0, 10, 300)

    tau_A = linear_tau(t, omega=1.0)
    tau_B = structured_tau(t, omega=1.0, freq=3.0)
    return analyze_interference_from_tau_fields(
        t,
        tau_A,
        tau_B,
        plot=True,
        show=show,
        tau_A_label=r"$\tau_A$ (linear)",
        tau_B_label=r"$\tau_B$ (structured)",
    )


FREQ_SWEEP_VALUES = (0.5, 1.0, 2.0, 3.0, 5.0)
AMPLITUDE_SWEEP_VALUES = (0.1, 0.5, 1.0, 2.0)

SWEEP_CSV_FIELDS = [
    "sweep",
    "structured_freq",
    "oscillatory_amplitude",
    "mean_abs_error",
    "max_abs_error",
    "visibility",
    "dominant_freq_direct_raw",
    "dominant_freq_quantum_raw",
    "dominant_freq_direct_detrended",
    "dominant_freq_quantum_detrended",
    "overlap_correlation",
]


def _sweep_row_from_result(res: dict[str, Any], **sweep_cols: Any) -> dict[str, Any]:
    return {
        **sweep_cols,
        "mean_abs_error": res["mean_abs_error"],
        "max_abs_error": res["max_abs_error"],
        "visibility": res["visibility"],
        "dominant_freq_direct_raw": res["dominant_freq_direct_raw"],
        "dominant_freq_quantum_raw": res["dominant_freq_quantum_raw"],
        "dominant_freq_direct_detrended": res["dominant_freq_direct_detrended"],
        "dominant_freq_quantum_detrended": res["dominant_freq_quantum_detrended"],
        "overlap_correlation": res["overlap_correlation"],
    }


def run_interference_parameter_sweep(
    t: np.ndarray | None = None,
    *,
    output_dir: Path | None = None,
    plot: bool = True,
    show: bool = False,
) -> list[dict[str, Any]]:
    """
    Sweep structured ``freq`` (τ_B) and oscillatory ``amplitude`` (τ_A); log metrics to CSV.

    Writes ``interference_sweep.csv`` with raw/detrended dominant frequencies and
    ``overlap_correlation``.
    """
    if t is None:
        t = np.linspace(0, 10, 300)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "interference_sweep.csv"

    rows: list[dict[str, Any]] = []

    for freq in FREQ_SWEEP_VALUES:
        tau_A = linear_tau(t, omega=1.0)
        tau_B = structured_tau(t, omega=1.0, freq=freq)
        res = analyze_interference_from_tau_fields(t, tau_A, tau_B, plot=False)
        rows.append(
            _sweep_row_from_result(
                res,
                sweep="structured_freq",
                structured_freq=freq,
                oscillatory_amplitude="",
            )
        )

    for amplitude in AMPLITUDE_SWEEP_VALUES:
        tau_A = oscillatory_tau(t, omega=1.0, amplitude=amplitude)
        tau_B = structured_tau(t, omega=1.0, freq=3.0)
        res = analyze_interference_from_tau_fields(t, tau_A, tau_B, plot=False)
        rows.append(
            _sweep_row_from_result(
                res,
                sweep="oscillatory_amplitude",
                structured_freq=3.0,
                oscillatory_amplitude=amplitude,
            )
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote interference sweep ({len(rows)} rows) to {csv_path}")

    if plot:
        freq_rows = [r for r in rows if r["sweep"] == "structured_freq"]
        amp_rows = [r for r in rows if r["sweep"] == "oscillatory_amplitude"]

        fx = [float(r["structured_freq"]) for r in freq_rows]
        fy = [r["mean_abs_error"] for r in freq_rows]
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(fx, fy, marker="o", color="C0")
        ax1.set_xlabel("structured_tau frequency ν")
        ax1.set_ylabel("mean |cos(Δτ) − overlap|")
        ax1.set_title("TDF interference error vs structured frequency")
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        p_freq = output_dir / "interference_sweep_mean_error_vs_freq.png"
        fig1.savefig(p_freq, dpi=150)

        ax_vals = [float(r["oscillatory_amplitude"]) for r in amp_rows]
        ay = [r["mean_abs_error"] for r in amp_rows]
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(ax_vals, ay, marker="o", color="C1")
        ax2.set_xlabel("oscillatory_tau amplitude A")
        ax2.set_ylabel("mean |cos(Δτ) − overlap|")
        ax2.set_title("TDF interference error vs oscillatory amplitude")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        p_amp = output_dir / "interference_sweep_mean_error_vs_amplitude.png"
        fig2.savefig(p_amp, dpi=150)

        print(f"Saved {p_freq}")
        print(f"Saved {p_amp}")

        if show:
            plt.show()

    return rows
