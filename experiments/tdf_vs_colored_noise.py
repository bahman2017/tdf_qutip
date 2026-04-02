"""
Discrimination: TDF τ-structure vs standard colored noise on the same two-qubit generator.

TDF uses :math:`H = E_\\tau(t)\\,G` with :math:`E=\\dot\\tau` from structured :math:`\\tau(t)`.
Colored-noise baselines use :math:`H = (\\omega + \\xi(t))\\,G` with no τ-phase derivative
structure (OU or pink :math:`\\xi`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt

from analysis.correlation_metrics import (
    dominant_frequency_hz,
    pearson_r,
    spectral_entropy_nats_fft,
)
from analysis.multi_observable_coupling import (
    METRIC_COLUMNS,
    cross_observable_coupling_matrix,
    interpret_tdf_vs_colored_noise,
    joint_coupling_score,
    metrics_dataframe,
    plot_coupling_heatmap,
    plot_metrics_vs_control,
    print_coupling_comparison,
)
from core.hamiltonians import two_qubit_zz_sum_operator, tau_to_two_qubit_tdf_hamiltonian
from core.tau_model import correlated_stochastic_tau
from core.evolution import run_evolution
from experiments.correlation_test import (
    bell_phi_plus,
    chsh_operators,
    correlation_pauli_tensors,
)


def _scalar_coupling_hamiltonian(t: np.ndarray, coeff: np.ndarray) -> list:
    """``H(t) = coeff(t) G`` with ``G = σ_z⊗I + I⊗σ_z``, QuTiP list format."""
    t_flat = np.asarray(t, dtype=float).ravel()
    c_flat = np.asarray(coeff, dtype=float).ravel()
    if t_flat.size != c_flat.size:
        raise ValueError("t and coeff must have the same length")
    order = np.argsort(t_flat, kind="mergesort")
    t_sorted = t_flat[order]
    c_sorted = c_flat[order]
    t_u, inv = np.unique(t_sorted, return_inverse=True)
    counts = np.bincount(inv)
    c_nodes = np.bincount(inv, weights=c_sorted) / counts
    t_nodes = t_u
    t_min, t_max = float(t_nodes[0]), float(t_nodes[-1])

    def drive(time: float, args=None) -> float:
        tt = float(time)
        if tt <= t_min:
            return float(c_nodes[0])
        if tt >= t_max:
            return float(c_nodes[-1])
        return float(np.interp(tt, t_nodes, c_nodes))

    G = two_qubit_zz_sum_operator()
    return [[G, drive]]


def ou_xi_trace(
    t: np.ndarray,
    correlation_time: float,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Zero-mean Ornstein–Uhlenbeck noise ξ(t) (same discrete scheme as ``correlated_stochastic_tau``).
    """
    t_arr = np.asarray(t, dtype=float).ravel()
    if correlation_time <= 0:
        raise ValueError("correlation_time must be positive")
    if seed is not None:
        np.random.seed(int(seed))
    xi = np.zeros_like(t_arr, dtype=float)
    for i in range(1, t_arr.size):
        dt_i = float(t_arr[i] - t_arr[i - 1])
        if dt_i <= 0:
            raise ValueError("t must be strictly increasing")
        xi[i] = (
            xi[i - 1]
            - (xi[i - 1] / correlation_time) * dt_i
            + sigma * np.sqrt(dt_i) * np.random.standard_normal()
        )
    return xi


def pink_xi_trace(
    t: np.ndarray,
    noise_strength: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Approximate 1/f-like real noise via frequency-domain shaping (phenomenological pink drive).
    """
    t_arr = np.asarray(t, dtype=float).ravel()
    n = t_arr.size
    if n < 4:
        return np.zeros_like(t_arr)
    dt = float(np.mean(np.diff(t_arr)))
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, d=dt)
    freqs[0] = freqs[1] if freqs.size > 1 else 1.0
    amp = 1.0 / np.sqrt(np.maximum(np.abs(freqs), 1e-6))
    amp[0] = 0.0
    shaped = spec * amp
    x = np.fft.irfft(shaped, n=n)
    x -= x[0]
    std = float(np.std(x))
    if std < 1e-15:
        return np.zeros_like(t_arr)
    return noise_strength * (x / std)


def _simulate_correlations(
    H: list,
    t: np.ndarray,
    psi0: qt.Qobj,
) -> dict[str, np.ndarray]:
    XX, YY, ZZ = correlation_pauli_tensors()
    A0, A1, B0, B1 = chsh_operators()
    e_ops = [
        XX,
        YY,
        ZZ,
        qt.tensor(A0, B0),
        qt.tensor(A0, B1),
        qt.tensor(A1, B0),
        qt.tensor(A1, B1),
    ]
    res = run_evolution(H, psi0, t, e_ops=e_ops)
    cxx = np.asarray(res.expect[0], dtype=float)
    cyy = np.asarray(res.expect[1], dtype=float)
    e00 = np.asarray(res.expect[3], dtype=float)
    e01 = np.asarray(res.expect[4], dtype=float)
    e10 = np.asarray(res.expect[5], dtype=float)
    e11 = np.asarray(res.expect[6], dtype=float)
    chsh = e00 + e01 + e10 - e11
    return {"cxx": cxx, "cyy": cyy, "chsh": chsh}


def _reference_standard(
    t: np.ndarray,
    omega: float,
    psi0: qt.Qobj,
) -> dict[str, np.ndarray]:
    G = two_qubit_zz_sum_operator()
    H0 = float(omega) * G
    return _simulate_correlations(H0, t, psi0)


def lindblad_cxx_mismatch_rmse(
    t: np.ndarray,
    omega: float,
    cxx_target: np.ndarray,
    psi0: qt.Qobj,
    *,
    gamma_grid: np.ndarray | None = None,
) -> float:
    """
    Best RMSE between ``C_xx`` from independent dephasing on each qubit and ``cxx_target``.

    Collapse ops ``√γ σ_z ⊗ I`` and ``√γ I ⊗ σ_z``; scan ``γ`` on a log-spaced grid.
    """
    t = np.asarray(t, dtype=float).ravel()
    cxx_target = np.asarray(cxx_target, dtype=float).ravel()
    if gamma_grid is None:
        gamma_grid = np.logspace(-3, 1.5, 48)
    G = two_qubit_zz_sum_operator()
    H = float(omega) * G
    XX = qt.tensor(qt.sigmax(), qt.sigmax())
    sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
    sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())
    best = float("inf")
    for g in gamma_grid:
        if g < 0:
            continue
        c_ops = [np.sqrt(float(g)) * sz1, np.sqrt(float(g)) * sz2]
        res = qt.mesolve(H, psi0, t, c_ops, e_ops=[XX])
        cxx_lb = np.asarray(res.expect[0], dtype=float)
        err = float(np.sqrt(np.mean((cxx_lb - cxx_target) ** 2)))
        if err < best:
            best = err
    return best


def compute_observable_row(
    t: np.ndarray,
    omega: float,
    traces: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    psi0: qt.Qobj,
) -> dict[str, float]:
    """
    Scalar metrics for one sweep point: entropy, overlaps, Lindblad mismatch, CHSH entropy, f_dom.
    """
    cxx = traces["cxx"]
    cyy = traces["cyy"]
    chsh = traces["chsh"]
    # FFT entropy of C_xx as the Bell-channel proxy for “⟨σ_x⟩-like” spectral width
    s_ent = spectral_entropy_nats_fft(t, cxx)
    chsh_ent = spectral_entropy_nats_fft(t, chsh)
    rxx = pearson_r(cxx, ref["cxx"])
    ryy = pearson_r(cyy, ref["cyy"])
    rsh = pearson_r(chsh, ref["chsh"])
    overlap = float(np.nanmean([rxx, ryy, rsh]))
    dec_rmse = lindblad_cxx_mismatch_rmse(t, omega, cxx, psi0)
    f_dom = dominant_frequency_hz(t, cxx)
    return {
        "spectrum_entropy_sx": float(s_ent),
        "overlap_correlation": overlap,
        "decoherence_mismatch_lindblad": float(dec_rmse),
        "chsh_spectral_entropy": float(chsh_ent),
        "dominant_freq_cxx": float(f_dom),
    }


def run_tdf_vs_colored_noise_discrimination(
    t: np.ndarray | None = None,
    *,
    omega: float = 1.0,
    tau_freq: float = 3.0,
    tau_noise_strength: float = 0.5,
    tau_c_values: np.ndarray | None = None,
    ou_correlation_times: np.ndarray | None = None,
    pink_noise_strengths: np.ndarray | None = None,
    ou_sigma: float = 0.5,
    base_seed: int = 4242,
    output_dir: str | Path | None = None,
    save_csv: bool = True,
    save_plots: bool = True,
    show: bool = False,
) -> dict[str, Any]:
    """
    Sweep control parameters for TDF (``τ_c``), OU (``correlation_time``), and pink (strength).

    Returns combined dataframe, per-model coupling matrices, joint scores, and interpretation.
    """
    if t is None:
        t = np.linspace(0.0, 10.0, 280)
    t = np.asarray(t, dtype=float).ravel()
    if tau_c_values is None:
        tau_c_values = np.linspace(0.35, 2.2, 7)
    if ou_correlation_times is None:
        ou_correlation_times = np.linspace(0.35, 2.2, 7)
    if pink_noise_strengths is None:
        pink_noise_strengths = np.linspace(0.15, 1.2, 7)

    psi0 = bell_phi_plus()
    ref = _reference_standard(t, omega, psi0)

    rows: list[dict[str, Any]] = []

    for i, tc in enumerate(tau_c_values):
        tau = correlated_stochastic_tau(
            t,
            omega=omega,
            freq=tau_freq,
            noise_strength=tau_noise_strength,
            correlation_time=float(tc),
            seed=base_seed + 1000 * i,
        )
        H = tau_to_two_qubit_tdf_hamiltonian(tau, t, hbar=1.0)
        tr = _simulate_correlations(H, t, psi0)
        m = compute_observable_row(t, omega, tr, ref, psi0)
        rows.append(
            {
                "model": "tdf",
                "control_param": "tau_c",
                "control_value": float(tc),
                **m,
            }
        )

    for i, tc_ou in enumerate(ou_correlation_times):
        xi = ou_xi_trace(t, float(tc_ou), ou_sigma, seed=base_seed + 2000 * i)
        H = _scalar_coupling_hamiltonian(t, omega + xi)
        tr = _simulate_correlations(H, t, psi0)
        m = compute_observable_row(t, omega, tr, ref, psi0)
        rows.append(
            {
                "model": "ou_colored",
                "control_param": "correlation_time",
                "control_value": float(tc_ou),
                **m,
            }
        )

    for i, pn in enumerate(pink_noise_strengths):
        xi = pink_xi_trace(t, float(pn), seed=base_seed + 3000 * i)
        H = _scalar_coupling_hamiltonian(t, omega + xi)
        tr = _simulate_correlations(H, t, psi0)
        m = compute_observable_row(t, omega, tr, ref, psi0)
        rows.append(
            {
                "model": "pink",
                "control_param": "noise_strength",
                "control_value": float(pn),
                **m,
            }
        )

    df = metrics_dataframe(rows)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coupling_mats: dict[str, tuple[np.ndarray, list[str]]] = {}
    scores: dict[str, float] = {}

    fig_paths: dict[str, Path] = {}

    for mname in ("tdf", "ou_colored", "pink"):
        sub = df[df["model"] == mname].copy()
        R, labels = cross_observable_coupling_matrix(sub, METRIC_COLUMNS)
        coupling_mats[mname] = (R, labels)
        scores[mname] = joint_coupling_score(R)

        if save_plots:
            plot_metrics_vs_control(
                sub,
                model_label=mname,
                control_col="control_value",
                output_path=output_dir / f"tdf_vs_noise_metrics_{mname}.png",
            )
            plot_coupling_heatmap(
                R,
                labels,
                title=f"Cross-observable coupling ({mname})",
                output_path=output_dir / f"tdf_vs_noise_coupling_{mname}.png",
            )
            fig_paths[f"metrics_{mname}"] = output_dir / f"tdf_vs_noise_metrics_{mname}.png"
            fig_paths[f"coupling_{mname}"] = output_dir / f"tdf_vs_noise_coupling_{mname}.png"

    if save_plots:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        names = list(scores.keys())
        vals = [scores[k] for k in names]
        ax.bar(names, vals, color=["C0", "C1", "C2"], edgecolor="k")
        ax.set_ylabel("joint coupling score")
        ax.set_title("Mean |off-diagonal| Pearson (across sweep)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p_cmp = output_dir / "tdf_vs_colored_noise_joint_scores.png"
        fig.savefig(p_cmp, dpi=150)
        if not show:
            plt.close(fig)
        fig_paths["joint_scores"] = p_cmp

    interpretation = interpret_tdf_vs_colored_noise(scores)

    csv_path = None
    summary_path = None
    if save_csv:
        csv_path = output_dir / "tdf_vs_colored_noise_sweep.csv"
        df.to_csv(csv_path, index=False)
        model_summary = pd.DataFrame(
            [{"model": k, "joint_coupling_score": scores[k]} for k in scores]
        )
        summary_path = output_dir / "tdf_vs_colored_noise_coupling_summary.csv"
        model_summary.to_csv(summary_path, index=False)

    print()
    print("Joint cross-observable coupling scores (mean |off-diag| Pearson):")
    print_coupling_comparison(scores)
    print()
    print(interpretation)

    if show and save_plots:
        plt.show()

    return {
        "dataframe": df,
        "coupling_matrices": coupling_mats,
        "joint_scores": scores,
        "interpretation": interpretation,
        "figure_paths": fig_paths,
        "csv_path": csv_path,
        "coupling_summary_csv_path": summary_path,
    }


if __name__ == "__main__":
    from scripts.pipeline_demo import step_tdf_vs_colored_noise

    step_tdf_vs_colored_noise(None)

