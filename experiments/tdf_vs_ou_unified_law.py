"""
TDF vs OU discrimination via **unified cross-observable laws** along a control sweep.

Compares low-order fits y(x) between metrics (not the raw Pearson coupling matrix of
metrics vs sweep index). OU must match TDF-like **stable functional relations** across
windows of the sweep to score well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.unified_law_metrics import (
    analyze_unified_laws,
    interpret_unified_tdf_vs_ou,
    plot_relation_fits,
    plot_window_coefficient_stability,
    print_unified_law_summary,
)
from core.evolution import run_evolution
from core.hamiltonians import tau_to_two_qubit_tdf_hamiltonian
from core.tau_model import correlated_stochastic_tau
from experiments.correlation_test import bell_phi_plus
from experiments.tdf_vs_colored_noise import (
    compute_observable_row,
    ou_xi_trace,
    _reference_standard,
    _scalar_coupling_hamiltonian,
    _simulate_correlations,
)


def collect_unified_law_sweeps(
    t: np.ndarray,
    *,
    omega: float = 1.0,
    tau_freq: float = 3.0,
    tau_noise_strength: float = 0.5,
    tau_c_values: np.ndarray,
    ou_correlation_times: np.ndarray,
    ou_sigma: float = 0.5,
    base_seed: int = 4242,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build TDF and OU sweep tables (metrics per control value) for unified-law analysis.

    Shared helper for :mod:`experiments.tdf_vs_ou_unified_law_robust`.
    """
    t = np.asarray(t, dtype=float).ravel()
    psi0 = bell_phi_plus()
    ref = _reference_standard(t, omega, psi0)

    rows_tdf: list[dict[str, Any]] = []
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
        rows_tdf.append({"model": "tdf", "control_value": float(tc), **m})

    rows_ou: list[dict[str, Any]] = []
    for i, tc_ou in enumerate(ou_correlation_times):
        xi = ou_xi_trace(t, float(tc_ou), ou_sigma, seed=base_seed + 2000 * i)
        H = _scalar_coupling_hamiltonian(t, omega + xi)
        tr = _simulate_correlations(H, t, psi0)
        m = compute_observable_row(t, omega, tr, ref, psi0)
        rows_ou.append({"model": "ou", "control_value": float(tc_ou), **m})

    return pd.DataFrame(rows_tdf), pd.DataFrame(rows_ou)


def run_tdf_vs_ou_unified_law(
    t: np.ndarray | None = None,
    *,
    omega: float = 1.0,
    tau_freq: float = 3.0,
    tau_noise_strength: float = 0.5,
    tau_c_values: np.ndarray | None = None,
    ou_correlation_times: np.ndarray | None = None,
    ou_sigma: float = 0.5,
    base_seed: int = 4242,
    n_windows: int = 3,
    output_dir: str | Path | None = None,
    save_plots: bool = True,
    show: bool = False,
) -> dict[str, Any]:
    """
    Sweep ``tau_c`` (TDF) and ``correlation_time`` (OU on ``ω+ξ``); fit unified laws; compare.

    Parameters
    ----------
    ou_sigma
        OU diffusion scale for ξ(t) (same discrete scheme as τ noise in ``correlated_stochastic_tau``).
    n_windows
        Number of contiguous segments for coefficient-stability diagnostics.
    """
    if t is None:
        t = np.linspace(0.0, 8.0, 220)
    t = np.asarray(t, dtype=float).ravel()
    if tau_c_values is None:
        tau_c_values = np.linspace(0.35, 2.2, 9)
    if ou_correlation_times is None:
        ou_correlation_times = np.linspace(0.35, 2.2, 9)

    df_tdf, df_ou = collect_unified_law_sweeps(
        t,
        omega=omega,
        tau_freq=tau_freq,
        tau_noise_strength=tau_noise_strength,
        tau_c_values=np.asarray(tau_c_values),
        ou_correlation_times=np.asarray(ou_correlation_times),
        ou_sigma=ou_sigma,
        base_seed=base_seed,
    )

    rep_tdf = analyze_unified_laws(df_tdf, n_windows=n_windows)
    rep_ou = analyze_unified_laws(df_ou, n_windows=n_windows)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if save_plots:
        p1 = output_dir / "tdf_vs_ou_unified_law_relations_tdf.png"
        p2 = output_dir / "tdf_vs_ou_unified_law_relations_ou.png"
        p3 = output_dir / "tdf_vs_ou_unified_law_stability_tdf.png"
        p4 = output_dir / "tdf_vs_ou_unified_law_stability_ou.png"
        p5 = output_dir / "tdf_vs_ou_unified_law_scores.png"

        plot_relation_fits(df_tdf, rep_tdf, title_prefix="TDF (τ_c sweep)", output_path=p1, show=show)
        plot_relation_fits(df_ou, rep_ou, title_prefix="OU (correlation_time sweep)", output_path=p2, show=show)
        plot_window_coefficient_stability(
            df_tdf,
            rep_tdf,
            title_prefix="TDF",
            n_windows=rep_tdf.n_windows,
            output_path=p3,
            show=show,
        )
        plot_window_coefficient_stability(
            df_ou,
            rep_ou,
            title_prefix="OU",
            n_windows=rep_ou.n_windows,
            output_path=p4,
            show=show,
        )

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.bar(
            ["TDF", "OU"],
            [rep_tdf.unified_score, rep_ou.unified_score],
            color=["C0", "C1"],
            edgecolor="k",
        )
        ax.set_ylabel("unified law score")
        ax.set_title("Higher ⇔ tighter fits + stable coeffs across windows")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(p5, dpi=150)
        if not show:
            plt.close(fig)

        paths = {
            "relations_tdf": p1,
            "relations_ou": p2,
            "stability_tdf": p3,
            "stability_ou": p4,
            "scores": p5,
        }

    interp = interpret_unified_tdf_vs_ou(rep_tdf, rep_ou)
    print_unified_law_summary("TDF (τ_c)", rep_tdf)
    print_unified_law_summary("OU (ξ correlation time)", rep_ou)
    print()
    print(interp)

    csv_tdf = output_dir / "tdf_vs_ou_unified_law_sweep_tdf.csv"
    csv_ou = output_dir / "tdf_vs_ou_unified_law_sweep_ou.csv"
    df_tdf.to_csv(csv_tdf, index=False)
    df_ou.to_csv(csv_ou, index=False)

    if show and save_plots:
        plt.show()

    return {
        "dataframe_tdf": df_tdf,
        "dataframe_ou": df_ou,
        "report_tdf": rep_tdf,
        "report_ou": rep_ou,
        "interpretation": interp,
        "figure_paths": paths if save_plots else {},
        "csv_paths": (csv_tdf, csv_ou),
    }


if __name__ == "__main__":
    run_tdf_vs_ou_unified_law()
