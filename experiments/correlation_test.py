"""
Two-qubit Bell-state experiment: standard vs TDF (τ-driven) dynamics on correlation observables.

TDF uses E(t) = ℏ dτ/dt multiplying the same two-qubit Z-generator as the standard benchmark,
so differences come solely from the structured / correlated τ field.

Note: |Φ+⟩ is an eigenstate of σ_z⊗σ_z, and H ∝ (σ_z⊗I+I⊗σ_z) preserves that eigenspace, so
⟨σ_z⊗σ_z⟩ stays constant; C_xx and C_yy carry the main contrast with standard evolution.

See also :mod:`analysis.tau_extraction` to fit a parametric τ(t) to the TDF correlation traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from analysis.correlation_metrics import summarize_cxx_comparison
from core.hamiltonians import two_qubit_zz_sum_operator, tau_to_two_qubit_tdf_hamiltonian
from core.tau_model import correlated_stochastic_tau
from core.evolution import run_evolution


def bell_phi_plus() -> qt.Qobj:
    """Maximally entangled state (|00⟩ + |11⟩) / √2."""
    b0 = qt.basis(2, 0)
    b1 = qt.basis(2, 1)
    return (qt.tensor(b0, b0) + qt.tensor(b1, b1)).unit()


def chsh_operators() -> tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    """
    CHSH local observables (same as common qubit CHSH setup):

    A0 = σ_z, A1 = σ_x,
    B0 ∝ σ_z + σ_x, B1 ∝ σ_z − σ_x (normalized).
    """
    A0 = qt.sigmaz()
    A1 = qt.sigmax()
    B0 = (qt.sigmaz() + qt.sigmax()).unit()
    B1 = (qt.sigmaz() - qt.sigmax()).unit()
    return A0, A1, B0, B1


def correlation_pauli_tensors() -> tuple[qt.Qobj, qt.Qobj, qt.Qobj]:
    """σ_x⊗σ_x, σ_y⊗σ_y, σ_z⊗σ_z."""
    sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    return qt.tensor(sx, sx), qt.tensor(sy, sy), qt.tensor(sz, sz)


def run_two_qubit_correlation_experiment(
    t: np.ndarray | None = None,
    *,
    omega: float = 1.0,
    output_dir: str | Path | None = None,
    tau_freq: float = 3.0,
    tau_noise_strength: float = 0.5,
    tau_correlation_time: float = 1.0,
    tau_seed: int = 4242,
    plot: bool = True,
    show: bool = False,
) -> dict[str, Any]:
    """
    Evolve |Φ+⟩ under H_std = ω G and H_tdf(t) = E_τ(t) G, compare correlations and CHSH.

    Parameters
    ----------
    t
        Time list (default ``linspace(0, 10, 300)``).
    omega
        Coefficient for the standard time-independent Hamiltonian.
    output_dir
        Directory for PNGs (default ``<repo>/outputs``).
    tau_freq, tau_noise_strength, tau_correlation_time, tau_seed
        Passed to :func:`correlated_stochastic_tau` for τ(t).

    Returns
    -------
    dict
        Trajectories, CHSH traces, metrics, and saved figure paths.
    """
    if t is None:
        t = np.linspace(0.0, 10.0, 300)
    t = np.asarray(t, dtype=float)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    psi0 = bell_phi_plus()
    G = two_qubit_zz_sum_operator()
    H_std = float(omega) * G

    tau = correlated_stochastic_tau(
        t,
        omega=omega,
        freq=tau_freq,
        noise_strength=tau_noise_strength,
        correlation_time=tau_correlation_time,
        seed=tau_seed,
    )
    H_tdf = tau_to_two_qubit_tdf_hamiltonian(tau, t, hbar=1.0)

    XX, YY, ZZ = correlation_pauli_tensors()
    A0, A1, B0, B1 = chsh_operators()
    A0B0 = qt.tensor(A0, B0)
    A0B1 = qt.tensor(A0, B1)
    A1B0 = qt.tensor(A1, B0)
    A1B1 = qt.tensor(A1, B1)

    e_ops = [XX, YY, ZZ, A0B0, A0B1, A1B0, A1B1]

    res_std = run_evolution(H_std, psi0, t, e_ops=e_ops)
    res_tdf = run_evolution(H_tdf, psi0, t, e_ops=e_ops)

    cxx_s = np.asarray(res_std.expect[0], dtype=float)
    cyy_s = np.asarray(res_std.expect[1], dtype=float)
    czz_s = np.asarray(res_std.expect[2], dtype=float)

    cxx_t = np.asarray(res_tdf.expect[0], dtype=float)
    cyy_t = np.asarray(res_tdf.expect[1], dtype=float)
    czz_t = np.asarray(res_tdf.expect[2], dtype=float)

    e00_s, e01_s, e10_s, e11_s = (
        np.asarray(res_std.expect[3], dtype=float),
        np.asarray(res_std.expect[4], dtype=float),
        np.asarray(res_std.expect[5], dtype=float),
        np.asarray(res_std.expect[6], dtype=float),
    )
    e00_td, e01_td, e10_td, e11_td = (
        np.asarray(res_tdf.expect[3], dtype=float),
        np.asarray(res_tdf.expect[4], dtype=float),
        np.asarray(res_tdf.expect[5], dtype=float),
        np.asarray(res_tdf.expect[6], dtype=float),
    )

    chsh_std = e00_s + e01_s + e10_s - e11_s
    chsh_tdf = e00_td + e01_td + e10_td - e11_td

    metrics_xx = summarize_cxx_comparison(t, cxx_s, cxx_t)
    metrics_yy = {"rmse_cyy_std_vs_tdf": float(np.sqrt(np.mean((cyy_s - cyy_t) ** 2)))}
    metrics_zz = {"rmse_czz_std_vs_tdf": float(np.sqrt(np.mean((czz_s - czz_t) ** 2)))}
    metrics_chsh = {
        "rmse_chsh_std_vs_tdf": float(np.sqrt(np.mean((chsh_std - chsh_tdf) ** 2))),
    }

    paths: dict[str, Path] = {}

    def _save_compare(
        y_std: np.ndarray,
        y_tdf: np.ndarray,
        ylabel: str,
        title: str,
        filename: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t, y_std, label="Standard", color="C0", linewidth=1.8)
        ax.plot(t, y_tdf, label="TDF (τ)", color="C1", linewidth=1.8, alpha=0.9)
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = output_dir / filename
        fig.savefig(p, dpi=150)
        if not show:
            plt.close(fig)
        return p

    if plot:
        paths["cxx"] = _save_compare(cxx_s, cxx_t, r"$\langle\sigma_x\otimes\sigma_x\rangle$", r"$C_{xx}(t)$", "correlation_xx.png")
        paths["cyy"] = _save_compare(cyy_s, cyy_t, r"$\langle\sigma_y\otimes\sigma_y\rangle$", r"$C_{yy}(t)$", "correlation_yy.png")
        paths["czz"] = _save_compare(czz_s, czz_t, r"$\langle\sigma_z\otimes\sigma_z\rangle$", r"$C_{zz}(t)$", "correlation_zz.png")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t, chsh_std, label=r"Standard $S(t)$", color="C0", linewidth=1.8)
        ax.plot(t, chsh_tdf, label=r"TDF $S(t)$", color="C1", linewidth=1.8, alpha=0.9)
        ax.axhline(2.0, color="k", linestyle=":", linewidth=1.0, label="classical bound ($2$)")
        ax.axhline(2 * np.sqrt(2), color="gray", linestyle=":", linewidth=1.0, label=rf"Tsirelson ($2\sqrt{{2}}$)")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$S = \langle A_0B_0\rangle + \langle A_0B_1\rangle + \langle A_1B_0\rangle - \langle A_1B_1\rangle$")
        ax.set_title("CHSH-style correlator (two-qubit)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pch = output_dir / "chsh_tdf_vs_standard.png"
        fig.savefig(pch, dpi=150)
        paths["chsh"] = pch
        if not show:
            plt.close(fig)

        for key, pth in paths.items():
            print(f"Saved {pth}")

        print()
        print("Two-qubit correlation metrics (TDF vs standard):")
        print(f"  C_xx RMSE: {metrics_xx['rmse_cxx_std_vs_tdf']:.6g}")
        print(f"  C_xx Pearson r: {metrics_xx['pearson_r_cxx']:.6g}")
        print(f"  C_xx (TDF) spectral entropy: {metrics_xx['spectral_entropy_cxx_tdf_nats']:.6g} nats")
        print(f"  C_xx (TDF) dominant freq: {metrics_xx['dominant_frequency_cxx_tdf']:.6g}")
        print(f"  C_yy RMSE: {metrics_yy['rmse_cyy_std_vs_tdf']:.6g}")
        print(f"  C_zz RMSE: {metrics_zz['rmse_czz_std_vs_tdf']:.6g}")
        print(f"  CHSH S(t) RMSE: {metrics_chsh['rmse_chsh_std_vs_tdf']:.6g}")

    if show and plot:
        plt.show()

    return {
        "t": t,
        "tau": tau,
        "cxx_standard": cxx_s,
        "cxx_tdf": cxx_t,
        "cyy_standard": cyy_s,
        "cyy_tdf": cyy_t,
        "czz_standard": czz_s,
        "czz_tdf": czz_t,
        "chsh_standard": chsh_std,
        "chsh_tdf": chsh_tdf,
        "metrics_cxx": metrics_xx,
        "metrics_cyy": metrics_yy,
        "metrics_czz": metrics_zz,
        "metrics_chsh": metrics_chsh,
        "figure_paths": paths,
        "output_dir": output_dir,
    }
