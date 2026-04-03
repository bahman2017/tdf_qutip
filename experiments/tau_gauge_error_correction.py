"""
Minimal τ-based “gauge” correction demo on a noisy two-qubit Bell trajectory.

|Φ+⟩ evolves under H₀ = ω G (G = σ_z⊗I + I⊗σ_z) with independent local Z dephasing.
Reference phase τ(t) = ω t + a sin(ν t) defines the noiseless TDF Hamiltonian
H_ref(t) = E(t) G with E = ℏ dτ/dt. The target CHSH trace S_exp(t) comes from
noiseless evolution under H_ref.

**Correction** (piecewise, causal): at time t_i before stepping to t_{i+1},

    u(t_i) = (E(t_i) − ω) − k (S_meas(t_i) − S_exp(t_i)).

The first term is **τ feedforward** from the known gauge (no extra measurement).
The second term is the requested **error signal** law using CHSH.
Pure CHSH-only feedback (k without feedforward) is a poor stabilizer here because
local Z Lindbladians commute with G; the feedforward part is what realigns H(t)
with H_ref when τ is modulated.

Outputs: fidelity vs |ψ_ref(t)⟩ and CHSH vs time for uncorrected vs corrected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from core.evolution import evolve_open, run_evolution
from core.hamiltonians import (
    tau_to_energy,
    tau_to_two_qubit_tdf_hamiltonian,
    two_qubit_zz_sum_operator,
)


def bell_phi_plus() -> qt.Qobj:
    b0, b1 = qt.basis(2, 0), qt.basis(2, 1)
    return (qt.tensor(b0, b0) + qt.tensor(b1, b1)).unit()


def chsh_operators() -> tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    A0, A1 = qt.sigmaz(), qt.sigmax()
    B0 = (qt.sigmaz() + qt.sigmax()).unit()
    B1 = (qt.sigmaz() - qt.sigmax()).unit()
    return A0, A1, B0, B1


def _chsh_expect(rho: qt.Qobj) -> float:
    A0, A1, B0, B1 = chsh_operators()
    a0b0 = qt.tensor(A0, B0)
    a0b1 = qt.tensor(A0, B1)
    a1b0 = qt.tensor(A1, B0)
    a1b1 = qt.tensor(A1, B1)
    e00 = qt.expect(a0b0, rho)
    e01 = qt.expect(a0b1, rho)
    e10 = qt.expect(a1b0, rho)
    e11 = qt.expect(a1b1, rho)
    return float(np.real(e00 + e01 + e10 - e11))


def _fidelity_ref(rho: qt.Qobj, psi_ref: qt.Qobj) -> float:
    """Tr[ρ |ψ⟩⟨ψ|] = ⟨ψ|ρ|ψ⟩."""
    P = psi_ref * psi_ref.dag()
    return float(np.real(qt.expect(P, rho)))


def run_tau_gauge_error_correction_demo(
    t: np.ndarray | None = None,
    *,
    omega: float = 1.0,
    a: float = 0.35,
    nu: float = 2.0,
    gamma: float = 0.045,
    k_gain: float = 0.25,
    tau_feedforward: bool = True,
    hbar: float = 1.0,
    output_dir: str | Path | None = None,
    plot: bool = True,
    show: bool = False,
) -> dict[str, Any]:
    """
    Compare open-system trajectories with and without τ-gauge correction.

    Parameters
    ----------
    t
        Time samples (monotone). Default ``linspace(0, 8, 320)``.
    omega, a, nu
        τ(t) = ω t + a sin(ν t).
    gamma
        Local dephasing strength (Lindblad √(γ) Z on each qubit).
    k_gain
        Gain k on the CHSH error term in
        ``u = (E − ω) − k (S_meas − S_exp)`` (feedforward term omitted if
        ``tau_feedforward=False``).
    tau_feedforward
        If True (default), include ``E(t) − ω`` from ℏ dτ/dt. If False,
        ``u = −k (S_meas − S_exp)`` only (often needs hand-tuned k, γ).
    """
    if t is None:
        t = np.linspace(0.0, 8.0, 320)
    t = np.asarray(t, dtype=float).ravel()
    if t.size < 2:
        raise ValueError("Need at least two time points")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    psi0 = bell_phi_plus()
    rho0 = psi0 * psi0.dag()
    G = two_qubit_zz_sum_operator()
    H0 = float(omega) * G

    tau = omega * t + a * np.sin(nu * t)
    H_ref = tau_to_two_qubit_tdf_hamiltonian(tau, t, hbar=hbar)
    E_tau = np.asarray(tau_to_energy(tau, t, hbar=hbar), dtype=float)

    sz1 = np.sqrt(gamma) * qt.tensor(qt.sigmaz(), qt.qeye(2))
    sz2 = np.sqrt(gamma) * qt.tensor(qt.qeye(2), qt.sigmaz())
    c_ops = [sz1, sz2]

    # --- Noiseless reference (pure): CHSH and |ψ_ref(t)⟩ ---
    res_ref = run_evolution(H_ref, psi0, t)
    psi_ref_list = res_ref.states
    chsh_exp = np.array([_chsh_expect(s * s.dag()) for s in psi_ref_list], dtype=float)

    # --- Noisy, no correction ---
    res_bare = evolve_open(H0, rho0, t, c_ops)
    rho_bare = res_bare.states
    chsh_bare = np.array([_chsh_expect(r) for r in rho_bare], dtype=float)
    fid_bare = np.array(
        [_fidelity_ref(rho_bare[i], psi_ref_list[i]) for i in range(len(t))],
        dtype=float,
    )

    # --- Noisy + causal piecewise feedback ---
    rho_c = rho0.copy()
    chsh_corr = np.empty_like(t)
    fid_corr = np.empty_like(t)
    chsh_corr[0] = _chsh_expect(rho_c)
    fid_corr[0] = _fidelity_ref(rho_c, psi_ref_list[0])

    for i in range(len(t) - 1):
        u_ff = float(E_tau[i] - omega) if tau_feedforward else 0.0
        u_fb = -float(k_gain) * (chsh_corr[i] - chsh_exp[i])
        u = u_ff + u_fb
        H_fb = (float(omega) + u) * G
        seg = np.array([t[i], t[i + 1]], dtype=float)
        step = evolve_open(H_fb, rho_c, seg, c_ops)
        rho_c = step.states[-1]
        chsh_corr[i + 1] = _chsh_expect(rho_c)
        fid_corr[i + 1] = _fidelity_ref(rho_c, psi_ref_list[i + 1])

    paths: dict[str, Path] = {}
    if plot:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t, fid_bare, label="No correction", color="C0", lw=1.6, alpha=0.9)
        ax.plot(t, fid_corr, label="τ gauge + CHSH term", color="C2", lw=1.6)
        ax.plot(t, np.ones_like(t), "k--", lw=0.8, alpha=0.35, label="Ideal (noiseless ref)")
        ax.set_xlabel("t")
        ax.set_ylabel(r"Fidelity $\langle\psi_{\mathrm{ref}}(t)|\rho(t)|\psi_{\mathrm{ref}}(t)\rangle$")
        ax.set_title("Overlap with noiseless τ-reference state (local Z dephasing)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.05)
        fig.tight_layout()
        p1 = output_dir / "tau_gauge_fidelity.png"
        fig.savefig(p1, dpi=150)
        if not show:
            plt.close(fig)
        paths["fidelity"] = p1

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.plot(t, chsh_exp, "k--", lw=1.2, alpha=0.7, label="S_exp (noiseless τ-drive)")
        ax2.plot(t, chsh_bare, label="S_meas (no correction)", color="C0", lw=1.5)
        ax2.plot(t, chsh_corr, label="S_meas (corrected)", color="C2", lw=1.5)
        ax2.set_xlabel("t")
        ax2.set_ylabel("CHSH expectation")
        ax2.set_title("CHSH: tracking the τ-reference under dephasing")
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        p2 = output_dir / "tau_gauge_chsh.png"
        fig2.savefig(p2, dpi=150)
        if not show:
            plt.close(fig2)
        paths["chsh"] = p2

    return {
        "t": t,
        "tau": tau,
        "chsh_expected": chsh_exp,
        "chsh_bare": chsh_bare,
        "chsh_corrected": chsh_corr,
        "fidelity_bare": fid_bare,
        "fidelity_corrected": fid_corr,
        "mean_fid_gain": float(np.mean(fid_corr - fid_bare)),
        "mean_chsh_rmse_bare": float(np.sqrt(np.mean((chsh_bare - chsh_exp) ** 2))),
        "mean_chsh_rmse_corr": float(np.sqrt(np.mean((chsh_corr - chsh_exp) ** 2))),
        "paths": paths,
        "params": {
            "omega": omega,
            "a": a,
            "nu": nu,
            "gamma": gamma,
            "k_gain": k_gain,
            "tau_feedforward": tau_feedforward,
        },
    }


if __name__ == "__main__":
    out = run_tau_gauge_error_correction_demo()
    print(
        f"Mean fidelity gain (corrected − bare): {out['mean_fid_gain']:.4f}\n"
        f"RMSE(CHSH − S_exp) bare / corrected: "
        f"{out['mean_chsh_rmse_bare']:.4f} / {out['mean_chsh_rmse_corr']:.4f}"
    )
    print("Saved:", out["paths"])
