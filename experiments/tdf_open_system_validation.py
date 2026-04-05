"""
Open-system validation: TDF phase-variance law vs ensemble-averaged qubit coherence (QuTiP).

Single qubit, ``ρ₀ = |+⟩⟨+|``. Per trajectory ``j`` and time ``t``, stochastic relative phase
``Δτ_j(t)`` is drawn from the **same** two-leg Wiener construction as
``experiments/tdf_phase_decoherence_test`` (Cases A/B/C: independent, shared, partially
correlated noise on ``τ₁, τ₂``).

Total ``σ_z`` rotation angle (commuting pieces combined):

    U_j(t) = exp\\!\\left(-\\frac{i}{2}\\bigl(ω t + Δτ_j(t)\\bigr) σ_z\\right)

with ``H = (ω/2) σ_z`` giving the deterministic part. Ensemble-averaged state
``ρ(t) = (1/N) Σ_j U_j ρ₀ U_j^†``.

**Coherence** (normalized off-diagonal):

    C(t) = |ρ₀₁(t)| / |ρ₀₁(0)|

**TDF / Gaussian prediction** (variance of ``Δτ`` across trajectories at fixed ``t``):

    P(t) = exp\\!\\left(-\\mathrm{Var}(Δτ)/2\\right)

Success: ``C(t) ≈ P(t)`` when the ensemble phase is approximately Gaussian.

**Lindblad extension:** ``L = \\sqrt{\\gamma}\\,\\sigma_z`` (pure dephasing). Model 1:
``mesolve`` only. Model 2 (Case A): each trajectory alternates a **dephasing** step
(``\\rho_{01} \\mapsto \\rho_{01} e^{-2\\gamma\\,dt}`` for ``L=\\sqrt{\\gamma}\\sigma_z``)
and an incremental ``\\sigma_z`` rotation from ``\\Delta\\tau``. We test whether

    C_{\\mathrm{tot}}(t) \\approx e^{-\\mathrm{Var}(\\Delta\\tau)/2}\\, C_L(t)

i.e. **multiplicative** splitting of τ-geometry and environment on coherence, and
compare to ``\\exp(-\\mathrm{Var}/2 - 2\\gamma t)`` (additive-in-log with Markovian rate
``2\\gamma`` for this ``L``).

Run::

    python -m experiments.tdf_open_system_validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt

from experiments.tdf_phase_decoherence_test import (
    _mse_r2_pearson,
    simulate_delta_tau_cases,
)


def _rho_plus() -> qt.Qobj:
    psi = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    return psi * psi.dag()


def _U_sigma_z(angle: float) -> qt.Qobj:
    """``exp(-i · (angle) · σ_z / 2)`` — matches ``H=(ω/2)σ_z`` over time for ``angle=ωt``."""
    return (-1j * float(angle) * qt.sigmaz() / 2.0).expm()


def coherence_and_prediction_from_delta_tau(
    t: np.ndarray,
    d_all: np.ndarray,
    *,
    omega: float,
    rho0: qt.Qobj,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``d_all`` shape ``(3, n_ensemble, n_times)`` for cases A, B, C.

    Returns ``C`` and ``P`` each shape ``(3, n_times)``.
    """
    n_cases, n_ens, n_times = d_all.shape
    rho01_0 = complex(rho0[0, 1])
    abs01_0 = float(np.abs(rho01_0))
    if abs01_0 < 1e-15:
        raise ValueError("|rho01(0)| too small")

    C = np.zeros((n_cases, n_times), dtype=float)
    P = np.zeros((n_cases, n_times), dtype=float)

    for ci in range(n_cases):
        for kt in range(n_times):
            tt = float(t[kt])
            dtau = d_all[ci, :, kt]
            P[ci, kt] = float(np.exp(-0.5 * float(np.var(dtau, ddof=0))))

            rho_sum = qt.Qobj(np.zeros((2, 2), dtype=complex), dims=[[2], [2]])
            for j in range(n_ens):
                ang = omega * tt + float(dtau[j])
                U = _U_sigma_z(ang)
                rho_j = U * rho0 * U.dag()
                rho_sum += rho_j
            rho_avg = rho_sum / n_ens
            C[ci, kt] = float(np.abs(complex(rho_avg[0, 1]))) / abs01_0

    return C, P


def run_open_system_validation(
    *,
    omega: float = 0.0,
    sigma: float = 0.3,
    rho: float = 0.6,
    t_max: float = 2.0,
    n_times: int = 200,
    n_ensemble: int = 1000,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, float(t_max), int(n_times), dtype=float)
    d_all = simulate_delta_tau_cases(t, omega, sigma, n_ensemble, rho, seed)
    rho0 = _rho_plus()

    C, P = coherence_and_prediction_from_delta_tau(t, d_all, omega=omega, rho0=rho0)

    df_data = pd.DataFrame(
        {
            "time": t,
            "C_caseA": C[0],
            "C_caseB": C[1],
            "C_caseC": C[2],
            "P_caseA": P[0],
            "P_caseB": P[1],
            "P_caseC": P[2],
        }
    )
    p_data = output_dir / "tdf_open_system_validation_data.csv"
    df_data.to_csv(p_data, index=False)

    case_labels = ("A", "B", "C")
    metrics_rows: list[dict[str, Any]] = []
    for i, lab in enumerate(case_labels):
        mse, r2, pear = _mse_r2_pearson(C[i], P[i])
        metrics_rows.append(
            {"case": lab, "mse": mse, "r2": r2, "pearson_corr": pear}
        )
    p_metrics = output_dir / "tdf_open_system_validation_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(p_metrics, index=False)

    # Plot 1: C(t) all cases
    fig1, ax1 = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax1.plot(t, C[0], lw=2.0, label="Case A (independent)")
    ax1.plot(t, C[1], lw=2.0, label="Case B (shared)")
    ax1.plot(t, C[2], lw=2.0, label="Case C (partial)")
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"normalized coherence $|\rho_{01}(t)|/|\rho_{01}(0)|$")
    ax1.set_title("QuTiP ensemble-averaged qubit coherence")
    ax1.grid(True, alpha=0.35)
    ax1.legend(frameon=False, fontsize=9)
    fig1.tight_layout()
    pc1 = output_dir / "tdf_open_system_validation_coherence.png"
    fig1.savefig(pc1, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: C vs P per case
    fig2, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), dpi=150, sharey=True)
    titles = ("A: independent", "B: shared", "C: partial")
    for ax, i, title in zip(axes, (0, 1, 2), titles):
        ax.plot(t, C[i], "C0", lw=2.0, label=r"$C(t)$")
        ax.plot(t, P[i], "C1", ls="--", lw=2.0, label=r"$e^{-\mathrm{Var}(\Delta\tau)/2}$")
        ax.set_xlabel("time")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("value")
    fig2.suptitle(
        r"Density-matrix coherence vs TDF phase-variance prediction",
        fontsize=11,
        y=1.02,
    )
    fig2.tight_layout()
    pc2 = output_dir / "tdf_open_system_validation_compare.png"
    fig2.savefig(pc2, bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote {p_data}")
    print(f"Wrote {p_metrics}")
    print(f"Wrote {pc1}")
    print(f"Wrote {pc2}")
    print(
        f"Parameters: ω={omega}, σ={sigma}, ρ={rho}, T={t_max}, "
        f"n_times={n_times}, N={n_ensemble}, seed={seed}"
    )
    print()
    print("--- Coherence C vs prediction P = exp(-Var(Δτ)/2) ---")
    for row in metrics_rows:
        pc = row["pearson_corr"]
        pc_s = f"{pc:.6f}" if np.isfinite(pc) else "nan"
        print(
            f"Case {row['case']}:  MSE={row['mse']:.6e}  R²={row['r2']:.6f}  Corr={pc_s}"
        )

    return {
        "t": t,
        "C": C,
        "P": P,
        "data_csv": p_data,
        "metrics_csv": p_metrics,
        "figures": (pc1, pc2),
        "output_dir": output_dir,
    }


def _lindblad_sigmaz_dephasing_step(rho: qt.Qobj, gamma: float, dt: float) -> qt.Qobj:
    """
    One Euler step for ``L = \\sqrt{\\gamma}\\sigma_z`` (zero Hamiltonian):

    ``d\\rho_{01}/dt = -2\\gamma\\rho_{01}`` ⇒ ``\\rho_{01} \\leftarrow \\rho_{01} e^{-2\\gamma dt}``.
    Diagonals unchanged for this channel on a qubit.
    """
    m = rho.full().copy()
    d = float(np.exp(-2.0 * gamma * dt))
    m[0, 1] *= d
    m[1, 0] *= d
    return qt.Qobj(m, dims=rho.dims)


def _mesolve_coherence_C_L(
    rho0: qt.Qobj,
    t: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Reference ``C_L(t) = |\\rho_{01}(t)|/|\\rho_{01}(0)|`` from ``mesolve``."""
    abs01_0 = float(np.abs(complex(rho0[0, 1])))
    H = 0 * qt.qeye(2)
    c_ops = [np.sqrt(float(gamma)) * qt.sigmaz()]
    res = qt.mesolve(H, rho0, t, c_ops, [])
    out = np.zeros(len(t), dtype=float)
    for k, st in enumerate(res.states):
        out[k] = float(np.abs(complex(st[0, 1]))) / abs01_0
    return out


def simulate_C_total_lindblad_tdf_case_a(
    t: np.ndarray,
    delta_tau: np.ndarray,
    *,
    omega: float,
    gamma: float,
    rho0: qt.Qobj,
    n_ensemble: int,
) -> np.ndarray:
    """
    Interleaved dephasing + incremental ``\\sigma_z`` phase (Case A paths ``delta_tau``).

    ``delta_tau`` shape ``(n_ensemble, n_times)``.
    """
    n_times = int(t.size)
    abs01_0 = float(np.abs(complex(rho0[0, 1])))
    C = np.zeros(n_times, dtype=float)
    C[0] = 1.0

    rhos = [rho0.copy() for _ in range(int(n_ensemble))]

    for k in range(1, n_times):
        dt = float(t[k] - t[k - 1])
        rho_sum = qt.Qobj(np.zeros((2, 2), dtype=complex), dims=[[2], [2]])
        for j in range(int(n_ensemble)):
            rhos[j] = _lindblad_sigmaz_dephasing_step(rhos[j], gamma, dt)
            dang = omega * dt + float(delta_tau[j, k] - delta_tau[j, k - 1])
            U = _U_sigma_z(dang)
            rhos[j] = U * rhos[j] * U.dag()
            rho_sum += rhos[j]
        rho_avg = rho_sum / float(n_ensemble)
        C[k] = float(np.abs(complex(rho_avg[0, 1]))) / abs01_0

    return C


def run_lindblad_validation(
    *,
    omega: float = 0.0,
    sigma: float = 0.3,
    rho_mix: float = 0.6,
    gamma: float = 0.08,
    t_max: float = 2.0,
    n_times: int = 200,
    n_ensemble: int = 800,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Lindblad + TDF (Case A only): compare ``C_tot`` to factorized and additive laws.
    """
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, float(t_max), int(n_times), dtype=float)
    d_all = simulate_delta_tau_cases(t, omega, sigma, n_ensemble, rho_mix, seed)
    delta_a = d_all[0, :, :]
    rho0 = _rho_plus()

    Var_t = np.var(delta_a, axis=0, ddof=0).astype(float)
    phase_factor = np.exp(-0.5 * Var_t)

    C_L = _mesolve_coherence_C_L(rho0, t, gamma)
    C_tot = simulate_C_total_lindblad_tdf_case_a(
        t,
        delta_a,
        omega=omega,
        gamma=gamma,
        rho0=rho0,
        n_ensemble=n_ensemble,
    )

    pred_product = phase_factor * C_L
    pred_additive = np.exp(-0.5 * Var_t - 2.0 * float(gamma) * t)

    mse_p, r2_p, corr_p = _mse_r2_pearson(C_tot, pred_product)
    mse_a, r2_a, corr_a = _mse_r2_pearson(C_tot, pred_additive)

    rows = [
        {
            "comparison": "C_tot vs exp(-Var/2)*C_L",
            "mse": mse_p,
            "r2": r2_p,
            "pearson_corr": corr_p,
        },
        {
            "comparison": "C_tot vs exp(-Var/2 - 2*gamma*t)",
            "mse": mse_a,
            "r2": r2_a,
            "pearson_corr": corr_a,
        },
    ]
    p_met = output_dir / "tdf_lindblad_metrics.csv"
    pd.DataFrame(rows).to_csv(p_met, index=False)

    # --- Figures ---
    fig_v, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 5.5), dpi=150, sharex=True)
    ax0.plot(t, C_L, lw=2.0, label=r"$C_L$ (Lindblad only)")
    ax0.plot(t, C_tot, lw=2.0, label=r"$C_{\mathrm{tot}}$ (TDF Case A + Lindblad)")
    ax0.plot(t, phase_factor, lw=2.0, ls=":", label=r"$e^{-\mathrm{Var}(\Delta\tau)/2}$")
    ax0.set_ylabel("coherence / factor")
    ax0.grid(True, alpha=0.35)
    ax0.legend(frameon=False, fontsize=8)
    ax0.set_title("Lindblad validation: environment vs τ-geometry pieces")

    ax1.plot(t, np.log(np.maximum(C_tot, 1e-15)), lw=2.0, label=r"$\log C_{\mathrm{tot}}$")
    ax1.plot(
        t,
        np.log(np.maximum(pred_product, 1e-15)),
        lw=2.0,
        ls="--",
        label=r"$\log(e^{-\mathrm{Var}/2} C_L)$",
    )
    ax1.plot(
        t,
        -0.5 * Var_t - 2.0 * float(gamma) * t,
        lw=1.5,
        ls=":",
        label=r"$-\mathrm{Var}/2 - 2\gamma t$",
    )
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$\log$ coherence / log-pred.")
    ax1.grid(True, alpha=0.35)
    ax1.legend(frameon=False, fontsize=7)
    fig_v.tight_layout()
    pv = output_dir / "tdf_lindblad_validation.png"
    fig_v.savefig(pv, bbox_inches="tight")
    plt.close(fig_v)

    fig_c, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(t, C_tot, "C0", lw=2.0, label=r"$C_{\mathrm{tot}}$ (simulation)")
    ax.plot(t, pred_product, "C1", ls="--", lw=2.0, label=r"$e^{-\mathrm{Var}/2}\,C_L$ (product law)")
    ax.plot(t, pred_additive, "C2", ls="-.", lw=2.0, label=r"$e^{-\mathrm{Var}/2 - 2\gamma t}$ (additive)")
    ax.set_xlabel("time")
    ax.set_ylabel(r"normalized $|\rho_{01}|$")
    ax.set_title("Total coherence vs TDF + Lindblad predictions (Case A)")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    fig_c.tight_layout()
    pc = output_dir / "tdf_lindblad_compare.png"
    fig_c.savefig(pc, bbox_inches="tight")
    plt.close(fig_c)

    print()
    print(f"Wrote {pv}")
    print(f"Wrote {pc}")
    print(f"Wrote {p_met}")
    print(
        f"Lindblad: γ={gamma} (L=√γ·σ_z ⇒ Markovian log-coherence rate 2γ on ρ₀₁), "
        f"σ_TDF={sigma}, N={n_ensemble}, Case A"
    )
    print()
    print("--- Lindblad + TDF: does C_tot ≈ exp(-Var/2)·C_L ? ---")
    for row in rows:
        c = row["pearson_corr"]
        c_s = f"{c:.6f}" if np.isfinite(c) else "nan"
        print(
            f"  {row['comparison']}:  MSE={row['mse']:.6e}  R²={row['r2']:.6f}  Corr={c_s}"
        )
    mean_log_resid = float(
        np.mean(np.abs(np.log(np.maximum(C_tot, 1e-15)) - np.log(np.maximum(pred_product, 1e-15))))
    )
    print()
    print(
        f"Mean |log C_tot - log(exp(-Var/2)·C_L)| = {mean_log_resid:.6e}  "
        "(0 if exact factorization in log space)"
    )

    return {
        "t": t,
        "C_L": C_L,
        "C_tot": C_tot,
        "Var_delta_tau": Var_t,
        "pred_product": pred_product,
        "pred_additive": pred_additive,
        "metrics_csv": p_met,
        "figures": (pv, pc),
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    run_open_system_validation()
    run_lindblad_validation()
