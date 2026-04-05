"""
Two-qubit entanglement under classical TDF-style local ``σ_z`` phase noise (QuTiP).

Initial state: Bell ``|Φ⁺⟩ = (|00⟩+|11⟩)/√2``. Per trajectory ``j`` and time ``t``,

    U_j(t) = e^{-i τ₁(t) (σ_z ⊗ I)/2} · e^{-i τ₂(t) (I ⊗ σ_z)/2}

with ``τ_k(t) = ω t + σ W_k(t)`` and the **same** three Wiener constructions as
``experiments/tdf_phase_decoherence_test``:

* **Case A** — independent ``W₁, W₂``
* **Case B** — shared ``W`` on both legs (``τ₁ = τ₂``) ⇒ local phases identical on the
  two tensor factors that preserve maximal entanglement trajectory-wise
* **Case C** — partial correlation ``τ₂ ∝ ρ W₁ + √(1-ρ²) W₃``

Ensemble-averaged state ``ρ̄(t) = (1/N) Σ_j U_j ρ₀ U_j^†``; **concurrence** ``C(t)``.

**Variance of** ``Δτ = τ₁ - τ₂`` **(requested, same as phase test):**

    P_Δ(t) = exp(-Var(Δτ) / 2)

For this Bell + local ``σ_z`` model, the ``|00⟩``–``|11⟩`` coherence picks up phase
``∝ -(τ₁+τ₂)``, so a Gaussian proxy that tracks **common-mode** noise is

    P_Σ(t) = exp(-Var(τ₁ + τ₂) / 2).

For **Case A** (independent identical legs), ``Var(Δτ) = Var(τ₁+τ₂)``; for **Case B**
(shared leg Wiener, ``τ₁=τ₂``), ``Var(Δτ)=0`` but ``Var(τ₁+τ₂)=4 Var(τ)`` — trajectory-wise
concurrence stays 1, while **ensemble-averaged** concurrence can still decay.

Plots compare ``C(t)`` to ``P_Δ`` (dashed) and ``P_Σ`` (dotted); **metrics** report MSE /
R² / Pearson for both. Intuition: **independent** legs give the strongest decay of
``ρ̄``; **partial** correlation interpolates; **shared** leg noise (``τ₁=τ₂``) still
allows decay of **ensemble-averaged** concurrence because the common phase varies across
trajectories — only ``P_Σ`` (not ``P_Δ``) tracks that decay.

Run::

    python -m experiments.tdf_entanglement_decay
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt

from core.noise_models import wiener_path
from core.tau_model import linear_tau
from experiments.tdf_phase_decoherence_test import _mse_r2_pearson


def _bell_state() -> qt.Qobj:
    z = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    o = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
    psi = (z + o).unit()
    return psi * psi.dag()


def simulate_tau1_tau2_cases(
    t: np.ndarray,
    omega: float,
    sigma: float,
    n_ensemble: int,
    rho: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``τ₁``, ``τ₂`` for cases A/B/C; each array shape ``(3, n_ensemble, n_times)``.
    """
    n_times = int(t.size)
    dt = float(t[1] - t[0]) if n_times > 1 else 1.0
    drift = linear_tau(t, omega)

    tau1 = np.zeros((3, n_ensemble, n_times), dtype=float)
    tau2 = np.zeros((3, n_ensemble, n_times), dtype=float)

    rng = np.random.default_rng(seed)
    sqrt_1mr2 = float(np.sqrt(max(0.0, 1.0 - rho * rho)))

    for i in range(n_ensemble):
        w1 = wiener_path(rng, n_times, dt)
        w2 = wiener_path(rng, n_times, dt)
        w3 = wiener_path(rng, n_times, dt)

        t1a = drift + sigma * w1
        t2a = drift + sigma * w2
        tau1[0, i, :] = t1a
        tau2[0, i, :] = t2a

        t1b = drift + sigma * w1
        t2b = drift + sigma * w1
        tau1[1, i, :] = t1b
        tau2[1, i, :] = t2b

        t1c = drift + sigma * w1
        t2c = drift + sigma * (rho * w1 + sqrt_1mr2 * w3)
        tau1[2, i, :] = t1c
        tau2[2, i, :] = t2c

    return tau1, tau2


def concurrence_and_prediction_from_tau(
    tau1: np.ndarray,
    tau2: np.ndarray,
    *,
    rho0: qt.Qobj,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns ``conc``, ``pred_dtau``, ``var_dtau``, ``pred_sum``, ``var_sum``
    each with shape ``(3, n_times)``.
    """
    n_cases, n_ens, n_times = tau1.shape
    sz_i = qt.tensor(qt.sigmaz(), qt.qeye(2))
    isz = qt.tensor(qt.qeye(2), qt.sigmaz())

    conc = np.zeros((n_cases, n_times), dtype=float)
    pred_dtau = np.zeros((n_cases, n_times), dtype=float)
    var_dtau = np.zeros((n_cases, n_times), dtype=float)
    pred_sum = np.zeros((n_cases, n_times), dtype=float)
    var_sum = np.zeros((n_cases, n_times), dtype=float)

    for ci in range(n_cases):
        for kt in range(n_times):
            d = tau1[ci, :, kt] - tau2[ci, :, kt]
            s = tau1[ci, :, kt] + tau2[ci, :, kt]
            vd = float(np.var(d, ddof=0))
            vs = float(np.var(s, ddof=0))
            var_dtau[ci, kt] = vd
            var_sum[ci, kt] = vs
            pred_dtau[ci, kt] = float(np.exp(-0.5 * vd))
            pred_sum[ci, kt] = float(np.exp(-0.5 * vs))

            acc = np.zeros((4, 4), dtype=complex)
            for j in range(n_ens):
                a1 = float(tau1[ci, j, kt])
                a2 = float(tau2[ci, j, kt])
                u1 = (-1j * a1 * sz_i / 2.0).expm()
                u2 = (-1j * a2 * isz / 2.0).expm()
                u = u1 * u2
                r = u * rho0 * u.dag()
                acc += r.full()
            rho_bar = qt.Qobj(acc / n_ens, dims=rho0.dims)
            conc[ci, kt] = float(qt.concurrence(rho_bar))

    return conc, pred_dtau, var_dtau, pred_sum, var_sum


def run_entanglement_decay_experiment(
    *,
    omega: float = 1.0,
    sigma: float = 0.3,
    rho: float = 0.6,
    t_max: float = 2.0,
    n_times: int = 200,
    n_ensemble: int = 800,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, float(t_max), int(n_times), dtype=float)
    tau1, tau2 = simulate_tau1_tau2_cases(t, omega, sigma, n_ensemble, rho, seed)
    rho0 = _bell_state()
    conc, pred_dtau, var_dtau, pred_sum, var_sum = concurrence_and_prediction_from_tau(
        tau1, tau2, rho0=rho0
    )

    labels = ("caseA", "caseB", "caseC")
    display_names = ("A: independent", "B: shared", "C: partial")

    metrics_rows: list[dict[str, Any]] = []
    for name, disp in zip(labels, display_names):
        k = labels.index(name)
        m_d, r_d, p_d = _mse_r2_pearson(conc[k], pred_dtau[k])
        m_s, r_s, p_s = _mse_r2_pearson(conc[k], pred_sum[k])
        metrics_rows.append(
            {
                "case": disp,
                "mse_vs_exp_neg_var_dtau_over_2": m_d,
                "r2_vs_exp_neg_var_dtau_over_2": r_d,
                "pearson_vs_exp_neg_var_dtau_over_2": p_d,
                "mse_vs_exp_neg_var_sum_over_2": m_s,
                "r2_vs_exp_neg_var_sum_over_2": r_s,
                "pearson_vs_exp_neg_var_sum_over_2": p_s,
                "var_dtau_at_tmax": float(var_dtau[k, -1]),
                "var_sum_at_tmax": float(var_sum[k, -1]),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "tdf_entanglement_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # --- Fig 1: concurrence vs time (three cases) ---
    fig1, ax1 = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax1.plot(t, conc[0], lw=2.0, label="A: independent W₁, W₂")
    ax1.plot(t, conc[1], lw=2.0, label="B: shared W (τ₁=τ₂)")
    ax1.plot(t, conc[2], lw=2.0, label=f"C: partial ρ={rho}")
    ax1.set_xlabel("time")
    ax1.set_ylabel("concurrence")
    ax1.set_title(r"Ensemble-averaged concurrence (Bell + local $σ_z$ phase noise)")
    ax1.grid(True, alpha=0.35)
    ax1.legend(frameon=False, fontsize=9)
    fig1.tight_layout()
    p1 = output_dir / "tdf_entanglement_decay.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)

    # --- Fig 2: three panels, conc vs prediction ---
    fig2, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), dpi=150, sharey=True)
    titles = (
        "A: independent noise",
        "B: shared noise",
        f"C: partial (ρ={rho})",
    )
    for ax, ci, title in zip(axes, (0, 1, 2), titles):
        ax.plot(t, conc[ci], "C0", lw=2.0, label=r"$C(t)$ (concurrence)")
        ax.plot(
            t,
            pred_dtau[ci],
            "C1",
            ls="--",
            lw=2.0,
            label=r"$\exp(-\mathrm{Var}(\Delta\tau)/2)$",
        )
        ax.plot(
            t,
            pred_sum[ci],
            "C2",
            ls=":",
            lw=2.0,
            label=r"$\exp(-\mathrm{Var}(\tau_1{+}\tau_2)/2)$",
        )
        ax.set_xlabel("time")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("value")
    fig2.suptitle(
        r"Concurrence vs $\exp(-\mathrm{Var}(\Delta\tau)/2)$ and "
        r"$\exp(-\mathrm{Var}(\tau_1+\tau_2)/2)$ (Bell common-mode phase)",
        fontsize=10,
        y=1.03,
    )
    fig2.tight_layout()
    p2 = output_dir / "tdf_entanglement_compare.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {metrics_path}")
    print(
        f"Parameters: ω={omega}, σ={sigma}, T={t_max}, n_times={n_times}, "
        f"N={n_ensemble}, ρ(partial)={rho}, seed={seed}"
    )
    print()
    print("--- Concurrence vs exp(-Var(Δτ)/2) ---")
    for row in metrics_rows:
        pc = row["pearson_vs_exp_neg_var_dtau_over_2"]
        pc_s = f"{pc:.6f}" if np.isfinite(pc) else "nan"
        print(
            f"{row['case']}: MSE={row['mse_vs_exp_neg_var_dtau_over_2']:.6e}  "
            f"R²={row['r2_vs_exp_neg_var_dtau_over_2']:.6f}  Corr={pc_s}"
        )
    print()
    print("--- Concurrence vs exp(-Var(τ₁+τ₂)/2) (common-mode) ---")
    for row in metrics_rows:
        pc = row["pearson_vs_exp_neg_var_sum_over_2"]
        pc_s = f"{pc:.6f}" if np.isfinite(pc) else "nan"
        print(
            f"{row['case']}: MSE={row['mse_vs_exp_neg_var_sum_over_2']:.6e}  "
            f"R²={row['r2_vs_exp_neg_var_sum_over_2']:.6f}  Corr={pc_s}"
        )

    return {
        "t": t,
        "concurrence": conc,
        "prediction_dtau": pred_dtau,
        "prediction_sum": pred_sum,
        "var_dtau": var_dtau,
        "var_sum": var_sum,
        "metrics": metrics_df,
        "figure_decay": p1,
        "figure_compare": p2,
        "metrics_csv_path": metrics_path,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    run_entanglement_decay_experiment()
