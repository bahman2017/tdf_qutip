"""
Three-qubit GHZ coherence under classical TDF-style local ``σ_z`` phase noise (QuTiP).

Initial state ``|ψ₀⟩ = (|000⟩+|111⟩)/√2``. Per trajectory ``j`` and time ``t``,

    U_j = e^{-i τ₁ (σ_z ⊗ I ⊗ I)/2}
          e^{-i τ₂ (I ⊗ σ_z ⊗ I)/2}
          e^{-i τ₃ (I ⊗ I ⊗ σ_z)/2}

with ``τ_k(t) = ω t + σ W_k(t)`` (after drift). **Cases:**

* **A** — independent Wieners ``W₁, W₂, W₃``
* **B** — one shared ``W``: ``τ₁ = τ₂ = τ₃``
* **C** — partial correlation (same pattern as the 2-qubit phase test, extended):
  ``τ₁ = ωt + σ W₁``, ``τ₂ = ωt + σ(ρ W₁ + √(1-ρ²) W₂)``,
  ``τ₃ = ωt + σ(ρ W₁ + √(1-ρ²) W₃)`` with ``W₂, W₃`` independent of each other and of ``W₁``.

Ensemble average ``ρ̄(t)``. The ``|000⟩``–``|111⟩`` matrix element picks up phase
``-(τ₁+τ₂+τ₃)``, so (Gaussian proxy)

    C(t) ≈ exp(-Var(τ₁+τ₂+τ₃) / 2)

after normalizing ``C(0)=1`` via ``C(t) = |ρ_{000,111}(t)| / |ρ_{000,111}(0)|``.

**Expectation:** independent legs → fastest decay of ``ρ̄`` coherence; shared leg noise
preserves coherence **per trajectory** but ensemble averaging still smears ``ρ_{000,111}``
when the common phase varies across trajectories; partial correlation is intermediate.

Run::

    python -m experiments.tdf_ghz_decay
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


def _ghz_state() -> qt.Qobj:
    zzz = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
    ooo = qt.tensor(qt.basis(2, 1), qt.basis(2, 1), qt.basis(2, 1))
    psi = (zzz + ooo).unit()
    return psi * psi.dag()


def _ghz_coherence_index(rho: qt.Qobj) -> complex:
    """Off-diagonal ``⟨000|ρ|111⟩`` in computational basis (flattened 8×8)."""
    return complex(rho[0, 7])


def simulate_tau_ghz_cases(
    t: np.ndarray,
    omega: float,
    sigma: float,
    n_ensemble: int,
    rho: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ``τ₁``, ``τ₂``, ``τ₃`` for cases A/B/C; each array shape ``(3, n_ensemble, n_times)``.
    """
    n_times = int(t.size)
    dt = float(t[1] - t[0]) if n_times > 1 else 1.0
    drift = linear_tau(t, omega)
    sqrt_1mr2 = float(np.sqrt(max(0.0, 1.0 - rho * rho)))

    tau1 = np.zeros((3, n_ensemble, n_times), dtype=float)
    tau2 = np.zeros((3, n_ensemble, n_times), dtype=float)
    tau3 = np.zeros((3, n_ensemble, n_times), dtype=float)

    rng = np.random.default_rng(seed)
    for i in range(n_ensemble):
        w1 = wiener_path(rng, n_times, dt)
        w2 = wiener_path(rng, n_times, dt)
        w3 = wiener_path(rng, n_times, dt)
        w4 = wiener_path(rng, n_times, dt)

        # A: independent
        tau1[0, i, :] = drift + sigma * w1
        tau2[0, i, :] = drift + sigma * w2
        tau3[0, i, :] = drift + sigma * w3

        # B: shared
        tau1[1, i, :] = drift + sigma * w1
        tau2[1, i, :] = drift + sigma * w1
        tau3[1, i, :] = drift + sigma * w1

        # C: τ₁ drives correlated part on τ₂, τ₃ (symmetric mixed legs)
        tau1[2, i, :] = drift + sigma * w1
        tau2[2, i, :] = drift + sigma * (rho * w1 + sqrt_1mr2 * w2)
        tau3[2, i, :] = drift + sigma * (rho * w1 + sqrt_1mr2 * w4)

    return tau1, tau2, tau3


def coherence_C_and_prediction(
    tau1: np.ndarray,
    tau2: np.ndarray,
    tau3: np.ndarray,
    *,
    rho0: qt.Qobj,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns ``C`` (normalized ``|ρ_{0,7}|``), ``pred`` = ``exp(-Var(sum τ)/2)``,
    ``var_sum``; each shape ``(3, n_times)``.
    """
    n_cases, n_ens, n_times = tau1.shape
    sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(2))
    sz2 = qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(2))
    sz3 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmaz())

    rho07_0 = _ghz_coherence_index(rho0)
    abs0 = float(np.abs(rho07_0))
    if abs0 < 1e-15:
        raise ValueError("|rho_{000,111}(0)| too small")

    C = np.zeros((n_cases, n_times), dtype=float)
    pred = np.zeros((n_cases, n_times), dtype=float)
    var_sum = np.zeros((n_cases, n_times), dtype=float)

    for ci in range(n_cases):
        for kt in range(n_times):
            s = tau1[ci, :, kt] + tau2[ci, :, kt] + tau3[ci, :, kt]
            vs = float(np.var(s, ddof=0))
            var_sum[ci, kt] = vs
            pred[ci, kt] = float(np.exp(-0.5 * vs))

            acc = np.zeros((8, 8), dtype=complex)
            for j in range(n_ens):
                a1 = float(tau1[ci, j, kt])
                a2 = float(tau2[ci, j, kt])
                a3 = float(tau3[ci, j, kt])
                u = (
                    (-1j * a1 * sz1 / 2.0).expm()
                    * (-1j * a2 * sz2 / 2.0).expm()
                    * (-1j * a3 * sz3 / 2.0).expm()
                )
                r = u * rho0 * u.dag()
                acc += r.full()
            rho_bar = qt.Qobj(acc / n_ens, dims=rho0.dims)
            C[ci, kt] = float(np.abs(_ghz_coherence_index(rho_bar))) / abs0

    return C, pred, var_sum


def run_ghz_decay_experiment(
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
    tau1, tau2, tau3 = simulate_tau_ghz_cases(t, omega, sigma, n_ensemble, rho, seed)
    rho0 = _ghz_state()
    C, pred, var_sum = coherence_C_and_prediction(tau1, tau2, tau3, rho0=rho0)

    display_names = ("A: independent", "B: shared", "C: partial")
    metrics_rows: list[dict[str, Any]] = []
    for ci, disp in enumerate(display_names):
        mse, r2, pear = _mse_r2_pearson(C[ci], pred[ci])
        metrics_rows.append(
            {
                "case": disp,
                "mse": mse,
                "r2": r2,
                "pearson_corr": pear,
                "var_sum_tau_at_tmax": float(var_sum[ci, -1]),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "tdf_ghz_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    fig1, ax1 = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax1.plot(t, C[0], lw=2.0, label="A: independent W₁, W₂, W₃")
    ax1.plot(t, C[1], lw=2.0, label="B: shared W (τ₁=τ₂=τ₃)")
    ax1.plot(t, C[2], lw=2.0, label=f"C: partial ρ={rho}")
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$C(t)=|\rho_{000,111}|/|\rho_{000,111}(0)|$")
    ax1.set_title(r"GHZ $(|000\rangle+|111\rangle)/\sqrt{2}$ + local $σ_z$ phase noise")
    ax1.grid(True, alpha=0.35)
    ax1.legend(frameon=False, fontsize=9)
    fig1.tight_layout()
    p1 = output_dir / "tdf_ghz_decay.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), dpi=150, sharey=True)
    titles = (
        "A: independent noise",
        "B: shared noise",
        f"C: partial (ρ={rho})",
    )
    for ax, ci, title in zip(axes, (0, 1, 2), titles):
        ax.plot(t, C[ci], "C0", lw=2.0, label=r"$C(t)$")
        ax.plot(
            t,
            pred[ci],
            "C1",
            ls="--",
            lw=2.0,
            label=r"$\exp(-\mathrm{Var}(\tau_1{+}\tau_2{+}\tau_3)/2)$",
        )
        ax.set_xlabel("time")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("value")
    fig2.suptitle(
        r"GHZ coherence vs $\exp(-\mathrm{Var}(\tau_1+\tau_2+\tau_3)/2)$",
        fontsize=11,
        y=1.02,
    )
    fig2.tight_layout()
    p2 = output_dir / "tdf_ghz_compare.png"
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
    print("--- C(t) vs exp(-Var(τ₁+τ₂+τ₃)/2) ---")
    for row in metrics_rows:
        pc = row["pearson_corr"]
        pc_s = f"{pc:.6f}" if np.isfinite(pc) else "nan"
        print(f"{row['case']}: MSE={row['mse']:.6e}  R²={row['r2']:.6f}  Corr={pc_s}")

    return {
        "t": t,
        "C": C,
        "prediction": pred,
        "var_sum": var_sum,
        "metrics": metrics_df,
        "figure_decay": p1,
        "figure_compare": p2,
        "metrics_csv_path": metrics_path,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    run_ghz_decay_experiment()
