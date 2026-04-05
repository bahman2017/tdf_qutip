"""
Phase-decoherence toy test for the TDF picture: coherence from ensemble-averaged phase factors.

For relative phase Δτ(t) = τ₁(t) − τ₂(t) across **N** independent trajectory pairs, define

    V(t) = |⟨exp(i Δτ)⟩| = |(1/N) Σⱼ exp(i Δτⱼ(t))|

For **Gaussian** Δτ with mean zero, ⟨exp(iΔτ)⟩ = exp(−Var(Δτ)/2) (real and positive), so

    V(t) ≈ exp(−Var(Δτ)/2).

The script reports **MSE**, **R²**, and **Pearson r** between **V** and **exp(−Var/2)** per case, saves
``tdf_phase_decoherence_metrics.csv``, plots **residuals** ``V − exp(−Var/2)``, and optionally fits
**V(t) ≈ exp(−α t)** (compare **α** to **σ²** for the independent-noise scaling).

This script compares that prediction to empirical **V** from Wiener-driven τ paths:

* **Case A** — independent noise on two legs: τₖ = ωt + σ Wₖ, different W.
* **Case B** — shared noise: same W on both legs ⇒ Δτ ≡ 0 ⇒ V ≡ 1.
* **Case C** — partial correlation: τ₂ uses ρ W₁ + √(1−ρ²) W₂.

Run::

    python -m experiments.tdf_phase_decoherence_test
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from core.noise_models import wiener_path
from core.tau_model import linear_tau


def simulate_delta_tau_cases(
    t: np.ndarray,
    omega: float,
    sigma: float,
    n_ensemble: int,
    rho: float,
    seed: int,
) -> np.ndarray:
    """
    Return Δτ trajectories shape (3, n_ensemble, n_times) for cases A, B, C.
    """
    n_times = int(t.size)
    dt = float(t[1] - t[0]) if n_times > 1 else 1.0
    drift = linear_tau(t, omega)

    dA = np.zeros((n_ensemble, n_times), dtype=float)
    dB = np.zeros((n_ensemble, n_times), dtype=float)
    dC = np.zeros((n_ensemble, n_times), dtype=float)

    rng = np.random.default_rng(seed)
    sqrt_1mr2 = float(np.sqrt(max(0.0, 1.0 - rho * rho)))

    for i in range(n_ensemble):
        w1 = wiener_path(rng, n_times, dt)
        w2 = wiener_path(rng, n_times, dt)
        w3 = wiener_path(rng, n_times, dt)

        tau1_a = drift + sigma * w1
        tau2_a = drift + sigma * w2
        dA[i, :] = tau1_a - tau2_a

        tau1_b = drift + sigma * w1
        tau2_b = drift + sigma * w1
        dB[i, :] = tau1_b - tau2_b

        tau1_c = drift + sigma * w1
        tau2_c = drift + sigma * (rho * w1 + sqrt_1mr2 * w3)
        dC[i, :] = tau1_c - tau2_c

    return np.stack([dA, dB, dC], axis=0)


def V_and_var_from_delta_tau(delta_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    delta_tau shape (n_ensemble, n_times).

    V(t) = |mean_j exp(i Δτ_j)|, Var(t) = sample variance of Δτ across ensemble (ddof=0).
    """
    z = np.exp(1j * delta_tau)
    V = np.abs(np.mean(z, axis=0))
    var_dt = np.var(delta_tau, axis=0, ddof=0)
    return V.astype(float), var_dt.astype(float)


def _mse_r2_pearson(V: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """MSE and R² treating ``pred = exp(-Var/2)`` as model for ``V``; Pearson corr(V, pred)."""
    V = np.asarray(V, dtype=float)
    pred = np.asarray(pred, dtype=float)
    resid = V - pred
    mse = float(np.mean(resid**2))
    ss_res = float(np.sum(resid**2))
    v_mean = float(np.mean(V))
    ss_tot = float(np.sum((V - v_mean) ** 2))
    if ss_tot < 1e-30:
        r2 = 1.0 if ss_res < 1e-20 else 0.0
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    s_v = float(np.std(V))
    s_p = float(np.std(pred))
    if s_v < 1e-14 and s_p < 1e-14:
        pearson = 1.0
    elif s_v < 1e-14 or s_p < 1e-14:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(V, pred)[0, 1])
    return mse, r2, pearson


def _fit_exp_neg_alpha_t(t: np.ndarray, V: np.ndarray, *, sigma_hint: float) -> float:
    """
    Fit V(t) ≈ exp(-α t) with α ≥ 0. Perfect flat V≡1 → α = 0.
    """
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)
    if t.size < 2:
        return float("nan")
    if np.max(np.abs(V - 1.0)) < 1e-12:
        return 0.0

    def model(tt: np.ndarray, a: float) -> np.ndarray:
        return np.exp(-a * tt)

    p0 = max(float(sigma_hint) ** 2, 1e-6)
    try:
        popt, _ = curve_fit(
            model,
            t,
            V,
            p0=[p0],
            bounds=([0.0], [1e6]),
            maxfev=20_000,
        )
        return float(popt[0])
    except (RuntimeError, ValueError):
        return float("nan")


def run_phase_decoherence_experiment(
    *,
    omega: float = 1.0,
    sigma: float = 0.3,
    t_max: float = 2.0,
    n_times: int = 200,
    n_ensemble: int = 1000,
    rho: float = 0.6,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, float(t_max), int(n_times), dtype=float)
    d_all = simulate_delta_tau_cases(t, omega, sigma, n_ensemble, rho, seed)

    labels = ("caseA", "caseB", "caseC")
    V_out: dict[str, np.ndarray] = {}
    Var_out: dict[str, np.ndarray] = {}
    pred_out: dict[str, np.ndarray] = {}

    for k, name in enumerate(labels):
        V, Var = V_and_var_from_delta_tau(d_all[k])
        V_out[name] = V
        Var_out[name] = Var
        pred_out[name] = np.exp(-0.5 * Var)

    residual: dict[str, np.ndarray] = {}
    for name in labels:
        residual[name] = V_out[name] - pred_out[name]

    df = pd.DataFrame(
        {
            "time": t,
            "V_caseA": V_out["caseA"],
            "V_caseB": V_out["caseB"],
            "V_caseC": V_out["caseC"],
            "Var_caseA": Var_out["caseA"],
            "Var_caseB": Var_out["caseB"],
            "Var_caseC": Var_out["caseC"],
            "residual_caseA": residual["caseA"],
            "residual_caseB": residual["caseB"],
            "residual_caseC": residual["caseC"],
        }
    )
    csv_path = output_dir / "tdf_phase_decoherence_data.csv"
    df.to_csv(csv_path, index=False)

    # --- Quantitative validation: V vs exp(-Var/2) ---
    display_names = ("A", "B", "C")
    metrics_rows: list[dict[str, Any]] = []
    alpha_expected = float(sigma) ** 2

    for name, disp in zip(labels, display_names):
        mse, r2, pear = _mse_r2_pearson(V_out[name], pred_out[name])
        alpha_fit = _fit_exp_neg_alpha_t(t, V_out[name], sigma_hint=float(sigma))
        metrics_rows.append(
            {
                "case": disp,
                "mse": mse,
                "r2": r2,
                "pearson_corr": pear,
                "alpha_fit": alpha_fit,
                "alpha_expected_sigma2": alpha_expected,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)[
        ["case", "mse", "r2", "pearson_corr", "alpha_fit", "alpha_expected_sigma2"]
    ]
    metrics_path = output_dir / "tdf_phase_decoherence_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Plot 1: V(t) all cases
    fig1, ax1 = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax1.plot(t, V_out["caseA"], label="A: independent W₁, W₂", lw=2.0)
    ax1.plot(t, V_out["caseB"], label="B: shared W (Δτ≡0)", lw=2.0)
    ax1.plot(t, V_out["caseC"], label=f"C: partial ρ={rho}", lw=2.0)
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$V(t) = |\langle e^{i\Delta\tau}\rangle|$")
    ax1.set_title("Ensemble phase coherence vs time (TDF phase-variance test)")
    ax1.grid(True, alpha=0.35)
    ax1.legend(frameon=False, fontsize=9)
    fig1.tight_layout()
    p1 = output_dir / "tdf_phase_decoherence_V.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: exp(-Var/2) vs V per case (3 panels)
    fig2, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), dpi=150, sharey=True)
    titles = (
        "A: independent noise",
        "B: shared noise",
        f"C: partial (ρ={rho})",
    )
    for ax, name, title in zip(axes, labels, titles):
        ax.plot(t, V_out[name], "C0", lw=2.0, label=r"$V(t)$ empirical")
        ax.plot(t, pred_out[name], "C1", ls="--", lw=2.0, label=r"$\exp(-\mathrm{Var}(\Delta\tau)/2)$")
        ax.set_xlabel("time")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("value")
    fig2.suptitle(
        r"Gaussian prediction $\exp(-\mathrm{Var}/2)$ vs empirical $V(t)$",
        fontsize=11,
        y=1.02,
    )
    fig2.tight_layout()
    p2 = output_dir / "tdf_phase_decoherence_compare.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: residuals V - exp(-Var/2)
    fig3, ax3 = plt.subplots(figsize=(8.0, 4.2), dpi=150)
    ax3.plot(t, residual["caseA"], lw=2.0, label="Case A: independent")
    ax3.plot(t, residual["caseB"], lw=2.0, label="Case B: shared")
    ax3.plot(t, residual["caseC"], lw=2.0, label="Case C: partial")
    ax3.axhline(0.0, color="k", ls=":", lw=0.9, alpha=0.5)
    ax3.set_xlabel("time")
    ax3.set_ylabel(r"residual $= V(t) - e^{-\mathrm{Var}(\Delta\tau)/2}$")
    ax3.set_title("Residuals: empirical coherence minus Gaussian phase-variance prediction")
    ax3.grid(True, alpha=0.35)
    ax3.legend(frameon=False, fontsize=9)
    fig3.tight_layout()
    p3 = output_dir / "tdf_phase_decoherence_residuals.png"
    fig3.savefig(p3, bbox_inches="tight")
    plt.close(fig3)

    print(f"Wrote {csv_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p3}")
    print(
        f"Parameters: ω={omega}, σ={sigma}, T={t_max}, n_times={n_times}, "
        f"N_ensemble={n_ensemble}, ρ={rho}, seed={seed}"
    )
    print()
    print("--- Validation: V vs exp(-Var(Δτ)/2) ---")
    print(f"Reference decay scale σ² = {alpha_expected:.6g} (compare α_fit for Case A)")
    for row in metrics_rows:
        pc = row["pearson_corr"]
        pc_s = f"{pc:.6f}" if np.isfinite(pc) else "nan"
        af = row["alpha_fit"]
        af_s = f"{af:.6f}" if np.isfinite(af) else "nan"
        print(f"Case {row['case']}:")
        print(f"  MSE: {row['mse']:.6e}")
        print(f"  R²:  {row['r2']:.6f}")
        print(f"  Corr: {pc_s}")
        print(f"  α_fit (V≈exp(-αt)): {af_s}  (expected ≈ σ² = {alpha_expected:.6g} for A; B→0)")

    return {
        "t": t,
        "V": V_out,
        "Var": Var_out,
        "prediction": pred_out,
        "residual": residual,
        "metrics": metrics_df,
        "csv_path": csv_path,
        "metrics_csv_path": metrics_path,
        "figure_V": p1,
        "figure_compare": p2,
        "figure_residuals": p3,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    run_phase_decoherence_experiment()
