"""
Discriminator: TDF-style phase noise vs Ornstein–Uhlenbeck (standard OU) phase noise.

**Model A (TDF phase / Wiener difference):** two legs
``τ_k = ω t + σ_{TDF} W_k`` with independent Wieners; ``Δτ = τ_1 - τ_2``;
ensemble coherence ``V_{TDF}(t) = |⟨e^{iΔτ}⟩|`` (same construction as
``experiments/tdf_phase_decoherence_test`` Case A).

**Model B (OU / Markovian-reverting phase):** two independent scalar OU processes
``dφ = -γ φ dt + σ_{OU} dW``; ``Δφ = φ_1 - φ_2``;
``V_{OU}(t) = |⟨e^{iΔφ}⟩|``.

We compare decay curves, ``d/dt log V``, a single-exponential fit ``V ≈ e^{-α t}``,
and the mean absolute **curvature** ``|d^2 log V / dt^2|``. A **parameter-matched**
pass adjusts ``σ_OU`` so the OU short-time rate
``α_eff = -(d/dt)\\log V|_{t\\approx 0}`` matches the TDF rate, then writes
``tdf_vs_ou_matched_*.png/csv`` including an **exponential deviation score**
``D = (1/T) \\int_0^T |\\log V(t) + α t|\\,dt`` with ``α`` from the exponential fit.

**Reuse:** :mod:`core.tau_model` (linear drift), :mod:`core.noise_models` (Wiener, OU paths).
For **QuTiP** evolution on kets / density matrices, use :mod:`core.evolution` (e.g.
``run_evolution``, ``evolve_open``); Lindblad dephasing benchmarks live in
:mod:`experiments.decoherence` — this file stays in the **classical phase-ensemble**
picture so both models use the same ``V(t)`` observable.

Run::

    python -m experiments.tdf_vs_standard_decoherence
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

from core.noise_models import ou_path, wiener_path
from core.tau_model import linear_tau


def _ensemble_coherence_V(delta: np.ndarray) -> np.ndarray:
    """delta shape (n_ensemble, n_times) → V(t) = |mean_k exp(i Δ_k)|."""
    z = np.exp(1j * np.asarray(delta, dtype=float))
    return np.abs(np.mean(z, axis=0)).astype(float)


def simulate_V_tdf(
    t: np.ndarray,
    omega: float,
    sigma_tdf: float,
    n_ensemble: int,
    seed: int,
) -> np.ndarray:
    n_times = int(t.size)
    dt = float(t[1] - t[0]) if n_times > 1 else 1.0
    drift = linear_tau(t, omega)
    rng = np.random.default_rng(seed)
    d = np.zeros((n_ensemble, n_times), dtype=float)
    for i in range(n_ensemble):
        w1 = wiener_path(rng, n_times, dt)
        w2 = wiener_path(rng, n_times, dt)
        d[i, :] = (drift + sigma_tdf * w1) - (drift + sigma_tdf * w2)
    return _ensemble_coherence_V(d)


def simulate_V_ou(
    t: np.ndarray,
    gamma: float,
    sigma_ou: float,
    n_ensemble: int,
    seed: int,
) -> np.ndarray:
    n_times = int(t.size)
    dt = float(t[1] - t[0]) if n_times > 1 else 1.0
    rng = np.random.default_rng(seed + 10_000)
    d = np.zeros((n_ensemble, n_times), dtype=float)
    for i in range(n_ensemble):
        p1 = ou_path(rng, n_times, dt, gamma, sigma_ou, phi0=0.0)
        p2 = ou_path(rng, n_times, dt, gamma, sigma_ou, phi0=0.0)
        d[i, :] = p1 - p2
    return _ensemble_coherence_V(d)


def estimate_alpha_short_time(
    t: np.ndarray,
    V: np.ndarray,
    *,
    frac: float = 0.05,
    k_min: int = 8,
) -> float:
    """
    Short-time decay rate ``α_eff ≈ -(d/dt) log V`` near ``t=0``.

    Fit ``log V ≈ c + s t`` on the first ``max(k_min, frac·n)`` samples; return ``-s``
    (positive when ``V`` decays).
    """
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)
    n = int(t.size)
    k = max(int(k_min), int(frac * n))
    k = min(max(k, 2), n)
    Vs = np.maximum(V[:k], 1e-15)
    logv = np.log(Vs)
    slope = float(np.polyfit(t[:k], logv, 1)[0])
    return float(max(-slope, 0.0))


def _exp_deviation_score(t: np.ndarray, logv: np.ndarray, alpha: float) -> float:
    r"""``D = (1/T) \int |\log V(t) + \alpha t|\,dt`` with ``T = t_{\max} - t_{\min}``."""
    if not np.isfinite(alpha):
        return float("nan")
    t = np.asarray(t, dtype=float)
    logv = np.asarray(logv, dtype=float)
    Tspan = float(t[-1] - t[0])
    if Tspan <= 0.0:
        return float("nan")
    integrand = np.abs(logv + alpha * t)
    return float(np.trapz(integrand, t) / Tspan)


def _fit_exp_decay_alpha(t: np.ndarray, V: np.ndarray, *, p0: float) -> float:
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)

    def model(tt: np.ndarray, a: float) -> np.ndarray:
        return np.exp(-a * tt)

    if np.max(np.abs(V - 1.0)) < 1e-12:
        return 0.0
    try:
        popt, _ = curve_fit(
            model,
            t,
            V,
            p0=[max(p0, 1e-8)],
            bounds=([0.0], [1e6]),
            maxfev=20_000,
        )
        return float(popt[0])
    except (RuntimeError, ValueError):
        return float("nan")


def _metrics_curve(
    t: np.ndarray,
    V: np.ndarray,
    *,
    alpha_p0: float,
    include_exp_deviation: bool = False,
) -> dict[str, Any]:
    dt = float(t[1] - t[0]) if t.size > 1 else 1.0
    V = np.asarray(V, dtype=float)
    Vs = np.maximum(V, 1e-15)
    logv = np.log(Vs)
    d1 = np.gradient(logv, dt)
    d2 = np.gradient(d1, dt)
    curvature_mean = float(np.mean(np.abs(d2)))

    alpha_fit = _fit_exp_decay_alpha(t, V, p0=alpha_p0)
    pred = np.exp(-alpha_fit * t) if np.isfinite(alpha_fit) else np.ones_like(t)
    mse_exp_fit = float(np.mean((V - pred) ** 2))

    out: dict[str, Any] = {
        "alpha_fit": alpha_fit,
        "alpha_matched": alpha_fit,
        "curvature_mean": curvature_mean,
        "mse_exp_fit": mse_exp_fit,
        "logV": logv,
        "d_logV_dt": d1,
    }
    if include_exp_deviation:
        out["exp_deviation_score"] = _exp_deviation_score(t, logv, alpha_fit)
    return out


def _calibrate_sigma_ou_to_match_alpha_short(
    t: np.ndarray,
    *,
    alpha_target: float,
    gamma: float,
    n_ensemble: int,
    seed: int,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 8.0,
) -> float:
    """
    Find ``σ_OU`` such that OU short-time ``α_eff`` matches ``alpha_target``.

    Short-time ``α_eff(σ)`` is increasing in ``σ`` for the OU phase-difference model;
    we expand the upper bracket if needed, then use ``brentq``.
    """

    def alpha_for_sigma(sig: float) -> float:
        V = simulate_V_ou(t, gamma, float(sig), n_ensemble, seed)
        return estimate_alpha_short_time(t, V)

    sig_hi = float(sigma_hi)
    a_lo = alpha_for_sigma(sigma_lo)
    a_hi = alpha_for_sigma(sig_hi)
    for _ in range(25):
        if a_hi >= alpha_target - 1e-10:
            break
        sig_hi *= 1.4
        a_hi = alpha_for_sigma(sig_hi)
        if sig_hi > 200.0:
            break

    if alpha_target <= a_lo + 1e-12:
        return float(sigma_lo)
    if alpha_target >= a_hi - 1e-12:
        return float(sig_hi)
    if not (a_lo < alpha_target < a_hi):
        return float("nan")

    def f(sig: float) -> float:
        return alpha_for_sigma(sig) - alpha_target

    try:
        return float(brentq(f, float(sigma_lo), float(sig_hi), xtol=1e-5, maxiter=100))
    except ValueError:
        return float("nan")


def run_tdf_vs_standard_decoherence(
    *,
    omega: float = 1.0,
    sigma_tdf: float = 0.3,
    gamma: float = 1.0,
    sigma_ou: float = 0.3,
    t_max: float = 2.0,
    n_times: int = 400,
    n_ensemble: int = 2000,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, float(t_max), int(n_times), dtype=float)
    dt = float(t[1] - t[0]) if n_times > 1 else 1.0

    V_tdf = simulate_V_tdf(t, omega, sigma_tdf, n_ensemble, seed)
    V_ou = simulate_V_ou(t, gamma, sigma_ou, n_ensemble, seed)

    # Rough scale for exponential fit initial guess (TDF short-time ~ σ²)
    alpha_guess_tdf = float(sigma_tdf) ** 2
    alpha_guess_ou = float(sigma_ou**2 / max(gamma, 1e-12))

    m_tdf = _metrics_curve(t, V_tdf, alpha_p0=alpha_guess_tdf)
    m_ou = _metrics_curve(t, V_ou, alpha_p0=alpha_guess_ou)

    alpha_tdf_short = estimate_alpha_short_time(t, V_tdf)
    alpha_ou_short_before = estimate_alpha_short_time(t, V_ou)

    rows = [
        {
            "model": "TDF",
            "alpha_fit": m_tdf["alpha_fit"],
            "curvature_mean": m_tdf["curvature_mean"],
            "mse_exp_fit": m_tdf["mse_exp_fit"],
        },
        {
            "model": "OU",
            "alpha_fit": m_ou["alpha_fit"],
            "curvature_mean": m_ou["curvature_mean"],
            "mse_exp_fit": m_ou["mse_exp_fit"],
        },
    ]
    metrics_path = output_dir / "tdf_vs_ou_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    # --- Parameter-matched OU (σ_OU tuned to match short-time TDF rate) ---
    sigma_ou_matched = _calibrate_sigma_ou_to_match_alpha_short(
        t,
        alpha_target=alpha_tdf_short,
        gamma=gamma,
        n_ensemble=n_ensemble,
        seed=seed,
    )
    if not np.isfinite(sigma_ou_matched):
        V_ou_m = simulate_V_ou(t, gamma, sigma_ou, n_ensemble, seed)
        sigma_ou_matched = float(sigma_ou)
    else:
        V_ou_m = simulate_V_ou(t, gamma, sigma_ou_matched, n_ensemble, seed)

    alpha_ou_short_after = estimate_alpha_short_time(t, V_ou_m)

    m_tdf_m = _metrics_curve(
        t, V_tdf, alpha_p0=alpha_guess_tdf, include_exp_deviation=True
    )
    m_ou_m = _metrics_curve(
        t,
        V_ou_m,
        alpha_p0=float(sigma_ou_matched**2 / max(gamma, 1e-12)),
        include_exp_deviation=True,
    )

    rows_m = [
        {
            "model": "TDF",
            "alpha_matched": m_tdf_m["alpha_matched"],
            "curvature_mean": m_tdf_m["curvature_mean"],
            "mse_exp_fit": m_tdf_m["mse_exp_fit"],
            "exp_deviation_score": m_tdf_m["exp_deviation_score"],
        },
        {
            "model": "OU",
            "alpha_matched": m_ou_m["alpha_matched"],
            "curvature_mean": m_ou_m["curvature_mean"],
            "mse_exp_fit": m_ou_m["mse_exp_fit"],
            "exp_deviation_score": m_ou_m["exp_deviation_score"],
        },
    ]
    matched_csv = output_dir / "tdf_vs_ou_matched_metrics.csv"
    pd.DataFrame(rows_m).to_csv(matched_csv, index=False)

    # Plot V
    fig1, ax1 = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax1.plot(t, V_tdf, lw=2.0, label=r"$V_{TDF}(t)$ (Wiener phase diff.)")
    ax1.plot(t, V_ou, lw=2.0, label=r"$V_{OU}(t)$ (OU phase diff.)")
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$V(t) = |\langle e^{i\Delta\theta}\rangle|$")
    ax1.set_title("Ensemble coherence: TDF-style vs OU phase noise")
    ax1.grid(True, alpha=0.35)
    ax1.legend(frameon=False, fontsize=9)
    fig1.tight_layout()
    p1 = output_dir / "tdf_vs_ou_V.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)

    # Plot log V and decay rate
    fig2, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=150, sharex=True)
    ax_a.plot(t, m_tdf["logV"], lw=2.0, label=r"$\log V_{TDF}$")
    ax_a.plot(t, m_ou["logV"], lw=2.0, label=r"$\log V_{OU}$")
    ax_a.set_ylabel(r"$\log V(t)$")
    ax_a.grid(True, alpha=0.35)
    ax_a.legend(frameon=False, fontsize=9)
    ax_a.set_title("Log coherence and instantaneous decay rate")

    ax_b.plot(t, m_tdf["d_logV_dt"], lw=2.0, label=r"$\frac{d}{dt}\log V_{TDF}$")
    ax_b.plot(t, m_ou["d_logV_dt"], lw=2.0, label=r"$\frac{d}{dt}\log V_{OU}$")
    ax_b.set_xlabel("time")
    ax_b.set_ylabel("decay rate")
    ax_b.grid(True, alpha=0.35)
    ax_b.legend(frameon=False, fontsize=9)
    fig2.tight_layout()
    p2 = output_dir / "tdf_vs_ou_logV.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)

    # Matched plots: TDF vs calibrated OU
    fig3, ax3 = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax3.plot(t, V_tdf, lw=2.0, label=r"$V_{TDF}$")
    sig_lbl = (
        f"{sigma_ou_matched:.4g}"
        if np.isfinite(sigma_ou_matched)
        else "nan"
    )
    ax3.plot(
        t,
        V_ou_m,
        lw=2.0,
        label=rf"$V_{{OU}}$ (matched $\sigma_{{OU}}={sig_lbl}$)",
    )
    ax3.set_xlabel("time")
    ax3.set_ylabel(r"$V(t)$")
    ax3.set_title("Short-time decay matched: TDF vs OU")
    ax3.grid(True, alpha=0.35)
    ax3.legend(frameon=False, fontsize=9)
    fig3.tight_layout()
    p3 = output_dir / "tdf_vs_ou_matched_V.png"
    fig3.savefig(p3, bbox_inches="tight")
    plt.close(fig3)

    fig4, (ax_ma, ax_mb) = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=150, sharex=True)
    ax_ma.plot(t, m_tdf_m["logV"], lw=2.0, label=r"$\log V_{TDF}$")
    ax_ma.plot(t, m_ou_m["logV"], lw=2.0, label=r"$\log V_{OU}$ (matched)")
    ax_ma.set_ylabel(r"$\log V(t)$")
    ax_ma.grid(True, alpha=0.35)
    ax_ma.legend(frameon=False, fontsize=9)
    ax_ma.set_title("Matched parameters: log coherence and decay rate")

    ax_mb.plot(t, m_tdf_m["d_logV_dt"], lw=2.0, label=r"$\frac{d}{dt}\log V_{TDF}$")
    ax_mb.plot(t, m_ou_m["d_logV_dt"], lw=2.0, label=r"$\frac{d}{dt}\log V_{OU}$")
    ax_mb.set_xlabel("time")
    ax_mb.set_ylabel("decay rate")
    ax_mb.grid(True, alpha=0.35)
    ax_mb.legend(frameon=False, fontsize=9)
    fig4.tight_layout()
    p4 = output_dir / "tdf_vs_ou_matched_logV.png"
    fig4.savefig(p4, bbox_inches="tight")
    plt.close(fig4)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {matched_csv}")
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p3}")
    print(f"Wrote {p4}")
    print(
        f"Parameters: ω={omega}, σ_TDF={sigma_tdf}, γ={gamma}, σ_OU(initial)={sigma_ou}, "
        f"T={t_max}, n_times={n_times}, N={n_ensemble}, seed={seed}, dt={dt:.6g}"
    )
    print()
    print("========== Before matching ==========")
    print(
        f"  Short-time α_eff (TDF): {alpha_tdf_short:.6g}\n"
        f"  Short-time α_eff (OU):  {alpha_ou_short_before:.6g}"
    )
    print("  Full-window metrics:")
    for row in rows:
        print(
            f"    {row['model']}:  α_fit={row['alpha_fit']:.6g}  "
            f"curvature_mean={row['curvature_mean']:.6g}  "
            f"mse_exp_fit={row['mse_exp_fit']:.6e}"
        )
    print()
    print("========== After matching ==========")
    print(
        f"  Calibrated σ_OU: {sigma_ou_matched:.6g}  (target α_short ≈ {alpha_tdf_short:.6g})\n"
        f"  Short-time α_eff (OU): {alpha_ou_short_after:.6g}"
    )
    print("  Matched-window metrics (α from exp fit → exp_deviation_score):")
    for row in rows_m:
        print(
            f"    {row['model']}:  alpha_matched={row['alpha_matched']:.6g}  "
            f"curvature_mean={row['curvature_mean']:.6g}  "
            f"mse_exp_fit={row['mse_exp_fit']:.6e}  "
            f"D={row['exp_deviation_score']:.6e}"
        )
    print()
    print(
        "D = (1/T)∫|log V + α t|dt with α = exponential-fit α_matched. "
        "Pure V=exp(−αt) ⇒ D=0."
    )

    return {
        "t": t,
        "V_tdf": V_tdf,
        "V_ou": V_ou,
        "V_ou_matched": V_ou_m,
        "sigma_ou_matched": sigma_ou_matched,
        "alpha_tdf_short": alpha_tdf_short,
        "alpha_ou_short_before": alpha_ou_short_before,
        "alpha_ou_short_after": alpha_ou_short_after,
        "metrics_tdf": m_tdf,
        "metrics_ou": m_ou,
        "metrics_tdf_matched": m_tdf_m,
        "metrics_ou_matched": m_ou_m,
        "metrics_csv": metrics_path,
        "matched_metrics_csv": matched_csv,
        "figure_V": p1,
        "figure_logV": p2,
        "figure_matched_V": p3,
        "figure_matched_logV": p4,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    run_tdf_vs_standard_decoherence()
