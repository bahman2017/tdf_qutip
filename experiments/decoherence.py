"""
Stochastic τ noise vs standard Lindblad dephasing.

A single stochastic-τ trajectory is **unitary** and does not decohere by itself.
**Decoherence-like decay** appears only after **ensemble averaging** ρ = (1/N)Σ|ψᵢ⟩⟨ψᵢ|.
The question addressed here is whether that emergent curve can be **mimicked** by
Markovian **Lindblad pure dephasing**, or whether **residual non-Lindbladian** structure remains.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from core.evolution import create_superposition_state, evolve_open, run_evolution
from core.hamiltonians import constant_hamiltonian, tau_to_hamiltonian
from core.tau_model import stochastic_tau, structured_stochastic_tau

NOISE_STRENGTHS_DEFAULT = (0.1, 0.5, 1.0, 2.0)


def transverse_coherence(sig_x: np.ndarray, sig_y: np.ndarray) -> np.ndarray:
    """√(⟨σ_x⟩² + ⟨σ_y⟩²) — in-plane Bloch length (1 for |+⟩ at t=0 in σ_z basis)."""
    sx = np.asarray(sig_x, dtype=float)
    sy = np.asarray(sig_y, dtype=float)
    return np.sqrt(sx**2 + sy**2)


def _stochastic_trajectory_states(
    t: np.ndarray,
    omega: float,
    noise_strength: float,
    psi0: qt.Qobj,
) -> list[qt.Qobj]:
    tau = stochastic_tau(t, omega, noise_strength)
    H = tau_to_hamiltonian(tau, t)
    res = run_evolution(H, psi0, t)
    return list(res.states)


def _ensemble_mixed_transverse_coherence(states_per_time: list[list[qt.Qobj]]) -> np.ndarray:
    """
    ρ(t) = (1/N) Σ_i |ψ_i⟩⟨ψ_i|, then C(t) = √(⟨σ_x⟩_ρ² + ⟨σ_y⟩_ρ²).
    """
    n_times = len(states_per_time)
    out = np.zeros(n_times, dtype=float)
    sx_op = qt.sigmax()
    sy_op = qt.sigmay()
    for k in range(n_times):
        kets = states_per_time[k]
        rho = kets[0] * kets[0].dag()
        for psi in kets[1:]:
            rho += psi * psi.dag()
        rho /= len(kets)
        sx = float(np.real(qt.expect(sx_op, rho)))
        sy = float(np.real(qt.expect(sy_op, rho)))
        out[k] = float(np.sqrt(sx**2 + sy**2))
    return out


def collect_tdf_ensemble_coherence_curves(
    t: np.ndarray,
    omega: float,
    noise_strengths: tuple[float, ...],
    n_ensemble: int,
    ensemble_base_seed: int,
    psi0: qt.Qobj,
) -> dict[float, np.ndarray]:
    """TDF transverse coherence C_tdf(t) for each noise strength."""
    curves: dict[float, np.ndarray] = {}
    for j, sigma in enumerate(noise_strengths):
        bucket: list[list[qt.Qobj]] = [[] for _ in range(len(t))]
        for i in range(n_ensemble):
            np.random.seed(ensemble_base_seed + j * 10_000 + i)
            seq = _stochastic_trajectory_states(t, omega, sigma, psi0)
            for k, psi in enumerate(seq):
                bucket[k].append(psi)
        curves[float(sigma)] = _ensemble_mixed_transverse_coherence(bucket)
    return curves


def _structured_stochastic_trajectory_states(
    t: np.ndarray,
    omega: float,
    freq: float,
    noise_strength: float,
    psi0: qt.Qobj,
) -> list[qt.Qobj]:
    tau = structured_stochastic_tau(
        t, omega=omega, freq=freq, noise_strength=noise_strength
    )
    H = tau_to_hamiltonian(tau, t)
    res = run_evolution(H, psi0, t)
    return list(res.states)


def collect_structured_stochastic_ensemble_coherence_curves(
    t: np.ndarray,
    omega: float,
    freq: float,
    noise_strengths: tuple[float, ...],
    n_ensemble: int,
    ensemble_base_seed: int,
    psi0: qt.Qobj,
) -> dict[float, np.ndarray]:
    """
    Ensemble-ρ transverse coherence for τ(t) = ω t + sin(ν t) + ξ(t) with Gaussian ξ.

    ``noise_strength`` in the outer loop is the standard deviation of ξ at each time.
    """
    curves: dict[float, np.ndarray] = {}
    for j, sigma in enumerate(noise_strengths):
        bucket: list[list[qt.Qobj]] = [[] for _ in range(len(t))]
        for i in range(n_ensemble):
            np.random.seed(ensemble_base_seed + j * 10_000 + i)
            seq = _structured_stochastic_trajectory_states(
                t, omega, freq, sigma, psi0
            )
            for k, psi in enumerate(seq):
                bucket[k].append(psi)
        curves[float(sigma)] = _ensemble_mixed_transverse_coherence(bucket)
    return curves


def compare_structured_stochastic_tdf_vs_fitted_lindblad(
    t: np.ndarray | None = None,
    noise_strengths: tuple[float, ...] | None = None,
    n_ensemble: int = 40,
    omega: float = 1.0,
    freq: float = 3.0,
    initial_state: str = "plus",
    fit_fraction: float = 0.2,
    gamma_grid: np.ndarray | None = None,
    ensemble_base_seed: int = 42,
) -> dict[str, Any]:
    """
    Same Lindblad comparison as :func:`compare_tdf_vs_fitted_lindblad`, but ensemble τ is
    :func:`structured_stochastic_tau` (drift + sin(ν t) + noise) instead of ``stochastic_tau``.

    Does not write CSV or figures; returns ``rows`` and curves for discrimination pipelines.
    """
    if t is None:
        t = np.linspace(0.0, 10.0, 300)
    if noise_strengths is None:
        noise_strengths = NOISE_STRENGTHS_DEFAULT
    if gamma_grid is None:
        gamma_grid = np.linspace(0.001, 1.0, 200)

    if initial_state.lower() != "plus":
        raise ValueError('initial_state must be "plus" for this experiment')
    psi0 = create_superposition_state()
    n_initial = max(3, int(fit_fraction * len(t)))

    curves = collect_structured_stochastic_ensemble_coherence_curves(
        t, omega, freq, noise_strengths, n_ensemble, ensemble_base_seed, psi0
    )

    rows: list[dict[str, Any]] = []
    lindblad_by_sigma: dict[float, np.ndarray] = {}

    for sigma in noise_strengths:
        c_tdf = curves[float(sigma)]
        best_gamma, _ = fit_best_gamma(
            t, omega, psi0, c_tdf, n_initial, gamma_grid
        )
        c_lind = lindblad_coherence_curve(t, omega, best_gamma, psi0)
        lindblad_by_sigma[float(sigma)] = c_lind

        residual = c_tdf - c_lind
        rmse_full = float(np.sqrt(np.mean(residual**2)))
        rmse_initial = float(np.sqrt(np.mean((c_tdf[:n_initial] - c_lind[:n_initial]) ** 2)))
        long_time_difference = float(
            abs(np.mean(c_tdf[-10:]) - np.mean(c_lind[-10:]))
        )
        oscillation_residual = float(np.std(residual))
        residual_peak_to_peak = float(np.max(residual) - np.min(residual))

        slope_tdf = compute_decay_slope(t, c_tdf, n_initial)
        slope_lb = compute_decay_slope(t, c_lind, n_initial)

        rows.append(
            {
                "noise_strength": sigma,
                "best_gamma": best_gamma,
                "rmse_full": rmse_full,
                "rmse_initial": rmse_initial,
                "long_time_difference": long_time_difference,
                "oscillation_residual": oscillation_residual,
                "residual_peak_to_peak": residual_peak_to_peak,
                "tdf_final_coherence": float(np.mean(c_tdf[-10:])),
                "lindblad_final_coherence": float(np.mean(c_lind[-10:])),
                "initial_decay_slope_tdf": slope_tdf,
                "initial_decay_slope_lindblad": slope_lb,
                "n_ensemble": n_ensemble,
            }
        )

    return {
        "t": t,
        "noise_strengths": noise_strengths,
        "omega": omega,
        "freq": freq,
        "n_ensemble": n_ensemble,
        "n_initial": n_initial,
        "tdf_curves": curves,
        "lindblad_curves": lindblad_by_sigma,
        "rows": rows,
        "gamma_grid": gamma_grid,
    }


def lindblad_coherence_curve(
    t: np.ndarray,
    omega: float,
    gamma: float,
    psi0: qt.Qobj,
) -> np.ndarray:
    """
    Pure dephasing: H = ω σ_z, c_ops = [√γ σ_z]; C(t) = √(⟨σ_x⟩²+⟨σ_y⟩²) from mesolve.
    """
    rho0 = psi0 * psi0.dag()
    H = constant_hamiltonian(omega)
    c_ops = [np.sqrt(float(max(gamma, 1e-15))) * qt.sigmaz()]
    res = evolve_open(H, rho0, t, c_ops, e_ops=[qt.sigmax(), qt.sigmay()])
    return transverse_coherence(res.expect[0], res.expect[1])


def compute_decay_slope(t: np.ndarray, y: np.ndarray, n_fit: int) -> float:
    """Linear least-squares slope dy/dt using the first ``n_fit`` samples."""
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    k = max(3, min(int(n_fit), len(y_arr)))
    return float(np.polyfit(t_arr[:k], y_arr[:k], 1)[0])


def fit_best_gamma(
    t: np.ndarray,
    omega: float,
    psi0: qt.Qobj,
    c_tdf_fit_target: np.ndarray,
    n_initial: int,
    gamma_grid: np.ndarray,
) -> tuple[float, float]:
    """
    Grid search: minimize RMSE between C_tdf and C_Lindblad on indices ``0:n_initial``.

    Returns
    -------
    best_gamma, best_rmse_initial
    """
    t_arr = np.asarray(t, dtype=float)
    y_tar = np.asarray(c_tdf_fit_target, dtype=float)
    k = max(3, min(n_initial, len(y_tar)))
    best_g = float(gamma_grid[0])
    best_e = np.inf
    for g in gamma_grid:
        cl = lindblad_coherence_curve(t_arr, omega, float(g), psi0)
        e = y_tar[:k] - cl[:k]
        rmse = float(np.sqrt(np.mean(e**2)))
        if rmse < best_e:
            best_e = rmse
            best_g = float(g)
    return best_g, best_e


def _fit_exponential_envelope(
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """Envelope A exp(−λt)+y∞ on Gaussian-smoothed y (legacy overlay plot)."""
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    sigma_pix = max(2.0, len(y_arr) / 50.0)
    y_smooth = gaussian_filter1d(y_arr, sigma=sigma_pix, mode="nearest")

    def model(tt: np.ndarray, A: float, lam: float, y_inf: float) -> np.ndarray:
        return A * np.exp(-lam * tt) + y_inf

    p0 = (
        float(np.clip(y_smooth[0] - y_smooth[-1], 0.05, 2.0)),
        0.08,
        float(np.mean(y_smooth[-max(8, len(y_smooth) // 15) :])),
    )
    try:
        popt, _ = curve_fit(
            model,
            t_arr,
            y_smooth,
            p0=p0,
            bounds=([0.0, 0.0, -0.5], [3.0, 50.0, 1.5]),
            maxfev=20000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError):
        return float("nan"), float("nan"), float("nan")


def _sigma_filename_tag(sigma: float) -> str:
    return str(sigma).replace(".", "p")


def compare_tdf_vs_fitted_lindblad(
    t: np.ndarray | None = None,
    noise_strengths: tuple[float, ...] | None = None,
    n_ensemble: int = 40,
    omega: float = 1.0,
    initial_state: str = "plus",
    smoothing_sigma: float = 2.0,
    fit_fraction: float = 0.2,
    gamma_grid: np.ndarray | None = None,
    output_dir: Path | str | None = None,
    plot: bool = True,
    show: bool = False,
    ensemble_base_seed: int = 42,
) -> dict[str, Any]:
    """
    Compare TDF ensemble decoherence to a best-fit Lindblad pure-dephasing model.

    γ is chosen by **grid search** to minimize RMSE between **raw** ``C_tdf[:k]`` and
    ``C_Lindblad[:k]`` with ``k = int(fit_fraction * len(t))``. All reported RMSE and
    residuals also use the **raw** ensemble curve. If ``smoothing_sigma > 0``, a smoothed
    TDF trace is drawn faintly on per-σ plots as a visual guide only (not used in the fit).

    Writes ``decoherence_comparison.csv``, ``decoherence_comparison_summary.png``, and
    per-σ figures ``decoherence_compare_sigma_<tag>.png``.
    """
    if t is None:
        t = np.linspace(0.0, 10.0, 300)
    if noise_strengths is None:
        noise_strengths = NOISE_STRENGTHS_DEFAULT
    if gamma_grid is None:
        gamma_grid = np.linspace(0.001, 1.0, 200)

    if output_dir is None:
        output_dir = Path("outputs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if initial_state.lower() != "plus":
        raise ValueError('initial_state must be "plus" for this experiment')
    psi0 = create_superposition_state()

    n_initial = max(3, int(fit_fraction * len(t)))

    curves = collect_tdf_ensemble_coherence_curves(
        t, omega, noise_strengths, n_ensemble, ensemble_base_seed, psi0
    )

    rows: list[dict[str, Any]] = []
    lindblad_by_sigma: dict[float, np.ndarray] = {}

    for sigma in noise_strengths:
        c_tdf = curves[float(sigma)]

        best_gamma, _ = fit_best_gamma(
            t, omega, psi0, c_tdf, n_initial, gamma_grid
        )
        c_lind = lindblad_coherence_curve(t, omega, best_gamma, psi0)
        lindblad_by_sigma[float(sigma)] = c_lind

        residual = c_tdf - c_lind
        rmse_full = float(np.sqrt(np.mean(residual**2)))
        rmse_initial = float(np.sqrt(np.mean((c_tdf[:n_initial] - c_lind[:n_initial]) ** 2)))
        long_time_difference = float(
            abs(np.mean(c_tdf[-10:]) - np.mean(c_lind[-10:]))
        )
        oscillation_residual = float(np.std(residual))
        residual_peak_to_peak = float(np.max(residual) - np.min(residual))

        slope_tdf = compute_decay_slope(t, c_tdf, n_initial)
        slope_lb = compute_decay_slope(t, c_lind, n_initial)

        rows.append(
            {
                "noise_strength": sigma,
                "best_gamma": best_gamma,
                "rmse_full": rmse_full,
                "rmse_initial": rmse_initial,
                "long_time_difference": long_time_difference,
                "oscillation_residual": oscillation_residual,
                "residual_peak_to_peak": residual_peak_to_peak,
                "tdf_final_coherence": float(np.mean(c_tdf[-10:])),
                "lindblad_final_coherence": float(np.mean(c_lind[-10:])),
                "initial_decay_slope_tdf": slope_tdf,
                "initial_decay_slope_lindblad": slope_lb,
                "n_ensemble": n_ensemble,
            }
        )

    csv_path = output_dir / "decoherence_comparison.csv"
    fieldnames = [
        "noise_strength",
        "best_gamma",
        "rmse_full",
        "rmse_initial",
        "long_time_difference",
        "oscillation_residual",
        "residual_peak_to_peak",
        "tdf_final_coherence",
        "lindblad_final_coherence",
        "initial_decay_slope_tdf",
        "initial_decay_slope_lindblad",
        "n_ensemble",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {csv_path}")

    sigmas = [float(r["noise_strength"]) for r in rows]
    gammas = [r["best_gamma"] for r in rows]
    rmse_f = [r["rmse_full"] for r in rows]
    rmse_i = [r["rmse_initial"] for r in rows]
    ltd = [r["long_time_difference"] for r in rows]
    osc = [r["oscillation_residual"] for r in rows]

    if plot:
        for sigma in noise_strengths:
            c_tdf = curves[float(sigma)]
            c_lind = lindblad_by_sigma[float(sigma)]
            best_g = float(
                next(
                    r["best_gamma"]
                    for r in rows
                    if float(r["noise_strength"]) == float(sigma)
                )
            )
            res = c_tdf - c_lind

            fig_p, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
            )
            ax_top.plot(t, c_tdf, label=r"TDF $C(t)$ (ensemble $\rho$)", color="C0")
            if smoothing_sigma > 0:
                ax_top.plot(
                    t,
                    gaussian_filter1d(
                        c_tdf, sigma=float(smoothing_sigma), mode="nearest"
                    ),
                    alpha=0.45,
                    color="C0",
                    linestyle=":",
                    label="TDF (Gaussian smoothed, visual)",
                )
            ax_top.plot(
                t,
                c_lind,
                "--",
                label=rf"best-fit Lindblad ($\gamma^* \approx {best_g:.4f}$)",
                color="C3",
                linewidth=2.0,
            )
            ax_top.axvspan(
                float(t[0]),
                float(t[min(n_initial, len(t)) - 1]),
                alpha=0.12,
                color="gray",
                label="fit window",
            )
            ax_top.set_ylabel(r"$C(t)$")
            ax_top.legend(loc="upper right", fontsize=8)
            ax_top.grid(True, alpha=0.3)
            ax_top.set_title(
                rf"TDF vs best-fit Lindblad decoherence ($\sigma={sigma}$, $\gamma^*={best_g:.4f}$)"
            )

            ax_bot.plot(t, res, color="C2", label=r"$C_\mathrm{TDF}-C_\mathrm{Lind}$")
            ax_bot.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)
            ax_bot.set_xlabel("t")
            ax_bot.set_ylabel("residual")
            ax_bot.legend(loc="upper right", fontsize=8)
            ax_bot.grid(True, alpha=0.3)
            fig_p.tight_layout()
            p_sigma = output_dir / f"decoherence_compare_sigma_{_sigma_filename_tag(sigma)}.png"
            fig_p.savefig(p_sigma, dpi=150)
            plt.close(fig_p)
            print(f"Saved {p_sigma}")

        fig_s, axes = plt.subplots(2, 2, figsize=(10, 7))
        ax_bg, ax_rmse, ax_lt, ax_os = axes.ravel()
        ax_bg.plot(sigmas, gammas, "o-", color="C0")
        ax_bg.set_xlabel(r"$\sigma$ (τ noise)")
        ax_bg.set_ylabel(r"best $\gamma$")
        ax_bg.set_title("Fitted Lindblad rate")
        ax_bg.grid(True, alpha=0.3)

        ax_rmse.plot(sigmas, rmse_f, "s-", color="C1", label="RMSE full")
        ax_rmse.plot(sigmas, rmse_i, "^--", color="C2", label="RMSE initial window")
        ax_rmse.set_xlabel(r"$\sigma$")
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.legend(fontsize=8)
        ax_rmse.set_title("Mismatch vs noise strength")
        ax_rmse.grid(True, alpha=0.3)

        ax_lt.plot(sigmas, ltd, "D-", color="C3")
        ax_lt.set_xlabel(r"$\sigma$")
        ax_lt.set_ylabel(r"$|\langle C_\mathrm{tail}\rangle_\mathrm{TDF}-\mathrm{Lind}|$")
        ax_lt.set_title("Long-time difference (last 10 pts)")
        ax_lt.grid(True, alpha=0.3)

        ax_os.plot(sigmas, osc, "v-", color="C4")
        ax_os.set_xlabel(r"$\sigma$")
        ax_os.set_ylabel(r"std(residual)")
        ax_os.set_title("Oscillation residual")
        ax_os.grid(True, alpha=0.3)

        fig_s.suptitle("TDF vs fitted Lindblad — summary", fontsize=12)
        fig_s.tight_layout()
        p_sum = output_dir / "decoherence_comparison_summary.png"
        fig_s.savefig(p_sum, dpi=150)
        print(f"Saved {p_sum}")
        if show:
            plt.show()
        else:
            plt.close(fig_s)

    med_i = float(np.median(rmse_i))
    med_f = float(np.median(rmse_f))
    med_os = float(np.median(osc))
    med_lt = float(np.median(ltd))
    print()
    print("--- TDF vs fitted Lindblad (interpretation) ---")
    print(
        f"Initial-window RMSE (median over σ) ≈ {med_i:.4f}; full-time RMSE median ≈ {med_f:.4f}."
    )
    if med_f > 2.5 * med_i + 1e-3:
        print(
            "→ Full-curve mismatch exceeds the early window: Lindblad with a single γ "
            "often tracks the **start** better than the **bulk/long-time** shape."
        )
    else:
        print(
            "→ Full RMSE is not dramatically larger than the initial-window RMSE on this grid."
        )
    print(
        f"Residual std (median) ≈ {med_os:.4f}; long-time |Δ| (median) ≈ {med_lt:.4f}."
    )
    if med_os > 0.08 or med_lt > 0.12:
        print(
            "→ Substantial **residual structure** and/or **tail mismatch** suggests the "
            "ensemble-TDF curve is **not** fully reducible to one Markovian dephasing channel."
        )
    else:
        print(
            "→ Residuals are relatively small here; increase σ, time window, or N_ensemble "
            "to stress-test non-Lindbladian effects."
        )
    print("---------------------------------------------")

    return {
        "t": t,
        "noise_strengths": noise_strengths,
        "omega": omega,
        "n_ensemble": n_ensemble,
        "n_initial": n_initial,
        "tdf_curves": curves,
        "lindblad_curves": lindblad_by_sigma,
        "rows": rows,
        "csv_path": csv_path,
        "gamma_grid": gamma_grid,
    }


def run_stochastic_tau_decoherence_experiment(
    t: np.ndarray | None = None,
    *,
    omega: float = 1.0,
    noise_strengths: tuple[float, ...] = NOISE_STRENGTHS_DEFAULT,
    n_ensemble: int = 40,
    ensemble_base_seed: int = 42,
    lindblad_gamma: float = 0.08,
    show: bool = False,
) -> dict[str, Any]:
    """
    Overlay TDF ensemble-ρ coherence curves with a **fixed-γ** Lindblad reference and
    exponential-envelope guides. For **fitted-γ** comparison and CSV, use
    :func:`compare_tdf_vs_fitted_lindblad`.
    """
    if t is None:
        t = np.linspace(0.0, 8.0, 400)

    psi0 = create_superposition_state()
    curves = collect_tdf_ensemble_coherence_curves(
        t, omega, noise_strengths, n_ensemble, ensemble_base_seed, psi0
    )
    decay_rates = {
        float(s): _fit_exponential_envelope(t, curves[float(s)]) for s in noise_strengths
    }

    lindblad_coh = lindblad_coherence_curve(t, omega, lindblad_gamma, psi0)

    fig, ax = plt.subplots(figsize=(9, 5))
    for sigma in noise_strengths:
        y = curves[float(sigma)]
        A, lam, y_inf = decay_rates[float(sigma)]
        lam_s = f"{lam:.3f}" if not np.isnan(lam) else "n/a"
        label = f"τ-noise σ={sigma} (⟨ρ⟩ transverse, λ_fit≈{lam_s})"
        ax.plot(t, y, label=label, alpha=0.9)
        if not np.isnan(lam):
            ax.plot(
                t,
                A * np.exp(-lam * t) + y_inf,
                linestyle="--",
                alpha=0.45,
                color=ax.lines[-1].get_color(),
            )

    ax.plot(
        t,
        lindblad_coh,
        color="k",
        linewidth=2.0,
        linestyle="-.",
        label=f"Lindblad γ={lindblad_gamma} (fixed ref.)",
    )
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\sqrt{\langle\sigma_x\rangle_\rho^2 + \langle\sigma_y\rangle_\rho^2}$")
    ax.set_title(
        r"Stochastic τ: transverse length of ensemble-averaged $\rho$ vs Lindblad"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    print(
        "Stochastic τ — λ from exponential fit to **Gaussian-smoothed** ensemble coherence "
        "(oscillations smoothed for envelope fit only):"
    )
    for sigma in noise_strengths:
        lam = decay_rates[float(sigma)][1]
        print(f"  noise_strength={sigma}: λ ≈ {lam:.4f}")
    print("For γ grid-fit vs TDF, run compare_tdf_vs_fitted_lindblad().")

    if show:
        plt.show()

    return {
        "t": t,
        "omega": omega,
        "noise_strengths": noise_strengths,
        "n_ensemble": n_ensemble,
        "ensemble_rho_transverse_coherence": curves,
        "mean_transverse_coherence": curves,
        "exponential_fit_A_lam_yinf": decay_rates,
        "lindblad_gamma": lindblad_gamma,
        "lindblad_transverse_coherence": lindblad_coh,
        "figure": fig,
    }


def compare_noise_models(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible alias for :func:`run_stochastic_tau_decoherence_experiment`."""
    return run_stochastic_tau_decoherence_experiment(**kwargs)
