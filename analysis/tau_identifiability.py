"""
Multi-start τ fitting to probe non-uniqueness of extracted τ(t) from correlation data.

If many random restarts converge to the same τ(t), the field is more **identifiable**;
scattered solutions suggest an **effective** (non-unique) parametrization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from analysis.correlation_metrics import rmse
from analysis.tau_extraction import PARAM_NAMES, fit_tau_from_correlations


def _pairwise_tau_rmse_matrix(taus: np.ndarray) -> np.ndarray:
    """Symmetric matrix of RMSE(τ_i, τ_j); diagonal is zero."""
    n = taus.shape[0]
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            e = rmse(taus[i], taus[j])
            M[i, j] = e
            M[j, i] = e
    return M


def interpret_identifiability(summary: dict[str, Any]) -> str:
    """
    Heuristic label from multi-start spread.

    Compares the typical pairwise RMSE between fitted τ curves to the typical
    RMS variation of τ across runs (per time sample). If curves agree up to
    that scale, treat τ as **identifiable**; otherwise **non-unique**.

    Returns
    -------
    str
        ``\"τ is identifiable (likely physical)\"`` or ``\"τ is not unique (effective model)\"``.
    """
    M = np.asarray(summary["tau_rmse_matrix"], dtype=float)
    n = M.shape[0]
    if n < 2:
        return "τ is identifiable (likely physical)"

    iu = np.triu_indices(n, k=1)
    median_pairwise = float(np.median(M[iu]))

    taus = np.asarray(summary["all_tau"], dtype=float)
    per_time_std = np.std(taus, axis=0)
    width = float(np.mean(per_time_std))

    # Pairwise agreement within ~15% of mean cross-run temporal spread → clustered
    if median_pairwise <= 0.15 * max(width, 1e-9) or median_pairwise < 1e-3:
        return "τ is identifiable (likely physical)"
    return "τ is not unique (effective model)"


def multi_start_tau_fit(
    t: np.ndarray,
    cxx_data: np.ndarray,
    cyy_data: np.ndarray,
    chsh_data: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    *,
    n_runs: int = 20,
    seed: int = 42,
    noise_seed: int = 42,
    method: str = "L-BFGS-B",
    maxiter: int = 200,
    output_dir: str | Path | None = None,
    save_plots: bool = True,
    show: bool = False,
) -> dict[str, Any]:
    """
    Repeat :func:`fit_tau_from_correlations` from uniform random starts inside ``bounds``.

    Parameters
    ----------
    n_runs
        Number of independent optimizations.
    seed
        RNG seed **only** for sampling initial guesses.
    noise_seed
        Passed through to each fit so the OU map η(t; σ, τ_c) is consistent across runs.

    Returns
    -------
    dict
        ``tau_rmse_matrix``, ``param_std``, ``loss_mean``, ``loss_std``, ``all_params``,
        ``all_tau`` (shape ``(n_runs, len(t))``), ``all_losses``, ``interpretation``,
        optional ``figure_paths`` (``tau_overlay``, ``param_histograms``).
    """
    t = np.asarray(t, dtype=float)
    bounds_list = list(bounds)
    if len(bounds_list) != len(PARAM_NAMES):
        raise ValueError(f"bounds must have {len(PARAM_NAMES)} tuples")

    rng = np.random.default_rng(seed)
    lows = np.array([b[0] for b in bounds_list], dtype=float)
    highs = np.array([b[1] for b in bounds_list], dtype=float)

    all_params: list[dict[str, float]] = []
    all_tau: list[np.ndarray] = []
    all_losses: list[float] = []

    for k in range(n_runs):
        x0 = rng.uniform(lows, highs)
        guess = {name: float(val) for name, val in zip(PARAM_NAMES, x0)}

        fit = fit_tau_from_correlations(
            t,
            cxx_data,
            cyy_data,
            chsh_data,
            initial_guess=guess,
            bounds=bounds_list,
            noise_seed=noise_seed,
            method=method,
            maxiter=maxiter,
            save_plots=False,
            show=False,
            verbose=False,
        )
        all_params.append(fit["params"])
        all_tau.append(np.asarray(fit["tau"], dtype=float).ravel())
        all_losses.append(float(fit["loss"]))
        print(f"  identifiability run {k + 1}/{n_runs}: loss={all_losses[-1]:.6g}")

    tau_stack = np.stack(all_tau, axis=0)
    M = _pairwise_tau_rmse_matrix(tau_stack)

    param_std: dict[str, float] = {
        name: float(
            np.std([float(p[name]) for p in all_params], ddof=1)
            if n_runs > 1
            else 0.0
        )
        for name in PARAM_NAMES
    }

    losses = np.asarray(all_losses, dtype=float)
    summary: dict[str, Any] = {
        "tau_rmse_matrix": M,
        "param_std": param_std,
        "loss_mean": float(np.mean(losses)),
        "loss_std": float(np.std(losses, ddof=1)) if n_runs > 1 else 0.0,
        "all_params": all_params,
        "all_tau": tau_stack,
        "all_losses": losses,
        "n_runs": n_runs,
        "bounds": bounds_list,
    }

    figure_paths: dict[str, Path] = {}
    if save_plots:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "outputs"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, tau_i in enumerate(tau_stack):
            ax.plot(t, tau_i, alpha=0.35, linewidth=1.2, color="C0")
        ax.plot(t, np.mean(tau_stack, axis=0), color="k", linewidth=2.0, label="mean τ")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\tau(t)$")
        ax.set_title(f"Multi-start τ trajectories (n={n_runs})")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p1 = output_dir / "tau_identifiability_tau_overlay.png"
        fig.savefig(p1, dpi=150)
        if not show:
            plt.close(fig)
        figure_paths["tau_overlay"] = p1

        fig2, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(14, 3))
        for ax, name in zip(axes, PARAM_NAMES):
            vals = [float(p[name]) for p in all_params]
            ax.hist(vals, bins=max(5, min(15, n_runs // 2)), color="C1", edgecolor="k", alpha=0.85)
            ax.set_xlabel(name)
            ax.set_ylabel("count")
        fig2.suptitle("Parameter spread across random restarts", y=1.02)
        fig2.tight_layout()
        p2 = output_dir / "tau_identifiability_param_histograms.png"
        fig2.savefig(p2, dpi=150)
        if not show:
            plt.close(fig2)
        figure_paths["param_histograms"] = p2

        for pth in figure_paths.values():
            print(f"Saved {pth}")
        summary["figure_paths"] = figure_paths

    if show and save_plots:
        plt.show()

    summary["interpretation"] = interpret_identifiability(summary)

    print()
    print("Identifiability summary:")
    print(f"  loss mean ± std: {summary['loss_mean']:.6g} ± {summary['loss_std']:.6g}")
    print(f"  param std: {param_std}")
    print(f"  → {summary['interpretation']}")

    return summary
