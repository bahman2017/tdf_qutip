"""
Fit PCA kernel modes φ_k(t) to simple effective 1D field equations (damped / relaxation).

Interpretation is **effective**: good fits suggest the temporal mode could arise from a
hidden τ-field eigenmode with oscillatory or relaxational dynamics, not a proof of the
full 5D structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from analysis.correlation_metrics import rmse


def _nan_fit_result(model: str) -> dict[str, Any]:
    """Placeholder when a fit is skipped or fails."""
    nan = float("nan")
    return {
        "model": model,
        "rmse": nan,
        "phi_pred": np.array([]),
        "error": "fit_failed",
    }


def fit_damped_oscillator_mode(t: np.ndarray, phi: np.ndarray) -> dict[str, Any]:
    """
    Fit φ'' + γ φ' + ω² φ = 0 by least squares on
    φ'' = -γ φ' - ω² φ, then reconstruct via ``solve_ivp``.

    Parameters
    ----------
    t, phi
        Same length; ``t`` should be strictly increasing.

    Returns
    -------
    dict
        ``model``, ``gamma``, ``omega2``, ``rmse``, ``phi_pred`` (aligned with ``t``),
        optional ``error`` on failure.
    """
    t = np.asarray(t, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    out_base: dict[str, Any] = {"model": "damped_oscillator"}

    if t.shape != phi.shape or t.size < 4:
        r = _nan_fit_result("damped_oscillator")
        r["gamma"] = float("nan")
        r["omega2"] = float("nan")
        return {**out_base, **r}

    if np.any(np.diff(t) <= 0):
        r = _nan_fit_result("damped_oscillator")
        r["gamma"] = float("nan")
        r["omega2"] = float("nan")
        r["error"] = "t_not_strictly_increasing"
        return {**out_base, **r}

    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(t)):
        r = _nan_fit_result("damped_oscillator")
        r["gamma"] = float("nan")
        r["omega2"] = float("nan")
        return {**out_base, **r}

    amp = float(np.max(np.abs(phi)))
    if amp < 1e-15 or float(np.std(phi)) < 1e-30:
        r = _nan_fit_result("damped_oscillator")
        r["gamma"] = float("nan")
        r["omega2"] = float("nan")
        r["error"] = "tiny_amplitude"
        return {**out_base, **r}

    try:
        phi_dot = np.gradient(phi, t, edge_order=2)
        phi_ddot = np.gradient(phi_dot, t, edge_order=2)
        X = np.column_stack([-phi_dot, -phi])
        coef, *_ = np.linalg.lstsq(X, phi_ddot, rcond=None)
        gamma, omega2 = float(coef[0]), float(coef[1])
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        r = _nan_fit_result("damped_oscillator")
        r["gamma"] = float("nan")
        r["omega2"] = float("nan")
        return {**out_base, **r}

    y0 = [float(phi[0]), float(phi_dot[0])]

    def ode(_t: float, y: list[float]) -> list[float]:
        p, pd = y
        return [pd, -gamma * pd - omega2 * p]

    try:
        sol = solve_ivp(
            ode,
            (float(t[0]), float(t[-1])),
            y0,
            t_eval=t,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        if not sol.success or sol.y.size == 0:
            raise RuntimeError(sol.message)
        phi_pred = np.asarray(sol.y[0], dtype=float)
    except (ValueError, RuntimeError, FloatingPointError):
        r = _nan_fit_result("damped_oscillator")
        r["gamma"] = gamma
        r["omega2"] = omega2
        return {**out_base, **r}

    err = rmse(phi, phi_pred)
    return {
        **out_base,
        "gamma": gamma,
        "omega2": omega2,
        "rmse": err,
        "phi_pred": phi_pred,
    }


def fit_relaxation_mode(t: np.ndarray, phi: np.ndarray) -> dict[str, Any]:
    """
    Fit φ' + μ φ = 0 (i.e. φ' = -μ φ) by least squares on ``phi_dot`` vs ``-φ``.

    Reconstruction: φ_pred(t) = φ(t*) exp(-μ (t - t*)) with t* chosen at max |φ|
    when φ(0) is negligible.
    """
    t = np.asarray(t, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    out_base: dict[str, Any] = {"model": "relaxation"}

    if t.shape != phi.shape or t.size < 3:
        r = _nan_fit_result("relaxation")
        r["mu"] = float("nan")
        return {**out_base, **r}

    if np.any(np.diff(t) <= 0):
        r = _nan_fit_result("relaxation")
        r["mu"] = float("nan")
        r["error"] = "t_not_strictly_increasing"
        return {**out_base, **r}

    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(t)):
        r = _nan_fit_result("relaxation")
        r["mu"] = float("nan")
        return {**out_base, **r}

    amp = float(np.max(np.abs(phi)))
    if amp < 1e-15 or float(np.std(phi)) < 1e-30:
        r = _nan_fit_result("relaxation")
        r["mu"] = float("nan")
        r["error"] = "tiny_amplitude"
        return {**out_base, **r}

    try:
        phi_dot = np.gradient(phi, t, edge_order=2)
        # mask near-zeros in phi for more stable μ
        scale = max(amp, 1.0)
        mask = np.abs(phi) > 1e-12 * scale
        if np.count_nonzero(mask) < 2:
            mask = np.ones_like(phi, dtype=bool)
        ph = phi[mask]
        pd = phi_dot[mask]
        X = (-ph).reshape(-1, 1)
        coef, *_ = np.linalg.lstsq(X, pd, rcond=None)
        mu = float(coef[0])
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        r = _nan_fit_result("relaxation")
        r["mu"] = float("nan")
        return {**out_base, **r}

    i0 = 0
    phi0 = float(phi[0])
    if abs(phi0) < 1e-12 * scale:
        i0 = int(np.argmax(np.abs(phi)))
        phi0 = float(phi[i0])
        t0 = float(t[i0])
    else:
        t0 = float(t[0])

    if abs(phi0) < 1e-30:
        r = _nan_fit_result("relaxation")
        r["mu"] = mu
        r["error"] = "no_valid_ic"
        return {**out_base, **r}

    try:
        phi_pred = phi0 * np.exp(-mu * (t - t0))
    except FloatingPointError:
        r = _nan_fit_result("relaxation")
        r["mu"] = mu
        return {**out_base, **r}

    err = rmse(phi, phi_pred)
    return {
        **out_base,
        "mu": mu,
        "rmse": err,
        "phi_pred": phi_pred.astype(float),
    }


def fit_kernel_modes_to_field_equations(
    t: np.ndarray,
    kernel_result: dict[str, Any],
    *,
    n_modes: int = 3,
    output_dir: str | Path | None = None,
    save_plots: bool = True,
    show: bool = False,
) -> dict[str, Any]:
    """
    Fit the first ``n_modes`` PCA modes from ``run_tau_kernel_mode_analysis`` to
    damped-oscillator and relaxation models; pick the smaller RMSE per mode.

    Parameters
    ----------
    kernel_result
        Output of :func:`analysis.tau_kernel_modes.run_tau_kernel_mode_analysis`
        (needs ``phi_modes``, ``explained_variance``; may include ``t``).
    t
        Time grid; used if ``kernel_result`` has no ``t``.
    """
    t = np.asarray(t, dtype=float).ravel()
    if "t" in kernel_result:
        tk = np.asarray(kernel_result["t"], dtype=float).ravel()
        if tk.size != t.size:
            raise ValueError("kernel_result['t'] and t must have the same length")
        atol = 1e-9 * (float(np.max(np.abs(t))) + 1.0)
        if not np.allclose(tk, t, rtol=0, atol=atol):
            raise ValueError("kernel_result['t'] must match the provided time grid t")

    phi_modes = np.asarray(kernel_result["phi_modes"], dtype=float)
    explained = np.asarray(
        kernel_result.get("explained_variance", np.full(phi_modes.shape[0], np.nan)),
        dtype=float,
    )

    if phi_modes.ndim != 2 or phi_modes.shape[1] != t.size:
        raise ValueError("kernel_result['phi_modes'] must be (n_modes, len(t))")

    n_fit = min(int(n_modes), phi_modes.shape[0])
    mode_fits: list[dict[str, Any]] = []
    n_damped_best = 0
    n_relax_best = 0
    n_none = 0

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}

    for k in range(n_fit):
        phi_row = phi_modes[k]
        ev = float(explained[k]) if k < explained.size else float("nan")

        damped = fit_damped_oscillator_mode(t, phi_row)
        relax = fit_relaxation_mode(t, phi_row)

        rd = float(damped["rmse"])
        rr = float(relax["rmse"])

        best: str
        if not np.isfinite(rd) and not np.isfinite(rr):
            best = "none"
            n_none += 1
        elif not np.isfinite(rd):
            best = "relaxation"
            n_relax_best += 1
        elif not np.isfinite(rr):
            best = "damped_oscillator"
            n_damped_best += 1
        elif rd < rr:
            best = "damped_oscillator"
            n_damped_best += 1
        else:
            best = "relaxation"
            n_relax_best += 1

        mode_fits.append(
            {
                "mode_index": k,
                "explained_variance": ev,
                "best_model": best,
                "damped_fit": damped,
                "relax_fit": relax,
            }
        )

        if save_plots:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t, phi_row, "k-", linewidth=2.0, label=r"$\phi_k(t)$ (PCA)")

            if (
                damped.get("phi_pred") is not None
                and np.asarray(damped["phi_pred"]).size == t.size
                and np.isfinite(rd)
            ):
                ax.plot(
                    t,
                    damped["phi_pred"],
                    "--",
                    linewidth=1.5,
                    alpha=0.9,
                    label=rf"damped (RMSE={rd:.4g})",
                )
            if (
                relax.get("phi_pred") is not None
                and np.asarray(relax["phi_pred"]).size == t.size
                and np.isfinite(rr)
            ):
                ax.plot(
                    t,
                    relax["phi_pred"],
                    ":",
                    linewidth=1.5,
                    alpha=0.9,
                    label=rf"relax (RMSE={rr:.4g})",
                )

            tit = (
                f"mode {k + 1}  |  EV={ev:.3g}  |  best: {best}  "
                f"(damped RMSE={rd:.4g}, relax RMSE={rr:.4g})"
            )
            ax.set_title(tit, fontsize=10)
            ax.set_xlabel("t")
            ax.set_ylabel(r"$\phi$")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p_mode = output_dir / f"tau_mode_fit_mode{k + 1}.png"
            fig.savefig(p_mode, dpi=150)
            figure_paths[f"mode_{k + 1}"] = p_mode
            if not show:
                plt.close(fig)

    if save_plots and n_fit > 0:
        fig2, ax2 = plt.subplots(figsize=(7.5, 4))
        xpos = np.arange(n_fit)
        evs = [mode_fits[i]["explained_variance"] for i in range(n_fit)]
        ax2.bar(xpos, evs, color="C0", edgecolor="k")
        for i in range(n_fit):
            m = mode_fits[i]["best_model"]
            ax2.annotate(
                m.replace("_", " "),
                xy=(xpos[i], evs[i]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                rotation=45,
            )
        ax2.set_xticks(xpos)
        ax2.set_xticklabels([f"{i + 1}" for i in range(n_fit)])
        ax2.set_xlabel("mode index")
        ax2.set_ylabel("explained variance ratio")
        ax2.set_title("Kernel modes: variance share and best effective model")
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        p_sum = output_dir / "tau_mode_fit_summary.png"
        fig2.savefig(p_sum, dpi=150)
        figure_paths["summary"] = p_sum
        if not show:
            plt.close(fig2)

    if show and save_plots:
        plt.show()

    summary_dict = {
        "n_modes": n_fit,
        "n_damped_best": n_damped_best,
        "n_relax_best": n_relax_best,
        "n_none": n_none,
    }
    return {
        "mode_fits": mode_fits,
        "summary": summary_dict,
        "figure_paths": figure_paths if save_plots else {},
        "interpretation": interpret_mode_field_fits(
            {"mode_fits": mode_fits, "summary": summary_dict}
        ),
    }


def interpret_mode_field_fits(result: dict[str, Any]) -> str:
    """
    Summarize whether leading fits prefer damped oscillators vs relaxation.

    Uses ``result['mode_fits']`` and ``result['summary']`` (counts of ``best_model``).
    """
    mf = result.get("mode_fits", [])
    summ = result.get("summary", {})
    n_modes = int(summ.get("n_modes", len(mf)))
    if n_modes <= 0:
        return "Kernel modes are mixed; no single effective 5D dynamics dominates."

    n_d = int(summ.get("n_damped_best", 0))
    n_r = int(summ.get("n_relax_best", 0))

    # Leading = strict majority among classified modes (exclude 'none')
    classified = n_d + n_r
    if classified == 0:
        return "Kernel modes are mixed; no single effective 5D dynamics dominates."

    if n_d > n_r and n_d > classified / 2:
        return (
            "Leading kernel modes behave like hidden 5D eigenmodes with oscillatory dynamics."
        )
    if n_r > n_d and n_r > classified / 2:
        return "Leading kernel modes behave like soft relaxational hidden modes."

    return "Kernel modes are mixed; no single effective 5D dynamics dominates."


# Example (not executed):
# from analysis.tau_mode_field_fit import fit_kernel_modes_to_field_equations
# result = fit_kernel_modes_to_field_equations(
#     t=t,
#     kernel_result=kernel_result,
#     n_modes=3,
#     output_dir="outputs",
# )


if __name__ == "__main__":
    from scripts.pipeline_demo import step_mode_field_fit

    step_mode_field_fit(None)
