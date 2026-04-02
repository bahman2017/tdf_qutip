"""
Infer a parametric τ(t) from two-qubit correlation and CHSH trajectories.

Model: τ(t) = ω t + a sin(ν t) + η(t), with optional Ornstein–Uhlenbeck η (σ, τ_c).

The pipeline checks whether this class of τ-fields can reproduce observable
correlation signatures from TDF evolution (compare fits to a Markovian / standard
benchmark separately in your analysis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import differential_evolution, minimize

from analysis.correlation_metrics import rmse
from core.evolution import run_evolution
from core.hamiltonians import tau_to_two_qubit_tdf_hamiltonian
from experiments.correlation_test import (
    bell_phi_plus,
    chsh_operators,
    correlation_pauli_tensors,
)

PARAM_NAMES = ("omega", "a", "nu", "sigma", "tau_c")


def build_parametric_tau(
    t: np.ndarray,
    omega: float,
    a: float,
    nu: float,
    *,
    sigma: float = 0.0,
    tau_c: float = 1.0,
    seed: int | None = 42,
) -> np.ndarray:
    """
    τ(t) = ω t + a sin(ν t) + η(t), with η an OU process when σ > 0.

    OU update (same discrete scheme as :func:`core.tau_model.correlated_stochastic_tau`):
    η_k = η_{k-1} - (η_{k-1}/τ_c) Δt + σ √(Δt) ξ_k.
    """
    t_arr = np.asarray(t, dtype=float)
    structured = float(omega) * t_arr + float(a) * np.sin(float(nu) * t_arr)
    if float(sigma) <= 0.0:
        return structured.astype(float)

    if float(tau_c) <= 0.0:
        raise ValueError("tau_c must be positive when sigma > 0")

    if seed is not None:
        np.random.seed(int(seed))

    noise = np.zeros_like(t_arr, dtype=float)
    n = int(t_arr.size)
    if n <= 1:
        return structured

    flat = t_arr.ravel()
    nf = flat.size
    noise_flat = np.zeros(nf, dtype=float)
    for i in range(1, nf):
        dt_i = float(flat[i] - flat[i - 1])
        if dt_i <= 0.0:
            raise ValueError("t must be strictly increasing")
        noise_flat[i] = (
            noise_flat[i - 1]
            - (noise_flat[i - 1] / float(tau_c)) * dt_i
            + float(sigma) * np.sqrt(dt_i) * np.random.standard_normal()
        )
    noise = noise_flat.reshape(t_arr.shape)
    return structured + noise


def simulate_correlations_from_tau(
    t: np.ndarray,
    omega: float,
    a: float,
    nu: float,
    sigma: float = 0.0,
    tau_c: float = 1.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Build τ(t), two-qubit TDF Hamiltonian H(t) = E(t) G, evolve |Φ+⟩, return correlators.

    Returns
    -------
    dict
        Keys ``cxx``, ``cyy``, ``czz``, ``chsh`` — expectation values vs ``t``.
    """
    t = np.asarray(t, dtype=float)
    tau = build_parametric_tau(
        t, omega, a, nu, sigma=sigma, tau_c=tau_c, seed=seed
    )
    H = tau_to_two_qubit_tdf_hamiltonian(tau, t, hbar=1.0)
    psi0 = bell_phi_plus()
    XX, YY, ZZ = correlation_pauli_tensors()
    A0, A1, B0, B1 = chsh_operators()
    A0B0 = qt.tensor(A0, B0)
    A0B1 = qt.tensor(A0, B1)
    A1B0 = qt.tensor(A1, B0)
    A1B1 = qt.tensor(A1, B1)
    e_ops = [XX, YY, ZZ, A0B0, A0B1, A1B0, A1B1]

    res = run_evolution(H, psi0, t, e_ops=e_ops)
    cxx = np.asarray(res.expect[0], dtype=float)
    cyy = np.asarray(res.expect[1], dtype=float)
    czz = np.asarray(res.expect[2], dtype=float)
    e00 = np.asarray(res.expect[3], dtype=float)
    e01 = np.asarray(res.expect[4], dtype=float)
    e10 = np.asarray(res.expect[5], dtype=float)
    e11 = np.asarray(res.expect[6], dtype=float)
    chsh = e00 + e01 + e10 - e11

    return {"cxx": cxx, "cyy": cyy, "czz": czz, "chsh": chsh, "tau": tau}


def weighted_correlation_loss(
    cxx_data: np.ndarray,
    cyy_data: np.ndarray,
    chsh_data: np.ndarray,
    cxx_model: np.ndarray,
    cyy_model: np.ndarray,
    chsh_model: np.ndarray,
    *,
    w_xx: float = 1.0,
    w_yy: float = 1.0,
    w_s: float = 1.0,
) -> float:
    """
    Weighted sum of RMSEs between data and model traces (default weights all 1).
    """
    return (
        float(w_xx) * rmse(cxx_data, cxx_model)
        + float(w_yy) * rmse(cyy_data, cyy_model)
        + float(w_s) * rmse(chsh_data, chsh_model)
    )


def _x_to_params(x: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(PARAM_NAMES, map(float, x))}


def _guess_to_x0(
    initial_guess: Sequence[float] | dict[str, float],
) -> np.ndarray:
    if isinstance(initial_guess, dict):
        return np.array([float(initial_guess[k]) for k in PARAM_NAMES], dtype=float)
    arr = np.asarray(initial_guess, dtype=float).ravel()
    if arr.size != len(PARAM_NAMES):
        raise ValueError(
            f"initial_guess must have {len(PARAM_NAMES)} values or dict with keys {PARAM_NAMES}"
        )
    return arr


def fit_tau_from_correlations(
    t: np.ndarray,
    cxx_data: np.ndarray,
    cyy_data: np.ndarray,
    chsh_data: np.ndarray,
    initial_guess: Sequence[float] | dict[str, float],
    bounds: Sequence[tuple[float, float]],
    *,
    w_xx: float = 1.0,
    w_yy: float = 1.0,
    w_s: float = 1.0,
    noise_seed: int = 42,
    method: str = "L-BFGS-B",
    maxiter: int = 250,
    output_dir: str | Path | None = None,
    save_plots: bool = True,
    show: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Fit (ω, a, ν, σ, τ_c) by minimizing a weighted RMSE loss on C_xx, C_yy, and S_CHSH.

    Parameters
    ----------
    t, cxx_data, cyy_data, chsh_data
        Target trajectories (same length).
    initial_guess
        Length-5 sequence (ω, a, ν, σ, τ_c) or dict with those keys.
    bounds
        ``scipy`` bounds, one (low, high) per parameter in the same order.
    method
        ``"L-BFGS-B"`` (local) or ``"differential_evolution"`` (global).
    noise_seed
        Fixed RNG seed when evaluating η(t) so the loss is deterministic in (σ, τ_c).
    verbose
        If False, omit progress prints (e.g. for multi-start wrappers).

    Returns
    -------
    dict
        ``params``, ``tau`` (best τ(t)), ``traces`` (model curves), ``loss``,
        ``optimization_result``, ``figure_paths`` (if saved).
    """
    t = np.asarray(t, dtype=float)
    cxx_data = np.asarray(cxx_data, dtype=float).ravel()
    cyy_data = np.asarray(cyy_data, dtype=float).ravel()
    chsh_data = np.asarray(chsh_data, dtype=float).ravel()
    n = t.size
    if cxx_data.size != n or cyy_data.size != n or chsh_data.size != n:
        raise ValueError("all arrays must match length of t")

    if len(bounds) != len(PARAM_NAMES):
        raise ValueError(f"bounds must have {len(PARAM_NAMES)} tuples")

    x0 = _guess_to_x0(initial_guess)
    bounds_list = list(bounds)

    def objective(x: np.ndarray) -> float:
        p = _x_to_params(x)
        sim = simulate_correlations_from_tau(
            t,
            p["omega"],
            p["a"],
            p["nu"],
            sigma=p["sigma"],
            tau_c=p["tau_c"],
            seed=noise_seed,
        )
        return weighted_correlation_loss(
            cxx_data,
            cyy_data,
            chsh_data,
            sim["cxx"],
            sim["cyy"],
            sim["chsh"],
            w_xx=w_xx,
            w_yy=w_yy,
            w_s=w_s,
        )

    if method == "differential_evolution":
        opt = differential_evolution(
            objective,
            bounds=bounds_list,
            maxiter=maxiter,
            seed=noise_seed,
            polish=True,
            workers=1,
        )
        x_best = opt.x
        fun = float(opt.fun)
    else:
        opt = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds_list,
            options={"maxiter": maxiter},
        )
        x_best = opt.x
        fun = float(opt.fun)

    best = _x_to_params(x_best)
    sim = simulate_correlations_from_tau(
        t,
        best["omega"],
        best["a"],
        best["nu"],
        sigma=best["sigma"],
        tau_c=best["tau_c"],
        seed=noise_seed,
    )

    out: dict[str, Any] = {
        "params": best,
        "tau": sim["tau"],
        "traces": {k: sim[k] for k in ("cxx", "cyy", "czz", "chsh")},
        "loss": fun,
        "optimization_result": opt,
        "weights": {"w_xx": w_xx, "w_yy": w_yy, "w_s": w_s},
        "noise_seed": noise_seed,
    }

    figure_paths: dict[str, Path] = {}
    if save_plots:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "outputs"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def _plot_pair(
            y_data: np.ndarray,
            y_model: np.ndarray,
            ylabel: str,
            title: str,
            fname: str,
        ) -> Path:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(t, y_data, "k.", markersize=3, alpha=0.6, label="data")
            ax.plot(t, y_model, "-", color="C1", linewidth=1.8, label="τ-model")
            ax.set_xlabel("t")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = output_dir / fname
            fig.savefig(path, dpi=150)
            if not show:
                plt.close(fig)
            return path

        figure_paths["cxx"] = _plot_pair(
            cxx_data,
            sim["cxx"],
            r"$C_{xx}$",
            r"$C_{xx}$: data vs fitted τ-model",
            "tau_extraction_Cxx_fit.png",
        )
        figure_paths["cyy"] = _plot_pair(
            cyy_data,
            sim["cyy"],
            r"$C_{yy}$",
            r"$C_{yy}$: data vs fitted τ-model",
            "tau_extraction_Cyy_fit.png",
        )
        figure_paths["chsh"] = _plot_pair(
            chsh_data,
            sim["chsh"],
            r"$S_{\mathrm{CHSH}}$",
            "CHSH: data vs fitted τ-model",
            "tau_extraction_CHSH_fit.png",
        )

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(t, sim["tau"], color="C0", linewidth=1.5)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\tau(t)$")
        ax.set_title(
            rf"Extracted $\tau(t)$: $\omega$={best['omega']:.4g}, $a$={best['a']:.4g}, "
            rf"$\nu$={best['nu']:.4g}, $\sigma$={best['sigma']:.4g}, $\tau_c$={best['tau_c']:.4g}"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        ptau = output_dir / "tau_extraction_tau.png"
        fig.savefig(ptau, dpi=150)
        if not show:
            plt.close(fig)
        figure_paths["tau"] = ptau

        if verbose:
            for pth in figure_paths.values():
                print(f"Saved {pth}")

        out["figure_paths"] = figure_paths

    if show and save_plots:
        plt.show()

    if verbose:
        print()
        print("τ extraction — best parameters:", best)
        print(f"Final weighted loss: {fun:.6g}")

    return out


if __name__ == "__main__":
    from scripts.pipeline_demo import step_tau_extraction

    step_tau_extraction(None)
