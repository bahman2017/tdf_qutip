"""
Coarse vs fine τ structure: do observables track the manifold mean or residual detail?

Uses ``multi_start_tau_fit`` τ trajectories; compares evolution under true τ, mean τ, and
mean + shuffled residual (same δτ power spectrum scrambled in time).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from analysis.correlation_metrics import rmse
from core.evolution import run_evolution
from core.hamiltonians import tau_to_two_qubit_tdf_hamiltonian
from experiments.correlation_test import bell_phi_plus, chsh_operators, correlation_pauli_tensors


def decompose_tau_manifold(summary: dict[str, Any]) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Split multi-start τ samples into mean and per-run residuals.

    Parameters
    ----------
    summary
        Dict with ``all_tau`` of shape ``(n_runs, len(t))``.

    Returns
    -------
    tau_mean :
        ``(len(t),)`` average τ across runs.
    tau_residuals :
        List of length ``n_runs`` with ``τ_i(t) - τ_mean(t)``.
    """
    all_tau = np.asarray(summary["all_tau"], dtype=float)
    if all_tau.ndim != 2:
        raise ValueError("summary['all_tau'] must be 2-D (n_runs, n_times)")
    tau_mean = np.mean(all_tau, axis=0)
    tau_residuals = [np.asarray(all_tau[i] - tau_mean, dtype=float) for i in range(all_tau.shape[0])]
    return tau_mean, tau_residuals


def simulate_from_tau(
    t: np.ndarray,
    tau: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Two-qubit Bell evolution with TDF Hamiltonian built from τ(t).

    Returns ``cxx``, ``cyy``, ``chsh`` expectation values on ``t``.
    """
    t = np.asarray(t, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    if tau.shape != t.shape:
        raise ValueError("tau and t must have the same shape")

    H = tau_to_two_qubit_tdf_hamiltonian(tau, t, hbar=1.0)
    psi0 = bell_phi_plus()
    XX, YY, ZZ = correlation_pauli_tensors()
    A0, A1, B0, B1 = chsh_operators()
    e_ops = [
        XX,
        YY,
        ZZ,
        qt.tensor(A0, B0),
        qt.tensor(A0, B1),
        qt.tensor(A1, B0),
        qt.tensor(A1, B1),
    ]

    res = run_evolution(H, psi0, t, e_ops=e_ops)
    cxx = np.asarray(res.expect[0], dtype=float)
    cyy = np.asarray(res.expect[1], dtype=float)
    e00 = np.asarray(res.expect[3], dtype=float)
    e01 = np.asarray(res.expect[4], dtype=float)
    e10 = np.asarray(res.expect[5], dtype=float)
    e11 = np.asarray(res.expect[6], dtype=float)
    chsh = e00 + e01 + e10 - e11
    return {"cxx": cxx, "cyy": cyy, "chsh": chsh}


def _rmse_abc(
    obs_a: dict[str, np.ndarray],
    obs_b: dict[str, np.ndarray],
) -> float:
    """Mean RMSE over ``cxx``, ``cyy``, ``chsh``."""
    return float(
        np.mean(
            [
                rmse(obs_a["cxx"], obs_b["cxx"]),
                rmse(obs_a["cyy"], obs_b["cyy"]),
                rmse(obs_a["chsh"], obs_b["chsh"]),
            ]
        )
    )


def run_tau_decomposition_test(
    summary: dict[str, Any],
    t: np.ndarray,
    *,
    seed: int = 42,
    output_dir: str | Path | None = None,
    save_plots: bool = True,
    show: bool = False,
    ratio_threshold: float = 0.35,
) -> dict[str, Any]:
    """
    For three representative τ (closest / farthest / random to τ_mean), compare:

    * **A**: full τ_i
    * **B**: τ_mean only (coarse)
    * **C**: τ_mean + time-shuffled residual (fine structure permuted)

    If observables match **B** much better than **C**, dynamics are mostly driven by coarse τ.

    Parameters
    ----------
    ratio_threshold
        Declare “mainly coarse” when ``mean(RMSE_AB) < ratio_threshold * mean(RMSE_AC)``.
    """
    t = np.asarray(t, dtype=float).ravel()
    all_tau = np.asarray(summary["all_tau"], dtype=float)
    if all_tau.shape[1] != t.size:
        raise ValueError("len(t) must match summary['all_tau'].shape[1]")

    n = all_tau.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 τ trajectories in summary['all_tau']")

    tau_mean, tau_residuals = decompose_tau_manifold(summary)
    rng = np.random.default_rng(seed)

    resid_norms = np.array([float(np.linalg.norm(all_tau[i] - tau_mean)) for i in range(n)])
    idx_closest = int(np.argmin(resid_norms))
    idx_farthest = int(np.argmax(resid_norms))
    idx_random = int(rng.integers(0, n))
    # Prefer three distinct indices when possible
    if n >= 3:
        tries = 0
        while idx_random in (idx_closest, idx_farthest) and tries < 50:
            idx_random = int(rng.integers(0, n))
            tries += 1

    labels = ("closest", "farthest", "random")
    indices = (idx_closest, idx_farthest, idx_random)

    per_rep: list[dict[str, Any]] = []
    rmse_ab_list: list[float] = []
    rmse_ac_list: list[float] = []

    obs_for_plot: dict[str, Any] | None = None

    for label, idx in zip(labels, indices):
        tau_i = all_tau[idx]
        resid = tau_residuals[idx]
        shuffled_resid = resid.copy()
        rng.shuffle(shuffled_resid)
        tau_c = tau_mean + shuffled_resid

        obs_a = simulate_from_tau(t, tau_i)
        obs_b = simulate_from_tau(t, tau_mean)
        obs_c = simulate_from_tau(t, tau_c)

        rab = _rmse_abc(obs_a, obs_b)
        rac = _rmse_abc(obs_a, obs_c)
        rmse_ab_list.append(rab)
        rmse_ac_list.append(rac)

        rep = {
            "label": label,
            "index": idx,
            "rmse_ab": rab,
            "rmse_ac": rac,
            "obs_a": obs_a,
            "obs_b": obs_b,
            "obs_c": obs_c,
            "tau_i": tau_i,
        }
        per_rep.append(rep)
        if label == "farthest":
            obs_for_plot = rep

    rmse_ab_mean = float(np.mean(rmse_ab_list))
    rmse_ac_mean = float(np.mean(rmse_ac_list))

    kernel_coarse = rmse_ab_mean < ratio_threshold * max(rmse_ac_mean, 1e-15)
    conclusion = (
        "Observables depend mainly on coarse τ → kernel confirmed"
        if kernel_coarse
        else "Fine τ structure affects observables"
    )

    print()
    print("τ decomposition test:")
    for rep in per_rep:
        print(
            f"  [{rep['label']:8s}] RMSE(A,B)={rep['rmse_ab']:.6g}  "
            f"RMSE(A,C)={rep['rmse_ac']:.6g}"
        )
    print(f"  mean RMSE(A,B)={rmse_ab_mean:.6g}  mean RMSE(A,C)={rmse_ac_mean:.6g}")
    if rmse_ab_mean < ratio_threshold * max(rmse_ac_mean, 1e-15):
        print("Observables depend mainly on coarse τ → kernel confirmed")
    else:
        print("Fine τ structure affects observables")

    figure_paths: dict[str, Path] = {}
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_plots:
        # τ: mean + three representatives
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(t, tau_mean, "k-", linewidth=2.2, label=r"$\tau_{\mathrm{mean}}$")
        colors = ("C0", "C1", "C2")
        for rep, c in zip(per_rep, colors):
            ax.plot(
                t,
                rep["tau_i"],
                "--",
                color=c,
                alpha=0.85,
                linewidth=1.4,
                label=rf"$\tau$ ({rep['label']})",
            )
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\tau$")
        ax.set_title(r"Representative $\tau_i(t)$ vs ensemble mean")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p1 = output_dir / "tau_decomposition_tau.png"
        fig.savefig(p1, dpi=150)
        if not show:
            plt.close(fig)
        figure_paths["tau"] = p1

        # Observables — farthest case (largest δτ): A vs B vs C
        assert obs_for_plot is not None
        oa, ob, oc = obs_for_plot["obs_a"], obs_for_plot["obs_b"], obs_for_plot["obs_c"]
        fig2, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        titles = (r"$C_{xx}$", r"$C_{yy}$", r"CHSH $S$")
        keys = ("cxx", "cyy", "chsh")
        for ax_, title, key in zip(axes, titles, keys):
            ax_.plot(t, oa[key], label="A: full τ", color="C0", linewidth=1.6)
            ax_.plot(t, ob[key], label="B: τ_mean", color="C1", linewidth=1.6, alpha=0.9)
            ax_.plot(t, oc[key], label="C: mean + shuffled δτ", color="C2", linewidth=1.6, alpha=0.9)
            ax_.set_ylabel(title)
            ax_.grid(True, alpha=0.3)
            ax_.legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("t")
        axes[0].set_title(r"Observables (representative: farthest from $\tau_{\mathrm{mean}}$)")
        fig2.tight_layout()
        p2 = output_dir / "tau_decomposition_observables.png"
        fig2.savefig(p2, dpi=150)
        if not show:
            plt.close(fig2)
        figure_paths["observables"] = p2

        for pth in figure_paths.values():
            print(f"Saved {pth}")

    if show and save_plots:
        plt.show()

    return {
        "tau_mean": tau_mean,
        "tau_residuals": tau_residuals,
        "representative_indices": dict(zip(labels, indices)),
        "per_representative": per_rep,
        "rmse_ab": np.asarray(rmse_ab_list, dtype=float),
        "rmse_ac": np.asarray(rmse_ac_list, dtype=float),
        "rmse_ab_mean": rmse_ab_mean,
        "rmse_ac_mean": rmse_ac_mean,
        "ratio_threshold": ratio_threshold,
        "conclusion": conclusion,
        "kernel_coarse_dominant": bool(kernel_coarse),
        "figure_paths": figure_paths if save_plots else {},
    }
