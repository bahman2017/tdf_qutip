"""
Kernel-style PCA modes φ_k(t) on the τ degeneracy manifold and observable sensitivity.

Uses ``tau_good`` from :func:`analyze_degeneracy_manifold`; residual PCA yields modes
orthogonal to τ_mean in sample space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from analysis.correlation_metrics import rmse
from analysis.tau_decomposition_test import simulate_from_tau


def extract_kernel_modes_from_manifold(
    manifold: dict[str, Any],
    summary: dict[str, Any],
    t: np.ndarray,
    *,
    n_modes_max: int | None = None,
) -> dict[str, Any]:
    """
    PCA on τ residuals (τ_i − τ_mean) over the low-loss manifold.

    Parameters
    ----------
    manifold
        Output of :func:`analyze_degeneracy_manifold` (needs ``tau_good``).
    summary
        Full multi-start summary; used only to check shape consistency with ``t``.
    t
        Time grid ``(T,)``.

    Returns
    -------
    dict
        ``tau_mean``, ``phi_modes`` ``(K, T)``, ``explained_variance`` ``(K,)``,
        ``coefficients`` ``(n_good, K)``, ``pca`` model, ``tau_good``.
    """
    t = np.asarray(t, dtype=float).ravel()
    tau_good = np.asarray(manifold["tau_good"], dtype=float)
    if tau_good.ndim != 2 or tau_good.shape[1] != t.size:
        raise ValueError("manifold['tau_good'] must be (n_good, len(t))")

    all_tau = np.asarray(summary["all_tau"], dtype=float)
    if all_tau.shape[1] != t.size:
        raise ValueError("summary['all_tau'] time dimension must match t")

    n_good, Tdim = tau_good.shape
    if n_good < 2:
        raise ValueError("Need at least 2 τ curves in manifold['tau_good']")

    tau_mean = np.mean(tau_good, axis=0)
    R = tau_good - tau_mean

    k_max = min(n_good - 1, Tdim)
    if n_modes_max is not None:
        k_max = min(k_max, int(n_modes_max))
    k_max = max(1, k_max)

    pca = PCA(n_components=k_max)
    coefficients = pca.fit_transform(R)
    phi_modes = np.asarray(pca.components_, dtype=float)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=float)

    return {
        "tau_mean": tau_mean,
        "phi_modes": phi_modes,
        "explained_variance": explained,
        "coefficients": coefficients,
        "pca": pca,
        "tau_good": tau_good,
        "t": t,
    }


def reconstruct_tau(extraction: dict[str, Any], i: int, n_modes: int) -> np.ndarray:
    """
    τ_rec = τ_mean + Σ_{k < n_modes} c_{ik} φ_k.

    Parameters
    ----------
    extraction
        Output of :func:`extract_kernel_modes_from_manifold`.
    i
        Index into the manifold samples (row of ``tau_good``).
    n_modes
        Number of PCA modes to include.
    """
    tau_mean = extraction["tau_mean"]
    phi = extraction["phi_modes"]
    c = extraction["coefficients"]
    if i < 0 or i >= c.shape[0]:
        raise IndexError("i out of range for coefficients")
    nm = min(int(n_modes), phi.shape[0])
    return tau_mean + c[i, :nm] @ phi[:nm]


def _mean_obs_rmse(
    obs_test: dict[str, np.ndarray],
    obs_ref: dict[str, np.ndarray],
) -> float:
    return float(
        np.mean(
            [
                rmse(obs_test["cxx"], obs_ref["cxx"]),
                rmse(obs_test["cyy"], obs_ref["cyy"]),
                rmse(obs_test["chsh"], obs_ref["chsh"]),
            ]
        )
    )


def _label_mode(
    var_share: float,
    max_rmse_off_zero: float,
    *,
    var_share_large: float = 0.12,
    rmse_hidden_threshold: float = 0.08,
) -> str:
    """Heuristic: large variance share but small observable response → kernel-like."""
    if var_share >= var_share_large and max_rmse_off_zero < rmse_hidden_threshold:
        return "hidden / kernel-like mode"
    return "observable mode"


def run_tau_kernel_mode_analysis(
    manifold: dict[str, Any],
    summary: dict[str, Any],
    t: np.ndarray,
    *,
    lambdas: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    output_dir: str | Path | None = None,
    save_plots: bool = True,
    show: bool = False,
    n_modes_max: int | None = None,
    var_share_large: float = 0.12,
    rmse_hidden_threshold: float = 0.08,
) -> dict[str, Any]:
    """
    Extract φ modes, plot summaries, and sweep τ_mean + λ φ_k for observable sensitivity.

    Returns extraction dict extended with ``sensitivity_rmse``, ``mode_labels``, paths.
    """
    t = np.asarray(t, dtype=float).ravel()
    ext = extract_kernel_modes_from_manifold(
        manifold, summary, t, n_modes_max=n_modes_max
    )
    tau_mean = ext["tau_mean"]
    phi_modes = ext["phi_modes"]
    explained = ext["explained_variance"]
    n_modes = phi_modes.shape[0]

    obs_ref = simulate_from_tau(t, tau_mean)
    lam_list = list(lambdas)
    sensitivity_rmse = np.zeros((n_modes, len(lam_list)), dtype=float)

    for k in range(n_modes):
        phi_k = phi_modes[k]
        for j, lam in enumerate(lam_list):
            tau_lam = tau_mean + float(lam) * phi_k
            obs_l = simulate_from_tau(t, tau_lam)
            sensitivity_rmse[k, j] = _mean_obs_rmse(obs_l, obs_ref)

    mode_labels: list[str] = []
    mode_meta: list[dict[str, Any]] = []
    for k in range(n_modes):
        off_zero = [sensitivity_rmse[k, j] for j, lam in enumerate(lam_list) if lam != 0]
        max_off = float(np.max(off_zero)) if off_zero else 0.0
        var_share = float(explained[k]) if k < explained.size else 0.0
        lbl = _label_mode(
            var_share,
            max_off,
            var_share_large=var_share_large,
            rmse_hidden_threshold=rmse_hidden_threshold,
        )
        mode_labels.append(lbl)
        mode_meta.append(
            {
                "index": k,
                "explained_variance_ratio": var_share,
                "max_rmse_lambda_neq0": max_off,
                "label": lbl,
            }
        )

    figure_paths: dict[str, Path] = {}
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_plots:
        # τ_mean + first 3 φ_k(t)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(t, tau_mean, "k-", linewidth=2.0, label=r"$\tau_{\mathrm{mean}}$")
        n_show = min(3, n_modes)
        for k in range(n_show):
            ax.plot(
                t,
                phi_modes[k],
                "--",
                linewidth=1.4,
                alpha=0.9,
                label=rf"$\phi_{k}$ (EV={explained[k]:.2f})",
            )
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\tau$ / mode amplitude")
        ax.set_title(r"Mean $\tau$ and leading kernel modes")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p1 = output_dir / "tau_kernel_modes.png"
        fig.savefig(p1, dpi=150)
        if not show:
            plt.close(fig)
        figure_paths["modes"] = p1

        # Explained variance bar chart
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.bar(np.arange(n_modes), explained, color="C0", edgecolor="k")
        ax2.set_xlabel("mode index $k$")
        ax2.set_ylabel("explained variance ratio")
        ax2.set_title("PCA on τ residuals (manifold)")
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        p2 = output_dir / "tau_kernel_variance.png"
        fig2.savefig(p2, dpi=150)
        if not show:
            plt.close(fig2)
        figure_paths["variance"] = p2

        # Sensitivity: RMSE vs λ for each mode
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        for k in range(n_modes):
            ax3.plot(
                lam_list,
                sensitivity_rmse[k],
                "o-",
                label=rf"$\phi_{k}$ — {mode_labels[k]}",
            )
        ax3.set_xlabel(r"$\lambda$ in $\tau = \tau_{\mathrm{mean}} + \lambda \phi_k$")
        ax3.set_ylabel("mean RMSE vs τ_mean observables")
        ax3.set_title("Observable sensitivity along each mode")
        ax3.legend(loc="best", fontsize=7)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        p3 = output_dir / "tau_kernel_sensitivity.png"
        fig3.savefig(p3, dpi=150)
        if not show:
            plt.close(fig3)
        figure_paths["sensitivity"] = p3

        for pth in figure_paths.values():
            print(f"Saved {pth}")

    if show and save_plots:
        plt.show()

    print()
    print("Kernel mode labels:")
    for m in mode_meta:
        print(f"  mode {m['index']}: EV={m['explained_variance_ratio']:.4f}, "
              f"max RMSE (λ≠0)={m['max_rmse_lambda_neq0']:.6g} → {m['label']}")

    out: dict[str, Any] = {
        **ext,
        "lambdas": np.asarray(lam_list, dtype=float),
        "sensitivity_rmse": sensitivity_rmse,
        "mode_labels": mode_labels,
        "mode_meta": mode_meta,
        "obs_ref": obs_ref,
        "figure_paths": figure_paths if save_plots else {},
    }
    return out


# Alias matching the design doc naming
extract_kernel_modes = extract_kernel_modes_from_manifold


if __name__ == "__main__":
    from scripts.pipeline_demo import step_kernel_modes

    step_kernel_modes(None)
