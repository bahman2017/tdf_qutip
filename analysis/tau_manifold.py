"""
Degeneracy manifold for τ-models: visualize near-equivalent fits from :func:`multi_start_tau_fit`.

Selects low-loss restarts, joins **parameter** and **τ(t)** PCA views, and summarizes spread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from analysis.tau_extraction import PARAM_NAMES


def _effective_dimension(explained_variance_ratio: np.ndarray, threshold: float = 0.9) -> int:
    """Smallest number of PCs whose cumulative explained variance reaches ``threshold``."""
    ev = np.asarray(explained_variance_ratio, dtype=float)
    ev = ev[ev > 1e-15]
    if ev.size == 0:
        return 0
    c = np.cumsum(ev)
    ok = c >= threshold
    if not np.any(ok):
        return int(ev.size)
    return int(np.argmax(ok) + 1)


def _stack_params(all_params: list[dict[str, float]]) -> np.ndarray:
    return np.array([[float(p[k]) for k in PARAM_NAMES] for p in all_params], dtype=float)


def analyze_degeneracy_manifold(
    summary: dict[str, Any],
    t: np.ndarray,
    epsilon: float,
    *,
    output_dir: str | Path | None = None,
    save_plots: bool = True,
    show: bool = False,
    variance_threshold: float = 0.9,
) -> dict[str, Any]:
    """
    Keep solutions with loss within ``epsilon`` of the best loss; PCA in parameter and τ space.

    Parameters
    ----------
    summary
        Output of :func:`analysis.tau_identifiability.multi_start_tau_fit` (needs
        ``all_losses``, ``all_params``, ``all_tau``).
    t
        Time axis, length matching ``summary['all_tau'].shape[1]``.
    epsilon
        **Additive** margin: ``good`` if ``loss <= min(loss) + epsilon``.

    Returns
    -------
    dict
        Indices, filtered arrays, PCA results, manifold metrics, ``interpretation``,
        optional ``figure_paths``.
    """
    t = np.asarray(t, dtype=float).ravel()
    losses = np.asarray(summary["all_losses"], dtype=float).ravel()
    all_tau = np.asarray(summary["all_tau"], dtype=float)
    all_params = summary["all_params"]

    if all_tau.ndim != 2 or all_tau.shape[1] != t.size:
        raise ValueError("summary['all_tau'] must have shape (n_runs, len(t))")
    if len(all_params) != all_tau.shape[0] or len(losses) != all_tau.shape[0]:
        raise ValueError("all_params, all_losses, and all_tau must have the same n_runs")

    min_loss = float(np.min(losses))
    good_mask = losses <= min_loss + float(epsilon)
    good_idx = np.flatnonzero(good_mask)
    n_good = int(good_idx.size)

    if n_good == 0:
        raise ValueError(
            f"No runs satisfy loss <= min_loss + epsilon ({min_loss + epsilon:g}); increase epsilon."
        )

    losses_good = losses[good_idx]
    params_good = _stack_params([all_params[i] for i in good_idx])
    tau_good = all_tau[good_idx]

    # --- Parameter PCA (first 2 coords for plotting)
    n_comp_p = min(2, max(1, n_good - 1), params_good.shape[1])
    if n_good == 1:
        pca_param = None
        projected_params_2d = np.zeros((1, 2), dtype=float)
        explained_param = np.array([1.0, 0.0], dtype=float)
        pca_components_param = np.zeros((len(PARAM_NAMES), 2), dtype=float)
    else:
        pca_param = PCA(n_components=n_comp_p)
        proj_p = pca_param.fit_transform(params_good)
        explained_param = np.pad(
            pca_param.explained_variance_ratio_,
            (0, max(0, 2 - n_comp_p)),
            constant_values=0.0,
        )
        if proj_p.shape[1] == 1:
            projected_params_2d = np.column_stack([proj_p, np.zeros(n_good)])
        else:
            projected_params_2d = proj_p[:, :2].copy()
        pca_components_param = np.zeros((len(PARAM_NAMES), 2), dtype=float)
        pca_components_param[:, :n_comp_p] = pca_param.components_.T

    intrinsic_dim_param = (
        _effective_dimension(pca_param.explained_variance_ratio_, variance_threshold)
        if pca_param is not None
        else 0
    )

    # --- τ(t) PCA (each row is one τ curve)
    Tdim = tau_good.shape[1]
    n_comp_t = min(2, max(1, n_good - 1), Tdim)
    if n_good == 1:
        pca_tau = None
        projected_tau_2d = np.zeros((1, 2), dtype=float)
        explained_tau = np.array([1.0, 0.0], dtype=float)
        pca_components_tau = None
    else:
        pca_tau = PCA(n_components=n_comp_t)
        proj_t = pca_tau.fit_transform(tau_good)
        explained_tau = np.pad(
            pca_tau.explained_variance_ratio_,
            (0, max(0, 2 - n_comp_t)),
            constant_values=0.0,
        )
        if proj_t.shape[1] == 1:
            projected_tau_2d = np.column_stack([proj_t, np.zeros(n_good)])
        else:
            projected_tau_2d = proj_t[:, :2].copy()
        pca_components_tau = pca_tau.components_

    intrinsic_dim_tau = (
        _effective_dimension(pca_tau.explained_variance_ratio_, variance_threshold)
        if pca_tau is not None
        else 0
    )

    param_spread = {
        name: float(np.std(params_good[:, j], ddof=1)) if n_good > 1 else 0.0
        for j, name in enumerate(PARAM_NAMES)
    }
    tau_std_time = np.std(tau_good, axis=0)
    tau_spread_mean = float(np.mean(tau_std_time))

    metrics = {
        "intrinsic_dimension_param": intrinsic_dim_param,
        "intrinsic_dimension_tau": intrinsic_dim_tau,
        "explained_variance_param": explained_param,
        "explained_variance_tau": explained_tau,
        "parameter_spread": param_spread,
        "tau_spread_over_time": tau_std_time,
        "tau_spread_mean": tau_spread_mean,
        "n_good": n_good,
        "min_loss": min_loss,
        "epsilon": float(epsilon),
    }

    interpretation = interpret_manifold(metrics)

    out: dict[str, Any] = {
        "good_idx": good_idx,
        "params_good": params_good,
        "tau_good": tau_good,
        "losses_good": losses_good,
        "projected_params_2d": projected_params_2d,
        "projected_tau_2d": projected_tau_2d,
        "pca_components_param": pca_components_param,
        "pca_components_tau": pca_components_tau,
        "pca_param": pca_param,
        "pca_tau": pca_tau,
        "metrics": metrics,
        "interpretation": interpretation,
    }

    if save_plots:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "outputs"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        # (a) Parameter space — 2D projection colored by loss
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(
            projected_params_2d[:, 0],
            projected_params_2d[:, 1],
            c=losses_good,
            cmap="viridis",
            s=45,
            edgecolors="k",
            linewidths=0.3,
        )
        plt.colorbar(sc, ax=ax, label="loss")
        ax.set_xlabel("PC1 (parameters)")
        ax.set_ylabel("PC2 (parameters)")
        ax.set_title(f"Parameter manifold (n_good={n_good}, ε={epsilon:g})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p_a = output_dir / "tau_manifold_param_space.png"
        fig.savefig(p_a, dpi=150)
        if not show:
            plt.close(fig)
        paths["param_space"] = p_a

        # (b) τ space — overlay + mean ± std
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for row in tau_good:
            ax.plot(t, row, color="C0", alpha=0.25, linewidth=1.0)
        mean_tau = np.mean(tau_good, axis=0)
        ax.plot(t, mean_tau, color="k", linewidth=2.0, label="mean τ")
        ax.fill_between(
            t,
            mean_tau - tau_std_time,
            mean_tau + tau_std_time,
            color="k",
            alpha=0.15,
            label="±1 std (across good fits)",
        )
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\tau(t)$")
        ax.set_title(r"$\tau(t)$ manifold (low-loss ensemble)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p_b = output_dir / "tau_manifold_tau_overlay.png"
        fig.savefig(p_b, dpi=150)
        if not show:
            plt.close(fig)
        paths["tau_overlay"] = p_b

        # (c) τ embedding — first two PCs of τ trajectories
        fig, ax = plt.subplots(figsize=(6, 5))
        sc2 = ax.scatter(
            projected_tau_2d[:, 0],
            projected_tau_2d[:, 1],
            c=losses_good,
            cmap="plasma",
            s=45,
            edgecolors="k",
            linewidths=0.3,
        )
        plt.colorbar(sc2, ax=ax, label="loss")
        ax.set_xlabel("PC1 (τ curves)")
        ax.set_ylabel("PC2 (τ curves)")
        ax.set_title("τ trajectory embedding (PCA)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p_c = output_dir / "tau_manifold_embedding.png"
        fig.savefig(p_c, dpi=150)
        if not show:
            plt.close(fig)
        paths["embedding"] = p_c

        for pth in paths.values():
            print(f"Saved {pth}")
        out["figure_paths"] = paths

    if show and save_plots:
        plt.show()

    print()
    print("Degeneracy manifold:")
    print(f"  good runs: {n_good} (loss ≤ min + ε = {min_loss + epsilon:.6g})")
    print(
        f"  intrinsic dim (param, τ) at {variance_threshold:.0%} var: "
        f"{intrinsic_dim_param}, {intrinsic_dim_tau}"
    )
    print(f"  interpretation: {interpretation}")

    return out


def interpret_manifold(metrics: dict[str, Any]) -> str:
    """
    Coarse label from PCA intrinsic dimensions (at 90% variance by default in analysis).

    Returns
    -------
    str
        ``\"τ nearly unique\"`` or ``\"τ lies on a degenerate manifold\"``.
    """
    if int(metrics.get("n_good", 0)) <= 1:
        return "τ nearly unique"

    d_p = int(metrics.get("intrinsic_dimension_param", 99))
    d_t = int(metrics.get("intrinsic_dimension_tau", 99))

    if d_p <= 1 and d_t <= 1:
        return "τ nearly unique"
    return "τ lies on a degenerate manifold"


if __name__ == "__main__":
    from scripts.pipeline_demo import step_manifold

    step_manifold(None)
