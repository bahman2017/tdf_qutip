"""
Cross-observable coupling across control-parameter sweeps (phenomenological discrimination).

Builds a Pearson correlation matrix between scalar metrics evaluated at each sweep point;
the mean absolute off-diagonal summarizes how tightly those observables co-vary with the knob.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Columns used for coupling analysis (must exist in the sweep dataframe)
METRIC_COLUMNS: tuple[str, ...] = (
    "spectrum_entropy_sx",
    "overlap_correlation",
    "decoherence_mismatch_lindblad",
    "chsh_spectral_entropy",
    "dominant_freq_cxx",
)


def metrics_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Stack sweep rows into a :class:`~pandas.DataFrame`."""
    return pd.DataFrame(rows)


def cross_observable_coupling_matrix(
    df: pd.DataFrame,
    metric_cols: Sequence[str] = METRIC_COLUMNS,
) -> tuple[np.ndarray, list[str]]:
    """
    Pearson correlation matrix between metric columns (across sweep index).

    Returns
    -------
    R, labels
        ``R[i,j]`` is correlation between column ``labels[i]`` and ``labels[j]``.
    """
    cols = [c for c in metric_cols if c in df.columns]
    if len(cols) < 2:
        return np.full((0, 0), np.nan), cols
    X = df[cols].to_numpy(dtype=float)
    if X.shape[0] < 2:
        return np.full((len(cols), len(cols)), np.nan), cols
    R = np.corrcoef(X, rowvar=False)
    R = np.asarray(R, dtype=float)
    np.fill_diagonal(R, 1.0)
    return R, cols


def joint_coupling_score(R: np.ndarray) -> float:
    """
    Mean absolute off-diagonal entry of a correlation matrix.

    Larger values mean metrics tend to move together as the control is swept.
    """
    if R.size == 0 or R.shape[0] < 2:
        return float("nan")
    m = R.shape[0]
    off = [abs(R[i, j]) for i in range(m) for j in range(m) if i != j]
    return float(np.nanmean(off)) if off else float("nan")


def plot_metrics_vs_control(
    df: pd.DataFrame,
    *,
    model_label: str,
    control_col: str = "control_value",
    metric_cols: Sequence[str] = METRIC_COLUMNS,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> None:
    """One subplot per metric vs the swept control parameter."""
    cols = [c for c in metric_cols if c in df.columns]
    if not cols or control_col not in df.columns:
        return
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    x = df[control_col].to_numpy(dtype=float)
    for ax, c in zip(axes, cols):
        y = df[c].to_numpy(dtype=float)
        ax.plot(x, y, "o-", linewidth=1.2, markersize=5)
        ax.set_ylabel(c, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(control_col)
    fig.suptitle(title or f"{model_label}: metrics vs control", fontsize=11)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def plot_coupling_heatmap(
    R: np.ndarray,
    labels: list[str],
    *,
    title: str,
    output_path: str | Path | None = None,
) -> None:
    """Heatmap of the coupling matrix with metric labels."""
    if R.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(R, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def print_coupling_comparison(scores: dict[str, float]) -> None:
    """Pretty-print joint-coupling scores by model name."""
    for k, v in sorted(scores.items(), key=lambda kv: (-(np.nan_to_num(kv[1], nan=-1)), kv[0])):
        print(f"  {k}: joint_coupling = {v:.6g}")


def interpret_tdf_vs_colored_noise(
    scores: dict[str, float],
    *,
    margin: float = 1.08,
) -> str:
    """
    If TDF's joint score exceeds both standard colored-noise models by ``margin``, return
    the unified-structure sentence; otherwise a neutral diagnostic string.
    """
    tdf = float(scores.get("tdf", np.nan))
    ou = float(scores.get("ou_colored", np.nan))
    pink = float(scores.get("pink", np.nan))
    if not np.isfinite(tdf):
        return "Insufficient data for TDF vs colored-noise coupling comparison."
    rivals = [x for x in (ou, pink) if np.isfinite(x)]
    if not rivals:
        return "No colored-noise baselines scored; cannot compare."
    best_rival = max(rivals)
    if tdf > margin * best_rival:
        return (
            "TDF exhibits a unified multi-observable structure not reproduced by "
            "standard colored noise."
        )
    return (
        "TDF does not clearly dominate the joint cross-observable coupling score "
        "relative to the colored-noise baselines on this sweep."
    )
