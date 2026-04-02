"""
Unified multi-observable laws along a control-parameter sweep.

Fits low-order curves y(x) between scalar metrics and scores **fit quality** plus
**cross-window stability** of coefficients—not raw Pearson coupling of metrics alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# (name, x_col, y_col)
RELATION_SPECS: tuple[tuple[str, str, str], ...] = (
    ("chsh_vs_sx", "spectrum_entropy_sx", "chsh_spectral_entropy"),
    ("dec_vs_sx", "spectrum_entropy_sx", "decoherence_mismatch_lindblad"),
    ("overlap_vs_fdom", "dominant_freq_cxx", "overlap_correlation"),
)


def _poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if degree == 1:
        return np.column_stack([np.ones_like(x), x])
    if degree == 2:
        return np.column_stack([np.ones_like(x), x, x**2])
    raise ValueError("degree must be 1 or 2")


def fit_poly_ls(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return ``(coef, rmse, y_pred)`` for least-squares polynomial in x."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < degree + 1:
        nan_coef = np.full(degree + 1, np.nan)
        return nan_coef, float("nan"), np.full_like(x, np.nan)
    X = _poly_design(x, degree)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return coef.astype(float), rmse, pred


def choose_degree_by_rmse(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[int, np.ndarray, float, np.ndarray]:
    """
    Prefer quadratic only if it clearly beats linear and enough points exist.

    Returns
    -------
    degree, coef, rmse, pred_on_masked_x
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    c1, r1, p1 = fit_poly_ls(x, y, 1)
    if n < 4:
        return 1, c1, r1, p1
    c2, r2, p2 = fit_poly_ls(x, y, 2)
    if r2 < 0.85 * r1:
        return 2, c2, r2, p2
    return 1, c1, r1, p1


def coefficient_variation_across_windows(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    n_windows: int,
) -> float:
    """
    Mean relative std of fitted coefficients across contiguous index windows.

    Lower is more stable. Returns ``inf`` if insufficient segments.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n < n_windows + degree:
        return float("inf")
    idx = np.arange(n)
    splits = np.array_split(idx, n_windows)
    coefs: list[np.ndarray] = []
    for s in splits:
        if s.size < degree + 1:
            continue
        c, rm, _ = fit_poly_ls(x[s], y[s], degree)
        if np.isfinite(rm):
            coefs.append(c)
    if len(coefs) < 2:
        return float("inf")
    C = np.row_stack(coefs)
    rel = []
    for j in range(C.shape[1]):
        col = C[:, j]
        m = float(np.mean(np.abs(col)))
        rel.append(float(np.std(col) / (m + 1e-12)))
    return float(np.mean(rel))


@dataclass
class RelationResult:
    name: str
    x_col: str
    y_col: str
    degree: int
    coef: np.ndarray
    rmse: float
    coef_cv_windows: float
    n_points: int


@dataclass
class UnifiedLawReport:
    relations: list[RelationResult] = field(default_factory=list)
    mean_rmse: float = float("nan")
    mean_coef_cv: float = float("nan")
    unified_score: float = float("nan")
    n_windows: int = 3


def analyze_unified_laws(
    df: pd.DataFrame,
    *,
    n_windows: int = 3,
    rmse_scale: float = 1.0,
    cv_scale: float = 1.0,
) -> UnifiedLawReport:
    """
    Fit pairwise laws, windowed coefficient variation, and a combined score.

    ``unified_score`` increases when relation RMSEs are small and coefficient
    variation across windows is small (stable law along the sweep).
    """
    n_windows = int(max(2, min(n_windows, max(2, len(df) // 2))))
    rel_results: list[RelationResult] = []

    for name, xcol, ycol in RELATION_SPECS:
        if xcol not in df.columns or ycol not in df.columns:
            continue
        x = df[xcol].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=float)
        deg, coef, rmse, _ = choose_degree_by_rmse(x, y)
        cv = coefficient_variation_across_windows(x, y, deg, n_windows)
        rel_results.append(
            RelationResult(
                name=name,
                x_col=xcol,
                y_col=ycol,
                degree=deg,
                coef=coef,
                rmse=rmse,
                coef_cv_windows=cv,
                n_points=int(np.sum(np.isfinite(x) & np.isfinite(y))),
            )
        )

    if not rel_results:
        return UnifiedLawReport(n_windows=n_windows)

    rmses = np.array([r.rmse for r in rel_results], dtype=float)
    cvs = np.array([r.coef_cv_windows for r in rel_results], dtype=float)
    mean_rmse = float(np.nanmean(rmses))
    mean_cv = float(np.nanmean(cvs[np.isfinite(cvs)])) if np.any(np.isfinite(cvs)) else float("inf")

    # High score ⇔ low RMSE and low coef CV (stable laws)
    inv_rmse = 1.0 / (mean_rmse / rmse_scale + 1e-6)
    stab = 1.0 / (mean_cv / cv_scale + 1e-6)
    unified = float(np.sqrt(inv_rmse * stab))

    return UnifiedLawReport(
        relations=rel_results,
        mean_rmse=mean_rmse,
        mean_coef_cv=mean_cv,
        unified_score=unified,
        n_windows=n_windows,
    )


def plot_relation_fits(
    df: pd.DataFrame,
    report: UnifiedLawReport,
    *,
    title_prefix: str,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Scatter y vs x with global fit curve for each relation."""
    n = len(report.relations)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    axes = axes[0]
    for ax, r in zip(axes, report.relations):
        x = df[r.x_col].to_numpy(dtype=float)
        y = df[r.y_col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        ax.scatter(x, y, s=40, c="k", alpha=0.75, zorder=3, label="sweep")
        if x.size and np.all(np.isfinite(r.coef)):
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 80)
            Xp = _poly_design(xs, r.degree)
            ax.plot(xs, Xp @ r.coef, "C0-", lw=2, label=f"deg-{r.degree} fit")
        ax.set_xlabel(r.x_col, fontsize=8)
        ax.set_ylabel(r.y_col, fontsize=8)
        ax.set_title(f"{r.name}\nRMSE={r.rmse:.4g}, coef_CV={r.coef_cv_windows:.4g}", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{title_prefix}: unified observable relations", fontsize=10)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        if not show:
            plt.close(fig)
    elif not show:
        plt.close(fig)


def plot_window_coefficient_stability(
    df: pd.DataFrame,
    report: UnifiedLawReport,
    *,
    title_prefix: str,
    n_windows: int,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Bar chart of per-window leading coefficient (slope) for each relation."""
    nrel = len(report.relations)
    if nrel == 0:
        return
    fig, axes = plt.subplots(1, nrel, figsize=(4 * nrel, 3.5), squeeze=False)
    axes = axes[0]
    idx_all = np.arange(len(df))
    splits = np.array_split(idx_all, n_windows)

    for ax, r in zip(axes, report.relations):
        x = df[r.x_col].to_numpy(dtype=float)
        y = df[r.y_col].to_numpy(dtype=float)
        slopes: list[float] = []
        labels: list[str] = []
        for k, s in enumerate(splits):
            if s.size < r.degree + 1:
                continue
            c, _, _ = fit_poly_ls(x[s], y[s], r.degree)
            # slope = c[1] for linear; for quadratic show c[1] as "linear part"
            slopes.append(float(c[1]) if c.size > 1 else float("nan"))
            labels.append(f"w{k+1}")
        if slopes:
            ax.bar(labels, slopes, color="C2", edgecolor="k")
        ax.set_title(r.name, fontsize=9)
        ax.set_ylabel("coef[1] (linear term)" if r.degree >= 1 else "coef")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{title_prefix}: refit per window ({n_windows} segments)", fontsize=10)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        if not show:
            plt.close(fig)
    elif not show:
        plt.close(fig)


def interpret_unified_tdf_vs_ou(
    tdf_report: UnifiedLawReport,
    ou_report: UnifiedLawReport,
    *,
    rmse_margin: float = 1.05,
    cv_margin: float = 1.05,
) -> str:
    """
    TDF wins if mean relation RMSE is lower and mean coef CV is lower (by margins).
    """
    t_rm = tdf_report.mean_rmse
    o_rm = ou_report.mean_rmse
    t_cv = tdf_report.mean_coef_cv
    o_cv = ou_report.mean_coef_cv

    if not np.isfinite(t_rm) or not np.isfinite(o_rm):
        return "Insufficient sweep data for unified-law comparison."

    tdf_better_rmse = t_rm * rmse_margin < o_rm
    tdf_better_cv = np.isfinite(t_cv) and np.isfinite(o_cv) and t_cv * cv_margin < o_cv

    if tdf_better_rmse and tdf_better_cv:
        return (
            "TDF exhibits a stable unified multi-observable law not reproduced by standard OU noise."
        )
    return (
        "Unified-law metrics do not clearly favor TDF over OU on this sweep "
        "(compare mean RMSE and coefficient variation across windows)."
    )


def print_unified_law_summary(label: str, report: UnifiedLawReport) -> None:
    print(f"\n--- {label} ---")
    print(f"  mean relation RMSE: {report.mean_rmse:.6g}")
    print(f"  mean coef CV (windows): {report.mean_coef_cv:.6g}")
    print(f"  unified_score: {report.unified_score:.6g}")
    for r in report.relations:
        print(
            f"  {r.name}: deg={r.degree}, RMSE={r.rmse:.6g}, coef_CV={r.coef_cv_windows:.6g}"
        )
