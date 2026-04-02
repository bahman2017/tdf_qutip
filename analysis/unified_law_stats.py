"""
Bootstrap, permutation, and sensitivity helpers for unified-law scores.

Builds on :func:`analysis.unified_law_metrics.analyze_unified_laws` without duplicating
the core fit logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.unified_law_metrics import (
    RELATION_SPECS,
    analyze_unified_laws,
    choose_degree_by_rmse,
    fit_poly_ls,
    _poly_design,
)


def bootstrap_unified_scores(
    df: pd.DataFrame,
    *,
    n_windows: int = 3,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """
    Resample sweep rows with replacement; recompute ``unified_score`` each time.

    Returns
    -------
    scores, point_estimate, ci_low, ci_high
        Point estimate is the score on the unresampled ``df``; CI is 2.5–97.5%
        percentiles of bootstrap scores.
    """
    rng = rng or np.random.default_rng()
    n = len(df)
    if n < 3:
        return np.array([]), float("nan"), float("nan"), float("nan")

    point = analyze_unified_laws(df, n_windows=n_windows).unified_score
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        df_b = df.iloc[idx].reset_index(drop=True)
        out[b] = analyze_unified_laws(df_b, n_windows=n_windows).unified_score
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return out, float(point), float(lo), float(hi)


def permutation_unified_scores(
    df: pd.DataFrame,
    *,
    n_windows: int = 3,
    n_perm: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray, float]:
    """
    Break x–y pairings by shuffling each relation's *y* column across rows.

    Returns
    -------
    score_real, null_scores, p_value
        One-sided ``p = mean(null_scores >= score_real)``.
    """
    rng = rng or np.random.default_rng()
    n = len(df)
    if n < 3:
        return float("nan"), np.array([]), float("nan")

    real = analyze_unified_laws(df, n_windows=n_windows).unified_score
    nulls = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        df_p = df.copy()
        for _name, xc, yc in RELATION_SPECS:
            if xc in df_p.columns and yc in df_p.columns:
                df_p[yc] = df[yc].to_numpy()[rng.permutation(n)]
        nulls[p] = analyze_unified_laws(df_p, n_windows=n_windows).unified_score
    pval = float(np.mean(nulls >= real))
    return float(real), nulls, pval


def mean_relation_train_test_rmse(
    df: pd.DataFrame,
    *,
    frac_train: float = 0.5,
) -> float:
    """
    For each relation: fit polynomial (degree chosen on **train** half), RMSE on **test** half.

    Rows are assumed ordered along the control sweep (first half = earlier indices).
    """
    n = len(df)
    n_tr = max(3, int(n * frac_train))
    if n - n_tr < 2:
        return float("nan")

    idx_train = np.arange(0, n_tr)
    idx_test = np.arange(n_tr, n)
    rmses: list[float] = []

    for _name, xcol, ycol in RELATION_SPECS:
        if xcol not in df.columns or ycol not in df.columns:
            continue
        x = df[xcol].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=float)
        xt, yt = x[idx_train], y[idx_train]
        xe, ye = x[idx_test], y[idx_test]
        m_tr = np.isfinite(xt) & np.isfinite(yt)
        m_te = np.isfinite(xe) & np.isfinite(ye)
        xt, yt = xt[m_tr], yt[m_tr]
        xe, ye = xe[m_te], ye[m_te]
        if xt.size < 3 or xe.size < 2:
            continue
        deg, coef, _, _ = choose_degree_by_rmse(xt, yt)
        if deg > xe.size - 1:
            deg = min(deg, max(1, xe.size - 1))
            coef, _, _ = fit_poly_ls(xt, yt, deg)
        Xe = _poly_design(xe, deg)
        pred = Xe @ coef
        rmses.append(float(np.sqrt(np.mean((ye - pred) ** 2))))

    if not rmses:
        return float("nan")
    return float(np.mean(rmses))


def delta_confidence_interval_normal(
    deltas: np.ndarray,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Mean, sample std, and normal-approx CI for the mean of ``deltas``.

    Uses Student-t critical value with ``n-1`` dof when SciPy is available;
    otherwise falls back to the normal critical value ``z_{1-α/2}``.
    """
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    if n == 1:
        return mean_d, std_d, mean_d, mean_d
    try:
        from scipy import stats

        crit = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    except ImportError:
        crit = 1.96 if abs(alpha - 0.05) < 1e-9 else 2.0
    half = crit * std_d / np.sqrt(n)
    return mean_d, std_d, mean_d - half, mean_d + half


def sensitivity_unified_score_vs_windows(
    df: pd.DataFrame,
    window_grid: tuple[int, ...] = (2, 3, 4),
) -> dict[int, float]:
    """Unified score for each ``n_windows`` value."""
    out: dict[int, float] = {}
    for nw in window_grid:
        if len(df) < 2 * nw:
            continue
        out[int(nw)] = analyze_unified_laws(df, n_windows=nw).unified_score
    return out
