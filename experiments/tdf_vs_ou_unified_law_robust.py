"""
Statistical robustness layer for :mod:`experiments.tdf_vs_ou_unified_law`.

Multi-seed runs, bootstrap CIs on unified scores, permutation nulls, window/sweep
sensitivity, and optional train/test generalization metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.unified_law_metrics import analyze_unified_laws
from analysis.unified_law_stats import (
    bootstrap_unified_scores,
    delta_confidence_interval_normal,
    mean_relation_train_test_rmse,
    permutation_unified_scores,
)
from experiments.tdf_vs_ou_unified_law import collect_unified_law_sweeps


def run_tdf_vs_ou_unified_law_robust(
    t: np.ndarray | None = None,
    *,
    omega: float = 1.0,
    tau_freq: float = 3.0,
    tau_noise_strength: float = 0.5,
    tau_c_values: np.ndarray | None = None,
    ou_correlation_times: np.ndarray | None = None,
    ou_sigma: float = 0.5,
    n_seeds: int = 20,
    seed_offset: int = 4242,
    n_windows: int = 3,
    n_boot: int = 500,
    n_perm: int = 500,
    window_sensitivity_grid: tuple[int, ...] = (2, 3, 4),
    sweep_grids: dict[str, np.ndarray] | None = None,
    output_dir: str | Path | None = None,
    save_plots: bool = True,
    show: bool = False,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """
    Parameters
    ----------
    fast_mode
        If True, uses ``n_seeds=5``, ``n_boot=100``, ``n_perm=100`` (overrides counts).
    sweep_grids
        Optional mapping name → 1-D control grid for sweep-length sensitivity
        (default: ``n7`` and ``n11`` on the same ``τ_c`` / OU range).
    """
    if fast_mode:
        n_seeds = min(n_seeds, 5)
        n_boot = min(n_boot, 100)
        n_perm = min(n_perm, 100)

    if t is None:
        t = np.linspace(0.0, 8.0, 220)
    t = np.asarray(t, dtype=float).ravel()
    if tau_c_values is None:
        tau_c_values = np.linspace(0.35, 2.2, 9)
    if ou_correlation_times is None:
        ou_correlation_times = np.linspace(0.35, 2.2, 9)
    tau_c_values = np.asarray(tau_c_values, dtype=float)
    ou_correlation_times = np.asarray(ou_correlation_times, dtype=float)

    if sweep_grids is None:
        sweep_grids = {
            "n7": np.linspace(0.35, 2.2, 7),
            "n11": np.linspace(0.35, 2.2, 11),
        }

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(seed_offset)

    seed_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    perm_rows: list[dict[str, Any]] = []

    for s in range(n_seeds):
        base_seed = int(seed_offset + s * 10_001)
        df_tdf, df_ou = collect_unified_law_sweeps(
            t,
            omega=omega,
            tau_freq=tau_freq,
            tau_noise_strength=tau_noise_strength,
            tau_c_values=tau_c_values,
            ou_correlation_times=ou_correlation_times,
            ou_sigma=ou_sigma,
            base_seed=base_seed,
        )
        rep_t = analyze_unified_laws(df_tdf, n_windows=n_windows)
        rep_o = analyze_unified_laws(df_ou, n_windows=n_windows)
        delta = rep_t.unified_score - rep_o.unified_score

        tt_tdf = mean_relation_train_test_rmse(df_tdf)
        tt_ou = mean_relation_train_test_rmse(df_ou)

        seed_rows.append(
            {
                "seed": s,
                "base_seed": base_seed,
                "unified_score_tdf": rep_t.unified_score,
                "unified_score_ou": rep_o.unified_score,
                "mean_rmse_tdf": rep_t.mean_rmse,
                "mean_rmse_ou": rep_o.mean_rmse,
                "mean_coef_cv_tdf": rep_t.mean_coef_cv,
                "mean_coef_cv_ou": rep_o.mean_coef_cv,
                "delta_score": delta,
                "train_test_rmse_tdf": tt_tdf,
                "train_test_rmse_ou": tt_ou,
            }
        )

        for label, df in ("tdf", df_tdf), ("ou", df_ou):
            rng_b = np.random.default_rng(rng_master.integers(0, 2**31))
            boot_s, pt, lo, hi = bootstrap_unified_scores(
                df, n_windows=n_windows, n_boot=n_boot, rng=rng_b
            )
            boot_rows.append(
                {
                    "seed": s,
                    "model": label,
                    "point_score": pt,
                    "boot_mean": float(np.nanmean(boot_s)) if boot_s.size else float("nan"),
                    "boot_ci_low": lo,
                    "boot_ci_high": hi,
                    "n_boot": n_boot,
                }
            )

        for label, df in ("tdf", df_tdf), ("ou", df_ou):
            rng_p = np.random.default_rng(rng_master.integers(0, 2**31))
            real, nulls, pval = permutation_unified_scores(
                df, n_windows=n_windows, n_perm=n_perm, rng=rng_p
            )
            perm_rows.append(
                {
                    "seed": s,
                    "model": label,
                    "score_real": real,
                    "p_value": pval,
                    "n_perm": n_perm,
                    "null_mean": float(np.mean(nulls)) if nulls.size else float("nan"),
                }
            )

    df_seed = pd.DataFrame(seed_rows)
    df_boot = pd.DataFrame(boot_rows)
    df_perm = pd.DataFrame(perm_rows)

    deltas = df_seed["delta_score"].to_numpy(dtype=float)
    mean_delta, std_delta, ci_lo, ci_hi = delta_confidence_interval_normal(deltas)
    win_rate = float(np.mean(deltas > 0)) if deltas.size else 0.0

    p_tdf_vals = df_perm[df_perm["model"] == "tdf"]["p_value"].to_numpy(dtype=float)
    p_ou_vals = df_perm[df_perm["model"] == "ou"]["p_value"].to_numpy(dtype=float)
    p_tdf_mean = float(np.mean(p_tdf_vals))
    p_ou_mean = float(np.mean(p_ou_vals))
    p_tdf_median = float(np.median(p_tdf_vals))
    p_ou_median = float(np.median(p_ou_vals))

    # Window sensitivity: full multi-seed for each n_windows
    sens_rows: list[dict[str, Any]] = []
    tdf_better_all_windows = True
    for nw in window_sensitivity_grid:
        if len(tau_c_values) < 2 * nw:
            continue
        dlist: list[float] = []
        for s in range(n_seeds):
            base_seed = int(seed_offset + s * 10_001)
            df_tdf, df_ou = collect_unified_law_sweeps(
                t,
                omega=omega,
                tau_freq=tau_freq,
                tau_noise_strength=tau_noise_strength,
                tau_c_values=tau_c_values,
                ou_correlation_times=ou_correlation_times,
                ou_sigma=ou_sigma,
                base_seed=base_seed,
            )
            st = analyze_unified_laws(df_tdf, n_windows=nw).unified_score
            so = analyze_unified_laws(df_ou, n_windows=nw).unified_score
            dlist.append(st - so)
        md = float(np.mean(dlist))
        sens_rows.append(
            {
                "n_windows": nw,
                "mean_delta_score": md,
                "win_rate_tdf": float(np.mean(np.asarray(dlist) > 0)),
            }
        )
        if md <= 0:
            tdf_better_all_windows = False

    # Sweep grid sensitivity (same n_windows as main)
    grid_rows: list[dict[str, Any]] = []
    tdf_better_all_grids = True
    for gname, grid in sweep_grids.items():
        g = np.asarray(grid, dtype=float).ravel()
        dlist: list[float] = []
        for s in range(n_seeds):
            base_seed = int(seed_offset + s * 10_001)
            df_tdf, df_ou = collect_unified_law_sweeps(
                t,
                omega=omega,
                tau_freq=tau_freq,
                tau_noise_strength=tau_noise_strength,
                tau_c_values=g,
                ou_correlation_times=g,
                ou_sigma=ou_sigma,
                base_seed=base_seed,
            )
            st = analyze_unified_laws(df_tdf, n_windows=n_windows).unified_score
            so = analyze_unified_laws(df_ou, n_windows=n_windows).unified_score
            dlist.append(st - so)
        md = float(np.mean(dlist))
        grid_rows.append(
            {
                "grid_name": gname,
                "n_points": len(g),
                "mean_delta_score": md,
                "win_rate_tdf": float(np.mean(np.asarray(dlist) > 0)),
            }
        )
        if md <= 0:
            tdf_better_all_grids = False

    mean_tt_tdf = float(np.nanmean(df_seed["train_test_rmse_tdf"]))
    mean_tt_ou = float(np.nanmean(df_seed["train_test_rmse_ou"]))

    robust = (
        mean_delta > 0
        and ci_lo > 0
        and win_rate >= 0.7
        and p_tdf_median < 0.05
        and tdf_better_all_windows
        and tdf_better_all_grids
    )

    summary = {
        "n_seeds": n_seeds,
        "n_windows": n_windows,
        "n_boot": n_boot,
        "n_perm": n_perm,
        "mean_delta_score": mean_delta,
        "std_delta_score": std_delta,
        "delta_ci_95_low": ci_lo,
        "delta_ci_95_high": ci_hi,
        "win_rate_tdf": win_rate,
        "mean_perm_p_tdf": p_tdf_mean,
        "mean_perm_p_ou": p_ou_mean,
        "median_perm_p_tdf": p_tdf_median,
        "median_perm_p_ou": p_ou_median,
        "mean_train_test_rmse_tdf": mean_tt_tdf,
        "mean_train_test_rmse_ou": mean_tt_ou,
        "tdf_better_all_window_settings": tdf_better_all_windows,
        "tdf_better_all_sweep_grids": tdf_better_all_grids,
        "statistically_robust_tdf_better": robust,
    }

    df_summary = pd.DataFrame([summary])
    df_sens = pd.DataFrame(sens_rows)
    df_grid = pd.DataFrame(grid_rows)

    p_seed = output_dir / "tdf_vs_ou_unified_law_seed_scores.csv"
    p_boot = output_dir / "tdf_vs_ou_unified_law_bootstrap.csv"
    p_perm = output_dir / "tdf_vs_ou_unified_law_permutation.csv"
    p_sum = output_dir / "tdf_vs_ou_unified_law_robust_summary.csv"

    df_seed.to_csv(p_seed, index=False)
    df_boot.to_csv(p_boot, index=False)
    df_perm.to_csv(p_perm, index=False)
    # Summary + sensitivity tables in one CSV (top = scalar summary, then blank, then window sens, then grid sens)
    with open(p_sum, "w", encoding="utf-8") as fh:
        df_summary.to_csv(fh, index=False)
        fh.write("\n# window_sensitivity\n")
        df_sens.to_csv(fh, index=False)
        fh.write("\n# sweep_grid_sensitivity\n")
        df_grid.to_csv(fh, index=False)

    if save_plots:
        # 1) Histogram of delta_score
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        ax1.hist(deltas, bins=min(15, max(5, n_seeds // 2)), color="C0", edgecolor="k", alpha=0.85)
        ax1.axvline(0.0, color="k", linestyle="--", linewidth=1)
        ax1.axvline(mean_delta, color="C3", linewidth=2, label=f"mean={mean_delta:.3g}")
        ax1.set_xlabel(r"$\Delta$ score (TDF − OU)")
        ax1.set_ylabel("count")
        ax1.set_title("Unified-law score difference across seeds")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(output_dir / "tdf_vs_ou_robust_delta_hist.png", dpi=150)
        if not show:
            plt.close(fig1)

        # 2) Bootstrap: first seed violin TDF vs OU
        s0_tdf = df_boot[(df_boot["seed"] == 0) & (df_boot["model"] == "tdf")]
        s0_ou = df_boot[(df_boot["seed"] == 0) & (df_boot["model"] == "ou")]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        # Re-run bootstrap samples for plotting (seed 0 only) — cheap
        df_t0, df_o0 = collect_unified_law_sweeps(
            t,
            omega=omega,
            tau_freq=tau_freq,
            tau_noise_strength=tau_noise_strength,
            tau_c_values=tau_c_values,
            ou_correlation_times=ou_correlation_times,
            ou_sigma=ou_sigma,
            base_seed=int(seed_offset),
        )
        rng_v = np.random.default_rng(seed_offset + 999)
        b_t, _, bt_lo, bt_hi = bootstrap_unified_scores(
            df_t0, n_windows=n_windows, n_boot=n_boot, rng=rng_v
        )
        rng_v2 = np.random.default_rng(seed_offset + 998)
        b_o, _, bo_lo, bo_hi = bootstrap_unified_scores(
            df_o0, n_windows=n_windows, n_boot=n_boot, rng=rng_v2
        )
        ax2.violinplot(
            [b_t[np.isfinite(b_t)], b_o[np.isfinite(b_o)]],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
        )
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(["TDF", "OU"])
        ax2.set_ylabel("unified_score (bootstrap)")
        ax2.set_title(f"Bootstrap distributions (seed 0, B={n_boot})\n95% CI bars from table")
        pt_tdf = float(s0_tdf["point_score"].iloc[0]) if len(s0_tdf) else float("nan")
        pt_ou = float(s0_ou["point_score"].iloc[0]) if len(s0_ou) else float("nan")
        for i, (lo, hi, pt) in enumerate(
            ((bt_lo, bt_hi, pt_tdf), (bo_lo, bo_hi, pt_ou)),
            start=1,
        ):
            ax2.plot([i, i], [lo, hi], color="k", linewidth=3, solid_capstyle="round")
            ax2.plot(i, pt, "ws", markeredgecolor="k", markersize=8)
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(output_dir / "tdf_vs_ou_robust_bootstrap_violin.png", dpi=150)
        if not show:
            plt.close(fig2)

        # 3) Permutation nulls (seed 0)
        rng_p0 = np.random.default_rng(seed_offset + 111)
        r_t, n_t, pv_t = permutation_unified_scores(
            df_t0, n_windows=n_windows, n_perm=n_perm, rng=rng_p0
        )
        rng_p1 = np.random.default_rng(seed_offset + 222)
        r_o, n_o, pv_o = permutation_unified_scores(
            df_o0, n_windows=n_windows, n_perm=n_perm, rng=rng_p1
        )
        fig3, (axa, axb) = plt.subplots(1, 2, figsize=(9, 3.8))
        axa.hist(n_t, bins=30, color="C0", alpha=0.75, edgecolor="k")
        axa.axvline(r_t, color="C3", linewidth=2, label=f"real={r_t:.3g}, p={pv_t:.3g}")
        axa.set_title("TDF: permutation null")
        axa.legend(fontsize=8)
        axa.grid(True, alpha=0.3)
        axb.hist(n_o, bins=30, color="C1", alpha=0.75, edgecolor="k")
        axb.axvline(r_o, color="C3", linewidth=2, label=f"real={r_o:.3g}, p={pv_o:.3g}")
        axb.set_title("OU: permutation null")
        axb.legend(fontsize=8)
        axb.grid(True, alpha=0.3)
        fig3.suptitle(f"Unified score under shuffled y-pairings (seed 0, n={n_perm})")
        fig3.tight_layout()
        fig3.savefig(output_dir / "tdf_vs_ou_robust_permutation.png", dpi=150)
        if not show:
            plt.close(fig3)

        # 4) Window sensitivity
        if sens_rows:
            fig4, ax4 = plt.subplots(figsize=(5.5, 3.6))
            xs = [r["n_windows"] for r in sens_rows]
            ys = [r["mean_delta_score"] for r in sens_rows]
            ax4.plot(xs, ys, "o-", color="C2", linewidth=2, markersize=8)
            ax4.axhline(0.0, color="k", linestyle="--", linewidth=1)
            ax4.set_xticks(xs)
            ax4.set_xlabel("n_windows")
            ax4.set_ylabel("mean Δ (TDF − OU)")
            ax4.set_title("Window count sensitivity (all seeds)")
            ax4.grid(True, alpha=0.3)
            fig4.tight_layout()
            fig4.savefig(output_dir / "tdf_vs_ou_robust_window_sensitivity.png", dpi=150)
            if not show:
                plt.close(fig4)

    print("\n=== TDF vs OU unified-law robust summary ===")
    print(f"  n_seeds={n_seeds}, n_windows={n_windows}, n_boot={n_boot}, n_perm={n_perm}")
    print(f"  mean Δscore (TDF−OU): {mean_delta:.6g}  (std {std_delta:.6g})")
    print(f"  95% CI for mean Δ:    [{ci_lo:.6g}, {ci_hi:.6g}]")
    print(f"  TDF win rate (Δ>0):   {win_rate:.3f}")
    print(f"  permutation p (TDF) mean / median: {p_tdf_mean:.4f} / {p_tdf_median:.4f}")
    print(f"  permutation p (OU)  mean / median: {p_ou_mean:.4f} / {p_ou_median:.4f}")
    print(f"  mean train/test RMSE — TDF: {mean_tt_tdf:.6g}, OU: {mean_tt_ou:.6g}")
    print(f"  TDF better all window settings: {tdf_better_all_windows}")
    print(f"  TDF better all sweep grids:     {tdf_better_all_grids}")
    print()
    if robust:
        print(
            "CONCLUSION: TDF is **statistically robustly better** on this battery "
            "(mean Δ>0, CI excludes 0, win_rate≥0.7, median p_TDF<0.05, stable windows/grids)."
        )
    else:
        print(
            "CONCLUSION: Evidence does **not** meet all robustness criteria — "
            "see flags in summary CSV and individual p-values per seed."
        )
    print(f"\nSaved: {p_seed.name}, {p_boot.name}, {p_perm.name}, {p_sum.name}")

    if show and save_plots:
        plt.show()

    return {
        "seed_scores": df_seed,
        "bootstrap": df_boot,
        "permutation": df_perm,
        "summary": summary,
        "window_sensitivity": df_sens,
        "grid_sensitivity": df_grid,
        "paths": {
            "seed_scores": p_seed,
            "bootstrap": p_boot,
            "permutation": p_perm,
            "summary": p_sum,
        },
    }


if __name__ == "__main__":
    import sys

    fast = "--fast" in sys.argv
    run_tdf_vs_ou_unified_law_robust(fast_mode=fast)
