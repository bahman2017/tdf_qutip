"""
Effective hidden-sector eigenvalues from fitted kernel-mode field models.

Maps each PCA mode's best-fit effective dynamics to a scalar ``lambda_eff`` and
optionally compares oscillatory (damped) ``omega2`` values to a compact ``n^2`` spectrum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd


def _float_or_nan(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def records_from_field_fit(field_fit_result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build one record per fitted mode from :func:`fit_kernel_modes_to_field_equations` output.
    """
    rows: list[dict[str, Any]] = []
    for m in field_fit_result.get("mode_fits", []):
        best = str(m.get("best_model", "none"))
        d = m.get("damped_fit") or {}
        r = m.get("relax_fit") or {}
        gamma = _float_or_nan(d.get("gamma"))
        omega2 = _float_or_nan(d.get("omega2"))
        mu = _float_or_nan(r.get("mu"))

        if best == "damped_oscillator":
            lambda_eff = omega2
        elif best == "relaxation":
            lambda_eff = mu
        else:
            lambda_eff = float("nan")

        rows.append(
            {
                "mode_index": int(m["mode_index"]),
                "explained_variance": float(m.get("explained_variance", float("nan"))),
                "best_model": best,
                "gamma": gamma,
                "omega2": omega2,
                "mu": mu,
                "lambda_eff": float(lambda_eff),
            }
        )
    return rows


def pairwise_lambda_ratios_oscillatory(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, list[int]]:
    """
    Oscillatory modes only, sorted by ``mode_index``.

    Returns
    -------
    ratio_matrix
        ``R[j, i] = lambda_j / lambda_i`` using the sorted oscillatory sequence.
    mode_indices
        Corresponding ``mode_index`` for each row/column of ``R``.
    """
    osc = [
        dict(r)
        for r in records
        if r.get("best_model") == "damped_oscillator"
        and np.isfinite(float(r.get("lambda_eff", np.nan)))
        and float(r["lambda_eff"]) != 0.0
    ]
    osc.sort(key=lambda row: int(row["mode_index"]))
    idxs = [int(r["mode_index"]) for r in osc]
    lams = np.array([float(r["lambda_eff"]) for r in osc], dtype=float)
    n = lams.size
    R = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if lams[i] != 0.0:
                R[j, i] = lams[j] / lams[i]
    return R, idxs


def compare_to_compact_spectrum(
    lambda_values: np.ndarray | Sequence[float],
    mode_numbers: np.ndarray | Sequence[int] | None = None,
) -> dict[str, Any]:
    """
    Fit ``lambda_n ≈ A * n^2`` with least squares (``n`` = compact KK tower index).

    Parameters
    ----------
    lambda_values
        Effective eigenvalues for oscillatory modes (e.g. ``omega2``), one per mode.
    mode_numbers
        Mode labels ``n``; default ``1, 2, … len(lambda_values)``.

    Returns
    -------
    dict
        ``A``, ``rmse``, ``lambda_pred``, ``mode_numbers``, ``n_modes``.
    """
    y = np.asarray(lambda_values, dtype=float).ravel()
    if y.size == 0:
        return {
            "A": float("nan"),
            "rmse": float("nan"),
            "lambda_pred": np.array([]),
            "mode_numbers": np.array([], dtype=int),
            "n_modes": 0,
        }

    if mode_numbers is None:
        nvec = np.arange(1, y.size + 1, dtype=float)
    else:
        nvec = np.asarray(mode_numbers, dtype=float).ravel()
    if nvec.shape != y.shape:
        raise ValueError("mode_numbers must match length of lambda_values")

    mask = np.isfinite(y) & np.isfinite(nvec) & (nvec > 0)
    y = y[mask]
    nvec = nvec[mask]
    if y.size == 0:
        return {
            "A": float("nan"),
            "rmse": float("nan"),
            "lambda_pred": np.array([]),
            "mode_numbers": nvec.astype(int),
            "n_modes": 0,
        }

    x = nvec ** 2
    A, _, _, _ = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)
    A = float(A[0])
    pred = A * x
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {
        "A": A,
        "rmse": rmse,
        "lambda_pred": pred,
        "mode_numbers": nvec.astype(int),
        "n_modes": int(y.size),
    }


def interpret_hidden_spectrum(analysis: dict[str, Any]) -> str:
    """
    Heuristic: increasing oscillatory ``lambda_eff`` and low compact-fit RMSE → compact-like.
    """
    osc_lams = np.asarray(analysis.get("oscillatory_lambda_eff", []), dtype=float)
    compact_rmse = float(analysis.get("compact_rmse", np.nan))
    n_osc = int(analysis.get("n_oscillatory", 0))

    if n_osc < 2:
        return (
            "Oscillatory kernel modes do not match a simple compact spectrum; "
            "more general hidden geometry may be required."
        )

    order_ok = bool(np.all(np.diff(osc_lams) >= -1e-9))
    if order_ok and len(osc_lams) >= 2:
        rel = compact_rmse / (float(np.mean(osc_lams)) + 1e-12)
        small = rel < 0.35 and np.isfinite(compact_rmse)
    else:
        small = False

    if order_ok and small:
        return (
            "Oscillatory kernel modes are compatible with a compact 5D spectrum."
        )

    return (
        "Oscillatory kernel modes do not match a simple compact spectrum; "
        "more general hidden geometry may be required."
    )


def extract_hidden_spectrum(
    field_fit_result: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    save_plots: bool = True,
    show: bool = False,
    csv_filename: str = "tau_hidden_spectrum.csv",
) -> dict[str, Any]:
    """
    Build spectrum table, pairwise ratios, compact-spectrum fit, plots, and CSV.

    Parameters
    ----------
    field_fit_result
        Output of :func:`analysis.tau_mode_field_fit.fit_kernel_modes_to_field_equations`.

    Returns
    -------
    dict
        ``records``, ``table`` (DataFrame), ``pairwise_ratio_matrix``,
        ``pairwise_mode_indices``, ``compact_fit``, ``interpretation``, ``figure_paths``, ``csv_path``.
    """
    records = records_from_field_fit(field_fit_result)
    df = pd.DataFrame(records)

    R, idxs_pair = pairwise_lambda_ratios_oscillatory(records)
    n_osc_r = R.shape[0]
    R_hyp = np.full((n_osc_r, n_osc_r), np.nan, dtype=float)
    if n_osc_r > 0:
        nn = np.arange(1, n_osc_r + 1, dtype=float)
        for i in range(n_osc_r):
            for j in range(n_osc_r):
                if nn[i] != 0:
                    R_hyp[j, i] = (nn[j] / nn[i]) ** 2

    osc_sorted = sorted(
        [
            r
            for r in records
            if r["best_model"] == "damped_oscillator"
            and np.isfinite(r["lambda_eff"])
        ],
        key=lambda r: int(r["mode_index"]),
    )
    osc_positive = [r for r in osc_sorted if float(r["lambda_eff"]) > 0.0]
    osc_lams = np.array([float(r["lambda_eff"]) for r in osc_positive], dtype=float)
    n_osc = int(osc_lams.size)

    compact = compare_to_compact_spectrum(osc_lams) if n_osc else compare_to_compact_spectrum([])

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}

    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False)

    if save_plots:
        # Bar: lambda_eff by mode_index
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(records))
        heights = [float(r["lambda_eff"]) if np.isfinite(r["lambda_eff"]) else 0.0 for r in records]
        labels = [str(r["mode_index"]) for r in records]
        colors = []
        for r in records:
            bm = r["best_model"]
            if bm == "damped_oscillator":
                colors.append("C0")
            elif bm == "relaxation":
                colors.append("C1")
            else:
                colors.append("0.7")
        ax.bar(x, heights, color=colors, edgecolor="k")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("mode_index (PCA order)")
        ax.set_ylabel(r"$\lambda_{\mathrm{eff}}$")
        ax.set_title("Effective hidden spectrum from kernel mode fits")
        ax.grid(True, axis="y", alpha=0.3)

        leg = [
            Patch(facecolor="C0", edgecolor="k", label="damped → ω²"),
            Patch(facecolor="C1", edgecolor="k", label="relax → μ"),
            Patch(facecolor="0.7", edgecolor="k", label="none"),
        ]
        ax.legend(handles=leg, loc="best", fontsize=8)
        fig.tight_layout()
        p1 = output_dir / "hidden_spectrum_bar.png"
        fig.savefig(p1, dpi=150)
        figure_paths["bar"] = p1
        if not show:
            plt.close(fig)

        # Compact fit: oscillatory only
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if n_osc > 0:
            nvec = np.asarray(compact["mode_numbers"], dtype=float).ravel()
            if nvec.size != osc_lams.size:
                nvec = np.arange(1, n_osc + 1, dtype=float)
            ax2.scatter(nvec, osc_lams, s=60, c="C0", edgecolors="k", zorder=3, label=r"$\lambda_{\mathrm{eff}}$")
            if n_osc > 0 and np.isfinite(compact.get("A", np.nan)):
                nn = np.linspace(max(0.5, float(np.min(nvec)) - 0.2), float(np.max(nvec)) + 0.5, 120)
                ax2.plot(nn, compact["A"] * nn**2, "k--", linewidth=1.5, label=rf"fit $A n^2$, A={compact['A']:.4g}")
            ax2.set_xlabel("mode number n (compact tower)")
            ax2.set_ylabel(r"$\lambda_{\mathrm{eff}} = \omega^2$")
            ax2.set_title(
                f"Compact spectrum hypothesis (RMSE={compact['rmse']:.4g})"
            )
            ax2.legend(loc="best", fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No oscillatory modes", ha="center", va="center", transform=ax2.transAxes)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        p2 = output_dir / "hidden_spectrum_compact_fit.png"
        fig2.savefig(p2, dpi=150)
        figure_paths["compact_fit"] = p2
        if not show:
            plt.close(fig2)

    if show and save_plots:
        plt.show()

    analysis_stats = {
        "oscillatory_lambda_eff": osc_lams,
        "compact_rmse": compact.get("rmse", float("nan")),
        "n_oscillatory": n_osc,
    }
    interpretation = interpret_hidden_spectrum(analysis_stats)

    return {
        "records": records,
        "table": df,
        "pairwise_ratio_matrix": R,
        "pairwise_mode_indices": idxs_pair,
        "compact_hypothesis_ratio_matrix": R_hyp,
        "pairwise_note": "R[j,i] = lambda_j / lambda_i; R_hyp[j,i] = (n_j/n_i)^2 with n=1..K",
        "compact_fit": compact,
        "interpretation": interpretation,
        "figure_paths": figure_paths if save_plots else {},
        "csv_path": csv_path,
    }


# Example (not executed):
# from analysis.tau_hidden_spectrum import extract_hidden_spectrum
# out = extract_hidden_spectrum(field_fit_result, output_dir="outputs")
