"""
χ-geometry hypotheses vs effective hidden-sector spectrum from kernel-mode extraction.

Consumes the output of :func:`analysis.tau_hidden_spectrum.extract_hidden_spectrum`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _records_from_spectrum_result(spectrum_result: dict[str, Any]) -> list[dict[str, Any]]:
    if "records" in spectrum_result and spectrum_result["records"]:
        return list(spectrum_result["records"])
    if "table" in spectrum_result and spectrum_result["table"] is not None:
        return spectrum_result["table"].to_dict("records")
    return []


def extract_usable_spectrum(
    spectrum_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Split oscillatory (damped) vs relaxational (soft) ``lambda_eff``, sorted by ``mode_index``.

    Oscillatory tower fits use modes with positive ``lambda_eff`` only (``ω²`` KK ladder).

    Returns
    -------
    dict
        ``oscillatory_records``, ``soft_records``, ``lambda_osc``, ``mode_numbers``,
        ``lambda_soft`` (scalar mean of soft modes or NaN), ``mode_indices_osc``.
    """
    rows = _records_from_spectrum_result(spectrum_result)
    soft_recs = sorted(
        [
            dict(r)
            for r in rows
            if str(r.get("best_model")) == "relaxation"
            and np.isfinite(float(r.get("lambda_eff", np.nan)))
        ],
        key=lambda r: int(r["mode_index"]),
    )
    osc_all = sorted(
        [
            dict(r)
            for r in rows
            if str(r.get("best_model")) == "damped_oscillator"
            and np.isfinite(float(r.get("lambda_eff", np.nan)))
        ],
        key=lambda r: int(r["mode_index"]),
    )
    osc_pos = [r for r in osc_all if float(r["lambda_eff"]) > 0.0]
    lambda_osc = np.array([float(r["lambda_eff"]) for r in osc_pos], dtype=float)
    n_os = lambda_osc.size
    mode_numbers = np.arange(1, n_os + 1, dtype=int) if n_os else np.array([], dtype=int)

    if soft_recs:
        lambda_soft = float(np.mean([float(r["lambda_eff"]) for r in soft_recs]))
    else:
        lambda_soft = float("nan")

    return {
        "oscillatory_records": osc_pos,
        "soft_records": soft_recs,
        "lambda_osc": lambda_osc,
        "mode_numbers": mode_numbers,
        "lambda_soft": lambda_soft,
        "mode_indices_osc": [int(r["mode_index"]) for r in osc_pos],
    }


def fit_flat_compact_spectrum(
    lambda_vals: np.ndarray | Sequence[float],
    mode_numbers: np.ndarray | Sequence[int] | None = None,
) -> dict[str, Any]:
    """Hypothesis A: ``λ_n = A n^2``."""
    y = np.asarray(lambda_vals, dtype=float).ravel()
    if y.size == 0:
        return _empty_fit("flat_compact")
    if mode_numbers is None:
        n = np.arange(1, y.size + 1, dtype=float)
    else:
        n = np.asarray(mode_numbers, dtype=float).ravel()
    if n.shape != y.shape:
        raise ValueError("mode_numbers must match lambda_vals length")
    mask = np.isfinite(y) & np.isfinite(n) & (n > 0)
    y, n = y[mask], n[mask]
    if y.size == 0:
        return _empty_fit("flat_compact")
    try:
        X = (n ** 2).reshape(-1, 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        A = float(coef[0])
        pred = A * n ** 2
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return _empty_fit("flat_compact")
    return {
        "model": "flat_compact",
        "A": A,
        "rmse": rmse,
        "lambda_pred": pred,
        "mode_numbers": n.astype(int),
    }


def fit_offset_compact_spectrum(
    lambda_vals: np.ndarray | Sequence[float],
    mode_numbers: np.ndarray | Sequence[int] | None = None,
) -> dict[str, Any]:
    """Hypothesis B: ``λ_n = λ₀ + A n^2``."""
    y = np.asarray(lambda_vals, dtype=float).ravel()
    if y.size == 0:
        return _empty_fit_offset()
    if mode_numbers is None:
        n = np.arange(1, y.size + 1, dtype=float)
    else:
        n = np.asarray(mode_numbers, dtype=float).ravel()
    if n.shape != y.shape:
        raise ValueError("mode_numbers must match lambda_vals length")
    mask = np.isfinite(y) & np.isfinite(n) & (n > 0)
    y, n = y[mask], n[mask]
    if y.size == 0:
        return _empty_fit_offset()
    try:
        X = np.column_stack([np.ones_like(n), n ** 2])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        lambda0, A = float(coef[0]), float(coef[1])
        pred = lambda0 + A * n ** 2
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return _empty_fit_offset()
    return {
        "model": "offset_compact",
        "lambda0": lambda0,
        "A": A,
        "rmse": rmse,
        "lambda_pred": pred,
        "mode_numbers": n.astype(int),
    }


def fit_warped_compact_spectrum(
    lambda_vals: np.ndarray | Sequence[float],
    mode_numbers: np.ndarray | Sequence[int] | None = None,
) -> dict[str, Any]:
    """
    Hypothesis C: ``λ_n = λ₀ + A n² + B/(n+1)``.

    Requires at least **three** oscillatory points; otherwise returns NaN parameters
    (two points leave the three-parameter model underdetermined).
    """
    y = np.asarray(lambda_vals, dtype=float).ravel()
    if y.size == 0:
        return _empty_fit_warped()
    if mode_numbers is None:
        n = np.arange(1, y.size + 1, dtype=float)
    else:
        n = np.asarray(mode_numbers, dtype=float).ravel()
    if n.shape != y.shape:
        raise ValueError("mode_numbers must match lambda_vals length")
    mask = np.isfinite(y) & np.isfinite(n) & (n > 0)
    y, n = y[mask], n[mask]
    if y.size == 0:
        return _empty_fit_warped()
    if y.size < 3:
        # three parameters need at least three tower points for a meaningful fit
        return _empty_fit_warped()
    try:
        X = np.column_stack([np.ones_like(n), n ** 2, 1.0 / (n + 1.0)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        lambda0, A, B = float(coef[0]), float(coef[1]), float(coef[2])
        pred = lambda0 + A * n ** 2 + B / (n + 1.0)
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return _empty_fit_warped()
    return {
        "model": "warped_compact",
        "lambda0": lambda0,
        "A": A,
        "B": B,
        "rmse": rmse,
        "lambda_pred": pred,
        "mode_numbers": n.astype(int),
    }


def _empty_fit(name: str) -> dict[str, Any]:
    return {
        "model": name,
        "A": float("nan"),
        "rmse": float("nan"),
        "lambda_pred": np.array([]),
        "mode_numbers": np.array([], dtype=int),
    }


def _empty_fit_offset() -> dict[str, Any]:
    d = _empty_fit("offset_compact")
    d["model"] = "offset_compact"
    d["lambda0"] = float("nan")
    return d


def _empty_fit_warped() -> dict[str, Any]:
    d = _empty_fit("warped_compact")
    d["model"] = "warped_compact"
    d["lambda0"] = float("nan")
    d["B"] = float("nan")
    return d


def compare_chi_geometry_models(spectrum_result: dict[str, Any]) -> dict[str, Any]:
    """
    Fit flat / offset / warped hypotheses to oscillatory tower; summarize soft ratio.

    Parameters
    ----------
    spectrum_result
        Output of :func:`analysis.tau_hidden_spectrum.extract_hidden_spectrum`.
    """
    spec = extract_usable_spectrum(spectrum_result)
    lam = spec["lambda_osc"]
    nvec = spec["mode_numbers"]

    flat = fit_flat_compact_spectrum(lam, nvec)
    off = fit_offset_compact_spectrum(lam, nvec)
    warp = fit_warped_compact_spectrum(lam, nvec)

    candidates = [
        ("flat_compact", float(flat["rmse"])),
        ("offset_compact", float(off["rmse"])),
        ("warped_compact", float(warp["rmse"])),
    ]
    finite = [(k, r) for k, r in candidates if np.isfinite(r)]
    if finite:
        best_model = min(finite, key=lambda x: x[1])[0]
    else:
        best_model = "none"

    soft_present = bool(spec["soft_records"]) and np.isfinite(spec["lambda_soft"])
    mean_osc = float(np.mean(lam)) if lam.size else float("nan")
    if soft_present and np.isfinite(mean_osc) and abs(mean_osc) > 1e-15:
        soft_ratio = float(spec["lambda_soft"] / mean_osc)
    else:
        soft_ratio = float("nan")

    return {
        "flat_compact": flat,
        "offset_compact": off,
        "warped_compact": warp,
        "best_model": best_model,
        "soft_mode_present": soft_present,
        "lambda_soft": spec["lambda_soft"],
        "soft_ratio": soft_ratio,
        "spectrum_split": spec,
    }


def interpret_chi_geometry(compare_result: dict[str, Any]) -> str:
    """Short narrative from :func:`compare_chi_geometry_models` output."""
    best = str(compare_result.get("best_model", "none"))
    parts: list[str] = []

    rmses: list[tuple[str, float]] = []
    for k in ("flat_compact", "offset_compact", "warped_compact"):
        d = compare_result.get(k, {})
        r = float(d.get("rmse", np.nan))
        if np.isfinite(r):
            rmses.append((k, r))
    rmses.sort(key=lambda x: x[1])
    clearly = False
    if len(rmses) >= 2:
        clearly = rmses[0][1] < 0.72 * rmses[1][1]
    elif len(rmses) == 1:
        clearly = True

    if best == "flat_compact":
        if clearly:
            parts.append("χ is compatible with a flat compact geometry.")
        else:
            parts.append(
                "A flat compact χ spectrum is competitive but not clearly preferred."
            )
    elif best == "offset_compact":
        parts.append(
            "χ shows a compact geometry with a mass gap / curvature offset."
        )
    elif best == "warped_compact":
        parts.append("χ likely has an effective warped/trapping structure.")
    else:
        parts.append(
            "No single χ-geometry hypothesis clearly dominates given the available tower modes."
        )

    soft_ok = bool(compare_result.get("soft_mode_present"))
    sr = float(compare_result.get("soft_ratio", np.nan))
    if soft_ok and np.isfinite(sr) and abs(sr) < 0.2:
        parts.append(
            "plus a soft hidden sector weakly coupled to observables."
        )

    return " ".join(parts)


def analyze_chi_geometry(
    spectrum_result: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    save_plots: bool = True,
    show: bool = False,
    csv_filename: str = "tau_chi_geometry_models.csv",
) -> dict[str, Any]:
    """
    Run model comparison, save CSV and figures under ``output_dir``.
    """
    cmp_out = compare_chi_geometry_models(spectrum_result)
    spec = cmp_out["spectrum_split"]
    lam = spec["lambda_osc"]
    nvec = spec["mode_numbers"].astype(float)
    interp = interpret_chi_geometry(cmp_out)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}

    # --- CSV ---
    rows: list[dict[str, Any]] = []
    for key in ("flat_compact", "offset_compact", "warped_compact"):
        d = cmp_out[key]
        row: dict[str, Any] = {
            "model": d.get("model", key),
            "is_best": key == cmp_out["best_model"],
            "rmse": d.get("rmse"),
            "A": d.get("A", np.nan),
        }
        if key != "flat_compact":
            row["lambda0"] = d.get("lambda0", np.nan)
        else:
            row["lambda0"] = np.nan
        if key == "warped_compact":
            row["B"] = d.get("B", np.nan)
        else:
            row["B"] = np.nan
        rows.append(row)
    df_m = pd.DataFrame(rows)
    csv_path = output_dir / csv_filename
    df_m.to_csv(csv_path, index=False)
    summary_stats = pd.DataFrame(
        [
            {
                "best_model": cmp_out["best_model"],
                "soft_mode_present": cmp_out["soft_mode_present"],
                "lambda_soft": cmp_out["lambda_soft"],
                "soft_ratio": cmp_out["soft_ratio"],
                "interpretation": interp,
            }
        ]
    )
    with open(csv_path, "a", encoding="utf-8") as fh:
        fh.write("\n")
    summary_stats.to_csv(csv_path, mode="a", index=False)

    if save_plots:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        if lam.size > 0:
            ax.scatter(
                nvec,
                lam,
                s=70,
                c="k",
                zorder=5,
                label=r"$\lambda_{\mathrm{eff}}$ (oscillatory)",
            )
            nn = np.linspace(
                max(0.5, float(np.min(nvec)) - 0.15),
                float(np.max(nvec)) + 0.45,
                150,
            )
            fc = cmp_out["flat_compact"]
            oc = cmp_out["offset_compact"]
            wc = cmp_out["warped_compact"]
            if np.isfinite(fc.get("A", np.nan)):
                ax.plot(
                    nn,
                    fc["A"] * nn**2,
                    "--",
                    linewidth=1.5,
                    label=rf"flat $A n^2$ (RMSE={fc['rmse']:.3g})",
                )
            if np.isfinite(oc.get("A", np.nan)) and np.isfinite(
                oc.get("lambda0", np.nan)
            ):
                ax.plot(
                    nn,
                    oc["lambda0"] + oc["A"] * nn**2,
                    "-.",
                    linewidth=1.5,
                    label=rf"offset $\lambda_0+A n^2$ (RMSE={oc['rmse']:.3g})",
                )
            if (
                np.isfinite(wc.get("A", np.nan))
                and np.isfinite(wc.get("lambda0", np.nan))
                and np.isfinite(wc.get("B", np.nan))
            ):
                ax.plot(
                    nn,
                    wc["lambda0"] + wc["A"] * nn**2 + wc["B"] / (nn + 1.0),
                    "-",
                    linewidth=1.5,
                    label=rf"warped (RMSE={wc['rmse']:.3g})",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No oscillatory tower modes (positive ω²)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel("mode number n (compact tower)")
        ax.set_ylabel(r"$\lambda_{\mathrm{eff}}$")
        ax.set_title("χ-geometry models vs oscillatory spectrum")
        if lam.size > 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p1 = output_dir / "chi_geometry_spectrum_models.png"
        fig.savefig(p1, dpi=150)
        figure_paths["spectrum_models"] = p1
        if not show:
            plt.close(fig)

    # Soft vs oscillatory bars
    if save_plots:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        labels: list[str] = []
        vals: list[float] = []
        if cmp_out["soft_mode_present"] and np.isfinite(cmp_out["lambda_soft"]):
            labels.append("soft (relax)")
            vals.append(float(cmp_out["lambda_soft"]))
        for i, (mi, lv) in enumerate(zip(spec["mode_indices_osc"], lam)):
            labels.append(f"osc n={i+1}\n(idx {mi})")
            vals.append(float(lv))
        if labels:
            xb = np.arange(len(labels))
            ax2.bar(xb, vals, color=["C1"] * (1 if cmp_out["soft_mode_present"] else 0) + ["C0"] * lam.size, edgecolor="k")
            ax2.set_xticks(xb)
            ax2.set_xticklabels(labels, fontsize=8)
            ax2.set_ylabel(r"$\lambda_{\mathrm{eff}}$")
            ax2.set_title(
                "Soft mode vs oscillatory tower "
                + (
                    f"(soft_ratio={cmp_out['soft_ratio']:.3g})"
                    if np.isfinite(cmp_out["soft_ratio"])
                    else ""
                )
            )
            ax2.grid(True, axis="y", alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No spectrum data", ha="center", va="center", transform=ax2.transAxes)
        fig2.tight_layout()
        p2 = output_dir / "chi_geometry_soft_mode.png"
        fig2.savefig(p2, dpi=150)
        figure_paths["soft_mode"] = p2
        if not show:
            plt.close(fig2)

    if show and save_plots:
        plt.show()

    return {
        **cmp_out,
        "interpretation": interp,
        "figure_paths": figure_paths if save_plots else {},
        "csv_path": csv_path,
    }


# Example (not executed):
# from analysis.tau_chi_geometry import analyze_chi_geometry
# out = analyze_chi_geometry(spectrum_result, output_dir="outputs")


if __name__ == "__main__":
    from scripts.pipeline_demo import step_chi_geometry

    step_chi_geometry(None)
