"""
τ-model discrimination: spectrum + interference + shared stochastic decoherence baseline.

Framed as **phenomenological comparison** and discrimination vs ad-hoc noise models,
not as confirmation of a fundamental TDF hypothesis.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from core.tau_model import linear_tau
from analysis.tau_model_spectrum import (
    analyze_tau_models,
    tau_models_specs,
    tau_models_specs_v2,
)
from experiments.decoherence import (
    compare_structured_stochastic_tdf_vs_fitted_lindblad,
    compare_tdf_vs_fitted_lindblad,
)
from experiments.interference import analyze_interference_from_tau_fields


def _aggregate_deco_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Means over noise strengths for stochastic-τ vs Lindblad diagnostics."""
    if not rows:
        return {
            "final_coherence": float("nan"),
            "lindblad_rmse_full": float("nan"),
            "long_time_difference": float("nan"),
            "oscillation_residual": float("nan"),
        }
    return {
        "final_coherence": float(
            np.mean([float(r["tdf_final_coherence"]) for r in rows])
        ),
        "lindblad_rmse_full": float(np.mean([float(r["rmse_full"]) for r in rows])),
        "long_time_difference": float(
            np.mean([float(r["long_time_difference"]) for r in rows])
        ),
        "oscillation_residual": float(
            np.mean([float(r["oscillation_residual"]) for r in rows])
        ),
    }


def _worst_sigma_by_rmse(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """(noise_strength, rmse_full) with largest full-curve RMSE."""
    if not rows:
        return float("nan"), float("nan")
    i = int(np.argmax([float(r["rmse_full"]) for r in rows]))
    return float(rows[i]["noise_strength"]), float(rows[i]["rmse_full"])


def run_tdf_discrimination_summary(
    t: np.ndarray,
    output_dir: str | Path,
    *,
    decoherence_result: dict[str, Any] | None = None,
    ref_tau_label: str = "linear_tau",
) -> tuple[Path, Path]:
    """
    Build per-τ-model spectrum and interference metrics; attach pooled stochastic-τ
    decoherence metrics (same on every row).

    Interference: τ_A = ``linear_tau`` (reference), τ_B = each benchmark model with
    the same kwargs as :func:`analysis.tau_model_spectrum.tau_models_specs`.

    Parameters
    ----------
    decoherence_result
        Optional return dict of :func:`compare_tdf_vs_fitted_lindblad`. If omitted,
        that function is run with ``plot=False`` (writes ``decoherence_comparison.csv``).
    ref_tau_label
        Name of the reference τ used for Δτ (default linear).

    Returns
    -------
    csv_path, md_path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, spec_rows = analyze_tau_models(t)

    ref_kw = {"omega": 1.0}
    tau_a = linear_tau(t, **ref_kw)

    specs = tau_models_specs()
    inter_by_name: dict[str, dict[str, Any]] = {}
    for name, tau_fn, kw in specs:
        tau_b = tau_fn(t, **kw)
        inter_by_name[name] = analyze_interference_from_tau_fields(
            t,
            tau_a,
            tau_b,
            plot=False,
            tau_A_label=rf"$\tau_A$ ({ref_tau_label})",
            tau_B_label=rf"$\tau_B$ ({name})",
        )

    if decoherence_result is None:
        decoherence_result = compare_tdf_vs_fitted_lindblad(
            t=t,
            output_dir=output_dir,
            plot=False,
            show=False,
        )
    dec_rows = decoherence_result.get("rows", [])
    dec_agg = _aggregate_deco_rows(dec_rows)
    worst_sigma, worst_rmse = _worst_sigma_by_rmse(dec_rows)

    combined: list[dict[str, Any]] = []
    for r in spec_rows:
        name = str(r["name"])
        ir = inter_by_name[name]
        combined.append(
            {
                "tau_model": name,
                "dominant_frequency": r["dominant_freq"],
                "spectral_entropy": r["spectral_entropy"],
                "bandwidth_90pct_power": r["bandwidth_90"],
                "mean_abs_error": ir["mean_abs_error"],
                "overlap_correlation": ir["overlap_correlation"],
                "final_coherence_stochastic": dec_agg["final_coherence"],
                "lindblad_rmse_full_mean_over_sigma": dec_agg["lindblad_rmse_full"],
                "long_time_difference_mean_over_sigma": dec_agg["long_time_difference"],
                "oscillation_residual_mean_over_sigma": dec_agg["oscillation_residual"],
                "decoherence_note": (
                    "Identical for all rows: mean over noise_strengths in "
                    "stochastic_tau ensemble vs best-fit Lindblad per σ; "
                    "not specific to deterministic τ families."
                ),
            }
        )

    csv_path = output_dir / "tdf_discrimination_summary.csv"
    fieldnames = [
        "tau_model",
        "dominant_frequency",
        "spectral_entropy",
        "bandwidth_90pct_power",
        "mean_abs_error",
        "overlap_correlation",
        "final_coherence_stochastic",
        "lindblad_rmse_full_mean_over_sigma",
        "long_time_difference_mean_over_sigma",
        "oscillation_residual_mean_over_sigma",
        "decoherence_note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in combined:
            w.writerow(row)

    md_path = output_dir / "tdf_discrimination_summary.md"
    _write_markdown_report(
        md_path,
        combined,
        ref_tau_label=ref_tau_label,
        worst_sigma=worst_sigma,
        worst_rmse=worst_rmse,
        n_ensemble=int(dec_rows[0]["n_ensemble"]) if dec_rows else 0,
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return csv_path, md_path


def _write_markdown_report(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    ref_tau_label: str,
    worst_sigma: float,
    worst_rmse: float,
    n_ensemble: int,
) -> None:
    """Cautious discrimination narrative + rankings."""

    def _f(x: Any) -> str:
        v = float(x)
        if np.isnan(v):
            return "nan"
        return f"{v:.6g}"

    lines: list[str] = []
    lines.append("# τ-model discrimination summary")
    lines.append("")
    lines.append(
        "This note **does not** establish time-domain field (TDF) mechanics or rule out "
        "equivalent descriptions. It is a **discrimination-style** checklist: how much "
        "structure each *ad hoc* τ(t) template exhibits in (A) Ramsey-channel spectrum, "
        "(B) interference agreement between cos(Δτ) and quantum overlap with a fixed "
        "reference τ, and (C) a **separate** stochastic-τ ensemble benchmark versus a "
        "best-fit Markovian dephasing model. Section C is **not** tied to individual "
        "deterministic τ families—values repeat across rows by construction."
    )
    lines.append("")
    lines.append("## Combined metrics")
    lines.append("")
    lines.append(
        "| τ model | f_dom | S_spec | BW_90 | |Δ|_mean | r_overlap | "
        "C_final (stoch.) | RMSE_L (mean σ) | |Δ|_tail (mean) | osc_res (mean) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        lines.append(
            f"| {r['tau_model']} | {_f(r['dominant_frequency'])} | "
            f"{_f(r['spectral_entropy'])} | {_f(r['bandwidth_90pct_power'])} | "
            f"{_f(r['mean_abs_error'])} | {_f(r['overlap_correlation'])} | "
            f"{_f(r['final_coherence_stochastic'])} | "
            f"{_f(r['lindblad_rmse_full_mean_over_sigma'])} | "
            f"{_f(r['long_time_difference_mean_over_sigma'])} | "
            f"{_f(r['oscillation_residual_mean_over_sigma'])} |"
        )
    lines.append("")
    lines.append(
        f"- **Reference τ for interference:** `{ref_tau_label}` vs each row's τ_B. "
        f"**Stochastic decoherence:** N_ensemble={n_ensemble}, metrics averaged over "
        f"the σ grid in `decoherence_comparison.csv`."
    )
    lines.append("")

    # Rankings (spectrum: max entropy; interference: min MAE among τ_B ≠ reference)
    best_ent = max(rows, key=lambda x: float(x["spectral_entropy"]))
    non_ref = [r for r in rows if r["tau_model"] != ref_tau_label]
    by_mae = sorted(non_ref, key=lambda x: float(x["mean_abs_error"]))
    by_r = sorted(
        non_ref,
        key=lambda x: float(x["overlap_correlation"])
        if not np.isnan(float(x["overlap_correlation"]))
        else -1e9,
        reverse=True,
    )

    lines.append("## Rankings (descriptive only)")
    lines.append("")
    lines.append(
        f"- **Richest Ramsey ⟨σ_x⟩ spectrum (largest spectral entropy):** "
        f"`{best_ent['tau_model']}` (S_spec ≈ {_f(best_ent['spectral_entropy'])} nats)."
    )
    lines.append(
        f"- **Strongest interference agreement (lowest mean |cos(Δτ) − overlap|, "
        f"excluding `{ref_tau_label}` where Δτ≡0):** "
        f"`{by_mae[0]['tau_model']}` (MAE ≈ {_f(by_mae[0]['mean_abs_error'])}). "
        f"The `{ref_tau_label}` row is a numerical self-check only."
    )
    lines.append(
        f"- **Highest overlap correlation (among non-reference τ_B):** `{by_r[0]['tau_model']}` "
        f"(r ≈ {_f(by_r[0]['overlap_correlation'])}). "
        "When MAE and r disagree slightly, treat both as noisy phenomenological guides."
    )
    lines.append(
        f"- **Largest Lindblad mismatch (among σ in the stochastic benchmark):** "
        f"σ ≈ {_f(worst_sigma)} with full-curve RMSE ≈ {_f(worst_rmse)} "
        "(same underlying curve for all deterministic τ rows; see "
        "`outputs/decoherence_comparison.csv` for per-σ detail)."
    )
    lines.append("")

    # Unified family: heuristic score (no proof)
    def _score(row: dict[str, Any]) -> float:
        s_ent = float(row["spectral_entropy"])
        mae = float(row["mean_abs_error"])
        r_ov = float(row["overlap_correlation"])
        if np.isnan(r_ov):
            r_ov = 0.0
        # normalize crudely: reward entropy & correlation, penalize MAE (scale ~1)
        return s_ent + 2.0 * r_ov - 5.0 * mae

    ranked = sorted(rows, key=_score, reverse=True)
    lines.append("## One τ family for “everything”?")
    lines.append("")
    lines.append(
        "Spectrum and interference **do** respond to the deterministic τ shape. "
        "The Lindblad comparison, however, probes **stochastic** τ—a different protocol. "
        "Therefore there is **no single row-wise score** that fairly “explains all three” "
        "without mixing assumptions."
    )
    lines.append("")
    lines.append(
        "A **purely heuristic** combined ordering (entropy + correlation − scaled MAE; "
        "decoherence not included) would place models as: "
        + ", ".join(f"`{r['tau_model']}`" for r in ranked)
        + ". This is **not** a statistical test and does not favor TDF over other "
        "parameterizations of effective Hamiltonians or noise."
    )
    lines.append("")
    lines.append(
        "**Takeaway:** Use this file to **compare ad hoc τ templates** under fixed numerical "
        "settings. Claims beyond phenomenological discrimination require independent "
        "constraints (data, identifiable parameters, and alternative noise models)."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


STRUCTURED_STOCHASTIC = "structured_stochastic_tau"


def run_tdf_discrimination_summary_v2(
    t: np.ndarray,
    output_dir: str | Path,
    *,
    decoherence_result: dict[str, Any] | None = None,
    structured_stochastic_decoherence: dict[str, Any] | None = None,
    ref_tau_label: str = "linear_tau",
) -> tuple[Path, Path]:
    """
    Like :func:`run_tdf_discrimination_summary`, but includes ``structured_stochastic_tau``
    in spectrum and interference, and assigns **row-specific** decoherence metrics:
    the combined model uses :func:`compare_structured_stochastic_tdf_vs_fitted_lindblad`;
    other rows keep the pooled ``stochastic_tau`` vs Lindblad means.

    Writes ``tdf_discrimination_summary_v2.csv`` and ``tdf_discrimination_summary_v2.md``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs_v2 = tau_models_specs_v2()
    _, spec_rows = analyze_tau_models(t, models=specs_v2)

    ref_kw = {"omega": 1.0}
    tau_a = linear_tau(t, **ref_kw)

    inter_by_name: dict[str, dict[str, Any]] = {}
    for name, tau_fn, kw in specs_v2:
        tau_b = tau_fn(t, **kw)
        inter_by_name[name] = analyze_interference_from_tau_fields(
            t,
            tau_a,
            tau_b,
            plot=False,
            tau_A_label=rf"$\tau_A$ ({ref_tau_label})",
            tau_B_label=rf"$\tau_B$ ({name})",
        )

    if decoherence_result is None:
        decoherence_result = compare_tdf_vs_fitted_lindblad(
            t=t,
            output_dir=output_dir,
            plot=False,
            show=False,
        )
    if structured_stochastic_decoherence is None:
        structured_stochastic_decoherence = compare_structured_stochastic_tdf_vs_fitted_lindblad(
            t=t,
            omega=1.0,
            freq=3.0,
        )

    dec_rows = decoherence_result.get("rows", [])
    dec_agg = _aggregate_deco_rows(dec_rows)
    worst_sigma, worst_rmse = _worst_sigma_by_rmse(dec_rows)

    ss_rows = structured_stochastic_decoherence.get("rows", [])
    ss_agg = _aggregate_deco_rows(ss_rows)
    ss_worst_sigma, ss_worst_rmse = _worst_sigma_by_rmse(ss_rows)

    baseline_note = (
        "Mean over σ: stochastic_tau(ω, σ) ensemble vs best-fit Lindblad per σ "
        "(same for all deterministic τ rows)."
    )
    ss_note = (
        "Mean over σ: structured_stochastic_tau(ω=1, ν=3, noise_strength=σ) ensemble "
        "vs best-fit Lindblad per σ (row-specific combined ansatz)."
    )

    combined: list[dict[str, Any]] = []
    for r in spec_rows:
        name = str(r["name"])
        ir = inter_by_name[name]
        if name == STRUCTURED_STOCHASTIC:
            d_final = ss_agg["final_coherence"]
            d_rmse = ss_agg["lindblad_rmse_full"]
            d_lt = ss_agg["long_time_difference"]
            d_osc = ss_agg["oscillation_residual"]
            note = ss_note
        else:
            d_final = dec_agg["final_coherence"]
            d_rmse = dec_agg["lindblad_rmse_full"]
            d_lt = dec_agg["long_time_difference"]
            d_osc = dec_agg["oscillation_residual"]
            note = baseline_note

        combined.append(
            {
                "tau_model": name,
                "dominant_frequency": r["dominant_freq"],
                "spectral_entropy": r["spectral_entropy"],
                "bandwidth_90pct_power": r["bandwidth_90"],
                "mean_abs_error": ir["mean_abs_error"],
                "overlap_correlation": ir["overlap_correlation"],
                "final_coherence_stochastic": d_final,
                "lindblad_rmse_full_mean_over_sigma": d_rmse,
                "long_time_difference_mean_over_sigma": d_lt,
                "oscillation_residual_mean_over_sigma": d_osc,
                "decoherence_note": note,
            }
        )

    csv_path = output_dir / "tdf_discrimination_summary_v2.csv"
    fieldnames = [
        "tau_model",
        "dominant_frequency",
        "spectral_entropy",
        "bandwidth_90pct_power",
        "mean_abs_error",
        "overlap_correlation",
        "final_coherence_stochastic",
        "lindblad_rmse_full_mean_over_sigma",
        "long_time_difference_mean_over_sigma",
        "oscillation_residual_mean_over_sigma",
        "decoherence_note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in combined:
            w.writerow(row)

    md_path = output_dir / "tdf_discrimination_summary_v2.md"
    _write_markdown_report_v2(
        md_path,
        combined,
        ref_tau_label=ref_tau_label,
        baseline_worst_sigma=worst_sigma,
        baseline_worst_rmse=worst_rmse,
        ss_worst_sigma=ss_worst_sigma,
        ss_worst_rmse=ss_worst_rmse,
        baseline_rmse_mean=dec_agg["lindblad_rmse_full"],
        ss_rmse_mean=ss_agg["lindblad_rmse_full"],
        n_ensemble=int(dec_rows[0]["n_ensemble"]) if dec_rows else 0,
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return csv_path, md_path


def _write_markdown_report_v2(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    ref_tau_label: str,
    baseline_worst_sigma: float,
    baseline_worst_rmse: float,
    ss_worst_sigma: float,
    ss_worst_rmse: float,
    baseline_rmse_mean: float,
    ss_rmse_mean: float,
    n_ensemble: int,
) -> None:
    """V2 report: per-row decoherence where applicable + structured_stochastic highlight."""

    def _f(x: Any) -> str:
        v = float(x)
        if np.isnan(v):
            return "nan"
        return f"{v:.6g}"

    lines: list[str] = []
    lines.append("# τ-model discrimination summary (v2)")
    lines.append("")
    lines.append(
        "Phenomenological comparison only—not a proof of TDF. V2 adds "
        f"**`{STRUCTURED_STOCHASTIC}`**: τ(t)=ωt+sin(νt)+ξ(t) with fixed **seed** for "
        "single-trajectory spectrum/interference, and a **dedicated** ensemble-ρ "
        "decoherence benchmark vs best-fit Lindblad for that same functional form "
        "(noise strength σ swept as in the baseline experiment). Other τ rows still "
        "show the **pooled** `stochastic_tau` vs Lindblad means for columns (C)."
    )
    lines.append("")
    lines.append("## Combined metrics")
    lines.append("")
    lines.append(
        "| τ model | f_dom | S_spec | BW_90 | |Δ|_mean | r_overlap | "
        "C_final | RMSE_L (mean σ) | |Δ|_tail (mean) | osc_res (mean) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['tau_model']} | {_f(r['dominant_frequency'])} | "
            f"{_f(r['spectral_entropy'])} | {_f(r['bandwidth_90pct_power'])} | "
            f"{_f(r['mean_abs_error'])} | {_f(r['overlap_correlation'])} | "
            f"{_f(r['final_coherence_stochastic'])} | "
            f"{_f(r['lindblad_rmse_full_mean_over_sigma'])} | "
            f"{_f(r['long_time_difference_mean_over_sigma'])} | "
            f"{_f(r['oscillation_residual_mean_over_sigma'])} |"
        )
    lines.append("")
    lines.append(
        f"- **Reference τ for interference:** `{ref_tau_label}` vs each τ_B. "
        f"**Ensemble size:** N={n_ensemble}. "
        f"**`{STRUCTURED_STOCHASTIC}` row:** decoherence columns use the structured+noise "
        f"ensemble; other rows use mean-over-σ from the baseline `stochastic_tau` run "
        f"(see `decoherence_note` in CSV)."
    )
    lines.append("")

    # Rankings
    best_ent = max(rows, key=lambda x: float(x["spectral_entropy"]))
    non_ref = [r for r in rows if r["tau_model"] != ref_tau_label]
    by_mae = sorted(non_ref, key=lambda x: float(x["mean_abs_error"]))
    by_r = sorted(
        non_ref,
        key=lambda x: float(x["overlap_correlation"])
        if not np.isnan(float(x["overlap_correlation"]))
        else -1e9,
        reverse=True,
    )

    lines.append("## Rankings (descriptive only)")
    lines.append("")
    lines.append(
        f"- **Richest spectrum (largest S_spec):** `{best_ent['tau_model']}` "
        f"(≈ {_f(best_ent['spectral_entropy'])} nats)."
    )
    lines.append(
        f"- **Strongest interference (lowest MAE, τ_B ≠ `{ref_tau_label}`):** "
        f"`{by_mae[0]['tau_model']}` (MAE ≈ {_f(by_mae[0]['mean_abs_error'])})."
    )
    lines.append(
        f"- **Highest overlap correlation (non-reference τ_B):** `{by_r[0]['tau_model']}` "
        f"(r ≈ {_f(by_r[0]['overlap_correlation'])})."
    )
    lines.append(
        f"- **Baseline `stochastic_tau` — worst σ by full RMSE:** σ ≈ {_f(baseline_worst_sigma)} "
        f"(RMSE ≈ {_f(baseline_worst_rmse)})."
    )
    lines.append(
        f"- **`{STRUCTURED_STOCHASTIC}` ensemble — worst σ by full RMSE:** "
        f"σ ≈ {_f(ss_worst_sigma)} (RMSE ≈ {_f(ss_worst_rmse)})."
    )
    lines.append("")

    # Ranks for structured_stochastic within spectrum / interference
    ss_row = next((r for r in rows if r["tau_model"] == STRUCTURED_STOCHASTIC), None)
    if ss_row is not None:
        ent_sorted = sorted(rows, key=lambda x: float(x["spectral_entropy"]), reverse=True)
        rank_ent = 1 + ent_sorted.index(ss_row)
        mae_sorted = sorted(non_ref, key=lambda x: float(x["mean_abs_error"]))
        rank_mae = 1 + mae_sorted.index(ss_row)

        lines.append(f"## `{STRUCTURED_STOCHASTIC}` — all three tests")
        lines.append("")
        lines.append(
            f"- **(A) Spectrum:** rank **{rank_ent}/{len(rows)}** by spectral entropy "
            f"(higher is richer structure in this FFT diagnostic)."
        )
        lines.append(
            f"- **(B) Interference:** rank **{rank_mae}/{len(non_ref)}** among non-reference "
            f"τ_B by mean |cos(Δτ) − overlap| (1 = best agreement)."
        )
        lines.append(
            f"- **(C) Decoherence:** mean full-curve RMSE vs best-fit Lindblad ≈ {_f(ss_rmse_mean)} "
            f"(structured+noise ensemble); baseline `stochastic_tau` pool mean ≈ {_f(baseline_rmse_mean)}. "
            "Lower RMSE means the Markovian dephasing surrogate tracks the ensemble curve more closely "
            "on average—**not** that the physics is Lindbladian."
        )
        lines.append("")
        lines.append(
            "**Cautious read:** High spectral entropy does not imply a “best” τ field. Interference uses "
            f"a **linear** reference; large |Δτ| from the stochastic part can **hurt** (B) even when "
            f"(A) is rich. Here: (A) rank **{rank_ent}/{len(rows)}**, (B) rank **{rank_mae}/{len(non_ref)}** "
            "among non-reference τ_B. (C) RMSE often tracks the baseline when both ensembles share the same "
            "σ grid—compare protocols, not only scalars. Not proof of TDF."
        )
        lines.append("")

    lines.append("## Does one combined τ “win” all three?")
    lines.append("")
    lines.append(
        f"The same ansatz may lead in **(A)** but not in **(B)** (as for `{STRUCTURED_STOCHASTIC}` vs "
        f"`{ref_tau_label}` here). Treat the three blocks as **different probes**: FFT structure on one "
        "trajectory, cos(Δτ) vs overlap for a chosen τ pair, and ensemble ρ vs a fitted Markovian channel. "
        "Uniform excellence is neither required nor sufficient for a useful discrimination test."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
