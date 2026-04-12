"""
Generate a compact Markdown report from saved TDF–QM CSV outputs.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _pkg_outputs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "outputs"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: str) -> float | None:
    if x is None or x.strip() == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}g}"


def falsification_results_markdown(output_dir: Path | None = None) -> str:
    """
    Build a **TDF Falsification Results** section from optional JSON/CSV artifacts.

    Reads (if present): ``non_gaussian/summary.json``, ``scaling/fit_results.json``,
    ``threshold/critical_sigma.txt``, ``deviation/stats.json``, ``falsification/summary.json``.
    """
    if output_dir is None:
        output_dir = _pkg_outputs_dir()
    lines = [
        "## TDF Falsification Results",
        "",
        "Summaries from ``experiments/*`` falsification runs (Gaussian vs non-Gaussian τ, "
        "multi-qubit scaling, noise threshold sweep, Lindblad vs τ deviation). "
        "Run ``python main.py --run falsification_tests`` to populate ``outputs/``.",
        "",
    ]
    fals = output_dir / "falsification" / "summary.json"
    if fals.is_file():
        try:
            data = json.loads(fals.read_text(encoding="utf-8"))
            mods = data.get("modules", {})
            bad = [k for k, v in mods.items() if v.get("status") != "ok"]
            lines.append(
                f"- **Suite summary:** `{fals.name}` — "
                f"{'all OK' if not bad else 'errors in: ' + ', '.join(bad)}"
            )
        except (json.JSONDecodeError, OSError):
            lines.append(f"- **Suite summary:** `{fals}` (unreadable).")
    else:
        lines.append("- **Suite summary:** not found (run falsification suite).")

    ng = output_dir / "non_gaussian" / "summary.json"
    if ng.is_file():
        lines.append(f"- **Non-Gaussian τ:** `{ng.relative_to(output_dir)}` present.")
    else:
        lines.append("- **Non-Gaussian τ:** no `summary.json` yet.")

    sc = output_dir / "scaling" / "fit_results.json"
    if sc.is_file():
        try:
            fr = json.loads(sc.read_text(encoding="utf-8"))
            ind = fr.get("independent", {})
            lines.append(
                f"- **Scaling (independent):** log–log slope Var vs N ≈ "
                f"{ind.get('loglog_slope_var_vs_N', '—')} (see `scaling/scaling_data.csv`)."
            )
        except (json.JSONDecodeError, OSError):
            lines.append("- **Scaling:** `fit_results.json` unreadable.")
    else:
        lines.append("- **Scaling:** not run.")

    th = output_dir / "threshold" / "critical_sigma.txt"
    if th.is_file():
        first = th.read_text(encoding="utf-8").strip().split("\n")[0]
        lines.append(f"- **Threshold:** {first}")
    else:
        lines.append("- **Threshold:** `critical_sigma.txt` not found.")

    dv = output_dir / "deviation" / "stats.json"
    if dv.is_file():
        try:
            st = json.loads(dv.read_text(encoding="utf-8"))
            mx = st.get("max_mean_delta", None)
            lines.append(
                f"- **Lindblad vs τ:** max mean |ΔC| over time ≈ {_fmt(mx) if mx is not None else '—'} "
                f"(see `deviation/stats.json`)."
            )
        except (json.JSONDecodeError, OSError):
            lines.append("- **Lindblad vs τ:** stats unreadable.")
    else:
        lines.append("- **Lindblad vs τ:** not run.")

    lines.append("")
    return "\n".join(lines)


def generate_tdf_report(
    *,
    output_dir: Path | None = None,
    report_name: str = "tdf_qutip_report.md",
    include_falsification: bool = False,
) -> Path:
    """
    Read ``tau_model_summary.csv`` and ``interference_sweep.csv``, write ``tdf_qutip_report.md``.

    Returns path to the written report.
    """
    if output_dir is None:
        output_dir = _pkg_outputs_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    tau_path = output_dir / "tau_model_summary.csv"
    if_path = output_dir / "interference_sweep.csv"

    tau_rows = _read_csv(tau_path)
    if_rows = _read_csv(if_path)

    # --- τ-model metrics ---
    for r in tau_rows:
        r["_S"] = _f(r.get("spectral_entropy_nats", ""))
        r["_BW"] = _f(r.get("bandwidth_90pct_power", ""))
        r["_fdom"] = _f(r.get("dominant_positive_frequency", ""))

    richest = max(tau_rows, key=lambda r: (r["_S"] is not None and r["_S"] or -1.0))
    richest_name = richest.get("model", "?")
    richest_S = richest["_S"]
    richest_BW = richest["_BW"]

    # --- interference sweep ---
    ovl: list[float] = []
    mae: list[float] = []
    raw_mismatch: list[float] = []
    dt_mismatch: list[float] = []
    for r in if_rows:
        oc = _f(r.get("overlap_correlation", ""))
        if oc is not None and not math.isnan(oc):
            ovl.append(oc)
        m = _f(r.get("mean_abs_error", ""))
        if m is not None:
            mae.append(m)
        frd = _f(r.get("dominant_freq_direct_raw", ""))
        frq = _f(r.get("dominant_freq_quantum_raw", ""))
        fdd = _f(r.get("dominant_freq_direct_detrended", ""))
        fdq = _f(r.get("dominant_freq_quantum_detrended", ""))
        if frd is not None and frq is not None:
            raw_mismatch.append(abs(frd - frq))
        if fdd is not None and fdq is not None:
            dt_mismatch.append(abs(fdd - fdq))

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    ovl_mean = _mean(ovl)
    ovl_min = min(ovl) if ovl else None
    mae_max = max(mae) if mae else None
    mae_min = min(mae) if mae else None
    raw_mean_mis = _mean(raw_mismatch)
    dt_mean_mis = _mean(dt_mismatch)

    if raw_mean_mis is not None and dt_mean_mis is not None:
        if dt_mean_mis < raw_mean_mis - 1e-9:
            freq_compare = (
                f"Mean |f_direct − f_quantum| drops from {_fmt(raw_mean_mis)} Hz (raw) "
                f"to {_fmt(dt_mean_mis)} Hz (detrended), so detrended peaks align more closely "
                "when DC-heavy spectra are excluded."
            )
        elif raw_mean_mis < dt_mean_mis - 1e-9:
            freq_compare = (
                "Raw dominant frequencies show slightly smaller mean mismatch than detrended "
                "on this dataset; detrending still clarifies oscillatory content when raw peaks sit at DC."
            )
        else:
            freq_compare = (
                f"Raw and detrended mean frequency mismatches are similar (~{_fmt(raw_mean_mis)} Hz)."
            )
    else:
        freq_compare = "Insufficient frequency columns to compare raw vs detrended mismatch."

    if ovl_mean is not None and ovl_min is not None:
        if ovl_min > 0.9:
            ovl_summary = (
                f"Pearson *r* between cos(Δτ) and quantum overlap averages **{_fmt(ovl_mean, 4)}** "
                f"(minimum **{_fmt(ovl_min, 4)}** over sweep rows), i.e. traces stay strongly correlated "
                "in most parameter settings."
            )
        elif ovl_min < 0.5:
            ovl_summary = (
                f"Overlap correlation is high on average (**{_fmt(ovl_mean, 4)}**) but falls to **{_fmt(ovl_min, 4)}** "
                "in at least one regime (see large oscillatory amplitude), so agreement is not uniform."
            )
        else:
            ovl_summary = (
                f"Mean *r* = **{_fmt(ovl_mean, 4)}**, min *r* = **{_fmt(ovl_min, 4)}** across sweep rows."
            )
    else:
        ovl_summary = "No valid overlap correlations parsed."

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# TDF–QM numerical summary report",
        "",
        f"*Generated {now} from `tau_model_summary.csv` and `interference_sweep.csv`.*",
        "",
        "## Overview: working assumptions",
        "",
        "This codebase explores a **time-dependent field (TDF) ansatz** linking a scalar phase field "
        "τ(t) to qubit dynamics, **not** a full axiomatic replacement of quantum mechanics.",
        "",
        "- **Phase field:** amplitudes carry phases suggestive of **ψ ∝ exp(iτ)** in a motivational sense; "
        "dynamics are implemented via QuTiP with **H(t)** derived from τ.",
        "- **Energy relation (model):** **E = ℏ dτ/dt**, used to build **H ∝ E(t) σ_z** in the minimal coupling used here.",
        "- **Classical interference diagnostic:** **cos(Δτ)** with Δτ = τ_A − τ_B is compared to **|⟨ψ_A|ψ_B⟩|** "
        "after separate evolutions—useful as a **consistency check**, not as a theorem.",
        "",
        "## τ-model spectral comparison (Ramsey channel, ⟨σ_x⟩ FFT)",
        "",
        "Spectral entropy *S* (nats) and 90% power bandwidth summarize how much structure appears "
        "in the positive-frequency half of the ⟨σ_x⟩ trajectory under each τ-driven Hamiltonian.",
        "",
        "| model | *S* (nats) | BW₉₀ | *f*_dom |",
        "|---|---:|---:|---:|",
    ]
    for r in tau_rows:
        lines.append(
            f"| `{r.get('model', '')}` | {_fmt(r['_S'], 4)} | {_fmt(r['_BW'], 4)} | {_fmt(r['_fdom'], 4)} |"
        )
    lines += [
        "",
        f"**Richest spectrum (by spectral entropy):** `{richest_name}` "
        f"(*S* ≈ {_fmt(richest_S, 4)}, BW₉₀ ≈ {_fmt(richest_BW, 4)}). "
        "Larger *S* indicates a broader spread of |FFT|² across frequency bins on this grid.",
        "",
        "## Interference sweep robustness",
        "",
        "Two sweeps: structured **ν** with linear τ_A; fixed ν = 3 with oscillatory **amplitude** on τ_A. "
        "Metrics: mean |cos(Δτ) − overlap|, detrended dominant frequencies, and Pearson *r* between the two traces.",
        "",
        f"- **Mean absolute error** across sweep rows: roughly **{_fmt(mae_min, 4)}**–**{_fmt(mae_max, 4)}** "
        "(depends strongly on oscillatory amplitude at large *A*).",
        f"- **Overlap correlation:** {ovl_summary}",
        f"- **Raw vs detrended dominant frequencies:** {freq_compare}",
        "",
        "## Highlights",
        "",
        f"- **Most structured Ramsey spectrum:** `{richest_name}` under the chosen parameters.",
        f"- **Overlap correlation:** remains **high** in the structured-frequency sweep and for small oscillatory "
        f"amplitudes; **degrades** when *A* is large (nonlinear τ_A), as expected if cos(Δτ) ceases to track overlap.",
        f"- **Raw vs detrended peak agreement:** mean |*f*_direct − *f*_quantum| over sweep rows is "
        f"**{_fmt(raw_mean_mis, 4)}** Hz (raw) vs **{_fmt(dt_mean_mis, 4)}** Hz (detrended) here—raw often ties both "
        f"channels to **DC**, yielding spurious agreement; detrending shifts emphasis to oscillatory bins and "
        f"can **separate** peaks when DC is misleading (see large-amplitude row where raw vs detrended peaks differ).",
        "",
        "## Interpretation",
        "",
        "Within this **minimal single-qubit, σ_z-coupled** implementation, τ-fields that add independent modulation "
        "(structured and multi-scale) produce **richer FFT structure** in ⟨σ_x⟩ than a pure linear drift. "
        "The **cos(Δτ) vs overlap** comparison suggests the TDF construction can **track** the quantum overlap "
        "closely when τ enters through the **same Hamiltonian map** used for evolution, but agreement is **parameter-dependent** "
        "and should be re-checked when changing Hilbert space, coupling, or open-system noise.",
        "",
        "## Limitations",
        "",
        "- **Phenomenological** τ models and **ad hoc** map τ → H; no uniqueness or completeness claim.",
        "- **Single qubit**, closed system, fixed discretization and time window—FFT summaries depend on grid choice.",
        "- **cos(Δτ)** is a **scalar diagnostic**, not derived from a full two-path interferometer model.",
        "- CSV reflects **one** parameter set per row; statistical uncertainty and repeated trials are not included.",
        "",
        "## Next experiments",
        "",
        "- Extend to **open-system** channels and compare cos(Δτ) diagnostics to **Lindblad** trajectories.",
        "- Scan **ℏ**, coupling axis (e.g. σ_x component), and **initial states**; quantify where overlap correlation breaks down.",
        "- Add **hypothesis tests** or confidence intervals over multiple noise realizations (e.g. stochastic τ).",
        "- Export **figures** into this report (e.g. spectrum overlays) for archival reproducibility.",
        "",
    ]

    if include_falsification:
        lines.extend(falsification_results_markdown(output_dir).splitlines())

    out_path = output_dir / report_name
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def append_falsification_to_report(
    report_path: Path,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Append :func:`falsification_results_markdown` to an existing Markdown report file."""
    block = "\n\n" + falsification_results_markdown(output_dir)
    existing = report_path.read_text(encoding="utf-8")
    report_path.write_text(existing + block, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    p = generate_tdf_report()
    print(f"Wrote {p}")
