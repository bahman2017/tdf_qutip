"""
Repeated baseline vs ``tau_symmetric`` runs (IBM or Aer) for statistical summaries.

Uses :func:`experiments.ibm_tau_hardware_ready.run_experiment_once` (same transpile + execute path).

Example::

    PYTHONPATH=. python3 experiments/ibm_tau_symmetric_stats.py --backend ibm --plot
    PYTHONPATH=. python3 experiments/ibm_tau_symmetric_stats.py --backend aer --n-runs 4

After a run, per-depth **mean ΔCHSH**, **95% CI** (normal approx.), and **win_rate** are printed and
saved to ``outputs/ibm_tau_symmetric_stats_by_depth.csv``. Raw rows stay in
``outputs/ibm_tau_symmetric_stats.csv`` (``tau_symmetric`` vs baseline only).

Re-summarize an existing raw CSV::

    PYTHONPATH=. python3 experiments/ibm_tau_symmetric_stats.py --analyze-csv outputs/ibm_tau_symmetric_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from experiments.ibm_tau_hardware_ready import (
    SingleDepthExperimentResult,
    get_backend,
    is_aer_backend,
    run_experiment_once,
)

Z95 = 1.96
DEFAULT_DEPTHS = [4, 6, 8, 10, 12, 14]
DEFAULT_N_RUNS = 8
DEFAULT_SHOTS = 1024
MAX_RETRIES = 3
RETRY_SLEEP_S = 4.0
BY_DEPTH_CSV_NAME = "ibm_tau_symmetric_stats_by_depth.csv"


def _effective_shots(backend, shots: int) -> int:
    s = int(shots)
    if not is_aer_backend(backend):
        s = min(s, 1024)
    return s


def _run_with_retries(
    *,
    n_steps: int,
    backend,
    shots: int,
    t_max: float,
    omega: float,
    a: float,
    nu: float,
    seed: int,
) -> tuple[SingleDepthExperimentResult | None, str | None]:
    last_err: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return (
                run_experiment_once(
                    n_steps,
                    backend=backend,
                    shots=shots,
                    t_max=t_max,
                    omega=omega,
                    a=a,
                    nu=nu,
                    seed=seed,
                ),
                None,
            )
        except Exception as e:
            last_err = repr(e)
            if attempt + 1 < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_S * (attempt + 1))
    return None, last_err


def _per_depth_from_raw_rows(
    rows_out: list[dict[str, int | float]],
    depths: list[int],
) -> list[dict[str, float | int | bool]]:
    by_depth: dict[int, list[float]] = {d: [] for d in depths}
    for r in rows_out:
        by_depth[int(r["depth"])].append(float(r["delta_chsh"]))

    per_depth_stats: list[dict[str, float | int | bool]] = []
    for d in depths:
        xs = np.array(by_depth[d], dtype=float)
        n = int(xs.size)
        if n == 0:
            per_depth_stats.append(
                {
                    "depth": d,
                    "n": 0,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "win_rate": float("nan"),
                    "passes": False,
                }
            )
            continue
        mean = float(np.mean(xs))
        std = float(np.std(xs, ddof=1)) if n > 1 else 0.0
        if n > 1:
            se = std / np.sqrt(n)
            ci_lo = mean - Z95 * se
            ci_hi = mean + Z95 * se
        else:
            ci_lo = float("nan")
            ci_hi = float("nan")
        win_rate = float(np.mean(xs > 0.0))
        ci_above_zero = np.isfinite(ci_lo) and ci_lo > 0.0
        if n <= 1:
            passes = False
        else:
            passes = bool(mean > 0.0 and ci_above_zero and win_rate >= 0.7)
        per_depth_stats.append(
            {
                "depth": d,
                "n": n,
                "mean": mean,
                "std": std,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "win_rate": win_rate,
                "passes": passes,
            }
        )
    return per_depth_stats


def _write_by_depth_csv(path: Path, per_depth_stats: list[dict[str, float | int | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "depth",
                "n_runs",
                "mean_delta_chsh",
                "std_delta_chsh",
                "ci95_lo",
                "ci95_hi",
                "win_rate",
            ],
        )
        w.writeheader()
        for s in per_depth_stats:
            w.writerow(
                {
                    "depth": s["depth"],
                    "n_runs": s["n"],
                    "mean_delta_chsh": "" if s["n"] == 0 else f'{s["mean"]:.10g}',
                    "std_delta_chsh": "" if s["n"] == 0 else f'{s["std"]:.10g}',
                    "ci95_lo": "" if s["n"] <= 1 or not np.isfinite(s["ci_lo"]) else f'{s["ci_lo"]:.10g}',
                    "ci95_hi": "" if s["n"] <= 1 or not np.isfinite(s["ci_hi"]) else f'{s["ci_hi"]:.10g}',
                    "win_rate": "" if s["n"] == 0 else f'{s["win_rate"]:.10g}',
                }
            )


def _print_symmetric_summary_table(per_depth_stats: list[dict[str, float | int | bool]]) -> None:
    print()
    print("--- τ_symmetric vs baseline: mean ΔCHSH, CI₉₅, win_rate (per depth) ---")
    for s in per_depth_stats:
        d = int(s["depth"])
        n = int(s["n"])
        if n == 0:
            print(f"depth={d}  n=0  (no runs)")
            continue
        mean = s["mean"]
        wr = s["win_rate"]
        if n > 1 and np.isfinite(s["ci_lo"]) and np.isfinite(s["ci_hi"]):
            print(
                f"depth={d}  n={n}  mean ΔCHSH={mean:+.6f}  "
                f"CI₉₅=[{s['ci_lo']:+.6f}, {s['ci_hi']:+.6f}]  win_rate={wr:.3f}"
            )
        else:
            print(
                f"depth={d}  n={n}  mean ΔCHSH={mean:+.6f}  "
                f"CI₉₅=undefined  win_rate={wr:.3f}"
            )


def analyze_csv_only(
    raw_csv: Path,
    *,
    out_dir: Path | None = None,
    write_by_depth: bool = True,
) -> int:
    """Load raw per-run CSV; print and optionally write per-depth summary."""
    if not raw_csv.is_file():
        print(f"Missing CSV: {raw_csv}", file=sys.stderr)
        return 1
    with raw_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Empty CSV", file=sys.stderr)
        return 1
    depths = sorted({int(r["depth"]) for r in rows})
    rows_out: list[dict[str, int | float]] = []
    for r in rows:
        rows_out.append(
            {
                "depth": int(r["depth"]),
                "run_id": int(r["run_id"]),
                "delta_chsh": float(r["delta_chsh"]),
                "delta_fidelity": float(r["delta_fidelity"]),
            }
        )
    stats = _per_depth_from_raw_rows(rows_out, depths)
    _print_symmetric_summary_table(stats)
    if write_by_depth:
        od = out_dir if out_dir is not None else raw_csv.parent
        by_path = od / BY_DEPTH_CSV_NAME
        _write_by_depth_csv(by_path, stats)
        print(f"Wrote {by_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical wrapper: repeated τ_symmetric vs baseline at fixed depths",
    )
    parser.add_argument(
        "--backend",
        choices=("ibm", "aer"),
        default="ibm",
        help="IBM hardware or Aer (default: ibm; use --aer-fallback if IBM fails)",
    )
    parser.add_argument("--ibm-backend", type=str, default=None)
    parser.add_argument(
        "--aer-fallback",
        action="store_true",
        help="If IBM backend unavailable, use AerSimulator",
    )
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=DEFAULT_DEPTHS,
        help="n_steps values (default: 4 6 8 10 12 14)",
    )
    parser.add_argument("--t-max", type=float, default=0.8)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.3)
    parser.add_argument("--nu", type=float, default=2.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <repo>/outputs",
    )
    parser.add_argument(
        "--apikey-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write outputs/ibm_tau_symmetric_stats.png (default: true)",
    )
    parser.add_argument("--seed-base", type=int, default=2026)
    parser.add_argument(
        "--analyze-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Only read raw per-run CSV and print/write per-depth summary (no hardware)",
    )
    args = parser.parse_args()

    if args.analyze_csv is not None:
        od = args.output_dir
        if od is None:
            od = Path(__file__).resolve().parent.parent / "outputs"
        sys.exit(analyze_csv_only(args.analyze_csv, out_dir=od))

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ibm_tau_symmetric_stats.csv"

    use_ibm = args.backend == "ibm"
    backend = get_backend(
        use_ibm=use_ibm,
        name=args.ibm_backend,
        apikey_path=args.apikey_json,
    )
    if backend is None and use_ibm and args.aer_fallback:
        from qiskit_aer import AerSimulator

        print("IBM unavailable; using AerSimulator (--aer-fallback).")
        backend = AerSimulator()
    if backend is None:
        raise SystemExit("No backend: fix IBM credentials or pass --aer-fallback")

    shots_eff = _effective_shots(backend, args.shots)
    depths = [int(d) for d in args.depths]
    n_runs = max(1, int(args.n_runs))

    print(f"Backend: {backend}")
    print(f"depths={depths}, n_runs={n_runs}, shots={shots_eff}")

    rows_out: list[dict[str, int | float]] = []

    for depth in depths:
        for run_id in range(n_runs):
            seed = int(args.seed_base) + depth * 1000 + run_id * 17
            res, err = _run_with_retries(
                n_steps=depth,
                backend=backend,
                shots=shots_eff,
                t_max=args.t_max,
                omega=args.omega,
                a=args.a,
                nu=args.nu,
                seed=seed,
            )
            if res is None:
                print(f"  depth={depth} run_id={run_id} FAILED after retries: {err}")
                continue
            d_chsh = float(res.chsh_tau - res.chsh_baseline)
            d_fid = float(res.fidelity_tau - res.fidelity_baseline)
            rows_out.append(
                {
                    "depth": depth,
                    "run_id": run_id,
                    "delta_chsh": d_chsh,
                    "delta_fidelity": d_fid,
                }
            )
            print(f"  depth={depth} run_id={run_id} ΔCHSH={d_chsh:+.4f} ΔFid={d_fid:+.4f}")

    if not rows_out:
        raise SystemExit("No successful runs; nothing to save.")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["depth", "run_id", "delta_chsh", "delta_fidelity"],
        )
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {csv_path}")

    # --- Per-depth analysis (τ_symmetric − baseline, ΔCHSH only) ---
    per_depth_stats = _per_depth_from_raw_rows(rows_out, depths)
    _print_symmetric_summary_table(per_depth_stats)

    by_depth_path = out_dir / BY_DEPTH_CSV_NAME
    _write_by_depth_csv(by_depth_path, per_depth_stats)
    print(f"Wrote {by_depth_path}")

    # Require every depth with data to pass; depths with n=0 fail the gate
    depths_with_data = [s for s in per_depth_stats if s["n"] > 0]
    all_pass_strict = bool(depths_with_data) and all(s["passes"] for s in depths_with_data)

    print()
    if all_pass_strict:
        print("Statistically consistent τ advantage")
    else:
        print("No statistically significant τ advantage")

    if args.plot and depths_with_data:
        try:
            import matplotlib.pyplot as plt

            ds = [s["depth"] for s in depths_with_data]
            means = [s["mean"] for s in depths_with_data]
            yerr = []
            for s in depths_with_data:
                if s["n"] > 1 and np.isfinite(s["ci_lo"]) and np.isfinite(s["ci_hi"]):
                    half = 0.5 * (s["ci_hi"] - s["ci_lo"])
                else:
                    half = 0.0
                yerr.append(half)
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
            ax.errorbar(ds, means, yerr=yerr, fmt="o-", capsize=4, color="C0", lw=2, ms=6)
            ax.axhline(0.0, color="k", ls=":", lw=0.9, alpha=0.5)
            ax.set_xlabel(r"$n_{\mathrm{steps}}$ (depth)")
            ax.set_ylabel(r"mean $\Delta$CHSH (τ_symmetric $-$ baseline)")
            ax.set_title("IBM τ_symmetric vs baseline — mean ΔCHSH ± 95% CI (normal approx.)")
            ax.grid(True, alpha=0.35)
            fig.tight_layout()
            png_path = out_dir / "ibm_tau_symmetric_stats.png"
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {png_path}")
        except ImportError:
            print("matplotlib not installed; skip plot")


if __name__ == "__main__":
    main()
