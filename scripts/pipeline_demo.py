"""
Shared reproducible steps for docs/REPRO.md and ``python -m`` module entry points.

Run from repository root ``tdf_qutip`` with ``PYTHONPATH=.`` (see ``docs/REPRO.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Default output directory for all figures/CSVs
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _t_default() -> np.ndarray:
    return np.linspace(0.0, 5.0, 80)


def _bounds() -> list[tuple[float, float]]:
    return [
        (0.2, 2.5),
        (0.05, 2.0),
        (0.5, 8.0),
        (0.05, 2.0),
        (0.1, 3.0),
    ]


def _midpoint_guess() -> dict[str, float]:
    b = _bounds()
    from analysis.tau_extraction import PARAM_NAMES

    return {name: 0.5 * (lo + hi) for name, (lo, hi) in zip(PARAM_NAMES, b)}


def step_correlation(
    t: np.ndarray | None = None,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    from experiments.correlation_test import run_two_qubit_correlation_experiment

    t = np.asarray(t if t is not None else np.linspace(0.0, 10.0, 300), dtype=float)
    od = output_dir or OUTPUT_DIR
    return run_two_qubit_correlation_experiment(
        t=t, output_dir=od, plot=True, show=False
    )


def step_tau_extraction(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    from experiments.correlation_test import run_two_qubit_correlation_experiment
    from analysis.tau_extraction import fit_tau_from_correlations

    t = _t_default()
    if ctx and "t" in ctx:
        t = np.asarray(ctx["t"], dtype=float).ravel()
        exp = ctx
    else:
        exp = run_two_qubit_correlation_experiment(t=t, plot=False, show=False)
    bounds = _bounds()
    fit = fit_tau_from_correlations(
        t,
        exp["cxx_tdf"],
        exp["cyy_tdf"],
        exp["chsh_tdf"],
        _midpoint_guess(),
        bounds,
        save_plots=True,
        output_dir=OUTPUT_DIR,
        verbose=True,
        maxiter=200,
    )
    return {"t": t, **exp, "tau_extraction": fit}


def step_identifiability(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    from experiments.correlation_test import run_two_qubit_correlation_experiment
    from analysis.tau_identifiability import multi_start_tau_fit

    t = _t_default()
    if ctx and "cxx_tdf" in ctx:
        exp = ctx
    else:
        exp = run_two_qubit_correlation_experiment(t=t, plot=False, show=False)
    summary = multi_start_tau_fit(
        t,
        exp["cxx_tdf"],
        exp["cyy_tdf"],
        exp["chsh_tdf"],
        _bounds(),
        n_runs=12,
        save_plots=True,
        show=False,
        maxiter=80,
        output_dir=OUTPUT_DIR,
    )
    return {"t": t, **exp, "identifiability": summary}


def step_manifold(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    from analysis.tau_manifold import analyze_degeneracy_manifold

    ctx = step_identifiability(ctx)
    t = ctx["t"]
    summary = ctx["identifiability"]
    losses = np.asarray(summary["all_losses"])
    epsilon = max(float(np.percentile(losses - losses.min(), 60)), 0.05)
    manifold = analyze_degeneracy_manifold(
        summary, t, epsilon, output_dir=OUTPUT_DIR, save_plots=True, show=False
    )
    return {**ctx, "manifold": manifold, "epsilon": epsilon}


def step_kernel_modes(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    from analysis.tau_kernel_modes import run_tau_kernel_mode_analysis

    ctx = step_manifold(ctx)
    kernel = run_tau_kernel_mode_analysis(
        ctx["manifold"],
        ctx["identifiability"],
        ctx["t"],
        output_dir=OUTPUT_DIR,
        save_plots=True,
        show=False,
    )
    return {**ctx, "kernel": kernel}


def step_mode_field_fit(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    from analysis.tau_mode_field_fit import fit_kernel_modes_to_field_equations

    ctx = step_kernel_modes(ctx)
    field = fit_kernel_modes_to_field_equations(
        ctx["t"],
        ctx["kernel"],
        n_modes=3,
        output_dir=OUTPUT_DIR,
        save_plots=True,
        show=False,
    )
    return {**ctx, "field_fit": field}


def step_hidden_spectrum(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    from analysis.tau_hidden_spectrum import extract_hidden_spectrum

    ctx = step_mode_field_fit(ctx)
    hidden = extract_hidden_spectrum(
        ctx["field_fit"],
        output_dir=OUTPUT_DIR,
        save_plots=True,
        show=False,
    )
    return {**ctx, "hidden_spectrum": hidden}


def step_chi_geometry(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    from analysis.tau_chi_geometry import analyze_chi_geometry

    ctx = step_hidden_spectrum(ctx)
    chi = analyze_chi_geometry(
        ctx["hidden_spectrum"],
        output_dir=OUTPUT_DIR,
        save_plots=True,
        show=False,
    )
    return {**ctx, "chi_geometry": chi}


def step_tdf_vs_colored_noise(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    from experiments.tdf_vs_colored_noise import run_tdf_vs_colored_noise_discrimination

    t = np.linspace(0.0, 8.0, 200)
    return run_tdf_vs_colored_noise_discrimination(
        t=t,
        tau_c_values=np.linspace(0.35, 2.2, 7),
        ou_correlation_times=np.linspace(0.35, 2.2, 7),
        pink_noise_strengths=np.linspace(0.15, 1.2, 7),
        output_dir=OUTPUT_DIR,
        save_plots=True,
        save_csv=True,
        show=False,
    )


def run_all() -> dict[str, Any]:
    """Execute the full chain through χ-geometry, then the discrimination experiment."""
    ctx = step_chi_geometry(None)
    disc = step_tdf_vs_colored_noise(None)
    return {**ctx, "discrimination": disc}


STEP_REGISTRY: dict[str, Any] = {
    "correlation": lambda: step_correlation(),
    "tau_extraction": lambda: step_tau_extraction(None),
    "identifiability": lambda: step_identifiability(None),
    "manifold": lambda: step_manifold(None),
    "kernel_modes": lambda: step_kernel_modes(None),
    "mode_field_fit": lambda: step_mode_field_fit(None),
    "hidden_spectrum": lambda: step_hidden_spectrum(None),
    "chi_geometry": lambda: step_chi_geometry(None),
    "tdf_vs_colored_noise": lambda: step_tdf_vs_colored_noise(None),
}


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="TDF-QM reproducible pipeline steps")
    p.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline (χ-geometry chain + discrimination)",
    )
    p.add_argument(
        "--step",
        choices=list(STEP_REGISTRY.keys()),
        help="Run a single named step (each step may recompute upstream data)",
    )
    args = p.parse_args(argv)
    if args.all:
        run_all()
        print("Full pipeline finished. Outputs in", OUTPUT_DIR)
    elif args.step:
        STEP_REGISTRY[args.step]()
        print(f"Step {args.step!r} finished. Outputs in", OUTPUT_DIR)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
