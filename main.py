"""
Entry point for quick runs or batch experiments.

Ramsey analysis: standard vs TDF τ-model comparison plots and phase observables.

CLI::

    python main.py                      # default Ramsey + pipeline demo
    python main.py --run falsification_tests
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.discrimination import (
    run_tdf_discrimination_summary,
    run_tdf_discrimination_summary_v2,
)
from analysis.plots import plot_timeseries
from analysis.report import (
    append_falsification_to_report,
    falsification_results_markdown,
    generate_tdf_report,
)
from analysis.tau_extraction import fit_tau_from_correlations
from analysis.tau_model_spectrum import analyze_tau_models, phase_magnitude
from experiments.correlation_test import run_two_qubit_correlation_experiment
from experiments.decoherence import compare_tdf_vs_fitted_lindblad
from experiments.interference import run_interference_parameter_sweep
from experiments.ramsey import run_ramsey_experiment


def print_tau_model_summary_table(rows: list[dict]) -> None:
    """Print aligned table of τ-model metrics."""
    headers = (
        "model",
        "f_dom",
        "peak1",
        "peak2",
        "peak3",
        "S_spec",
        "BW_90",
        "⟨|φ|⟩",
    )
    w_name = max(len(headers[0]), max(len(r["name"]) for r in rows))
    w_f = 10
    w_ent = 10
    w_bw = 10
    w_ph = 10

    def fnum(x: float, width: int) -> str:
        if np.isnan(x):
            return f"{'nan':>{width}}"
        return f"{x:>{width}.6g}"

    line = (
        f"{headers[0]:<{w_name}}  "
        f"{headers[1]:>{w_f}}  "
        f"{headers[2]:>{w_f}}  "
        f"{headers[3]:>{w_f}}  "
        f"{headers[4]:>{w_f}}  "
        f"{headers[5]:>{w_ent}}  "
        f"{headers[6]:>{w_bw}}  "
        f"{headers[7]:>{w_ph}}"
    )
    sep = "-" * len(line)
    print(sep)
    print("τ-model quantitative summary (FFT of ⟨σ_x⟩, positive half)")
    print(sep)
    print(line)
    print(sep)
    for r in rows:
        print(
            f"{r['name']:<{w_name}}  "
            f"{fnum(r['dominant_freq'], w_f)}  "
            f"{fnum(r['peak1'], w_f)}  "
            f"{fnum(r['peak2'], w_f)}  "
            f"{fnum(r['peak3'], w_f)}  "
            f"{fnum(r['spectral_entropy'], w_ent)}  "
            f"{fnum(r['bandwidth_90'], w_bw)}  "
            f"{fnum(r['phase_mag_mean'], w_ph)}"
        )
    print(sep)
    print("f_dom: dominant positive frequency; S_spec: spectral entropy (nats);")
    print("BW_90: Hz width of shortest band with ≥90% |FFT|² power; ⟨|φ|⟩: mean √(⟨σ_x⟩²+⟨σ_y⟩²).")
    print(sep)


def save_tau_model_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "dominant_positive_frequency",
        "peak_frequency_1",
        "peak_frequency_2",
        "peak_frequency_3",
        "spectral_entropy_nats",
        "bandwidth_90pct_power",
        "phase_magnitude_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "model": r["name"],
                    "dominant_positive_frequency": r["dominant_freq"],
                    "peak_frequency_1": r["peak1"],
                    "peak_frequency_2": r["peak2"],
                    "peak_frequency_3": r["peak3"],
                    "spectral_entropy_nats": r["spectral_entropy"],
                    "bandwidth_90pct_power": r["bandwidth_90"],
                    "phase_magnitude_mean": r["phase_mag_mean"],
                }
            )


def run_tau_model_frequency_comparison(
    t: np.ndarray | None = None,
) -> list[dict]:
    """
    For each τ model, evolve TDF, FFT ⟨σ_x⟩, plot per-model and combined spectra.

    Returns
    -------
    list[dict]
        Metric rows (same structure as :func:`analyze_tau_models`).
    """
    if t is None:
        t = np.linspace(0, 10, 300)

    print("Comparing τ models (FFT of ⟨σ_x⟩)...")

    freq_pos, rows = analyze_tau_models(t)
    labels = [r["name"] for r in rows]
    spectra = [r["mag_half"] for r in rows]

    for r in rows:
        plot_timeseries(
            freq_pos,
            [r["mag_half"]],
            labels=[r["name"]],
            title=f"Frequency spectrum (σ_x): {r['name']}",
            xlabel="Frequency",
            ylabel="Amplitude",
            figsize=(8, 3.5),
        )
        plt.show()

    plot_timeseries(
        freq_pos,
        spectra,
        labels=labels,
        title=r"Frequency spectrum (σ_x): all τ models compared",
        xlabel="Frequency",
        ylabel="Amplitude",
        figsize=(10, 5),
    )
    plt.show()

    return rows


def run_falsification_tests_entry(*, fast: bool = False) -> None:
    """Run TDF falsification suite and append results to the Markdown report."""
    from experiments.falsification_suite import run_all_falsification_tests

    out_dir = Path(__file__).resolve().parent / "outputs"
    print("Running TDF falsification test suite...")
    summary = run_all_falsification_tests(output_root=out_dir, fast=fast)
    print("Falsification suite finished:", summary.get("summary_path", ""))
    snippet = out_dir / "tdf_qutip_falsification_section.md"
    snippet.write_text(falsification_results_markdown(out_dir), encoding="utf-8")
    print(f"Wrote standalone falsification Markdown: {snippet}")
    report_md = out_dir / "tdf_qutip_report.md"
    if report_md.is_file():
        append_falsification_to_report(report_md, output_dir=out_dir)
        print(f"Appended falsification section to {report_md}")


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--run" and argv[1] == "falsification_tests":
        fast = "--fast" in argv
        run_falsification_tests_entry(fast=fast)
        return

    print("Running TDF vs Standard comparison...")

    t, result_standard, result_tau, tau = run_ramsey_experiment()

    std_x = result_standard.expect[0]
    std_y = result_standard.expect[1]
    tau_x = result_tau.expect[0]
    tau_y = result_tau.expect[1]

    plot_timeseries(
        t,
        [std_x, tau_x],
        labels=["Standard QM (X)", "TDF (X)"],
        title="Phase dynamics (σ_x)",
        ylabel=r"$\langle\sigma_x\rangle$",
    )
    plt.show()

    plot_timeseries(
        t,
        [std_y, tau_y],
        labels=["Standard QM (Y)", "TDF (Y)"],
        title="Phase dynamics (σ_y)",
        ylabel=r"$\langle\sigma_y\rangle$",
    )
    plt.show()

    std_fft = np.fft.fft(np.asarray(std_x, dtype=float))
    tau_fft = np.fft.fft(np.asarray(tau_x, dtype=float))
    freq = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
    n_half = len(freq) // 2

    plot_timeseries(
        freq[:n_half],
        [np.abs(std_fft)[:n_half], np.abs(tau_fft)[:n_half]],
        labels=["Standard", "TDF"],
        title="Frequency Spectrum (σ_x)",
        xlabel="Frequency",
        ylabel="Amplitude",
    )
    plt.show()

    std_peak = freq[np.argmax(np.abs(std_fft))]
    tau_peak = freq[np.argmax(np.abs(tau_fft))]
    print("Standard peak frequency:", std_peak)
    print("TDF peak frequency:", tau_peak)

    std_phase = phase_magnitude(std_x, std_y)
    tau_phase = phase_magnitude(tau_x, tau_y)

    print("Standard mean phase magnitude:", float(np.mean(std_phase)))
    print("TDF mean phase magnitude:", float(np.mean(tau_phase)))

    plot_timeseries(
        t,
        [std_phase, tau_phase],
        labels=["Standard", "TDF"],
        title="Phase coherence magnitude",
        ylabel=r"$\sqrt{\langle\sigma_x\rangle^2 + \langle\sigma_y\rangle^2}$",
    )
    plt.show()

    plot_timeseries(
        t,
        [tau],
        labels=[r"$\tau(t)$"],
        title="Tau Field",
        ylabel=r"$\tau$",
    )
    plt.show()

    tau_rows = run_tau_model_frequency_comparison(t)
    print()
    print_tau_model_summary_table(tau_rows)
    out_csv = Path(__file__).resolve().parent / "outputs" / "tau_model_summary.csv"
    save_tau_model_summary_csv(tau_rows, out_csv)
    print(f"Wrote {out_csv}")

    print()
    print("Interference parameter sweep (cos(Δτ) vs quantum overlap)...")
    out_dir = Path(__file__).resolve().parent / "outputs"
    run_interference_parameter_sweep(t=t, output_dir=out_dir, plot=True, show=False)

    report_path = generate_tdf_report(output_dir=out_dir, include_falsification=False)
    print(f"Wrote {report_path}")

    print()
    print("TDF vs fitted Lindblad decoherence comparison...")
    decoherence_result = compare_tdf_vs_fitted_lindblad(
        t=t, output_dir=out_dir, plot=True, show=False
    )

    print()
    print("τ-model discrimination summary (spectrum + interference + decoherence baseline)...")
    run_tdf_discrimination_summary(
        t, out_dir, decoherence_result=decoherence_result
    )

    print()
    print("τ-model discrimination summary v2 (includes structured_stochastic_tau)...")
    run_tdf_discrimination_summary_v2(
        t, out_dir, decoherence_result=decoherence_result
    )

    print()
    print("Two-qubit Bell correlations: standard vs TDF (correlated τ)...")
    corr_res = run_two_qubit_correlation_experiment(
        t=t, output_dir=out_dir, plot=True, show=False
    )

    print()
    print("τ extraction: fit parametric τ to TDF correlation data...")
    fit_tau_from_correlations(
        corr_res["t"],
        corr_res["cxx_tdf"],
        corr_res["cyy_tdf"],
        corr_res["chsh_tdf"],
        initial_guess={
            "omega": 1.0,
            "a": 1.0,
            "nu": 3.0,
            "sigma": 0.5,
            "tau_c": 1.0,
        },
        bounds=[
            (0.2, 2.5),
            (0.05, 2.0),
            (0.8, 5.0),
            (0.0, 1.2),
            (0.12, 5.0),
        ],
        method="L-BFGS-B",
        output_dir=out_dir,
        save_plots=True,
        show=False,
    )


if __name__ == "__main__":
    main()
