"""
TDF falsification test suite — orchestrates all falsification experiments and writes
``outputs/falsification/summary.json``.

Used by ``python main.py --run falsification_tests``.

Individual modules can still be run with ``python -m experiments.<name>``.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from experiments.non_gaussian_tau_test import run_non_gaussian_tau_test
from experiments.tau_scaling_test import run_tau_scaling_test
from experiments.tau_threshold_test import run_tau_threshold_test
from experiments.tdf_vs_lindblad_deviation import run_tdf_vs_lindblad_deviation


def run_all_falsification_tests(
    *,
    output_root: Path | None = None,
    fast: bool = False,
) -> dict[str, Any]:
    """
    Run non-Gaussian, scaling, threshold, and Lindblad-deviation experiments.

    Parameters
    ----------
    fast
        If True, use smaller ensembles and fewer σ points for CI smoke tests.
    """
    root = output_root or Path(__file__).resolve().parent.parent / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    fals_dir = root / "falsification"
    fals_dir.mkdir(parents=True, exist_ok=True)

    ne = 280 if fast else 500
    nb = 12 if fast else 35

    results: dict[str, Any] = {"modules": {}}

    def _run(name: str, fn: Any, **kwargs: Any) -> None:
        try:
            out = fn(**kwargs)
            results["modules"][name] = {"status": "ok", "return_keys": list(out.keys())}
        except Exception as e:  # noqa: BLE001
            results["modules"][name] = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    _run(
        "non_gaussian_tau_test",
        run_non_gaussian_tau_test,
        n_ensemble=ne,
        n_times=100 if fast else 120,
    )
    _run(
        "tau_scaling_test",
        run_tau_scaling_test,
        n_ensemble=ne,
    )
    _run(
        "tau_threshold_test",
        run_tau_threshold_test,
        n_ensemble=ne,
        sigma_grid=np.linspace(0.08, 1.0, 10 if fast else 14),
    )
    _run(
        "tdf_vs_lindblad_deviation",
        run_tdf_vs_lindblad_deviation,
        n_ensemble=ne,
        n_bootstrap=nb,
    )

    summary_path = fals_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["summary_path"] = str(summary_path)
    return results


if __name__ == "__main__":
    out = run_all_falsification_tests()
    print(json.dumps(out, indent=2))
