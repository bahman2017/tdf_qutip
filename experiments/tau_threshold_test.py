"""
τ-noise **threshold** / crossover sweep (falsification).

Sweeps ``σ`` (τ noise strength) for single-qubit ``|+⟩`` **coherence lifetime** (time when
``C(t)`` drops below 0.5) and two-qubit Bell **CHSH** (same operators as ``tdf_chsh_decay``).

Detects an **elbow** in coherence lifetime vs ``σ`` and writes ``critical_sigma.txt`` as the
largest σ before lifetime collapses below a fraction of the baseline range (heuristic).

Outputs under ``outputs/threshold/``.

Run::

    python -m experiments.tau_threshold_test
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt
from scipy.optimize import curve_fit

from experiments.tdf_chsh_decay import chsh_S_from_E, chsh_expectations


def _coherence_lifetime(
    sigma: float,
    *,
    t: np.ndarray,
    n_ensemble: int,
    rng: np.random.Generator,
    omega: float,
) -> tuple[float, float]:
    """
    Returns (time when single-qubit C drops below 0.5, CHSH at last time on Bell).

    Uses **marginal** law ``τ(t) ~ 𝒩(ω t, σ² t)`` (Wiener at fixed ``t``) for speed.
    """
    n_times = t.size
    rho_p = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    rho_p = rho_p * rho_p.dag()
    z = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    o = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
    bell = (z + o).unit() * (z + o).unit().dag()
    sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
    sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())

    C_t = np.zeros(n_times)
    S_t = np.zeros(n_times)
    for kt in range(n_times):
        tt = float(t[kt])
        std = float(sigma * np.sqrt(max(tt, 1e-15)))
        acc01 = 0j
        acc_rho = np.zeros((4, 4), dtype=complex)
        for _ in range(n_ensemble):
            tau = float(rng.normal(omega * tt, std))
            U = (-1j * tau * qt.sigmaz() / 2.0).expm()
            r = U * rho_p * U.dag()
            acc01 += complex(r[0, 1])
            t1 = float(rng.normal(omega * tt, std))
            t2 = float(rng.normal(omega * tt, std))
            U1 = (-1j * t1 * sz1 / 2.0).expm()
            U2 = (-1j * t2 * sz2 / 2.0).expm()
            rb = U1 * U2 * bell * U2.dag() * U1.dag()
            acc_rho += rb.full()
        rho_bar = qt.Qobj(acc_rho / n_ensemble, dims=bell.dims)
        C_t[kt] = float(np.abs(acc01 / n_ensemble)) / float(
            np.abs(complex(rho_p[0, 1]))
        )
        e00, e01, e10, e11 = chsh_expectations(rho_bar)
        S_t[kt] = chsh_S_from_E(e00, e01, e10, e11)

    target = 0.5
    idx = np.where(C_t < target)[0]
    if idx.size == 0:
        t_half = float(t[-1])
    else:
        t_half = float(t[idx[0]])
    return t_half, float(S_t[-1])


def run_tau_threshold_test(
    *,
    omega: float = 0.0,
    sigma_grid: np.ndarray | None = None,
    t_max: float = 2.0,
    n_times: int = 120,
    n_ensemble: int = 350,
    seed: int = 2,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    out_root = output_dir or Path(__file__).resolve().parent.parent / "outputs" / "threshold"
    out_root.mkdir(parents=True, exist_ok=True)
    if sigma_grid is None:
        sigma_grid = np.linspace(0.05, 1.2, 14)

    t = np.linspace(0.0, t_max, n_times)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    lifetimes: list[float] = []
    chsh_end: list[float] = []

    for sig in sigma_grid:
        rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        tl, se = _coherence_lifetime(
            float(sig), t=t, n_ensemble=n_ensemble, rng=rng, omega=omega
        )
        lifetimes.append(tl)
        chsh_end.append(se)
        rows.append(
            {
                "sigma_tau": float(sig),
                "coherence_lifetime_t05": tl,
                "chsh_end_time": se,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "phase_transition.csv", index=False)

    # elbow: largest drop in d(lifetime)/dσ
    sg = sigma_grid.astype(float)
    lt = np.array(lifetimes)
    dlt = np.gradient(lt, sg)
    elbow_i = int(np.argmin(dlt))
    crit = float(sg[elbow_i])
    (out_root / "critical_sigma.txt").write_text(
        f"critical_sigma_heuristic={crit:.6f}\n"
        f"(index of min d(lifetime)/dσ in sigma_grid; interpret as crossover scale)\n",
        encoding="utf-8",
    )

    # sigmoid fit on CHSH end vs sigma
    def sigmoid(s, a, b, c, d):
        return a / (1.0 + np.exp(-b * (s - c))) + d

    try:
        popt, _ = curve_fit(
            sigmoid,
            sg,
            np.array(chsh_end),
            p0=[2.8, 8.0, 0.5, 0.0],
            maxfev=8000,
        )
        sigmoid_fit = {k: float(v) for k, v in zip(("a", "b", "c", "d"), popt)}
    except (RuntimeError, ValueError):
        sigmoid_fit = {"error": "fit_failed"}

    summary = {
        "critical_sigma_heuristic": crit,
        "sigmoid_chsh_vs_sigma": sigmoid_fit,
        "csv": str(out_root / "phase_transition.csv"),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    ax.plot(sg, lt, "o-", label="coherence lifetime (t @ C<0.5)")
    ax.set_xlabel("σ_τ")
    ax.set_ylabel("time")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(sg, chsh_end, "s--", color="C1", label="CHSH at t_max")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_root / "threshold_sweep.png", bbox_inches="tight")
    plt.close(fig)

    return {"output_dir": out_root, "summary": summary}


if __name__ == "__main__":
    run_tau_threshold_test()
