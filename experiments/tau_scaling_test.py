"""
Multi-qubit τ scaling falsification test.

For ``N ∈ {1, 2, 3, 5, 8}`` qubits, local ``σ_z`` phase noise with **independent** or
**correlated** Wiener ``τ_i``. Records ``Var(\\sum_i τ_i)`` at ``t_\\mathrm{max}`` and
normalized **GHZ** coherence (``N ≥ 2``) or ``|+⟩`` coherence (``N = 1``).

Outputs: ``outputs/scaling/scaling_data.csv``, ``fit_results.json``, plots.

Run::

    python -m experiments.tau_scaling_test
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt

from core.noise_models import wiener_path


def _ghz_dm(N: int) -> qt.Qobj:
    up = [qt.basis(2, 0)] * N
    dn = [qt.basis(2, 1)] * N
    psi = (qt.tensor(*up) + qt.tensor(*dn)).unit()
    return psi * psi.dag()


def _sz_k(N: int, k: int) -> qt.Qobj:
    ops = [qt.qeye(2)] * N
    ops[k] = qt.sigmaz()
    return qt.tensor(*ops)


def _ensemble_coherence_last(
    N: int,
    t: np.ndarray,
    dt: float,
    *,
    mode: str,
    sigma: float,
    omega: float,
    alpha_shared: float,
    n_ensemble: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return (Var(sum τ) at last time, normalized coherence at last time)."""
    n_times = int(t.size)
    kt = n_times - 1
    if N == 1:
        rho0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        rho0 = rho0 * rho0.dag()
        i0, i1 = 0, 1
    else:
        rho0 = _ghz_dm(N)
        i0, i1 = 0, 2**N - 1
    c0 = float(np.abs(complex(rho0[i0, i1])))

    sum_at_kt: list[float] = []
    coh_at_kt: list[float] = []

    for _ in range(n_ensemble):
        taus = np.zeros((N, n_times))
        if mode == "independent":
            for i in range(N):
                taus[i, :] = omega * t + sigma * wiener_path(rng, n_times, dt)
        else:
            w0 = wiener_path(rng, n_times, dt)
            a = float(alpha_shared)
            b = float(np.sqrt(max(0.0, 1.0 - a * a)))
            for i in range(N):
                wi = wiener_path(rng, n_times, dt)
                taus[i, :] = omega * t + sigma * (a * w0 + b * wi)
        ssum = np.sum(taus[:, kt])
        sum_at_kt.append(ssum)

        sz_ops = [_sz_k(N, k) for k in range(N)]
        if N == 1:
            U = (-1j * float(taus[0, kt]) * qt.sigmaz() / 2.0).expm()
            r = U * rho0 * U.dag()
        else:
            U_tot = qt.tensor([qt.qeye(2)] * N)
            for i in range(N):
                ang = float(taus[i, kt])
                Ui = (-1j * ang * sz_ops[i] / 2.0).expm()
                U_tot = Ui * U_tot
            r = U_tot * rho0 * U_tot.dag()
        coh_at_kt.append(float(np.abs(complex(r[i0, i1]))) / c0)

    var_sum = float(np.var(np.array(sum_at_kt), ddof=0))
    coh_mean = float(np.mean(coh_at_kt))
    return var_sum, coh_mean


def run_tau_scaling_test(
    *,
    omega: float = 0.0,
    sigma: float = 0.28,
    t_max: float = 1.5,
    n_times: int = 80,
    n_ensemble: int = 400,
    alpha_shared: float = 0.65,
    seed: int = 1,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    out_root = output_dir or Path(__file__).resolve().parent.parent / "outputs" / "scaling"
    out_root.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, t_max, n_times)
    dt = float(t[1] - t[0]) if t.size > 1 else 1.0
    Ns = [1, 2, 3, 5, 8]
    modes = ("independent", "correlated")
    rows: list[dict[str, Any]] = []

    for mode in modes:
        rng = np.random.default_rng(seed + hash(mode) % 10_000)
        for N in Ns:
            vs, ch = _ensemble_coherence_last(
                N,
                t,
                dt,
                mode=mode,
                sigma=sigma,
                omega=omega,
                alpha_shared=alpha_shared,
                n_ensemble=n_ensemble,
                rng=rng,
            )
            rows.append(
                {
                    "N": N,
                    "mode": mode,
                    "var_sum_tau": vs,
                    "coherence_last_t": ch,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "scaling_data.csv", index=False)

    fit_results: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=140)
    for ax, mode in zip(axes, modes):
        sub = df[df["mode"] == mode]
        x = np.log(sub["N"].values.astype(float))
        y = np.log(np.maximum(sub["var_sum_tau"].values, 1e-20))
        slope, intercept = np.polyfit(x, y, 1)
        fit_results[mode] = {
            "loglog_slope_var_vs_N": float(slope),
            "loglog_intercept": float(intercept),
            "hypothesis_hint": (
                "Var ~ N^a with a≈1 independent Wiener sum at fixed t"
                if mode == "independent"
                else "Var ~ N^a with larger a when shared component aligns"
            ),
        }
        ax.scatter(sub["N"], sub["var_sum_tau"], s=60, label="data")
        Nf = np.linspace(min(Ns), max(Ns), 50)
        ax.plot(Nf, np.exp(intercept) * (Nf**slope), "k--", lw=1.2, label=f"fit N^{slope:.2f}")
        ax.set_xlabel("N")
        ax.set_ylabel("Var(Σ τ_i) at t_max")
        ax.set_title(mode)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Scaling of Var(Σ τ) vs qubit count", y=1.02)
    fig.tight_layout()
    fig.savefig(out_root / "scaling_var_vs_N.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5.0, 3.6), dpi=140)
    for mode, m in zip(modes, ("o", "s")):
        sub = df[df["mode"] == mode]
        ax2.plot(sub["N"], sub["coherence_last_t"], m + "-", label=mode)
    ax2.set_xlabel("N")
    ax2.set_ylabel("mean coherence (last t)")
    ax2.set_title("Coherence vs N (ensemble mean at t_max)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_root / "scaling_coherence_vs_N.png", bbox_inches="tight")
    plt.close(fig2)

    fit_results["note"] = (
        "Compare slope to +1 (Var ∝ N independent), ~0 (flat), negative (1/N-like decay of Var per sum scaling — uncommon here)."
    )
    (out_root / "fit_results.json").write_text(
        json.dumps(fit_results, indent=2), encoding="utf-8"
    )
    return {"output_dir": out_root, "fit_results": fit_results, "dataframe": df}


if __name__ == "__main__":
    run_tau_scaling_test()
