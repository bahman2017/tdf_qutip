"""
Non-Gaussian τ statistics falsification test.

Compares Gaussian (Wiener) cumulative τ to **Student-t**, **skew-normal**, and **bimodal**
increment models (:mod:`core.tau_non_gaussian`) on ensemble-averaged observables:

* single-qubit ``|+⟩`` coherence,
* two-qubit Bell ``(|00⟩+|11⟩)/√2`` ``|ρ_{0,3}|``,
* three-qubit GHZ ``|ρ_{0,7}|``.

**Key metric:** deviation of empirical decay from ``exp(-Var/2)`` (Gaussian phase law)
and qualitative detection of **non-monotone** / revival structure.

Outputs under ``outputs/non_gaussian/`` (CSV, PNG, ``summary.json``).

Run::

    python -m experiments.non_gaussian_tau_test
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt

from core.tau_non_gaussian import (
    cumulative_path_bimodal_gaussian,
    cumulative_path_gaussian_wiener,
    cumulative_path_skew_normal,
    cumulative_path_student_t,
)


def _rho_plus() -> qt.Qobj:
    p = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    return p * p.dag()


def _rho_bell() -> qt.Qobj:
    z = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    o = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
    psi = (z + o).unit()
    return psi * psi.dag()


def _rho_ghz() -> qt.Qobj:
    z = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
    o = qt.tensor(qt.basis(2, 1), qt.basis(2, 1), qt.basis(2, 1))
    psi = (z + o).unit()
    return psi * psi.dag()


def _U1(angle: float) -> qt.Qobj:
    return (-1j * angle * qt.sigmaz() / 2.0).expm()


def run_non_gaussian_tau_test(
    *,
    omega: float = 0.0,
    sigma: float = 0.35,
    t_max: float = 2.5,
    n_times: int = 150,
    n_ensemble: int = 600,
    seed: int = 0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    out_root = output_dir or Path(__file__).resolve().parent.parent / "outputs" / "non_gaussian"
    out_root.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, t_max, n_times)
    rng = np.random.default_rng(seed)

    models: dict[str, tuple[Callable[..., np.ndarray], dict[str, Any]]] = {
        "gaussian": (cumulative_path_gaussian_wiener, {"omega": omega, "sigma": sigma}),
        "student_t": (cumulative_path_student_t, {"omega": omega, "sigma": sigma, "df": 5.0}),
        "skew_normal": (
            cumulative_path_skew_normal,
            {"omega": omega, "sigma": sigma, "alpha": 4.0},
        ),
        "bimodal": (
            cumulative_path_bimodal_gaussian,
            {"omega": omega, "sigma": sigma, "p": 0.5, "delta": 0.65},
        ),
    }

    rows_coh: list[dict[str, Any]] = []
    rows_dev: list[dict[str, Any]] = []
    series_plot: dict[str, dict[str, np.ndarray]] = {}

    for sys_name, rho0, mode in (
        ("single_qubit", _rho_plus(), "single"),
        ("bell", _rho_bell(), "bell"),
        ("ghz3", _rho_ghz(), "ghz3"),
    ):
        sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2)) if mode != "single" else None
        sz2 = qt.tensor(qt.qeye(2), qt.sigmaz()) if mode == "bell" else None
        sz3 = (
            qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(2))
            if mode == "ghz3"
            else None
        )
        sz3b = (
            qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(2))
            if mode == "ghz3"
            else None
        )
        sz3c = (
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmaz())
            if mode == "ghz3"
            else None
        )

        for dist_name, (pfn, kw) in models.items():
            taus = []
            for _ in range(n_ensemble):
                if mode == "single":
                    taus.append(pfn(rng, t, **kw))
                elif mode == "bell":
                    taus.append(
                        (
                            pfn(rng, t, **kw),
                            pfn(rng, t, **kw),
                        )
                    )
                else:
                    taus.append(
                        (
                            pfn(rng, t, **kw),
                            pfn(rng, t, **kw),
                            pfn(rng, t, **kw),
                        )
                    )

            C = np.zeros(n_times)
            pred = np.zeros(n_times)
            for kt in range(n_times):
                if mode == "single":
                    tau = np.array([taus[j][kt] for j in range(n_ensemble)])
                    v = float(np.var(tau, ddof=0))
                    pred[kt] = float(np.exp(-0.5 * v))
                    acc = 0j
                    for j in range(n_ensemble):
                        U = _U1(float(taus[j][kt]))
                        r = U * rho0 * U.dag()
                        acc += complex(r[0, 1])
                    C[kt] = float(np.abs(acc / n_ensemble)) / float(
                        np.abs(complex(rho0[0, 1]))
                    )
                elif mode == "bell":
                    t1 = np.array([taus[j][0][kt] for j in range(n_ensemble)])
                    t2 = np.array([taus[j][1][kt] for j in range(n_ensemble)])
                    s = t1 + t2
                    v = float(np.var(s, ddof=0))
                    pred[kt] = float(np.exp(-0.5 * v))
                    acc = np.zeros((4, 4), dtype=complex)
                    for j in range(n_ensemble):
                        a1, a2 = float(t1[j]), float(t2[j])
                        U = (
                            (-1j * a1 * sz1 / 2.0).expm()
                            * (-1j * a2 * sz2 / 2.0).expm()
                        )
                        r = U * rho0 * U.dag()
                        acc += r.full()
                    rho_bar = qt.Qobj(acc / n_ensemble, dims=rho0.dims)
                    C[kt] = float(np.abs(complex(rho_bar[0, 3]))) / float(
                        np.abs(complex(rho0[0, 3]))
                    )
                else:
                    t1 = np.array([taus[j][0][kt] for j in range(n_ensemble)])
                    t2 = np.array([taus[j][1][kt] for j in range(n_ensemble)])
                    t3 = np.array([taus[j][2][kt] for j in range(n_ensemble)])
                    s = t1 + t2 + t3
                    v = float(np.var(s, ddof=0))
                    pred[kt] = float(np.exp(-0.5 * v))
                    acc = np.zeros((8, 8), dtype=complex)
                    for j in range(n_ensemble):
                        U = (
                            (-1j * float(t1[j]) * sz3 / 2.0).expm()
                            * (-1j * float(t2[j]) * sz3b / 2.0).expm()
                            * (-1j * float(t3[j]) * sz3c / 2.0).expm()
                        )
                        r = U * rho0 * U.dag()
                        acc += r.full()
                    rho_bar = qt.Qobj(acc / n_ensemble, dims=rho0.dims)
                    C[kt] = float(np.abs(complex(rho_bar[0, 7]))) / float(
                        np.abs(complex(rho0[0, 7]))
                    )

            dev = C - pred
            max_dev = float(np.max(np.abs(dev)))
            revival = bool(np.any(np.diff(C) > 0.02) and np.min(C) < 0.95)
            rows_dev.append(
                {
                    "system": sys_name,
                    "distribution": dist_name,
                    "max_abs_deviation": max_dev,
                    "mean_signed_deviation": float(np.mean(dev)),
                    "revival_flag": revival,
                }
            )
            for kt in range(n_times):
                rows_coh.append(
                    {
                        "time": float(t[kt]),
                        "system": sys_name,
                        "distribution": dist_name,
                        "coherence": float(C[kt]),
                        "gaussian_law_pred": float(pred[kt]),
                        "deviation_C_minus_pred": float(dev[kt]),
                    }
                )
            series_plot[f"{sys_name}_{dist_name}"] = {"t": t, "C": C, "pred": pred}

    pd.DataFrame(rows_coh).to_csv(out_root / "coherence_vs_time.csv", index=False)
    pd.DataFrame(rows_dev).to_csv(out_root / "deviation_from_gaussian.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), dpi=140, sharey=False)
    for ax, sys_name in zip(axes, ("single_qubit", "bell", "ghz3")):
        for dist_name, sty in zip(
            models.keys(),
            ("-", "--", "-.", ":"),
        ):
            k = f"{sys_name}_{dist_name}"
            d = series_plot[k]
            ax.plot(d["t"], d["C"], ls=sty, lw=1.8, label=f"{dist_name}")
        ax.set_title(sys_name)
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("normalized coherence")
    fig.suptitle("Non-Gaussian τ: coherence decay vs distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(out_root / "coherence_non_gaussian_compare.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.0, 4.0), dpi=140)
    df_d = pd.DataFrame(rows_dev)
    piv = df_d.pivot(index="system", columns="distribution", values="max_abs_deviation")
    piv.plot(kind="bar", ax=ax2, rot=0)
    ax2.set_ylabel("max |C − exp(−Var/2)|")
    ax2.set_title("Deviation from Gaussian phase law")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(title="distribution", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_root / "deviation_summary.png", bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "parameters": {
            "omega": omega,
            "sigma": sigma,
            "t_max": t_max,
            "n_times": n_times,
            "n_ensemble": n_ensemble,
            "seed": seed,
        },
        "deviation_table": rows_dev,
        "outputs": {
            "coherence_vs_time_csv": str(out_root / "coherence_vs_time.csv"),
            "deviation_from_gaussian_csv": str(out_root / "deviation_from_gaussian.csv"),
            "plots": [
                str(out_root / "coherence_non_gaussian_compare.png"),
                str(out_root / "deviation_summary.png"),
            ],
        },
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return {"output_dir": out_root, "summary": summary}


if __name__ == "__main__":
    run_non_gaussian_tau_test()
