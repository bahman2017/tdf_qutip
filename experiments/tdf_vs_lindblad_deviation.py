"""
Lindblad vs τ-stochastic **deviation** test (falsification).

Same single-qubit ``|+⟩`` system:

* **A)** Markovian dephasing ``L = \\sqrt{γ}\\,σ_z`` (QuTiP ``mesolve``).
* **B)** Classical τ noise: ``U = \\exp(-i τ σ_z/2)`` with ``τ = σ W(t)`` (Wiener marginal at each ``t``),
  ensemble-averaged ``ρ``.

**Short-time match:** choose ``γ = σ²/4`` so ``|ρ₀₁| ≈ \\exp(-2γ t) = \\exp(-σ² t/2)`` matches
Gaussian phase ``\\exp(-\\mathrm{Var}(τ)/2)`` with ``\\mathrm{Var}(τ)=σ² t`` at leading order.

**Long-time:** compare coherence, von Neumann entropy, and (optional) embedded Bell CHSH proxy on
``ρ ⊗ ρ`` is too heavy — we stick to **single-qubit** observables as specified.

Bootstrap over seeds for mean ``Δ(t) = |C_τ(t) − C_L(t)|``.

Outputs under ``outputs/deviation/``.

Run::

    python -m experiments.tdf_vs_lindblad_deviation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import qutip as qt

from experiments.tdf_chsh_decay import chsh_expectations, chsh_S_from_E


def _rho_plus() -> qt.Qobj:
    p = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    return p * p.dag()


def lindblad_coherence_curve(
    t: np.ndarray,
    *,
    gamma: float,
    rho0: qt.Qobj,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (C(t), S_vn(t)) for dephasing L = sqrt(gamma) sz."""
    c_ops = [np.sqrt(max(gamma, 0.0)) * qt.sigmaz()]
    out = qt.mesolve(0.0 * qt.qeye(2), rho0, t, c_ops, [])
    C = np.zeros(len(t))
    S = np.zeros(len(t))
    r0 = complex(rho0[0, 1])
    a0 = float(np.abs(r0))
    for k, st in enumerate(out.states):
        rho = st
        C[k] = float(np.abs(complex(rho[0, 1]))) / a0
        S[k] = float(qt.entropy_vn(rho))
    return C, S


def tau_ensemble_coherence_curve(
    t: np.ndarray,
    *,
    sigma: float,
    n_ensemble: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Wiener marginal τ ~ N(0, σ² t); ensemble-mean ρ then coherence and entropy."""
    rho0 = _rho_plus()
    r0 = complex(rho0[0, 1])
    a0 = float(np.abs(r0))
    C = np.zeros(len(t))
    S = np.zeros(len(t))
    for k, tt in enumerate(t):
        std = float(sigma * np.sqrt(max(float(tt), 1e-15)))
        acc = np.zeros((2, 2), dtype=complex)
        for _ in range(n_ensemble):
            tau = float(rng.normal(0.0, std))
            U = (-1j * tau * qt.sigmaz() / 2.0).expm()
            r = U * rho0 * U.dag()
            acc += r.full()
        rho_bar = qt.Qobj(acc / n_ensemble, dims=rho0.dims)
        C[k] = float(np.abs(complex(rho_bar[0, 1]))) / a0
        S[k] = float(qt.entropy_vn(rho_bar))
    return C, S


def _bell_dm() -> qt.Qobj:
    z = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    o = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
    psi = (z + o).unit()
    return psi * psi.dag()


def bell_chsh_curves(
    t: np.ndarray,
    *,
    sigma: float,
    gamma: float,
    n_ensemble: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent dephasing on each qubit vs independent τ noise; CHSH on ``ρ̄``."""
    rho0 = _bell_dm()
    sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
    sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())
    g = float(np.sqrt(max(gamma, 0.0)))
    c_ops = [g * sz1, g * sz2]
    out = qt.mesolve(0.0 * qt.tensor(qt.qeye(2), qt.qeye(2)), rho0, t, c_ops, [])
    S_L = np.zeros(len(t))
    for k, st in enumerate(out.states):
        e = chsh_expectations(st)
        S_L[k] = chsh_S_from_E(*e)

    S_T = np.zeros(len(t))
    for k, tt in enumerate(t):
        std = float(sigma * np.sqrt(max(float(tt), 1e-15)))
        acc = np.zeros((4, 4), dtype=complex)
        for _ in range(n_ensemble):
            t1 = float(rng.normal(0.0, std))
            t2 = float(rng.normal(0.0, std))
            U1 = (-1j * t1 * sz1 / 2.0).expm()
            U2 = (-1j * t2 * sz2 / 2.0).expm()
            r = U1 * U2 * rho0 * U2.dag() * U1.dag()
            acc += r.full()
        rho_bar = qt.Qobj(acc / n_ensemble, dims=rho0.dims)
        e = chsh_expectations(rho_bar)
        S_T[k] = chsh_S_from_E(*e)
    return S_T, S_L


def run_tdf_vs_lindblad_deviation(
    *,
    sigma: float = 0.45,
    t_max: float = 3.0,
    n_times: int = 100,
    n_ensemble: int = 800,
    n_bootstrap: int = 40,
    base_seed: int = 7,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    out_root = output_dir or Path(__file__).resolve().parent.parent / "outputs" / "deviation"
    out_root.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, t_max, n_times)
    gamma = float(sigma**2 / 4.0)
    rho0 = _rho_plus()
    C_L, S_L = lindblad_coherence_curve(t, gamma=gamma, rho0=rho0)

    delta_rows: list[dict[str, Any]] = []
    rng_master = np.random.default_rng(base_seed)
    all_delta: list[np.ndarray] = []

    for b in range(n_bootstrap):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        C_T, S_T = tau_ensemble_coherence_curve(
            t, sigma=sigma, n_ensemble=n_ensemble, rng=rng
        )
        dC = np.abs(C_T - C_L)
        dS = np.abs(S_T - S_L)
        all_delta.append(dC)
        for k in range(n_times):
            delta_rows.append(
                {
                    "bootstrap_id": b,
                    "time": float(t[k]),
                    "delta_coherence": float(dC[k]),
                    "delta_entropy": float(dS[k]),
                }
            )

    pd.DataFrame(delta_rows).to_csv(out_root / "deviation_vs_time.csv", index=False)

    stack = np.stack(all_delta, axis=0)
    rng_ch = np.random.default_rng(base_seed + 99)
    S_T_bell, S_L_bell = bell_chsh_curves(
        t, sigma=sigma, gamma=gamma, n_ensemble=n_ensemble, rng=rng_ch
    )
    delta_chsh = np.abs(S_T_bell - S_L_bell)

    stats = {
        "sigma_tau": sigma,
        "gamma_matched": gamma,
        "mean_delta_coherence": [float(x) for x in np.mean(stack, axis=0)],
        "std_delta_coherence": [float(x) for x in np.std(stack, axis=0, ddof=1)],
        "time": [float(x) for x in t],
        "max_mean_delta": float(np.max(np.mean(stack, axis=0))),
        "delta_chsh_bell_single_seed": [float(x) for x in delta_chsh],
        "note": "Short-time slopes matched via gamma=sigma^2/4; long-time divergence indicates non-Markovian memory in tau ensemble.",
    }
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return {"output_dir": out_root, "stats": stats}


if __name__ == "__main__":
    run_tdf_vs_lindblad_deviation()
