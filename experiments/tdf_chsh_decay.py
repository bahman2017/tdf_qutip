"""
CHSH (Bell) parameter under TDF-style local ``σ_z`` phase noise (QuTiP).

Two qubits, initial Bell ``|Φ⁺⟩ = (|00⟩+|11⟩)/√2``. Same commuting unitaries and
``τ₁, τ₂`` cases as ``experiments/tdf_entanglement_decay`` (independent / shared /
partial Wiener legs).

**CHSH** (product observables on ``ρ``):

    A₀ = σ_z ⊗ I,   A₁ = σ_x ⊗ I
    B₀ = I ⊗ (σ_z + σ_x)/√2,   B₁ = I ⊗ (σ_z - σ_x)/√2

    E(A_i, B_j) = Tr(ρ (A_i ⊗ B_j)),  (i,j) ∈ {0,1}²

    S = |E(A₀,B₀) + E(A₀,B₁) + E(A₁,B₀) − E(A₁,B₁)|

For pure ``|Φ⁺⟩`` with these settings, ``S(0) = 2√2`` (Tsirelson). Classical LHV bound:
``S ≤ 2``.

**Component TDF test:** with ``\\mathrm{Var}_{\\mathrm{sum}}(t) = \\mathrm{Var}(τ₁+τ₂)``
across trajectories,

    E_{ij}^{\\mathrm{pred}}(t) = E_{ij}(0)\\,\\exp(-\\mathrm{Var}_{\\mathrm{sum}}/2).

Reconstructed CHSH magnitude:

    S_{\\mathrm{pred}} = |E_{00}^{\\mathrm{pred}} + E_{01}^{\\mathrm{pred}}
        + E_{10}^{\\mathrm{pred}} - E_{11}^{\\mathrm{pred}}|.

Because each term uses the **same** exponential factor, this equals
``|E_{00}(0)+\\cdots|\\,\\exp(-\\mathrm{Var}/2) = S(0)\\exp(-\\mathrm{Var}/2)`` — the
same as the naive single-exponential ``S`` prediction.

**Multiphase (per-component variance):** use ensemble variances

    \\mathrm{Var}_{00}=\\mathrm{Var}(τ₁+τ₂),\\quad
    \\mathrm{Var}_{01}=\\mathrm{Var}(τ₁-τ₂),\\quad
    \\mathrm{Var}_{10}=\\mathrm{Var}(-τ₁+τ₂)=\\mathrm{Var}_{01},\\quad
    \\mathrm{Var}_{11}=\\mathrm{Var}(-τ₁-τ₂)=\\mathrm{Var}_{00},

so ``E_{ij}^{\\mathrm{pred}} = E_{ij}(0)\\exp(-\\mathrm{Var}_{ij}/2)`` and
``S_{\\mathrm{pred}}^{\\mathrm{multi}} = |\\sum (\\pm) E_{ij}^{\\mathrm{pred}}|``.
This can differ from ``S(0)\\exp(-\\mathrm{Var}(τ₁+τ₂)/2)`` when
``\\mathrm{Var}(τ₁+τ₂)\\neq\\mathrm{Var}(τ₁-τ₂)`` (e.g. partial / independent cases).

**Characteristic-function (CF) component model:** for phases
``\\phi_{00}=τ₁+τ₂``, ``\\phi_{01}=τ₁-τ₂``, ``\\phi_{10}=-τ₁+τ₂``, ``\\phi_{11}=-τ₁-τ₂``,

    \\mathrm{CF}_{ij}(t) = \\mathbb{E}_j[e^{i\\phi_{ij}}],\\quad
    E_{ij}^{\\mathrm{pred}} = E_{ij}(0)\\,\\mathrm{Re}(\\mathrm{CF}_{ij}).

This uses the **full** empirical phase distribution (modulo this ansatz), not only
variance. For exact Gaussian ``\\phi``, ``\\mathrm{Re}(\\mathrm{CF})=\\exp(-\\mathrm{Var}(\\phi)/2)``,
recovering the variance law.

Run::

    python -m experiments.tdf_chsh_decay
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt

from experiments.tdf_entanglement_decay import simulate_tau1_tau2_cases
from experiments.tdf_phase_decoherence_test import _mse_r2_pearson


def _bell_dm() -> qt.Qobj:
    z = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    o = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
    psi = (z + o).unit()
    return psi * psi.dag()


def _chsh_joint_operators() -> list[qt.Qobj]:
    sz, sx = qt.sigmaz(), qt.sigmax()
    b0 = (sz + sx) / np.sqrt(2.0)
    b1 = (sz - sx) / np.sqrt(2.0)
    return [
        qt.tensor(sz, b0),
        qt.tensor(sz, b1),
        qt.tensor(sx, b0),
        qt.tensor(sx, b1),
    ]


def chsh_expectations(rho: qt.Qobj) -> tuple[float, float, float, float]:
    """``(E00, E01, E10, E11)`` for two-qubit ``ρ``."""
    o00, o01, o10, o11 = _chsh_joint_operators()
    return (
        float(qt.expect(o00, rho)),
        float(qt.expect(o01, rho)),
        float(qt.expect(o10, rho)),
        float(qt.expect(o11, rho)),
    )


def chsh_S_from_E(
    e00: float, e01: float, e10: float, e11: float
) -> float:
    return float(abs(e00 + e01 + e10 - e11))


def chsh_S(rho: qt.Qobj) -> float:
    return chsh_S_from_E(*chsh_expectations(rho))


def chsh_simulation_from_tau(
    tau1: np.ndarray,
    tau2: np.ndarray,
    *,
    rho0: qt.Qobj,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns

    * ``E`` — shape ``(3, 4, n_times)``: ``E[case, 0..3, t]`` = ``E00..E11``
    * ``S_out`` — shape ``(3, n_times)``
    * ``pred_S`` — ``S(0) * exp(-Var(τ₁+τ₂)/2)``
    * ``var_sum`` — ``Var(τ₁+τ₂)``
    * ``var_diff`` — ``Var(τ₁-τ₂)`` (same as ``Var(-τ₁+τ₂)``)
    """
    n_cases, n_ens, n_times = tau1.shape
    sz_i = qt.tensor(qt.sigmaz(), qt.qeye(2))
    isz = qt.tensor(qt.qeye(2), qt.sigmaz())

    e00_0, e01_0, e10_0, e11_0 = chsh_expectations(rho0)
    s0 = chsh_S_from_E(e00_0, e01_0, e10_0, e11_0)

    E = np.zeros((n_cases, 4, n_times), dtype=float)
    S_out = np.zeros((n_cases, n_times), dtype=float)
    pred_S = np.zeros((n_cases, n_times), dtype=float)
    var_sum = np.zeros((n_cases, n_times), dtype=float)
    var_diff = np.zeros((n_cases, n_times), dtype=float)

    for ci in range(n_cases):
        for kt in range(n_times):
            s = tau1[ci, :, kt] + tau2[ci, :, kt]
            d = tau1[ci, :, kt] - tau2[ci, :, kt]
            vs = float(np.var(s, ddof=0))
            vd = float(np.var(d, ddof=0))
            var_sum[ci, kt] = vs
            var_diff[ci, kt] = vd
            pred_S[ci, kt] = float(s0 * np.exp(-0.5 * vs))

            acc = np.zeros((4, 4), dtype=complex)
            for j in range(n_ens):
                a1 = float(tau1[ci, j, kt])
                a2 = float(tau2[ci, j, kt])
                u1 = (-1j * a1 * sz_i / 2.0).expm()
                u2 = (-1j * a2 * isz / 2.0).expm()
                u = u1 * u2
                r = u * rho0 * u.dag()
                acc += r.full()
            rho_bar = qt.Qobj(acc / n_ens, dims=rho0.dims)
            e00, e01, e10, e11 = chsh_expectations(rho_bar)
            E[ci, 0, kt] = e00
            E[ci, 1, kt] = e01
            E[ci, 2, kt] = e10
            E[ci, 3, kt] = e11
            S_out[ci, kt] = chsh_S_from_E(e00, e01, e10, e11)

    return E, S_out, pred_S, var_sum, var_diff


def chsh_cf_predictions_from_tau(
    tau1: np.ndarray,
    tau2: np.ndarray,
    e00_0: float,
    e01_0: float,
    e10_0: float,
    e11_0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Empirical characteristic functions of phase combinations and CHSH reconstruction.

    Returns
    -------
    CF
        Complex, shape ``(3, 4, n_times)`` — ``CF_{00}..CF_{11}``.
    E_pred_cf
        ``E_ij(0) * Re(CF_ij)``, shape ``(3, 4, n_times)``.
    S_pred_cf
        ``|E00 + E01 + E10 - E11|`` from ``E_pred_cf``, shape ``(3, n_times)``.
    """
    n_cases, _, n_times = tau1.shape
    CF = np.zeros((n_cases, 4, n_times), dtype=np.complex128)
    for ci in range(n_cases):
        t1 = tau1[ci]
        t2 = tau2[ci]
        ph00 = t1 + t2
        ph01 = t1 - t2
        ph10 = -t1 + t2
        ph11 = -t1 - t2
        CF[ci, 0] = np.mean(np.exp(1j * ph00), axis=0)
        CF[ci, 1] = np.mean(np.exp(1j * ph01), axis=0)
        CF[ci, 2] = np.mean(np.exp(1j * ph10), axis=0)
        CF[ci, 3] = np.mean(np.exp(1j * ph11), axis=0)

    cf_r = CF.real
    E_cf = np.zeros((n_cases, 4, n_times), dtype=float)
    E_cf[:, 0, :] = e00_0 * cf_r[:, 0, :]
    E_cf[:, 1, :] = e01_0 * cf_r[:, 1, :]
    E_cf[:, 2, :] = e10_0 * cf_r[:, 2, :]
    E_cf[:, 3, :] = e11_0 * cf_r[:, 3, :]
    S_cf = np.abs(
        E_cf[:, 0, :] + E_cf[:, 1, :] + E_cf[:, 2, :] - E_cf[:, 3, :]
    )
    return CF, E_cf, S_cf


def run_chsh_decay_experiment(
    *,
    omega: float = 1.0,
    sigma: float = 0.3,
    rho: float = 0.6,
    t_max: float = 2.0,
    n_times: int = 200,
    n_ensemble: int = 800,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rho0 = _bell_dm()
    e00_0, e01_0, e10_0, e11_0 = chsh_expectations(rho0)
    s0 = chsh_S_from_E(e00_0, e01_0, e10_0, e11_0)
    if abs(s0 - 2.0 * np.sqrt(2.0)) > 1e-6:
        raise RuntimeError(f"unexpected S(0)={s0}, expected 2√2")

    t = np.linspace(0.0, float(t_max), int(n_times), dtype=float)
    tau1, tau2 = simulate_tau1_tau2_cases(t, omega, sigma, n_ensemble, rho, seed)
    E, S_arr, pred_S, var_sum, var_diff = chsh_simulation_from_tau(
        tau1, tau2, rho0=rho0
    )

    exp_factor = np.exp(-0.5 * var_sum)
    E_pred = np.zeros_like(E)
    for k in range(4):
        e0 = (e00_0, e01_0, e10_0, e11_0)[k]
        E_pred[:, k, :] = e0 * exp_factor

    S_recon = np.abs(
        E_pred[:, 0, :]
        + E_pred[:, 1, :]
        + E_pred[:, 2, :]
        - E_pred[:, 3, :]
    )

    # Multiphase: Var_00,Var_11 -> Var(τ1+τ2); Var_01,Var_10 -> Var(τ1-τ2)
    exp_sum = np.exp(-0.5 * var_sum)
    exp_diff = np.exp(-0.5 * var_diff)
    E_pred_multi = np.zeros_like(E)
    E_pred_multi[:, 0, :] = e00_0 * exp_sum
    E_pred_multi[:, 1, :] = e01_0 * exp_diff
    E_pred_multi[:, 2, :] = e10_0 * exp_diff
    E_pred_multi[:, 3, :] = e11_0 * exp_sum
    S_pred_multi = np.abs(
        E_pred_multi[:, 0, :]
        + E_pred_multi[:, 1, :]
        + E_pred_multi[:, 2, :]
        - E_pred_multi[:, 3, :]
    )

    CF, E_pred_cf, S_pred_cf = chsh_cf_predictions_from_tau(
        tau1, tau2, e00_0, e01_0, e10_0, e11_0
    )

    display_names = ("A: independent", "B: shared", "C: partial")
    case_keys = ("A", "B", "C")
    comp_labels = ("E00", "E01", "E10", "E11")

    # --- components CSV: time, case, E00, E01, E10, E11 ---
    comp_rows: list[dict[str, Any]] = []
    for ci, ck in enumerate(case_keys):
        for kt in range(len(t)):
            comp_rows.append(
                {
                    "time": float(t[kt]),
                    "case": ck,
                    "E00": float(E[ci, 0, kt]),
                    "E01": float(E[ci, 1, kt]),
                    "E10": float(E[ci, 2, kt]),
                    "E11": float(E[ci, 3, kt]),
                }
            )
    components_path = output_dir / "tdf_chsh_components.csv"
    pd.DataFrame(comp_rows).to_csv(components_path, index=False)

    # --- per-component metrics (TDF law): include case for clarity ---
    component_metric_rows: list[dict[str, Any]] = []
    for ci, ck in enumerate(case_keys):
        for k, clab in enumerate(comp_labels):
            mse, r2, pear = _mse_r2_pearson(E[ci, k, :], E_pred[ci, k, :])
            component_metric_rows.append(
                {
                    "component": clab,
                    "case": ck,
                    "mse": mse,
                    "r2": r2,
                    "pearson_corr": pear,
                }
            )
    comp_metrics_df = pd.DataFrame(component_metric_rows)[
        ["component", "case", "mse", "r2", "pearson_corr"]
    ]
    comp_metrics_path = output_dir / "tdf_chsh_component_metrics.csv"
    comp_metrics_df.to_csv(comp_metrics_path, index=False)

    # --- legacy S-level metrics CSV ---
    pred_norm = exp_factor
    metrics_rows: list[dict[str, Any]] = []
    for ci, disp in enumerate(display_names):
        mse, r2, pear = _mse_r2_pearson(S_arr[ci], pred_S[ci])
        mse_n, r2_n, _ = _mse_r2_pearson(S_arr[ci] / s0, pred_norm[ci])
        mse_rec, r2_rec, pear_rec = _mse_r2_pearson(S_arr[ci], S_recon[ci])
        metrics_rows.append(
            {
                "case": disp,
                "mse_S_vs_S0_exp_neg_varsum_over_2": mse,
                "r2_S_vs_S0_exp_neg_varsum_over_2": r2,
                "pearson_S_vs_prediction": pear,
                "mse_S_over_S0_vs_exp_neg_varsum_over_2": mse_n,
                "r2_S_over_S0_vs_exp_neg_varsum_over_2": r2_n,
                "mse_S_vs_S_reconstructed_from_components": mse_rec,
                "r2_S_vs_S_reconstructed_from_components": r2_rec,
                "pearson_S_vs_S_reconstructed": pear_rec,
                "var_sum_tau_at_tmax": float(var_sum[ci, -1]),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "tdf_chsh_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    multiphase_metric_rows: list[dict[str, Any]] = []
    for ci, ck in enumerate(case_keys):
        m_sv, r_sv, p_sv = _mse_r2_pearson(S_arr[ci], pred_S[ci])
        multiphase_metric_rows.append(
            {
                "case": ck,
                "model_type": "single_var",
                "mse": m_sv,
                "r2": r_sv,
                "pearson_corr": p_sv,
            }
        )
        m_mp, r_mp, p_mp = _mse_r2_pearson(S_arr[ci], S_pred_multi[ci])
        multiphase_metric_rows.append(
            {
                "case": ck,
                "model_type": "multiphase",
                "mse": m_mp,
                "r2": r_mp,
                "pearson_corr": p_mp,
            }
        )
    multiphase_metrics_df = pd.DataFrame(multiphase_metric_rows)[
        ["model_type", "case", "mse", "r2", "pearson_corr"]
    ]
    multiphase_metrics_path = output_dir / "tdf_chsh_multiphase_metrics.csv"
    multiphase_metrics_df.to_csv(multiphase_metrics_path, index=False)

    cf_metric_rows: list[dict[str, Any]] = []
    for ci, ck in enumerate(case_keys):
        m_sv, r_sv, p_sv = _mse_r2_pearson(S_arr[ci], pred_S[ci])
        cf_metric_rows.append(
            {
                "model_type": "single_var",
                "case": ck,
                "mse": m_sv,
                "r2": r_sv,
                "pearson_corr": p_sv,
            }
        )
        m_mp, r_mp, p_mp = _mse_r2_pearson(S_arr[ci], S_pred_multi[ci])
        cf_metric_rows.append(
            {
                "model_type": "multiphase_var",
                "case": ck,
                "mse": m_mp,
                "r2": r_mp,
                "pearson_corr": p_mp,
            }
        )
        m_cf, r_cf, p_cf = _mse_r2_pearson(S_arr[ci], S_pred_cf[ci])
        cf_metric_rows.append(
            {
                "model_type": "characteristic_function",
                "case": ck,
                "mse": m_cf,
                "r2": r_cf,
                "pearson_corr": p_cf,
            }
        )
    cf_metrics_df = pd.DataFrame(cf_metric_rows)[
        ["model_type", "case", "mse", "r2", "pearson_corr"]
    ]
    cf_metrics_path = output_dir / "tdf_chsh_cf_metrics.csv"
    cf_metrics_df.to_csv(cf_metrics_path, index=False)

    cf_comp_rows: list[dict[str, Any]] = []
    for ci, ck in enumerate(case_keys):
        for kt in range(len(t)):
            cf_comp_rows.append(
                {
                    "time": float(t[kt]),
                    "case": ck,
                    "CF00_real": float(CF[ci, 0, kt].real),
                    "CF01_real": float(CF[ci, 1, kt].real),
                    "CF10_real": float(CF[ci, 2, kt].real),
                    "CF11_real": float(CF[ci, 3, kt].real),
                    "E00_pred_cf": float(E_pred_cf[ci, 0, kt]),
                    "E01_pred_cf": float(E_pred_cf[ci, 1, kt]),
                    "E10_pred_cf": float(E_pred_cf[ci, 2, kt]),
                    "E11_pred_cf": float(E_pred_cf[ci, 3, kt]),
                }
            )
    cf_components_path = output_dir / "tdf_chsh_cf_components.csv"
    pd.DataFrame(cf_comp_rows).to_csv(cf_components_path, index=False)

    classical = 2.0
    tsirelson = float(s0)

    fig1, ax1 = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax1.axhline(classical, color="k", ls="--", lw=1.2, label="classical bound $S=2$")
    ax1.axhline(
        tsirelson,
        color="0.5",
        ls=":",
        lw=1.0,
        alpha=0.85,
        label=r"Tsirelson $S=2\sqrt{2}$ (noiseless)",
    )
    ax1.plot(t, S_arr[0], lw=2.0, label="A: independent")
    ax1.plot(t, S_arr[1], lw=2.0, label="B: shared")
    ax1.plot(t, S_arr[2], lw=2.0, label=f"C: partial ρ={rho}")
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"CHSH $S$")
    ax1.set_title(r"Bell $|\Phi^+\rangle$ + local $σ_z$ phase noise: CHSH vs time")
    ax1.grid(True, alpha=0.35)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    fig1.tight_layout()
    p1 = output_dir / "tdf_chsh_decay.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), dpi=150, sharey=True)
    titles = (
        "A: independent noise",
        "B: shared noise",
        f"C: partial (ρ={rho})",
    )
    for ax, ci, title in zip(axes, (0, 1, 2), titles):
        ax.plot(t, S_arr[ci], "C0", lw=2.0, label=r"$S(t)$")
        ax.plot(
            t,
            pred_S[ci],
            "C1",
            ls="--",
            lw=2.0,
            label=r"$S(0)\,e^{-\mathrm{Var}(\tau_1+\tau_2)/2}$",
        )
        ax.axhline(classical, color="k", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel("time")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel(r"$S$")
    fig2.suptitle(
        r"CHSH vs $S(0)\exp(-\mathrm{Var}(\tau_1+\tau_2)/2)$ (dotted: $S=2$ classical)",
        fontsize=10,
        y=1.03,
    )
    fig2.tight_layout()
    p2 = output_dir / "tdf_chsh_compare.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)

    # --- 4 panels: E_ij vs E_ij(0) exp(-Var/2) for all cases ---
    fig_c, axes_c = plt.subplots(2, 2, figsize=(9.5, 7.0), dpi=150, sharex=True)
    colors = ("C0", "C1", "C2")
    for ax, k, title in zip(
        axes_c.flat,
        range(4),
        (r"$E_{00}$", r"$E_{01}$", r"$E_{10}$", r"$E_{11}$"),
    ):
        for ci, ck, col in zip(range(3), case_keys, colors):
            ax.plot(t, E[ci, k, :], color=col, lw=2.0, label=f"{ck} data")
            ax.plot(
                t,
                E_pred[ci, k, :],
                color=col,
                ls="--",
                lw=1.2,
                alpha=0.85,
                label=f"{ck} $E(0)e^{{-\\mathrm{{Var}}/2}}$",
            )
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=6, ncol=2)
    for ax in axes_c[1, :]:
        ax.set_xlabel("time")
    axes_c[0, 0].set_ylabel("expectation")
    axes_c[1, 0].set_ylabel("expectation")
    fig_c.suptitle(
        r"CHSH correlations vs $E_{ij}(0)\exp(-\mathrm{Var}(\tau_1+\tau_2)/2)$",
        fontsize=11,
        y=1.01,
    )
    fig_c.tight_layout()
    p_components = output_dir / "tdf_chsh_components.png"
    fig_c.savefig(p_components, bbox_inches="tight")
    plt.close(fig_c)

    # --- Reconstructed S_pred vs actual S ---
    fig_r, axes_r = plt.subplots(1, 3, figsize=(12.0, 3.8), dpi=150, sharey=True)
    for ax, ci, title in zip(axes_r, (0, 1, 2), titles):
        ax.plot(t, S_arr[ci], "C0", lw=2.0, label=r"$S(t)$ actual")
        ax.plot(
            t,
            S_recon[ci],
            "C3",
            ls="--",
            lw=2.0,
            label=r"$S_{\mathrm{pred}}$ from components",
        )
        ax.plot(
            t,
            pred_S[ci],
            "C1",
            ls=":",
            lw=1.5,
            alpha=0.8,
            label=r"$S(0)e^{-\mathrm{Var}/2}$",
        )
        ax.axhline(classical, color="k", ls=":", lw=0.9, alpha=0.5)
        ax.set_xlabel("time")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=6)
    axes_r[0].set_ylabel(r"$S$")
    fig_r.suptitle(
        r"Actual $S$ vs $S_{\mathrm{pred}}$ from components "
        r"($S_{\mathrm{pred}}\equiv S(0)e^{-\mathrm{Var}/2}$ with shared $\mathrm{Var}(\tau_1+\tau_2)$)",
        fontsize=10,
        y=1.03,
    )
    fig_r.tight_layout()
    p_recon = output_dir / "tdf_chsh_reconstructed.png"
    fig_r.savefig(p_recon, bbox_inches="tight")
    plt.close(fig_r)

    fig_mp, axes_mp = plt.subplots(1, 3, figsize=(12.0, 3.8), dpi=150, sharey=True)
    for ax, ci, title in zip(axes_mp, (0, 1, 2), titles):
        ax.plot(t, S_arr[ci], "C0", lw=2.2, label=r"$S(t)$ actual")
        ax.plot(
            t,
            pred_S[ci],
            "C1",
            ls="--",
            lw=2.0,
            label=r"single $\mathrm{Var}(\tau_1+\tau_2)$",
        )
        ax.plot(
            t,
            S_pred_multi[ci],
            "C2",
            ls="-.",
            lw=2.0,
            label=r"multiphase $\mathrm{Var}_{ij}$",
        )
        ax.axhline(classical, color="k", ls=":", lw=1.0, alpha=0.65)
        ax.set_xlabel("time")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=6)
    axes_mp[0].set_ylabel(r"$S$")
    fig_mp.suptitle(
        r"CHSH: actual $S$ vs single-var vs multiphase reconstructed $S_{\mathrm{pred}}$",
        fontsize=10,
        y=1.03,
    )
    fig_mp.tight_layout()
    p_multiphase = output_dir / "tdf_chsh_multiphase_compare.png"
    fig_mp.savefig(p_multiphase, bbox_inches="tight")
    plt.close(fig_mp)

    fig_cf, axes_cf = plt.subplots(1, 3, figsize=(12.5, 3.9), dpi=150, sharey=True)
    for ax, ci, title in zip(axes_cf, (0, 1, 2), titles):
        ax.plot(t, S_arr[ci], "C0", lw=2.3, label=r"$S(t)$ actual")
        ax.plot(
            t,
            pred_S[ci],
            "C1",
            ls="--",
            lw=2.0,
            label="single-var",
        )
        ax.plot(
            t,
            S_pred_multi[ci],
            "C2",
            ls="-.",
            lw=2.0,
            label="multiphase-var",
        )
        ax.plot(
            t,
            S_pred_cf[ci],
            "C3",
            ls=(0, (3, 1, 1, 1)),
            lw=2.0,
            label="char. fn.",
        )
        ax.axhline(classical, color="k", ls=":", lw=1.0, alpha=0.65)
        ax.set_xlabel("time")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=5.5)
    axes_cf[0].set_ylabel(r"$S$")
    fig_cf.suptitle(
        r"CHSH: actual $S$ vs single-var, multiphase-var, and characteristic-function $S_{\mathrm{pred}}$",
        fontsize=10,
        y=1.03,
    )
    fig_cf.tight_layout()
    p_cf_compare = output_dir / "tdf_chsh_cf_compare.png"
    fig_cf.savefig(p_cf_compare, bbox_inches="tight")
    plt.close(fig_cf)

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p_components}")
    print(f"Wrote {p_recon}")
    print(f"Wrote {components_path}")
    print(f"Wrote {comp_metrics_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {multiphase_metrics_path}")
    print(f"Wrote {p_multiphase}")
    print(f"Wrote {cf_metrics_path}")
    print(f"Wrote {cf_components_path}")
    print(f"Wrote {p_cf_compare}")
    print(f"S(0) = {s0:.6f} (2√2)")
    print(
        f"Parameters: ω={omega}, σ={sigma}, T={t_max}, n_times={n_times}, "
        f"N={n_ensemble}, ρ(partial)={rho}, seed={seed}"
    )
    print()
    print("--- S(t) vs S(0) exp(-Var(τ₁+τ₂)/2) (CHSH scale) ---")
    for row in metrics_rows:
        pc = row["pearson_S_vs_prediction"]
        pc_s = f"{pc:.6f}" if np.isfinite(pc) else "nan"
        print(
            f"{row['case']}: MSE={row['mse_S_vs_S0_exp_neg_varsum_over_2']:.6e}  "
            f"R²={row['r2_S_vs_S0_exp_neg_varsum_over_2']:.6f}  Corr={pc_s}"
        )
    print()
    print("--- S/S(0) vs exp(-Var(τ₁+τ₂)/2) (same Pearson as above) ---")
    for row in metrics_rows:
        print(
            f"{row['case']}: MSE={row['mse_S_over_S0_vs_exp_neg_varsum_over_2']:.6e}  "
            f"R²={row['r2_S_over_S0_vs_exp_neg_varsum_over_2']:.6f}"
        )
    print()
    print("--- S actual vs S_pred from component fits (same exp for all E_ij) ---")
    for row in metrics_rows:
        pc = row["pearson_S_vs_S_reconstructed"]
        pc_s = f"{pc:.6f}" if np.isfinite(pc) else "nan"
        print(
            f"{row['case']}: MSE={row['mse_S_vs_S_reconstructed_from_components']:.6e}  "
            f"R²={row['r2_S_vs_S_reconstructed_from_components']:.6f}  Corr={pc_s}"
        )
    recon_vs_naive = float(np.max(np.abs(S_recon - pred_S)))
    print()
    print(
        f"max |S_reconstructed - S(0)exp(-Var/2)| = {recon_vs_naive:.3e} "
        "(same formula: one exp(-Var/2) on all four E_ij(0))"
    )
    print()
    print(
        "Key: S_pred from component TDF models is *identical* to S(0)exp(-Var/2) here, "
        "so it cannot improve on the naive CHSH-vs-exponential mismatch. "
        "Per-component metrics show where each E_ij tracks E_ij(0)exp(-Var/2); "
        "nonlinearity remains in |·| and in any time the CHSH *sum* of true E_ij "
        "deviates from that shared exponential."
    )
    print()
    print("--- Single-var vs multiphase CHSH prediction (see tdf_chsh_multiphase_metrics.csv) ---")
    for ci, disp in enumerate(display_names):
        row_sv = multiphase_metric_rows[2 * ci]
        row_mp = multiphase_metric_rows[2 * ci + 1]
        d_mse = float(row_sv["mse"] - row_mp["mse"])
        print(
            f"{disp}: MSE single={row_sv['mse']:.6e}  multiphase={row_mp['mse']:.6e}  "
            f"ΔMSE(single−multi)={d_mse:.6e}"
        )
    print()
    print(
        "--- CHSH vs three models (see tdf_chsh_cf_metrics.csv): "
        "single_var | multiphase_var | characteristic_function ---"
    )
    cf_strict_best_count = 0
    for ci, disp in enumerate(display_names):
        base = 3 * ci
        r_sv = cf_metric_rows[base]
        r_mp = cf_metric_rows[base + 1]
        r_cf = cf_metric_rows[base + 2]
        mse_sv = float(r_sv["mse"])
        mse_mp = float(r_mp["mse"])
        mse_cf = float(r_cf["mse"])
        best = min(mse_sv, mse_mp, mse_cf)
        if mse_cf < min(mse_sv, mse_mp) - 1e-15:
            cf_strict_best_count += 1
        print(
            f"{disp}: MSE  single_var={mse_sv:.6e}  multiphase_var={mse_mp:.6e}  "
            f"CF={mse_cf:.6e}  (best MSE={best:.6e})"
        )
    max_abs_cf_vs_multi = float(np.max(np.abs(S_pred_cf - S_pred_multi)))
    print()
    print(f"max |S_pred_CF - S_pred_multiphase| = {max_abs_cf_vs_multi:.6e}")
    print()
    if cf_strict_best_count == 3:
        print(
            "Summary: characteristic_function strictly beats **both** variance models on "
            "all three cases (this run): full phase distribution beyond second moment matters for S."
        )
    elif cf_strict_best_count > 0:
        print(
            "Summary: CF strictly beats both variance models on some cases only; "
            "see tdf_chsh_cf_compare.png. Non-Gaussianity or finite-N CF can separate CF from Var."
        )
    elif max_abs_cf_vs_multi < 1e-6:
        print(
            "Summary: S_pred_CF ≈ S_pred_multiphase numerically — for this τ law, "
            "Re(CF) agrees with exp(-Var/2) per component (Gaussian / symmetric regime)."
        )
    else:
        print(
            "Summary: CF differs from multiphase Var but does not uniformly win on MSE vs S; "
            "|·| on the CHSH combination still dominates the residual."
        )

    return {
        "t": t,
        "E": E,
        "E_pred": E_pred,
        "E_pred_multiphase": E_pred_multi,
        "S": S_arr,
        "S_reconstructed": S_recon,
        "S_pred_multiphase": S_pred_multi,
        "S_pred_cf": S_pred_cf,
        "CF": CF,
        "E_pred_cf": E_pred_cf,
        "prediction": pred_S,
        "var_sum": var_sum,
        "var_diff": var_diff,
        "S0": s0,
        "metrics": metrics_df,
        "multiphase_metrics": multiphase_metrics_df,
        "cf_metrics": cf_metrics_df,
        "component_metrics": comp_metrics_df,
        "figure_decay": p1,
        "figure_compare": p2,
        "figure_components": p_components,
        "figure_reconstructed": p_recon,
        "figure_multiphase_compare": p_multiphase,
        "figure_cf_compare": p_cf_compare,
        "components_csv_path": components_path,
        "component_metrics_csv_path": comp_metrics_path,
        "metrics_csv_path": metrics_path,
        "multiphase_metrics_csv_path": multiphase_metrics_path,
        "cf_metrics_csv_path": cf_metrics_path,
        "cf_components_csv_path": cf_components_path,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    run_chsh_decay_experiment()
