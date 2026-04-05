# From τ(t) to Hidden 5D Geometry: A Reproducible Pipeline

This note is the **big-picture** story of the TDF–QM (`tdf_qutip`) repository: what was built, why, and how far the evidence goes. For commands and file paths, see [`PIPELINE_STEPS.md`](PIPELINE_STEPS.md) and [`REPRO.md`](REPRO.md).

---

## 1. Motivation

In many open-system treatments, noise is summarized by **Markovian** (Lindblad) maps or by **unstructured** stochastic drives. Those choices are practical, but they fold all temporal structure into effective rates or a generic spectrum.

This project explores a contrasting picture: a **single phase field** τ(t) such that amplitudes acquire phase through τ, and the **energy** fed into the Hamiltonian follows from how fast τ advances. If τ is smooth, oscillatory, or temporally correlated, that structure can appear **together** in several observables—correlations, CHSH-style combinations, and spectral fingerprints—not as independent knobs.

**Interpretation vs observation:** We do **not** claim a unique fundamental theory. We build a **minimal phenomenological pipeline** (simulate → fit → manifold → modes → spectrum → geometry tests) and report where the data look **structured** versus **degenerate** or **mimickable** by simpler noise.

---

## 2. Core idea (minimal model)

- **Phase / amplitude:** dynamics are still standard quantum evolution; τ(t) enters through a derived **coupling energy** (implementation: `E(t) = ℏ dτ/dt` on a fixed generator `G`).
- **Hamiltonian:** `H(t) = E(t) G` on the chosen system (here, a two-qubit Bell setup with the same `G` as the standard benchmark).
- **Consequence:** one τ trajectory **jointly** drives `C_xx`, `C_yy`, CHSH correlators, etc., so cross-observable structure is a natural diagnostic—not an add-on.

---

## 3. What was built

- **τ models:** linear, structured, stochastic, and **correlated** (OU) τ(t) (`core/tau_model.py`).
- **Two-qubit experiment:** standard `ωG` vs TDF τ-drive; saves correlation and CHSH traces (`experiments/correlation_test.py`).
- **Inverse problem:** fit parametric τ to match `C_xx`, `C_yy`, CHSH (`analysis/tau_extraction.py`).
- **Non-uniqueness:** multi-start fits + spread metrics (`analysis/tau_identifiability.py`).
- **Degeneracy manifold:** low-loss ensemble, PCA in parameter and τ space (`analysis/tau_manifold.py`).
- **Kernel modes:** PCA on τ residuals, observable sensitivity along modes (`analysis/tau_kernel_modes.py`).
- **Mode dynamics:** phenomenological ODE fits (relaxation vs damped oscillator) (`analysis/tau_mode_field_fit.py`).
- **Hidden spectrum:** map modes to effective λ_eff; compact-tower-style checks (`analysis/tau_hidden_spectrum.py`).
- **χ-geometry hypotheses:** flat / offset / warped compact-style fits (`analysis/tau_chi_geometry.py`).
- **Discrimination:** TDF vs OU and pink noise on the same `G`, joint cross-observable coupling score (`experiments/tdf_vs_colored_noise.py`).
- **Unified-law comparison (TDF vs OU):** same sweeps, but a **multi-relation** score that rewards consistent polynomial structure across observables (`analysis/unified_law_metrics.py`, `experiments/tdf_vs_ou_unified_law.py`). This is a **different scalar** from the coupling score above.
- **Robust statistics:** multi-seed replication, bootstrap CIs, permutation nulls, and window/grid sensitivity (`analysis/unified_law_stats.py`, `experiments/tdf_vs_ou_unified_law_robust.py`). The robust script applies a **strict** rule before claiming TDF is “robustly better” (positive mean Δscore with CI above zero, high win rate, small median permutation p-value, stability across settings).
- **Qiskit τ on Aer / IBM:** depth-swept Bell circuits with τ-modulated layers vs a depth-matched baseline; CHSH, fidelity proxy, Pauli reads; optional cross-observable coupling diagnostic; separate script for **repeated τ_symmetric vs baseline** jobs with per-depth mean ΔCHSH, CI, and win rate (`experiments/ibm_tau_hardware_ready.py`, `experiments/ibm_tau_symmetric_stats.py`). This branch is **orthogonal** to the QuTiP Lindblad pipeline (see [`REPRO.md`](REPRO.md)).
- **Ensemble TDF phase laws (QuTiP):** Wiener-driven **τ₁, τ₂** (and three-qubit **τ₃**) with independent / shared / partially correlated legs; tests **Gaussian variance** laws and extensions on **coherence**, **open system + Lindblad dephasing**, **concurrence**, **GHZ** off-diagonal, and **CHSH** (including **multiphase variance** per correlator and **characteristic-function** `Re ⟨e^{iφ}⟩` fits). See [`PIPELINE_STEPS.md`](PIPELINE_STEPS.md) §13 and [`REPRO.md`](REPRO.md).

---

## 4. Key findings (current default pipeline)

* **τ is not unique** from correlations alone: multi-start fits land on a **low-loss manifold** (effective degeneracy).
* **Kernel modes** on that manifold are **low-dimensional and structured**—not i.i.d. jitter.
* **Mode dynamics:** typically **mixed**—e.g. a dominant mode better fit as **relaxational**, subleading modes as **oscillatory** (depends on seed and grid).
* **Effective spectrum:** a **small** soft-scale mode plus **larger** oscillatory ω²-like scales in the table produced by the hidden-spectrum step.
* **χ-geometry:** with only **two** oscillatory tower points, the **three-parameter “warped” fit is disabled**; **offset-compact** (λ₀ + A n²) often wins over **flat** A n² on the demo settings.
* **Colored-noise discrimination:** the **joint coupling score** does **not** consistently single out TDF over OU/pink on the default sweep—**not decisive** as implemented.
* **Unified-law metric:** on the default robust battery, TDF can score **higher** than OU with **seed-level** and **resampling** support; claims still depend on sweep design, physics choices, and the printed robustness checklist (see [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md), [`REPRO.md`](REPRO.md)).
* **Ensemble phase laws:** for local `σ_z` τ-noise on standard states, **scalar** coherence often tracks `exp(-Var/2)`; **multi-qubit** coherences may need **Var(∑τ)** (not only `Var(Δτ)`). **CHSH S** is a **nonlinear** functional of `ρ̄`, so a single variance scaling is a poor global fit; **empirical characteristic functions** of linear τ-combinations can match **component** expectations and reconstructed **S** much more closely than variance-only models when `ω≠0` (mean phase) or distributions depart from Gaussian-second-moment equivalence.

---

## 5. What it means (short)

The pipeline **moves the narrative** from “we added structured noise” toward “we summarize data with an **effective spectrum** and **geometry-inspired** hypotheses,” while keeping **honest limits**: many fits are **degenerate**, and **standard colored drives** on the same generator can still look competitive on scalar summaries.

**Status:** promising as a **structured phenomenology**; **not** a uniqueness proof against Markovian or generic noise without stronger experimental design or metrics.

---

## Further reading

| Doc | Role |
|-----|------|
| [`PIPELINE_STEPS.md`](PIPELINE_STEPS.md) | Ordered steps, modules, outputs |
| [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) | Figures + captions + takeaways |
| [`REPRO.md`](REPRO.md) | Exact reproduction commands |
| [`FIGURES_INDEX.md`](FIGURES_INDEX.md) | List of PNGs in `outputs/` |
