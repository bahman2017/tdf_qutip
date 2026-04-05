# Results summary (figures + takeaways)

Figures live under [`outputs/`](../outputs/). If a file is missing locally, run the pipeline ([`REPRO.md`](REPRO.md)).

**Legend:** **Observed** = what the plot/CSV shows. **Interpretation** = how we read it in this project (not a uniqueness proof).

---

## Degeneracy (multi-start τ)

![Identifiability overlay](../outputs/tau_identifiability_tau_overlay.png)

* **`tau_identifiability_tau_overlay.png`**  
  **Observed:** Many fitted τ(t) curves from different restarts lie close in loss but differ in shape.  
  **Interpretation:** The inverse map from observables to τ is **non-unique**; fits explore a **manifold** of near-equivalent τ.

---

## Kernel modes (PCA on manifold)

![Kernel modes](../outputs/tau_kernel_modes.png)

* **`tau_kernel_modes.png`**  
  **Observed:** Mean τ plus a few smooth orthogonal modes φ_k(t) with decaying explained variance.  
  **Interpretation:** Degeneracy is **low-dimensional structure**, not unstructured jitter.

---

## Mode dynamics (ODE phenomenology)

![Mode 1 fit](../outputs/tau_mode_fit_mode1.png)

* **`tau_mode_fit_mode1.png`**  
  **Observed:** Leading PCA mode often fits better as **relaxation** (or damped) depending on run.  
  **Interpretation:** Dominant manifold direction can look **soft** / slowly varying.

![Mode 2 fit](../outputs/tau_mode_fit_mode2.png)

* **`tau_mode_fit_mode2.png`**  
  **Observed:** Higher modes often look **oscillatory** under damped-oscillator fit.  
  **Interpretation:** Subleading directions carry **faster** effective dynamics.

---

## Hidden spectrum (λ_eff)

![Spectrum bar](../outputs/hidden_spectrum_bar.png)

* **`hidden_spectrum_bar.png`**  
  **Observed:** Small |λ| on relaxation-labeled mode; larger ω²-like values on damped-labeled modes.  
  **Interpretation:** Effective picture: **soft scale** + **discrete higher** scales (phenomenological).

---

## χ-geometry model comparison

![χ geometry models](../outputs/chi_geometry_spectrum_models.png)

* **`chi_geometry_spectrum_models.png`**  
  **Observed:** With two oscillatory points, **offset-compact** λ₀ + A n² often beats **flat** A n²; **warped** fit is skipped until ≥3 points.  
  **Interpretation:** Data are **compatible with a gap/offset** in a toy compact tower; not a proof of extra dimensions.

---

## TDF vs colored noise (coupling score)

![Joint scores](../outputs/tdf_vs_colored_noise_joint_scores.png)

* **`tdf_vs_colored_noise_joint_scores.png`**  
  **Observed:** Mean |off-diagonal| Pearson correlations across sweep points for three models.  
  **Interpretation:** On default settings, **OU/pink can match or exceed TDF**; the test is **not decisive** as a discriminator.

---

## TDF vs OU (unified-law score)

Single-run figures from `experiments/tdf_vs_ou_unified_law.py` (e.g. **`tdf_vs_ou_unified_law_scores.png`**, relation and stability panels) summarize **one** seed and one window setting. The **unified_score** aggregates how well low-order polynomial relations across observables agree (RMSE, coefficient variability); it is **not** the same as the joint coupling scalar in the previous section.

---

## TDF vs OU (robust statistics)

After `experiments/tdf_vs_ou_unified_law_robust.py`:

* **`tdf_vs_ou_robust_delta_hist.png`** — Spread of Δscore = score_TDF − score_OU over seeds.  
* **`tdf_vs_ou_robust_bootstrap_violin.png`** — Bootstrap distributions of the unified score (illustrative seed).  
* **`tdf_vs_ou_robust_permutation.png`** — Null from shuffling relation targets vs observed score.  
* **`tdf_vs_ou_robust_window_sensitivity.png`** — Whether the ordering TDF vs OU persists when `n_windows` and sweep grid length change.

**Observed:** CSVs under `outputs/` (`tdf_vs_ou_unified_law_seed_scores.csv`, `…_bootstrap.csv`, `…_permutation.csv`, `tdf_vs_ou_unified_law_robust_summary.csv`) store the numbers behind the printed summary.

**Interpretation:** Treat the console **“statistically robustly better”** line as applying only when **all** built-in checks pass (mean Δscore > 0, 95% CI for mean Δ excludes 0, win rate ≥ 0.7, median permutation p for TDF < 0.05, TDF ahead across window and grid sensitivity tables). Otherwise the comparison is **inconclusive** or **setting-dependent** under this battery.

---

## IBM / Aer τ embeddings (Qiskit)

**Sweep** (`experiments/ibm_tau_hardware_ready.py`):

* **`ibm_tau_hardware_comparison.png`** (and `…_ibm.png` from `--pipeline`) — CHSH $S$ and fidelity proxy vs circuit depth for baseline and selected τ models.  
* **`ibm_tau_results.csv`** — Per row: `n_steps`, `model`, `chsh`, `fidelity`, `zz`, `xx`, `yy`.

**Observed:** Aer traces are near ideal; IBM hardware shows lower CHSH and noisier Pauli moments. **Interpretation:** Use as a **hardware probe** of which τ embedding survives noise; not the same object as the QuTiP Lindblad pipeline above.

**Coupling block (console):** When the run includes both **baseline** and **tau_symmetric**, the script prints a **3×3** correlation matrix of **rolling-mean (w=3)** of **finite-difference** sequences for CHSH, ⟨XX⟩, ⟨YY⟩ across depth, a scalar coupling score, and a line comparing τ vs baseline (see script docstring).

**Repeated symmetric stats** (`experiments/ibm_tau_symmetric_stats.py`):

* **`ibm_tau_symmetric_stats.png`** — Mean **ΔCHSH** (τ_symmetric − baseline) vs depth with **95% CI** error bars (normal approximation).  
* **`ibm_tau_symmetric_stats.csv`** — One row per successful run; **`ibm_tau_symmetric_stats_by_depth.csv`** — Aggregated **mean ΔCHSH**, **CI**, **win_rate** per depth.

**Observed:** Numbers depend on backend, device, and shot noise. **Interpretation:** The printed **“Statistically consistent τ advantage”** line applies only when **every** depth with $n\ge2$ runs passes strict built-in gates (positive mean, CI lower bound $>0$, win rate $\ge 0.7$).

Commands: [`REPRO.md`](REPRO.md) § IBM Quantum / Aer.

---

## TDF ensemble phase laws (QuTiP)

Run all modules in one block: [`REPRO.md`](REPRO.md) § TDF ensemble phase laws.

* **`tdf_phase_decoherence_V.png` / `tdf_phase_decoherence_compare.png`** — Empirical `V(t) = |⟨e^{iΔτ}⟩|` vs `exp(-Var(Δτ)/2)` for cases A/B/C; residuals panel. **Interpretation:** checks the **Gaussian phase** ansatz on the toy scalar order parameter.

* **`tdf_open_system_validation_coherence.png`** — Ensemble-averaged single-qubit coherence vs the same prediction. **`tdf_lindblad_compare.png`** — Total coherence with **Markovian dephasing** vs factorized **τ × environment** law (`e^{-Var/2} C_L`).

* **`tdf_entanglement_decay.png` / `tdf_entanglement_compare.png`** — **Concurrence** of `ρ̄` for Bell + local τ noise. **Interpretation:** `Var(Δτ)` alone can miss **common-mode** decay; `exp(-Var(τ₁+τ₂)/2)` aligns better with `\|00⟩`–`\|11⟩` structure (see metrics CSV).

* **`tdf_ghz_decay.png` / `tdf_ghz_compare.png`** — GHZ `\|ρ_{000,111}\|` (normalized) vs `exp(-Var(τ₁+τ₂+τ₃)/2)`.

* **`tdf_chsh_cf_compare.png`** — Actual **CHSH S** vs **single-variance**, **multiphase-variance**, and **characteristic-function** reconstructed S. **`tdf_chsh_components.png`** — Per-correlator data vs single-variance component fits. **Interpretation:** variance-only scalings are weak for **S**; **`tdf_chsh_cf_metrics.csv`** compares `model_type` ∈ {`single_var`, `multiphase_var`, `characteristic_function`}; CF uses full empirical `⟨e^{iφ_{ij}}⟩` per τ combination (see experiment docstring).

**Observed:** Defaults are seed-dependent; CHSH script prints a short verdict comparing the three S models.

---

## Takeaways

* **Observed:** Correlation data do not pin down a unique τ; a structured **low-loss manifold** appears in multi-start fits.  
* **Observed:** PCA modes and ODE fits show **mixed** relaxational and oscillatory character.  
* **Interpretation:** That mix motivates an **effective spectrum** (soft + tower) rather than a single noise rate.  
* **Observed:** Offset-style χ fits can beat flat n² on **sparse** tower data.  
* **Observed:** The colored-noise baseline on the **same** generator remains **competitive** on the joint-coupling scalar—**stronger metrics or data** would be needed for a sharp claim.
* **Observed:** The **unified-law** comparison uses a **different** scalar; the **robust** script adds seeds, bootstrap CIs, permutations, and sensitivity—read the printed checklist and CSVs before generalizing.
* **Observed:** **Qiskit** τ sweeps on Aer vs IBM separate **circuit-level** embeddings from the QuTiP analysis chain; use **`ibm_tau_symmetric_stats_by_depth.csv`** for aggregated ΔCHSH / CI / win rate after repeated runs.
* **Observed:** **Ensemble τ** scripts validate **phase-variance** laws and **Lindblad × τ** factorization on minimal QuTiP models; **CHSH** highlights **nonlinearity** of S in `ρ̄` and the value of **characteristic-function** (full distribution) modeling over variance-only fits for correlator-based reconstruction.

---

See also [`FIGURES_INDEX.md`](FIGURES_INDEX.md) for every PNG under `outputs/`.
