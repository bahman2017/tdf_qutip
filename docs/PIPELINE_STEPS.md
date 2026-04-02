# Pipeline steps (module map)

Each step lists **purpose**, **code**, **data flow**, and **typical figures** under [`outputs/`](../outputs/). Paths are relative to the `tdf_qutip` repository root.

---

### 1. Generate τ and simulate observables

| | |
|---|---|
| **Purpose** | Baseline two-qubit Bell evolution: standard `ωG` vs TDF `E_τ(t) G`; produces `C_xx`, `C_yy`, `C_zz`, CHSH. |
| **Module** | [`experiments/correlation_test.py`](../experiments/correlation_test.py) — `run_two_qubit_correlation_experiment` |
| **Input → output** | Time grid `t` → trajectories + optional PNGs. |
| **Figures** | `correlation_xx.png`, `correlation_yy.png`, `correlation_zz.png`, `chsh_tdf_vs_standard.png` |

---

### 2. Fit τ from observables

| | |
|---|---|
| **Purpose** | Single-start optimization: find `(ω, a, ν, σ, τ_c)` so simulated correlations match TDF data traces. |
| **Module** | [`analysis/tau_extraction.py`](../analysis/tau_extraction.py) — `fit_tau_from_correlations` |
| **Input → output** | `t`, `C_xx`, `C_yy`, `CHSH` data + bounds → best params, fitted curves, τ(t) plot. |
| **Figures** | `tau_extraction_Cxx_fit.png`, `tau_extraction_Cyy_fit.png`, `tau_extraction_CHSH_fit.png`, `tau_extraction_tau.png` |

---

### 3. Multi-start identifiability

| | |
|---|---|
| **Purpose** | Many random restarts; quantify whether τ is **clustered** or **spread** (non-unique inverse). |
| **Module** | [`analysis/tau_identifiability.py`](../analysis/tau_identifiability.py) — `multi_start_tau_fit` |
| **Input → output** | Same correlation data + bounds → `all_tau`, losses, pairwise RMSE matrix, interpretation string. |
| **Figures** | `tau_identifiability_tau_overlay.png`, `tau_identifiability_param_histograms.png` |

---

### 4. Degeneracy manifold (PCA)

| | |
|---|---|
| **Purpose** | Restrict to **low-loss** fits; PCA in parameter space and in τ trajectories; 2D embedding. |
| **Module** | [`analysis/tau_manifold.py`](../analysis/tau_manifold.py) — `analyze_degeneracy_manifold` |
| **Input → output** | Multi-start `summary`, `t`, margin `ε` → `tau_good`, PCA metrics, interpretation. |
| **Figures** | `tau_manifold_param_space.png`, `tau_manifold_tau_overlay.png`, `tau_manifold_embedding.png` |

---

### 5. Kernel modes + observable sensitivity

| | |
|---|---|
| **Purpose** | PCA on τ residuals on the manifold; sweep τ_mean + λφ_k; RMSE on observables. |
| **Module** | [`analysis/tau_kernel_modes.py`](../analysis/tau_kernel_modes.py) — `run_tau_kernel_mode_analysis` |
| **Input → output** | Manifold dict + multi-start summary + `t` → modes, explained variance, sensitivity matrix. |
| **Figures** | `tau_kernel_modes.png`, `tau_kernel_variance.png`, `tau_kernel_sensitivity.png` |

---

### 6. Mode dynamics (ODE fits)

| | |
|---|---|
| **Purpose** | Fit each leading mode to **relaxation** vs **damped oscillator**; pick best RMSE per mode. |
| **Module** | [`analysis/tau_mode_field_fit.py`](../analysis/tau_mode_field_fit.py) — `fit_kernel_modes_to_field_equations` |
| **Input → output** | Kernel analysis dict → per-mode fits + summary counts. |
| **Figures** | `tau_mode_fit_mode1.png`, `tau_mode_fit_mode2.png`, …, `tau_mode_fit_summary.png` |

---

### 7. Hidden spectrum λ_eff

| | |
|---|---|
| **Purpose** | Map best-fit dynamics to scalars λ_eff (ω² or μ); compact n² reference; CSV table. |
| **Module** | [`analysis/tau_hidden_spectrum.py`](../analysis/tau_hidden_spectrum.py) — `extract_hidden_spectrum` |
| **Input → output** | Field-fit result → records, pairwise ratios, compact fit, interpretation. |
| **Figures / data** | `hidden_spectrum_bar.png`, `hidden_spectrum_compact_fit.png`, `tau_hidden_spectrum.csv` |

---

### 8. χ-geometry reconstruction

| | |
|---|---|
| **Purpose** | Compare oscillatory tower to **flat**, **offset**, and **warped** (if ≥3 points) phenomenological spectra. |
| **Module** | [`analysis/tau_chi_geometry.py`](../analysis/tau_chi_geometry.py) — `analyze_chi_geometry` |
| **Input → output** | Hidden-spectrum dict → model RMSEs, best model, soft-mode bar chart, CSV. |
| **Figures / data** | `chi_geometry_spectrum_models.png`, `chi_geometry_soft_mode.png`, `tau_chi_geometry_models.csv` |

---

### 9. TDF vs colored noise

| | |
|---|---|
| **Purpose** | Sweep **τ_c** (TDF), **correlation_time** (OU on `ω+ξ`), **noise_strength** (pink); joint coupling score. |
| **Module** | [`experiments/tdf_vs_colored_noise.py`](../experiments/tdf_vs_colored_noise.py) — `run_tdf_vs_colored_noise_discrimination` |
| **Input → output** | Time grid + sweep arrays → DataFrame, coupling matrices, scores, interpretation line. |
| **Figures / data** | `tdf_vs_noise_metrics_*.png`, `tdf_vs_noise_coupling_*.png`, `tdf_vs_colored_noise_joint_scores.png`, CSVs |

---

### 10. TDF vs OU unified laws

| | |
|---|---|
| **Purpose** | Same generator sweep as the discrimination experiment, but score **consistency of polynomial relations** across observables (unified-law metric in [`analysis/unified_law_metrics.py`](../analysis/unified_law_metrics.py)). |
| **Module** | [`experiments/tdf_vs_ou_unified_law.py`](../experiments/tdf_vs_ou_unified_law.py) — `run_tdf_vs_ou_unified_law` |
| **Input → output** | Lindblad-style sweeps for TDF (`τ_c`) and OU (correlation time) → sweep tables, per-relation fits, scalar **unified_score**. |
| **Figures / data** | `tdf_vs_ou_unified_law_relations_*.png`, `tdf_vs_ou_unified_law_stability_*.png`, `tdf_vs_ou_unified_law_scores.png`, `tdf_vs_ou_unified_law_sweep_tdf.csv`, `tdf_vs_ou_unified_law_sweep_ou.csv` |

---

### 11. Unified-law robustness (statistics)

| | |
|---|---|
| **Purpose** | Make the TDF vs OU unified-law comparison **defensible across seeds**: bootstrap CIs on the score, permutation p-values (shuffled relation targets), sensitivity to `n_windows` and sweep length, optional train/test RMSE. |
| **Modules** | [`experiments/tdf_vs_ou_unified_law_robust.py`](../experiments/tdf_vs_ou_unified_law_robust.py); helpers in [`analysis/unified_law_stats.py`](../analysis/unified_law_stats.py) |
| **Input → output** | Repeated `collect_unified_law_sweeps` with different seeds / grids → CSV summaries + diagnostic plots. |
| **Figures / data** | `tdf_vs_ou_robust_delta_hist.png`, `tdf_vs_ou_robust_bootstrap_violin.png`, `tdf_vs_ou_robust_permutation.png`, `tdf_vs_ou_robust_window_sensitivity.png`; `tdf_vs_ou_unified_law_seed_scores.csv`, `tdf_vs_ou_unified_law_bootstrap.csv`, `tdf_vs_ou_unified_law_permutation.csv`, `tdf_vs_ou_unified_law_robust_summary.csv` |

---

### Orchestration

Shared step runners live in [`scripts/pipeline_demo.py`](../scripts/pipeline_demo.py) (used by `python -m …` entry points and optional `--all` run). See [`REPRO.md`](REPRO.md).
