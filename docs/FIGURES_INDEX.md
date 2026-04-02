# Figures index

PNG files under [`outputs/`](../outputs/) (regenerate locally; some may be gitignored).

To refresh this list after a pipeline run:

```bash
cd tdf_qutip
PYTHONPATH=. python3 scripts/generate_docs_index.py
```

---

* [`chsh_tdf_vs_standard.png`](../outputs/chsh_tdf_vs_standard.png) — CHSH-style S(t): standard vs TDF.
* [`correlated_tau_predictivity_trends.png`](../outputs/correlated_tau_predictivity_trends.png) — Figure: correlated tau predictivity trends.
* [`correlation_xx.png`](../outputs/correlation_xx.png) — Two-qubit ⟨XX⟩: standard vs TDF τ-drive.
* [`correlation_yy.png`](../outputs/correlation_yy.png) — Two-qubit ⟨YY⟩: standard vs TDF.
* [`correlation_zz.png`](../outputs/correlation_zz.png) — Two-qubit ⟨ZZ⟩: standard vs TDF.
* [`decoherence_compare_sigma_0p1.png`](../outputs/decoherence_compare_sigma_0p1.png) — Figure: decoherence compare sigma 0p1.
* [`decoherence_compare_sigma_0p5.png`](../outputs/decoherence_compare_sigma_0p5.png) — Figure: decoherence compare sigma 0p5.
* [`decoherence_compare_sigma_1p0.png`](../outputs/decoherence_compare_sigma_1p0.png) — Figure: decoherence compare sigma 1p0.
* [`decoherence_compare_sigma_2p0.png`](../outputs/decoherence_compare_sigma_2p0.png) — Figure: decoherence compare sigma 2p0.
* [`decoherence_comparison.png`](../outputs/decoherence_comparison.png) — Figure: decoherence comparison.
* [`decoherence_comparison_summary.png`](../outputs/decoherence_comparison_summary.png) — Figure: decoherence comparison summary.
* [`hidden_spectrum_bar.png`](../outputs/hidden_spectrum_bar.png) — Effective λ_eff per mode (damped vs relaxation).
* [`hidden_spectrum_compact_fit.png`](../outputs/hidden_spectrum_compact_fit.png) — Oscillatory λ vs n with A n² compact reference.
* [`interference_sweep_mean_error_vs_amplitude.png`](../outputs/interference_sweep_mean_error_vs_amplitude.png) — Figure: interference sweep mean error vs amplitude.
* [`interference_sweep_mean_error_vs_freq.png`](../outputs/interference_sweep_mean_error_vs_freq.png) — Figure: interference sweep mean error vs freq.
* [`tau_decomposition_observables.png`](../outputs/tau_decomposition_observables.png) — Figure: tau decomposition observables.
* [`tau_decomposition_tau.png`](../outputs/tau_decomposition_tau.png) — Figure: tau decomposition tau.
* [`tau_extraction_CHSH_fit.png`](../outputs/tau_extraction_CHSH_fit.png) — Fitted τ-model vs data (CHSH).
* [`tau_extraction_Cxx_fit.png`](../outputs/tau_extraction_Cxx_fit.png) — Fitted τ-model vs data (C_xx).
* [`tau_extraction_Cyy_fit.png`](../outputs/tau_extraction_Cyy_fit.png) — Fitted τ-model vs data (C_yy).
* [`tau_extraction_tau.png`](../outputs/tau_extraction_tau.png) — Extracted τ(t) from correlation fit.
* [`tau_identifiability_param_histograms.png`](../outputs/tau_identifiability_param_histograms.png) — Parameter histograms across restarts.
* [`tau_identifiability_tau_overlay.png`](../outputs/tau_identifiability_tau_overlay.png) — Multi-start τ curves (identifiability).
* [`tau_manifold_embedding.png`](../outputs/tau_manifold_embedding.png) — 2D PCA embedding of τ trajectories.
* [`tau_manifold_param_space.png`](../outputs/tau_manifold_param_space.png) — Low-loss runs in parameter PCA space.
* [`tau_manifold_tau_overlay.png`](../outputs/tau_manifold_tau_overlay.png) — τ(t) ensemble on degeneracy manifold.

---

### TDF vs OU unified law and robust statistics

Produced by [`experiments/tdf_vs_ou_unified_law.py`](../experiments/tdf_vs_ou_unified_law.py) and [`experiments/tdf_vs_ou_unified_law_robust.py`](../experiments/tdf_vs_ou_unified_law_robust.py) (see [`REPRO.md`](REPRO.md)). After a run, refresh the full PNG list with `PYTHONPATH=. python3 scripts/generate_docs_index.py`.

**Single-run unified law**

* [`tdf_vs_ou_unified_law_relations_tdf.png`](../outputs/tdf_vs_ou_unified_law_relations_tdf.png) — Polynomial relations across sweep points (TDF).
* [`tdf_vs_ou_unified_law_relations_ou.png`](../outputs/tdf_vs_ou_unified_law_relations_ou.png) — Same for OU (correlation-time sweep).
* [`tdf_vs_ou_unified_law_stability_tdf.png`](../outputs/tdf_vs_ou_unified_law_stability_tdf.png) — Fit coefficient stability vs window count (TDF).
* [`tdf_vs_ou_unified_law_stability_ou.png`](../outputs/tdf_vs_ou_unified_law_stability_ou.png) — Same (OU).
* [`tdf_vs_ou_unified_law_scores.png`](../outputs/tdf_vs_ou_unified_law_scores.png) — Scalar unified-law scores: TDF vs OU.

**Robust battery (multi-seed, bootstrap, permutation)**

* [`tdf_vs_ou_robust_delta_hist.png`](../outputs/tdf_vs_ou_robust_delta_hist.png) — Distribution of Δscore = score_TDF − score_OU over seeds.
* [`tdf_vs_ou_robust_bootstrap_violin.png`](../outputs/tdf_vs_ou_robust_bootstrap_violin.png) — Bootstrap score distributions and 95% CIs (illustrative seed).
* [`tdf_vs_ou_robust_permutation.png`](../outputs/tdf_vs_ou_robust_permutation.png) — Permutation nulls vs observed unified score.
* [`tdf_vs_ou_robust_window_sensitivity.png`](../outputs/tdf_vs_ou_robust_window_sensitivity.png) — Scores vs `n_windows` and sweep grid choice.

**Tables (CSV, under [`outputs/`](../outputs/))**

* `tdf_vs_ou_unified_law_sweep_tdf.csv`, `tdf_vs_ou_unified_law_sweep_ou.csv` — sweep rows used by the analyzer.
* `tdf_vs_ou_unified_law_seed_scores.csv` — per seed / model: unified score, mean RMSE, mean coef CV, train/test RMSE.
* `tdf_vs_ou_unified_law_bootstrap.csv`, `tdf_vs_ou_unified_law_permutation.csv` — resampling details per seed.
* `tdf_vs_ou_unified_law_robust_summary.csv` — headline statistics and sensitivity tables.
