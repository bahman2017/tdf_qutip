# τ-model discrimination summary (v2)

Phenomenological comparison only—not a proof of TDF. V2 adds **`structured_stochastic_tau`**: τ(t)=ωt+sin(νt)+ξ(t) with fixed **seed** for single-trajectory spectrum/interference, and a **dedicated** ensemble-ρ decoherence benchmark vs best-fit Lindblad for that same functional form (noise strength σ swept as in the baseline experiment). Other τ rows still show the **pooled** `stochastic_tau` vs Lindblad means for columns (C).

## Combined metrics

| τ model | f_dom | S_spec | BW_90 | |Δ|_mean | r_overlap | C_final | RMSE_L (mean σ) | |Δ|_tail (mean) | osc_res (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear_tau | 0.299 | 0.572744 | 0.0996667 | 1.55801e-16 | nan | 0.455535 | 0.223827 | 0.252976 | 0.157011 |
| oscillatory_tau | 0.299 | 1.42915 | 0.498333 | 3.19874e-05 | 1 | 0.455535 | 0.223827 | 0.252976 | 0.157011 |
| structured_tau | 0.797333 | 2.07965 | 1.09633 | 0.00110521 | 1 | 0.455535 | 0.223827 | 0.252976 | 0.157011 |
| multi_scale_tau | 0.299 | 1.49533 | 0.598 | 0.00032481 | 0.999995 | 0.455535 | 0.223827 | 0.252976 | 0.157011 |
| structured_stochastic_tau | 0.199333 | 3.13646 | 4.485 | 0.166084 | 0.714129 | 0.455544 | 0.223834 | 0.252986 | 0.157013 |
| correlated_stochastic_tau | 0.199333 | 2.38619 | 1.09633 | 0.0136366 | 0.9957 | 0.455535 | 0.223827 | 0.252976 | 0.157011 |

- **Reference τ for interference:** `linear_tau` vs each τ_B. **Ensemble size:** N=40. **`structured_stochastic_tau` row:** decoherence columns use the structured+noise ensemble; other rows use mean-over-σ from the baseline `stochastic_tau` run (see `decoherence_note` in CSV).

## Rankings (descriptive only)

- **Richest spectrum (largest S_spec):** `structured_stochastic_tau` (≈ 3.13646 nats).
- **Strongest interference (lowest MAE, τ_B ≠ `linear_tau`):** `oscillatory_tau` (MAE ≈ 3.19874e-05).
- **Highest overlap correlation (non-reference τ_B):** `oscillatory_tau` (r ≈ 1).
- **Baseline `stochastic_tau` — worst σ by full RMSE:** σ ≈ 0.5 (RMSE ≈ 0.434688).
- **`structured_stochastic_tau` ensemble — worst σ by full RMSE:** σ ≈ 0.5 (RMSE ≈ 0.434713).

## `structured_stochastic_tau` — all three tests

- **(A) Spectrum:** rank **1/6** by spectral entropy (higher is richer structure in this FFT diagnostic).
- **(B) Interference:** rank **5/5** among non-reference τ_B by mean |cos(Δτ) − overlap| (1 = best agreement).
- **(C) Decoherence:** mean full-curve RMSE vs best-fit Lindblad ≈ 0.223834 (structured+noise ensemble); baseline `stochastic_tau` pool mean ≈ 0.223827. Lower RMSE means the Markovian dephasing surrogate tracks the ensemble curve more closely on average—**not** that the physics is Lindbladian.

**Cautious read:** High spectral entropy does not imply a “best” τ field. Interference uses a **linear** reference; large |Δτ| from the stochastic part can **hurt** (B) even when (A) is rich. Here: (A) rank **1/6**, (B) rank **5/5** among non-reference τ_B. (C) RMSE often tracks the baseline when both ensembles share the same σ grid—compare protocols, not only scalars. Not proof of TDF.

## Does one combined τ “win” all three?

The same ansatz may lead in **(A)** but not in **(B)** (as for `structured_stochastic_tau` vs `linear_tau` here). Treat the three blocks as **different probes**: FFT structure on one trajectory, cos(Δτ) vs overlap for a chosen τ pair, and ensemble ρ vs a fitted Markovian channel. Uniform excellence is neither required nor sufficient for a useful discrimination test.
