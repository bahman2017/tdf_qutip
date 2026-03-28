# τ-model discrimination summary

This note **does not** establish time-domain field (TDF) mechanics or rule out equivalent descriptions. It is a **discrimination-style** checklist: how much structure each *ad hoc* τ(t) template exhibits in (A) Ramsey-channel spectrum, (B) interference agreement between cos(Δτ) and quantum overlap with a fixed reference τ, and (C) a **separate** stochastic-τ ensemble benchmark versus a best-fit Markovian dephasing model. Section C is **not** tied to individual deterministic τ families—values repeat across rows by construction.

## Combined metrics

| τ model | f_dom | S_spec | BW_90 | |Δ|_mean | r_overlap | C_final (stoch.) | RMSE_L (mean σ) | |Δ|_tail (mean) | osc_res (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear_tau | 0.299 | 0.572744 | 0.0996667 | 1.55801e-16 | nan | 0.455535 | 0.223827 | 0.252976 | 0.157011 |
| oscillatory_tau | 0.299 | 1.42915 | 0.498333 | 3.19874e-05 | 1 | 0.455535 | 0.223827 | 0.252976 | 0.157011 |
| structured_tau | 0.797333 | 2.07965 | 1.09633 | 0.00110521 | 1 | 0.455535 | 0.223827 | 0.252976 | 0.157011 |
| multi_scale_tau | 0.299 | 1.49533 | 0.598 | 0.00032481 | 0.999995 | 0.455535 | 0.223827 | 0.252976 | 0.157011 |

- **Reference τ for interference:** `linear_tau` vs each row's τ_B. **Stochastic decoherence:** N_ensemble=40, metrics averaged over the σ grid in `decoherence_comparison.csv`.

## Rankings (descriptive only)

- **Richest Ramsey ⟨σ_x⟩ spectrum (largest spectral entropy):** `structured_tau` (S_spec ≈ 2.07965 nats).
- **Strongest interference agreement (lowest mean |cos(Δτ) − overlap|, excluding `linear_tau` where Δτ≡0):** `oscillatory_tau` (MAE ≈ 3.19874e-05). The `linear_tau` row is a numerical self-check only.
- **Highest overlap correlation (among non-reference τ_B):** `oscillatory_tau` (r ≈ 1). When MAE and r disagree slightly, treat both as noisy phenomenological guides.
- **Largest Lindblad mismatch (among σ in the stochastic benchmark):** σ ≈ 0.5 with full-curve RMSE ≈ 0.434688 (same underlying curve for all deterministic τ rows; see `outputs/decoherence_comparison.csv` for per-σ detail).

## One τ family for “everything”?

Spectrum and interference **do** respond to the deterministic τ shape. The Lindblad comparison, however, probes **stochastic** τ—a different protocol. Therefore there is **no single row-wise score** that fairly “explains all three” without mixing assumptions.

A **purely heuristic** combined ordering (entropy + correlation − scaled MAE; decoherence not included) would place models as: `structured_tau`, `multi_scale_tau`, `oscillatory_tau`, `linear_tau`. This is **not** a statistical test and does not favor TDF over other parameterizations of effective Hamiltonians or noise.

**Takeaway:** Use this file to **compare ad hoc τ templates** under fixed numerical settings. Claims beyond phenomenological discrimination require independent constraints (data, identifiable parameters, and alternative noise models).
