# TDF–QM numerical summary report

*Generated 2026-03-28 22:42 UTC from `tau_model_summary.csv` and `interference_sweep.csv`.*

## Overview: working assumptions

This codebase explores a **time-dependent field (TDF) ansatz** linking a scalar phase field τ(t) to qubit dynamics, **not** a full axiomatic replacement of quantum mechanics.

- **Phase field:** amplitudes carry phases suggestive of **ψ ∝ exp(iτ)** in a motivational sense; dynamics are implemented via QuTiP with **H(t)** derived from τ.
- **Energy relation (model):** **E = ℏ dτ/dt**, used to build **H ∝ E(t) σ_z** in the minimal coupling used here.
- **Classical interference diagnostic:** **cos(Δτ)** with Δτ = τ_A − τ_B is compared to **|⟨ψ_A|ψ_B⟩|** after separate evolutions—useful as a **consistency check**, not as a theorem.

## τ-model spectral comparison (Ramsey channel, ⟨σ_x⟩ FFT)

Spectral entropy *S* (nats) and 90% power bandwidth summarize how much structure appears in the positive-frequency half of the ⟨σ_x⟩ trajectory under each τ-driven Hamiltonian.

| model | *S* (nats) | BW₉₀ | *f*_dom |
|---|---:|---:|---:|
| `linear_tau` | 0.5727 | 0.09967 | 0.299 |
| `oscillatory_tau` | 1.429 | 0.4983 | 0.299 |
| `structured_tau` | 2.08 | 1.096 | 0.7973 |
| `multi_scale_tau` | 1.495 | 0.598 | 0.299 |

**Richest spectrum (by spectral entropy):** `structured_tau` (*S* ≈ 2.08, BW₉₀ ≈ 1.096). Larger *S* indicates a broader spread of |FFT|² across frequency bins on this grid.

## Interference sweep robustness

Two sweeps: structured **ν** with linear τ_A; fixed ν = 3 with oscillatory **amplitude** on τ_A. Metrics: mean |cos(Δτ) − overlap|, detrended dominant frequencies, and Pearson *r* between the two traces.

- **Mean absolute error** across sweep rows: roughly **0.0002**–**0.4832** (depends strongly on oscillatory amplitude at large *A*).
- **Overlap correlation:** Overlap correlation is high on average (**0.912**) but falls to **0.2938** in at least one regime (see large oscillatory amplitude), so agreement is not uniform.
- **Raw vs detrended dominant frequencies:** Raw dominant frequencies show slightly smaller mean mismatch than detrended on this dataset; detrending still clarifies oscillatory content when raw peaks sit at DC.

## Highlights

- **Most structured Ramsey spectrum:** `structured_tau` under the chosen parameters.
- **Overlap correlation:** remains **high** in the structured-frequency sweep and for small oscillatory amplitudes; **degrades** when *A* is large (nonlinear τ_A), as expected if cos(Δτ) ceases to track overlap.
- **Raw vs detrended peak agreement:** mean |*f*_direct − *f*_quantum| over sweep rows is **0.03306** Hz (raw) vs **0.07713** Hz (detrended) here—raw often ties both channels to **DC**, yielding spurious agreement; detrending shifts emphasis to oscillatory bins and can **separate** peaks when DC is misleading (see large-amplitude row where raw vs detrended peaks differ).

## Interpretation

Within this **minimal single-qubit, σ_z-coupled** implementation, τ-fields that add independent modulation (structured and multi-scale) produce **richer FFT structure** in ⟨σ_x⟩ than a pure linear drift. The **cos(Δτ) vs overlap** comparison suggests the TDF construction can **track** the quantum overlap closely when τ enters through the **same Hamiltonian map** used for evolution, but agreement is **parameter-dependent** and should be re-checked when changing Hilbert space, coupling, or open-system noise.

## Limitations

- **Phenomenological** τ models and **ad hoc** map τ → H; no uniqueness or completeness claim.
- **Single qubit**, closed system, fixed discretization and time window—FFT summaries depend on grid choice.
- **cos(Δτ)** is a **scalar diagnostic**, not derived from a full two-path interferometer model.
- CSV reflects **one** parameter set per row; statistical uncertainty and repeated trials are not included.

## Next experiments

- Extend to **open-system** channels and compare cos(Δτ) diagnostics to **Lindblad** trajectories.
- Scan **ℏ**, coupling axis (e.g. σ_x component), and **initial states**; quantify where overlap correlation breaks down.
- Add **hypothesis tests** or confidence intervals over multiple noise realizations (e.g. stochastic τ).
- Export **figures** into this report (e.g. spectrum overlays) for archival reproducibility.
