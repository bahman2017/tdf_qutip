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

## Takeaways

* **Observed:** Correlation data do not pin down a unique τ; a structured **low-loss manifold** appears in multi-start fits.  
* **Observed:** PCA modes and ODE fits show **mixed** relaxational and oscillatory character.  
* **Interpretation:** That mix motivates an **effective spectrum** (soft + tower) rather than a single noise rate.  
* **Observed:** Offset-style χ fits can beat flat n² on **sparse** tower data.  
* **Observed:** The colored-noise baseline on the **same** generator remains **competitive** on the joint-coupling scalar—**stronger metrics or data** would be needed for a sharp claim.

---

See also [`FIGURES_INDEX.md`](FIGURES_INDEX.md) for every PNG under `outputs/`.
