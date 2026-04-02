# Reproduction guide

All commands assume a **terminal**, **QuTiP + SciPy + NumPy + matplotlib + pandas + scikit-learn** installed (see `requirements.txt`), and repository root:

```bash
cd /path/to/tdf_qutip
export PYTHONPATH=.
export MPLBACKEND=Agg    # optional: headless plotting
```

Use `python3` instead of `python` if that is what your system provides.

---

## One-shot full pipeline

Runs the τ chain through χ-geometry **and** the colored-noise discrimination (slowest option):

```bash
python3 scripts/pipeline_demo.py --all
```

**Rough runtime:** on a typical laptop, **tens of minutes** (many optimizations + QuTiP evolutions). The discrimination step alone is on the order of **minutes** (7×7×7 sweeps × Lindblad scans).

**Outputs:** figures and CSVs under [`outputs/`](../outputs/) (may be gitignored).

`--all` does **not** run the TDF vs OU **unified-law** experiments; invoke those scripts separately (below).

---

## Step-by-step (`python -m` entry points)

Each command **re-runs upstream computations** needed for that stage (see [`scripts/pipeline_demo.py`](../scripts/pipeline_demo.py)).

Exact sequence (use `python` or `python3` per your install):

```bash
export PYTHONPATH=.
python -m experiments.correlation_test
python -m analysis.tau_extraction
python -m analysis.tau_identifiability
python -m analysis.tau_manifold
python -m analysis.tau_kernel_modes
python -m analysis.tau_mode_field_fit
python -m analysis.tau_hidden_spectrum
python -m analysis.tau_chi_geometry
python -m experiments.tdf_vs_colored_noise
```

### TDF vs OU (unified-law score, optional)

Single-run comparison on the **multi-relation unified-law** metric (not the joint coupling score):

```bash
python3 experiments/tdf_vs_ou_unified_law.py
```

**Statistical robustness** (multi-seed, bootstrap on sweep rows, permutation nulls, window/grid sensitivity, train/test RMSE). Default settings are heavy (many sweeps × resamples); use `--fast` for a shorter smoke test:

```bash
python3 experiments/tdf_vs_ou_unified_law_robust.py --fast
python3 experiments/tdf_vs_ou_unified_law_robust.py   # full defaults: e.g. 20 seeds, larger B
```

**Outputs:** figures `tdf_vs_ou_unified_law_*.png`, `tdf_vs_ou_robust_*.png`, and CSVs `tdf_vs_ou_unified_law_*.csv` under [`outputs/`](../outputs/). The script prints mean Δscore, 95% CI, win rate, permutation p-values, and a strict **“robustly better”** verdict when all criteria pass.

Single named step without re-running the whole chain from your own session:

```bash
python3 scripts/pipeline_demo.py --step correlation
python3 scripts/pipeline_demo.py --step tau_extraction
# … identifiability | manifold | kernel_modes | mode_field_fit |
#   hidden_spectrum | chi_geometry | tdf_vs_colored_noise
```

---

## Refresh the figures index

After generating PNGs:

```bash
python3 scripts/generate_docs_index.py
```

Updates [`FIGURES_INDEX.md`](FIGURES_INDEX.md).

---

## Troubleshooting

| Issue | Suggestion |
|--------|------------|
| `ModuleNotFoundError: scripts` | Run from `tdf_qutip` root with `PYTHONPATH=.` |
| QuTiP solver warnings | Reduce time steps or shorten `t` in `pipeline_demo.py` |
| Empty `outputs/` | Run at least `python3 -m experiments.correlation_test` |

---

## Documentation map

| File | Content |
|------|---------|
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | Narrative |
| [`PIPELINE_STEPS.md`](PIPELINE_STEPS.md) | Step table |
| [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) | Plots + takeaways |
| [`FIGURES_INDEX.md`](FIGURES_INDEX.md) | PNG list (includes unified-law / robust figures when generated) |
