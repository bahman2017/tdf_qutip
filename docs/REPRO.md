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
| [`FIGURES_INDEX.md`](FIGURES_INDEX.md) | PNG list |
