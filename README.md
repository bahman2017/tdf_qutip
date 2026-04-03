# tdf_qutip

QuTiP-based research code for simulating quantum systems with **custom noise models** and TDF τ-field structure.

## Layout

| Path | Role |
|------|------|
| `core/` | Hamiltonians, τ-model, noise channels, time evolution |
| `experiments/` | Ramsey, Δτ interference, decoherence, TDF vs OU **unified-law** + **robust** stats; **Qiskit** τ sweep on Aer / IBM (`ibm_tau_hardware_ready.py`, `ibm_tau_symmetric_stats.py`) |
| `analysis/` | Metrics (coherence, correlation, Q(τ)), **unified_law_metrics** / **unified_law_stats** |
| `config/` | Shared simulation parameters |
| `notebooks/` | Exploratory workflows |
| `docs/` | **Project narrative & reproducible pipeline** (start with [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)) |
| `scripts/` | `pipeline_demo.py` (repro steps), `generate_docs_index.py` |

## Setup

```bash
cd tdf_qutip
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Optional (IBM Quantum / Aer τ experiments):** `pip install -r requirements-qiskit.txt`, then see [`docs/REPRO.md`](docs/REPRO.md) § IBM τ hardware-ready.

## Documentation (pipeline narrative)

- **[`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)** — motivation, what was built, findings, limits  
- **[`docs/PIPELINE_STEPS.md`](docs/PIPELINE_STEPS.md)** — step → module → outputs  
- **[`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md)** — key figures + captions  
- **[`docs/REPRO.md`](docs/REPRO.md)** — exact commands  
- **[`docs/FIGURES_INDEX.md`](docs/FIGURES_INDEX.md)** — all PNGs under `outputs/` (run `python3 scripts/generate_docs_index.py` to refresh)

## Status

Active experiments: two-qubit τ extraction, manifold / kernel / spectrum / χ-geometry analysis, TDF vs colored-noise discrimination, **TDF vs OU** on a **multi-relation unified-law** score with optional **multi-seed bootstrap / permutation** robustness, and **hardware-style** τ embeddings on **Aer / IBM Quantum** (CHSH + fidelity proxy, optional multi-observable coupling summary, repeated **τ_symmetric** statistics) ([`docs/REPRO.md`](docs/REPRO.md), [`docs/PIPELINE_STEPS.md`](docs/PIPELINE_STEPS.md) §12).

## License

Add your license here.
