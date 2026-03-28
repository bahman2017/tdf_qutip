# tdf_qutip

QuTiP-based research code for simulating quantum systems with **custom noise models** and TDF τ-field structure.

## Layout

| Path | Role |
|------|------|
| `core/` | Hamiltonians, τ-model, noise channels, time evolution |
| `experiments/` | Ramsey, Δτ interference, decoherence comparisons |
| `analysis/` | Metrics (coherence, correlation, Q(τ)) and plotting helpers |
| `config/` | Shared simulation parameters |
| `notebooks/` | Exploratory workflows |

## Setup

```bash
cd tdf_qutip
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Status

Stubs only: implement physics and numerics in each module as the project matures.

## License

Add your license here.
