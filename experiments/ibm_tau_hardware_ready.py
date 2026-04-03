"""
Hardware-ready τ-modulation experiment (Qiskit + IBM Quantum Runtime).

At each depth ``n_steps`` we always include **baseline** (depth-matched ``RZ(0)`` ⊗ ``RZ(0)``)
and one or more τ evolutions derived from the same ``Δτ_k``:

* **tau_naive** — ``RZ(Δτ)`` on each qubit (same angle).
* **tau_symmetric** — ``RZ(+Δτ)``, ``RZ(−Δτ)`` (cancels relative phase on |Φ+⟩ in the ideal).
* **tau_zz** — ``RZZ(2·Δτ)`` on the pair (ZZ-type interaction).
* **tau_zz_mix** — ``RZZ(2·Δτ)`` — ``RX(ε)`` on each qubit — ``RZZ(2·Δτ)`` per slice (small basis mixing).

Select variants with ``--tau-mode`` (default ``all`` = naive + symmetric + zz + zz_mix + baseline).

After each run, if both **baseline** and **tau_symmetric** are present, the script prints a
**multi-observable coupling** block: differential signals ``dCHSH``, ``dXX``, ``dYY`` along the
shared depth grid, then a length-3 **rolling mean** (``w=3``) on each diff series; the **3×3
correlation matrix** between those smoothed local-dynamics traces; coupling score = mean
|off-diagonal|; verdict if τ score exceeds baseline by >5%. CSV columns ``zz``, ``xx``, ``yy`` are
saved (coupling uses CHSH and Pauli reads ``xx``, ``yy``).

Requires::

    pip install -r requirements-qiskit.txt

IBM Quantum (free tier): put your token in **local** ``apikey.json`` at the repo
root (gitignored). Copy ``apikey.example.json`` → ``apikey.json`` and paste the
token. Optional: ``--apikey-json /path/to/file.json``. Then::

    PYTHONPATH=. python3 experiments/ibm_tau_hardware_ready.py --backend ibm

One-shot Aer + IBM::

    PYTHONPATH=. python3 experiments/ibm_tau_hardware_ready.py --pipeline --n-steps-max 8 --shots 1024 --plot
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Install Qiskit: pip install -r requirements-qiskit.txt"
    ) from e

# ---------------------------------------------------------------------------
# A. τ definition
# ---------------------------------------------------------------------------


def tau(t: np.ndarray | float, omega: float, a: float, nu: float) -> np.ndarray:
    """τ(t) = ω t + a sin(ν t) (vectorized)."""
    t_arr = np.asarray(t, dtype=float)
    return omega * t_arr + a * np.sin(nu * t_arr)


def tau_deltas(tlist: np.ndarray, omega: float, a: float, nu: float) -> np.ndarray:
    """Δτ_k between consecutive times (length len(tlist) - 1)."""
    vals = tau(tlist, omega, a, nu)
    return np.diff(vals)


# ---------------------------------------------------------------------------
# C. Circuit builders
# ---------------------------------------------------------------------------


def build_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def add_identity_rz(qc: QuantumCircuit, n_steps: int) -> None:
    for _ in range(n_steps):
        qc.rz(0.0, 0)
        qc.rz(0.0, 1)


def add_tau_naive(qc: QuantumCircuit, tlist: np.ndarray, omega: float, a: float, nu: float) -> None:
    """Naive: ``RZ(Δτ) ⊗ RZ(Δτ)`` each slice."""
    dtheta = tau_deltas(tlist, omega, a, nu)
    for d in dtheta:
        qc.rz(float(d), 0)
        qc.rz(float(d), 1)


def add_tau_symmetric(qc: QuantumCircuit, tlist: np.ndarray, omega: float, a: float, nu: float) -> None:
    """Symmetric local phases: ``RZ(+Δτ)``, ``RZ(−Δτ)``."""
    dtheta = tau_deltas(tlist, omega, a, nu)
    for d in dtheta:
        qc.rz(float(d), 0)
        qc.rz(float(-d), 1)


def add_tau_zz(qc: QuantumCircuit, tlist: np.ndarray, omega: float, a: float, nu: float) -> None:
    """ZZ-style layer: ``RZZ(2·Δτ)`` on qubits 0–1."""
    dtheta = tau_deltas(tlist, omega, a, nu)
    for d in dtheta:
        qc.rzz(float(2.0 * d), 0, 1)


def add_tau_zz_mix(
    qc: QuantumCircuit,
    tlist: np.ndarray,
    omega: float,
    a: float,
    nu: float,
    eps: float = 0.1,
) -> None:
    """τ ZZ sandwich with small ``RX(ε)`` on each qubit between ZZ layers (basis mixing / interference)."""
    deltas = tau_deltas(tlist, omega, a, nu)
    for dtheta in deltas:
        qc.rzz(float(2.0 * dtheta), 0, 1)
        qc.rx(float(eps), 0)
        qc.rx(float(eps), 1)
        qc.rzz(float(2.0 * dtheta), 0, 1)


# ---------------------------------------------------------------------------
# D. Measurement circuits
# ---------------------------------------------------------------------------


def _measure_zz(qc: QuantumCircuit) -> None:
    qc.measure_all()


def _measure_xx(qc: QuantumCircuit) -> None:
    qc.h(0)
    qc.h(1)
    qc.measure_all()


def _measure_yy(qc: QuantumCircuit) -> None:
    qc.sdg(0)
    qc.h(0)
    qc.sdg(1)
    qc.h(1)
    qc.measure_all()


def _append_chsh_ry(qc: QuantumCircuit, use_a0: bool, use_b0: bool) -> None:
    ta = 0.0 if use_a0 else np.pi / 2.0
    tb = np.pi / 4.0 if use_b0 else -np.pi / 4.0
    qc.ry(ta, 0)
    qc.ry(tb, 1)


def _copy_and_measure(
    base: QuantumCircuit,
    attach: Callable[[QuantumCircuit], None],
) -> QuantumCircuit:
    qc = base.copy()
    attach(qc)
    return qc


def build_measurement_circuits(base: QuantumCircuit) -> dict[str, QuantumCircuit]:
    def mk_chsh(use_a0: bool, use_b0: bool) -> QuantumCircuit:
        qc = base.copy()
        _append_chsh_ry(qc, use_a0, use_b0)
        qc.measure_all()
        return qc

    return {
        "zz": _copy_and_measure(base, _measure_zz),
        "xx": _copy_and_measure(base, _measure_xx),
        "yy": _copy_and_measure(base, _measure_yy),
        "chsh_00": mk_chsh(True, True),
        "chsh_01": mk_chsh(True, False),
        "chsh_10": mk_chsh(False, True),
        "chsh_11": mk_chsh(False, False),
    }


OBS_KEYS = ("zz", "xx", "yy", "chsh_00", "chsh_01", "chsh_10", "chsh_11")

# Stable plot / flatten order (baseline first, then τ variants)
MODEL_ORDER = ("baseline", "tau_naive", "tau_symmetric", "tau_zz", "tau_zz_mix")
TAU_MODELS = ("tau_naive", "tau_symmetric", "tau_zz", "tau_zz_mix")


def resolve_active_tau_modes(tau_mode: str) -> frozenset[str]:
    """
    Which τ implementations to include (always in addition to ``baseline``).

    ``all`` → naive, symmetric, zz, zz_mix; otherwise a single variant name.
    """
    m = tau_mode.strip().lower()
    if m == "all":
        return frozenset({"naive", "symmetric", "zz", "zz_mix"})
    if m in ("naive", "symmetric", "zz", "zz_mix"):
        return frozenset({m})
    raise ValueError(
        f"Unknown --tau-mode {tau_mode!r} (use all, naive, symmetric, zz, zz_mix)"
    )


# ---------------------------------------------------------------------------
# E. Metrics
# ---------------------------------------------------------------------------


def expectation_zz(counts: dict[str, int]) -> float:
    """⟨Z⊗Z⟩ from computational-basis shots (Qiskit bit order: q0 rightmost)."""
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    acc = 0.0
    for bitstring, n in counts.items():
        b0 = int(bitstring[-1])
        b1 = int(bitstring[-2]) if len(bitstring) >= 2 else 0
        z0 = 1.0 - 2.0 * b0
        z1 = 1.0 - 2.0 * b1
        acc += n * z0 * z1
    return acc / total


def _e_ab_chsh(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    s = 0.0
    for bitstring, n in counts.items():
        b0 = int(bitstring[-1])
        b1 = int(bitstring[-2]) if len(bitstring) >= 2 else 0
        a = 1.0 - 2.0 * b0
        b = 1.0 - 2.0 * b1
        s += n * a * b
    return s / total


def chsh_from_counts(
    c00: dict[str, int],
    c01: dict[str, int],
    c10: dict[str, int],
    c11: dict[str, int],
) -> float:
    """S = ⟨A0B0⟩ + ⟨A0B1⟩ + ⟨A1B0⟩ − ⟨A1B1⟩."""
    return (
        _e_ab_chsh(c00)
        + _e_ab_chsh(c01)
        + _e_ab_chsh(c10)
        - _e_ab_chsh(c11)
    )


def fidelity_proxy(zz: float, xx: float, yy: float) -> float:
    """F ≈ (1 + ZZ + XX − YY) / 4 for target |Φ+⟩ (Qiskit Pauli convention)."""
    return float((1.0 + zz + xx - yy) / 4.0)


@dataclass
class RowMetrics:
    chsh: float
    fidelity: float
    zz: float
    xx: float
    yy: float


def metrics_from_count_list(counts_map: dict[str, dict[str, int]]) -> RowMetrics:
    zz = expectation_zz(counts_map["zz"])
    xx = expectation_zz(counts_map["xx"])
    yy = expectation_zz(counts_map["yy"])
    chsh = chsh_from_counts(
        counts_map["chsh_00"],
        counts_map["chsh_01"],
        counts_map["chsh_10"],
        counts_map["chsh_11"],
    )
    return RowMetrics(
        chsh=chsh,
        fidelity=fidelity_proxy(zz, xx, yy),
        zz=zz,
        xx=xx,
        yy=yy,
    )


# ---------------------------------------------------------------------------
# F–H. Backend, transpile, run
# ---------------------------------------------------------------------------


def init_ibm(apikey_path: Path | None = None):
    """
    Initialize IBM Quantum Runtime (uses ``apikey.json`` via ``config.ibm_token``).
    Returns ``QiskitRuntimeService`` or ``None`` on failure.
    """
    try:
        from config.ibm_token import qiskit_runtime_service

        service = qiskit_runtime_service(apikey_path)
        print("IBM Quantum service initialized")
        return service
    except Exception as e:
        print("IBM init failed:", e)
        return None


def backend_from_ibm_service(service, name: str | None = None):
    """Pick a backend from an initialized Runtime service."""
    if name:
        return service.backend(name)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=2)


def is_aer_backend(backend) -> bool:
    return "AerSimulator" in type(backend).__name__


def get_backend(
    *,
    use_ibm: bool,
    name: str | None = None,
    apikey_path: Path | None = None,
):
    """
    Return a backend instance, or ``None`` if IBM path fails (caller may fall back to Aer).
    """
    if use_ibm:
        service = init_ibm(apikey_path)
        if service is None:
            return None
        try:
            return backend_from_ibm_service(service, name)
        except Exception as e:
            print("IBM backend selection failed:", e)
            return None

    from qiskit_aer import AerSimulator

    return AerSimulator()


def transpile_safe(
    circuits: Sequence[QuantumCircuit],
    backend,
) -> list[QuantumCircuit]:
    return list(
        transpile(
            list(circuits),
            backend,
            optimization_level=1,
            layout_method="sabre",
            routing_method="sabre",
        )
    )


def _sampler_pub_to_counts(pub_result) -> dict[str, int]:
    """Extract shot counts from a Sampler V2 primitive result entry."""
    if hasattr(pub_result, "join_data"):
        jd = pub_result.join_data()
        if hasattr(jd, "get_counts"):
            return dict(jd.get_counts())
    data = getattr(pub_result, "data", None)
    if data is not None:
        reg = getattr(data, "c", None)
        if reg is not None and hasattr(reg, "get_counts"):
            return dict(reg.get_counts())
    raise TypeError(f"Cannot read counts from sampler result: {type(pub_result)!r}")


def execute_circuits_and_get_counts(
    circuits: Sequence[QuantumCircuit],
    backend,
    shots: int,
    *,
    seed_simulator: int | None = None,
) -> list[dict[str, int]]:
    """
    Aer: ``backend.run`` + ``Result.get_counts``.
    IBM Quantum: ``SamplerV2`` (``backend.run`` is no longer supported on IBM backends).
    """
    circs = list(circuits)
    if is_aer_backend(backend):
        kwargs: dict = {"shots": int(shots)}
        if seed_simulator is not None:
            kwargs["seed_simulator"] = int(seed_simulator)
        job = backend.run(circs, **kwargs)
        res = job.result()
        return [res.get_counts(i) for i in range(len(circs))]

    from qiskit_ibm_runtime import SamplerV2 as Sampler

    sampler = Sampler(mode=backend)
    job = sampler.run(circs, shots=int(shots))
    prim = job.result()
    out: list[dict[str, int]] = []
    for i in range(len(circs)):
        out.append(_sampler_pub_to_counts(prim[i]))
    return out


# ---------------------------------------------------------------------------
# I–J. Pipeline + sweep
# ---------------------------------------------------------------------------


def build_base_circuits(
    n_steps: int,
    t_max: float,
    omega: float,
    a: float,
    nu: float,
    *,
    active_tau: frozenset[str],
) -> dict[str, QuantumCircuit]:
    tlist = np.linspace(0.0, float(t_max), n_steps + 1)

    out: dict[str, QuantumCircuit] = {}

    qc_b = build_bell()
    add_identity_rz(qc_b, n_steps)
    out["baseline"] = qc_b

    if "naive" in active_tau:
        qc_n = build_bell()
        add_tau_naive(qc_n, tlist, omega, a, nu)
        out["tau_naive"] = qc_n
    if "symmetric" in active_tau:
        qc_s = build_bell()
        add_tau_symmetric(qc_s, tlist, omega, a, nu)
        out["tau_symmetric"] = qc_s
    if "zz" in active_tau:
        qc_z = build_bell()
        add_tau_zz(qc_z, tlist, omega, a, nu)
        out["tau_zz"] = qc_z
    if "zz_mix" in active_tau:
        qc_m = build_bell()
        add_tau_zz_mix(qc_m, tlist, omega, a, nu)
        out["tau_zz_mix"] = qc_m

    return out


def flatten_measurement_jobs(bases: dict[str, QuantumCircuit]) -> tuple[list[QuantumCircuit], list[tuple[str, str]]]:
    """Return all circuits and (model, obs_key) index."""
    circuits: list[QuantumCircuit] = []
    index: list[tuple[str, str]] = []
    for model in MODEL_ORDER:
        if model not in bases:
            continue
        obs_circs = build_measurement_circuits(bases[model])
        for key in OBS_KEYS:
            circuits.append(obs_circs[key])
            index.append((model, key))
    return circuits, index


def run_sweep(
    n_steps_list: Sequence[int],
    *,
    backend,
    shots: int,
    t_max: float,
    omega: float,
    a: float,
    nu: float,
    active_tau: frozenset[str],
    seed: int = 42,
) -> list[dict[str, str | int | float]]:
    """One hardware job per depth; returns flat rows for CSV."""
    rows: list[dict[str, str | int | float]] = []
    use_aer = "AerSimulator" in type(backend).__name__

    for n_steps in n_steps_list:
        bases = build_base_circuits(
            n_steps,
            t_max,
            omega,
            a,
            nu,
            active_tau=active_tau,
        )
        circuits, idx = flatten_measurement_jobs(bases)
        t_circs = transpile_safe(circuits, backend)

        sim_seed = (seed + int(n_steps)) if use_aer else None
        counts_list = execute_circuits_and_get_counts(
            t_circs,
            backend,
            shots,
            seed_simulator=sim_seed,
        )

        # Reassemble counts per model
        per_model: dict[str, dict[str, dict[str, int]]] = {
            m: {k: {} for k in OBS_KEYS} for m in bases
        }
        for i, (model, obs_key) in enumerate(idx):
            per_model[model][obs_key] = counts_list[i]

        for model in MODEL_ORDER:
            if model not in bases:
                continue
            m = metrics_from_count_list(per_model[model])
            rows.append(
                {
                    "n_steps": n_steps,
                    "model": model,
                    "chsh": m.chsh,
                    "fidelity": m.fidelity,
                    "zz": m.zz,
                    "xx": m.xx,
                    "yy": m.yy,
                }
            )
    return rows


@dataclass
class SingleDepthExperimentResult:
    """Baseline + ``tau_symmetric`` at one ``n_steps`` (single transpile + execute batch)."""

    n_steps: int
    chsh_baseline: float
    chsh_tau: float
    fidelity_baseline: float
    fidelity_tau: float


def run_experiment_once(
    n_steps: int,
    *,
    backend,
    shots: int,
    t_max: float = 0.8,
    omega: float = 1.0,
    a: float = 0.3,
    nu: float = 2.5,
    seed: int | None = None,
) -> SingleDepthExperimentResult:
    """
    One experiment: depth-matched baseline (``RZ(0)`` identity layers) and ``tau_symmetric``,
    same measurement / CHSH / fidelity pipeline and ``transpile_safe`` settings as ``run_sweep``.
    """
    active_tau = frozenset({"symmetric"})
    bases = build_base_circuits(
        n_steps,
        t_max,
        omega,
        a,
        nu,
        active_tau=active_tau,
    )
    if "baseline" not in bases or "tau_symmetric" not in bases:
        raise RuntimeError("run_experiment_once requires baseline and tau_symmetric in circuit set")

    circuits, idx = flatten_measurement_jobs(bases)
    t_circs = transpile_safe(circuits, backend)
    use_aer = is_aer_backend(backend)
    sim_seed = int(seed) if seed is not None and use_aer else None
    counts_list = execute_circuits_and_get_counts(
        t_circs,
        backend,
        shots,
        seed_simulator=sim_seed,
    )

    per_model: dict[str, dict[str, dict[str, int]]] = {
        m: {k: {} for k in OBS_KEYS} for m in bases
    }
    for i, (model, obs_key) in enumerate(idx):
        per_model[model][obs_key] = counts_list[i]

    mb = metrics_from_count_list(per_model["baseline"])
    mt = metrics_from_count_list(per_model["tau_symmetric"])
    return SingleDepthExperimentResult(
        n_steps=int(n_steps),
        chsh_baseline=float(mb.chsh),
        chsh_tau=float(mt.chsh),
        fidelity_baseline=float(mb.fidelity),
        fidelity_tau=float(mt.fidelity),
    )


# ---------------------------------------------------------------------------
# K–L. Plot + CSV
# ---------------------------------------------------------------------------


def save_csv(rows: list[dict[str, str | int | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["n_steps", "model", "chsh", "fidelity", "zz", "xx", "yy"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_comparison(
    rows: list[dict[str, str | int | float]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    # Pivot by model
    models = sorted({str(r["model"]) for r in rows})
    steps = sorted({int(r["n_steps"]) for r in rows})

    def series(model: str, key: str) -> list[float]:
        out = []
        for ns in steps:
            hit = [r for r in rows if r["model"] == model and int(r["n_steps"]) == ns]
            out.append(float(hit[0][key]) if hit else float("nan"))
        return out

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=150)

    styles = {
        "baseline": dict(color="0.2", ls="-", marker="o", ms=5, lw=2.0),
        "tau_naive": dict(color="C0", ls="--", marker="s", ms=4),
        "tau_symmetric": dict(color="C1", ls="-.", marker="D", ms=4),
        "tau_zz": dict(color="C2", ls="-", marker="^", ms=4),
        "tau_zz_mix": dict(color="red", ls="-", marker="v", ms=4, lw=2.0),
    }

    plot_order = [m for m in MODEL_ORDER if m in models] + [
        m for m in models if m not in MODEL_ORDER
    ]
    for m in plot_order:
        st = styles.get(m, dict(color="0.5", ls="-", marker="x", ms=4))
        ax0.plot(steps, series(m, "chsh"), label=m, **st)
        ax1.plot(steps, series(m, "fidelity"), label=m, **st)

    ax0.axhline(2.0, color="k", lw=0.8, ls=":", alpha=0.45)
    ax0.axhline(2 * np.sqrt(2), color="k", lw=0.8, ls=":", alpha=0.25)
    ax0.set_xlabel(r"$n_{\mathrm{steps}}$")
    ax0.set_ylabel("CHSH $S$")
    ax0.set_title("CHSH vs depth")
    ax0.grid(True, alpha=0.35)
    ax0.legend(frameon=False, fontsize=8, loc="best")

    ax1.set_xlabel(r"$n_{\mathrm{steps}}$")
    ax1.set_ylabel(r"Fidelity proxy $(1+\langle ZZ\rangle+\langle XX\rangle-\langle YY\rangle)/4$")
    ax1.set_title("Fidelity proxy vs depth")
    ax1.set_ylim(-0.05, 1.08)
    ax1.grid(True, alpha=0.35)
    ax1.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("τ-modulation vs depth-matched baseline (IBM Quantum / Aer)", fontsize=11, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_rows_from_csv(path: Path) -> list[dict[str, str | int | float]]:
    """Reload experiment rows from a saved CSV (for validation / reporting)."""
    with path.open(newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    out: list[dict[str, str | int | float]] = []
    for r in raw:
        row: dict[str, str | int | float] = {
            "n_steps": int(r["n_steps"]),
            "model": r["model"],
            "chsh": float(r["chsh"]),
            "fidelity": float(r["fidelity"]),
        }
        for k in ("zz", "xx", "yy"):
            if k in r and r[k] != "":
                row[k] = float(r[k])
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Multi-observable coupling (differentials + rolling window, cross-observable C)
# ---------------------------------------------------------------------------

_ROLL_W = 3


def rolling_window(x: np.ndarray | Sequence[float], w: int = 3) -> np.ndarray:
    """Rolling mean along the depth index (local smoothing, no cumulative history)."""
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < w:
        return np.array([], dtype=float)
    return np.array([float(np.mean(x[i : i + w])) for i in range(n - w + 1)], dtype=float)


def _sorted_model_rows(rows: list[dict[str, str | int | float]], model: str) -> list[dict[str, str | int | float]]:
    sub = [r for r in rows if str(r["model"]) == model]
    sub.sort(key=lambda r: int(r["n_steps"]))
    return sub


def _rows_have_coupling_observables(rs: Sequence[dict[str, str | int | float]]) -> bool:
    """Coupling uses CHSH + ⟨XX⟩, ⟨YY⟩ along depth (same grid as baseline / τ)."""
    for r in rs:
        for k in ("xx", "yy"):
            if k not in r:
                return False
    return True


_COUPLE_METRIC_LABELS = ("roll(dCHSH)", "roll(dXX)", "roll(dYY)")


def _pairwise_corr_3x3(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Symmetric correlation matrix; pairs with near-zero std → 0 (no linear co-movement)."""
    p, L = int(X.shape[0]), int(X.shape[1])
    R = np.eye(p, dtype=float)
    for i in range(p):
        for j in range(i + 1, p):
            a, b = X[i], X[j]
            sa = float(np.std(a, ddof=1)) if L > 1 else 0.0
            sb = float(np.std(b, ddof=1)) if L > 1 else 0.0
            if sa < eps or sb < eps:
                c_ij = 0.0
            else:
                c_ij = float(np.corrcoef(a, b)[0, 1])
                if not np.isfinite(c_ij):
                    c_ij = 0.0
            R[i, j] = R[j, i] = c_ij
    return R


def _diff_and_roll_coupling_matrix(
    rows_sorted: Sequence[dict[str, str | int | float]],
    *,
    w: int = _ROLL_W,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build metrics [dCHSH, dXX, dYY], apply rolling mean (``w``) to each,
    return (X 3×L, R 3×3, score) where score = mean |off-diagonal(R)|.
    """
    chsh = np.array([float(r["chsh"]) for r in rows_sorted], dtype=float)
    xx = np.array([float(r["xx"]) for r in rows_sorted], dtype=float)
    yy = np.array([float(r["yy"]) for r in rows_sorted], dtype=float)
    d_chsh = np.diff(chsh)
    d_xx = np.diff(xx)
    d_yy = np.diff(yy)
    r_chsh = rolling_window(d_chsh, w)
    r_xx = rolling_window(d_xx, w)
    r_yy = rolling_window(d_yy, w)
    L = min(int(r_chsh.size), int(r_xx.size), int(r_yy.size))
    # Pearson corr of three series needs enough rolling windows; L=3 is rank-deficient / fragile.
    _min_L = 4
    if L < _min_L:
        return (
            np.empty((3, 0)),
            np.full((3, 3), np.nan, dtype=float),
            float("nan"),
        )
    X = np.vstack([r_chsh[:L], r_xx[:L], r_yy[:L]])
    R = _pairwise_corr_3x3(X)
    iu = np.triu_indices(3, k=1)
    return X, R, float(np.mean(np.abs(R[iu])))


def print_multi_observable_coupling(rows: list[dict[str, str | int | float]]) -> None:
    """Baseline vs τ_symmetric: cross-observable correlation of rolled differential traces."""
    present = {str(r["model"]) for r in rows}
    if "baseline" not in present or "tau_symmetric" not in present:
        return

    rb = _sorted_model_rows(rows, "baseline")
    rt = _sorted_model_rows(rows, "tau_symmetric")
    if not _rows_have_coupling_observables(rb) or not _rows_have_coupling_observables(rt):
        print()
        print("--- Multi-observable coupling ---")
        print("  Skipped: CSV rows need xx, yy (re-run experiment to refresh CSV).")
        return
    if len(rb) != len(rt):
        print()
        print("--- Multi-observable coupling ---")
        print("  Skipped: baseline and tau_symmetric depth counts differ.")
        return
    for a, b in zip(rb, rt):
        if int(a["n_steps"]) != int(b["n_steps"]):
            print()
            print("--- Multi-observable coupling ---")
            print("  Skipped: misaligned n_steps between baseline and tau_symmetric.")
            return

    Xb, Rb, cb = _diff_and_roll_coupling_matrix(rb, w=_ROLL_W)
    Xt, Rt, ct = _diff_and_roll_coupling_matrix(rt, w=_ROLL_W)

    print()
    print("--- Multi-observable coupling (local dynamics: diff + rolling mean w=3) ---")
    print("  Same n_steps grid; metrics roll(mean(diff(·))) for CHSH, ⟨XX⟩, ⟨YY⟩.")
    Lb, Lt = int(Xb.shape[1]), int(Xt.shape[1])
    print(f"  Rolled series length: baseline L={Lb}, tau_symmetric L={Lt}")

    print("  Coupling matrix C = corr( roll(dCHSH), roll(dXX), roll(dYY) ) — baseline:")
    print("        " + "  ".join(f"{lab:>14}" for lab in _COUPLE_METRIC_LABELS))
    for lab, row in zip(_COUPLE_METRIC_LABELS, Rb):
        print(f"    {lab:16}" + "".join(f"{v:14.3f}" if np.isfinite(v) else f"{'nan':>14}" for v in row))
    print(f"  Coupling score (mean |off-diagonal|): {cb:.6f}")

    print("  Coupling matrix C — tau_symmetric:")
    print("        " + "  ".join(f"{lab:>14}" for lab in _COUPLE_METRIC_LABELS))
    for lab, row in zip(_COUPLE_METRIC_LABELS, Rt):
        print(f"    {lab:16}" + "".join(f"{v:14.3f}" if np.isfinite(v) else f"{'nan':>14}" for v in row))
    print(f"  Coupling score (mean |off-diagonal|): {ct:.6f}")

    if np.isfinite(cb) and np.isfinite(ct):
        if ct > cb * 1.05:
            print("τ shows structured multi-observable coupling")
        else:
            print("No structured coupling detected")
    else:
        print(
            "  Coupling scores unavailable (need enough depth points: "
            f"len(roll(diff)) ≥ 4 after w={_ROLL_W}, e.g. ≥7 sweep points on this grid)."
        )
        print("No structured coupling detected")


def _pivot_by_depth(rows: list[dict[str, str | int | float]]) -> dict[int, dict[str, dict[str, float]]]:
    by_ns: dict[int, dict[str, dict[str, float]]] = {}
    for r in rows:
        ns = int(r["n_steps"])
        by_ns.setdefault(ns, {})[str(r["model"])] = {
            "chsh": float(r["chsh"]),
            "fidelity": float(r["fidelity"]),
        }
    return by_ns


def best_delta_vs_baseline(
    rows: list[dict[str, str | int | float]],
    tau_model: str,
    *,
    key: str = "chsh",
) -> tuple[float, int]:
    """Max (tau_model − baseline) for ``key`` over depths; returns (best_delta, n_steps)."""
    by_ns = _pivot_by_depth(rows)
    best_d = float("-inf")
    best_step = -1
    for ns, d in by_ns.items():
        if "baseline" not in d or tau_model not in d:
            continue
        delta = d[tau_model][key] - d["baseline"][key]
        if delta > best_d:
            best_d = delta
            best_step = ns
    if best_step < 0:
        return float("nan"), -1
    return best_d, best_step


def print_run_summary(
    rows: list[dict[str, str | int | float]],
    *,
    label: str,
    csv_path: Path | None = None,
) -> None:
    """Print best ΔCHSH (and ΔFidelity) per τ model vs baseline."""
    present = {str(r["model"]) for r in rows}
    print()
    print(f"--- Summary ({label}) ---")
    if csv_path is not None:
        print(f"  CSV: {csv_path}")
    print("Best ΔCHSH vs baseline (max over depths):")
    best_chsh_any = float("-inf")
    best_name = ""
    for name in TAU_MODELS:
        if name not in present:
            continue
        dc, ns_c = best_delta_vs_baseline(rows, name, key="chsh")
        print(f"  {name}: {dc:+.4f}  (at n_steps={ns_c})")
        if not np.isnan(dc) and dc > best_chsh_any:
            best_chsh_any = dc
            best_name = name
    print("Best ΔFidelity vs baseline (max over depths):")
    for name in TAU_MODELS:
        if name not in present:
            continue
        df, ns_f = best_delta_vs_baseline(rows, name, key="fidelity")
        print(f"  {name}: {df:+.4f}  (at n_steps={ns_f})")

    if best_name and not np.isnan(best_chsh_any) and best_chsh_any > 0:
        print(f"τ shows improvement in correlation stability (best: {best_name}, ΔCHSH={best_chsh_any:+.4f})")
    else:
        print("No clear τ advantage observed (vs baseline on CHSH)")


def validate_outputs(csv_path: Path, png_path: Path | None, *, plot: bool) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    rows = load_rows_from_csv(csv_path)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    if plot and png_path is not None and not png_path.is_file():
        raise FileNotFoundError(f"Missing plot: {png_path}")


def execute_sweep_to_disk(
    *,
    backend,
    n_steps_list: Sequence[int],
    shots: int,
    t_max: float,
    omega: float,
    a: float,
    nu: float,
    active_tau: frozenset[str],
    out_dir: Path,
    csv_name: str,
    png_name: str,
    plot: bool,
    seed: int = 42,
) -> list[dict[str, str | int | float]]:
    """Run sweep, save CSV (and optional plot). Returns rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shots_eff = min(int(shots), 1024) if not is_aer_backend(backend) else int(shots)
    rows = run_sweep(
        n_steps_list,
        backend=backend,
        shots=shots_eff,
        t_max=t_max,
        omega=omega,
        a=a,
        nu=nu,
        active_tau=active_tau,
        seed=seed,
    )
    csv_path = out_dir / csv_name
    png_path = out_dir / png_name
    save_csv(rows, csv_path)
    if plot:
        plot_comparison(rows, png_path)
    return rows


# ---------------------------------------------------------------------------
# M–N. CLI + main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="τ hardware-ready sweep: baseline vs tau_naive / tau_symmetric / tau_zz / tau_zz_mix",
    )
    parser.add_argument(
        "--backend",
        choices=("ibm", "aer"),
        default="aer",
        help="IBM Quantum hardware or local Aer (default: aer)",
    )
    parser.add_argument("--ibm-backend", type=str, default=None, help="Override IBM backend name")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit (≤1024 typical for free tier)")
    parser.add_argument(
        "--n-steps-max",
        type=int,
        default=10,
        help="Largest depth (even steps 2..max)",
    )
    parser.add_argument(
        "--tau-mode",
        choices=("all", "naive", "symmetric", "zz", "zz_mix"),
        default="all",
        help=(
            "Which τ evolutions to include besides baseline: "
            "all = tau_naive + tau_symmetric + tau_zz + tau_zz_mix; or a single variant."
        ),
    )
    parser.add_argument(
        "--aer",
        action="store_true",
        help="Force Aer simulator (same as --backend aer)",
    )
    parser.add_argument("--t-max", type=float, default=0.8, help="Time span for τ sampling")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.3)
    parser.add_argument("--nu", type=float, default=2.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <repo>/outputs",
    )
    parser.add_argument(
        "--apikey-json",
        type=Path,
        default=None,
        help="Path to apikey.json (default: <repo>/apikey.json if it exists)",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write comparison PNG (default: true; use --no-plot to skip)",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help=(
            "Run Aer first (sanity), then IBM hardware. Saves ibm_tau_results.csv + plot "
            "from Aer, then ibm_tau_results_ibm.csv (+ _ibm.png) if hardware succeeds."
        ),
    )
    args = parser.parse_args()

    if args.aer:
        args.backend = "aer"

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "outputs"

    n_steps_list = [n for n in range(2, int(args.n_steps_max) + 1, 2)]
    if not n_steps_list:
        raise SystemExit("n-steps-max must be >= 2")

    apikey_path = args.apikey_json
    plot = bool(args.plot)
    active_tau = resolve_active_tau_modes(args.tau_mode)

    if args.pipeline:
        from qiskit_aer import AerSimulator

        # --- Aer sanity ---
        print("=== Pipeline: Step 1 — Aer simulator (sanity) ===")
        aer_backend = AerSimulator()
        print(f"Backend: {aer_backend}")
        print(f"n_steps: {n_steps_list}, shots={args.shots}, tau-mode={args.tau_mode}")
        rows_aer = execute_sweep_to_disk(
            backend=aer_backend,
            n_steps_list=n_steps_list,
            shots=args.shots,
            t_max=args.t_max,
            omega=args.omega,
            a=args.a,
            nu=args.nu,
            active_tau=active_tau,
            out_dir=out_dir,
            csv_name="ibm_tau_results.csv",
            png_name="ibm_tau_hardware_comparison.png",
            plot=plot,
            seed=42,
        )
        csv_aer = out_dir / "ibm_tau_results.csv"
        png_aer = out_dir / "ibm_tau_hardware_comparison.png"
        validate_outputs(csv_aer, png_aer if plot else None, plot=plot)
        print_run_summary(rows_aer, label="Aer", csv_path=csv_aer)
        print_multi_observable_coupling(rows_aer)
        print()
        print("Aer run completed")

        # --- IBM hardware ---
        print()
        print("=== Pipeline: Step 2 — IBM Quantum hardware ===")
        ibm_backend = get_backend(
            use_ibm=True,
            name=args.ibm_backend,
            apikey_path=apikey_path,
        )
        if ibm_backend is None:
            print("IBM unavailable; Aer outputs are complete.")
            print()
            print("τ branch vs baseline comparison complete (Aer only)")
            return

        print(f"Backend: {ibm_backend}")
        ibm_shots = min(int(args.shots), 1024)
        print(f"n_steps: {n_steps_list}, shots={ibm_shots} (capped ≤1024 for IBM), tau-mode={args.tau_mode}")
        try:
            rows_ibm = execute_sweep_to_disk(
                backend=ibm_backend,
                n_steps_list=n_steps_list,
                shots=ibm_shots,
                t_max=args.t_max,
                omega=args.omega,
                a=args.a,
                nu=args.nu,
                active_tau=active_tau,
                out_dir=out_dir,
                csv_name="ibm_tau_results_ibm.csv",
                png_name="ibm_tau_hardware_comparison_ibm.png",
                plot=plot,
                seed=43,
            )
        except Exception as e:
            print("IBM execution failed:", e)
            print("Falling back: keeping Aer-only results under outputs/ibm_tau_results.csv")
            return

        csv_ibm = out_dir / "ibm_tau_results_ibm.csv"
        png_ibm = out_dir / "ibm_tau_hardware_comparison_ibm.png"
        validate_outputs(csv_ibm, png_ibm if plot else None, plot=plot)
        print_run_summary(rows_ibm, label="IBM hardware", csv_path=csv_ibm)
        print_multi_observable_coupling(rows_ibm)

        print()
        print("τ branch vs baseline comparison complete (Aer + IBM)")
        return

    # --- Single backend mode ---
    use_ibm = args.backend == "ibm"
    backend = get_backend(
        use_ibm=use_ibm,
        name=args.ibm_backend,
        apikey_path=apikey_path,
    )
    if backend is None and use_ibm:
        print("Falling back to AerSimulator.")
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        use_ibm = False

    print(f"Backend: {backend}")
    shots_eff = min(int(args.shots), 1024) if not is_aer_backend(backend) else int(args.shots)
    print(f"n_steps: {n_steps_list}, shots={shots_eff}, tau-mode={args.tau_mode}")

    csv_path = out_dir / "ibm_tau_results.csv"
    png_path = out_dir / "ibm_tau_hardware_comparison.png"
    rows = execute_sweep_to_disk(
        backend=backend,
        n_steps_list=n_steps_list,
        shots=shots_eff,
        t_max=args.t_max,
        omega=args.omega,
        a=args.a,
        nu=args.nu,
        active_tau=active_tau,
        out_dir=out_dir,
        csv_name="ibm_tau_results.csv",
        png_name="ibm_tau_hardware_comparison.png",
        plot=plot,
        seed=42,
    )

    validate_outputs(csv_path, png_path if plot else None, plot=plot)
    print_run_summary(rows, label=str(backend), csv_path=csv_path)
    print_multi_observable_coupling(rows)
    if is_aer_backend(backend):
        print()
        print("Aer run completed")

    print()
    print("τ branch vs baseline comparison complete")
    if plot:
        print(f"  Saved plot: {png_path}")


if __name__ == "__main__":
    main()
