"""
Minimal IBM Quantum / Qiskit demo: Bell state with optional τ-like RZ layers.

Compares a **baseline** depth-matched circuit (``RZ(0)`` slices — no coherent τ)
to a **τ-modulated** circuit (incremental ``RZ(Δθ)`` on both qubits from Δτ(t),
matching the Z⊗I + I⊗Z generator picture used elsewhere in this repo).

**Noiseless simulators:** the baseline leaves |Φ+⟩ in the +1 eigenspace of ZZ while
τ layers add relative phase between |00⟩ and |11⟩, so CHSH / XX / YY move by design.
**Noisy Aer or IBM hardware** is the regime where you can ask which trajectory
keeps correlations higher under gate noise and decoherence.

Runs on **Aer** by default. If ``qiskit-ibm-runtime`` is installed and the
environment is configured (``QISKIT_IBM_TOKEN`` or saved account), optionally
runs on a real IBM Quantum backend (free-tier friendly: few shots, small depth sweep).

Install::

    pip install -r requirements-qiskit.txt

Local token file (recommended, gitignored): copy ``apikey.example.json`` to
``apikey.json`` in the repo root, or pass ``--apikey-json PATH``. See
``config/ibm_token.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Qiskit is required. Install with: pip install -r requirements-qiskit.txt"
    ) from e


def _tau_trajectory(
    t: np.ndarray,
    *,
    omega: float,
    a: float,
    nu: float,
) -> np.ndarray:
    return omega * t + a * np.sin(nu * t)


def bell_prep() -> QuantumCircuit:
    """|Φ+⟩ = (|00⟩+|11⟩)/√2 on qubits 0,1."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def append_evolution_slices(
    qc: QuantumCircuit,
    n_steps: int,
    *,
    t_max: float,
    omega: float,
    a: float,
    nu: float,
    tau_modulated: bool,
) -> None:
    """
    Append ``n_steps`` slices on ``[0, t_max]``.

    τ version: ``RZ(Δτ_k)`` on each qubit with Δτ_k = τ(t_{k+1}) - τ(t_k).
    Baseline: ``RZ(0)`` on each qubit (matched depth / gate type for noise models).
    """
    if n_steps <= 0:
        return
    edges = np.linspace(0.0, float(t_max), n_steps + 1)
    tau = _tau_trajectory(edges, omega=omega, a=a, nu=nu)
    for k in range(n_steps):
        dtheta = float(tau[k + 1] - tau[k])
        if tau_modulated:
            qc.rz(dtheta, 0)
            qc.rz(dtheta, 1)
        else:
            # Same two-qubit gate count as τ branch (fair under gate-wise noise models).
            qc.rz(0.0, 0)
            qc.rz(0.0, 1)


def _measure_zz(qc: QuantumCircuit) -> None:
    qc.measure_all()


def _measure_xx(qc: QuantumCircuit) -> None:
    qc.h(0)
    qc.h(1)
    qc.measure_all()


def _measure_yy(qc: QuantumCircuit) -> None:
    # Y basis: S† then H maps Y → Z
    qc.sdg(0)
    qc.h(0)
    qc.sdg(1)
    qc.h(1)
    qc.measure_all()


def _append_chsh_rotations(qc: QuantumCircuit, use_a0: bool, use_b0: bool) -> None:
    """
    CHSH measurement after |Φ+⟩ prep: local ``Ry`` then Z⊗Z readout.

    Settings (maximal violation): A0→``Ry(0)``, A1→``Ry(π/2)`` on q0;
    B0→``Ry(π/4)``, B1→``Ry(−π/4)`` on q1. Same geometry as common Qiskit CHSH demos.
    """
    ta = 0.0 if use_a0 else np.pi / 2.0
    tb = np.pi / 4.0 if use_b0 else -np.pi / 4.0
    qc.ry(ta, 0)
    qc.ry(tb, 1)


def copy_and_measure(
    base: QuantumCircuit,
    attach_measure: Callable[[QuantumCircuit], None],
) -> QuantumCircuit:
    qc = base.copy()
    attach_measure(qc)
    return qc


@dataclass
class ObservablesResult:
    zz: float
    xx: float
    yy: float
    chsh: float
    fidelity_proxy: float


def _counts_to_zz_expectation(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    acc = 0.0
    for bitstring, n in counts.items():
        # Qiskit: rightmost bit is q0
        b0 = int(bitstring[-1])
        b1 = int(bitstring[-2]) if len(bitstring) >= 2 else 0
        z0 = 1.0 - 2.0 * b0
        z1 = 1.0 - 2.0 * b1
        acc += n * z0 * z1
    return acc / total


def _counts_to_chsh(counts00: dict[str, int], counts01: dict[str, int], counts10: dict[str, int], counts11: dict[str, int]) -> float:
    """
    CHSH = ⟨A0B0⟩ + ⟨A0B1⟩ + ⟨A1B0⟩ − ⟨A1B1⟩ with ±1 outcomes from Z⊗Z counts
    after local rotations.
    """

    def e_ab(cts: dict[str, int]) -> float:
        tot = sum(cts.values())
        if tot == 0:
            return float("nan")
        s = 0.0
        for bitstring, n in cts.items():
            b0 = int(bitstring[-1])
            b1 = int(bitstring[-2]) if len(bitstring) >= 2 else 0
            a = 1.0 - 2.0 * b0
            b = 1.0 - 2.0 * b1
            s += n * a * b
        return s / tot

    return e_ab(counts00) + e_ab(counts01) + e_ab(counts10) - e_ab(counts11)


def build_observable_circuits(base: QuantumCircuit) -> dict[str, QuantumCircuit]:
    """Return labeled circuits for ZZ, XX, YY, and four CHSH settings."""

    def mk_chsh(use_a0: bool, use_b0: bool) -> QuantumCircuit:
        qc = base.copy()
        _append_chsh_rotations(qc, use_a0, use_b0)
        qc.measure_all()
        return qc

    return {
        "zz": copy_and_measure(base, _measure_zz),
        "xx": copy_and_measure(base, _measure_xx),
        "yy": copy_and_measure(base, _measure_yy),
        "chsh_00": mk_chsh(True, True),
        "chsh_01": mk_chsh(True, False),
        "chsh_10": mk_chsh(False, True),
        "chsh_11": mk_chsh(False, False),
    }


def observables_from_counts(counts_map: dict[str, dict[str, int]]) -> ObservablesResult:
    zz = _counts_to_zz_expectation(counts_map["zz"])
    xx = _counts_to_zz_expectation(counts_map["xx"])
    yy = _counts_to_zz_expectation(counts_map["yy"])
    chsh = _counts_to_chsh(
        counts_map["chsh_00"],
        counts_map["chsh_01"],
        counts_map["chsh_10"],
        counts_map["chsh_11"],
    )
    # |Φ+⟩⟨Φ+| = (I + XX − YY + ZZ) / 4 in the Pauli convention used by Qiskit (⟨YY⟩ = −1).
    fidelity_proxy = float((1.0 + zz + xx - yy) / 4.0)
    return ObservablesResult(
        zz=zz,
        xx=xx,
        yy=yy,
        chsh=chsh,
        fidelity_proxy=fidelity_proxy,
    )


def run_circuits_aer(
    circuits: Sequence[QuantumCircuit],
    *,
    shots: int = 4096,
    seed: int = 123,
    noise: bool = False,
) -> list[dict[str, int]]:
    from qiskit_aer import AerSimulator

    noise_model = None
    if noise:
        try:
            from qiskit_aer.noise import NoiseModel, depolarizing_error

            noise_model = NoiseModel()
            e1 = depolarizing_error(0.002, 1)
            e2 = depolarizing_error(0.01, 2)
            noise_model.add_all_qubit_quantum_error(e1, ["id", "rz", "h", "ry"])
            noise_model.add_all_qubit_quantum_error(e2, ["cx"])
        except ImportError:
            noise_model = None

    if noise_model is not None:
        backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    else:
        backend = AerSimulator(seed_simulator=seed)

    t_circs = transpile(list(circuits), backend, optimization_level=0)
    job = backend.run(t_circs, shots=shots)
    result = job.result()
    return [result.get_counts(i) for i in range(len(circuits))]


def _sampler_pub_to_counts(pub_result) -> dict[str, int]:
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


def run_circuits_ibm(
    circuits: Sequence[QuantumCircuit],
    *,
    shots: int = 1024,
    backend_name: str | None = None,
    apikey_path: Path | None = None,
) -> list[dict[str, int]]:
    """
    Run on IBM Quantum via ``SamplerV2`` (``backend.run`` is not supported on IBM backends).
    Uses ``apikey.json`` when present. See ``config.ibm_token``.
    """
    try:
        from config.ibm_token import qiskit_runtime_service
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    except ImportError as e:
        raise ImportError(
            "Install qiskit-ibm-runtime and configure your IBM Quantum account "
            "(see module docstring)."
        ) from e

    service = qiskit_runtime_service(apikey_path)
    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)

    t_circs = transpile(list(circuits), backend, optimization_level=1)
    try:
        sampler = Sampler(mode=backend)
        job = sampler.run(t_circs, shots=int(shots))
        prim = job.result()
    except Exception as e:
        raise RuntimeError(
            "IBM Sampler run failed. Check account, queue, and circuit limits. "
            f"Original error: {e}"
        ) from e
    return [_sampler_pub_to_counts(prim[i]) for i in range(len(t_circs))]


def run_ibm_tau_demo(
    *,
    depths: Sequence[int] | None = None,
    t_max: float = 0.6,
    omega: float = 1.0,
    a: float = 0.25,
    nu: float = 2.5,
    shots: int = 4096,
    backend: str = "aer",
    ibm_backend: str | None = None,
    ibm_apikey_path: Path | None = None,
    noisy_aer: bool = False,
    seed: int = 123,
) -> dict[str, Any]:
    """
    For each depth in ``depths``, run baseline vs τ-modulated observable circuits.

    Returns structured metrics and optional ``comparison`` summary strings.
    """
    if depths is None:
        depths = (0, 2, 4, 6, 8)

    rows_baseline: list[ObservablesResult] = []
    rows_tau: list[ObservablesResult] = []

    for n_steps in depths:
        base_b = bell_prep()
        append_evolution_slices(
            base_b,
            n_steps,
            t_max=t_max,
            omega=omega,
            a=a,
            nu=nu,
            tau_modulated=False,
        )
        obs_b = build_observable_circuits(base_b)
        keys = ["zz", "xx", "yy", "chsh_00", "chsh_01", "chsh_10", "chsh_11"]
        circs_b = [obs_b[k] for k in keys]

        base_t = bell_prep()
        append_evolution_slices(
            base_t,
            n_steps,
            t_max=t_max,
            omega=omega,
            a=a,
            nu=nu,
            tau_modulated=True,
        )
        obs_t = build_observable_circuits(base_t)
        circs_t = [obs_t[k] for k in keys]

        if backend == "aer":
            cnt_b = run_circuits_aer(circs_b, shots=shots, seed=seed + n_steps, noise=noisy_aer)
            cnt_t = run_circuits_aer(circs_t, shots=shots, seed=seed + 1000 + n_steps, noise=noisy_aer)
        else:
            cnt_b = run_circuits_ibm(
                circs_b,
                shots=min(shots, 1024),
                backend_name=ibm_backend,
                apikey_path=ibm_apikey_path,
            )
            cnt_t = run_circuits_ibm(
                circs_t,
                shots=min(shots, 1024),
                backend_name=ibm_backend,
                apikey_path=ibm_apikey_path,
            )

        cm_b = {k: cnt_b[i] for i, k in enumerate(keys)}
        cm_t = {k: cnt_t[i] for i, k in enumerate(keys)}
        rows_baseline.append(observables_from_counts(cm_b))
        rows_tau.append(observables_from_counts(cm_t))

    return {
        "depths": list(depths),
        "baseline": rows_baseline,
        "tau": rows_tau,
        "params": {
            "t_max": t_max,
            "omega": omega,
            "a": a,
            "nu": nu,
            "shots": shots,
            "backend": backend,
        },
    }


def _print_table(result: dict[str, Any]) -> None:
    depths: list[int] = result["depths"]
    b = result["baseline"]
    t = result["tau"]
    print(f"backend={result['params']['backend']} shots={result['params']['shots']}")
    print(
        f"{'n':>3} | {'CHSH_b':>8} {'CHSH_t':>8} | {'F_b':>8} {'F_t':>8} | {'ZZ_b':>7} {'ZZ_t':>7}"
    )
    print("-" * 60)
    for i, n in enumerate(depths):
        print(
            f"{n:3d} | {b[i].chsh:8.4f} {t[i].chsh:8.4f} | "
            f"{b[i].fidelity_proxy:8.4f} {t[i].fidelity_proxy:8.4f} | "
            f"{b[i].zz:7.4f} {t[i].zz:7.4f}"
        )


def _maybe_plot(result: dict[str, Any], output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    depths = np.asarray(result["depths"], dtype=float)
    ch_b = [r.chsh for r in result["baseline"]]
    ch_t = [r.chsh for r in result["tau"]]
    f_b = [r.fidelity_proxy for r in result["baseline"]]
    f_t = [r.fidelity_proxy for r in result["tau"]]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.plot(depths, ch_b, "o-", label="baseline", color="C0")
    ax0.plot(depths, ch_t, "s-", label="τ RZ", color="C1")
    ax0.axhline(2.0, color="k", ls="--", lw=0.8, alpha=0.4, label="classical bound 2")
    ax0.set_xlabel("evolution slices")
    ax0.set_ylabel("CHSH")
    ax0.set_title("CHSH vs circuit depth")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1.plot(depths, f_b, "o-", label="baseline", color="C0")
    ax1.plot(depths, f_t, "s-", label="τ RZ", color="C1")
    ax1.set_xlabel("evolution slices")
    ax1.set_ylabel(r"F ≈ (1+⟨ZZ⟩+⟨XX⟩−⟨YY⟩)/4")
    ax1.set_title("Fidelity proxy vs depth")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "ibm_tau_demo_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="τ-modulated Bell demo (Qiskit)")
    parser.add_argument(
        "--backend",
        choices=("aer", "ibm"),
        default="aer",
        help="aer (local) or ibm (requires runtime + account)",
    )
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--noisy-aer", action="store_true", help="depolarizing noise on Aer")
    parser.add_argument("--ibm-backend", type=str, default=None, help="IBM backend name")
    parser.add_argument(
        "--apikey-json",
        type=Path,
        default=None,
        help="IBM token JSON (default: <repo>/apikey.json)",
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    shots = args.shots
    if args.backend == "ibm":
        shots = min(shots, 1024)

    out = run_ibm_tau_demo(
        backend=args.backend,
        shots=shots,
        noisy_aer=args.noisy_aer,
        ibm_backend=args.ibm_backend,
        ibm_apikey_path=args.apikey_json,
    )
    _print_table(out)
    print(
        "\nNote: coherent τ slices implement exp(−i Δτ Z)⊗exp(−i Δτ Z) each step, so they "
        "rotate |Φ+⟩ out of the maximal-CHSH plane unless Δτ ≡ 0. Baseline RZ(0) leaves "
        "the Bell state in that plane (noiseless). Compare slopes on IBM hardware or "
        "use --noisy-aer for a toy gate-noise study."
    )

    if args.plot:
        p = _maybe_plot(out, Path(__file__).resolve().parent.parent / "outputs")
        if p:
            print("Saved", p)


if __name__ == "__main__":
    main()
