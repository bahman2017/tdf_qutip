"""
Ramsey-type experiment simulation.

Compares standard constant-Hamiltonian qubit evolution to τ-driven TDF evolution
on the same initial state and time grid.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from core.evolution import create_superposition_state, run_evolution
from core.hamiltonians import constant_hamiltonian, tau_to_hamiltonian
from core.tau_model import linear_tau, structured_tau  # noqa: F401


def run_tdf_with_tau(t: np.ndarray, tau: np.ndarray):
    """
    Evolve the default superposition under H(t) built from τ(t).

    Returns ``sesolve`` result with ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ expectations.
    """
    H_tau = tau_to_hamiltonian(tau, t)
    psi0 = create_superposition_state()
    e_ops = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    return run_evolution(H_tau, psi0, t, e_ops=e_ops)


def run_ramsey_experiment():
    """
    Compare standard qubit evolution to τ-driven TDF evolution.

    Evolves the same superposition under H = σ_z (constant) vs H(t) built from
    structured τ(t), and returns trajectories plus the τ field used for TDF.

    Returns
    -------
    t : ndarray
        Time grid ``linspace(0, 10, 300)``.
    result_standard
        ``sesolve`` result for ``constant_hamiltonian(1.0)``.
    result_tau
        ``sesolve`` result for ``tau_to_hamiltonian(structured_tau(...), t)``.
    tau : ndarray
        Phase field ``structured_tau(t, omega=1.0, freq=3.0)``.
    """
    t = np.linspace(0, 10, 300)

    H_standard = constant_hamiltonian(1.0)
    tau = structured_tau(t, omega=1.0, freq=3.0)

    psi0 = create_superposition_state()
    e_ops = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    result_standard = run_evolution(H_standard, psi0, t, e_ops=e_ops)
    result_tau = run_tdf_with_tau(t, tau)

    return t, result_standard, result_tau, tau
