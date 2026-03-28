"""
Time evolution using QuTiP (unitary and open-system).

Central place for mesolve / sesolve wrappers and τ-dependent evolution hooks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import qutip as qt


def create_superposition_state() -> qt.Qobj:
    """
    Equal superposition (|0⟩ + |1⟩) / √2 on the computational basis of one qubit.
    """
    return (qt.basis(2, 0) + qt.basis(2, 1)).unit()


def run_evolution(
    H: qt.Qobj | list,
    psi0: qt.Qobj,
    tlist: np.ndarray,
    **kwargs: Any,
) -> Any:
    """
    Unitary Schrödinger evolution of a pure state (closed system).

    Wraps ``qutip.sesolve`` so experiments share one entry point.

    Parameters
    ----------
    H
        Time-independent ``Qobj`` or time-dependent list accepted by ``sesolve``.
    psi0
        Initial ket.
    tlist
        Times at which the state is returned.
    **kwargs
        Forwarded to ``qutip.sesolve``.

    Returns
    -------
    Any
        ``sesolve`` result (states, etc.).
    """
    return qt.sesolve(H, psi0, tlist, **kwargs)


def evolve_closed(
    H: qt.Qobj,
    psi0: qt.Qobj,
    tlist: np.ndarray,
    **kwargs: Any,
) -> Any:
    """
    Evolve a pure state under a (possibly time-dependent) Hamiltonian.

    Parameters
    ----------
    H
        Hamiltonian or list format accepted by ``qutip.sesolve``.
    psi0
        Initial state.
    tlist
        Time list for the solver.
    **kwargs
        Extra arguments forwarded to the solver (stub).

    Returns
    -------
    Any
        QuTiP solver result object (``sesolve`` output).
    """
    return run_evolution(H, psi0, tlist, **kwargs)


def evolve_open(
    H: qt.Qobj,
    rho0: qt.Qobj,
    tlist: np.ndarray,
    c_ops: list[qt.Qobj],
    **kwargs: Any,
) -> Any:
    """
    Evolve a density matrix with Lindblad collapse operators.

    Parameters
    ----------
    H
        Hamiltonian.
    rho0
        Initial density matrix.
    tlist
        Time list for the solver.
    c_ops
        Collapse operators.
    **kwargs
        Extra arguments forwarded to ``qutip.mesolve`` (stub).

    Returns
    -------
    Any
        QuTiP solver result object (``mesolve`` output).
    """
    return qt.mesolve(H, rho0, tlist, c_ops, **kwargs)
