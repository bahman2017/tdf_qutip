"""
System Hamiltonians for QuTiP simulations.

In the TDF picture used here, **τ** is the **phase field** (same role as in
``core.tau_model``): it tracks accumulated phase so that amplitudes pick up factors
``exp(i τ)``. **Energy is derived from how fast that phase advances**,

    E(t) = ℏ dτ/dt,

so specifying ``τ(t)`` fixes an effective instantaneous energy scale that can drive
the system. Mapping ``τ → E(t)`` and then ``E(t) → H(t)`` (e.g. ``E(t) σ_z``) is how
the phase field **drives** the quantum dynamics in this minimal model.
"""

from __future__ import annotations

import numpy as np
import qutip as qt


def tau_to_energy(tau: np.ndarray, t: np.ndarray, hbar: float = 1.0) -> np.ndarray:
    """
    Compute E(t) from τ(t) using the TDF relation E = ℏ dτ/dt.

    The phase field τ advances in time; its time derivative sets an instantaneous
    energy (up to the choice of how E couples into H, e.g. along ``σ_z``).

    Parameters
    ----------
    tau
        Phase field values (real), same shape as ``t``.
    t
        Time samples (may be unsorted). At equal ``t``, samples are merged by
        averaging ``τ`` before ``dτ/dt`` is taken on the unique-time grid; the
        returned energy is then interpolated back so each original pair
        ``(t, τ)`` gets a value (same ``t`` shares the same ``E``).

    hbar
        Reduced Planck constant in the chosen units (default 1).

    Returns
    -------
    ndarray
        Energy samples with the same shape as ``tau``.
    """
    tau_arr = np.asarray(tau, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    if tau_arr.shape != t_arr.shape:
        raise ValueError("tau and t must have the same shape")
    if tau_arr.size == 0:
        raise ValueError("tau and t must be non-empty")

    flat_tau = tau_arr.ravel()
    flat_t = t_arr.ravel()
    order = np.argsort(flat_t, kind="mergesort")
    t_s = flat_t[order]
    tau_s = flat_tau[order]

    t_u, inv, counts = np.unique(t_s, return_inverse=True, return_counts=True)
    tau_u = np.bincount(inv, weights=tau_s) / counts

    if t_u.size == 1:
        d_tau_dt_u = np.zeros_like(tau_u)
    else:
        d_tau_dt_u = np.gradient(tau_u, t_u)

    e_u = hbar * d_tau_dt_u
    e_on_sorted = np.interp(t_s, t_u, e_u)

    e_flat = np.empty_like(flat_t)
    e_flat[order] = e_on_sorted
    return e_flat.reshape(tau_arr.shape)


def constant_hamiltonian(omega: float) -> qt.Qobj:
    """
    Return a simple constant single-qubit Hamiltonian H = ω σ_z.

    Parameters
    ----------
    omega
        Energy splitting coefficient in front of ``sigmaz``.

    Returns
    -------
    qutip.Qobj
        Time-independent Hamiltonian operator.
    """
    return float(omega) * qt.sigmaz()


def tau_to_hamiltonian(
    tau: np.ndarray,
    t: np.ndarray,
    hbar: float = 1.0,
) -> list:
    """
    Build a QuTiP time-dependent Hamiltonian from τ(t).

    Steps: compute ``E(t) = ℏ dτ/dt`` via :func:`tau_to_energy`, then represent
    ``H(t) = E(t) σ_z`` in mesolve form ``[[sigmaz(), coeff]]`` where ``coeff``
    returns the **interpolated** energy at simulation time ``t`` (signature
    ``coeff(t, args=None)`` for QuTiP 4/5).

    This connects the **phase field** τ to **dynamics**: τ fixes how phase would
    accumulate in a frame; its time derivative supplies the energy that rotates
    the qubit when that energy is coupled to ``σ_z``.

    Parameters
    ----------
    tau
        Phase field τ(t), same shape as ``t``.
    t
        Tabulated times (any order; duplicates allowed). Interpolation uses strictly
        increasing unique times with energies averaged where needed.
    hbar
        Same meaning as in :func:`tau_to_energy`.

    Returns
    -------
    list
        QuTiP ``mesolve`` / ``sesolve``-compatible entry ``[[qt.sigmaz(), coeff]]``.
    """
    E = tau_to_energy(tau, t, hbar=hbar)
    E = np.asarray(np.real_if_close(E), dtype=float)

    t_flat = np.asarray(t, dtype=float).ravel()
    E_flat = E.ravel()
    if t_flat.size != E_flat.size:
        raise ValueError("Internal shape mismatch between t and energy")

    order = np.argsort(t_flat, kind="mergesort")
    t_sorted = t_flat[order]
    E_sorted = E_flat[order]
    t_u, inv = np.unique(t_sorted, return_inverse=True)
    counts = np.bincount(inv)
    E_nodes = np.bincount(inv, weights=E_sorted) / counts
    t_nodes = t_u

    t_min = float(t_nodes[0])
    t_max = float(t_nodes[-1])

    def coeff(time: float, args=None) -> float:
        # QuTiP may call with (t,) only or (t, args); keep args optional.
        tt = float(time)
        if tt <= t_min:
            return float(E_nodes[0])
        if tt >= t_max:
            return float(E_nodes[-1])
        return float(np.interp(tt, t_nodes, E_nodes))

    return [[qt.sigmaz(), coeff]]
