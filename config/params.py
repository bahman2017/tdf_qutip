"""
Central simulation parameters.

Single place for default Hilbert space sizes, time grids, noise rates, and τ-model knobs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationParams:
    """Container for default numerical and physical parameters (values are placeholders)."""

    n_qubits: int = 1
    t_max: float = 1.0
    n_steps: int = 100


def default_params() -> SimulationParams:
    """Return a fresh copy of default simulation parameters."""
    return SimulationParams()
