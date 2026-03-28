"""
Standard and custom noise models for open-system QuTiP simulations.

Wraps Lindblad / collapse operators and project-specific noise channels.
"""

from __future__ import annotations

from typing import Any

import qutip as qt


def standard_lindblad_ops(**kwargs: Any) -> list[qt.Qobj]:
    """
    Return collapse operators for a standard noise model (e.g. amplitude / phase damping).

    Parameters
    ----------
    **kwargs
        Rates and system size (stub).

    Returns
    -------
    list of qutip.Qobj
        Collapse operators (stub).
    """
    raise NotImplementedError


def custom_noise_ops(**kwargs: Any) -> list[qt.Qobj]:
    """
    Return collapse operators for the project's custom noise model.

    Parameters
    ----------
    **kwargs
        Custom rate tensors or correlation structure (stub).

    Returns
    -------
    list of qutip.Qobj
        Collapse operators (stub).
    """
    raise NotImplementedError
