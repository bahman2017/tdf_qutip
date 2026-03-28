"""
Plotting utilities for experiments and analysis.

Thin wrappers around matplotlib for consistent figure style.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_timeseries(
    t: np.ndarray,
    y: np.ndarray | Sequence[np.ndarray],
    ax: plt.Axes | None = None,
    *,
    labels: list[str] | None = None,
    title: str | None = None,
    xlabel: str = "t",
    ylabel: str | None = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Plot one or more timeseries sharing the same time axis.

    Parameters
    ----------
    t
        1D time samples.
    y
        Single array or sequence of arrays, each same length as ``t``.
    ax
        Optional axes; if None, a new figure and axes are created.
    labels
        Legend entries (one per series in ``y``).
    title
        Axes title.
    xlabel, ylabel
        Axis labels.
    **kwargs
        Forwarded to ``Axes.plot`` (e.g. ``linewidth``).

    Returns
    -------
    matplotlib.axes.Axes
        Axes used for plotting.
    """
    t_arr = np.asarray(t, dtype=float)
    series: list[np.ndarray]
    if isinstance(y, np.ndarray):
        series = [np.asarray(y, dtype=float)]
    else:
        series = [np.asarray(s, dtype=float) for s in y]

    plot_kw = dict(kwargs)
    figsize = plot_kw.pop("figsize", (8, 4))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, s in enumerate(series):
        lab = labels[i] if labels is not None and i < len(labels) else None
        ax.plot(t_arr, s, label=lab, **plot_kw)

    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if labels is not None:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def save_figure(path: str, **kwargs: Any) -> None:
    """
    Save the current figure to ``path``.

    Parameters
    ----------
    path
        Output file path.
    **kwargs
        Forwarded to ``savefig``.
    """
    plt.savefig(path, **kwargs)
