"""Croston and Syntetos-Boylan approximation for intermittent demand."""

from __future__ import annotations

import numpy as np


def croston_sba(values: list[float] | np.ndarray, alpha: float = 0.1) -> float:
    """Return a non-negative, one-step SBA forecast for non-negative demand."""
    demand = np.asarray(values, dtype=float)
    if demand.ndim != 1 or not 0 < alpha <= 1:
        raise ValueError("values must be one-dimensional and alpha must be in (0, 1]")
    if not np.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("demand must contain finite non-negative values")
    nonzero = np.flatnonzero(demand > 0)
    if len(nonzero) == 0:
        return 0.0
    size = float(demand[nonzero[0]])
    interval = float(nonzero[0] + 1)
    previous = int(nonzero[0])
    for index in nonzero[1:]:
        size += alpha * (demand[index] - size)
        interval += alpha * ((index - previous) - interval)
        previous = int(index)
    return max(0.0, (1.0 - alpha / 2.0) * size / interval)
