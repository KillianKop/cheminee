"""Interface every lift model must implement (dependency inversion for the planner)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class LiftModel(Protocol):
    """Dimensionless thermal quality index L(x, y, t) in [0, 1] over the terrain grid."""

    name: str

    def field(self, hour_local: float) -> np.ndarray:
        """Return the (ny, nx) lift index at a local time of the configured day."""
        ...
