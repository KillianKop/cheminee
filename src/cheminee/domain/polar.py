"""Glider polar: sink rate as a function of airspeed."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KMH = 1.0 / 3.6


@dataclass
class GliderPolar:
    """Quadratic polar through three (airspeed km/h, sink m/s) points. Default: EN-B/C paraglider."""

    points: tuple = ((30.0, 1.10), (38.0, 1.25), (52.0, 2.30))  # min sink, trim, accelerated
    trim_kmh: float = 38.0
    circling_sink_ms: float = 1.30  # sink rate while turning in a thermal
    n_speeds: int = 9  # airspeeds tested by the speed-to-fly optimiser
    _coef: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        v = np.array([p[0] for p in self.points]) * KMH
        self._coef = np.polyfit(v, [p[1] for p in self.points], 2)

    def sink(self, v_ms) -> np.ndarray:
        return np.polyval(self._coef, v_ms)

    @property
    def speeds(self) -> np.ndarray:
        """Candidate airspeeds (m/s) between min-sink and full-bar speed; includes trim."""
        v = np.linspace(self.points[0][0], self.points[-1][0], self.n_speeds)
        return np.sort(np.append(v, self.trim_kmh)) * KMH

    @property
    def trim_index(self) -> int:
        return int(np.abs(self.speeds - self.trim_kmh * KMH).argmin())

    @property
    def trim_glide_ratio(self) -> float:
        v = self.trim_kmh * KMH
        return float(v / self.sink(v))
