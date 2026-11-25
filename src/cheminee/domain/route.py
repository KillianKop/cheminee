"""Result of the planner: one candidate flight."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KIND_LABELS = {"free": "free distance", "triangle": "flat triangle", "fai": "FAI triangle", "goal": "goal"}


@dataclass
class Route:
    nodes: list[int]  # node indices (0 = takeoff)
    times: list[float]  # local hours when leaving / reaching each node
    distance_km: float  # scored distance (3-TP free distance or triangle perimeter)
    turnpoints: list[int]  # positions in `nodes` of the scoring points
    kind: str = "free"  # "free" | "triangle" | "fai" | "goal"
    points: float = 0.0  # distance x coefficient of the flight type
    xy: np.ndarray = field(default=None, repr=False)  # (n, 3): x, y, ground z of the nodes
    profile: np.ndarray = field(default=None, repr=False)  # (m, 4): x, y, flight altitude, hour
    legs: list = field(default=None, repr=False)  # glides of the profile, in order (planner data, see route_track)

    @property
    def duration_h(self) -> float:
        return self.times[-1] - self.times[0]

    @property
    def speed_kmh(self) -> float:
        return self.distance_km / max(self.duration_h, 1e-6)

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)
