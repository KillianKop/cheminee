"""Parameters of one planning request (day, weather, glider, margins, search)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .domain.polar import GliderPolar
from .weather.model import Weather

KMH = 1.0 / 3.6  # km/h -> m/s
OBJECTIVES = ("triangle", "fai", "free", "score")


@dataclass
class Params:
    """Times are local hours, altitudes AMSL."""

    # Day / time
    day_of_year: int = 180  # ~29 June
    utc_offset: float = 2.0  # CEST
    t_start: float = 11.0  # earliest takeoff time
    t_end: float = 18.0  # latest landing time
    # Atmosphere and glider
    weather: Weather = field(default_factory=Weather)
    polar: GliderPolar = field(default_factory=GliderPolar)
    # Lift index <-> climb rate: vertical air velocity = w_ref_ms * L
    w_ref_ms: float = 5.0
    min_climb_ms: float = 0.5  # a thermal is usable above this climb rate
    # Safety margins
    clearance_m: float = 60.0  # min terrain clearance during a glide
    arrival_margin_m: float = 120.0  # min height above a thermal node on arrival
    # Thermal nodes
    n_nodes: int = 70
    node_spacing_m: float = 1200.0
    # Search
    objective: str = "triangle"  # one of OBJECTIVES (triangle = flat or FAI, best points first)
    dt_min: float = 5.0  # time bin
    labels_per_state: int = 3  # routes kept per (node, time bin)
    n_sectors: int = 12  # neighbours: the nearest reachable ones per sector
    neighbours_per_sector: int = 3  # more: longer glides that skip weak thermals (2 -> 3: +8 % on the sample)
    max_visits: int = 2  # max visits of the same node (free distance)
    n_routes: int = 5  # number of (diverse) routes returned
    diversity: float = 0.5  # max Jaccard similarity between returned routes
    max_triangle_evals: int = 30000  # budget of the triangle optimiser (per departure time)
    partial_climbs: bool = True  # climb only what is needed, skip weak thermals, final glide (False: ceiling)
    dolphin_core_m: float = 150.0  # rising air crossed when a thermal is passed without circling
    triangle_departure_step_h: float = 0.5  # triangles also leave the takeoff later, every step (0: first time only)
    # Scoring: CFD coefficients (free distance over 3 turnpoints, flat triangle, FAI triangle)
    coef_free: float = 1.0
    coef_triangle: float = 1.2
    coef_fai: float = 1.4
    closing_fraction: float = 0.05  # a triangle is closed within this fraction of its perimeter
    fai_min_leg: float = 0.28  # FAI triangle: shortest side >= 28 % of the perimeter

    @property
    def lift_min(self) -> float:
        """Lift index giving exactly the minimum usable climb rate."""
        return (self.min_climb_ms + self.polar.circling_sink_ms) / self.w_ref_ms

    def to_utc(self, hour_local: float) -> float:
        return hour_local - self.utc_offset

    def to_solar(self, hour_local: float, lon_deg: float) -> float:
        return hour_local - self.utc_offset + lon_deg / 15.0

    def from_solar(self, solar_hour: float, lon_deg: float) -> float:
        return solar_hour + self.utc_offset - lon_deg / 15.0
