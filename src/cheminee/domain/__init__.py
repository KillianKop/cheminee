"""Pure domain layer: terrain, sun geometry, routes and scoring (no I/O framework, no ML)."""

from .polar import GliderPolar
from .route import Route
from .scoring import three_tp_distance
from .sun import solar_declination, sun_position, sun_vector
from .terrain import Terrain

__all__ = ["GliderPolar", "Route", "Terrain", "solar_declination", "sun_position", "sun_vector", "three_tp_distance"]
