"""IGC flight logs: parsing, thermal extraction and synthetic generation."""

from .parser import IgcTrack, read_igc, write_igc
from .thermals import Thermal, ThermalConfig, detect_thermals, estimate_wind

__all__ = ["IgcTrack", "Thermal", "ThermalConfig", "detect_thermals", "estimate_wind", "read_igc", "write_igc"]
