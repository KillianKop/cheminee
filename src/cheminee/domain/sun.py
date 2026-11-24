"""Solar geometry (vectorised). Angles in degrees, time in UTC hours."""

from __future__ import annotations

import numpy as np


def solar_declination(day_of_year) -> np.ndarray:
    """Solar declination in radians (Cooper's formula)."""
    return np.radians(23.44) * np.sin(2 * np.pi * (284 + np.asarray(day_of_year)) / 365.0)


def sun_position(day_of_year, hour_utc, lat_deg: float, lon_deg: float):
    """Return (azimuth, elevation) in degrees. Azimuth: 0 = north, 90 = east.

    `day_of_year` and `hour_utc` may be scalars or arrays (broadcast together).
    """
    day = np.asarray(day_of_year, dtype=float)
    b = 2 * np.pi * (day - 81) / 364.0
    eq_time_min = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    decl = solar_declination(day)
    solar_time = np.asarray(hour_utc, dtype=float) + lon_deg / 15.0 + eq_time_min / 60.0
    h = np.radians(15.0 * (solar_time - 12.0))  # hour angle
    lat = np.radians(lat_deg)
    sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(h)
    el = np.arcsin(np.clip(sin_el, -1, 1))
    az = np.arctan2(np.sin(h), np.cos(h) * np.sin(lat) - np.tan(decl) * np.cos(lat)) + np.pi
    return np.degrees(az) % 360.0, np.degrees(el)


def sun_vector(az_deg, el_deg) -> np.ndarray:
    """Unit vector pointing to the sun, components (east, north, up) on the last axis."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], axis=-1)
