"""Input features of the thermal network, shared by training and inference.

A sample is a (ground point, instant, weather) triple. Features are split in:
- static features, depending on the terrain only (pre-computed once per DEM);
- dynamic features, depending on the sun position (day, hour) and on the weather.
"""

from __future__ import annotations

import numpy as np

from ..domain import Terrain, sun_position, sun_vector
from ..weather import Weather

STATIC_NAMES = ("z_km", "slope_rad", "normal_east", "normal_north", "tpi_100m", "uphill_east", "uphill_north")
DYNAMIC_NAMES = (
    "insolation_now",
    "insolation_1h_ago",
    "insolation_2h_ago",
    "sin_sun_elevation",
    "wind_exposure",
    "wind_speed",
    "cloud_shading",
    "thermal_strength",
    "ceiling_above_ground_km",
    "snow_covered",
    "sin_day",
    "cos_day",
    "sin_solar_hour",
    "cos_solar_hour",
)
FEATURE_NAMES = STATIC_NAMES + DYNAMIC_NAMES
SUN_LAGS_H = (0.0, 1.0, 2.0)  # the network learns the heating delay from these
WIND_REF_MS = 5.0  # wind normalisation


def static_stack(terrain: Terrain) -> np.ndarray:
    """(n_static, ny, nx) stack of terrain-only features."""
    up_x, up_y = terrain.uphill()
    return np.stack(
        [terrain.z / 1000.0, terrain.slope, terrain.normal[0], terrain.normal[1], terrain.tpi() / 100.0, up_x, up_y]
    ).astype(np.float32)


def sample_static(terrain: Terrain, stack: np.ndarray, x, y) -> np.ndarray:
    """(n_points, n_static) static features at metric coordinates (bilinear)."""
    return np.stack([terrain.sample(layer, x, y) for layer in stack], axis=1)


def assemble(static: np.ndarray, day_of_year, hour_utc, lat_deg: float, lon_deg: float, weather: Weather) -> np.ndarray:
    """Build the (n_points, n_features) matrix for ONE weather situation.

    `static` is (n_points, n_static); `day_of_year` is a scalar; `hour_utc` is a scalar or
    an (n_points,) array. Weather quantities are evaluated hour by hour.
    """
    n = len(static)
    hour = np.broadcast_to(np.asarray(hour_utc, dtype=float), (n,))
    solar = hour + lon_deg / 15.0
    z_m, slope, nx, ny = static[:, 0] * 1000.0, static[:, 1], static[:, 2], static[:, 3]
    nz = np.cos(slope)

    cols = [static]
    for lag in SUN_LAGS_H:
        az, el = sun_position(day_of_year, hour - lag, lat_deg, lon_deg)
        s = sun_vector(az, el)
        cols.append((np.clip(nx * s[:, 0] + ny * s[:, 1] + nz * s[:, 2], 0, 1) * (el > 0))[:, None])
        if lag == 0.0:
            sin_el = np.clip(np.sin(np.radians(el)), 0, 1)

    # Weather is piecewise evaluated on the distinct (rounded) hours of the samples.
    exposure, speed, cover, strength, ceiling = (np.zeros(n) for _ in range(5))
    slots = np.round(solar * 4) / 4
    for h in np.unique(slots):
        m = slots == h
        wx, wy = weather.surface_wind(z_m[m], static[m, 5], static[m, 6], float(h))
        exposure[m] = -(nx[m] * wx + ny[m] * wy) / WIND_REF_MS
        speed[m] = np.hypot(wx, wy) / WIND_REF_MS
        cover[m] = 1.0 - weather.sun_factor(float(h))  # radiation lost to clouds
        strength[m] = weather.strength(float(h))
        ceiling[m] = np.clip((weather.ceiling(float(h)) - z_m[m]) / 1000.0, 0, 3)
    snow = (z_m > weather.snow_line_m).astype(float)
    year, day_angle = 2 * np.pi * day_of_year / 365.0, 2 * np.pi * solar / 24.0
    cols += [
        c[:, None]
        for c in (
            sin_el,
            exposure,
            speed,
            cover,
            strength,
            ceiling,
            snow,
            np.full(n, np.sin(year)),
            np.full(n, np.cos(year)),
            np.sin(day_angle),
            np.cos(day_angle),
        )
    ]
    return np.hstack(cols).astype(np.float32)
