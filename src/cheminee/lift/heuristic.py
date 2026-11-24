"""Analytic lift model: insolation x slope x ridge x wind, modulated by the weather."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..config import KMH, Params
from ..domain import Terrain, solar_declination, sun_position, sun_vector


@dataclass
class HeuristicLiftConfig:
    thermal_lag_h: float = 1.0  # ground heating delay w.r.t. the sun
    slope_lo_deg: float = 15.0  # slope where the slope factor reaches 1
    slope_hi_deg: float = 65.0  # slope where the slope factor falls back to 0
    tpi_scale_m: float = 40.0  # ridge factor = sigmoid(TPI / tpi_scale_m)
    wind_gain: float = 1.5  # strength of windward bonus / lee penalty
    wind_max_kmh: float = 35.0  # thermals fully destroyed at this local wind speed
    snow_factor: float = 0.15  # residual lift over snow covered ground
    smooth_m: float = 150.0  # spatial smoothing of the field
    seasonal_exp: float = 0.0  # > 0: weaker thermals when the noon sun is low
    alt_opt_m: float | None = None  # optional preferred altitude band (centre)
    alt_width_m: float = 400.0  # ... and its width


class HeuristicLift:
    """L = strength * smooth[ insolation(t - lag) * clouds * shadows * slope * ridge * wind * snow ]."""

    name = "heuristic"

    def __init__(self, terrain: Terrain, params: Params, cfg: HeuristicLiftConfig | None = None):
        self.t, self.p, self.cfg = terrain, params, cfg or HeuristicLiftConfig()
        c, w = self.cfg, params.weather
        deg = np.degrees(terrain.slope)
        slope_f = np.clip(deg / c.slope_lo_deg, 0, 1) * np.clip((c.slope_hi_deg - deg) / 20.0, 0, 1)
        ridge_f = 1.0 / (1.0 + np.exp(-terrain.tpi() / c.tpi_scale_m))
        static = slope_f * ridge_f
        static = np.where(terrain.z > w.snow_line_m, static * c.snow_factor, static)
        if c.alt_opt_m is not None:
            static = static * (0.6 + 0.4 * np.exp(-(((terrain.z - c.alt_opt_m) / c.alt_width_m) ** 2)))
        if c.seasonal_exp > 0:
            lat = np.radians(terrain.lat_mid)
            noon = np.sin(np.pi / 2 - lat + solar_declination(params.day_of_year))
            best = np.sin(np.pi / 2 - lat + np.radians(23.44))
            static = static * float(noon / best) ** c.seasonal_exp
        self.static = static.astype(np.float32)
        self.sigma = c.smooth_m / terrain.dy

    def wind_factor(self, solar_hour: float) -> np.ndarray:
        """Windward faces are fed and anchored, lee faces are turbulent; strong wind kills thermals."""
        t, c = self.t, self.cfg
        wx, wy = self.p.weather.surface_wind(t.z, *t.uphill(), solar_hour)
        exposure = -(t.normal[0] * wx + t.normal[1] * wy)  # m/s blowing INTO the slope
        v_max = c.wind_max_kmh * KMH
        factor = np.clip(0.75 + c.wind_gain * exposure / (0.5 * v_max), 0.1, 1.0)
        return factor * np.clip(1.0 - (np.hypot(wx, wy) / v_max) ** 2, 0, 1)

    def field(self, hour_local: float) -> np.ndarray:
        p, t, w = self.p, self.t, self.p.weather
        solar = p.to_solar(hour_local, t.lon_mid)
        az, el = sun_position(p.day_of_year, p.to_utc(hour_local) - self.cfg.thermal_lag_h, t.lat_mid, t.lon_mid)
        if el <= 0:
            return np.zeros_like(self.static)
        insolation = np.clip(np.tensordot(sun_vector(az, el), t.normal, 1), 0, 1)
        if w.cast_shadows:
            insolation = insolation * t.shadow_mask(float(az), float(el))
        raw = insolation * self.static * self.wind_factor(solar) * w.sun_factor(solar - self.cfg.thermal_lag_h)
        return np.clip(w.strength(solar) * ndimage.gaussian_filter(raw, self.sigma), 0, 1).astype(np.float32)
