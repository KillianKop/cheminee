"""Vertical profile of the atmosphere (model sounding) and its analysis for soaring.

A `Sounding` holds, hour by hour, temperature, dew point and wind on pressure levels
above one point, plus the 2 m values. The analysis follows the classical emagram
reasoning: an air parcel heated at the ground rises along the dry adiabat while it is
warmer than its environment; it condenses at the lifted condensation level (cloud base).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DRY_LAPSE = 9.8e-3  # K/m, dry adiabatic lapse rate
SPREAD_TO_BASE = 125.0  # m of cloud base height per K of temperature / dew point spread


def moist_lapse(temp_c: float, alt_m: float) -> float:
    """Saturated (pseudo-)adiabatic lapse rate in K/m, pressure taken from the standard atmosphere."""
    t = temp_c + 273.15
    p = 1013.25 * (1.0 - 2.25577e-5 * alt_m) ** 5.25588  # hPa
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))  # hPa
    r = 0.622 * es / max(p - es, 1.0)  # kg/kg
    lv, rd, cp, eps, g = 2.5e6, 287.0, 1004.0, 0.622, 9.81
    return g * (1 + lv * r / (rd * t)) / (cp + lv**2 * r * eps / (rd * t**2))


@dataclass
class Sounding:
    hours_utc: np.ndarray  # (n_t,) hour of the day
    elevation_m: float  # model ground altitude at the point
    t2m: np.ndarray  # (n_t,) C
    td2m: np.ndarray  # (n_t,) C
    p_hpa: np.ndarray  # (n_l,) pressure levels, decreasing pressure
    z_m: np.ndarray  # (n_t, n_l) geopotential height
    temp: np.ndarray  # (n_t, n_l) C
    dewpoint: np.ndarray  # (n_t, n_l) C
    wind_kmh: np.ndarray  # (n_t, n_l)
    wind_dir: np.ndarray  # (n_t, n_l) direction the wind comes from, degrees
    wind10_kmh: np.ndarray | None = None  # (n_t,) optional 10 m wind
    wind10_dir: np.ndarray | None = None
    parcel_excess_c: float = 2.0  # how much warmer than the 2 m air the rising parcel starts
    base_margin_m: float = 100.0  # the pilot stays this far below the cloud base

    def index(self, hour_utc: float) -> int:
        return int(np.abs(self.hours_utc - hour_utc).argmin())

    def profile(self, k: int):
        """(altitude, temperature, dew point) from the ground up, underground levels removed."""
        ok = (self.z_m[k] > self.elevation_m + 20.0) & np.isfinite(self.temp[k]) & np.isfinite(self.z_m[k])
        order = np.argsort(self.z_m[k][ok])
        alt = np.r_[self.elevation_m, self.z_m[k][ok][order]]
        temp = np.r_[self.t2m[k], self.temp[k][ok][order]]
        dew = np.r_[
            self.td2m[k], np.where(np.isfinite(self.dewpoint[k][ok][order]), self.dewpoint[k][ok][order], np.nan)
        ]
        return alt, temp, dew

    def cloud_base(self, k: int) -> float:
        """Lifted condensation level of the surface parcel (m AMSL)."""
        return self.elevation_m + SPREAD_TO_BASE * max(self.t2m[k] - self.td2m[k], 0.0)

    def parcel(self, k: int, step_m: float = 25.0):
        """(altitude, temperature) of the rising parcel: dry adiabat, then moist above the cloud base."""
        alt_env, _, _ = self.profile(k)
        base = self.cloud_base(k)
        alts = np.arange(self.elevation_m, alt_env[-1] + step_m, step_m)
        temps = np.empty_like(alts)
        temps[0] = self.t2m[k] + self.parcel_excess_c
        for i in range(1, len(alts)):
            lapse = DRY_LAPSE if alts[i - 1] < base else moist_lapse(temps[i - 1], alts[i - 1])
            temps[i] = temps[i - 1] - lapse * step_m
        return alts, temps

    def thermal_top(self, k: int) -> float:
        """Altitude where the parcel stops being warmer than its environment (m AMSL)."""
        alt_env, temp_env, _ = self.profile(k)
        alts, temps = self.parcel(k)
        cooler = np.flatnonzero(temps <= np.interp(alts, alt_env, temp_env))
        return float(alts[cooler[0]] if len(cooler) else alts[-1])

    def ceiling(self, k: int) -> float:
        """Usable top of the climbs: thermal top, kept below the cloud base."""
        return float(min(self.thermal_top(k), self.cloud_base(k) - self.base_margin_m))

    def strength(self, k: int) -> float:
        """Crude instability index from the depth of the dry convective layer (ground to
        thermal top or cloud base, whichever is lower).

        The convective velocity scale grows like depth^(1/3); 1500 m of depth is mapped to 1.
        """
        depth = max(min(self.thermal_top(k), self.cloud_base(k)) - self.elevation_m, 0.0)
        return float(np.clip((depth / 1500.0) ** (1.0 / 3.0), 0.3, 1.3))

    def wind_levels(self, k: int):
        """(altitude, speed km/h, direction) from the ground up, including the 10 m wind if known."""
        ok = (self.z_m[k] > self.elevation_m + 20.0) & np.isfinite(self.wind_kmh[k])
        order = np.argsort(self.z_m[k][ok])
        alt, speed, direction = self.z_m[k][ok][order], self.wind_kmh[k][ok][order], self.wind_dir[k][ok][order]
        if self.wind10_kmh is not None and np.isfinite(self.wind10_kmh[k]):
            alt = np.r_[self.elevation_m + 10.0, alt]
            speed, direction = np.r_[self.wind10_kmh[k], speed], np.r_[self.wind10_dir[k], direction]
        return alt, speed, direction

    def wind_profile(self, lon_deg: float) -> dict:
        """Wind profile in the format of `Weather.wind_profile` (solar hours, mean level altitudes)."""
        alts = [self.wind_levels(k)[0] for k in range(len(self.hours_utc))]
        n = min(len(a) for a in alts)
        if n == 0:
            return {}
        u, v = [], []
        for k in range(len(self.hours_utc)):
            _, speed, direction = self.wind_levels(k)
            a = np.radians(direction[-n:])
            u.append((-speed[-n:] / 3.6 * np.sin(a)).round(3).tolist())
            v.append((-speed[-n:] / 3.6 * np.cos(a)).round(3).tolist())
        return {
            "hours": (self.hours_utc + lon_deg / 15.0).round(3).tolist(),
            "alt_m": np.mean([a[-n:] for a in alts], axis=0).round(1).tolist(),
            "u": u,
            "v": v,
        }
