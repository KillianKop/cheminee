"""Weather model of one flying day.

All time arguments are SOLAR hours (UTC + longitude / 15), so that the built-in diurnal
shapes follow the sun whatever the time zone. Scalars describe the best part of the
afternoon; any quantity listed in `HOURLY_KEYS` can instead be given hour by hour in
`hourly` (e.g. from a forecast), which then takes precedence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

KMH = 1.0 / 3.6
HOURLY_KEYS = (
    "ceiling_m",
    "wind_speed_kmh",
    "wind_dir_deg",
    "wind_aloft_speed_kmh",
    "wind_aloft_dir_deg",
    "cloud_cover",
    "thermal_strength",
)


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@dataclass
class Weather:
    # -- wind (meteorological convention: direction the wind comes FROM) ------------
    wind_dir_deg: float = 270.0  # low-level wind, valid at `wind_alt_m`
    wind_speed_kmh: float = 10.0
    wind_alt_m: float = 1000.0
    wind_aloft_dir_deg: float | None = None  # None -> same as the low-level wind
    wind_aloft_speed_kmh: float | None = None
    wind_aloft_alt_m: float = 3000.0
    # -- convection -------------------------------------------------------------------
    temp_c: float = 28.0  # afternoon temperature at the reference station
    dewpoint_c: float = 11.0  # dew point at the reference station
    station_alt_m: float = 600.0
    inversion_alt_m: float = 3500.0  # top of the convective layer (caps the thermals)
    base_margin_m: float = 100.0  # the pilot stays this far below the cloud base
    base_morning_fraction: float = 0.75  # cloud base height at 9 h solar / afternoon height
    thermal_strength: float = 1.0  # instability multiplier of the lift index
    cloud_cover: float = 0.2  # fraction of the sky (0-1), shades the ground
    # -- local effects ----------------------------------------------------------------
    breeze_kmh: float = 8.0  # peak up-slope / valley breeze
    snow_line_m: float = 9999.0  # ground above this altitude is snow covered
    cast_shadows: bool = False  # terrain shadows (slower)
    overdev_hour: float | None = None  # solar hour after which flying must stop (storms)
    # -- optional hourly series: name -> [[solar hours], [values]] --------------------
    hourly: dict = field(default_factory=dict)
    # -- optional full wind profile (from a model sounding); replaces the two wind levels:
    #    {"hours": [solar hours], "alt_m": [levels], "u": [[m/s per level] per hour], "v": [...]}
    wind_profile: dict = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------------------
    def _value(self, name: str, solar_hour, default=None):
        if name in self.hourly:
            hours, values = self.hourly[name]
            return np.interp(solar_hour, hours, values)
        value = getattr(self, name, default)
        return default if value is None else value

    def _wind_level(self, prefix: str, solar_hour) -> np.ndarray:
        """Wind vector (m/s, east/north, pointing where the air goes) of one level."""
        if f"{prefix}_dir_deg" in self.hourly and f"{prefix}_speed_kmh" in self.hourly:
            hours, dirs = self.hourly[f"{prefix}_dir_deg"]
            speeds = np.asarray(self.hourly[f"{prefix}_speed_kmh"][1]) * KMH
            a = np.radians(dirs)
            return -np.array(
                [np.interp(solar_hour, hours, speeds * np.sin(a)), np.interp(solar_hour, hours, speeds * np.cos(a))]
            )
        a = np.radians(self._value(f"{prefix}_dir_deg", solar_hour, self.wind_dir_deg))
        speed = self._value(f"{prefix}_speed_kmh", solar_hour, self.wind_speed_kmh) * KMH
        return -speed * np.array([np.sin(a), np.cos(a)])

    # -- wind -------------------------------------------------------------------------
    def wind_vec(self, alt_m, solar_hour: float = 14.0) -> np.ndarray:
        """Synoptic wind vector at altitude(s). Shape (2,) or (2, ...) for array altitudes."""
        if self.wind_profile:
            return self._profile_wind(alt_m, solar_hour)
        low, high = self._wind_level("wind", solar_hour), self._wind_level("wind_aloft", solar_hour)
        u = np.clip(
            (np.asarray(alt_m, dtype=float) - self.wind_alt_m) / max(self.wind_aloft_alt_m - self.wind_alt_m, 1.0),
            0.0,
            1.0,
        )
        return low.reshape((2,) + (1,) * u.ndim) * (1 - u) + high.reshape((2,) + (1,) * u.ndim) * u

    def _profile_wind(self, alt_m, solar_hour: float) -> np.ndarray:
        """Wind from the sounding profile: linear in time, then in altitude (clamped at both ends)."""
        wp = self.wind_profile
        hours, levels = np.asarray(wp["hours"]), np.asarray(wp["alt_m"])
        pos = float(np.interp(solar_hour, hours, np.arange(len(hours))))
        i0 = int(np.floor(pos))
        i1, f = min(i0 + 1, len(hours) - 1), pos - int(np.floor(pos))
        alt = np.asarray(alt_m, dtype=float)
        out = [
            np.interp(alt.ravel(), levels, (1 - f) * np.asarray(wp[c][i0]) + f * np.asarray(wp[c][i1]))
            for c in ("u", "v")
        ]
        return np.stack(out).reshape((2,) + alt.shape)

    def breeze_ms(self, solar_hour) -> float:
        """Up-slope breeze speed: zero at night, maximum in mid-afternoon."""
        return self.breeze_kmh * KMH * float(np.clip(np.sin(np.pi * (solar_hour - 8.5) / 10.5), 0, 1))

    def surface_wind(self, z_m, uphill_x, uphill_y, solar_hour: float):
        """Local low-level wind (wx, wy) = synoptic wind at terrain altitude + up-slope breeze."""
        w = self.wind_vec(z_m, solar_hour)
        b = self.breeze_ms(solar_hour)
        return w[0] + b * uphill_x, w[1] + b * uphill_y

    # -- convection -------------------------------------------------------------------
    def cloud_base(self, solar_hour) -> float:
        """Cloud base AMSL: 125 m per degree of temperature / dew point spread, growing during the day."""
        spread = max(self.temp_c - self.dewpoint_c, 0.0)
        growth = self.base_morning_fraction + (1 - self.base_morning_fraction) * _smoothstep((solar_hour - 9.0) / 6.0)
        return self.station_alt_m + 125.0 * spread * float(growth)

    def ceiling(self, solar_hour) -> float:
        """Usable top of the climbs (m AMSL): below the cloud base and below the inversion."""
        if "ceiling_m" in self.hourly:
            return float(self._value("ceiling_m", solar_hour))
        return float(min(self.cloud_base(solar_hour) - self.base_margin_m, self.inversion_alt_m))

    def sun_factor(self, solar_hour) -> float:
        """Fraction of the clear-sky radiation reaching the ground (Kasten & Czeplak)."""
        cover = float(np.clip(self._value("cloud_cover", solar_hour), 0, 1))
        return 1.0 - 0.75 * cover**3.4

    def strength(self, solar_hour) -> float:
        return float(self._value("thermal_strength", solar_hour))

    def override(self, name: str, value) -> None:
        """Set one value explicitly: it replaces the hourly series of that quantity (and the ones derived from it)."""
        if name not in self.__dataclass_fields__ or name in ("hourly", "wind_profile"):
            raise ValueError(f"Unknown weather parameter: {name}")
        setattr(self, name, value)
        self.hourly.pop(name, None)
        if name in ("temp_c", "dewpoint_c", "inversion_alt_m", "station_alt_m"):
            self.hourly.pop("ceiling_m", None)
        if name.startswith("wind"):
            self.wind_profile = {}

    # -- (de)serialisation ------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Weather:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"Unknown weather parameters: {sorted(unknown)}")
        bad = set(known.get("hourly", {})) - set(HOURLY_KEYS)
        if bad:
            raise ValueError(f"Unknown hourly series: {sorted(bad)} (allowed: {HOURLY_KEYS})")
        return cls(**known)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> Weather:
        with open(path) as f:
            return cls.from_dict(json.load(f))


PRESETS: dict[str, Weather] = {
    "good_day": Weather(),
    "blue_stable": Weather(
        temp_c=27.0,
        dewpoint_c=8.0,
        inversion_alt_m=2300.0,
        thermal_strength=0.7,
        cloud_cover=0.0,
        wind_speed_kmh=6.0,
        breeze_kmh=10.0,
    ),
    "strong_unstable": Weather(
        temp_c=30.0,
        dewpoint_c=10.0,
        inversion_alt_m=4200.0,
        thermal_strength=1.25,
        cloud_cover=0.35,
        wind_speed_kmh=8.0,
        overdev_hour=15.5,
    ),
    "north_wind": Weather(
        wind_dir_deg=340.0,
        wind_speed_kmh=22.0,
        wind_aloft_dir_deg=330.0,
        wind_aloft_speed_kmh=40.0,
        temp_c=24.0,
        dewpoint_c=6.0,
        cloud_cover=0.05,
        thermal_strength=0.9,
        breeze_kmh=4.0,
    ),
    "overcast": Weather(
        cloud_cover=0.85, temp_c=22.0, dewpoint_c=13.0, thermal_strength=0.75, wind_speed_kmh=8.0, breeze_kmh=4.0
    ),
    "spring_snow": Weather(
        temp_c=18.0,
        dewpoint_c=2.0,
        snow_line_m=1700.0,
        thermal_strength=1.1,
        cloud_cover=0.25,
        wind_dir_deg=200.0,
        wind_speed_kmh=12.0,
    ),
}
