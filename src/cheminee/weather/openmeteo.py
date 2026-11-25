"""Build a `Weather` (and a `Sounding`) from the Open-Meteo forecast API (https://open-meteo.com).

No API key is needed for non-commercial use. The conversion functions are separated from
the network call so that they can be tested offline on a saved answer:

    payload (dict)  --sounding_from_payload-->  Sounding
    payload (dict)  --weather_from_payload--->  Weather

`tests/data/openmeteo_digne_2026-11-15.json` is a real answer of the API.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.parse
import urllib.request

import numpy as np

from .model import Weather
from .sounding import Sounding

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"  # past dates, same parameters
DEFAULT_MODEL = "meteofrance_seamless"  # AROME where available, ARPEGE elsewhere
BEST_MATCH = "best_match"  # Open-Meteo's own choice: no `models` parameter

# Models offered by the interfaces: Open-Meteo `models` value -> (label, grid spacing km over the Alps,
# approximate forecast range in days, pressure levels, short name). Ranges change with the model runs: a forecast
# that does not cover the flying day is refused by `weather_from_payload` anyway. Without pressure levels the
# ceiling comes from the surface spread and the boundary layer height only.
FORECAST_MODELS = {
    "meteofrance_seamless": ("Météo-France AROME + ARPEGE", 1.5, 4, True, "MF seamless"),
    "meteofrance_arome_france_hd": ("Météo-France AROME HD", 1.5, 1.5, False, "AROME HD"),
    "meteofrance_arome_france": ("Météo-France AROME", 2.5, 2, True, "AROME"),
    "meteofrance_arpege_europe": ("Météo-France ARPEGE Europe", 11, 4, True, "ARPEGE"),
    "icon_seamless": ("DWD ICON (D2 + EU + global)", 2, 7, True, "ICON"),
    "icon_d2": ("DWD ICON-D2", 2, 2, True, "ICON-D2"),
    "ecmwf_ifs025": ("ECMWF IFS", 25, 10, True, "ECMWF"),
    "ukmo_seamless": ("UK Met Office", 10, 7, True, "UKMO"),
    "gfs_seamless": ("NOAA GFS", 25, 16, True, "GFS"),
    BEST_MATCH: ("Open-Meteo best match", None, 16, True, "best match"),
}
DAY_COVERAGE_MIN = 0.8  # share of the flying hours (11-17 h solar) a forecast must cover


class ForecastUnavailable(ValueError):
    """The answer does not cover the flying day (date beyond the model range, or outside its domain)."""

    def __init__(self, covered: int, total: int):
        super().__init__(
            f"the forecast covers {covered} of the {total} flying hours of the day "
            "(date beyond the model range, or place outside its domain)"
        )
        self.covered, self.total = covered, total


SURFACE_VARS = (
    "temperature_2m",
    "dew_point_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
    "cape",
    "freezing_level_height",
)
LEVEL_VARS = ("temperature", "dew_point", "geopotential_height", "wind_speed", "wind_direction")
LEVELS_HPA = (950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500)


def build_url(
    lat: float,
    lon: float,
    date: dt.date,
    model: str | None = DEFAULT_MODEL,
    levels: tuple = LEVELS_HPA,
    today: dt.date | None = None,
) -> str:
    """Request URL for one day. Dates older than a few days go to the archive host."""
    hourly = list(SURFACE_VARS) + [f"{v}_{p}hPa" for p in levels for v in LEVEL_VARS]
    query = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": ",".join(hourly),
        "wind_speed_unit": "kmh",
        "timezone": "GMT",
        "start_date": date.isoformat(),
        "end_date": date.isoformat(),
    }
    if model and model != BEST_MATCH:
        query["models"] = model
    old = (today or dt.date.today()) - date > dt.timedelta(days=5)
    return f"{ARCHIVE_URL if old else FORECAST_URL}?{urllib.parse.urlencode(query, safe=',')}"


def _series(hourly: dict, name: str) -> np.ndarray | None:
    if name not in hourly:
        return None
    return np.array([np.nan if v is None else v for v in hourly[name]], dtype=float)


def sounding_from_payload(payload: dict) -> Sounding | None:
    """Extract the pressure-level profile of an Open-Meteo answer. None if it has no level data."""
    h = payload["hourly"]
    levels = sorted({int(k.split("_")[-1][:-3]) for k in h if k.startswith("geopotential_height_")}, reverse=True)
    levels = [p for p in levels if f"temperature_{p}hPa" in h]
    if not levels or "temperature_2m" not in h or "dew_point_2m" not in h:
        return None
    n_t = len(h["time"])

    def stack(var):
        cols = [_series(h, f"{var}_{p}hPa") for p in levels]
        return np.column_stack([c if c is not None else np.full(n_t, np.nan) for c in cols])

    hours = np.array(
        [dt.datetime.fromisoformat(t).hour + dt.datetime.fromisoformat(t).minute / 60.0 for t in h["time"]]
    )
    return Sounding(
        hours,
        float(payload.get("elevation", 0.0)),
        _series(h, "temperature_2m"),
        _series(h, "dew_point_2m"),
        np.array(levels, dtype=float),
        stack("geopotential_height"),
        stack("temperature"),
        stack("dew_point"),
        stack("wind_speed"),
        stack("wind_direction"),
        _series(h, "wind_speed_10m"),
        _series(h, "wind_direction_10m"),
    )


def weather_from_payload(payload: dict, lon: float, base: Weather | None = None) -> Weather:
    """Convert an Open-Meteo JSON answer (hourly block, times in GMT) into a Weather.

    With pressure levels in the answer, the ceiling and the instability come from the
    parcel analysis of the sounding and the wind from its full profile. Without them,
    the ceiling falls back to the temperature / dew point spread and the boundary layer
    height, and the wind to the 10 m and 700 hPa values.
    """
    h = payload["hourly"]
    solar = np.array([dt.datetime.fromisoformat(t).hour for t in h["time"]], dtype=float) + lon / 15.0
    station_alt = float(payload.get("elevation", 600.0))

    def clean(values):
        ok = ~np.isnan(values)
        return [list(map(float, solar[ok])), list(map(float, values[ok]))]

    w = Weather(**(base.to_dict() if base else {}))
    w.station_alt_m = station_alt
    day = (solar >= 11) & (solar <= 17)
    temp, dew = _series(h, "temperature_2m"), _series(h, "dew_point_2m")
    covered = 0 if temp is None else int(np.sum(~np.isnan(temp[day])))
    if covered < DAY_COVERAGE_MIN * max(int(day.sum()), 1):
        raise ForecastUnavailable(covered, int(day.sum()))
    hourly = {}
    sounding = sounding_from_payload(payload)
    if sounding is not None:
        sounding.base_margin_m = w.base_margin_m
        ks = range(len(solar))
        ceiling = np.array([sounding.ceiling(k) for k in ks])
        hourly["ceiling_m"] = clean(ceiling)
        hourly["thermal_strength"] = clean(np.array([sounding.strength(k) for k in ks]))
        w.inversion_alt_m = float(max(sounding.thermal_top(k) for k in np.flatnonzero(day)))
        w.thermal_strength = float(np.mean([sounding.strength(k) for k in np.flatnonzero(day)]))
        w.wind_profile = sounding.wind_profile(lon)
    if temp is not None and dew is not None:
        w.temp_c, w.dewpoint_c = float(np.nanmax(temp[day])), float(np.nanmean(dew[day]))
        if sounding is None:
            ceiling = station_alt + 125.0 * np.clip(temp - dew, 0, None) - w.base_margin_m
            blh = _series(h, "boundary_layer_height")
            if blh is not None and not np.isnan(blh[day]).all():
                w.inversion_alt_m = float(station_alt + np.nanmax(blh[day]))
                ceiling = np.fmin(ceiling, station_alt + blh)
            hourly["ceiling_m"] = clean(ceiling)
    cover = _series(h, "cloud_cover")
    if cover is not None:
        hourly["cloud_cover"] = clean(cover / 100.0)
        w.cloud_cover = float(np.nanmean(cover[day]) / 100.0)
    for prefix, speed_key, dir_key in (
        ("wind", "wind_speed_10m", "wind_direction_10m"),
        ("wind_aloft", "wind_speed_700hPa", "wind_direction_700hPa"),
    ):
        speed, direction = _series(h, speed_key), _series(h, dir_key)
        if speed is not None and direction is not None and not np.isnan(speed).all():
            hourly[f"{prefix}_speed_kmh"], hourly[f"{prefix}_dir_deg"] = clean(speed), clean(direction)
    w.wind_alt_m, w.wind_aloft_alt_m = station_alt, 3000.0
    if w.wind_profile:  # scalar summary of the profile, for display and for the sliders
        for prefix, alt in (("wind", w.wind_alt_m), ("wind_aloft", w.wind_aloft_alt_m)):
            v = w.wind_vec(alt, 14.0)
            setattr(w, f"{prefix}_speed_kmh", float(np.hypot(*v) * 3.6))
            setattr(w, f"{prefix}_dir_deg", float(np.degrees(np.arctan2(-v[0], -v[1])) % 360.0))
    cape = _series(h, "cape")
    if cape is not None and not np.isnan(cape[day]).all():
        if sounding is None:  # crude mapping: no CAPE -> weak blue thermals, 600 J/kg and more -> strong day
            w.thermal_strength = float(np.clip(0.75 + 0.5 * np.nanmax(cape[day]) / 600.0, 0.75, 1.25))
        if np.nanmax(cape[day]) > 1200.0:
            w.overdev_hour = float(solar[day][np.nanargmax(cape[day])])
    frz = _series(h, "freezing_level_height")
    if frz is not None and not np.isnan(frz[day]).all() and np.nanmean(frz[day]) < 2500.0:
        w.snow_line_m = float(np.nanmean(frz[day]) - 300.0)
    w.hourly = hourly
    return w


def fetch_payload(
    lat: float, lon: float, date: dt.date, model: str | None = DEFAULT_MODEL, timeout: float = 30.0
) -> dict:
    url = build_url(lat, lon, date, model)
    print(f"[weather] GET {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def fetch(
    lat: float, lon: float, date: dt.date, base: Weather | None = None, model: str | None = DEFAULT_MODEL
) -> Weather:
    return weather_from_payload(fetch_payload(lat, lon, date, model), lon, base)
