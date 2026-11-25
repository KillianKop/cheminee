"""Weather at a few points of the flying area (e.g. the three turnpoints of a triangle).

A flight crosses tens of kilometres: the ceiling and the wind are not the same at the takeoff and at the far
turnpoint. A `WeatherPoint` holds the weather forecast at one place; between the points the weather is blended
with smooth inverse-distance weights (`point_weights`), so that the weather of a point rules around it.

The `.weather.json` sidecar of an IGC flight holds either one serialised `Weather` (older files, synthetic
tracks) or {"points": [...], "meta": {...}} written by `save_points`; `load_points` reads both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .model import Weather

BLEND_SMOOTH_M = 2000.0  # weights 1 / (d^2 + L^2): within ~L of a point, its weather rules


@dataclass
class WeatherPoint:
    lon: float | None  # degrees; None: position unknown (single-weather file), valid everywhere
    lat: float | None
    weather: Weather
    role: str = ""  # e.g. "takeoff", "TP1", "start", "middle", "far"
    hour_utc: float | None = None  # when the flight passed there (IGC), for information

    def to_dict(self) -> dict:
        out = dict(lon=self.lon, lat=self.lat, role=self.role, weather=self.weather.to_dict())
        if self.hour_utc is not None:
            out["hour_utc"] = self.hour_utc
        return out

    @classmethod
    def from_dict(cls, data: dict) -> WeatherPoint:
        return cls(
            lon=data.get("lon"),
            lat=data.get("lat"),
            weather=Weather.from_dict(data["weather"]),
            role=data.get("role", ""),
            hour_utc=data.get("hour_utc"),
        )


def point_weights(px, py, x, y, smooth_m: float = BLEND_SMOOTH_M) -> np.ndarray:
    """Blending weights (K, *x.shape) of K points at (px, py) for positions (x, y), metres; they sum to 1."""
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    shape = (len(px),) + (1,) * x.ndim
    d2 = (x[None] - px.reshape(shape)) ** 2 + (y[None] - py.reshape(shape)) ** 2
    w = 1.0 / (d2 + smooth_m**2)
    return w / w.sum(axis=0, keepdims=True)


def nearest_point(px, py, x, y) -> np.ndarray:
    """Index of the nearest of the points (px, py) for each position (x, y), metres."""
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    shape = (len(px),) + (1,) * x.ndim
    return np.argmin((x[None] - px.reshape(shape)) ** 2 + (y[None] - py.reshape(shape)) ** 2, axis=0)


def save_points(path: str, points: list[WeatherPoint], meta: dict | None = None) -> None:
    with open(path, "w") as f:
        json.dump(dict(points=[p.to_dict() for p in points], meta=meta or {}), f, indent=2)


def load_points(path: str) -> list[WeatherPoint]:
    """Weather points of a sidecar file; a file holding a single `Weather` gives one point without position."""
    with open(path) as f:
        data = json.load(f)
    if "points" in data:
        points = [WeatherPoint.from_dict(p) for p in data["points"]]
        if not points:
            raise ValueError(f"{path}: no weather point")
        return points
    return [WeatherPoint(None, None, Weather.from_dict(data))]
