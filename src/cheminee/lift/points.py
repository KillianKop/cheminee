"""Lift with the weather of several points: one field per point weather, blended with `point_weights`."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from ..config import Params
from ..domain import Terrain
from ..weather.points import point_weights


class PointsLift:
    """field(h) = sum_k w_k(x, y) * field_k(h), where field_k is the lift model under the weather of point k."""

    def __init__(self, terrain: Terrain, params: Params, factory: Callable):
        pts = params.weather_points
        self.models = [factory(terrain, dataclasses.replace(params, weather=q.weather, weather_points=())) for q in pts]
        px, py = terrain.lonlat_to_xy(np.array([q.lon for q in pts]), np.array([q.lat for q in pts]))
        rows, cols = np.indices(terrain.z.shape)
        x, y = terrain.rowcol_to_xy(rows, cols)
        self.weights = point_weights(px, py, x, y).astype(np.float16)  # (K, ny, nx); half size: up to 10 points
        self.norm = self.weights.sum(axis=0, dtype=np.float32)  # 1 up to float16 rounding: divided out
        self.name = f"{self.models[0].name}, {len(pts)} weather points"

    def field(self, hour_local: float) -> np.ndarray:
        out = np.zeros(self.weights.shape[1:], dtype=np.float32)
        for w, model in zip(self.weights, self.models):
            out += w * model.field(hour_local)
        return np.clip(out / self.norm, 0.0, 1.0)


def with_weather_points(factory: Callable) -> Callable:
    """Wrap a lift factory `(terrain, params) -> LiftModel` so that it honours `Params.weather_points`."""

    def make(terrain: Terrain, params: Params):
        if len(params.weather_points) > 1:
            return PointsLift(terrain, params, factory)
        return factory(terrain, params)

    return make
