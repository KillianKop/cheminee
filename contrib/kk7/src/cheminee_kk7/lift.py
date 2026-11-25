"""Lift model that blends the kk7 thermal climatology into a cheminee lift model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cheminee.config import Params
from cheminee.domain import Terrain, sun_position

from . import fetch, tiles

LiftFactory = Callable[[Terrain, Params], object]


def sunrise_local(terrain: Terrain, params: Params) -> float:
    """Local hour of sunrise at the centre of the DEM (upper limb at the horizon)."""
    hours_utc = np.arange(0.0, 24.0, 1.0 / 60.0)
    _, el = sun_position(params.day_of_year, hours_utc, terrain.lat_mid, terrain.lon_mid)
    up = np.nonzero(el > -0.83)[0]
    return (float(hours_utc[up[0]]) if len(up) else 12.0) + params.utc_offset


def layer_keys(season: str, time_of_day: str) -> list[str]:
    """kk7 layers needed: one season, and either one time of day or all of them (interpolated)."""
    times = list(tiles.TIMES_OF_DAY) if time_of_day == "auto" else [time_of_day]
    return [f"{season}_{t}" for t in times]


def load_layers(
    terrain: Terrain,
    keys: list[str],
    zoom: int = tiles.MAX_ZOOM,
    cache_dir: str = fetch.DEFAULT_CACHE,
    offline: bool = False,
) -> dict[str, np.ndarray]:
    """kk7 probability of each layer on the terrain grid (downloads only the tiles over the DEM)."""
    cover = tiles.tiles_covering(*tiles.terrain_bbox(terrain), zoom)
    return {
        key: tiles.to_terrain_grid(terrain, fetch.fetch_layer(cover, key, zoom, cache_dir, offline), zoom)
        for key in keys
    }


MODES = ("boost", "blend")
TOP_PERCENTILE = 99.9  # "maximum" of a field, robust to a few outlier cells


class Kk7Lift:
    """Lift = cheminee base model, reshaped by the kk7 thermal climatology.

    kk7 says WHERE thermals are usually found (a climatology built from flown tracks, per season and time of
    day); the base model (heuristic or network) says HOW STRONG lift is today (sun, clouds, wind, instability).
    With K = amplitude(base) * kk7 / max(kk7), where amplitude is the top of the base field at the same hour
    (a day without lift stays a day without lift, climb rates keep the scale of the base model):

    - mode "boost" (default): L = max(base, weight * K). kk7 raises lift where thermals are known and never
      removes any: kk7 is transparent wherever nobody flies, and missing data is not a missing thermal.
    - mode "blend": L = (1 - weight) * base + weight * K. Treats kk7 as complete: lift fades away from the
      kk7 spots. Only sensible where kk7 coverage is dense.
    """

    def __init__(
        self,
        terrain: Terrain,
        params: Params,
        base,
        layers: dict[str, np.ndarray],
        blend: float = 0.5,
        season: str = "auto",
        time_of_day: str = "auto",
        mode: str = "boost",
    ):
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {blend}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.t, self.p, self.base = terrain, params, base
        self.season = tiles.nearest_season(params.day_of_year) if season == "auto" else season
        self.time_of_day = time_of_day
        self.sunrise = sunrise_local(terrain, params)
        self.layers = layers  # probabilities, for display
        missing = [k for k in layer_keys(self.season, time_of_day) if k not in layers]
        if missing:
            raise ValueError(f"missing kk7 layers: {missing}")
        top = {k: float(np.percentile(v, TOP_PERCENTILE)) for k, v in layers.items()}
        self._normed = {k: np.minimum(v / top[k], 1.0) if top[k] > 0 else np.zeros_like(v) for k, v in layers.items()}
        self.blend = blend
        if blend > 0 and max(top.values(), default=0.0) <= 0:
            print("[kk7] warning: no kk7 data over this DEM, using the base lift model only")
            self.blend = 0.0
        self.name = f"{base.name} + kk7 {self.season} {mode} x{self.blend:.2f}"

    def layer_weights(self, hour_local: float) -> dict[str, float]:
        if self.time_of_day != "auto":
            return {f"{self.season}_{self.time_of_day}": 1.0}
        w = tiles.time_of_day_weights(hour_local - self.sunrise)
        return {f"{self.season}_{t}": v for t, v in w.items() if v > 0}

    def _mix(self, grids: dict[str, np.ndarray], hour_local: float) -> np.ndarray:
        return sum(w * grids[k] for k, w in self.layer_weights(hour_local).items())

    def probability(self, hour_local: float) -> np.ndarray:
        """kk7 thermal probability at this hour (time-of-day layers interpolated)."""
        return self._mix(self.layers, hour_local).astype(np.float32)

    def field(self, hour_local: float) -> np.ndarray:
        base = self.base.field(hour_local)
        if self.blend == 0.0:
            return base
        amplitude = float(np.percentile(base, TOP_PERCENTILE))
        if amplitude <= 0.0:
            return base
        kk7 = amplitude * self._mix(self._normed, hour_local)
        if self.mode == "boost":
            out = np.maximum(base, self.blend * kk7)
        else:
            out = (1.0 - self.blend) * base + self.blend * kk7
        return np.clip(out, 0.0, 1.0).astype(np.float32)


def kk7_factory(
    base_factory: LiftFactory,
    blend: float = 0.5,
    season: str = "auto",
    time_of_day: str = "auto",
    zoom: int = tiles.MAX_ZOOM,
    cache_dir: str = fetch.DEFAULT_CACHE,
    offline: bool = False,
    mode: str = "boost",
) -> LiftFactory:
    """Wrap a cheminee lift factory `(terrain, params) -> LiftModel` (as built by `cheminee.cli._flight_setup`)."""
    loaded: dict[tuple[int, tuple[str, ...]], dict[str, np.ndarray]] = {}  # the GUI calls the factory again

    def factory(terrain: Terrain, params: Params) -> Kk7Lift:
        s = tiles.nearest_season(params.day_of_year) if season == "auto" else season
        keys = layer_keys(s, time_of_day)
        cache_key = (id(terrain), tuple(keys))
        if cache_key not in loaded:
            loaded[cache_key] = load_layers(terrain, keys, zoom, cache_dir, offline)
        base = base_factory(terrain, params)
        return Kk7Lift(terrain, params, base, loaded[cache_key], blend, s, time_of_day, mode)

    return factory
