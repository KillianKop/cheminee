"""Compare lift models on the flights held out for validation (same flights and split as `train`).

Each model is asked for its lift FIELD, as the planner sees it, on a window of the DEM around the flight, at the
time of the samples and with the weather of the nearest weather point. Two families of scores:

- agreement with the measured lift, on the thermal and low glide samples (the background pseudo-absences are left
  out: they are not measurements): Pearson r, weighted mean absolute error, weighted binary cross-entropy;
- usefulness for the planner: where the pilot actually found each thermal, how high does the model rank that
  place among the ground within `radius_m` at that time? Mean percentile (0.5 = no better than chance, 1 = always
  the best spot around) and share of thermals in the top `top_fraction` of the area (chance = top_fraction).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import Params
from ..domain import Terrain
from .dataset import BACKGROUND, THERMAL, DatasetConfig, iter_flights, nearest_weather


@dataclass
class EvalConfig:
    val_fraction: float = 0.15  # same split as training (TrainConfig)
    seed: int = 0
    tile_m: float = 10000.0  # the samples of a flight are evaluated tile by tile (bounded windows of the DEM)
    margin_m: float = 3000.0  # around a tile: room for the terrain filters and the ranking disc
    radius_m: float = 3000.0  # ranking of a thermal among the ground within this distance
    top_fraction: float = 0.1
    bin_h: float = 0.5  # fields are computed every half hour
    max_flights: int | None = None  # evaluate only the first validation flights (quicker)


def crop(terrain: Terrain, x0: float, y0: float, x1: float, y1: float) -> Terrain:
    """Window of the DEM covering [x0, x1] x [y0, y1] (metres, clipped to the DEM)."""
    r0, c0 = (int(np.floor(v)) for v in terrain.xy_to_rowcol(x0, y1))
    r1, c1 = (int(np.ceil(v)) + 1 for v in terrain.xy_to_rowcol(x1, y0))
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, terrain.ny), min(c1, terrain.nx)
    cs = terrain.cell_deg
    return Terrain(terrain.z[r0:r1, c0:c1].copy(), terrain.lon0 + c0 * cs, terrain.lat0 + (terrain.ny - r1) * cs, cs)


def _rank(field: np.ndarray, win: Terrain, x: float, y: float, radius_m: float) -> float:
    """Percentile of the field at (x, y) among the cells within radius_m (ties count half)."""
    value = float(win.sample(field, x, y))
    r, c = (float(v) for v in win.xy_to_rowcol(x, y))
    kr, kc = int(radius_m / win.dy) + 1, int(radius_m / win.dx) + 1
    r0, r1 = max(int(r) - kr, 0), min(int(r) + kr + 1, win.ny)
    c0, c1 = max(int(c) - kc, 0), min(int(c) + kc + 1, win.nx)
    rows, cols = np.mgrid[r0:r1, c0:c1]
    disc = ((rows - r) * win.dy) ** 2 + ((cols - c) * win.dx) ** 2 <= radius_m**2
    around = field[r0:r1, c0:c1][disc]
    return float(np.mean(around < value) + 0.5 * np.mean(around == value))


def evaluate(
    terrain: Terrain,
    igc_paths: list,
    models: dict[str, Callable],
    cfg: EvalConfig | None = None,
    data_cfg: DatasetConfig | None = None,
) -> dict:
    """Scores of each model `name -> factory(terrain, params) -> LiftModel` on the validation flights."""
    from .train import validation_flights

    cfg, data_cfg = cfg or EvalConfig(), data_cfg or DatasetConfig()
    flights = list(iter_flights(terrain, igc_paths, data_cfg))
    val = set(validation_flights([k for k, *_ in flights], cfg.val_fraction, cfg.seed).tolist())
    chosen = [f for f in flights if f[0] in val][: cfg.max_flights]
    print(f"[evaluate] {len(flights)} usable flights, {len(val)} held out for validation, {len(chosen)} evaluated")

    pred = {name: [] for name in models}
    ranks = {name: [] for name in models}
    targets, weights = [], []
    n_fields = 0
    for n_done, (_, _, track, fs) in enumerate(chosen, 1):
        keep = fs.kind != BACKGROUND
        near = nearest_weather(fs, terrain)
        tiles = np.floor(fs.x / cfg.tile_m).astype(int) * 100000 + np.floor(fs.y / cfg.tile_m).astype(int)
        hours = fs.t_s / 3600.0
        bins = np.round(hours / cfg.bin_h) * cfg.bin_h
        out = {name: np.full(len(fs.x), np.nan) for name in models}
        # one ranking per thermal: its first sample (all the samples of a thermal share its entry point)
        first_of_thermal = np.zeros(len(fs.x), dtype=bool)
        is_thermal = np.flatnonzero(fs.kind == THERMAL)
        first_of_thermal[is_thermal[np.unique(fs.thermal[is_thermal], return_index=True)[1]]] = True
        lon, lat = terrain.xy_to_lonlat(fs.x, fs.y)
        for tile in np.unique(tiles[keep]):
            in_tile = keep & (tiles == tile)
            i0 = np.flatnonzero(in_tile)[0]
            tx, ty = np.floor(fs.x[i0] / cfg.tile_m), np.floor(fs.y[i0] / cfg.tile_m)
            m, size = cfg.margin_m, cfg.tile_m
            win = crop(terrain, tx * size - m, ty * size - m, (tx + 1) * size + m, (ty + 1) * size + m)
            wx, wy = win.lonlat_to_xy(lon, lat)
            for q in np.unique(near[in_tile]):
                weather = fs.points[q].weather if len(fs.points) > 1 else fs.weather
                params = Params(day_of_year=track.day_of_year, utc_offset=0.0, weather=weather)
                lifts = {name: factory(win, params) for name, factory in models.items()}
                group = in_tile & (near == q)
                for b in np.unique(bins[group]):
                    sel = np.flatnonzero(group & (bins == b))
                    for name, lift in lifts.items():
                        field = lift.field(float(b))  # utc_offset 0: local hour = UTC hour
                        n_fields += 1
                        out[name][sel] = win.sample(field, wx[sel], wy[sel])
                        for i in sel[first_of_thermal[sel]]:
                            ranks[name].append(_rank(field, win, float(wx[i]), float(wy[i]), cfg.radius_m))
        for name in models:
            pred[name].append(out[name][keep])
        targets.append(fs.target[keep])
        weights.append(fs.weight[keep])
        if n_done % 10 == 0 or n_done == len(chosen):
            print(f"[evaluate] {n_done}/{len(chosen)} flights, {n_fields} fields computed")

    y, w = np.concatenate(targets), np.concatenate(weights)
    scores = {}
    for name in models:
        p = np.concatenate(pred[name])
        ok = np.isfinite(p)
        scores[name] = _scores(p[ok], y[ok], w[ok], np.array(ranks[name]), cfg.top_fraction)
    const = np.full(len(y), float(np.average(y, weights=w)))
    scores["constant (mean target)"] = _scores(const, y, w, np.full(1, 0.5), cfg.top_fraction)
    scores["constant (mean target)"].update(rank_percentile=0.5, top_hit=cfg.top_fraction, n_thermals=0)
    return dict(scores=scores, n_samples=int(len(y)), n_flights=len(chosen), top_fraction=cfg.top_fraction)


def _scores(p: np.ndarray, y: np.ndarray, w: np.ndarray, ranks: np.ndarray, top_fraction: float) -> dict:
    eps = 1e-4
    q = np.clip(p, eps, 1 - eps)
    bce = -np.average(y * np.log(q) + (1 - y) * np.log(1 - q), weights=w)
    pearson = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else float("nan")
    return dict(
        pearson=pearson,
        mae=float(np.average(np.abs(p - y), weights=w)),
        bce=float(bce),
        rank_percentile=float(np.mean(ranks)) if len(ranks) else float("nan"),
        top_hit=float(np.mean(ranks >= 1 - top_fraction)) if len(ranks) else float("nan"),
        n_thermals=int(len(ranks)),
    )


def print_scores(result: dict) -> None:
    top = int(round(100 * result["top_fraction"]))
    print(f"[evaluate] {result['n_flights']} validation flights, {result['n_samples']} thermal and low glide samples")
    print(f"{'model':<28}{'Pearson r':>10}{'MAE':>8}{'BCE':>8}{'thermal rank':>14}{f'top {top} %':>10}{'thermals':>10}")
    for name, s in result["scores"].items():
        print(
            f"{name:<28}{s['pearson']:>10.3f}{s['mae']:>8.3f}{s['bce']:>8.4f}"
            f"{s['rank_percentile']:>14.3f}{s['top_hit']:>10.1%}{s['n_thermals']:>10d}"
        )
    print(
        "thermal rank: percentile of the place of each real thermal among the ground around it (0.5 = chance, "
        f"1 = always the best spot); top {top} %: share of thermals in the best {top} % of the ground around "
        f"(chance = {top} %)."
    )
