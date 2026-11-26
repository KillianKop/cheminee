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

import os
import sys
import time
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


def _disc(win: Terrain, x: float, y: float, radius_m: float):
    """(rows, cols, disc): the window of the grid around (x, y) and the cells of it within radius_m."""
    r, c = (float(v) for v in win.xy_to_rowcol(x, y))
    kr, kc = int(radius_m / win.dy) + 1, int(radius_m / win.dx) + 1
    r0, r1 = max(int(r) - kr, 0), min(int(r) + kr + 1, win.ny)
    c0, c1 = max(int(c) - kc, 0), min(int(c) + kc + 1, win.nx)
    rows, cols = np.mgrid[r0:r1, c0:c1]
    disc = ((rows - r) * win.dy) ** 2 + ((cols - c) * win.dx) ** 2 <= radius_m**2
    return slice(r0, r1), slice(c0, c1), disc


def _rank(field: np.ndarray, win: Terrain, x: float, y: float, radius_m: float) -> float:
    """Percentile of the field at (x, y) among the cells within radius_m (ties count half)."""
    value = float(win.sample(field, x, y))
    rows, cols, disc = _disc(win, x, y, radius_m)
    around = field[rows, cols][disc]
    return float(np.mean(around < value) + 0.5 * np.mean(around == value))


def _needed(win: Terrain, x: np.ndarray, y: np.ndarray, thermals: np.ndarray, radius_m: float) -> np.ndarray:
    """Cells an evaluation reads: the 2 x 2 cells around each sample (bilinear sampling) and the discs of the
    thermal rankings. A model that supports it computes its field there only."""
    mask = np.zeros(win.z.shape, dtype=bool)
    r, c = win.xy_to_rowcol(x, y)
    r0 = np.clip(np.floor(r).astype(int), 0, win.ny - 1)
    c0 = np.clip(np.floor(c).astype(int), 0, win.nx - 1)
    for dr in (0, 1):
        for dc in (0, 1):
            mask[np.minimum(r0 + dr, win.ny - 1), np.minimum(c0 + dc, win.nx - 1)] = True
    for i in thermals:
        rows, cols, disc = _disc(win, float(x[i]), float(y[i]), radius_m + max(win.dx, win.dy))
        mask[rows, cols] |= disc
    return mask


def _tile_boxes(terrain: Terrain, fs, cfg: EvalConfig):
    """(tile x, tile y, indices of the measured samples in it, window box x0, y0, x1, y1 in metres) of a flight."""
    keep = np.flatnonzero(fs.kind != BACKGROUND)
    tx, ty = np.floor(fs.x[keep] / cfg.tile_m), np.floor(fs.y[keep] / cfg.tile_m)
    m, size = cfg.margin_m, cfg.tile_m
    for a, b in sorted(set(zip(tx.tolist(), ty.tolist()))):
        idx = np.flatnonzero((tx == a) & (ty == b))
        yield idx, (a * size - m, b * size - m, (a + 1) * size + m, (b + 1) * size + m)


def windows(terrain: Terrain, flights: list, cfg: EvalConfig) -> list[tuple[float, float, float, float, int]]:
    """(lon_min, lat_min, lon_max, lat_max, day of year) of every DEM window the evaluation will ask the models
    for: what a model that needs outside data (tiles of a map...) has to fetch beforehand."""
    out = []
    for _, _, track, fs in flights:
        for _, (x0, y0, x1, y1) in _tile_boxes(terrain, fs, cfg):
            win = crop(terrain, x0, y0, x1, y1)
            lon0, lat0 = win.xy_to_lonlat(0.0, 0.0)
            lon1, lat1 = win.xy_to_lonlat(win.width, win.height)
            out.append((float(lon0), float(lat0), float(lon1), float(lat1), int(track.day_of_year)))
    return out


def _flight_task(terrain: Terrain, track, fs, cfg: EvalConfig) -> dict:
    """What a worker needs to score one flight: small DEM windows around its tiles and its measured samples."""
    keep = np.flatnonzero(fs.kind != BACKGROUND)
    near = nearest_weather(fs, terrain)[keep]
    x, y = fs.x[keep], fs.y[keep]
    bins = np.round(fs.t_s[keep] / 3600.0 / cfg.bin_h) * cfg.bin_h
    # one ranking per thermal: its first sample (all the samples of a thermal share its entry point)
    first = np.zeros(len(fs.x), dtype=bool)
    is_thermal = np.flatnonzero(fs.kind == THERMAL)
    first[is_thermal[np.unique(fs.thermal[is_thermal], return_index=True)[1]]] = True
    first = first[keep]
    lon, lat = terrain.xy_to_lonlat(x, y)
    tiles = []
    for idx, box in _tile_boxes(terrain, fs, cfg):
        win = crop(terrain, *box)
        tiles.append(dict(z=win.z, lon0=win.lon0, lat0=win.lat0, cs=win.cell_deg, idx=idx))
    weathers = [q.weather for q in fs.points] if len(fs.points) > 1 else [fs.weather]
    return dict(
        day=track.day_of_year, tiles=tiles, near=near, bins=bins, first=first, lon=lon, lat=lat,
        weathers=weathers, target=fs.target[keep], weight=fs.weight[keep],
    )  # fmt: skip


def _score_flight(task: dict, models: dict, radius_m: float) -> tuple[dict, dict, int]:
    """Predictions of every model at the measured samples of a flight, and the ranks of its thermals."""
    n = len(task["target"])
    out = {name: np.full(n, np.nan) for name in models}
    ranks = {name: [] for name in models}
    n_fields = 0
    for tile in task["tiles"]:
        win = Terrain(tile["z"], tile["lon0"], tile["lat0"], tile["cs"])
        idx = tile["idx"]
        wx, wy = win.lonlat_to_xy(task["lon"][idx], task["lat"][idx])
        for q in np.unique(task["near"][idx]):
            params = Params(day_of_year=task["day"], utc_offset=0.0, weather=task["weathers"][q])
            lifts = {name: factory(win, params) for name, factory in models.items()}
            in_q = task["near"][idx] == q
            for b in np.unique(task["bins"][idx][in_q]):
                sel = np.flatnonzero(in_q & (task["bins"][idx] == b))
                firsts = sel[task["first"][idx[sel]]]
                mask = None
                for name, lift in lifts.items():
                    if getattr(lift, "supports_mask", False):  # compute only the cells the scores read
                        if mask is None:
                            mask = _needed(win, wx[sel], wy[sel], np.searchsorted(sel, firsts), radius_m)
                        field = lift.field(float(b), mask=mask)  # utc_offset 0: local hour = UTC hour
                    else:
                        field = lift.field(float(b))
                    n_fields += 1
                    out[name][idx[sel]] = win.sample(field, wx[sel], wy[sel])
                    for i in firsts:
                        ranks[name].append(_rank(field, win, float(wx[i]), float(wy[i]), radius_m))
    return out, ranks, n_fields


def _init_worker() -> None:
    """One thread per worker: the workers already share the cores."""
    os.environ["OMP_NUM_THREADS"] = "1"
    if "torch" in sys.modules:  # imported with the package, before this initializer
        sys.modules["torch"].set_num_threads(1)


def evaluate(
    terrain: Terrain,
    igc_paths: list,
    models: dict[str, Callable],
    cfg: EvalConfig | None = None,
    data_cfg: DatasetConfig | None = None,
    workers: int = 1,
    validation: list | None = None,
    prepare: Callable[[list], None] | None = None,
) -> dict:
    """Scores of each model `name -> factory(terrain, params) -> LiftModel` on the validation flights.

    With `workers` > 1 the flights are scored in parallel processes; the factories must then be picklable
    (a class such as `HeuristicLift`, or `lift.neural.NeuralFactory`). `validation`: the validation flight
    indices recorded by `train` (only those flights are read); otherwise the split is rebuilt. `prepare`: called
    once with the `windows` of the evaluation before any model is asked for a field (downloads, caches)."""
    from .train import validation_flights

    cfg, data_cfg = cfg or EvalConfig(), data_cfg or DatasetConfig()
    t0 = time.perf_counter()
    if validation is not None:  # recorded by train: read the validation flights only
        val = set(int(k) for k in validation)
        chosen = list(iter_flights(terrain, igc_paths, data_cfg, only=val))[: cfg.max_flights]
        source = "the checkpoint"
    else:  # rebuild the split: every flight is read to know which ones are usable
        flights = list(iter_flights(terrain, igc_paths, data_cfg))
        val = set(validation_flights([k for k, *_ in flights], cfg.val_fraction, cfg.seed).tolist())
        chosen = [f for f in flights if f[0] in val][: cfg.max_flights]
        source = f"{len(flights)} usable flights"
    print(
        f"[evaluate] {len(val)} validation flights (from {source}), {len(chosen)} evaluated, {workers} worker(s) "
        f"({time.perf_counter() - t0:.0f} s to read the flights)"
    )
    if prepare is not None:
        prepare(windows(terrain, chosen, cfg))
    tasks = (_flight_task(terrain, track, fs, cfg) for _, _, track, fs in chosen)
    pred = {name: [] for name in models}
    ranks = {name: [] for name in models}
    targets, weights = [], []
    n_fields = n_done = 0

    def collect(task, result) -> None:
        nonlocal n_fields, n_done
        out, rk, nf = result
        for name in models:
            pred[name].append(out[name])
            ranks[name] += rk[name]
        targets.append(task["target"])
        weights.append(task["weight"])
        n_fields += nf
        n_done += 1
        if n_done % 10 == 0 or n_done == len(chosen):
            print(f"[evaluate] {n_done}/{len(chosen)} flights, {n_fields} fields ({time.perf_counter() - t0:.0f} s)")

    if workers <= 1:
        for task in tasks:
            collect(task, _score_flight(task, models, cfg.radius_m))
    else:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        with ProcessPoolExecutor(workers, initializer=_init_worker) as pool:
            running = {}
            for task in tasks:  # a few flights ahead of the workers only: the DEM windows stay few in memory
                running[pool.submit(_score_flight, task, models, cfg.radius_m)] = task
                if len(running) >= 2 * workers:
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for fut in done:
                        collect(running.pop(fut), fut.result())
            for fut in list(running):
                collect(running.pop(fut), fut.result())

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
