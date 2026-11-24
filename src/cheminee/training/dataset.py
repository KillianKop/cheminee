"""Build the supervised dataset of the thermal network from IGC tracks.

The target is a lift index in [0, 1] derived from the vertical velocity of the AIR:

    netto = vertical speed of the glider + still-air sink rate of the glider
    target = clip(netto / w_ref, 0, 1)

Three kinds of samples are extracted from each flight:
- thermal samples   : netto from the mean climb rate, located at the ENTRY point of the
                      thermal (proxy of its trigger on the ground, before wind drift);
- low glide samples : netto measured while gliding close to the terrain, located under
                      the pilot (graded information, including "no lift here");
- background samples: target = 0 with a small weight, random points of the DEM at random
                      instants of the flight ("pseudo-absences": nobody ever flew there).

Weather of a flight: read from `<track>.weather.json` (a serialised `Weather`) when the
file exists, otherwise estimated from the track (`weather_from_track`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..domain import Terrain
from ..igc import IgcTrack, ThermalConfig, detect_thermals, estimate_wind, read_igc
from ..igc.thermals import vertical_speed
from ..lift import features as ft
from ..weather import Weather


@dataclass
class DatasetConfig:
    w_ref_ms: float = 5.0  # vertical air velocity mapped to a lift index of 1
    sink_ms: float = 1.25  # still-air sink rate of the glider
    thermal_sample_s: float = 60.0  # one sample per minute of climb
    glide_sample_s: float = 4.0
    glide_agl_min_m: float = 30.0  # below: on the ground / landing
    glide_agl_max_m: float = 400.0  # above: the glide says nothing about the ground below
    vario_window_s: float = 20.0  # averaging window of the vertical speed in glides
    bg_ratio: float = 0.5  # background samples per thermal sample
    bg_weight: float = 0.03
    seed: int = 0


@dataclass
class ThermalDataset:
    X: np.ndarray  # (n, n_features)
    y: np.ndarray  # (n,) target lift index in [0, 1]
    w: np.ndarray  # (n,) sample weight
    group: np.ndarray  # (n,) flight index (used to split train / validation by flight)

    def __len__(self) -> int:
        return len(self.y)


def weather_from_track(track: IgcTrack, thermals: list) -> Weather:
    """Fallback weather of a flight without sidecar file, estimated from the track itself.

    Wind: mean drift of the thermals (same at every level, no breeze added).
    Ceiling: highest altitude reached. Cloud cover and instability: defaults.
    """
    wind_from, wind_ms = estimate_wind(thermals)
    top = float(np.percentile(track.alt, 99))
    return Weather(
        wind_dir_deg=wind_from,
        wind_speed_kmh=wind_ms * 3.6,
        breeze_kmh=0.0,
        hourly={"ceiling_m": [[0.0, 24.0], [top, top]]},
    )


def samples_from_track(
    track: IgcTrack,
    terrain: Terrain,
    stack: np.ndarray,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    thermal_cfg: ThermalConfig | None = None,
    weather: Weather | None = None,
):
    """Return (X, y, w) for one flight, or None if it has no usable thermal on the DEM."""
    x, y = terrain.lonlat_to_xy(track.lon, track.lat)
    thermals = [th for th in detect_thermals(track.t, x, y, track.alt, thermal_cfg) if terrain.contains(th.x, th.y)]
    if not thermals:
        return None
    if weather is None:
        weather = weather_from_track(track, thermals)

    # 1. thermal samples (same location, successive instants of the climb)
    px, py, pt, target = [], [], [], []
    for th in thermals:
        n = max(1, int(round(th.duration_s / cfg.thermal_sample_s)))
        times = th.t0 + (np.arange(n) + 0.5) * th.duration_s / n
        px += [th.x] * n
        py += [th.y] * n
        pt += list(times)
        target += [float(np.clip((th.climb_ms + cfg.sink_ms) / cfg.w_ref_ms, 0, 1))] * n
    n_pos = len(pt)
    weight = [1.0] * n_pos

    # 2. low glide samples
    in_thermal = np.zeros(len(track), dtype=bool)
    for th in thermals:
        in_thermal[th.i0 : th.i1 + 1] = True
    agl = track.alt - terrain.sample(terrain.z, x, y)
    ok = ~in_thermal & terrain.contains(x, y) & (agl > cfg.glide_agl_min_m) & (agl < cfg.glide_agl_max_m)
    netto = vertical_speed(track.t, track.alt, cfg.vario_window_s) + cfg.sink_ms
    step = max(1, int(round(cfg.glide_sample_s / np.median(np.diff(track.t)))))
    idx = np.flatnonzero(ok)[::step]
    px += list(x[idx])
    py += list(y[idx])
    pt += list(track.t[idx])
    target += list(np.clip(netto[idx] / cfg.w_ref_ms, 0, 1))
    weight += [1.0] * len(idx)

    # 3. background samples
    n_bg = int(cfg.bg_ratio * n_pos)
    px += list(rng.uniform(0, terrain.width, n_bg))
    py += list(rng.uniform(0, terrain.height, n_bg))
    pt += list(rng.uniform(thermals[0].t0, thermals[-1].t1, n_bg))
    target += [0.0] * n_bg
    weight += [cfg.bg_weight] * n_bg

    static = ft.sample_static(terrain, stack, np.array(px), np.array(py))
    X = ft.assemble(static, track.day_of_year, np.array(pt) / 3600.0, terrain.lat_mid, terrain.lon_mid, weather)
    return X, np.array(target, dtype=np.float32), np.array(weight, dtype=np.float32)


def build_dataset(terrain: Terrain, igc_paths: list, cfg: DatasetConfig | None = None) -> ThermalDataset:
    cfg = cfg or DatasetConfig()
    rng = np.random.default_rng(cfg.seed)
    stack = ft.static_stack(terrain)
    Xs, ys, ws, gs, skipped, n_sidecar = [], [], [], [], 0, 0
    for k, path in enumerate(sorted(Path(p) for p in igc_paths)):
        try:
            sidecar = path.with_suffix(".weather.json")
            weather = Weather.load(str(sidecar)) if sidecar.exists() else None
            n_sidecar += weather is not None
            res = samples_from_track(read_igc(str(path)), terrain, stack, cfg, rng, weather=weather)
        except ValueError as exc:
            print(f"[dataset] skipped: {exc}")
            res = None
        if res is None:
            skipped += 1
            continue
        Xs.append(res[0])
        ys.append(res[1])
        ws.append(res[2])
        gs.append(np.full(len(res[1]), k))
    if not Xs:
        raise ValueError("No usable flight: check that the IGC tracks overlap the DEM.")
    ds = ThermalDataset(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(gs))
    print(
        f"[dataset] {len(Xs)} flights ({skipped} skipped, {n_sidecar} with a weather file), {len(ds)} samples, "
        f"mean target {np.average(ds.y, weights=ds.w):.2f}"
    )
    return ds
