"""Build the supervised dataset of the thermal network from IGC tracks.

The target is a lift index in [0, 1] derived from the vertical velocity of the AIR:

    netto = vertical speed of the glider + still-air sink rate of the glider
    target = clip(netto / w_ref, 0, 1)

Three kinds of samples are extracted from each flight:
- thermal samples   : netto from the mean climb rate, located at the ENTRY point of the
                      thermal (proxy of its trigger on the ground, before wind drift);
- low glide samples : netto measured while gliding close to the terrain, located under
                      the pilot (graded information, including "no lift here");
- background samples: target = 0 with a small weight ("pseudo-absences": the pilot did not climb
                      there), random points of the DEM at random instants of the flight, or
                      (`DatasetConfig.background = "near_track"`) random points within 3 km of
                      where the pilot was, away from the thermals of the flight.

Weather of a flight: read from `<track>.weather.json` when the file exists, otherwise
estimated from the track (`weather_from_track`). The file holds one `Weather`, or the weather
at three points of the flight (`cheminee fetch-weather`): each sample then takes the weather
of the nearest point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..domain import Terrain
from ..igc import IgcTrack, ThermalConfig, detect_thermals, estimate_wind, read_igc
from ..igc.thermals import vertical_speed
from ..lift import features as ft
from ..weather import Weather, WeatherPoint, load_points
from ..weather.points import nearest_point


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
    # where the background pseudo-absences are drawn: "map" (anywhere on the DEM, at a random time of the flight)
    # or "near_track" (within bg_radius_m of where the pilot was at that time, at least bg_min_dist_m from the
    # thermals of the flight): the network then learns to tell a thermal from the ground around it
    background: str = "map"
    bg_radius_m: float = 3000.0
    bg_min_dist_m: float = 400.0
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


THERMAL, GLIDE, BACKGROUND = 0, 1, 2  # kinds of samples


@dataclass
class FlightSamples:
    """Where and when the samples of one flight are taken, with their target (before the features)."""

    x: np.ndarray  # (n,) metres, local frame of the terrain
    y: np.ndarray
    t_s: np.ndarray  # (n,) seconds UTC of the flight day
    target: np.ndarray  # (n,) lift index in [0, 1]
    weight: np.ndarray  # (n,)
    kind: np.ndarray  # (n,) THERMAL, GLIDE or BACKGROUND
    thermal: np.ndarray  # (n,) index of the thermal of a THERMAL sample, -1 otherwise
    weather: Weather  # weather of the flight (first point, or estimated from the track)
    points: list  # weather points with a position (more than one: each sample takes the nearest)


def flight_samples(
    track: IgcTrack,
    terrain: Terrain,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    thermal_cfg: ThermalConfig | None = None,
    weather: Weather | list[WeatherPoint] | None = None,
) -> FlightSamples | None:
    """Samples of one flight, or None if it has no usable thermal on the DEM."""
    x, y = terrain.lonlat_to_xy(track.lon, track.lat)
    thermals = [th for th in detect_thermals(track.t, x, y, track.alt, thermal_cfg) if terrain.contains(th.x, th.y)]
    if not thermals:
        return None
    points = [q for q in weather if q.lon is not None] if isinstance(weather, list) else []
    if isinstance(weather, list):
        weather = weather[0].weather if weather else None
    if weather is None:
        weather = weather_from_track(track, thermals)

    # 1. thermal samples (same location, successive instants of the climb)
    px, py, pt, target, tid = [], [], [], [], []
    for k, th in enumerate(thermals):
        n = max(1, int(round(th.duration_s / cfg.thermal_sample_s)))
        times = th.t0 + (np.arange(n) + 0.5) * th.duration_s / n
        px += [th.x] * n
        py += [th.y] * n
        pt += list(times)
        target += [float(np.clip((th.climb_ms + cfg.sink_ms) / cfg.w_ref_ms, 0, 1))] * n
        tid += [k] * n
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
    if cfg.background == "near_track":
        bx, by, bt = _near_track(x, y, track.t, thermals, terrain, cfg, rng, n_bg)
    elif cfg.background == "map":
        bx, by = rng.uniform(0, terrain.width, n_bg), rng.uniform(0, terrain.height, n_bg)
        bt = rng.uniform(thermals[0].t0, thermals[-1].t1, n_bg)
    else:
        raise ValueError(f"background: 'map' or 'near_track', got {cfg.background!r}")
    n_bg = len(bx)
    px += list(bx)
    py += list(by)
    pt += list(bt)
    target += [0.0] * n_bg
    weight += [cfg.bg_weight] * n_bg
    kind = np.concatenate([np.full(n_pos, THERMAL), np.full(len(idx), GLIDE), np.full(n_bg, BACKGROUND)])
    return FlightSamples(
        np.array(px, dtype=float),
        np.array(py, dtype=float),
        np.array(pt, dtype=float),
        np.array(target, dtype=np.float32),
        np.array(weight, dtype=np.float32),
        kind,
        np.concatenate([np.array(tid, dtype=int), np.full(len(idx) + n_bg, -1)]),
        weather,
        points,
    )


def _near_track(x, y, t, thermals, terrain: Terrain, cfg: DatasetConfig, rng: np.random.Generator, n: int):
    """n pseudo-absences within cfg.bg_radius_m of the pilot (uniform in the disc), at the time of a random fix
    between the first and the last thermal, on the DEM and at least cfg.bg_min_dist_m from every thermal."""
    fixes = np.flatnonzero(terrain.contains(x, y) & (t >= thermals[0].t0) & (t <= thermals[-1].t1))
    if n <= 0 or not len(fixes):
        return np.empty(0), np.empty(0), np.empty(0)
    tx, ty = np.array([th.x for th in thermals]), np.array([th.y for th in thermals])
    out_x, out_y, out_t = [], [], []
    for _ in range(4):  # a few rounds of candidates: some fall off the DEM or on a thermal
        i = rng.choice(fixes, 2 * n)
        r = cfg.bg_radius_m * np.sqrt(rng.uniform(0, 1, 2 * n))
        a = rng.uniform(0, 2 * np.pi, 2 * n)
        cx, cy = x[i] + r * np.sin(a), y[i] + r * np.cos(a)
        far = np.hypot(cx[:, None] - tx[None], cy[:, None] - ty[None]).min(axis=1) >= cfg.bg_min_dist_m
        ok = terrain.contains(cx, cy) & far
        out_x += list(cx[ok])
        out_y += list(cy[ok])
        out_t += list(t[i][ok])
        if len(out_x) >= n:
            break
    return np.array(out_x[:n]), np.array(out_y[:n]), np.array(out_t[:n])


def nearest_weather(fs: FlightSamples, terrain: Terrain) -> np.ndarray:
    """Index of the weather point of each sample (all 0 without positioned points)."""
    if len(fs.points) <= 1:
        return np.zeros(len(fs.x), dtype=int)
    qx, qy = terrain.lonlat_to_xy(np.array([q.lon for q in fs.points]), np.array([q.lat for q in fs.points]))
    return nearest_point(qx, qy, fs.x, fs.y)


def samples_from_track(
    track: IgcTrack,
    terrain: Terrain,
    stack: np.ndarray,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    thermal_cfg: ThermalConfig | None = None,
    weather: Weather | list[WeatherPoint] | None = None,
):
    """Return (X, y, w) for one flight, or None if it has no usable thermal on the DEM. `weather` may be the
    weather at a few points of the flight: each sample then takes the weather of the nearest point."""
    fs = flight_samples(track, terrain, cfg, rng, thermal_cfg, weather)
    if fs is None:
        return None
    hours = fs.t_s / 3600.0
    static = ft.sample_static(terrain, stack, fs.x, fs.y)
    if len(fs.points) > 1:  # weather of the nearest point of the flight
        near = nearest_weather(fs, terrain)
        X = np.zeros((len(fs.x), len(ft.FEATURE_NAMES)), dtype=np.float32)
        for k, q in enumerate(fs.points):
            m = near == k
            if m.any():
                X[m] = ft.assemble(static[m], track.day_of_year, hours[m], terrain.lat_mid, terrain.lon_mid, q.weather)
    else:
        X = ft.assemble(static, track.day_of_year, hours, terrain.lat_mid, terrain.lon_mid, fs.weather)
    return X, fs.target, fs.weight


def iter_flights(terrain: Terrain, igc_paths: list, cfg: DatasetConfig, sample=None):
    """Read the flights in a fixed order and yield (flight index, path, track, result of `sample`) for the usable
    ones; `sample(track, rng, weather)` defaults to `flight_samples`. Returns the number of flights skipped and
    of weather files. Training and evaluation share it: same flights, same indices, same train/validation split."""
    rng = np.random.default_rng(cfg.seed)
    sample = sample or (lambda track, rng, weather: flight_samples(track, terrain, cfg, rng, weather=weather))
    skipped = n_sidecar = 0
    for k, path in enumerate(sorted(Path(p) for p in igc_paths)):
        try:
            sidecar = path.with_suffix(".weather.json")
            weather = load_points(str(sidecar)) if sidecar.exists() else None
            n_sidecar += weather is not None
            track = read_igc(str(path))
            res = sample(track, rng, weather)
        except ValueError as exc:
            print(f"[dataset] skipped: {exc}")
            res = None
        if res is None:
            skipped += 1
            continue
        yield k, path, track, res
    return skipped, n_sidecar


def build_dataset(terrain: Terrain, igc_paths: list, cfg: DatasetConfig | None = None) -> ThermalDataset:
    cfg = cfg or DatasetConfig()
    stack = ft.static_stack(terrain)
    Xs, ys, ws, gs = [], [], [], []

    def sample(track, rng, weather):
        return samples_from_track(track, terrain, stack, cfg, rng, weather=weather)

    flights = iter_flights(terrain, igc_paths, cfg, sample)
    while True:
        try:
            k, _, _, res = next(flights)
        except StopIteration as stop:
            skipped, n_sidecar = stop.value
            break
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
