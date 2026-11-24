"""Synthetic IGC generator, used to exercise the training pipeline without real tracks.

Flights are simulated on top of a HIDDEN lift model (`TRUTH_CONFIG`) that deliberately
differs from the default heuristic (heating delay, slope optimum, wind sensitivity,
seasonal and altitude effects), under a random weather written next to each IGC file
(`<name>.weather.json`). A network trained on these tracks should recover the
hidden model better than the default heuristic does: this is a self-consistency test
of the pipeline, NOT a validation against real weather.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import KMH, Params
from ..domain import Terrain
from ..lift import HeuristicLift, HeuristicLiftConfig
from ..planning import Planner
from ..weather import Weather
from .parser import IgcTrack, write_igc

TRUTH_CONFIG = HeuristicLiftConfig(
    thermal_lag_h=1.7, slope_lo_deg=22.0, wind_gain=2.2, seasonal_exp=1.5, alt_opt_m=1500.0, alt_width_m=450.0
)


@dataclass
class SyntheticConfig:
    year: int = 2025
    day_min: int = 100  # ~10 April
    day_max: int = 270  # ~27 September
    wind_max_kmh: float = 22.0
    vary_weather: bool = True  # False: default weather, only the low-level wind changes
    utc_offset: float = 2.0
    fix_interval_s: float = 3.0
    lift_agl_full_m: float = 400.0  # below this height the glider feels the full lift
    lift_agl_zero_m: float = 700.0  # above this height a glide feels no terrain lift
    n_nodes: int = 150  # denser than the planner default: more varied tracks
    node_spacing_m: float = 800.0


def random_weather(rng: np.random.Generator, cfg: SyntheticConfig) -> Weather:
    """Draw the weather of one synthetic flying day."""
    wind_dir, wind_speed = float(rng.uniform(0, 360)), float(rng.uniform(0, cfg.wind_max_kmh))
    if not cfg.vary_weather:
        return Weather(wind_dir_deg=wind_dir, wind_speed_kmh=wind_speed, breeze_kmh=0.0)
    temp = float(rng.uniform(22, 31))
    return Weather(
        wind_dir_deg=wind_dir,
        wind_speed_kmh=wind_speed,
        wind_aloft_dir_deg=float((wind_dir + rng.normal(0, 25)) % 360),
        wind_aloft_speed_kmh=float(wind_speed * rng.uniform(1.0, 1.8)),
        temp_c=temp,
        dewpoint_c=float(temp - rng.uniform(12, 19)),
        inversion_alt_m=float(rng.uniform(2500, 3800)),
        cloud_cover=float(rng.uniform(0, 0.75)),
        thermal_strength=float(rng.uniform(0.75, 1.2)),
        breeze_kmh=float(rng.uniform(0, 12)),
    )


class _Flight:
    """Mutable state of one simulated flight."""

    def __init__(self, terrain: Terrain, params: Params, planner: Planner, rng, cfg: SyntheticConfig):
        self.t, self.p, self.pl, self.rng, self.cfg = terrain, params, planner, rng, cfg
        self.pos = np.array([planner.x[0], planner.y[0]])
        self.alt = float(planner.zg[0])
        self.hour = params.t_start
        self.fixes = [(self.hour * 3600.0, *self.pos, self.alt)]
        self.landed = False

    def _push(self, secs, xs, ys, alts):
        self.fixes.extend(zip(self.hour * 3600.0 + secs, xs, ys, alts))
        self.pos, self.alt = np.array([xs[-1], ys[-1]]), float(alts[-1])
        self.hour += secs[-1] / 3600.0

    def _wind(self) -> np.ndarray:
        return self.p.weather.wind_vec(self.alt, self.pl.solar(self.hour))

    def climb(self, node: int) -> bool:
        """Circle in the thermal of `node` up to the ceiling. False if there is no usable lift."""
        p, cfg, rng, pl = self.p, self.cfg, self.rng, self.pl
        b = pl.bin_floor(self.hour)
        lift = pl.node_lift[min(b, pl.n_bins), node]
        # climb rate = vertical air velocity - sink rate of the circling glider
        rate = p.w_ref_ms * lift * rng.lognormal(0, 0.12) - p.polar.circling_sink_ms
        if rate < p.min_climb_ms:
            return False
        top = pl.slots[pl.slot_of(b)].ceil - rng.uniform(0, 150)
        if self.alt >= top - 30:
            return True
        n = max(2, int((top - self.alt) / rate / cfg.fix_interval_s))
        secs = np.arange(1, n + 1) * cfg.fix_interval_s
        radius, phase = rng.uniform(40, 70), rng.uniform(0, 2 * np.pi)
        omega = rng.choice([-1, 1]) * p.polar.trim_kmh * KMH / radius
        centre = self.pos - radius * np.array([np.cos(phase), np.sin(phase)])
        w = self._wind()  # the thermal column drifts with the wind
        xs = centre[0] + w[0] * secs + radius * np.cos(phase + omega * secs)
        ys = centre[1] + w[1] * secs + radius * np.sin(phase + omega * secs)
        if not self.t.contains(xs, ys).all():
            self.landed = True
            return False
        self._push(secs, xs, ys, self.alt + rate * secs)
        return True

    def glide(self, target: np.ndarray, max_s: float = 3600.0) -> None:
        """Straight glide at trim speed toward `target`; ends early on ground contact."""
        p, cfg = self.p, self.cfg
        d = target - self.pos
        dist = float(np.hypot(*d))
        e = d / max(dist, 1e-6)
        w, va = self._wind(), p.polar.trim_kmh * KMH
        w_par, w_perp = float(w @ e), float(e[0] * w[1] - e[1] * w[0])
        vg = w_par + np.sqrt(max(va**2 - w_perp**2, 0.0))
        if vg < 1.0:
            self.landed = True
            return
        sink = float(p.polar.sink(va)) * self.rng.uniform(0.9, 1.15)
        n = max(1, int(min(dist / vg, max_s) / cfg.fix_interval_s))
        secs = np.arange(1, n + 1) * cfg.fix_interval_s
        xs, ys = self.pos[0] + e[0] * vg * secs, self.pos[1] + e[1] * vg * secs
        inside = self.t.contains(xs, ys)
        ground = self.t.sample(self.t.z, xs, ys)
        # Lift felt while gliding close to the terrain (the pilot does not stop to climb).
        agl = self.alt - sink * secs - ground
        felt = np.clip((cfg.lift_agl_zero_m - agl) / (cfg.lift_agl_zero_m - cfg.lift_agl_full_m), 0, 1)
        air = p.w_ref_ms * self.pl.lift_at(xs, ys, self.hour) * felt
        alts = self.alt - np.cumsum(sink - air) * cfg.fix_interval_s
        hit = np.flatnonzero((alts < ground + 15.0) | ~inside)
        if len(hit):
            k = max(int(hit[0]), 1)
            secs, xs, ys, alts = secs[:k], xs[:k], ys[:k], alts[:k]
            self.landed = True
        self._push(secs, xs, ys, alts)

    def land(self) -> None:
        heading = self.rng.uniform(0, 2 * np.pi)
        while not self.landed:
            self.glide(self.pos + 4000.0 * np.array([np.sin(heading), np.cos(heading)]), max_s=300.0)


def simulate_flight(
    terrain: Terrain, rng: np.random.Generator, cfg: SyntheticConfig | None = None
) -> tuple[IgcTrack, Params] | None:
    """Simulate one flight. Returns (track, true conditions) or None if no takeoff was found."""
    cfg = cfg or SyntheticConfig()
    p = Params(
        day_of_year=int(rng.integers(cfg.day_min, cfg.day_max)),
        utc_offset=cfg.utc_offset,
        t_start=float(rng.uniform(10.5, 13.5)),
        t_end=18.5,
        weather=random_weather(rng, cfg),
        n_nodes=cfg.n_nodes,
        node_spacing_m=cfg.node_spacing_m,
    )
    truth = HeuristicLift(terrain, p, TRUTH_CONFIG)

    # Takeoff: random high ground cell where the (hidden) lift is good shortly after takeoff time.
    f = truth.field(p.t_start + 0.5)
    ceiling = p.weather.ceiling(p.to_solar(p.t_start + 0.5, terrain.lon_mid))
    ok = (f >= p.lift_min + 0.1) & (terrain.z > np.percentile(terrain.z, 60)) & (terrain.z < ceiling - 400)
    rows, cols = np.nonzero(ok)
    if len(rows) == 0:
        return None
    k = rng.integers(len(rows))
    takeoff = tuple(float(v) for v in terrain.rowcol_to_xy(rows[k], cols[k]))
    pl = Planner(terrain, p, takeoff, truth, verbose=False)

    fl = _Flight(terrain, p, pl, rng, cfg)
    node, visited = 0, [0]
    while not fl.landed and fl.hour < pl.t_end:
        first = len(visited) == 1
        climbed = fl.climb(node)
        b = pl.bin_floor(fl.hour)
        if fl.landed or b >= pl.n_bins:
            break
        slot = pl.slots[pl.slot_of(b)]
        if climbed:  # at the ceiling: every glide of the graph is available
            nbrs = slot.nbrs[node]
        elif first:  # no house thermal: direct glide from the takeoff altitude
            nbrs = slot.low0_nbrs
        else:
            break
        nbrs = nbrs[pl.climb[b, nbrs] >= p.min_climb_ms]
        usable = [int(j) for j in nbrs if j not in visited[-3:]]
        if not usable:
            break
        weights = pl.node_lift[b, usable] ** 2
        node = usable[rng.choice(len(usable), p=weights / weights.sum())]
        visited.append(node)
        fl.glide(np.array([pl.x[node], pl.y[node]]))
        if fl.alt < pl.zg[node] + 30.0:
            break
    fl.land()

    a = np.array(fl.fixes)
    if len(a) < 100:
        return None
    x = a[:, 1] + rng.normal(0, 1.5, len(a))  # GPS noise
    y = a[:, 2] + rng.normal(0, 1.5, len(a))
    lon, lat = terrain.xy_to_lonlat(x, y)
    date = dt.date(cfg.year, 1, 1) + dt.timedelta(days=p.day_of_year - 1)
    track = IgcTrack(date, a[:, 0] - cfg.utc_offset * 3600.0, lat, lon, a[:, 3] + rng.normal(0, 1.0, len(a)))
    return track, p


def generate(
    terrain: Terrain, out_dir: str, n_flights: int, seed: int = 0, cfg: SyntheticConfig | None = None
) -> list[Path]:
    """Write `n_flights` synthetic IGC files, their weather sidecars and a manifest."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths, manifest = [], ["file,day_of_year,wind_dir_deg,wind_speed_kmh,cloud_cover,thermal_strength"]
    attempts = 0
    while len(paths) < n_flights and attempts < 5 * n_flights:
        attempts += 1
        res = simulate_flight(terrain, rng, cfg)
        if res is None:
            continue
        track, params = res
        path = out / f"synthetic_{len(paths):04d}.igc"
        write_igc(str(path), track)
        params.weather.save(str(path.with_suffix(".weather.json")))
        paths.append(path)
        w = params.weather
        manifest.append(
            f"{path.name},{params.day_of_year},{w.wind_dir_deg:.1f},{w.wind_speed_kmh:.1f},"
            f"{w.cloud_cover:.2f},{w.thermal_strength:.2f}"
        )
        if len(paths) % 20 == 0:
            print(f"[synth] {len(paths)}/{n_flights} flights written")
    (out / "manifest.csv").write_text("\n".join(manifest) + "\n")
    print(f"[synth] {len(paths)} IGC files in {out}")
    return paths
