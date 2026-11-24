"""Flight package: everything a navigation app needs to fly the computed routes.

The package is one JSON document, described by `schema/flight-package.v1.schema.json`
and documented in `docs/flight-package.md`:

    format, version    contract identifier ("cheminee.flight-package", 1)
    meta               generator, creation time, day, time zone, lift model, attributions
    extent             metric size of the map, lon / lat bounds
    terrain            map image (tint + hillshade + contours), JPEG data URI
    dem                down-sampled elevation grid (int16 little endian, base64)
    lift               hourly lift-index frames (grayscale PNG data URIs)
    polar              glider speeds used by the planner
    nodes              thermal nodes: position, ground altitude, slope aspect, climb rate every 15 min
    routes             candidate flights cut into "climb" and "glide" segments, with speed to fly
    weather            half-hourly summary: ceiling, cloud base, wind at two levels
    graph              glides feasible from the ceiling, per hourly slot (for in-flight rerouting)

All positions are in the local metric frame of the DEM: x east, y north, metres, origin at
the south-west corner. Times are local hours (decimal), altitudes metres above sea level.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .. import __version__
from ..config import Params
from ..domain import Route, Terrain
from ..planning import Planner, RouteOptimizer

FORMAT = "cheminee.flight-package"
FORMAT_VERSION = 1
ASPECTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
KIND_LABELS = {"free": "Free distance", "triangle": "Flat triangle", "fai": "FAI triangle", "goal": "Goal"}
COPERNICUS_DEM_NOTICE = (
    "Relief produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
    "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved"
)
OPEN_METEO_NOTICE = "Weather data by Open-Meteo.com (CC BY 4.0), from Météo-France models, processed by cheminee"
MIN_GLIDE_M = 300.0  # shorter glides between two neighbouring thermals are folded into one climb


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def _data_uri(img: Image.Image, fmt: str, **kw) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    mime = "jpeg" if fmt == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def render_map(terrain: Terrain, m_per_px: float = 12.0) -> Image.Image:
    """Light topographic map: hypsometric tint x hillshade, contours every 100 m (500 m bolder)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = int(round(terrain.width / m_per_px)), int(round(terrain.height / m_per_px))
    z = ndimage.zoom(terrain.z, (h / terrain.ny, w / terrain.nx), order=3)
    gy, gx = np.gradient(z, m_per_px)
    norm = np.sqrt(gx**2 + gy**2 + 1.0)
    shade = np.zeros_like(z)
    for az, weight in ((315.0, 0.65), (0.0, 0.2), (270.0, 0.15)):  # soft multi-directional light
        a, el = np.radians(az), np.radians(45.0)
        light = np.array([np.sin(a) * np.cos(el), np.cos(a) * np.cos(el), np.sin(el)])
        shade += weight * np.clip((-gx * light[0] + gy * light[1] + light[2]) / norm, 0, 1)
    stops = np.array([300, 700, 1100, 1500, 1900, 2400, 3200], dtype=float)
    colors = (
        np.array(
            [
                [214, 230, 205],
                [222, 234, 207],
                [234, 234, 210],
                [240, 231, 212],
                [244, 238, 228],
                [246, 244, 240],
                [252, 252, 252],
            ],
            dtype=float,
        )
        / 255
    )
    tint = np.stack([np.interp(z, stops, colors[:, c]) for c in range(3)], axis=-1)
    rgb = np.clip(tint * (0.58 + 0.52 * shade)[..., None], 0, 1)

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="nearest")
    lo, hi = int(z.min() // 100 * 100), int(z.max() // 100 * 100 + 100)
    ax.contour(z, levels=[v for v in range(lo, hi, 100) if v % 500], colors=[(0.42, 0.38, 0.30, 0.22)], linewidths=0.35)
    ax.contour(z, levels=list(range(lo - lo % 500, hi, 500)), colors=[(0.42, 0.38, 0.30, 0.45)], linewidths=0.6)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return Image.open(buf).convert("RGB")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def _aspect(terrain: Terrain, x: float, y: float) -> str:
    """Compass direction the slope faces (direction of steepest descent), or "flat"."""
    nx, ny = (float(terrain.sample(terrain.normal[k], x, y)) for k in (0, 1))
    if np.hypot(nx, ny) < 0.05:
        return "flat"
    return ASPECTS[int(((np.degrees(np.arctan2(nx, ny)) % 360) + 22.5) // 45) % 8]


def _glide_airspeed(pl: Planner, a: int, b: int, t0: float, loss: float) -> float:
    """Airspeed (km/h) whose wind triangle and polar sink give the planned height loss."""
    p, slot = pl.p, pl.slots[pl.slot_of(pl.bin_floor(t0))]
    d = pl.dist[a, b]
    e = np.array([pl.x[b] - pl.x[a], pl.y[b] - pl.y[a]]) / max(d, 1.0)
    w_par, w_perp = float(e @ slot.wind), float(e[0] * slot.wind[1] - e[1] * slot.wind[0])
    best, best_err = p.polar.trim_kmh, np.inf
    for v in p.polar.speeds:
        root = v**2 - w_perp**2
        if root <= 0:
            continue
        vg = w_par + np.sqrt(root)
        if vg <= 1.0:
            continue
        err = abs(d * float(p.polar.sink(v)) / vg - loss)
        if err < best_err:
            best, best_err = v * 3.6, err
    return float(best)


def route_segments(pl: Planner, route: Route) -> list[dict]:
    """Cut the altitude profile of a route into climb and glide segments."""
    prof = route.profile
    node_of = [int(np.argmin(np.hypot(pl.x - x, pl.y - y))) for x, y in prof[:, :2]]
    segs = []
    for k in range(len(prof) - 1):
        (x0, y0, a0, t0), (x1, y1, a1, t1) = (map(float, prof[k]), map(float, prof[k + 1]))
        n0, n1 = node_of[k], node_of[k + 1]
        if t1 <= t0 + 1e-6:
            continue
        if n0 == n1:
            if a1 > a0 + 1.0:
                segs.append(
                    dict(type="climb", node=n0, t0=round(t0, 5), t1=round(t1, 5), a0=round(a0), a1=round(a1), rate=0.0)
                )
            continue
        dist = float(np.hypot(x1 - x0, y1 - y0))
        dt_s = (t1 - t0) * 3600.0
        segs.append(
            dict(
                type="glide",
                src=n0,
                dst=n1,
                t0=round(t0, 5),
                t1=round(t1, 5),
                a0=round(a0),
                a1=round(a1),
                dist=round(dist),
                gs_kmh=round(dist / dt_s * 3.6, 1),
                sink_ms=round((a0 - a1) / dt_s, 2),
                airspeed_kmh=round(_glide_airspeed(pl, n0, n1, t0, a0 - a1)),
                margin_m=round(a1 - float(pl.zg[n1])),
            )
        )
    return merge_short_glides(segs)


def merge_short_glides(segs: list[dict], min_dist_m: float = MIN_GLIDE_M) -> list[dict]:
    """Fold glides shorter than `min_dist_m` (hop between two neighbouring thermals) into one climb.

    Climb rates are recomputed from the merged altitudes and times.
    """
    out: list[dict] = []
    for seg in segs:
        if seg["type"] == "glide" and seg["dist"] < min_dist_m:
            start = out.pop() if out and out[-1]["type"] == "climb" else dict(t0=seg["t0"], a0=seg["a0"])
            out.append(
                dict(
                    type="climb",
                    node=seg["dst"],
                    t0=start["t0"],
                    t1=seg["t1"],
                    a0=start["a0"],
                    a1=seg["a1"],
                    rate=0.0,
                    _open=True,
                )
            )
            continue
        if seg["type"] == "climb" and out and out[-1].get("_open") and out[-1]["node"] == seg["node"]:
            prev = out.pop()
            seg = dict(seg, t0=prev["t0"], a0=prev["a0"])
        out.append(seg)
    for seg in out:
        seg.pop("_open", None)
        if seg["type"] == "climb":
            seg["rate"] = round((seg["a1"] - seg["a0"]) / max((seg["t1"] - seg["t0"]) * 3600.0, 1.0), 2)
    return out


def route_dict(pl: Planner, route: Route) -> dict:
    tp_nodes = [int(route.nodes[k]) for k in route.turnpoints]
    if route.kind in ("triangle", "fai"):
        scoring = tp_nodes + tp_nodes[:1]
        labels = {n: f"B{i + 1}" for i, n in enumerate(tp_nodes)}
    else:  # free distance: start, P1, P2, P3, end
        scoring = tp_nodes
        labels = {n: f"P{i}" for i, n in enumerate(tp_nodes) if 0 < i < len(tp_nodes) - 1}
    segs = route_segments(pl, route)
    return dict(
        kind=route.kind,
        label=KIND_LABELS.get(route.kind, route.kind),
        distance_km=round(route.distance_km, 1),
        points=round(route.points, 1),
        t_start=segs[0]["t0"],
        t_end=segs[-1]["t1"],
        nodes=[int(n) for n in route.nodes],
        scoring=scoring,
        turnpoints={str(k): v for k, v in labels.items()},
        segments=segs,
    )


def candidate_routes(planner: Planner, n_triangles: int = 3, n_free: int = 1) -> list[Route]:
    """A few diverse candidates: the best triangles and the best free distances, by points."""
    routes = RouteOptimizer(planner).triangles()[:n_triangles] + planner.search()[:n_free]
    return sorted(routes, key=lambda r: -r.points)


# --------------------------------------------------------------------------- #
# Routing graph
# --------------------------------------------------------------------------- #
def routing_graph(pl: Planner) -> dict:
    """Glides feasible from the ceiling, for each hourly slot, at three airspeeds.

    Edge rows are `[i, j, glide_s_1, loss_m_1, glide_s_2, loss_m_2, glide_s_3, loss_m_3]`
    for the airspeeds of `speeds_kmh`; -1 marks an airspeed that does not clear the terrain.
    `takeoff_edges` rows are `[j, glide_s, loss_m]` at trim speed, starting from the takeoff
    altitude without climbing.
    """
    p = pl.p
    speeds = p.polar.speeds
    trim = p.polar.trim_index
    picks = sorted({trim, int(np.abs(speeds - 0.5 * (speeds[trim] + speeds[-1])).argmin()), len(speeds) - 1})
    slots = []
    for k, slot in enumerate(pl.slots):
        edges = []
        for i in range(pl.n):
            for j in slot.nbrs[i]:
                row, ok = [i, int(j)], False
                for vi in picks:
                    if slot.feas[vi, i, j]:
                        row += [round(float(slot.t_glide[vi, i, j])), round(float(slot.loss[vi, i, j]))]
                        ok = True
                    else:
                        row += [-1, -1]
                if ok:
                    edges.append(row)
        wind = slot.wind
        slots.append(
            dict(
                t0=round(float(pl.p.t_start + k), 3),
                t1=round(float(min(pl.p.t_start + k + 1, pl.t_end)), 3),
                ceiling_m=round(slot.ceil),
                wind_kmh=round(float(np.hypot(*wind) * 3.6), 1),
                wind_dir_deg=round(float(np.degrees(np.arctan2(-wind[0], -wind[1])) % 360)),
                edges=edges,
                takeoff_edges=[
                    [int(j), round(float(slot.low0_t[j])), round(float(slot.low0_loss[j]))] for j in slot.low0_nbrs
                ],
            )
        )
    return dict(
        speeds_kmh=[round(float(speeds[i] * 3.6), 1) for i in picks],
        circling_sink_ms=p.polar.circling_sink_ms,
        arrival_margin_m=p.arrival_margin_m,
        clearance_m=p.clearance_m,
        slots=slots,
    )


# --------------------------------------------------------------------------- #
# Package
# --------------------------------------------------------------------------- #
def build_flight_package(
    terrain: Terrain,
    params: Params,
    planner: Planner,
    routes: list[Route],
    attribution: list[str] | None = None,
    date: dt.date | None = None,
    with_graph: bool = True,
) -> dict:
    """Assemble the flight package (a JSON-serialisable dict)."""
    p, w = params, params.weather
    terrain_img = render_map(terrain)
    dem = np.round(terrain.z[::2, ::2]).astype("<i2")
    hours = list(np.arange(np.floor(p.t_start), np.ceil(planner.t_end) + 1e-6, 1.0))
    frames = []
    for hour in hours:
        f = planner.lift.field(float(hour))[::3, ::3]
        frames.append(
            _data_uri(Image.fromarray(np.round(np.clip(f, 0, 1) * 255).astype(np.uint8), "L"), "PNG", optimize=True)
        )
    every = max(1, int(round(15.0 / p.dt_min)))
    nodes = [
        dict(
            x=round(float(planner.x[i]), 1),
            y=round(float(planner.y[i]), 1),
            z=round(float(planner.zg[i])),
            aspect=_aspect(terrain, planner.x[i], planner.y[i]),
            climb=[round(float(c), 1) for c in planner.climb[::every, i]],
        )
        for i in range(planner.n)
    ]
    weather = []
    for hour in np.arange(p.t_start, planner.t_end + 1e-6, 0.5):
        s = planner.solar(float(hour))
        low, high = w.wind_vec(w.wind_alt_m, s), w.wind_vec(w.wind_aloft_alt_m, s)
        weather.append(
            dict(
                t=round(float(hour), 2),
                ceiling=round(w.ceiling(s)),
                base=round(w.cloud_base(s)),
                wind=[round(float(np.hypot(*low) * 3.6)), round(float(np.degrees(np.arctan2(-low[0], -low[1])) % 360))],
                wind_aloft=[
                    round(float(np.hypot(*high) * 3.6)),
                    round(float(np.degrees(np.arctan2(-high[0], -high[1])) % 360)),
                ],
            )
        )
    lon0, lat0 = terrain.xy_to_lonlat(0.0, 0.0)
    lon1, lat1 = terrain.xy_to_lonlat(terrain.width, terrain.height)
    package = dict(
        format=FORMAT,
        version=FORMAT_VERSION,
        meta=dict(
            generator=f"cheminee {__version__}",
            created=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            date=date.isoformat() if date else None,
            day_of_year=p.day_of_year,
            utc_offset=p.utc_offset,
            lift_model=planner.lift.name,
            attribution=list(attribution) if attribution is not None else [COPERNICUS_DEM_NOTICE],
        ),
        extent=dict(
            width=round(terrain.width, 1),
            height=round(terrain.height, 1),
            lon=[round(float(lon0), 5), round(float(lon1), 5)],
            lat=[round(float(lat0), 5), round(float(lat1), 5)],
        ),
        terrain=dict(image=_data_uri(terrain_img, "JPEG", quality=80, optimize=True), px=list(terrain_img.size)),
        dem=dict(nx=int(dem.shape[1]), ny=int(dem.shape[0]), data=base64.b64encode(dem.tobytes()).decode()),
        lift=dict(hours=[float(h) for h in hours], frames=frames),
        polar=dict(trim_kmh=p.polar.trim_kmh, min_kmh=p.polar.points[0][0], max_kmh=p.polar.points[-1][0]),
        takeoff=0,
        nodes=nodes,
        climb_times=[round(float(h), 4) for h in planner.hours[::every]],
        routes=[route_dict(planner, r) for r in routes],
        weather=weather,
    )
    if with_graph:
        package["graph"] = routing_graph(planner)
    return package


def write_flight_package(package: dict, path: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(
        f"[export] flight package written to {out} ({out.stat().st_size / 1e6:.2f} MB, {len(package['routes'])} routes)"
    )
    return out
