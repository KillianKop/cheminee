"""Local web interface: a small JSON API over the planner and the page that drives it (`cheminee serve`).

Standard library only (`http.server`). The page draws the 3D terrain with the plotly.js bundled in the plotly
Python package, served from disk without importing plotly, so the app works offline except for the optional
Open-Meteo forecast. The server binds to 127.0.0.1: it is a local tool without authentication.

API (JSON):
    GET  /api/config        DEM grid for the 3D view, presets, weather fields, defaults
    POST /api/weather       resolved weather of a request, hour by hour
    POST /api/plan          start a planning job -> {"job": id}
    GET  /api/jobs/<id>     job status, log lines and, when done, routes, thermal nodes and lift frames
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import gzip
import importlib.util
import io
import json
import math
import os
import sys
import threading
import time
import traceback
import uuid
import zlib
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ..config import OBJECTIVES, Params
from ..domain import Terrain
from ..weather import PRESETS, Weather

STATIC_DIR = Path(__file__).with_name("static")
MAX_GRID = 300  # cells along the longest side of the 3D grid sent to the page
MAX_JOBS = 20  # finished jobs kept in memory
MAX_LIFT_CACHE = 3  # lift models (with their hourly fields, ~30 MB each on the sample DEM) kept in memory
MAX_RESULT_CACHE = 12  # planning results kept for identical requests
MAX_WAIT_S = 20.0  # longest wait of a long-polling request
ROUTE_COLORS = ["#e6194b", "#1f77b4", "#f58231", "#911eb4", "#17a589", "#7f8c00", "#c2185b", "#00838f"]

# Weather values the page can set: attribute -> (label, unit, min, max, step)
WEATHER_FIELDS = {
    "wind_dir_deg": ("Wind direction (from)", "°", 0, 360, 5),
    "wind_speed_kmh": ("Wind speed", "km/h", 0, 60, 1),
    "wind_aloft_dir_deg": ("Wind at 3000 m, direction", "°", 0, 360, 5),
    "wind_aloft_speed_kmh": ("Wind at 3000 m, speed", "km/h", 0, 90, 1),
    "temp_c": ("Afternoon temperature", "°C", -15, 42, 0.5),
    "dewpoint_c": ("Dew point", "°C", -25, 30, 0.5),
    "station_alt_m": ("Station altitude", "m", 0, 3500, 10),
    "inversion_alt_m": ("Top of convection", "m", 800, 6500, 50),
    "cloud_cover": ("Cloud cover", "0-1", 0, 1, 0.05),
    "thermal_strength": ("Instability", "×", 0.3, 1.6, 0.05),
    "breeze_kmh": ("Breeze", "km/h", 0, 30, 1),
    "snow_line_m": ("Snow line", "m", 0, 9999, 50),
    "overdev_hour": ("Overdevelopment (solar hour)", "h", 9, 21, 0.25),
}


class RequestError(ValueError):
    """Invalid request: answered with HTTP 400 and the message."""


def _plotly_js() -> Path | None:
    """plotly.min.js shipped with the plotly Python package (located without importing plotly)."""
    spec = importlib.util.find_spec("plotly")
    for root in (spec.submodule_search_locations or []) if spec else []:
        path = Path(root) / "package_data" / "plotly.min.js"
        if path.exists():
            return path
    return None


def _number(req: dict, key: str, lo: float, hi: float, default=None) -> float | None:
    value = req.get(key, default)
    if value is None or value == "":
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise RequestError(f"{key}: not a number ({value!r})") from None
    if not math.isfinite(value) or not lo <= value <= hi:
        raise RequestError(f"{key}: {value} outside [{lo}, {hi}]")
    return value


def _to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _wind_from(vec: np.ndarray) -> tuple[float, float]:
    """(direction the wind comes FROM in degrees, speed km/h) of a model vector (pointing where the air goes)."""
    vx, vy = float(vec[0]), float(vec[1])
    return float(np.degrees(np.arctan2(-vx, -vy)) % 360.0), float(np.hypot(vx, vy) * 3.6)


class _JobLog(io.TextIOBase):
    """stdout replacement while a job runs: echoes to the console and keeps the lines for the page."""

    def __init__(self, console, lines: list[str], on_line: Callable[[], None]):
        self.console, self.lines, self.on_line, self._buf = console, lines, on_line, ""

    def write(self, text: str) -> int:
        self.console.write(text)
        self._buf += text
        *done, self._buf = self._buf.split("\n")
        done = [line for line in done if line.strip()]
        if done:
            self.lines.extend(done)
            self.on_line()  # wake up the pages waiting for news of this job
        return len(text)

    def flush(self) -> None:
        self.console.flush()


class _CachedLift:
    """LiftModel wrapper: each hourly field is computed once per weather / day, several hours in parallel.

    The planner, `default_takeoff` and the page ask for the same fields again and again (every half hour of
    the window, for each takeoff or objective tried): they are kept read-only, so that no caller can modify
    a shared array by mistake.
    """

    def __init__(self, lift):
        self.lift, self.name = lift, lift.name
        self.fields: dict[float, np.ndarray] = {}
        self.hits = self.misses = 0

    def _compute(self, hour: float) -> np.ndarray:
        f = np.asarray(self.lift.field(hour))
        f.setflags(write=False)
        return f

    def prefetch(self, hours) -> None:
        """Compute the missing fields concurrently (numpy and scipy release the GIL on large arrays)."""
        missing = sorted({round(float(h), 4) for h in hours} - set(self.fields))
        if not missing:
            return
        self.fields[missing[0]] = self._compute(missing[0])  # first call alone: fills the lazy terrain caches
        self.misses += 1
        rest = missing[1:]
        if rest:
            workers = min(len(rest), os.cpu_count() or 1, 8)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for h, f in zip(rest, pool.map(self._compute, rest)):
                    self.fields[h] = f
            self.misses += len(rest)

    def field(self, hour_local: float) -> np.ndarray:
        key = round(float(hour_local), 4)
        if key in self.fields:
            self.hits += 1
        else:
            self.fields[key] = self._compute(key)
            self.misses += 1
        return self.fields[key]


class WebApp:
    """State and logic of the web interface, independent of HTTP (tested directly and through the server)."""

    def __init__(
        self,
        terrain: Terrain,
        lift_factory: Callable,
        base_params: Params | None = None,
        fetch_payload: Callable | None = None,
        dem_name: str = "",
    ):
        """`fetch_payload(lat, lon, date, model) -> dict` downloads an Open-Meteo answer (None: forecasts off)."""
        self.t, self.lift_factory = terrain, lift_factory
        self.base_params = base_params or Params()
        self.fetch_payload = fetch_payload
        self.dem_name = dem_name
        self.stride = max(1, math.ceil(max(terrain.nx, terrain.ny) / MAX_GRID))
        self.jobs: dict[str, dict] = {}
        self._payloads: dict[tuple, dict] = {}  # Open-Meteo answers: keep request counts low
        self._run_lock = threading.Lock()  # one planning job at a time (they use all the CPU anyway)
        self._jobs_lock = threading.Lock()
        self._news = threading.Condition()  # notified when a job logs a line or finishes (long polling)
        self._lifts: OrderedDict[str, _CachedLift] = OrderedDict()  # LRU, key = everything the fields depend on
        self._results: OrderedDict[str, dict] = OrderedDict()  # LRU, key = the whole planning request
        self._config_bytes: tuple[str, bytes, bytes] | None = None

    def _notify(self) -> None:
        with self._news:
            self._news.notify_all()

    # -- configuration ------------------------------------------------------------------ #
    def config(self) -> dict:
        t, s, p = self.t, self.stride, self.base_params
        rows, cols = np.arange(t.ny)[::s], np.arange(t.nx)[::s]
        xs, _ = t.rowcol_to_xy(0, cols)
        _, ys = t.rowcol_to_xy(rows, 0)
        lon_min, lat_min = t.xy_to_lonlat(0.0, 0.0)
        lon_max, lat_max = t.xy_to_lonlat(t.width, t.height)
        return dict(
            dem=dict(
                name=self.dem_name,
                nx=t.nx,
                ny=t.ny,
                width_km=round(t.width / 1000, 3),
                height_km=round(t.height / 1000, 3),
                z_min=round(float(t.z.min())),
                z_max=round(float(t.z.max())),
                bounds=dict(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max),
            ),
            grid=dict(
                stride=s,
                x_km=np.round(xs / 1000, 3),
                y_km=np.round(ys / 1000, 3),
                z=np.round(t.z[::s, ::s]).astype(int),
            ),
            presets={name: w.to_dict() for name, w in PRESETS.items()},
            weather_fields={
                k: dict(label=label, unit=unit, min=lo, max=hi, step=step)
                for k, (label, unit, lo, hi, step) in WEATHER_FIELDS.items()
            },
            objectives=list(OBJECTIVES),
            forecast=self.fetch_payload is not None,
            defaults=dict(
                preset="good_day",
                objective=p.objective,
                t_start=p.t_start,
                t_end=p.t_end,
                utc_offset=p.utc_offset,
                day_of_year=p.day_of_year,
                n_routes=p.n_routes,
                forecast_model="meteofrance_seamless",
            ),
        )

    def config_bytes(self) -> tuple[str, bytes, bytes]:
        """(ETag, JSON, gzipped JSON) of `config()`, built once: the DEM does not change while serving."""
        if self._config_bytes is None:
            body = json.dumps(self.config(), default=_to_json, separators=(",", ":")).encode()
            etag = f'"{zlib.crc32(body):08x}-{len(body):x}"'
            self._config_bytes = (etag, body, gzip.compress(body, 6))
        return self._config_bytes

    # -- request parsing ------------------------------------------------------------------ #
    def takeoff_xy(self, req: dict) -> tuple[float, float] | None:
        """Takeoff from the request ({"x_km", "y_km"} clicked on the map, or {"lon", "lat"}); None = automatic."""
        tk = req.get("takeoff")
        if not tk:
            return None
        if "lon" in tk and "lat" in tk:
            x, y = (float(v) for v in self.t.lonlat_to_xy(_number(tk, "lon", -180, 180), _number(tk, "lat", -90, 90)))
        else:
            x = 1000.0 * _number(tk, "x_km", -1e6, 1e6)
            y = 1000.0 * _number(tk, "y_km", -1e6, 1e6)
        if not self.t.contains(x, y):
            raise RequestError("takeoff: outside the DEM")
        return x, y

    def _date(self, req: dict) -> dt.date | None:
        if not req.get("date"):
            return None
        try:
            return dt.date.fromisoformat(str(req["date"]))
        except ValueError:
            raise RequestError(f"date: expected YYYY-MM-DD, got {req['date']!r}") from None

    def resolve_weather(self, req: dict) -> tuple[Weather, list[str]]:
        """Preset, then forecast (optional), then explicit values: the same order as the command line."""
        preset = req.get("preset", "good_day")
        if preset not in PRESETS:
            raise RequestError(f"preset: unknown {preset!r} (known: {', '.join(PRESETS)})")
        weather = Weather.from_dict(PRESETS[preset].to_dict())
        sources = [f"preset {preset}"]
        if req.get("forecast"):
            date = self._date(req)
            if date is None:
                raise RequestError("forecast: needs a date")
            if self.fetch_payload is None:
                raise RequestError("forecast: disabled on this server")
            from ..weather import openmeteo

            xy = self.takeoff_xy(req)
            lon, lat = self.t.xy_to_lonlat(*xy) if xy else (self.t.lon_mid, self.t.lat_mid)
            model = str(req.get("forecast_model") or "meteofrance_seamless")
            key = (date.isoformat(), model, round(float(lat), 2), round(float(lon), 2))
            if key not in self._payloads:
                try:
                    self._payloads[key] = self.fetch_payload(float(lat), float(lon), date, model)
                except Exception as exc:  # network, HTTP error, bad model name
                    raise RequestError(f"forecast: Open-Meteo request failed ({exc})") from None
            weather = openmeteo.weather_from_payload(self._payloads[key], float(lon), weather)
            sources.append(f"Open-Meteo {model} {date.isoformat()} at {float(lat):.3f}, {float(lon):.3f}")
        overrides = req.get("weather") or {}
        for name, value in overrides.items():
            if name not in WEATHER_FIELDS:
                raise RequestError(f"weather: unknown parameter {name!r}")
            _, _, lo, hi, _ = WEATHER_FIELDS[name]
            value = _number(overrides, name, lo, hi)
            if value is not None:
                weather.override(name, value)
                sources.append(f"{name}={value:g}")
        weather.cast_shadows = bool(req.get("shadows", weather.cast_shadows))
        return weather, sources

    def params(self, req: dict, weather: Weather) -> Params:
        p = self.base_params
        date = self._date(req)
        day = date.timetuple().tm_yday if date else int(_number(req, "day_of_year", 1, 366, p.day_of_year))
        t_start = _number(req, "t_start", 6, 20, p.t_start)
        t_end = _number(req, "t_end", 7, 22, p.t_end)
        if t_end <= t_start + 0.5:
            raise RequestError("t_end must be at least 30 min after t_start")
        objective = req.get("objective", p.objective)
        if objective not in OBJECTIVES:
            raise RequestError(f"objective: unknown {objective!r} (known: {', '.join(OBJECTIVES)})")
        return dataclasses.replace(
            p,
            day_of_year=day,
            utc_offset=_number(req, "utc_offset", -12, 14, p.utc_offset),
            t_start=t_start,
            t_end=t_end,
            weather=weather,
            objective=objective,
            n_routes=int(_number(req, "n_routes", 1, 10, p.n_routes)),
        )

    # -- weather preview ----------------------------------------------------------------- #
    def weather_summary(self, req: dict) -> dict:
        weather, sources = self.resolve_weather(req)
        p, lon = self.params(req, weather), self.t.lon_mid
        rows = []
        for hour in np.arange(np.floor(p.t_start) - 1, np.ceil(p.t_end) + 1):
            s = p.to_solar(float(hour), lon)
            low_dir, low = _wind_from(weather.wind_vec(weather.wind_alt_m, s))
            high_dir, high = _wind_from(weather.wind_vec(weather.wind_aloft_alt_m, s))
            rows.append(
                dict(
                    hour=float(hour),
                    ceiling_m=round(weather.ceiling(s)),
                    cloud_base_m=round(weather.cloud_base(s)),
                    wind_dir_deg=round(low_dir),
                    wind_kmh=round(low),
                    wind_aloft_dir_deg=round(high_dir),
                    wind_aloft_kmh=round(high),
                    sun=round(weather.sun_factor(s), 2),
                    strength=round(weather.strength(s), 2),
                    breeze_kmh=round(weather.breeze_ms(s) * 3.6, 1),
                )
            )
        return dict(sources=sources, day_of_year=p.day_of_year, hours=rows, weather=weather.to_dict())

    # -- planning jobs --------------------------------------------------------------------- #
    @staticmethod
    def _lift_key(params: Params) -> str:
        """Everything the lift fields depend on: weather, day, time zone (not the takeoff, objective or window)."""
        return json.dumps([params.weather.to_dict(), params.day_of_year, params.utc_offset], sort_keys=True)

    def _cached_lift(self, params: Params) -> _CachedLift:
        key = self._lift_key(params)
        if key in self._lifts:
            self._lifts.move_to_end(key)
        else:
            self._lifts[key] = _CachedLift(self.lift_factory(self.t, params))
            while len(self._lifts) > MAX_LIFT_CACHE:
                self._lifts.popitem(last=False)
        return self._lifts[key]

    def start_plan(self, req: dict) -> str:
        weather, sources = self.resolve_weather(req)  # validate before queuing: errors go back as HTTP 400
        params = self.params(req, weather)
        takeoff = self.takeoff_xy(req)
        job_id = uuid.uuid4().hex[:12]
        job = dict(id=job_id, status="queued", log=[], sources=sources, created=time.time())
        result_key = json.dumps(
            [self._lift_key(params), takeoff, params.objective, params.n_routes, params.t_start, params.t_end]
        )
        with self._jobs_lock:
            self.jobs[job_id] = job
            done = [k for k, j in self.jobs.items() if j["status"] in ("done", "error")]
            for k in done[: max(0, len(done) - MAX_JOBS)]:
                del self.jobs[k]
            cached = self._results.get(result_key)
            if cached is not None:  # identical request: answer at once
                self._results.move_to_end(result_key)
                job.update(status="done", result=cached, elapsed_s=0.0, log=["[web] same request as before: reused"])
                return job_id
        threading.Thread(target=self._run, args=(job, params, takeoff, result_key), daemon=True).start()
        return job_id

    def job(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def wait_job(self, job_id: str, since: int = 0, timeout_s: float = 0.0) -> dict | None:
        """Long polling: return when the job has more than `since` log lines, is finished, or after the timeout."""
        job = self.jobs.get(job_id)
        if job is None or timeout_s <= 0:
            return job
        with self._news:
            self._news.wait_for(
                lambda: len(job["log"]) > since or job["status"] in ("done", "error"), min(timeout_s, MAX_WAIT_S)
            )
        return job

    def _run(self, job: dict, params: Params, takeoff: tuple[float, float] | None, result_key: str) -> None:
        with self._run_lock:
            job["status"] = "running"
            self._notify()
            console, t0 = sys.stdout, time.perf_counter()
            sys.stdout = _JobLog(console, job["log"], self._notify)
            try:
                result = self.plan(params, takeoff)
                with self._jobs_lock:
                    self._results[result_key] = result
                    while len(self._results) > MAX_RESULT_CACHE:
                        self._results.popitem(last=False)
                job["result"] = result
                job["status"] = "done"
            except Exception as exc:
                job["status"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
                traceback.print_exc(file=console)
            finally:
                sys.stdout = console
                job["elapsed_s"] = round(time.perf_counter() - t0, 2)
                self._notify()

    def plan(self, params: Params, takeoff: tuple[float, float] | None) -> dict:
        """Run the planner and return everything the page draws (km for positions, m for altitudes)."""
        from ..export.flight_package import route_dict
        from ..planning import Planner, RouteOptimizer, default_takeoff

        t, s = self.t, self.stride
        print(f"[web] objective {params.objective}, day {params.day_of_year}, {params.t_start:g}-{params.t_end:g} h")
        t0 = time.perf_counter()
        lift = self._cached_lift(params)
        n_cached = len(lift.fields)
        # the half-hourly fields of the planner, and the one `default_takeoff` uses
        lift.prefetch(list(np.arange(params.t_start, params.t_end + 1e-6, 0.5)) + [params.t_start + 1.0])
        print(
            f"[web] lift fields: {len(lift.fields) - n_cached} computed, {n_cached} reused "
            f"({time.perf_counter() - t0:.2f} s)"
        )
        if takeoff is None:
            takeoff = default_takeoff(t, params, lift)
            print(f"[web] automatic takeoff at x {takeoff[0] / 1000:.1f} km, y {takeoff[1] / 1000:.1f} km")
        planner = Planner(t, params, takeoff, lift)
        routes = RouteOptimizer(planner).optimise()
        print(f"[web] {len(routes)} routes found ({time.perf_counter() - t0:.2f} s)")

        out_routes = []
        for k, r in enumerate(routes):
            d = route_dict(planner, r)
            prof = r.profile
            tp = r.xy[r.turnpoints]
            if r.kind in ("triangle", "fai"):
                tp = np.vstack([tp, tp[:1]])
            d.update(
                color=ROUTE_COLORS[k % len(ROUTE_COLORS)],
                duration_h=round(r.duration_h, 2),
                speed_kmh=round(r.speed_kmh, 1),
                profile=np.round(np.column_stack([prof[:, :2] / 1000, prof[:, 2:3], prof[:, 3:4]]), 4),
                turnpoints_km=np.round(tp[:, :2] / 1000, 3),
            )
            out_routes.append(d)

        hourly = np.flatnonzero(np.isclose(planner.sample_hours % 1.0, 0.0))
        if len(hourly) == 0:
            hourly = np.arange(len(planner.sample_hours))
        lon, lat = t.xy_to_lonlat(*takeoff)
        return dict(
            lift_name=lift.name,
            takeoff=dict(
                x_km=round(takeoff[0] / 1000, 3),
                y_km=round(takeoff[1] / 1000, 3),
                z=round(float(planner.zg[0])),
                lon=round(float(lon), 5),
                lat=round(float(lat), 5),
            ),
            nodes=dict(
                x_km=np.round(planner.x[1:] / 1000, 3),
                y_km=np.round(planner.y[1:] / 1000, 3),
                z=np.round(planner.zg[1:]).astype(int),
            ),
            ceiling=dict(hours=planner.slot_hours, m=np.round(planner.ceil).astype(int)),
            lift=dict(  # integers in hundredths: float32 values would print as 0.27000001072883606 in JSON
                hours=planner.sample_hours[hourly],
                scale=0.01,
                frames=[np.round(100.0 * planner.fields[i][::s, ::s]).astype(np.int16) for i in hourly],
            ),
            routes=out_routes,
        )


# -- HTTP --------------------------------------------------------------------------------- #
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css"}
GZIP_MIN_BYTES = 1024
_static_cache: dict[Path, tuple] = {}  # file -> ((mtime, size), etag, raw bytes, gzipped bytes)
_static_lock = threading.Lock()


def _static(path: Path) -> tuple[str, bytes, bytes]:
    """(ETag, bytes, gzipped bytes) of a static file, compressed once and reloaded when the file changes."""
    st = path.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    with _static_lock:
        hit = _static_cache.get(path)
        if hit is None or hit[0] != stamp:
            raw = path.read_bytes()
            hit = (stamp, f'"{st.st_mtime_ns:x}-{st.st_size:x}"', raw, gzip.compress(raw, 6))
            _static_cache[path] = hit
    return hit[1], hit[2], hit[3]


def _query(path: str) -> dict[str, str]:
    from urllib.parse import parse_qsl

    return dict(parse_qsl(path.split("?", 1)[1])) if "?" in path else {}


class Handler(BaseHTTPRequestHandler):
    server_version = "cheminee-web"
    protocol_version = "HTTP/1.1"  # keep-alive: the page makes many small requests
    app: WebApp  # set on the class built by `make_server`

    def log_request(self, code="-", size="-") -> None:  # quieter than the default: only failed requests
        if str(code).isdigit() and int(code) >= 400:
            sys.stderr.write(f"[web] {self.command} {self.path} -> {code}\n")

    def log_message(self, fmt, *args) -> None:
        pass

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        cache: str = "no-store",
        etag: str | None = None,
        gzipped: bytes | None = None,
    ) -> None:
        """Send a body, gzipped when the browser accepts it and it is worth it."""
        if "gzip" in self.headers.get("Accept-Encoding", "") and len(body) >= GZIP_MIN_BYTES:
            body, encoding = (gzipped if gzipped is not None else gzip.compress(body, 5)), "gzip"
        else:
            encoding = None
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("Vary", "Accept-Encoding")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        if etag:
            self.send_header("ETag", etag)
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, file: Path, cache: str) -> None:
        self._send_cached(*_static(file), CONTENT_TYPES[file.suffix], cache)

    def _send_cached(self, etag: str, raw: bytes, gz: bytes, content_type: str, cache: str) -> None:
        if etag in self.headers.get("If-None-Match", ""):  # the browser already has it
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._send(200, raw, content_type, cache, etag, gz)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, default=_to_json, separators=(",", ":")).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def do_GET(self) -> None:
        path, query = self.path.split("?", 1)[0], _query(self.path)
        try:
            if path == "/api/config":
                return self._send_cached(*self.app.config_bytes(), "application/json", "no-cache")
            if path.startswith("/api/jobs/"):
                since, wait = int(query.get("since", 0)), float(query.get("wait", 0))
                job = self.app.wait_job(path.rsplit("/", 1)[1], since, wait)
                return self._json(job) if job else self._error(HTTPStatus.NOT_FOUND, "unknown job")
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        if path == "/static/plotly.min.js":
            js = _plotly_js()
            if js is None:
                return self._error(HTTPStatus.NOT_FOUND, "plotly.min.js not found: is plotly installed?")
            return self._send_static(js, "public, max-age=604800")  # a week, then revalidated by ETag
        name = "index.html" if path == "/" else path.removeprefix("/static/")
        file = (STATIC_DIR / name).resolve()
        if file.parent != STATIC_DIR.resolve() or not file.is_file() or file.suffix not in CONTENT_TYPES:
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        self._send_static(file, "no-cache")  # always revalidated: edits show at once, unchanged files cost a 304

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1_000_000:
                raise RequestError("request too large")
            req = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(req, dict):
                raise RequestError("expected a JSON object")
            if self.path == "/api/weather":
                return self._json(self.app.weather_summary(req))
            if self.path == "/api/plan":
                return self._json({"job": self.app.start_plan(req)})
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except (RequestError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # report, keep serving
            traceback.print_exc()
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")


def make_server(app: WebApp, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """HTTP server bound to the loopback interface (port 0 = any free port)."""
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(app: WebApp, port: int = 8765, open_browser: bool = False) -> None:
    server = make_server(app, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[web] cheminee running at {url} (Ctrl+C to stop)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[web] stopped")
    finally:
        server.server_close()
