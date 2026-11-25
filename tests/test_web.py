"""Web interface: JSON API and HTTP server on the analytic terrain. No network (forecast from a saved answer)."""

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from cheminee.lift import HeuristicLift
from cheminee.web import WebApp, make_server
from cheminee.web.server import RequestError

REAL_ANSWER = Path(__file__).parent / "data" / "openmeteo_digne_2026-11-15.json"


@pytest.fixture
def app(terrain, params):
    calls = []

    def fake_fetch(lat, lon, date, model):
        calls.append((lat, lon, date, model))
        return json.loads(REAL_ANSWER.read_text())

    web = WebApp(terrain, lambda t, p: HeuristicLift(t, p), params, fetch_payload=fake_fetch, dem_name="test")
    web.fetch_calls = calls
    return web


@pytest.fixture
def base_url(app):
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as answer:
        return answer.status, answer.headers.get("Content-Type"), answer.read()


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as answer:
            return answer.status, json.loads(answer.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _wait(app, job_id, timeout_s=120.0):
    t0 = time.time()
    while app.job(job_id)["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout_s, "planning job did not finish"
        time.sleep(0.1)
    return app.job(job_id)


def test_config_describes_the_map(app, terrain):
    cfg = app.config()
    b, g = cfg["dem"]["bounds"], cfg["lift_grid"]
    assert b["lon_min"] == pytest.approx(6.2) and b["lat_min"] == pytest.approx(44.0)
    assert (g["ny"], g["nx"]) == terrain.z[:: g["stride"], :: g["stride"]].shape
    west, north, east, south = g["bounds"]  # the lift image covers the DEM, within one image pixel
    pixel = g["stride"] * terrain.cell_deg
    assert abs(west - b["lon_min"]) < pixel and abs(north - b["lat_max"]) < pixel
    assert abs(east - b["lon_max"]) < pixel and abs(south - b["lat_min"]) < pixel
    assert "{z}/{x}/{y}" in cfg["dem"]["tiles"]["url"] and "Copernicus" in cfg["dem"]["attribution"]
    assert "good_day" in cfg["presets"] and cfg["forecast"] is True
    json.dumps(cfg, default=lambda o: o.tolist())  # serialisable


def test_weather_overrides_and_forecast(app):
    calm = app.weather_summary({"preset": "good_day", "weather": {"wind_speed_kmh": 0, "wind_aloft_speed_kmh": 0}})
    assert all(h["wind_kmh"] == 0 for h in calm["hours"])
    windy = app.weather_summary({"preset": "good_day", "weather": {"wind_speed_kmh": 25, "wind_dir_deg": 90}})
    afternoon = [h for h in windy["hours"] if 13 <= h["hour"] <= 15]
    assert all(abs(h["wind_dir_deg"] - 90) <= 1 for h in afternoon)  # direction the wind comes FROM
    hours = [h["hour"] for h in windy["hours"]]
    assert hours == sorted(hours)

    req = {"preset": "good_day", "forecast": True, "date": "2026-11-15", "takeoff": {"x_km": 3.0, "y_km": 3.0}}
    fc = app.weather_summary({**req, "weather_points": 1})
    assert fc["day_of_year"] == 319 and any("Open-Meteo" in s for s in fc["sources"])
    assert fc["weather"]["hourly"], "the forecast must bring hourly series"
    assert fc["points"] == []
    app.weather_summary({**req, "weather_points": 1})
    assert len(app.fetch_calls) == 1, "the same forecast must not be downloaded twice"
    three = app.weather_summary(req)  # default: the takeoff and two points of a triangle around it
    assert [q["role"] for q in three["points"]] == ["takeoff", "point 2", "point 3"]
    assert len(app.fetch_calls) == 3, "one request per new point, the takeoff one is reused"
    assert sum("Open-Meteo" in s for s in three["sources"]) == 3
    assert all(set(q["ceiling_m"]) == set(q["wind_aloft"]) and q["ceiling_m"] for q in three["points"])


def test_weather_points_placed_by_hand_and_planned(app, terrain):
    sites = [list(terrain.xy_to_lonlat(1000.0, 6000.0)), list(terrain.xy_to_lonlat(4500.0, 1000.0))]
    req = {"forecast": True, "date": "2026-11-15", "takeoff": {"x_km": 3.0, "y_km": 3.0}, "weather_sites": sites}
    summary = app.weather_summary(req)
    got = [(q["lon"], q["lat"]) for q in summary["points"][1:]]
    assert np.allclose(got, sites, atol=1e-5)
    with pytest.raises(RequestError, match="outside"):
        app.weather_summary({**req, "weather_sites": [[0.0, 0.0], sites[1]]})
    with pytest.raises(RequestError, match="weather_points"):
        app.weather_summary({**req, "weather_points": 2})
    job = _wait(app, app.start_plan({**req, "objective": "free", "n_routes": 1, "t_start": 11, "t_end": 15}))
    assert job["status"] == "done", job.get("error")
    result = job["result"]
    assert len(result["weather_points"]) == 3 and "weather points" in result["lift_name"]


@pytest.mark.parametrize(
    "req, message",
    [
        ({"preset": "nope"}, "preset"),
        ({"objective": "loop"}, "objective"),
        ({"t_start": 15, "t_end": 15.2}, "t_end"),
        ({"takeoff": {"x_km": -5, "y_km": 1}}, "outside"),
        ({"weather": {"wind_speed_kmh": 500}}, "wind_speed_kmh"),
        ({"weather": {"hourly": 1}}, "unknown parameter"),
        ({"forecast": True}, "date"),
        ({"date": "15/11/2026"}, "date"),
    ],
)
def test_bad_requests_are_refused(app, req, message):
    with pytest.raises(RequestError, match=message):
        app.start_plan(req)


def test_plan_from_a_clicked_takeoff(app, terrain):
    x_km, y_km = 1.6, 4.0  # on the western ridge
    job = _wait(app, app.start_plan({"takeoff": {"x_km": x_km, "y_km": y_km}, "objective": "free"}))
    assert job["status"] == "done", job.get("error")
    res = job["result"]
    assert res["takeoff"]["x_km"] == pytest.approx(x_km) and res["takeoff"]["y_km"] == pytest.approx(y_km)
    assert res["routes"], "\n".join(job["log"])
    assert any(line.startswith("[nodes]") for line in job["log"]), "planner logs are shown on the page"
    takeoff_lonlat = terrain.xy_to_lonlat(x_km * 1000, y_km * 1000)
    assert (res["takeoff"]["lon"], res["takeoff"]["lat"]) == pytest.approx(takeoff_lonlat, abs=1e-5)
    for r in res["routes"]:
        path = np.asarray(r["path"])  # lon, lat, flight altitude, hour, ground altitude
        assert path.shape[1] == 5
        assert np.allclose(path[0, :2], takeoff_lonlat, atol=1e-5)  # starts at the clicked takeoff
        assert np.all(np.diff(path[:, 3]) >= -1e-9)  # time never goes back
        x, y = terrain.lonlat_to_xy(path[:, 0], path[:, 1])
        assert np.allclose(path[:, 4], terrain.sample(terrain.z, x, y), atol=1.0)
        assert np.all(path[:, 2] >= path[:, 4] - 1.0)  # never under the ground
        assert [s["t0"] for s in r["segments"]] == sorted(s["t0"] for s in r["segments"])
    lon, lat = np.asarray(res["nodes"]["lonlat"]).T
    assert np.all(terrain.contains(*terrain.lonlat_to_xy(lon, lat)))
    g = app.config()["lift_grid"]
    frames = np.asarray(res["lift"]["frames"]) * res["lift"]["scale"]
    assert len(frames) == len(res["lift"]["hours"]) and frames[0].shape == (g["ny"], g["nx"])
    assert frames.min() >= 0 and frames.max() <= 1 and frames.max() > 0.1


def test_http_server_end_to_end(app, base_url):
    status, ctype, body = _get(base_url + "/")
    assert status == 200 and ctype.startswith("text/html") and b"maplibre-gl.js" in body
    status, ctype, body = _get(base_url + "/static/vendor/maplibre-gl-4.7.1/maplibre-gl.js")
    assert status == 200 and ctype == "text/javascript" and b"MapLibre" in body[:200]  # vendored, no CDN
    status, ctype, body = _get(base_url + "/static/track3d.js")
    assert status == 200 and b"class TrackLayer" in body and b"track3d.js" in _get(base_url + "/")[2]
    status, ctype, body = _get(base_url + "/api/config")
    assert json.loads(body)["dem"]["name"] == "test"

    status, answer = _post(base_url + "/api/weather", {"preset": "blue_stable"})
    assert status == 200 and answer["hours"]
    status, answer = _post(base_url + "/api/plan", {"objective": "triangle"})
    assert status == 200
    job = _wait(app, answer["job"])
    assert job["status"] == "done", job.get("error")
    status, ctype, body = _get(f"{base_url}/api/jobs/{answer['job']}")
    routes = json.loads(body)["result"]["routes"]
    assert routes and all(r["kind"] in ("triangle", "fai") for r in routes)
    for r in routes:  # closed scoring polyline
        assert r["turnpoints_lonlat"][0] == r["turnpoints_lonlat"][-1]

    status, answer = _post(base_url + "/api/plan", {"objective": "loop"})
    assert status == 400 and "objective" in answer["error"]
    for path in ("/static/../server.py", "/api/jobs/unknown", "/static/missing.js"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base_url + path)
        assert err.value.code == 404


def test_lift_fields_and_results_are_reused(app):
    first = _wait(app, app.start_plan({"takeoff": {"x_km": 1.6, "y_km": 4.0}, "objective": "free"}))
    assert any("0 reused" in line for line in first["log"])
    # another takeoff and objective, same weather and day: no lift field is computed again
    second = _wait(app, app.start_plan({"takeoff": {"x_km": 2.0, "y_km": 4.5}, "objective": "triangle"}))
    assert second["status"] == "done", second.get("error")
    assert any(line.startswith("[web] lift fields: 0 computed") for line in second["log"])
    # the exact same request: answered at once with the same result
    again = app.start_plan({"takeoff": {"x_km": 2.0, "y_km": 4.5}, "objective": "triangle"})
    assert app.job(again)["status"] == "done" and app.job(again)["result"] is second["result"]
    # another weather: new fields
    windy = {"takeoff": {"x_km": 2.0, "y_km": 4.5}, "objective": "free", "weather": {"wind_speed_kmh": 20}}
    third = _wait(app, app.start_plan(windy))
    assert any("0 reused" in line for line in third["log"])


def test_cached_fields_are_read_only(app, params):
    lift = app._cached_lift(params, {"model": "base", "options": {}})
    lift.prefetch([12.0, 12.5, 13.0])
    f = lift.field(12.5)
    assert np.array_equal(f, HeuristicLift(app.t, params).field(12.5))  # parallel = serial result
    with pytest.raises(ValueError):
        f[0, 0] = 1.0


def test_long_polling_returns_as_soon_as_the_job_ends(app):
    job_id = app.start_plan({"objective": "free"})
    t0 = time.time()
    job = app.wait_job(job_id, since=10_000, timeout_s=60)  # never enough log lines: only the end wakes it up
    assert job["status"] == "done" and time.time() - t0 < 30
    t0 = time.time()
    assert app.wait_job(job_id, since=10_000, timeout_s=5)["status"] == "done"
    assert time.time() - t0 < 0.5  # finished job: no wait


def test_http_compression_and_browser_cache(app, base_url):
    def get(path, headers):
        req = urllib.request.Request(base_url + path, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as answer:
                return answer.status, dict(answer.headers), answer.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read()

    vendor = "/static/vendor/maplibre-gl-4.7.1/maplibre-gl.js"
    status, headers, body = get(vendor, {"Accept-Encoding": "gzip"})
    assert status == 200 and headers["Content-Encoding"] == "gzip" and "immutable" in headers["Cache-Control"]
    import gzip

    assert len(body) < len(gzip.decompress(body)) / 2
    status, headers2, body = get(vendor, {"If-None-Match": headers["ETag"]})
    assert status == 304 and body == b""
    status, headers, body = get("/api/config", {"Accept-Encoding": "gzip"})
    assert headers["Content-Encoding"] == "gzip" and json.loads(gzip.decompress(body))["dem"]["name"] == "test"
    status, headers, body = get("/api/config", {})  # no gzip unless asked for
    assert "Content-Encoding" not in headers and json.loads(body)["dem"]["name"] == "test"
    status, _, _ = get("/api/jobs/x?since=a", {})
    assert status == 400
    status, headers, _ = get("/api/config", {})
    status, _, body = get("/api/config", {"If-None-Match": headers["ETag"]})
    assert status == 304 and body == b""  # the DEM grid is downloaded once per browser


def test_dem_tiles_and_elevation(app, base_url, terrain):
    from PIL import Image

    from cheminee.web.tiles import decode_terrain_rgb, tile_lonlat

    # a zoom 14 tile inside the analytic DEM (lon 6.2-6.267, lat 44.0-44.067)
    z, x, y = 14, 8475, 5955
    lon, lat = tile_lonlat(z, x, y)
    assert 6.2 < lon.min() and lat.max() < 44.067
    status, headers, body = _get(f"{base_url}/tiles/dem/{z}/{x}/{y}.png")
    assert status == 200 and headers == "image/png"
    heights = decode_terrain_rgb(np.asarray(Image.open(__import__("io").BytesIO(body)).convert("RGB")))
    rows, cols = [10, 128, 250], [5, 128, 240]
    for r, c in zip(rows, cols):  # same altitude as the DEM the planner uses, to the metre
        expected = terrain.sample(terrain.z, *terrain.lonlat_to_xy(lon[c], lat[r]))
        assert heights[r, c] == pytest.approx(float(expected), abs=0.55)
    # far from the DEM: flat tile at the lowest DEM altitude, shared by every such tile
    outside = decode_terrain_rgb(
        np.asarray(Image.open(__import__("io").BytesIO(_get(f"{base_url}/tiles/dem/14/0/0.png")[2])))
    )
    assert np.allclose(outside, float(terrain.z.min()), atol=0.55)
    assert app.dem_tile(14, 0, 0) is app.dem_tile(14, 1, 1)

    status, _, body = _get(f"{base_url}/api/elevation?lon=6.23&lat=44.03")
    point = json.loads(body)
    assert point["inside"] and point["z"] == round(float(terrain.sample(terrain.z, *terrain.lonlat_to_xy(6.23, 44.03))))
    assert json.loads(_get(f"{base_url}/api/elevation?lon=7&lat=45")[2]) == {
        "lon": 7.0,
        "lat": 45.0,
        "inside": False,
        "z": None,
    }
    for bad in ("/tiles/dem/99/0/0.png", "/tiles/dem/3/9/0.png", "/tiles/dem/a/b/c.png", "/api/elevation?lon=6.2"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base_url + bad)
        assert err.value.code == 400


def test_terrain_rgb_roundtrip():
    from cheminee.web.tiles import decode_terrain_rgb, encode_terrain_rgb

    h = np.array([[-50.0, 0.0, 509.27], [1971.04, 4807.8, 8848.86]])
    assert np.allclose(decode_terrain_rgb(encode_terrain_rgb(h)), h, atol=0.05)


def test_dem_tiles_slope_down_outside_the_dem(terrain):
    from cheminee.web.tiles import APRON_M, dem_heights, tile_lonlat

    z, x, y = 13, 4237, 2977  # straddles the western edge of the analytic DEM (lon 6.2)
    lon, lat = tile_lonlat(z, x, y)
    h = dem_heights(terrain, z, x, y)
    row = int(np.argmin(np.abs(lat - 44.03)))
    west = lon < 6.2
    assert west.any() and (~west).any()
    line = h[row]
    assert np.all(np.diff(line[west]) >= -1e-3), "the apron rises towards the DEM edge (no wall, no pit)"
    floor = float(terrain.z.min())
    assert np.all(line[west] >= floor - 1e-3) and np.all(line[west] <= line[west][-1] + 1e-3)
    assert APRON_M > 0


def test_track_is_dense_and_above_the_ground(app, terrain):
    from cheminee.web.server import TRACK_STEP_M, densify

    prof = np.array([[0.0, 0.0, 1000.0, 11.0], [0.0, 0.0, 1500.0, 11.2], [600.0, 800.0, 1400.0, 11.4]])
    dense = densify(prof)
    assert np.allclose(dense[[0, 1, -1]], prof)  # original points kept, climb untouched
    step = np.hypot(*np.diff(dense[:, :2], axis=0).T)
    assert step.max() <= TRACK_STEP_M + 1e-9 and np.all(np.diff(dense[:, 3]) >= 0)

    job = _wait(app, app.start_plan({"takeoff": {"x_km": 1.6, "y_km": 4.0}, "objective": "free"}))
    for r in job["result"]["routes"]:
        track = np.asarray(r["track"])  # lon, lat, flight altitude, ground altitude
        x, y = terrain.lonlat_to_xy(track[:, 0], track[:, 1])
        assert np.hypot(np.diff(x), np.diff(y)).max() <= TRACK_STEP_M + 1.0
        assert np.allclose(track[:, 3], terrain.sample(terrain.z, x, y), atol=1.0)
        assert np.all(track[:, 2] >= track[:, 3] - 1.0)
        assert np.allclose(track[0, :2], r["path"][0][:2]) and np.allclose(track[-1, :2], r["path"][-1][:2])
        assert track.shape[1] == 5 and np.all(np.diff(track[:, 4]) >= -1e-9)  # hour of every point, for the replay
        assert track[0, 4] == pytest.approx(r["path"][0][3], abs=1e-3)
        assert track[-1, 4] == pytest.approx(r["path"][-1][3], abs=1e-3)


def test_extra_lift_models_plugged_in_by_the_caller(terrain, params):
    """Add-ons (kk7) plug their lift model into the page without cheminee depending on them."""
    from cheminee.web import LiftChoice

    made = []

    class Scaled:
        def __init__(self, base, gain):
            self.base, self.gain, self.name = base, gain, f"{base.name} x{gain:g}"

        def field(self, hour):
            return np.clip(self.base.field(hour) * self.gain, 0, 1).astype(np.float32)

    def make(base_factory, options):
        made.append(options)
        return lambda t, p: Scaled(base_factory(t, p), options["gain"])

    choice = LiftChoice("scaled lift", make, {"gain": ("gain", "", 0.5, 2.0, 0.1, 1.2)}, "Data: someone, CC BY", "test")
    app = WebApp(terrain, lambda t, p: HeuristicLift(t, p), params, lift_choices={"scaled": choice})
    models = app.config()["lift_models"]
    assert set(models) == {"base", "scaled"} and models["scaled"]["options"]["gain"]["default"] == 1.2

    base = _wait(app, app.start_plan({"objective": "free", "takeoff": {"x_km": 1.6, "y_km": 4.0}}))["result"]
    assert base["lift_model"] == "base" and base["lift_attribution"] is None
    req = {
        "objective": "free",
        "takeoff": {"x_km": 1.6, "y_km": 4.0},
        "lift": {"model": "scaled", "options": {"gain": 1.5}},
    }
    res = _wait(app, app.start_plan(req))["result"]
    assert res["lift_model"] == "scaled" and res["lift_name"].endswith("x1.5")
    assert res["lift_attribution"] == "Data: someone, CC BY"
    assert np.asarray(res["lift"]["frames"]).max() > np.asarray(base["lift"]["frames"]).max()  # its own fields
    req["takeoff"] = {"x_km": 2.0, "y_km": 4.5}
    _wait(app, app.start_plan(req))
    assert made == [{"gain": 1.5}], "one factory per model and options (add-ons cache their data in it)"
    for bad, message in [
        ({"model": "nope"}, "unknown model"),
        ({"model": "scaled", "options": {"x": 1}}, "unknown options"),
        ({"model": "scaled", "options": {"gain": 9}}, "gain"),
    ]:
        with pytest.raises(RequestError, match=message):
            app.start_plan({"lift": bad})
    with pytest.raises(ValueError):
        WebApp(terrain, lambda t, p: HeuristicLift(t, p), params, lift_choices={"base": choice})


def test_weather_models_choice_and_comparison(terrain, params):
    def fetch(lat, lon, date, model):
        payload = json.loads(REAL_ANSWER.read_text())
        if model == "icon_d2":  # range ending in the morning
            for key, values in payload["hourly"].items():
                if key != "time":
                    payload["hourly"][key] = [v if k < 10 else None for k, v in enumerate(values)]
        return payload

    app = WebApp(terrain, lambda t, p: HeuristicLift(t, p), params, fetch_payload=fetch)
    ids = [m["id"] for m in app.config()["forecast_models"]]
    assert "meteofrance_seamless" in ids and "best_match" in ids
    base = {"forecast": True, "date": "2026-11-15"}
    with pytest.raises(RequestError, match="unknown"):
        app.weather_summary({**base, "forecast_model": "my_model&x=1"})
    with pytest.raises(RequestError, match="flying hours"):
        app.weather_summary({**base, "forecast_model": "icon_d2"})
    with pytest.raises(RequestError, match="date"):
        app.compare_models({})
    table = app.compare_models(base)["models"]
    assert [m["id"] for m in table] == ids
    rows = {m["id"]: m for m in table}
    assert not rows["icon_d2"]["ok"] and "flying hours" in rows["icon_d2"]["error"] and rows["icon_d2"]["short"]
    ok = rows["meteofrance_seamless"]
    assert ok["ok"] and ok["ceiling_max_m"] == max(ok["ceiling_m"].values()) and 0 <= ok["sun_14"] <= 1
