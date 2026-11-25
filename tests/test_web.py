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


def test_config_describes_the_grid(app, terrain):
    cfg = app.config()
    g = cfg["grid"]
    assert np.shape(g["z"]) == (len(g["y_km"]), len(g["x_km"]))
    assert np.all(np.diff(g["x_km"]) > 0) and np.all(np.diff(g["y_km"]) < 0)  # row 0 = north edge
    assert g["z"][0][0] == round(float(terrain.z[0, 0]))
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
    fc = app.weather_summary(req)
    assert fc["day_of_year"] == 319 and any("Open-Meteo" in s for s in fc["sources"])
    assert fc["weather"]["hourly"], "the forecast must bring hourly series"
    app.weather_summary(req)
    assert len(app.fetch_calls) == 1, "the same forecast must not be downloaded twice"


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
    for r in res["routes"]:
        prof = np.asarray(r["profile"])
        assert prof.shape[1] == 4
        assert np.allclose(prof[0, :2], [x_km, y_km], atol=1e-3)  # starts at the clicked takeoff
        assert np.all(np.diff(prof[:, 3]) >= -1e-9)  # time never goes back
        z_ground = terrain.sample(terrain.z, prof[:, 0] * 1000, prof[:, 1] * 1000)
        assert np.all(prof[:, 2] >= z_ground - 1.0)
        assert [s["t0"] for s in r["segments"]] == sorted(s["t0"] for s in r["segments"])
    frames = np.asarray(res["lift"]["frames"]) * res["lift"]["scale"]
    assert len(frames) == len(res["lift"]["hours"]) and frames[0].shape == np.shape(app.config()["grid"]["z"])
    assert frames.min() >= 0 and frames.max() <= 1 and frames.max() > 0.1


def test_http_server_end_to_end(app, base_url):
    status, ctype, body = _get(base_url + "/")
    assert status == 200 and ctype.startswith("text/html") and b"plotly.min.js" in body
    status, ctype, body = _get(base_url + "/static/plotly.min.js")
    assert status == 200 and len(body) > 1_000_000  # served from the plotly package, no CDN
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
        assert r["turnpoints_km"][0] == r["turnpoints_km"][-1]

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
    lift = app._cached_lift(params)
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

    status, headers, body = get("/static/plotly.min.js", {"Accept-Encoding": "gzip"})
    assert status == 200 and headers["Content-Encoding"] == "gzip" and "max-age" in headers["Cache-Control"]
    import gzip

    assert len(body) < len(gzip.decompress(body)) / 2
    status, headers2, body = get("/static/plotly.min.js", {"If-None-Match": headers["ETag"]})
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
