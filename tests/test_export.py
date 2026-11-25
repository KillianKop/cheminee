import base64
import dataclasses
import json
from pathlib import Path

import pytest

from cheminee.config import Params
from cheminee.export import FORMAT, FORMAT_VERSION, build_flight_package, candidate_routes, write_flight_package
from cheminee.export.flight_package import merge_short_glides
from cheminee.lift import HeuristicLift
from cheminee.lift.points import with_weather_points
from cheminee.planning import Planner, default_takeoff
from cheminee.weather import Weather, WeatherPoint

SCHEMA = Path(__file__).parents[1] / "schema" / "flight-package.v1.schema.json"


@pytest.fixture(scope="module")
def package(terrain):
    params = Params(n_nodes=30, node_spacing_m=600.0, weather=Weather(breeze_kmh=0.0))
    lift = HeuristicLift(terrain, params)
    planner = Planner(terrain, params, default_takeoff(terrain, params, lift), lift, verbose=False)
    routes = candidate_routes(planner, n_triangles=2, n_free=1)
    assert routes
    return build_flight_package(terrain, params, planner, routes)


def test_package_matches_the_schema(package):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(package, schema)
    assert package["format"] == FORMAT and package["version"] == FORMAT_VERSION


def test_package_content(package, terrain):
    assert package["extent"]["width"] == pytest.approx(terrain.width, abs=1)
    assert package["meta"]["attribution"], "the DEM notice must always be shipped"
    dem = package["dem"]
    assert len(base64.b64decode(dem["data"])) == dem["nx"] * dem["ny"] * 2
    assert len(package["lift"]["frames"]) == len(package["lift"]["hours"])
    assert all(len(n["climb"]) == len(package["climb_times"]) for n in package["nodes"])


def test_route_segments_are_continuous_and_alternate(package):
    for route in package["routes"]:
        segs = route["segments"]
        assert segs[0]["t0"] == route["t_start"] and segs[-1]["t1"] == route["t_end"]
        for a, b in zip(segs[:-1], segs[1:]):
            assert b["t0"] == pytest.approx(a["t1"], abs=1e-4)
            assert b["a0"] == pytest.approx(a["a1"], abs=1)
        for s in segs:
            if s["type"] == "glide":
                assert s["dist"] >= 300 and s["margin_m"] > 0
                # rising air can make a glide gain height: then its sink rate is negative
                assert s["sink_ms"] * (s["a0"] - s["a1"]) >= 0
                assert 25 <= s["airspeed_kmh"] <= 60
            else:
                assert s["a1"] > s["a0"] and s["rate"] > 0
        if route["kind"] in ("triangle", "fai"):
            assert route["scoring"][0] == route["scoring"][-1] and len(route["turnpoints"]) == 3


def test_routing_graph_is_consistent(package):
    graph, n = package["graph"], len(package["nodes"])
    width = 2 + 2 * len(graph["speeds_kmh"])
    assert graph["slots"]
    for slot in graph["slots"]:
        assert slot["t1"] > slot["t0"]
        for row in slot["edges"]:
            assert len(row) == width and 0 <= row[0] < n and 0 <= row[1] < n and row[0] != row[1]
            assert all(v >= 0 or v == -1 for v in row[3::2])  # heights lost: >= 0, or -1 when not feasible
            pairs = list(zip(row[2::2], row[3::2]))
            assert any(g > 0 for g, _ in pairs)
            assert all((g == -1) == (loss == -1) for g, loss in pairs)
            assert all(loss <= slot["ceiling_m"] for g, loss in pairs)
        assert all(row[2] >= 0 for row in slot["takeoff_edges"])  # cautious: a gain is reported as 0


def test_short_glides_are_folded_into_one_climb():
    segs = [
        dict(type="climb", node=0, t0=11.0, t1=11.4, a0=1200, a1=2100, rate=0),
        dict(type="glide", src=0, dst=1, t0=11.4, t1=11.41, a0=2100, a1=2090, dist=80),
        dict(type="climb", node=1, t0=11.41, t1=11.45, a0=2090, a1=2120, rate=0),
        dict(type="glide", src=1, dst=2, t0=11.45, t1=11.6, a0=2120, a1=1700, dist=4000),
    ]
    out = merge_short_glides(segs)
    assert [s["type"] for s in out] == ["climb", "glide"]
    assert out[0]["node"] == 1 and out[0]["t0"] == 11.0 and out[0]["a1"] == 2120
    assert out[0]["rate"] == pytest.approx(920 / (0.45 * 3600), abs=0.01)


def test_write_and_read_back(package, tmp_path):
    path = write_flight_package(package, str(tmp_path / "out" / "flight.json"))
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["routes"][0]["label"] == package["routes"][0]["label"]
    assert again["meta"]["attribution"] == package["meta"]["attribution"]


def test_ceiling_of_every_node_with_weather_points(terrain):
    """Weather points with a low ceiling in the west: `ceilings_m` gives the ceiling of every node of a slot
    (optional key, only when they differ); `ceiling_m` stays the one of the takeoff."""
    jsonschema = pytest.importorskip("jsonschema")
    params = Params(n_nodes=30, node_spacing_m=600.0, weather=Weather(breeze_kmh=0.0))
    pts = []
    for fx, fy, kw in [(0.1, 0.5, dict(inversion_alt_m=2200.0)), (0.9, 0.2, {}), (0.9, 0.8, {})]:
        lon, lat = terrain.xy_to_lonlat(fx * terrain.width, fy * terrain.height)
        pts.append(WeatherPoint(float(lon), float(lat), Weather(breeze_kmh=0.0, **kw)))
    params = dataclasses.replace(params, weather=pts[0].weather, weather_points=tuple(pts))
    lift = with_weather_points(lambda t, p: HeuristicLift(t, p))(terrain, params)
    planner = Planner(terrain, params, default_takeoff(terrain, params, lift), lift, verbose=False)
    package = build_flight_package(terrain, params, planner, candidate_routes(planner, n_triangles=1, n_free=1))
    jsonschema.validate(package, json.loads(SCHEMA.read_text()))
    slots = package["graph"]["slots"]
    given = [s for s in slots if "ceilings_m" in s]  # absent when every node has the same ceiling (morning)
    assert len(given) >= len(slots) // 2
    assert all(len(s["ceilings_m"]) == len(package["nodes"]) and s["ceiling_m"] == s["ceilings_m"][0] for s in given)
