"""Planner with the weather at several points: a ceiling per node, a wind per leg, blended lift."""

import dataclasses

import numpy as np

from cheminee.lift import HeuristicLift
from cheminee.lift.points import PointsLift, with_weather_points
from cheminee.planning import Planner, RouteOptimizer, default_takeoff
from cheminee.planning.planner import heights_along
from cheminee.planning.sites import weather_sites
from cheminee.weather import Weather, WeatherPoint

HIGH = dict(breeze_kmh=0.0)  # default afternoon ceiling, about 2600 m
LOW = dict(breeze_kmh=0.0, inversion_alt_m=2200.0)


def _points(terrain, specs):
    """Weather points at (fraction of width, fraction of height, weather kwargs)."""
    out = []
    for k, (fx, fy, kw) in enumerate(specs):
        lon, lat = terrain.xy_to_lonlat(fx * terrain.width, fy * terrain.height)
        out.append(WeatherPoint(float(lon), float(lat), Weather(**kw), f"p{k}"))
    return tuple(out)


def _planner(terrain, params):
    lift = with_weather_points(lambda t, p: HeuristicLift(t, p))(terrain, params)
    return Planner(terrain, params, default_takeoff(terrain, params, lift), lift, verbose=False)


# -- planner ------------------------------------------------------------------------------------------- #
def test_heights_along_with_a_ceiling_per_sample():
    rng = np.random.default_rng(2)
    drop = np.cumsum(rng.normal(0, 30, 63))
    ceil = np.linspace(1700.0, 1450.0, 63)
    h, expected = 1600.0, []
    for d0, d1, c in zip(np.concatenate([[0.0], drop[:-1]]), drop, ceil):
        h = min(c, h - (d1 - d0))
        expected.append(h)
    assert np.allclose(heights_along(1600.0, drop, ceil), expected)


def test_same_weather_at_three_points_changes_nothing(terrain, params):
    one = _planner(terrain, params)
    three = _planner(terrain, dataclasses.replace(params, weather_points=_points(terrain, [(0.2, 0.2, HIGH)] * 3)))
    assert np.allclose(one.fields, three.fields, atol=1e-5)
    assert np.allclose(three.ceil_nodes, one.ceil_nodes[:, :1]) and np.allclose(one.ceil, three.ceil)


def test_each_thermal_climbs_to_its_own_ceiling(terrain, params):
    """Weather points with a low ceiling in the west and a high one in the east: the nodes of the west climb
    less high, every glide still clears the terrain, routes stay below the local ceiling."""
    pts = _points(terrain, [(0.1, 0.5, LOW), (0.9, 0.2, HIGH), (0.9, 0.8, HIGH)])
    params = dataclasses.replace(params, weather=pts[0].weather, weather_points=pts)
    pl = _planner(terrain, params)
    assert isinstance(pl.lift, PointsLift) and "3 weather points" in pl.lift.name
    k = pl.n_slots // 2
    west, east = pl.x < 0.3 * terrain.width, pl.x > 0.7 * terrain.width
    assert pl.ceil_nodes[k, west].max() < pl.ceil_nodes[k, east].min()
    assert pl.ceil_nodes.min() >= pts[0].weather.ceiling(pl.solar(pl.slot_hours[0])) - 1.0
    for kind in ("triangle", "free"):
        for r in RouteOptimizer(pl).optimise(kind)[:2]:
            for g in r.legs:
                assert pl.glide_clear(g, tolerance_m=1.0)
                assert pl.glide_heights(g).max() <= pl.ceil_along(g["slot"], g["i"], g["j"]).max() + 1e-6
            node = [int(np.argmin(np.hypot(pl.x - x, pl.y - y))) for x, y in r.profile[:, :2]]
            assert np.all(r.profile[:, 2] <= pl.ceil_nodes[:, node].max(axis=0) + 1.0)


def test_weather_sites_around_the_takeoff(terrain):
    sites = weather_sites(terrain, (0.2 * terrain.width, 0.3 * terrain.height))
    assert [r for *_, r in sites] == ["takeoff", "point 2", "point 3"]
    for x, y, _ in sites:
        assert terrain.contains(x, y)
    (x0, y0, _), (x1, y1, _), (x2, y2, _) = sites
    d = [np.hypot(x1 - x0, y1 - y0), np.hypot(x2 - x0, y2 - y0), np.hypot(x2 - x1, y2 - y1)]
    assert max(d) < 1.05 * min(d) and min(d) > 1000.0  # equilateral, not a dot
    manual = weather_sites(terrain, (1000.0, 1000.0), [(2000.0, 3000.0), (4000.0, 1500.0)])
    assert [(x, y) for x, y, _ in manual[1:]] == [(2000.0, 3000.0), (4000.0, 1500.0)]
    one = weather_sites(terrain, (1000.0, 1000.0), [(2000.0, 3000.0)])  # the other point stays automatic
    assert len(one) == 3 and one[1][:2] == (2000.0, 3000.0) and one[2] == weather_sites(terrain, (1000.0, 1000.0))[2]
