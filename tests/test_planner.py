import dataclasses

import numpy as np
import pytest

from cheminee.lift import HeuristicLift
from cheminee.planning import Planner, RouteOptimizer, default_takeoff
from cheminee.weather import Weather


def _planner(terrain, params):
    lift = HeuristicLift(terrain, params)
    return Planner(terrain, params, default_takeoff(terrain, params, lift), lift, verbose=False)


def _check(route, planner, terrain):
    assert route.nodes[0] == 0
    assert np.all(np.diff(route.times) > 0) and route.times[-1] <= planner.t_end + 1e-9
    assert route.distance_km > 0 and route.points > 0
    prof = route.profile
    assert np.all(np.diff(prof[:, 3]) >= -1e-9), "profile times must not go back"
    assert prof[:, 2].max() <= planner.ceil.max() + 1e-6
    assert np.all(prof[:, 2] >= terrain.sample(terrain.z, prof[:, 0], prof[:, 1]) - 1.0)


def test_heuristic_lift_follows_the_sun(terrain, params):
    lift = HeuristicLift(terrain, params)
    east, west = terrain.normal[0] > 0.3, terrain.normal[0] < -0.3
    morning, evening = lift.field(10.0), lift.field(18.5)
    assert morning[east].mean() > morning[west].mean()
    assert evening[west].mean() > evening[east].mean()
    assert lift.field(3.0).max() == 0.0  # night


def test_free_distance_routes(terrain, params):
    planner = _planner(terrain, params)
    routes = RouteOptimizer(planner).optimise("free")
    assert routes and all(r.kind == "free" and len(r.turnpoints) == 5 for r in routes)
    for r in routes:
        _check(r, planner, terrain)
    assert routes == sorted(routes, key=lambda r: -r.distance_km)


def test_triangles_are_closed_and_fai_rule_holds(terrain, params):
    planner = _planner(terrain, params)
    opt = RouteOptimizer(planner)
    triangles = opt.optimise("triangle")
    assert triangles
    for r in triangles:
        _check(r, planner, terrain)
        tp = [r.nodes[k] for k in r.turnpoints]
        sides = [planner.dist[tp[0], tp[1]], planner.dist[tp[1], tp[2]], planner.dist[tp[2], tp[0]]]
        assert r.distance_km == pytest.approx(sum(sides) / 1000)
        gap = min(planner.dist[r.nodes[-1], 0], planner.dist[r.nodes[-1], tp[0]])
        assert gap <= params.closing_fraction * sum(sides) + 1e-6
        assert (r.kind == "fai") == (min(sides) >= params.fai_min_leg * sum(sides))
        assert r.points == pytest.approx(r.distance_km * (params.coef_fai if r.kind == "fai" else params.coef_triangle))
    assert all(r.kind == "fai" for r in opt.optimise("fai"))
    best = opt.optimise("score")
    assert best[0].points >= max(triangles[0].points, opt.optimise("free")[0].points) - 1e-6


def test_goal_route_is_the_fastest_path(terrain, params):
    planner = _planner(terrain, params)
    opt = RouteOptimizer(planner)
    target = int(np.argmax(planner.dist[0]))
    (route,) = opt.to_goal((planner.x[target], planner.y[target]))
    _check(route, planner, terrain)
    arr, _ = opt.earliest(0, planner.launch_hour(), first=True)
    # the search time includes a climb at the goal; the flight itself ends on arrival, no later
    assert route.kind == "goal" and route.times[-1] <= arr[route.nodes[-1]] + 1e-9
    assert route.times[-1] == pytest.approx(route.profile[-1, 3])


def test_tailwind_is_faster_than_headwind(terrain, params):
    """Nodes north of the takeoff are reached earlier with a south wind aloft than with a north wind."""
    calm = dataclasses.replace(params, weather=Weather(wind_speed_kmh=0.0, breeze_kmh=0.0))
    lift = HeuristicLift(terrain, calm)  # same thermals in both cases: only the glide wind differs

    def arrivals(direction):
        w = Weather(
            wind_dir_deg=direction,
            wind_speed_kmh=0.0,
            wind_aloft_dir_deg=direction,
            wind_aloft_speed_kmh=25.0,
            breeze_kmh=0.0,
        )
        planner = Planner(
            terrain,
            dataclasses.replace(params, weather=w),
            (terrain.width * 0.29, terrain.height * 0.15),
            lift,
            verbose=False,
        )
        return planner, RouteOptimizer(planner).earliest(0, planner.launch_hour(), first=True)[0]

    planner, south = arrivals(180.0)
    _, north = arrivals(0.0)
    common = np.isfinite(south) & np.isfinite(north) & (planner.y > planner.y[0] + 1500.0)
    assert common.any()
    assert np.all(south[common] < north[common])


@pytest.mark.parametrize(
    "weather",
    [
        Weather(wind_speed_kmh=34.0),
        Weather(cloud_cover=1.0, thermal_strength=0.5),
        Weather(temp_c=15.0, dewpoint_c=13.0),
    ],
)
def test_bad_weather_prevents_flying(terrain, params, weather):
    planner = _planner(terrain, dataclasses.replace(params, weather=weather))
    assert RouteOptimizer(planner).optimise("score") == []


def test_overdevelopment_shortens_the_day(terrain, params):
    stormy = dataclasses.replace(params, weather=dataclasses.replace(params.weather, overdev_hour=14.0))
    planner = _planner(terrain, stormy)
    assert planner.t_end < params.t_end
    assert all(r.times[-1] <= planner.t_end + 1e-9 for r in RouteOptimizer(planner).optimise("free"))


def test_triangles_can_leave_the_takeoff_later(terrain, params):
    """Regression: the triangle search used to leave only at the first possible time. On a slow November
    morning that start reaches no thermal, while leaving later gives triangles."""
    p = dataclasses.replace(
        params, day_of_year=330, utc_offset=1.0, t_start=9.5, t_end=16.0, weather=Weather(breeze_kmh=0.0)
    )
    lift = HeuristicLift(terrain, p)
    takeoff = default_takeoff(terrain, p, lift)
    first_only = Planner(terrain, dataclasses.replace(p, triangle_departure_step_h=0.0), takeoff, lift, verbose=False)
    assert RouteOptimizer(first_only).optimise("triangle") == []
    planner = Planner(terrain, p, takeoff, lift, verbose=False)
    triangles = RouteOptimizer(planner).optimise("triangle")
    assert triangles
    for r in triangles:
        _check(r, planner, terrain)
        assert r.kind in ("triangle", "fai")
        assert r.times[0] > planner.launch_hour()  # they leave after the first possible time


def test_final_glide_climbs_only_what_it_needs(terrain, params):
    """Regression: the last thermal was climbed to the ceiling and the flight ended high above the goal, with a
    final climb counted in its duration. Now the last climb stops at the height the final glide needs."""
    old = _planner(terrain, dataclasses.replace(params, final_glide=False))
    new = _planner(terrain, params)
    for kind in ("triangle", "free"):
        before = RouteOptimizer(old).optimise(kind)[0]
        after = new.make_route(
            list(before.nodes), list(before.times), before.kind, before.distance_km * 1000.0, before.turnpoints
        )
        _check(after, new, terrain)
        assert after.times[-1] == pytest.approx(after.profile[-1, 3])  # the flight ends on arrival
        assert after.times[-1] <= before.times[-1] + 1e-9 and after.distance_km == before.distance_km
        j = after.nodes[-1]
        arrival = after.profile[-1, 2]
        assert arrival >= new.zg[j] + params.arrival_margin_m - 1.0  # still the safety margin
        assert arrival <= before.profile[-1, 2] + 1e-6
        # the last glide stays clear of the terrain
        (x0, y0, z0, _), (x1, y1, z1, _) = after.profile[-2], after.profile[-1]
        f = np.linspace(0, 1, 50)
        ground = terrain.sample(terrain.z, x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
        assert np.all(z0 + (z1 - z0) * f >= ground - 1.0)
    # on this terrain at least one route saves time by not climbing to the ceiling at the end
    tri_old = RouteOptimizer(old).optimise("triangle")[0]
    tri_new = new.make_route(
        list(tri_old.nodes), list(tri_old.times), tri_old.kind, tri_old.distance_km * 1000.0, tri_old.turnpoints
    )
    assert tri_new.profile[-2, 2] < new.ceil.max() - 1.0 or tri_new.times[-1] < tri_old.profile[-1, 3]
