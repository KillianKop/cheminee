import itertools

import numpy as np

from cheminee.domain import sun_position, three_tp_distance


def test_sun_is_south_and_high_at_solar_noon_in_june():
    az, el = sun_position(172, 12.0 - 6.2 / 15.0, 44.0, 6.2)  # ~solar noon in UTC
    assert abs(az - 180.0) < 3.0
    assert abs(el - (90.0 - 44.0 + 23.44)) < 1.0


def test_sun_rises_east_sets_west_and_is_vectorised():
    az, el = sun_position(172, np.array([5.0, 17.5]), 44.0, 6.2)
    assert 45 < az[0] < 110 and 250 < az[1] < 315
    assert (el > 0).all()
    assert sun_position(172, 23.0, 44.0, 6.2)[1] < 0  # night


def test_terrain_coordinates_roundtrip(terrain):
    x, y = terrain.rowcol_to_xy(10, 20)
    r, c = terrain.xy_to_rowcol(x, y)
    assert np.allclose([r, c], [10, 20])
    lon, lat = terrain.xy_to_lonlat(x, y)
    assert np.allclose(terrain.lonlat_to_xy(lon, lat), [x, y])
    assert np.isclose(terrain.sample(terrain.z, x, y), terrain.z[10, 20], atol=1e-2)


def test_three_tp_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10_000, (12, 2))
    dist = np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1))
    path = [int(v) for v in rng.integers(0, 12, 9)]
    best, chain = three_tp_distance(path, dist)
    brute = max(
        sum(dist[path[a], path[b]] for a, b in zip(c[:-1], c[1:]))
        for c in itertools.combinations_with_replacement(range(9), 5)
    )
    assert np.isclose(best, brute)
    assert chain == sorted(chain) and len(chain) == 5
