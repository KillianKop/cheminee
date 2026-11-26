"""cheminee_kk7 tests: no network, synthetic tiles only (no kk7 data is stored in the repository)."""

from pathlib import Path

import numpy as np
import pytest
from cheminee_kk7 import fetch, tiles
from cheminee_kk7.lift import Kk7Lift, kk7_factory, layer_keys
from PIL import Image

from cheminee.config import Params
from cheminee.domain import Terrain
from cheminee.lift import HeuristicLift
from cheminee.weather import Weather


@pytest.fixture(scope="module")
def terrain() -> Terrain:
    """Same analytic DEM as the cheminee tests (~7.5 x 7.5 km near Digne)."""
    n = 240
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    z = (
        700.0
        + 1100.0 * np.exp(-(((j - 70) / 28.0) ** 2)) * (0.75 + 0.25 * np.cos(i / 25.0))
        + 900.0 * np.exp(-(((j - 170) / 30.0) ** 2 + ((i - 120) / 90.0) ** 2))
        + 60.0 * np.sin(i / 9.0) * np.cos(j / 11.0)
    )
    return Terrain(z, lon0=6.2, lat0=44.0, cellsize_deg=1.0 / 3600.0)


@pytest.fixture
def params() -> Params:
    return Params(n_nodes=30, node_spacing_m=600.0, weather=Weather(breeze_kmh=0.0))


def synthetic_tiles(terrain: Terrain, zoom: int, func) -> dict:
    """Decoded tiles whose probability is func(lon, lat) at every pixel centre."""
    out = {}
    for x, y in tiles.tiles_covering(*tiles.terrain_bbox(terrain), zoom):
        n = tiles.TILE_PX * 2**zoom
        px = (x * tiles.TILE_PX + np.arange(tiles.TILE_PX) + 0.5) / n
        py = (y * tiles.TILE_PX + np.arange(tiles.TILE_PX) + 0.5) / n
        lon = px * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * py))))
        out[(x, y)] = tiles.decode_tile(tiles.encode_tile(func(lon[None, :], lat[:, None])))
    return out


def test_decode_inverts_encode():
    p = np.linspace(0.0, tiles.RAMP_P_MAX, 400).reshape(20, 20)
    back = tiles.decode_tile(tiles.encode_tile(p))
    assert back.dtype == np.float32 and back.shape == p.shape
    assert np.abs(back - p).max() < 0.01
    assert back[0, 0] == 0.0  # transparent pixel


def test_decode_real_tile_colours():
    # colours seen on kk7 tiles: fade-in (0, 0, a/2, a), then the jet-like ramp
    rgba = np.array([[[0, 0, 0, 0], [0, 0, 56, 112], [0, 0, 255, 255], [255, 211, 0, 255], [128, 0, 0, 255]]], np.uint8)
    p = tiles.decode_tile(rgba)[0]
    assert p[0] == 0.0
    assert np.all(np.diff(p) > 0)
    assert p[-1] == pytest.approx(tiles.RAMP_P_MAX)


def test_tile_maths():
    x, y = tiles.lonlat_to_pixel(0.0, 0.0, 3)
    assert (float(x), float(y)) == pytest.approx((4 * 256, 4 * 256))
    assert tiles.tms_row(1488, 12) == 2607  # row of the Digne tile used to check the kk7 URLs
    cover = tiles.tiles_covering(6.148, 43.985, 6.373, 44.146, 12)
    assert {(2118, 1488), (2119, 1487)} <= set(cover)
    assert len(cover) == len(set(cover)) <= 20


def test_reprojection_follows_lonlat(terrain):
    zoom = 12
    func = lambda lon, lat: 0.4 * (lon - 6.2) / 0.07 * np.ones_like(lat)  # noqa: E731
    grid = tiles.to_terrain_grid(terrain, synthetic_tiles(terrain, zoom, func), zoom)
    assert grid.shape == (terrain.ny, terrain.nx)
    x, _ = terrain.rowcol_to_xy(0, np.arange(terrain.nx))
    lon, _ = terrain.xy_to_lonlat(x, 0.0)
    expected = np.clip(0.4 * (lon - 6.2) / 0.07, 0, tiles.RAMP_P_MAX)
    assert np.abs(grid[terrain.ny // 2] - expected).max() < 0.02
    # north / south: a latitude gradient must not be flipped
    func_lat = lambda lon, lat: 0.4 * (lat - 44.0) / 0.07 * np.ones_like(lon)  # noqa: E731
    grid = tiles.to_terrain_grid(terrain, synthetic_tiles(terrain, zoom, func_lat), zoom)
    assert grid[5, terrain.nx // 2] > grid[-5, terrain.nx // 2]  # row 0 = north edge


def test_missing_tiles_are_zero(terrain):
    cover = tiles.tiles_covering(*tiles.terrain_bbox(terrain), 12)
    grid = tiles.to_terrain_grid(terrain, dict.fromkeys(cover), 12)
    assert not grid.any()


def test_time_filter():
    assert tiles.nearest_season(196) == "jul"
    assert tiles.nearest_season(360) == "jan"  # circular year
    for h in (-2.0, 3.5, 6.0, 8.0, 9.7, 15.0):
        w = tiles.time_of_day_weights(h)
        assert sum(w.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in w.values())
    assert tiles.time_of_day_weights(8.0)["07"] == pytest.approx(1.0)
    assert tiles.time_of_day_weights(0.0)["04"] == pytest.approx(1.0)


def _layers(terrain, season="jul"):
    xs = np.linspace(0, 1, terrain.nx)[None, :] * np.ones((terrain.ny, 1))
    return {k: (0.1 + 0.3 * xs * (i + 1) / 3).astype(np.float32) for i, k in enumerate(layer_keys(season, "auto"))}


@pytest.mark.parametrize("mode", ["boost", "blend"])
def test_blend_limits(terrain, params, mode):
    base = HeuristicLift(terrain, params)
    layers = _layers(terrain)
    same = Kk7Lift(terrain, params, base, layers, blend=0.0, mode=mode)
    assert np.array_equal(same.field(13.0), base.field(13.0))
    full = Kk7Lift(terrain, params, base, layers, blend=1.0, mode=mode)
    f, b = full.field(13.0), base.field(13.0)
    assert f.shape == b.shape and f.dtype == np.float32
    assert 0.0 <= f.min() and f.max() <= 1.0
    # kk7 keeps the amplitude of the base model (no climb rate stronger than the day allows) ...
    assert np.percentile(f, 99.9) == pytest.approx(np.percentile(b, 99.9), rel=0.05)
    # ... and moves lift towards the kk7 spots (synthetic layers: max on the east edge)
    assert f[:, -10:].mean() > b[:, -10:].mean()
    assert not full.field(1.0).any()  # night: no lift whatever kk7 says
    with pytest.raises(ValueError):
        Kk7Lift(terrain, params, base, layers, blend=1.5)
    with pytest.raises(ValueError):
        Kk7Lift(terrain, params, base, layers, mode="replace")
    with pytest.raises(ValueError):
        Kk7Lift(terrain, params, base, {}, blend=0.5)


def test_boost_never_removes_lift(terrain, params):
    """kk7 is transparent where nobody flies: missing data must not erase the base lift."""
    base = HeuristicLift(terrain, params)
    layers = {k: np.zeros((terrain.ny, terrain.nx), np.float32) for k in layer_keys("jul", "auto")}
    for v in layers.values():
        v[100:110, 100:110] = 0.4  # one known thermal, nothing elsewhere
    lift = Kk7Lift(terrain, params, base, layers, blend=1.0, mode="boost")
    f, b = lift.field(13.0), base.field(13.0)
    assert np.all(f >= b)
    assert f[105, 105] >= np.percentile(b, 99.9) * 0.99
    assert np.array_equal(f[:90], b[:90])


def test_no_kk7_data_falls_back_to_base(terrain, params):
    base = HeuristicLift(terrain, params)
    layers = {k: np.zeros((terrain.ny, terrain.nx), np.float32) for k in layer_keys("jul", "auto")}
    lift = Kk7Lift(terrain, params, base, layers, blend=0.7)
    assert lift.blend == 0.0
    assert np.array_equal(lift.field(13.0), base.field(13.0))


def test_planner_runs_with_kk7(terrain, params):
    from cheminee.planning import Planner, default_takeoff

    lift = Kk7Lift(terrain, params, HeuristicLift(terrain, params), _layers(terrain), blend=0.5)
    planner = Planner(terrain, params, default_takeoff(terrain, params, lift), lift, verbose=False)
    assert planner.n > 1
    assert np.all(np.isfinite(planner.climb))


def test_offline_cache_no_network(terrain, params, tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", no_network)
    zoom = 12
    cover = tiles.tiles_covering(*tiles.terrain_bbox(terrain), zoom)
    rgba = tiles.encode_tile(np.full((256, 256), 0.3))
    for x, y in cover:  # pretend the tiles were downloaded earlier
        path = Path(fetch.tile_path(str(tmp_path), "jul_07", zoom, x, y))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, "RGBA").save(path)
    factory = kk7_factory(lambda t, p: HeuristicLift(t, p), 0.5, "jul", "07", zoom, str(tmp_path), offline=True)
    lift = factory(terrain, params)
    assert lift.probability(13.0).mean() == pytest.approx(0.3, abs=0.01)
    # a layer that is not cached gives no data offline, without any request
    factory = kk7_factory(lambda t, p: HeuristicLift(t, p), 0.5, "jan", "07", zoom, str(tmp_path), offline=True)
    assert factory(terrain, params).blend == 0.0


def test_kk7_lift_choice_in_the_web_interface(terrain, params, tmp_path, monkeypatch):
    """`cheminee-kk7 serve` offers kk7 as a lift model of the cheminee page, offline here (cached tiles)."""
    import argparse
    import time

    from cheminee_kk7.cli import KK7_NOTICE, kk7_lift_choice

    from cheminee.web import WebApp

    monkeypatch.setattr(
        fetch.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network"))
    )
    zoom = 12
    cover = tiles.tiles_covering(*tiles.terrain_bbox(terrain), zoom)
    rgba = tiles.encode_tile(np.full((256, 256), 0.3))
    for key in layer_keys("jul", "auto"):
        for x, y in cover:
            path = Path(fetch.tile_path(str(tmp_path), key, zoom, x, y))
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba, "RGBA").save(path)
    args = argparse.Namespace(
        kk7_blend=0.8,
        kk7_season="auto",
        kk7_time="auto",
        kk7_zoom=zoom,
        kk7_cache=str(tmp_path),
        kk7_offline=True,
        kk7_mode="boost",
    )
    app = WebApp(terrain, lambda t, p: HeuristicLift(t, p), params, lift_choices={"kk7": kk7_lift_choice(args)})
    assert app.config()["lift_models"]["kk7"]["attribution"] == KK7_NOTICE
    job_id = app.start_plan({"objective": "free", "lift": {"model": "kk7", "options": {"blend": 0.5}}})
    while app.job(job_id)["status"] in ("queued", "running"):
        time.sleep(0.05)
    job = app.job(job_id)
    assert job["status"] == "done", job.get("error")
    assert "kk7" in job["result"]["lift_name"] and job["result"]["lift_attribution"] == KK7_NOTICE


def test_evaluation_factory_and_prefetch(terrain, params, tmp_path, monkeypatch):
    """`prefetch` downloads the tiles of the windows once; `Kk7Factory` then reads them offline, survives
    pickling (parallel workers) and gives the climatology alone or blended into a base model."""
    import io
    import pickle

    from cheminee_kk7.lift import Kk7Climatology, Kk7Factory, prefetch

    buf = io.BytesIO()
    Image.fromarray(tiles.encode_tile(np.full((256, 256), 0.3)), "RGBA").save(buf, format="PNG")
    calls = []
    monkeypatch.setattr(fetch, "_get", lambda url, accept=None: calls.append(url) or buf.getvalue())
    windows = [(*tiles.terrain_bbox(terrain), params.day_of_year)]
    prefetch(windows, 12, str(tmp_path))
    n = len(calls)
    keys = layer_keys(tiles.nearest_season(params.day_of_year), "auto")
    assert n == len(keys) * len(tiles.tiles_covering(*tiles.terrain_bbox(terrain), 12))
    prefetch(windows, 12, str(tmp_path))
    assert len(calls) == n  # cached: no second request
    alone = pickle.loads(pickle.dumps(Kk7Factory(zoom=12, cache_dir=str(tmp_path))))(terrain, params)
    assert isinstance(alone, Kk7Climatology)
    assert alone.field(14.0).mean() == pytest.approx(0.3, abs=0.01)
    blended = pickle.loads(pickle.dumps(Kk7Factory(HeuristicLift, 0.5, zoom=12, cache_dir=str(tmp_path))))
    lift = blended(terrain, params)
    assert isinstance(lift, Kk7Lift) and np.all(lift.field(14.0) >= lift.base.field(14.0) - 1e-6)
    assert len(calls) == n  # offline by default
