"""Weather at three points of an IGC flight: blending weights, sidecar files, sites of a flight, fetch-weather,
training samples."""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from cheminee.igc import IgcTrack, read_igc, synthetic, write_igc
from cheminee.igc.sites import flight_sites
from cheminee.lift import features as ft
from cheminee.training.dataset import DatasetConfig, samples_from_track
from cheminee.weather import Weather, WeatherPoint, load_points, save_points
from cheminee.weather.points import nearest_point, point_weights

REAL_ANSWER = Path(__file__).parent / "data" / "openmeteo_digne_2026-11-15.json"
HIGH = dict(breeze_kmh=0.0)  # default afternoon ceiling, about 2600 m
LOW = dict(breeze_kmh=0.0, inversion_alt_m=2200.0)


# -- blending and files -------------------------------------------------------------------------------- #
def test_weights_sum_to_one_and_the_nearest_point_rules():
    px, py = np.array([0.0, 10000.0, 0.0]), np.array([0.0, 0.0, 10000.0])
    x, y = np.array([0.0, 5000.0, 9900.0]), np.array([0.0, 5000.0, 100.0])
    w = point_weights(px, py, x, y)
    assert w.shape == (3, 3) and np.allclose(w.sum(axis=0), 1.0)
    assert w[0, 0] > 0.9 and w[1, 2] > 0.9  # on a point: its weather
    assert list(nearest_point(px, py, x, y)) == [0, 0, 1]


def test_sidecar_files_old_and_new(tmp_path):
    old = tmp_path / "old.weather.json"
    Weather(temp_c=21.0).save(str(old))
    [single] = load_points(str(old))
    assert single.lon is None and single.weather.temp_c == 21.0
    new = tmp_path / "new.weather.json"
    pts = [WeatherPoint(6.2, 44.02, Weather(temp_c=20.0), "TP1", 11.5), WeatherPoint(6.24, 44.0, Weather(), "TP2")]
    save_points(str(new), pts, {"model": "x"})
    again = load_points(str(new))
    assert [(q.lon, q.lat, q.role, q.hour_utc) for q in again] == [(6.2, 44.02, "TP1", 11.5), (6.24, 44.0, "TP2", None)]
    assert again[0].weather.temp_c == 20.0 and json.loads(new.read_text())["meta"] == {"model": "x"}


# -- IGC: three points of a flight ------------------------------------------------------------------- #
def _track(corners_km, n_per_leg=120):
    """Track through the corners (km east, km north of 44 N 6.2 E), one fix every 10 s."""
    xy = np.vstack(
        [np.linspace(a, b, n_per_leg, endpoint=False) for a, b in zip(corners_km[:-1], corners_km[1:])]
        + [np.array(corners_km[-1:])]
    )
    lat = 44.0 + xy[:, 1] / 111.195
    lon = 6.2 + xy[:, 0] / (111.195 * np.cos(np.radians(44.0)))
    t = 10.0 * 3600.0 + 10.0 * np.arange(len(xy))
    return IgcTrack(dt.date(2026, 6, 20), t, lat, lon, np.full(len(xy), 1500.0))


def _km(track, idx):
    return np.array([(track.lon[idx] - 6.2) * 111.195 * np.cos(np.radians(44.0)), (track.lat[idx] - 44.0) * 111.195])


def test_triangle_flight_gives_its_turnpoints():
    corners = [(0.0, 0.0), (12.0, 2.0), (6.0, 11.0), (0.5, 0.5)]
    track = _track(corners)
    sites = flight_sites(track)
    assert [r for _, r in sites] == ["TP1", "TP2", "TP3"]
    assert [i for i, _ in sites] == sorted(i for i, _ in sites)  # in flight order
    for (idx, _), corner in zip(sites, [(0.0, 0.0), (12.0, 2.0), (6.0, 11.0)]):
        assert np.hypot(*(_km(track, idx) - corner)) < 0.7


@pytest.mark.parametrize("corners", [[(0, 0), (20, 4)], [(0, 0), (15, 0), (1, 0)]])
def test_straight_and_out_and_return_flights(corners):
    track = _track(corners)
    (i0, r0), (i1, r1), (i2, r2) = flight_sites(track)
    assert (r0, r1, r2) == ("start", "middle", "far") and i0 == 0
    far = _km(track, i2)
    assert np.hypot(*(far - corners[1])) < 0.3  # the farthest point from the takeoff
    assert np.hypot(*(_km(track, i1) - 0.5 * far)) < 0.3  # halfway to it


def test_fetch_weather_writes_three_rounded_points(tmp_path, monkeypatch):
    """One request per point (shared by the flights of the same day and place), positions rounded to ~2 km."""
    from cheminee import cli
    from cheminee.weather import openmeteo

    calls = []

    def fake(lat, lon, date, model):
        calls.append((lat, lon, date, model))
        return json.loads(REAL_ANSWER.read_text())

    monkeypatch.setattr(openmeteo, "fetch_payload", fake)
    track = _track([(0.0, 0.0), (12.0, 2.0), (6.0, 11.0), (0.5, 0.5)])
    write_igc(str(tmp_path / "a.igc"), track)
    write_igc(str(tmp_path / "b.IGC"), track)  # same flight twice: no new request
    cli.main(["fetch-weather", str(tmp_path), "--pause", "0"])
    assert len(calls) == 3
    for lat, lon, date, _ in calls:
        assert date == track.date
        assert abs(lat / 0.02 - round(lat / 0.02)) < 1e-6 and abs(lon / 0.02 - round(lon / 0.02)) < 1e-6
    points = load_points(str(tmp_path / "a.weather.json"))
    assert [q.role for q in points] == ["TP1", "TP2", "TP3"] and all(q.hour_utc is not None for q in points)
    meta = json.loads((tmp_path / "a.weather.json").read_text())["meta"]
    assert meta["position_rounding_deg"] == 0.02 and "Open-Meteo" in meta["source"]
    cli.main(["fetch-weather", str(tmp_path), "--pause", "0"])
    assert len(calls) == 3  # existing files are kept


def test_training_samples_take_the_weather_of_the_nearest_point(terrain, tmp_path):
    cfg = synthetic.SyntheticConfig(n_nodes=30, node_spacing_m=600.0, vary_weather=False)
    path = Path(synthetic.generate(terrain, str(tmp_path), 1, seed=5, cfg=cfg)[0])
    track = read_igc(str(path))
    x, y = terrain.lonlat_to_xy(track.lon, track.lat)
    west, east = np.argmin(x), np.argmax(x)
    points = [
        WeatherPoint(float(track.lon[west]), float(track.lat[west]), Weather(**LOW), "west"),
        WeatherPoint(float(track.lon[east]), float(track.lat[east]), Weather(**HIGH), "east"),
    ]
    stack = ft.static_stack(terrain)
    col = ft.FEATURE_NAMES.index("ceiling_above_ground_km")
    one = samples_from_track(track, terrain, stack, DatasetConfig(), np.random.default_rng(0), weather=Weather(**HIGH))
    two = samples_from_track(track, terrain, stack, DatasetConfig(), np.random.default_rng(0), weather=points)
    assert one[0].shape == two[0].shape and np.array_equal(one[1], two[1])
    lower = two[0][:, col] < one[0][:, col] - 1e-6
    assert 0 < lower.mean() < 1  # samples near the west point see its lower ceiling, the others do not
