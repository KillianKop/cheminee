import dataclasses

import numpy as np
import pytest

from cheminee.domain import GliderPolar
from cheminee.lift import HeuristicLift
from cheminee.weather import PRESETS, Weather
from cheminee.weather.openmeteo import build_url, sounding_from_payload, weather_from_payload

REAL_ANSWER = __import__("pathlib").Path(__file__).parent / "data" / "openmeteo_digne_2026-11-15.json"


def _real_payload():
    import json

    return json.loads(REAL_ANSWER.read_text())


def test_sounding_from_a_real_open_meteo_answer():
    """Digne-les-Bains, 15 Nov 2026: humid, low cloud base, WSW wind aloft."""
    s = sounding_from_payload(_real_payload())
    assert s.p_hpa.tolist() == [850.0, 700.0] and s.elevation_m == 603
    k = s.index(14.0)
    alt, temp, dew = s.profile(k)
    assert alt.tolist() == [603.0, 1484.0, 3049.0] and temp.tolist() == [11.6, 5.7, -2.7]
    assert s.cloud_base(k) == pytest.approx(603 + 125 * (11.6 - 8.8))
    assert s.ceiling(k) == pytest.approx(s.cloud_base(k) - 100)  # capped by the cloud base
    assert s.thermal_top(k) > s.cloud_base(k)  # the parcel keeps rising in cloud
    assert 0.3 <= s.strength(k) < 0.8  # shallow dry layer: weak thermals
    p_alt, p_temp = s.parcel(k)
    assert p_temp[0] == pytest.approx(13.6) and np.all(np.diff(p_temp) < 0)


def test_weather_from_a_real_open_meteo_answer(tmp_path):
    w = weather_from_payload(_real_payload(), lon=6.24)
    solar_noon_utc = 12.0 + 6.24 / 15.0
    assert np.hypot(*w.wind_vec(3053.0, solar_noon_utc)) * 3.6 == pytest.approx(37.9, abs=0.1)  # 700 hPa, 12 UTC
    assert np.hypot(*w.wind_vec(1490.0, solar_noon_utc)) * 3.6 == pytest.approx(4.7, abs=0.1)  # 850 hPa, 12 UTC
    blow_to = w.wind_vec(3053.0, solar_noon_utc)
    assert blow_to[0] > 0 and blow_to[1] > 0  # from 252 deg: toward the ENE
    assert 600 < w.ceiling(solar_noon_utc) < 1000 and w.thermal_strength < 0.8
    path = tmp_path / "w.json"
    w.save(str(path))
    assert Weather.load(str(path)) == w  # wind profile survives the JSON round trip


def test_archive_host_for_old_dates():
    import datetime as dt

    today = dt.date(2026, 11, 15)
    assert build_url(44.0, 6.2, dt.date(2026, 11, 14), today=today).startswith("https://api.open-meteo.com/")
    old = build_url(44.0, 6.2, dt.date(2025, 7, 1), today=today)
    assert old.startswith("https://historical-forecast-api.open-meteo.com/") and "temperature_850hPa" in old


def test_ceiling_follows_spread_inversion_and_time():
    w = Weather(temp_c=28, dewpoint_c=12, station_alt_m=600, inversion_alt_m=9000)
    assert w.cloud_base(15.0) == pytest.approx(600 + 125 * 16)
    assert w.ceiling(10.0) < w.ceiling(15.0) == pytest.approx(w.cloud_base(15.0) - w.base_margin_m)
    assert dataclasses.replace(w, inversion_alt_m=2000).ceiling(15.0) == 2000
    hourly = dataclasses.replace(w, hourly={"ceiling_m": [[10, 16], [1800, 2600]]})
    assert hourly.ceiling(13.0) == pytest.approx(2200)


def test_wind_profile_and_breeze():
    w = Weather(wind_dir_deg=270, wind_speed_kmh=10, wind_aloft_dir_deg=0, wind_aloft_speed_kmh=36)
    low, high = w.wind_vec(500.0), w.wind_vec(5000.0)
    assert np.allclose(low * 3.6, [10, 0], atol=1e-6)  # west wind blows toward the east
    assert np.allclose(high * 3.6, [0, -36], atol=1e-6)  # north wind blows toward the south
    assert w.wind_vec(np.array([500.0, 2000.0, 5000.0])).shape == (2, 3)
    assert w.breeze_ms(3.0) == 0 and w.breeze_ms(14.0) > w.breeze_ms(10.0) > 0


def test_weather_json_roundtrip_and_validation(tmp_path):
    path = tmp_path / "w.json"
    PRESETS["north_wind"].save(str(path))
    assert Weather.load(str(path)) == PRESETS["north_wind"]
    with pytest.raises(ValueError):
        Weather.from_dict({"wind_sped_kmh": 3})


def test_polar_is_monotonic_above_min_sink():
    polar = GliderPolar()
    sink = polar.sink(polar.speeds)
    assert np.all(np.diff(sink) > 0) and 7.5 < polar.trim_glide_ratio < 9.5


@pytest.mark.parametrize(
    "change, expected",
    [
        (dict(cloud_cover=0.9), "less"),
        (dict(thermal_strength=0.6), "less"),
        (dict(wind_speed_kmh=30.0), "less"),
        (dict(snow_line_m=0.0), "less"),
        (dict(thermal_strength=1.3), "more"),
    ],
)
def test_weather_changes_the_heuristic_lift(terrain, params, change, expected):
    ref = HeuristicLift(terrain, params).field(14.0).mean()
    other = dataclasses.replace(params, weather=dataclasses.replace(params.weather, **change))
    value = HeuristicLift(terrain, other).field(14.0).mean()
    assert value < ref if expected == "less" else value > ref


def test_cast_shadows_only_remove_light(terrain, params):
    assert terrain.shadow_mask(180.0, 60.0).mean() > terrain.shadow_mask(90.0, 8.0).mean()
    assert not terrain.shadow_mask(90.0, -5.0).any()
    shaded = dataclasses.replace(params, weather=dataclasses.replace(params.weather, cast_shadows=True))
    assert np.all(HeuristicLift(terrain, shaded).field(9.0) <= HeuristicLift(terrain, params).field(9.0) + 1e-6)


def test_open_meteo_payload_conversion():
    hours = list(range(24))
    payload = {
        "elevation": 650.0,
        "hourly": {
            "time": [f"2026-06-29T{h:02d}:00" for h in hours],
            "temperature_2m": [14 + 12 * np.sin(np.pi * (h - 5) / 14) if 5 <= h <= 19 else 14 for h in hours],
            "dew_point_2m": [10.0] * 24,
            "cloud_cover": [30] * 24,
            "wind_speed_10m": [8.0] * 24,
            "wind_direction_10m": [200] * 24,
            "wind_speed_700hPa": [25.0] * 24,
            "wind_direction_700hPa": [240] * 24,
            "boundary_layer_height": [300 + 150 * max(0, 8 - abs(h - 13)) for h in hours],
            "cape": [100.0] * 24,
            "freezing_level_height": [3900.0] * 24,
        },
    }
    w = weather_from_payload(payload, lon=6.2)
    assert set(w.hourly) == {
        "ceiling_m",
        "cloud_cover",
        "wind_speed_kmh",
        "wind_dir_deg",
        "wind_aloft_speed_kmh",
        "wind_aloft_dir_deg",
    }
    assert w.station_alt_m == 650.0 and w.cloud_cover == pytest.approx(0.3)
    assert 1000 < w.ceiling(13.5) <= 650 + 1500 and w.ceiling(13.5) > w.ceiling(8.0)
    assert np.hypot(*w.wind_vec(3000.0, 13.0)) * 3.6 == pytest.approx(25.0)
    assert "dew_point_2m" in build_url(44.0, 6.2, __import__("datetime").date(2026, 6, 29))


def test_forecast_not_covering_the_day_is_refused():
    """Regression: a high-resolution model whose range ends in the morning gave a ceiling at the station altitude
    all afternoon. Such an answer must be refused, not converted."""
    from cheminee.weather.openmeteo import ForecastUnavailable

    payload = _real_payload()
    hourly = payload["hourly"]
    cut = 10  # the model run ends at 10 UTC
    for key, values in hourly.items():
        if key != "time":
            hourly[key] = [v if k < cut else None for k, v in enumerate(values)]
    with pytest.raises(ForecastUnavailable, match="flying hours"):
        weather_from_payload(payload, lon=6.24)


def test_best_match_sends_no_model():
    import datetime as dt

    from cheminee.weather.openmeteo import FORECAST_MODELS, build_url

    day, today = dt.date(2026, 11, 27), dt.date(2026, 11, 25)
    assert "models=" not in build_url(44.0, 6.2, day, "best_match", today=today)
    assert "models=icon_d2" in build_url(44.0, 6.2, day, "icon_d2", today=today)
    assert all(len(v) == 5 for v in FORECAST_MODELS.values()) and "meteofrance_seamless" in FORECAST_MODELS
