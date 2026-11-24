import numpy as np
import pytest

from cheminee.config import Params
from cheminee.igc import detect_thermals, estimate_wind, read_igc, synthetic, write_igc
from cheminee.lift import features as ft
from cheminee.lift.neural import NeuralLift, load_checkpoint
from cheminee.training import TrainConfig, build_dataset, train
from cheminee.training.dataset import weather_from_track
from cheminee.weather import Weather


@pytest.fixture(scope="module")
def igc_dir(terrain, tmp_path_factory):
    out = tmp_path_factory.mktemp("igc")
    cfg = synthetic.SyntheticConfig(n_nodes=30, node_spacing_m=600.0, vary_weather=False)
    paths = synthetic.generate(terrain, str(out), 8, seed=3, cfg=cfg)
    assert len(paths) == 8
    return out


def test_igc_roundtrip(igc_dir, tmp_path):
    track = read_igc(str(igc_dir / "synthetic_0000.igc"))
    copy = tmp_path / "copy.igc"
    write_igc(str(copy), track)
    again = read_igc(str(copy))
    assert again.date == track.date and len(again) == len(track)
    assert np.allclose(again.lat, track.lat, atol=2e-5) and np.allclose(again.alt, track.alt, atol=1.0)


def test_thermals_and_wind_are_recovered(terrain, igc_dir):
    manifest = np.loadtxt(igc_dir / "manifest.csv", delimiter=",", skiprows=1, usecols=(2, 3))
    checked = 0
    for k, (wind_dir, wind_kmh) in enumerate(manifest):
        track = read_igc(str(igc_dir / f"synthetic_{k:04d}.igc"))
        x, y = terrain.lonlat_to_xy(track.lon, track.lat)
        thermals = detect_thermals(track.t, x, y, track.alt)
        if len(thermals) >= 3 and wind_kmh > 5:
            est_dir, est_ms = estimate_wind(thermals)
            assert abs((est_dir - wind_dir + 180) % 360 - 180) < 20
            assert abs(est_ms * 3.6 - wind_kmh) < 4
            checked += 1
    assert checked >= 1


def test_weather_sidecar_is_used(terrain, igc_dir):
    path = igc_dir / "synthetic_0000.igc"
    assert Weather.load(str(path.with_suffix(".weather.json"))).breeze_kmh == 0.0
    track = read_igc(str(path))
    x, y = terrain.lonlat_to_xy(track.lon, track.lat)
    fallback = weather_from_track(track, detect_thermals(track.t, x, y, track.alt))
    assert fallback.ceiling(14.0) == pytest.approx(np.percentile(track.alt, 99))


def test_feature_matrix_shape(terrain):
    stack = ft.static_stack(terrain)
    static = ft.sample_static(terrain, stack, np.array([1000.0, 2000.0]), np.array([1500.0, 3000.0]))
    X = ft.assemble(static, 180, np.array([9.0, 14.0]), terrain.lat_mid, terrain.lon_mid, Weather())
    assert X.shape == (2, len(ft.FEATURE_NAMES)) and np.isfinite(X).all()


def test_training_pipeline_end_to_end(terrain, igc_dir, tmp_path):
    ds = build_dataset(terrain, list(igc_dir.glob("*.igc")))
    assert len(ds) > 200 and ds.X.shape[1] == len(ft.FEATURE_NAMES)
    assert 0.0 <= ds.y.min() and ds.y.max() <= 1.0
    ckpt = tmp_path / "net.pt"
    _, history = train(ds, TrainConfig(epochs=8), str(ckpt))
    assert history["train_loss"][-1] < history["train_loss"][0] and np.isfinite(history["best_val_loss"])
    net, meta = load_checkpoint(str(ckpt))
    field = NeuralLift(terrain, Params(), net).field(14.0)
    assert field.shape == terrain.z.shape and 0.0 <= field.min() and field.max() <= 1.0
