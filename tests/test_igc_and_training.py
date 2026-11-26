import json

import numpy as np
import pytest

from cheminee.config import Params
from cheminee.igc import detect_thermals, estimate_wind, find_igc, read_igc, synthetic, write_igc
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


def test_find_igc_lists_each_file_once(tmp_path):
    # *.igc and *.IGC match the same files on Windows / macOS: each flight must still be listed once
    (tmp_path / "2024").mkdir()
    for name in ("a.igc", "2024/b.IGC", "2024/c.Igc"):
        (tmp_path / name).write_text("")
    (tmp_path / "a.weather.json").write_text("{}")
    found = find_igc(tmp_path)
    assert [p.name for p in found] == ["b.IGC", "c.Igc", "a.igc"]
    assert find_igc(tmp_path / "missing") == []


def test_evaluate_ranks_models_on_the_validation_flights(terrain, igc_dir):
    """`evaluate` scores every model on the same held-out flights; a model that knows the true lift (the
    heuristic that generated the synthetic tracks) must rank the real thermals well above chance, and a flat
    field must not."""
    import numpy as np

    from cheminee.lift import HeuristicLift
    from cheminee.training.evaluate import EvalConfig, crop, evaluate
    from cheminee.training.train import validation_flights

    class Flat:
        name = "flat"

        def __init__(self, t, p):
            self.shape = t.z.shape

        def field(self, hour):
            return np.full(self.shape, 0.3, dtype=np.float32)

    paths = sorted(igc_dir.glob("*.igc"))
    models = {"heuristic": lambda t, p: HeuristicLift(t, p), "flat": Flat}
    res = evaluate(terrain, paths, models, EvalConfig(val_fraction=0.5, tile_m=3000.0, margin_m=1000.0))
    s = res["scores"]
    assert res["n_flights"] == len(validation_flights(range(8), 0.5, 0)) and res["n_samples"] > 50
    assert s["heuristic"]["n_thermals"] > 0 and s["heuristic"]["rank_percentile"] > 0.6
    assert s["flat"]["rank_percentile"] == 0.5 and np.isnan(s["flat"]["pearson"])
    assert set(s) == {"heuristic", "flat", "constant (mean target)"}
    # parallel workers (picklable factories) give the same scores
    two = evaluate(terrain, paths, {"heuristic": HeuristicLift}, EvalConfig(0.5, 0, 3000.0, 1000.0), workers=2)
    assert two["scores"]["heuristic"] == pytest.approx(s["heuristic"], nan_ok=True)
    win = crop(terrain, 1000.0, 2000.0, 3000.0, 5000.0)  # a window keeps the geographic position of the cells
    lon, lat = terrain.xy_to_lonlat(2000.0, 3500.0)
    assert np.isclose(win.sample(win.z, *win.lonlat_to_xy(lon, lat)), terrain.sample(terrain.z, 2000.0, 3500.0), atol=1)


def test_near_track_background_samples(terrain, igc_dir):
    """`background="near_track"`: the pseudo-absences lie within the radius of where the pilot was at their time,
    away from the thermals; the default draws them anywhere on the DEM, as before."""
    import dataclasses

    import numpy as np

    from cheminee.training.dataset import BACKGROUND, THERMAL, DatasetConfig, flight_samples

    track = read_igc(str(igc_dir / "synthetic_0000.igc"))
    x, y = terrain.lonlat_to_xy(track.lon, track.lat)
    cfg = DatasetConfig(background="near_track", bg_radius_m=1500.0, bg_min_dist_m=300.0)
    near = flight_samples(track, terrain, cfg, np.random.default_rng(0))
    bg = near.kind == BACKGROUND
    assert bg.sum() > 0 and np.all(near.weight[bg] == cfg.bg_weight) and np.all(near.target[bg] == 0)
    for bx, by, bt in zip(near.x[bg], near.y[bg], near.t_s[bg]):
        i = int(np.argmin(np.abs(track.t - bt)))
        assert np.hypot(bx - x[i], by - y[i]) <= 1500.0 + 1e-6
    th = near.kind == THERMAL
    d = np.hypot(near.x[bg][:, None] - near.x[th][None], near.y[bg][:, None] - near.y[th][None])
    assert d.min() >= 300.0 - 1e-6
    default = flight_samples(track, terrain, DatasetConfig(), np.random.default_rng(0))
    assert np.ptp(default.x[default.kind == BACKGROUND]) > np.ptp(near.x[bg])  # anywhere on the map
    with pytest.raises(ValueError):
        flight_samples(track, terrain, dataclasses.replace(cfg, background="nope"), np.random.default_rng(0))


def test_neural_field_on_a_mask_and_masked_evaluation(terrain, igc_dir, tmp_path):
    """A neural field computed on a mask has the values of the full field there, and `evaluate` (which computes
    the network on the cells it reads only) gives the same scores as with full fields."""
    import numpy as np
    import torch

    from cheminee.lift.neural import NeuralFactory, NeuralLift, ThermalNet, save_checkpoint
    from cheminee.training.evaluate import EvalConfig, evaluate

    torch.manual_seed(0)
    net = ThermalNet(len(ft.FEATURE_NAMES)).eval()
    lift = NeuralLift(terrain, Params(), net)
    mask = np.zeros(terrain.z.shape, dtype=bool)
    mask[50:60, 30:200] = True
    mask[180, 100] = True
    assert np.allclose(lift.field(14.0, mask=mask)[mask], lift.field(14.0)[mask], atol=1e-6)

    ckpt = str(tmp_path / "net.pt")
    save_checkpoint(ckpt, net, (64, 64, 32), {})

    class Full:  # the same network, always on the whole grid
        def __init__(self, t, p):
            self.lift = NeuralFactory(ckpt)(t, p)

        def field(self, hour):
            return self.lift.field(hour)

    cfg = EvalConfig(val_fraction=0.5, tile_m=3000.0, margin_m=1000.0)
    paths = sorted(igc_dir.glob("*.igc"))
    s = evaluate(terrain, paths, {"masked": NeuralFactory(ckpt), "full": Full}, cfg)["scores"]
    assert s["masked"] == pytest.approx(s["full"], abs=1e-6)


def test_validation_flights_recorded_by_train(terrain, igc_dir, tmp_path):
    """`train` records the validation flights (indices and a fingerprint of the file names, not the names);
    `evaluate` then reads only those flights and gets the same scores as when it rebuilds the split."""
    import torch

    from cheminee.cli import main
    from cheminee.lift import HeuristicLift
    from cheminee.lift.neural import load_checkpoint
    from cheminee.training.dataset import flights_fingerprint
    from cheminee.training.evaluate import EvalConfig, evaluate

    dem = tmp_path / "dem.asc"
    with open(dem, "w") as f:
        f.write(f"ncols {terrain.nx}\nnrows {terrain.ny}\nxllcorner {terrain.lon0}\nyllcorner {terrain.lat0}\n")
        f.write(f"cellsize {terrain.cell_deg}\n")
        np.savetxt(f, terrain.z, fmt="%.2f")
    ckpt = tmp_path / "net.pt"
    torch.manual_seed(0)
    main(["train", str(dem), str(igc_dir), "--out", str(ckpt), "--epochs", "2"])
    val = load_checkpoint(str(ckpt))[1]["validation"]
    paths = sorted(igc_dir.glob("*.igc"))
    assert val["paths_sha256"] == flights_fingerprint(paths) and val["n_paths"] == len(paths)
    assert all(p.stem not in json.dumps(val) for p in paths)  # no file name stored
    cfg = EvalConfig(tile_m=3000.0, margin_m=1000.0)
    rebuilt = evaluate(terrain, paths, {"h": HeuristicLift}, cfg)
    recorded = evaluate(terrain, paths, {"h": HeuristicLift}, cfg, validation=val["flights"])
    assert recorded["n_flights"] == rebuilt["n_flights"] == len(val["flights"])
    assert recorded["scores"]["h"] == pytest.approx(rebuilt["scores"]["h"], nan_ok=True)
