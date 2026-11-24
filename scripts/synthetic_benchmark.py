"""Self-consistency benchmark on synthetic data.

Compares, over the whole DEM and for conditions never used as such in training, the
HIDDEN lift model that generated the synthetic IGC files with (a) the trained network
and (b) the default heuristic. Only meaningful for a checkpoint trained on synthetic
tracks (`cheminee synth-igc`): with real IGC files there is no ground-truth field.

    python scripts/synthetic_benchmark.py DEM.asc models/thermal_net.pt --out docs/img/synthetic_benchmark.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cheminee.config import Params
from cheminee.domain import Terrain
from cheminee.igc.synthetic import TRUTH_CONFIG, SyntheticConfig, random_weather
from cheminee.lift import HeuristicLift
from cheminee.lift.neural import NeuralLift, load_checkpoint
from cheminee.weather import Weather


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dem")
    ap.add_argument("model")
    ap.add_argument("--out", default="docs/img/synthetic_benchmark.png")
    ap.add_argument("--n-conditions", type=int, default=40)
    args = ap.parse_args()

    terrain = Terrain.from_asc(args.dem)
    net, _ = load_checkpoint(args.model)
    rng = np.random.default_rng(123)

    def models(p):
        return HeuristicLift(terrain, p, TRUTH_CONFIG), NeuralLift(terrain, p, net), HeuristicLift(terrain, p)

    # 1. correlation with the hidden truth over random (day, hour, weather) conditions
    cfg = SyntheticConfig()
    r_nn, r_heur = [], []
    for _ in range(args.n_conditions):
        p = Params(day_of_year=int(rng.integers(100, 255)), weather=random_weather(rng, cfg))
        hour = float(rng.uniform(11, 18))
        truth, nn, heur = (m.field(hour).ravel()[::7] for m in models(p))
        if truth.std() < 1e-6:
            continue
        r_nn.append(np.corrcoef(truth, nn)[0, 1])
        r_heur.append(np.corrcoef(truth, heur)[0, 1])
    print(f"[benchmark] Pearson r with the hidden truth over {len(r_nn)} random conditions:")
    print(f"[benchmark]   neural network : mean {np.mean(r_nn):.3f} (min {np.min(r_nn):.3f})")
    print(f"[benchmark]   heuristic      : mean {np.mean(r_heur):.3f} (min {np.min(r_heur):.3f})")

    # 2. response to the weather: mean lift index of the 5 % best cells, one parameter at a time
    def top_mean(weather, hour=14.5):
        p = Params(day_of_year=180, weather=weather)
        return [float(np.sort(m.field(hour).ravel())[-len(terrain.z.ravel()) // 20 :].mean()) for m in models(p)]

    print("[benchmark] mean lift index of the best 5 % of the terrain at 14:30 (truth / network / heuristic):")
    for label, weather in (
        ("reference", Weather()),
        ("cloud cover 0.7", Weather(cloud_cover=0.7)),
        ("instability 0.75", Weather(thermal_strength=0.75)),
        ("wind 20 km/h", Weather(wind_speed_kmh=20.0)),
    ):
        print(f"[benchmark]   {label:<18} " + " / ".join(f"{v:.2f}" for v in top_mean(weather)))

    # 3. maps for one day, three hours
    p = Params(day_of_year=135, weather=Weather(wind_dir_deg=200.0, wind_speed_kmh=12.0, cloud_cover=0.3))
    hours = (11.0, 14.0, 17.0)
    extent = [0, terrain.width / 1000, 0, terrain.height / 1000]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12.5), sharex=True, sharey=True, constrained_layout=True)
    titles = ("hidden truth (generator)", "neural network (trained on IGC)", "default heuristic")
    for row, hour in enumerate(hours):
        fields = [m.field(hour) for m in models(p)]
        for col, f in enumerate(fields):
            ax = axes[row, col]
            im = ax.imshow(f, extent=extent, cmap="YlOrRd", vmin=0, vmax=0.8)
            r = np.corrcoef(fields[0].ravel(), f.ravel())[0, 1]
            suffix = "" if col == 0 else f"  (r = {r:.2f})"
            ax.set_title(f"{titles[col]} - {int(hour)}:00{suffix}", fontsize=10)
            if col == 0:
                ax.set_ylabel("y north [km]")
            if row == 2:
                ax.set_xlabel("x east [km]")
    fig.colorbar(im, ax=axes, shrink=0.5, label="lift index")
    fig.suptitle(
        f"Lift index, day {p.day_of_year}, wind from {p.weather.wind_dir_deg:.0f} deg at "
        f"{p.weather.wind_speed_kmh:.0f} km/h, cloud cover 0.3  |  mean r over {len(r_nn)} random conditions: "
        f"network {np.mean(r_nn):.2f}, heuristic {np.mean(r_heur):.2f}",
        fontsize=11,
    )
    fig.savefig(args.out, dpi=90)
    print(f"[benchmark] figure saved to {args.out}")


if __name__ == "__main__":
    main()
