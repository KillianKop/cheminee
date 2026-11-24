"""Illustrate the speed-to-fly optimisation used by the planner for one leg.

python scripts/plot_speed_to_fly.py --out docs/img/speed_to_fly.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cheminee.domain import GliderPolar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/img/speed_to_fly.png")
    ap.add_argument("--distance-km", type=float, default=5.0)
    args = ap.parse_args()

    polar = GliderPolar()
    v = np.linspace(polar.points[0][0], polar.points[-1][0], 200) / 3.6  # airspeed, m/s
    sink = polar.sink(v)
    d = args.distance_km * 1000.0

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.3))
    a0.plot(v * 3.6, -sink, color="k")
    a0.scatter([p[0] for p in polar.points], [-p[1] for p in polar.points], color="k", zorder=3)
    a0.set(xlabel="airspeed [km/h]", ylabel="vertical speed [m/s]", title="Glider polar")
    a0.grid(alpha=0.3)

    def leg_minutes(climb, headwind):
        vg = v - headwind  # ground speed along the leg
        return (d / vg + d * sink / vg / climb) / 60.0

    for climb, color in ((1.0, "#1f77b4"), (2.0, "#f58231"), (3.5, "#e6194b")):
        t = leg_minutes(climb, 0.0)
        a1.plot(v * 3.6, t, color=color, label=f"next climb {climb:.1f} m/s")
        a1.scatter(v[t.argmin()] * 3.6, t.min(), color=color, zorder=3)
    a1.set(
        xlabel="airspeed [km/h]",
        ylabel="glide + climb back [min]",
        title=f"Time of a {args.distance_km:.0f} km leg, no wind",
    )
    a1.legend()
    a1.grid(alpha=0.3)

    for wind, color in ((-15.0, "#17a589"), (0.0, "#555555"), (15.0, "#911eb4")):
        t = leg_minutes(2.0, wind / 3.6)
        label = "no wind" if wind == 0 else f"{abs(wind):.0f} km/h {'headwind' if wind > 0 else 'tailwind'}"
        a2.plot(v * 3.6, t, color=color, label=label)
        a2.scatter(v[t.argmin()] * 3.6, t.min(), color=color, zorder=3)
    a2.set(xlabel="airspeed [km/h]", ylabel="glide + climb back [min]", title="Same leg, next climb 2.0 m/s")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.suptitle("Speed to fly: the dot is the airspeed chosen by the planner (minimum total time)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print(f"[plot] saved {args.out}")


if __name__ == "__main__":
    main()
