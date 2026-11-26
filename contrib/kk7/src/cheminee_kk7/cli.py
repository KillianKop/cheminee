"""`cheminee-kk7 <command> ...`: cheminee flights with the thermal.kk7.ch climatology blended into the lift.

Thermal data: thermal.kk7.ch, CC BY-NC-SA 4.0 (non-commercial use only).
"""

from __future__ import annotations

import argparse
import os

from cheminee.cli import _add_flight_args, _flight_setup, add_evaluate_args

from . import fetch, tiles
from .lift import MODES, Kk7Lift, kk7_factory, layer_keys, load_layers

KK7_NOTICE = "Thermal data: thermal.kk7.ch, CC BY-NC-SA 4.0"
KK7_TERMS = (
    "thermal.kk7.ch data: CC BY-NC-SA 4.0, non-commercial use only. Tiles are downloaded over the DEM on first "
    "use and cached in data/kk7_cache."
)


def _add_kk7_args(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("kk7 (thermal.kk7.ch, CC BY-NC-SA 4.0, non-commercial)")
    g.add_argument(
        "--kk7-mode",
        default="boost",
        choices=MODES,
        help="boost: kk7 raises lift at known thermals, never removes any (default); "
        "blend: linear mix, lift fades away from kk7 spots",
    )
    g.add_argument(
        "--kk7-blend", type=float, default=0.8, help="weight of kk7 in [0, 1] (0 = cheminee only, default 0.8)"
    )
    g.add_argument(
        "--kk7-season", default="auto", choices=["auto", "all", *tiles.SEASONS], help="season layer (auto: from --day)"
    )
    g.add_argument(
        "--kk7-time",
        default="auto",
        choices=["auto", "all", *tiles.TIMES_OF_DAY],
        help="time-of-day layer (auto: interpolated from the hours after sunrise)",
    )
    g.add_argument("--kk7-zoom", type=int, default=tiles.MAX_ZOOM, help=f"tile zoom (max {tiles.MAX_ZOOM})")
    g.add_argument("--kk7-cache", default=fetch.DEFAULT_CACHE, help="tile cache directory")
    g.add_argument("--kk7-offline", action="store_true", help="use cached tiles only (no network)")


def _setup(args):
    terrain, params, factory, takeoff, goal = _flight_setup(args)
    factory = kk7_factory(
        factory,
        args.kk7_blend,
        args.kk7_season,
        args.kk7_time,
        args.kk7_zoom,
        args.kk7_cache,
        args.kk7_offline,
        args.kk7_mode,
    )
    return terrain, params, factory, takeoff, goal


def cmd_plan(args) -> None:
    from cheminee.planning import Planner, RouteOptimizer, default_takeoff
    from cheminee.viz.scene3d import fmt_hour

    terrain, params, factory, takeoff, goal = _setup(args)
    lift = factory(terrain, params)
    planner = Planner(terrain, params, takeoff or default_takeoff(terrain, params, lift), lift)
    optimizer = RouteOptimizer(planner)
    routes = optimizer.to_goal(goal) if goal else optimizer.optimise()
    if not routes:
        print("[plan] no flight found with this weather")
    for k, r in enumerate(routes):
        print(
            f"[route {k + 1}] {r.label:<13} {r.distance_km:6.1f} km {r.points:6.1f} pts  "
            f"{r.duration_h:4.1f} h  {r.speed_kmh:5.1f} km/h  {fmt_hour(r.times[0])}-{fmt_hour(r.times[-1])}"
        )
    print(f"[kk7] {KK7_NOTICE}")


def cmd_gui(args) -> None:
    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    from cheminee.viz.gui import App

    terrain, params, factory, takeoff, goal = _setup(args)
    app = App(terrain, params, factory, takeoff)
    app.fig.text(0.035, 0.01, KK7_NOTICE, fontsize=7, color="0.4")
    if goal:
        app.goal = goal
        app.compute()
    if args.save:
        app.fig.savefig(args.save, dpi=110)
        print(f"[gui] saved {args.save}")
    else:
        app.plt.show()


def kk7_lift_choice(args):
    """The kk7 lift model as an extra choice of the cheminee web interface (cheminee itself never imports kk7)."""
    from cheminee.web import LiftChoice

    def make(base_factory, options):
        return kk7_factory(
            base_factory,
            options["blend"],
            args.kk7_season,
            args.kk7_time,
            args.kk7_zoom,
            args.kk7_cache,
            args.kk7_offline,
            args.kk7_mode,
        )

    return LiftChoice(
        label=f"cheminee + kk7 thermal climatology ({args.kk7_mode})",
        make=make,
        options_spec={"blend": ("kk7 weight", "0-1", 0.0, 1.0, 0.05, args.kk7_blend)},
        attribution=KK7_NOTICE,
        note=KK7_TERMS,
    )


def cmd_serve(args) -> None:
    """cheminee web interface, with "cheminee + kk7" in the lift model choice."""
    from cheminee.cli import lift_factory
    from cheminee.domain import Terrain
    from cheminee.web import WebApp, serve

    terrain = Terrain.from_asc(args.dem)
    fetch_payload = None
    if not args.no_forecast:
        from cheminee.weather import openmeteo

        fetch_payload = openmeteo.fetch_payload
    app = WebApp(
        terrain,
        lift_factory(args.model),
        fetch_payload=fetch_payload,
        dem_name=os.path.basename(args.dem),
        lift_choices={"kk7": kk7_lift_choice(args)},
    )
    serve(app, args.port, args.open)


def cmd_fetch(args) -> None:
    """Download (once, then cache) the kk7 layers over the DEM, e.g. before flying offline."""
    from cheminee.domain import Terrain

    terrain = Terrain.from_asc(args.dem)
    seasons = list(tiles.SEASONS) if args.kk7_season == "auto" else [args.kk7_season]
    keys = [k for s in seasons for k in layer_keys(s, args.kk7_time)]
    load_layers(terrain, keys, args.kk7_zoom, args.kk7_cache, offline=False)
    print(f"[kk7] {len(keys)} layers cached in {args.kk7_cache}")


def cmd_compare(args) -> None:
    """Figure: base lift field, kk7 probability (+ hotspots) and blended field at one hour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    terrain, params, factory, _, _ = _setup(args)
    lift: Kk7Lift = factory(terrain, params)
    hour = args.hour
    base, prob, mixed = lift.base.field(hour), lift.probability(hour), lift.field(hour)
    extent = [0, terrain.width / 1000, 0, terrain.height / 1000]
    shade = terrain.hillshade()

    keys = list(lift.layer_weights(hour))
    main_key = max(lift.layer_weights(hour).items(), key=lambda kv: kv[1])[0]
    hotspots = fetch.fetch_hotspots(
        tiles.terrain_bbox(terrain), main_key, cache_dir=args.kk7_cache, offline=args.kk7_offline
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), constrained_layout=True)
    panels = [
        (base, f"cheminee ({lift.base.name}) lift index", "magma", 0, max(float(base.max()), 1e-3)),
        (prob, f"kk7 thermal probability ({' + '.join(keys)})", "turbo", 0, max(float(prob.max()), 1e-3)),
        (
            mixed,
            f"lift index with kk7 ({lift.mode}, weight {lift.blend:.2f})",
            "magma",
            0,
            max(float(mixed.max()), 1e-3),
        ),
    ]
    for ax, (grid, title, cmap, lo, hi) in zip(axes, panels):
        ax.imshow(shade, cmap="gray", extent=extent, origin="upper")
        im = ax.imshow(np.ma.masked_less(grid, 0.02 * hi), cmap=cmap, vmin=lo, vmax=hi, alpha=0.7, extent=extent)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x east [km]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("y north [km]")
    if hotspots:
        lonlat = np.array([f["geometry"]["coordinates"] for f in hotspots.get("features", [])]).reshape(-1, 2)
        p = np.array([f["properties"].get("probability", 0.0) for f in hotspots.get("features", [])])
        x, y = terrain.lonlat_to_xy(lonlat[:, 0], lonlat[:, 1])
        inside = terrain.contains(x, y)
        axes[1].scatter(x[inside] / 1000, y[inside] / 1000, s=20 + 200 * p[inside], facecolors="none", edgecolors="k")
        axes[1].set_title(axes[1].get_title() + f"\nrings: {inside.sum()} kk7 hotspots", fontsize=10)
    fig.suptitle(f"{os.path.basename(args.dem)}  -  day {params.day_of_year}, {hour:.1f} h local", fontsize=11)
    fig.text(0.01, 0.005, KK7_NOTICE, fontsize=7, color="0.4")
    fig.savefig(args.out, dpi=110)
    print(f"[kk7] comparison saved to {args.out}")


def cmd_evaluate(args) -> None:
    """`cheminee evaluate` with the kk7 climatology alone and blended into the analytical model as extra
    models, on the same validation flights."""
    from cheminee.cli import evaluate_setup
    from cheminee.lift import HeuristicLift
    from cheminee.training.evaluate import evaluate, print_scores

    from .lift import Kk7Factory, prefetch

    terrain, paths, models, cfg, validation = evaluate_setup(args)
    common = dict(blend=args.kk7_blend, mode=args.kk7_mode, zoom=args.kk7_zoom, cache_dir=args.kk7_cache)
    models["kk7 climatology"] = Kk7Factory(**common)
    models[f"heuristic + kk7 {args.kk7_mode}"] = Kk7Factory(HeuristicLift, **common)

    def fetch_tiles(windows) -> None:
        prefetch(windows, args.kk7_zoom, args.kk7_cache, args.kk7_offline, args.kk7_max_tiles)

    result = evaluate(terrain, paths, models, cfg, workers=args.workers, validation=validation, prepare=fetch_tiles)
    print_scores(result)
    print(f"[kk7] {KK7_NOTICE}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="cheminee-kk7", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("plan", help="compute flights with kk7 blended into the lift, and print them")
    _add_flight_args(sp)
    _add_kk7_args(sp)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("gui", help="cheminee 2D interface with kk7 blended into the lift")
    _add_flight_args(sp)
    _add_kk7_args(sp)
    sp.add_argument("--save", help="render to an image file instead of opening a window")
    sp.set_defaults(func=cmd_gui)

    sp = sub.add_parser("compare", help="figure: cheminee lift vs kk7 probability vs blend at one hour")
    _add_flight_args(sp)
    _add_kk7_args(sp)
    sp.add_argument("--hour", type=float, default=14.0, help="local hour")
    sp.add_argument("--out", default=os.path.join("data", "kk7_compare.png"))
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("serve", help="cheminee web interface with the kk7 lift model as a choice")
    sp.add_argument("dem", help="ESRI ASCII grid (.asc), WGS84 lon/lat")
    sp.add_argument("--model", help="ThermalNet checkpoint; without it the heuristic lift model is used")
    sp.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1")
    sp.add_argument("--open", action="store_true", help="open the page in the default browser")
    sp.add_argument("--no-forecast", action="store_true", help="disable Open-Meteo forecasts")
    _add_kk7_args(sp)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser(
        "evaluate", help="`cheminee evaluate` with the kk7 climatology (alone and blended) as extra models"
    )
    add_evaluate_args(sp)
    _add_kk7_args(sp)
    sp.add_argument(
        "--kk7-max-tiles", type=int, default=3000, help="stop before downloading more tiles than this (kk7 policy)"
    )
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("fetch", help="download the kk7 layers over a DEM into the cache (all seasons with auto)")
    sp.add_argument("dem")
    _add_kk7_args(sp)
    sp.set_defaults(func=cmd_fetch)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
