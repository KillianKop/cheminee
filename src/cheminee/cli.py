"""Command line entry point: `cheminee <command> ...` (see README)."""

from __future__ import annotations

import argparse
import datetime as dt
import os

from .config import OBJECTIVES, Params
from .weather import PRESETS, Weather

# CLI option -> Weather attribute
WEATHER_OPTIONS = {
    "wind_dir": ("wind_dir_deg", "low-level wind: direction it comes from (deg)"),
    "wind_speed": ("wind_speed_kmh", "low-level wind speed (km/h)"),
    "wind_aloft_dir": ("wind_aloft_dir_deg", "wind direction at 3000 m (deg)"),
    "wind_aloft_speed": ("wind_aloft_speed_kmh", "wind speed at 3000 m (km/h)"),
    "temp": ("temp_c", "afternoon temperature at the reference station (C)"),
    "dewpoint": ("dewpoint_c", "dew point at the reference station (C)"),
    "station_alt": ("station_alt_m", "altitude of the reference station (m)"),
    "inversion": ("inversion_alt_m", "top of the convective layer (m AMSL)"),
    "cloud_cover": ("cloud_cover", "cloud cover fraction (0-1)"),
    "strength": ("thermal_strength", "instability multiplier (about 0.5-1.4)"),
    "breeze": ("breeze_kmh", "peak up-slope / valley breeze (km/h)"),
    "snow_line": ("snow_line_m", "snow line (m AMSL)"),
    "overdev": ("overdev_hour", "solar hour after which flying stops (storms)"),
}


def _add_flight_args(sp: argparse.ArgumentParser) -> None:
    d = Params()
    sp.add_argument("dem", help="ESRI ASCII grid (.asc), WGS84 lon/lat")
    sp.add_argument("--model", help="ThermalNet checkpoint; without it the heuristic lift model is used")
    sp.add_argument("--takeoff", nargs=2, type=float, metavar=("LON", "LAT"))
    sp.add_argument(
        "--goal",
        nargs=2,
        type=float,
        metavar=("LON", "LAT"),
        help="compute the fastest route to this point instead of optimising a score",
    )
    sp.add_argument("--objective", choices=OBJECTIVES, default=d.objective)
    sp.add_argument(
        "--glide-lift",
        type=float,
        default=d.glide_lift_efficiency,
        help="share of the thermal core speed met gliding straight through lift zones (0: still air on glides)",
    )
    sp.add_argument("--date", help="YYYY-MM-DD (sets the day of year; needed by --forecast)")
    sp.add_argument("--day", type=int, default=d.day_of_year, help="day of year (1-365)")
    sp.add_argument("--utc-offset", type=float, default=d.utc_offset)
    sp.add_argument("--t-start", type=float, default=d.t_start, help="takeoff time, local hours")
    sp.add_argument("--t-end", type=float, default=d.t_end, help="latest landing time, local hours")
    g = sp.add_argument_group("weather")
    g.add_argument(
        "--weather", default="good_day", help=f"preset ({', '.join(PRESETS)}) or path of a weather JSON file"
    )
    g.add_argument(
        "--forecast", action="store_true", help="download the forecast of --date from Open-Meteo (needs network)"
    )
    g.add_argument("--forecast-json", metavar="FILE", help="use a saved Open-Meteo answer instead of downloading it")
    g.add_argument(
        "--forecast-model",
        default="meteofrance_seamless",
        help="Open-Meteo `models` value (default: Meteo-France AROME / ARPEGE); "
        "the web interface lists tested ones (weather.openmeteo.FORECAST_MODELS)",
    )
    for opt, (_, text) in WEATHER_OPTIONS.items():
        g.add_argument(f"--{opt.replace('_', '-')}", type=float, help=text)
    g.add_argument("--shadows", action="store_true", help="account for terrain cast shadows (slower)")


def _forecast_payload(args, lat: float, lon: float) -> dict | None:
    """Open-Meteo answer from a saved file (--forecast-json) or from the network (--forecast)."""
    if args.forecast_json:
        import json

        with open(args.forecast_json) as f:
            return json.load(f)
    if args.forecast:
        if not args.date:
            raise SystemExit("--forecast needs --date YYYY-MM-DD")
        from .weather import openmeteo

        return openmeteo.fetch_payload(lat, lon, dt.date.fromisoformat(args.date), args.forecast_model)
    return None


def build_weather(args, lat: float, lon: float) -> Weather:
    """Preset or file, then forecast, then explicit command line overrides."""
    if args.weather in PRESETS:
        weather = Weather.from_dict(PRESETS[args.weather].to_dict())
    elif os.path.exists(args.weather):
        weather = Weather.load(args.weather)
    else:
        raise SystemExit(f"Unknown weather preset or file: {args.weather}")
    payload = _forecast_payload(args, lat, lon)
    if payload is not None:
        from .weather import openmeteo

        try:
            weather = openmeteo.weather_from_payload(payload, lon, weather)
        except openmeteo.ForecastUnavailable as exc:
            raise SystemExit(f"{args.forecast_model}: {exc}. Choose a model with a longer range.") from None
    for opt, (attr, _) in WEATHER_OPTIONS.items():
        value = getattr(args, opt)
        if value is not None:
            weather.override(attr, value)  # an explicit value replaces the hourly series
    weather.cast_shadows = weather.cast_shadows or args.shadows
    return weather


def lift_factory(model_path: str | None):
    """Lift model factory `(terrain, params) -> LiftModel`: ThermalNet with a checkpoint, else the heuristic."""
    if model_path:
        from .lift.neural import NeuralLift, load_checkpoint

        net, _ = load_checkpoint(model_path)
        return lambda t, p: NeuralLift(t, p, net)
    from .lift import HeuristicLift

    return lambda t, p: HeuristicLift(t, p)


def _flight_setup(args):
    """Common wiring (composition root): terrain, params, lift factory, takeoff, goal."""
    from .domain import Terrain

    terrain = Terrain.from_asc(args.dem)
    day = dt.date.fromisoformat(args.date).timetuple().tm_yday if args.date else args.day
    params = Params(
        day_of_year=day,
        utc_offset=args.utc_offset,
        t_start=args.t_start,
        t_end=args.t_end,
        weather=build_weather(args, terrain.lat_mid, terrain.lon_mid),
        objective=args.objective,
        glide_lift_efficiency=args.glide_lift,
    )
    factory = lift_factory(args.model)
    to_xy = lambda lonlat: tuple(float(v) for v in terrain.lonlat_to_xy(*lonlat))  # noqa: E731
    return (
        terrain,
        params,
        factory,
        to_xy(args.takeoff) if args.takeoff else None,
        to_xy(args.goal) if args.goal else None,
    )


def _plan(args):
    from .planning import Planner, RouteOptimizer, default_takeoff
    from .viz.scene3d import fmt_hour

    terrain, params, factory, takeoff, goal = _flight_setup(args)
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
    return terrain, params, planner, routes


def cmd_plan(args) -> None:
    _plan(args)


def cmd_gui(args) -> None:
    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    from .viz.gui import App

    terrain, params, factory, takeoff, goal = _flight_setup(args)
    app = App(terrain, params, factory, takeoff)
    if goal:
        app.goal = goal
        app.compute()
    if args.save:
        app.fig.savefig(args.save, dpi=110)
        print(f"[gui] saved {args.save}")
    else:
        app.plt.show()


def cmd_view3d(args) -> None:
    from .viz.scene3d import build_scene, write_html

    terrain, params, planner, routes = _plan(args)
    write_html(build_scene(terrain, planner, routes, params, stride=args.stride), args.out, offline=not args.cdn)


def cmd_weather(args) -> None:
    """Print the resolved weather and optionally save it as a JSON file."""
    from .domain import Terrain

    terrain = Terrain.from_asc(args.dem)
    weather = build_weather(args, terrain.lat_mid, terrain.lon_mid)
    p = Params(utc_offset=args.utc_offset)
    for key, value in weather.to_dict().items():
        if key != "hourly":
            print(f"  {key:<24} {value}")
    print(f"  hourly series            {sorted(weather.hourly) or 'none'}")
    print("  local hour  ceiling [m]  wind low / aloft [km/h]  cloud shading  breeze [km/h]")
    for hour in range(9, 20):
        s = p.to_solar(hour, terrain.lon_mid)
        low = float((weather.wind_vec(weather.wind_alt_m, s) ** 2).sum() ** 0.5) * 3.6
        high = float((weather.wind_vec(weather.wind_aloft_alt_m, s) ** 2).sum() ** 0.5) * 3.6
        print(
            f"  {hour:>10}  {weather.ceiling(s):>11.0f}  {low:>10.0f} / {high:<10.0f}  "
            f"{1 - weather.sun_factor(s):>13.2f}  {weather.breeze_ms(s) * 3.6:>13.1f}"
        )
    if args.out:
        weather.save(args.out)
        print(f"[weather] saved to {args.out}")


def cmd_sounding(args) -> None:
    """Analyse the model sounding of the day and draw its emagram."""
    from .domain import Terrain
    from .viz.emagram import plot_emagram
    from .weather import openmeteo

    terrain = Terrain.from_asc(args.dem)
    lon, lat = args.takeoff if args.takeoff else (terrain.lon_mid, terrain.lat_mid)
    payload = _forecast_payload(args, lat, lon)
    if payload is None:
        raise SystemExit("sounding needs --forecast (with --date) or --forecast-json FILE")
    sounding = openmeteo.sounding_from_payload(payload)
    if sounding is None:
        raise SystemExit("The forecast answer has no pressure-level data")
    if args.save_json:
        import json

        with open(args.save_json, "w") as f:
            json.dump(payload, f)
        print(f"[sounding] raw answer saved to {args.save_json}")
    print(f"[sounding] ground {sounding.elevation_m:.0f} m, levels {sounding.p_hpa.astype(int).tolist()} hPa")
    print("  UTC  cloud base  thermal top  ceiling  strength  wind by level [km/h @ m]")
    for k, hour in enumerate(sounding.hours_utc):
        if 7 <= hour <= 18:
            alt, speed, _ = sounding.wind_levels(k)
            winds = "  ".join(f"{s:.0f}@{a:.0f}" for a, s in zip(alt, speed))
            print(
                f"  {hour:>3.0f}  {sounding.cloud_base(k):>10.0f}  {sounding.thermal_top(k):>11.0f}  "
                f"{sounding.ceiling(k):>7.0f}  {sounding.strength(k):>8.2f}  {winds}"
            )
    plot_emagram(sounding, args.hour_utc, args.out)


SITE_ROUND_DEG = 0.02  # positions sent to Open-Meteo are rounded (~2 km): the route of a flight stays private


def cmd_fetch_weather(args) -> None:
    """Download the archived weather of each IGC flight at three points of it (`igc.sites.flight_sites`: the
    turnpoints of a triangle, or start / middle / far point of a straight cross) and write its `.weather.json`."""
    import time

    from .igc import find_igc, read_igc
    from .igc.sites import flight_sites
    from .weather import WeatherPoint, openmeteo, save_points

    paths = find_igc(args.igc_dir)
    payloads: dict[tuple, dict] = {}  # flights of the same day and place share their requests
    done = requests = 0
    for path in paths:
        sidecar = path.with_suffix(".weather.json")
        if sidecar.exists() and not args.overwrite:
            continue
        try:
            track = read_igc(str(path))
            points = []
            for idx, role in flight_sites(track):
                lat = round(round(float(track.lat[idx]) / SITE_ROUND_DEG) * SITE_ROUND_DEG, 4)
                lon = round(round(float(track.lon[idx]) / SITE_ROUND_DEG) * SITE_ROUND_DEG, 4)
                key = (track.date, args.forecast_model, lat, lon)
                if key not in payloads:
                    if requests:
                        time.sleep(args.pause)
                    payloads[key] = openmeteo.fetch_payload(lat, lon, track.date, args.forecast_model)
                    requests += 1
                try:
                    weather = openmeteo.weather_from_payload(payloads[key], lon, Weather(breeze_kmh=0.0))
                except openmeteo.ForecastUnavailable as exc:
                    print(f"[fetch-weather] {path.name} {role}: {exc}")
                    continue
                hour = round(float(track.t[idx]) / 3600.0, 3)
                points.append(WeatherPoint(lon, lat, weather, role, hour_utc=hour))
            if not points:
                continue
            url = openmeteo.build_url(points[0].lat, points[0].lon, track.date, args.forecast_model)
            meta = dict(
                source="Open-Meteo " + ("historical forecast" if url.startswith(openmeteo.ARCHIVE_URL) else "forecast"),
                model=args.forecast_model,
                date=track.date.isoformat(),
                position_rounding_deg=SITE_ROUND_DEG,
            )
            save_points(str(sidecar), points, meta)
            done += 1
        except Exception as exc:  # one failing flight must not stop the batch
            print(f"[fetch-weather] {path.name}: {exc}")
    print(f"[fetch-weather] {done} weather files written for {len(paths)} flights, {requests} Open-Meteo requests")


def cmd_export(args) -> None:
    """Compute flights and write the flight package read by the vautour app."""
    from .export import build_flight_package, candidate_routes, write_flight_package
    from .export.flight_package import COPERNICUS_DEM_NOTICE, OPEN_METEO_NOTICE
    from .planning import Planner, default_takeoff

    terrain, params, factory, takeoff, _ = _flight_setup(args)
    lift = factory(terrain, params)
    planner = Planner(terrain, params, takeoff or default_takeoff(terrain, params, lift), lift)
    routes = candidate_routes(planner, args.triangles, args.free)
    if not routes:
        raise SystemExit("No flight found with this weather: nothing to export")
    attribution = [args.dem_attribution or COPERNICUS_DEM_NOTICE]
    if args.forecast or args.forecast_json:
        attribution.append(OPEN_METEO_NOTICE)
    date = dt.date.fromisoformat(args.date) if args.date else None
    package = build_flight_package(terrain, params, planner, routes, attribution, date, with_graph=not args.no_graph)
    write_flight_package(package, args.out)


def cmd_serve(args) -> None:
    """Local web interface: 3D map, takeoff by click, weather, flights."""
    from .domain import Terrain
    from .web import WebApp, serve

    terrain = Terrain.from_asc(args.dem)
    fetch_payload = None
    if not args.no_forecast:
        from .weather import openmeteo

        fetch_payload = openmeteo.fetch_payload
    app = WebApp(terrain, lift_factory(args.model), fetch_payload=fetch_payload, dem_name=os.path.basename(args.dem))
    serve(app, args.port, args.open)


def cmd_synth(args) -> None:
    from .domain import Terrain
    from .igc import synthetic

    synthetic.generate(Terrain.from_asc(args.dem), args.out, args.n, seed=args.seed)


def cmd_train(args) -> None:
    from .domain import Terrain
    from .igc import find_igc
    from .training import TrainConfig, build_dataset, train
    from .training.train import plot_report

    paths = find_igc(args.igc_dir)
    if not paths:
        raise SystemExit(f"No IGC file found in {args.igc_dir}")
    dataset = build_dataset(Terrain.from_asc(args.dem), paths)
    _, history = train(dataset, TrainConfig(epochs=args.epochs, lr=args.lr, seed=args.seed), args.out)
    if args.report:
        plot_report(history, args.report)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="cheminee", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("gui", help="2D interactive interface")
    _add_flight_args(sp)
    sp.add_argument("--save", help="render to an image file instead of opening a window")
    sp.set_defaults(func=cmd_gui)

    sp = sub.add_parser("plan", help="compute flights and print them (no graphics)")
    _add_flight_args(sp)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("view3d", help="compute flights and export an interactive 3D HTML scene")
    _add_flight_args(sp)
    sp.add_argument("--out", default="xc_flights_3d.html")
    sp.add_argument("--stride", type=int, default=3, help="DEM decimation factor of the 3D surface")
    sp.add_argument("--cdn", action="store_true", help="load plotly.js from a CDN (smaller file, needs network)")
    sp.set_defaults(func=cmd_view3d)

    sp = sub.add_parser("weather", help="show (and save) the weather resulting from the options")
    _add_flight_args(sp)
    sp.add_argument("--out", help="save the weather as a JSON file usable with --weather")
    sp.set_defaults(func=cmd_weather)

    sp = sub.add_parser("sounding", help="analyse the model sounding of the day and draw its emagram")
    _add_flight_args(sp)
    sp.add_argument("--hour-utc", type=float, default=13.0, help="hour of the emagram (UTC)")
    sp.add_argument("--out", default="emagram.png")
    sp.add_argument("--save-json", help="also save the raw Open-Meteo answer (reusable with --forecast-json)")
    sp.set_defaults(func=cmd_sounding)

    sp = sub.add_parser("fetch-weather", help="download the archived weather of every IGC flight of a directory")
    sp.add_argument("igc_dir")
    sp.add_argument("--forecast-model", default="meteofrance_seamless")
    sp.add_argument("--overwrite", action="store_true", help="replace existing .weather.json files")
    sp.add_argument("--pause", type=float, default=0.5, help="seconds between two requests")
    sp.set_defaults(func=cmd_fetch_weather)

    sp = sub.add_parser("export", help="write the flight package (JSON) read by the vautour navigation app")
    _add_flight_args(sp)
    sp.add_argument("--out", default="flight.json", help="output file")
    sp.add_argument("--triangles", type=int, default=3, help="number of triangles in the package")
    sp.add_argument("--free", type=int, default=1, help="number of free-distance flights in the package")
    sp.add_argument("--dem-attribution", help="attribution notice of the DEM (default: Copernicus DEM notice)")
    sp.add_argument("--no-graph", action="store_true", help="leave out the routing graph (smaller file)")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("serve", help="local web interface: 3D map, takeoff by click, weather, flights")
    sp.add_argument("dem", help="ESRI ASCII grid (.asc), WGS84 lon/lat")
    sp.add_argument("--model", help="ThermalNet checkpoint; without it the heuristic lift model is used")
    sp.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1")
    sp.add_argument("--open", action="store_true", help="open the page in the default browser")
    sp.add_argument("--no-forecast", action="store_true", help="disable Open-Meteo forecasts (no network at all)")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("synth-igc", help="generate synthetic IGC files (pipeline testing)")
    sp.add_argument("dem")
    sp.add_argument("--out", default="data/igc_synthetic")
    sp.add_argument("-n", type=int, default=300, help="number of flights")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("train", help="train the thermal network from a directory of IGC files")
    sp.add_argument("dem")
    sp.add_argument("igc_dir")
    sp.add_argument("--out", default="models/thermal_net.pt")
    sp.add_argument("--epochs", type=int, default=40)
    sp.add_argument("--lr", type=float, default=2e-3)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--report", help="save a training report figure (PNG)")
    sp.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
