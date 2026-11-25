# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).
The flight package has its own version number (`version` in the JSON, see [docs/flight-package.md](docs/flight-package.md)).

## [Unreleased]

### Added

- Web interface: replay timeline under the map (play / pause, speed, follow): flown part bright, glider with a
  line to the ground, altitude, height above ground, climb rate and distance flown; lift overlay and ceiling
  follow the replay time. The 3D track carries the hour of every point.
- Web interface: extra lift models plugged in by the caller (`web.LiftChoice`), offered in a "Lift model"
  menu with their options, licence note and map attribution. `cheminee-kk7 serve` (contrib) offers the kk7
  climatology this way; cheminee itself does not depend on it.

### Fixed

- Triangle search: the takeoff was only left at the first possible time, so a slow morning (no thermal reachable
  yet) gave no triangle at all. Later departures are now tried every `Params.triangle_departure_step_h` (30 min)
  when they reach new thermals, and the results are merged. Example: forecast of 27 November 2026 near Digne,
  0 -> 1 triangle (17 km); unchanged results on the `good_day` preset (search time 0.3 -> 0.7 s).

### Added

- Local web interface (`cheminee serve DEM`): 3D terrain, takeoff picked by click, presets, Open-Meteo forecast
  and hand-set weather values, hourly weather preview, flight computation with live log, routes and lift field
  in 3D. Standard library HTTP server bound to 127.0.0.1, plotly.js served from the plotly package.
- Web map: ceiling of the selected hour drawn as a translucent plane over the computed area, hourly ceiling in
  the altitude profile, relief at true scale by default (x1.5 made heights look 1.5 times bigger), day of year
  and UTC offset hint under the date.
- Web map: flights drawn in 3D at their altitude by a WebGL custom layer (`static/track3d.js`): smooth line of
  constant screen width, spirals in the thermals, translucent curtain to the ground, aligned with the relief
  through MapLibre's terrain matrix and scaled with the relief exaggeration (flat on the map in 2D mode); the API
  gives each route a `track` densified every 60 m with the ground altitude.
- Web map with MapLibre GL JS 4.7.1 (vendored, BSD-3) instead of plotly: background map (OpenTopoMap, OpenStreetMap
  or none), 3D relief and hillshade from elevation tiles cut on the fly from the planner's DEM (Terrain-RGB,
  `/tiles/dem/{z}/{x}/{y}.png`), computed area outlined and its surroundings darkened, lift overlay per hour,
  thermal nodes coloured by climb rate, altitude profile of each route, `/api/elevation`. Page 1.6 MB -> 235 kB.
- Web interface speed: lift fields computed in parallel and cached per weather / day, identical requests answered
  from memory, gzip, ETag / browser cache for plotly.js and the DEM grid (5.1 MB -> 1.6 MB for a first load, 0 for
  a reload), long polling instead of a 600 ms poll.
- `Weather.override(name, value)`: an explicit value replaces the hourly series of that quantity (used by the
  command line and the web interface).

## [0.1.0] - 2026-11-24

First public version.

### Added

- Terrain from ESRI ASCII DEMs, sun geometry, glider polar, 3-point CFD scoring.
- Weather of the day: two-level or full-profile wind, breezes, cloud base, inversion, cloud cover, instability,
  snow line, cast shadows, overdevelopment, hourly series, presets and JSON files.
- Open-Meteo / Météo-France forecasts (`--forecast`, `--forecast-json`), sounding analysis by the parcel method
  and emagram (`cheminee sounding`), archived weather for IGC flights (`cheminee fetch-weather`).
- Analytical lift model and ThermalNet, a neural lift model trained on IGC tracks (`cheminee train`), with a
  synthetic track generator (`cheminee synth-igc`) and a self-consistency benchmark.
- Planner with hourly slots and MacCready speed to fly; optimiser for free distance, flat and FAI triangles and
  flight to a goal.
- matplotlib GUI, plotly 3D scene.
- Flight package format v1 (`cheminee export`) with its JSON schema, for the vautour navigation app.
