# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).
The flight package has its own version number (`version` in the JSON, see [docs/flight-package.md](docs/flight-package.md)).

## [Unreleased]

### Added

- Local web interface (`cheminee serve DEM`): 3D terrain, takeoff picked by click, presets, Open-Meteo forecast
  and hand-set weather values, hourly weather preview, flight computation with live log, routes and lift field
  in 3D. Standard library HTTP server bound to 127.0.0.1, plotly.js served from the plotly package.
- Web map: flights drawn in 3D at their altitude (fill-extrusion ribbons for glides, columns for thermal climbs,
  translucent curtain to the ground), scaled with the relief exaggeration; the API gives each route a `track`
  densified every 60 m with the ground altitude.
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
