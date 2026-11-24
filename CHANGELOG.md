# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).
The flight package has its own version number (`version` in the JSON, see [docs/flight-package.md](docs/flight-package.md)).

## [Unreleased]

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
