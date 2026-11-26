# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).
The flight package has its own version number (`version` in the JSON, see [docs/flight-package.md](docs/flight-package.md)).

## [Unreleased]

### Added

- `cheminee evaluate DEM IGC_DIR --model CKPT`: scores the analytical lift model and trained networks on the
  flights held out by `train` (same flights and split, shared through `training.dataset.iter_flights`), with the
  fields the planner sees: Pearson r, MAE and cross-entropy on the measured samples, and the rank of the real
  thermals among the ground around them (`training/evaluate.py`). Flights are scored in parallel processes
  (`--workers`, a quarter of the logical CPUs by default) on small DEM windows, so memory stays low.
- Lift features: the sun position is computed once per distinct hour instead of once per cell (identical
  values); a field of the neural lift model is about 30 % faster, in the planner too.
- Weather points. With a forecast, the planner takes the weather at 1 to 10 points: the takeoff and points
  placed by the user, or around the takeoff (3 by default: a triangle; `--weather-points N`,
  `--weather-point LON LAT` repeated). Web: *Click adds a weather point* places numbered blue points on the map
  like the takeoff, draggable, removed by a right click; automatic ones are dashed; each shows its ceiling. The
  forecasts of the points are downloaded a few at once. Between the points the weather is blended
  (`weather/points.py`): the lift field of each point weather (`lift/points.py`, for any lift model), a ceiling
  per thermal node, a wind per glide; the leg cost climbs back to the ceiling of the arrival node. A single
  weather gives exactly the former results.
- `fetch-weather` takes the weather of each IGC flight at three points of it (`igc/sites.py`: turnpoints of a
  triangle, or start / middle / farthest point), positions rounded to 0.02° and requests shared by the flights
  of the same day and place; the `.weather.json` file then holds the points and the source (`save_points`;
  files with one `Weather` are still read). Training gives each sample the weather of the nearest point.
- Flight package (still version 1): optional `graph.slots[].ceilings_m`, the ceiling of every node when the
  weather points give different ceilings; `ceiling_m` is the one of the takeoff. Readers that ignore it get leg
  durations off by the difference of ceilings divided by the climb rate.
- `data/dem/cop30/` for your own larger Copernicus DEM extracts (ignored by git, see `data/dem/README.md`).
- `data/igc/` for your own IGC tracks and `models/local/` for the networks trained on them, both ignored by
  git (only `data/igc/README.md` is versioned); `*.igc` is now ignored everywhere in the repository.
- Web interface: replay timeline under the map (play / pause, speed, follow): flown part bright, glider with a
  line to the ground, altitude, height above ground, climb rate and distance flown; lift overlay and ceiling
  follow the replay time. The 3D track carries the hour of every point.
- Web interface: extra lift models plugged in by the caller (`web.LiftChoice`), offered in a "Lift model"
  menu with their options, licence note and map attribution. `cheminee-kk7 serve` (contrib) offers the kk7
  climatology this way; cheminee itself does not depend on it.

- Weather models: tested Open-Meteo models listed in `weather.openmeteo.FORECAST_MODELS` (grid, approximate
  range, vertical profile); the web interface offers them in a menu, warns when the date is beyond the range
  and compares them all for a date (`POST /api/weather/compare`, one cached request per model).

- Planner: rising air on glides. In the lift zones met along a glide (working ridge, line of thermals) the
  pilot slows down to minimum sink and the air lifts the glider (`Params.glide_lift_efficiency`, option
  `--glide-lift`, web *Lift on glides*; 0.5 by default): glides lose less height and can gain some, never above
  the ceiling. Terrain clearance is checked on the real altitude curve, which the 3D track now shows. Sample
  DEM, June: free distance 87 -> 90 km, 13.2 -> 14.6 km/h, 25 -> 21 climbs.
- Flight package (still version 1): a glide segment can gain height, its `sink_ms` is then negative and
  `a1 > a0` (schema description and doc updated; readers must not assume that glides go down). Heights lost
  in the routing graph stay `>= 0`.
- Planner: climbs shared out like a pilot (`Params.partial_climbs`): climb only what is needed to reach a
  stronger thermal or the goal, cross weak thermals without circling, wait for a thermal (or a cloud base)
  that is not ready yet, final glide at the safety margin. The search
  keeps 3 neighbours per sector instead of 2 (`neighbours_per_sector`) to skip weak thermals. Sample DEM: June
  triangle 3.9 -> 3.3 h (14.9 -> 17.5 km/h), June free distance 80 -> 87 km, 27 November triangle 17 -> 24 km.

### Changed

- `cheminee plan / export --forecast` takes the forecast at the takeoff when one is given (it was the centre of
  the map), and at three points by default.
- The sample DEM moved to `data/dem/sample/`: extract it with
  `tar -xzf data/dem/sample/rasters_COP30_sample.tar.gz -C data/dem/sample/`, then use
  `data/dem/sample/output_hh.asc`.

### Fixed

- `train` and `fetch-weather` listed every IGC file twice on case-insensitive file systems (Windows, macOS:
  `*.igc` and `*.IGC` match the same files), so each flight was trained on twice and could fall in both the
  training and validation sets. New `igc.find_igc` lists each file once.
- The final glide added earlier in this release checked the terrain under the glide with the height lost
  from the end instead of from the start: a ridge near the goal could be passed below the clearance. Fixed in
  the shared-out climbs, with a test of the clearance along every glide.
- Final glide: the last thermal was climbed to the ceiling and routes ended far above the goal (640 m above
  ground on the sample triangle), with one more climb counted after arrival in their duration and speed. Routes
  now end on arrival, after a final glide at the safety margin (see the shared-out climbs above).
- Forecast not covering the flying day (date beyond the model range, e.g. AROME two days ahead) was converted
  into a ceiling at the station altitude all afternoon. It is now refused (`ForecastUnavailable`), with a clear
  message on the command line and in the web interface.

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
