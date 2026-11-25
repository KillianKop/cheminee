# Flight package (format v1)

The flight package is the **contract** between cheminee (which computes) and the vautour app (which guides the
pilot in flight). It is a single JSON document, produced by:

```bash
cheminee export data/output_hh.asc --out flight.json
```

The formal definition is the JSON schema [`schema/flight-package.v1.schema.json`](../schema/flight-package.v1.schema.json).
cheminee's tests validate every export against it, and vautour's tests validate the app's sample package
against a copy of the same file.

## Conventions

| Quantity | Convention |
|---|---|
| Position | Local frame of the DEM: x east, y north, in metres, origin at the south-west corner |
| Altitude | Metres above sea level |
| Time | Decimal local hours (14.5 = 14:30); `meta.utc_offset` gives the offset to UTC |
| Wind | `[speed in km/h, direction it blows FROM in degrees]` |
| Images | Base64 data URIs, north up, covering exactly the `extent` |
| Indices | Nodes are referred to by their index in `nodes` |
| Labels | English. Apps translate `kind` and `aspect` themselves; `label` is a fallback |

## Content

| Key | Content |
|---|---|
| `format`, `version` | `"cheminee.flight-package"` and `1` |
| `meta` | Generator and version, creation date, day of year, UTC offset, lift model, **attribution notices to display** |
| `extent` | Map size in metres, west/east longitudes and south/north latitudes |
| `terrain` | Map image (hypsometric tint, hillshade, 100 m contours), about 12 m per pixel |
| `dem` | Every other DEM sample: `nx × ny` little-endian 16-bit integers in base64, first row at the **north** |
| `lift` | Hourly lift field, greyscale PNG (0 = none, 255 = index 1) |
| `polar` | Glider speeds: trim, minimum, full bar (km/h) |
| `takeoff` | Index of the takeoff in `nodes` (always 0 today) |
| `nodes` | Thermals: position, ground altitude, slope aspect (`N` … `NW`, or `flat`), expected climb rate at every time of `climb_times` |
| `climb_times` | Times (every quarter of an hour) at which `nodes[].climb` is given |
| `routes` | Proposed routes, best first (see below) |
| `weather` | Every half hour: usable ceiling, cloud base, low-level wind and wind aloft |
| `graph` | Optional: feasible glides, to re-plan in flight (see below) |

## Routes (`routes[]`)

| Key | Content |
|---|---|
| `kind`, `label` | `fai`, `triangle` (flat), `free` (free distance) or `goal`, and an English label |
| `distance_km`, `points` | Counted distance (triangle perimeter or free distance via 3 points) and points (distance × CFD coefficient) |
| `t_start`, `t_end` | Planned takeoff and landing times |
| `nodes` | Nodes visited, in order |
| `scoring` | Counted polygon (closed for a triangle) |
| `turnpoints` | Node index (as a string) → `B1`, `B2`, `B3` for a triangle, `P1` to `P3` for free distance |
| `segments` | Continuous sequence of climbs (`climb`) and glides (`glide`) |

A `climb` segment gives the node, start and end times and altitudes (`t0`, `t1`, `a0`, `a1`) and the mean climb
rate `rate` (m/s). A `glide` segment also gives the departure node `src` and the arrival node `dst`, the
horizontal length `dist` (m), the mean ground speed `gs_kmh`, the sink rate `sink_ms`, the **airspeed to fly**
`airspeed_kmh` (chosen by the planner) and the **arrival margin** `margin_m` above the ground at `dst`.
The planner counts the rising air met along a glide (lift zones, where the pilot slows down to minimum sink): a
glide can therefore lose little height or even gain some. `sink_ms` is then small or **negative** and `a1 > a0`;
readers must not assume that a glide always goes down. The routing graph keeps heights lost `>= 0` (a glide from
the ceiling cannot climb above it; from the takeoff, a gain is reported as 0, the cautious value).

Segments are contiguous: the end of one is the start of the next, in time and in altitude. Glides shorter than
300 m between two neighbouring thermals are merged into a single climb.

## Routing graph (`graph`)

The graph lets the app re-plan a route when the pilot leaves the plan, without network. For each one-hour slot,
it gives the feasible glides **starting at the ceiling** of the slot.

- `speeds_kmh`: the three airspeeds considered (trim, intermediate, full bar).
- `slots[].edges`: rows `[i, j, glide_s, loss_m, glide_s, loss_m, glide_s, loss_m]`, one pair per speed of
  `speeds_kmh`; `-1` when that speed does not clear the terrain with the margin.
- `slots[].takeoff_edges`: rows `[j, glide_s, loss_m]` from the takeoff altitude, without climbing first.
- `circling_sink_ms`, `arrival_margin_m`, `clearance_m`: assumptions of the computation.

To estimate the duration of a leg i → j started at the ceiling at time t, add to the glide time the time needed
to climb back the height lost, in the thermal of j: `glide_s + loss_m / rate(j, t)`, where `rate(j, t)` is read
from `nodes[j].climb`. This is exactly what cheminee's planner computes.

## Changing the format

- Adding an **optional** key: compatible, `version` stays at 1. Update the schema, this document and the
  CHANGELOG.
- Renaming or removing a key, or changing its meaning: breaking change. Create
  `flight-package.v2.schema.json`, bump `version` to 2 (`FORMAT_VERSION` in `export/flight_package.py`), and
  update vautour in the same series of pull requests.
- In both cases, copy the schema into vautour (`schema/`) and regenerate its sample package:

```bash
# from cheminee, with vautour cloned next to it
cp schema/flight-package.v1.schema.json ../vautour/schema/
cheminee export data/output_hh.asc --out ../vautour/public/flight.json
```

With the default options this command reproduces vautour's sample byte for byte (apart from `meta.created`).

## Attribution notices

`meta.attribution` holds the notices that any app displaying the package **must** show to its users: the
Copernicus DEM notice by default (or the one given with `--dem-attribution`), and the Open-Meteo notice when a
forecast was used (`--forecast` or `--forecast-json`).
