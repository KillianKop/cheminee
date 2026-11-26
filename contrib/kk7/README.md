# cheminee-kk7: thermal.kk7.ch climatology for cheminee (optional add-on)

> **Status: prototype.** Like cheminee itself, nothing here is a validated flight aid.

[thermal.kk7.ch](https://thermal.kk7.ch) publishes thermal probability maps computed from flown tracks, per
season and time of day. This add-on downloads them over a cheminee DEM and blends them into the lift model, so
the planner prefers places where thermals are actually found.

It lives **outside** the `cheminee` package on purpose: cheminee, its flight package and its licences never
depend on kk7 data. Nothing from kk7 is stored in this repository.

## Licence of the data: read this first

kk7 data are **CC BY-NC-SA 4.0**. In short:

- **Non-commercial only**: no paid app, no app with ads, no commercial service, unless you have a separate
  agreement with kk7.
- **Attribution**: every output (figure, screenshot, file) must credit `thermal.kk7.ch` with the licence.
  The commands here print or draw that notice.
- **ShareAlike**: anything derived from the data (a lift field, routes computed with it) is shared under the
  same licence. This is why there is no `export` command here. Do not ship a vautour flight package computed
  with kk7 unless you accept those terms for it.

kk7 usage policy, followed by `fetch.py`: a `User-Agent` that identifies the client, `?src=cheminee` on tile
requests, only the tiles over the DEM, at most 4 requests/s (kk7 allows 10), a 30-day cache, no bulk
prefetching.

## Install and run

From the root of the cheminee checkout, with cheminee installed (`pip install -e ".[dev]"`):

```bash
pip install -e contrib/kk7 --no-deps
cheminee-kk7 compare data/dem/sample/output_hh.asc --hour 14  # figure -> data/kk7_compare.png
cheminee-kk7 plan data/dem/sample/output_hh.asc               # same options as `cheminee plan`
cheminee-kk7 gui data/dem/sample/output_hh.asc
cheminee-kk7 serve data/dem/sample/output_hh.asc              # web interface with "cheminee + kk7" as lift model
cheminee-kk7 fetch data/dem/sample/output_hh.asc              # cache every season over the DEM (offline use)
cheminee-kk7 evaluate $DEM data/igc --model models/local/thermal_net.pt   # cheminee evaluate + kk7 models
cheminee-kk7 plan data/dem/sample/output_hh.asc --kk7-offline # cached tiles only, no network
pytest contrib/kk7/tests                                      # no network
```

`--no-deps` keeps pip from looking up a `cheminee` package on PyPI: the add-on uses the cheminee you
installed from this checkout. `python -m cheminee_kk7 ...` works too.

The first run downloads 3 layers x 12 tiles for the sample DEM (about 15 s). They are cached in
`data/kk7_cache/` (git-ignored).

| Option | Default | Meaning |
| --- | --- | --- |
| `--kk7-mode` | `boost` | `boost`: kk7 raises lift at known thermals and never removes any. `blend`: linear mix, lift fades away from kk7 spots |
| `--kk7-blend` | `0.8` | weight of kk7 in [0, 1]. 0 gives the cheminee lift unchanged |
| `--kk7-season` | `auto` | `jan`, `apr`, `jul`, `oct` or `all`; `auto` takes the closest one to `--day` / `--date` |
| `--kk7-time` | `auto` | `04` (morning), `07` (midday), `10` (evening) or `all`; `auto` interpolates from the hours after sunrise |
| `--kk7-zoom` | `12` | tile zoom (12 is about 27 m per pixel at 44° N) |
| `--kk7-cache` | `data/kk7_cache` | tile cache |
| `--kk7-offline` | off | cached tiles only; missing tiles count as "no data" |

`--model` (the ThermalNet checkpoint) still works: kk7 is blended into whichever base model cheminee builds.

## Evaluation against flown tracks

`cheminee-kk7 evaluate` runs `cheminee evaluate` (same options, same validation flights) with three extra
models: `kk7 climatology` (the kk7 probability alone, season of the flight, time of day interpolated),
`heuristic + kk7 <mode>` (the analytical model reshaped by kk7, `--kk7-blend`, `--kk7-mode`). Before scoring it
downloads, once and rate-limited, only the tiles over the evaluated windows (the season of each flight, three
times of day) and stops if more than `--kk7-max-tiles` (3000) are missing from the cache; the workers then read
the cache only.

kk7 is built from flown tracks, possibly the very flights you evaluate on: its scores are an upper bound of
what it would do on a new flight, unlike the network, which never saw the validation flights.

## How it works

1. **Tiles** (`tiles.py`, pure). The kk7 `thermals_<season>_<time>` raster tiles are TMS Web Mercator PNGs.
   Each pixel colour is mapped back to a probability by finding the nearest point on the kk7 colour ramp.
   The ramp is a transparent-to-dark-blue fade-in followed by the 16 colours of the kk7 web client. The
   decoded mosaic is then resampled bilinearly onto the DEM cells.
2. **Scale.** kk7 does not document the probability of each colour. `RAMP_P_MAX = 0.5` (last colour = 50 %)
   makes the decoded tiles agree with the probabilities of the kk7 hotspot API on the Digne sample (for
   example, hotspot 0.356 gives tile 0.40 and 0.160 gives 0.15). Treat absolute values as approximate.
   The lift model only uses relative values.
3. **Lift** (`lift.py`). kk7 says *where* thermals usually are; the base model says *how strong* lift is today
   (sun, clouds, wind). The kk7 field is normalised and rescaled to the top (99.9th percentile) of the base
   field at the same hour, `K = amplitude(base) * kk7 / max(kk7)`. Then:
   - `boost`: `L = max(base, weight * K)`. kk7 is transparent wherever nobody flies, and missing data does not
     mean a missing thermal. This mode therefore never lowers the base lift.
   - `blend`: `L = (1 - weight) * base + weight * K`. It lowers lift everywhere kk7 has no data. On the sample
     DEM this removes most usable thermals and the triangle search finds nothing. Use it only where kk7
     coverage is dense.

   A day without lift stays a day without lift, and climb rates keep the scale of the base model.
4. **Network** (`fetch.py`) is the only module that does HTTP. It also reads the hotspot API
   (`/api/hotspots/geojson/<layer>/<S>,<W>,<N>,<E>`), which `compare` draws as rings to check the
   georeferencing.

## Known limits

- kk7 is a climatology: it knows nothing about today's weather. It only reshapes the base field.
- kk7 coverage follows where pilots fly. In areas nobody flies, the add-on has no effect in `boost` mode.
- The cheminee triangle search leaves the takeoff at the first possible time (`Planner.launch_hour`). A lift
  field that is weak in the late morning can make it find no triangle even when a later start would work.
  Free distance (`--objective free`) does not have this limit.
- The colour decoding and the 0.5 scale come from reading the kk7 web client and tiles in November 2026. If
  kk7 changes its rendering, `test_decode_real_tile_colours` will no longer describe reality: check
  `cheminee-kk7 compare` against the hotspot rings.
