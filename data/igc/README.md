# Your IGC tracks

Put the IGC files used to train the thermal network here. Everything in this folder except this README is
ignored by git (and `*.igc` is ignored everywhere in the repository): IGC files are personal data, never commit
them, and only use tracks whose pilots agreed.

## Layout

Sub-folders are fine (by year, by site, by pilot...); files are found recursively, `.igc` in any case.

```
data/igc/
├── 2024/
│   ├── 2024-06-12-digne.igc
│   └── 2024-06-12-digne.weather.json   # written by `cheminee fetch-weather`
└── 2025/
    └── ...
```

## Steps

```bash
cheminee fetch-weather data/igc --pause 1.0     # archived weather of each flight -> <flight>.weather.json
DEM=data/dem/cop30/<area>/output_hh.asc      # your larger extract, or data/dem/sample/output_hh.asc
cheminee train $DEM data/igc --out models/local/thermal_net.pt --report models/local/report.png
cheminee serve $DEM --model models/local/thermal_net.pt     # or plan / export --model ...
```

- `fetch-weather` makes one Open-Meteo archive request per flight and skips flights that already have their
  `.weather.json` (rerun it after adding tracks). Météo-France archives start in September 2022; older flights
  fall back to the weather estimated from the track.
- Only the part of a track inside the DEM is used. The sample DEM covers about 18 x 18 km around Digne: for
  flights that go further, train on a larger COP30 extract of the area (`data/dem/cop30/`, see
  [../dem/README.md](../dem/README.md)).
- Write the network trained on your tracks to `models/local/` (ignored by git). The default `--out`,
  `models/thermal_net.pt`, is the versioned synthetic model: overwriting it would put a model learnt from your
  flights in the next commit (a model can reveal where and when you fly).
