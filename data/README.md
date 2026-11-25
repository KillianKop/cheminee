# Data

- `rasters_COP30_sample.tar.gz`: sample Copernicus DEM GLO-30 extract (Digne-les-Bains area, 811 x 577 cells of
  1 arc-second, about 18 x 18 km). Extract it with `tar -xzf rasters_COP30_sample.tar.gz` → `output_hh.asc`.
  Licence: see the Copernicus notice in the main [README](../README.md#licence-and-data); any product made from
  it must carry that notice.
- `igc_synthetic/`: created by `cheminee synth-igc` (not versioned). Holds one `.weather.json` per flight and a
  `manifest.csv` with the true conditions (day, wind) of each simulated flight, handy to check wind estimation.
- `igc/`: the place for your real IGC tracks (sub-folders allowed), see [igc/README.md](igc/README.md). They are
  personal data: only use tracks whose pilots agreed, and never commit them. Everything in `data/` except the
  READMEs and the sample DEM is ignored by git, and `*.igc` is ignored everywhere.
