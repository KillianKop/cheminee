# Data

- `dem/`: elevation models, see [dem/README.md](dem/README.md). `dem/sample/` holds the versioned sample
  (Digne-les-Bains, about 18 x 18 km), `dem/cop30/` your own larger Copernicus extracts (not versioned).
- `igc_synthetic/`: created by `cheminee synth-igc` (not versioned). Holds one `.weather.json` per flight and a
  `manifest.csv` with the true conditions (day, wind) of each simulated flight, handy to check wind estimation.
- `igc/`: the place for your real IGC tracks (sub-folders allowed), see [igc/README.md](igc/README.md). They are
  personal data: only use tracks whose pilots agreed, and never commit them. Everything in `data/` except the
  READMEs and the sample DEM is ignored by git, and `*.igc` is ignored everywhere.
