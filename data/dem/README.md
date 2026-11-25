# Elevation models (DEM)

```
data/dem/
├── sample/
│   ├── rasters_COP30_sample.tar.gz   # versioned sample
│   └── output_hh.asc, .prj           # extracted here (not versioned)
└── cop30/                            # your own larger extracts (not versioned)
    └── <area>/output_hh.asc, .prj
```

Only this README and the sample archive are versioned; everything else in `data/dem/` is ignored by git.

## sample/

Copernicus DEM GLO-30 extract around Digne-les-Bains: 811 x 577 cells of 1 arc-second, about 18 x 18 km.

```bash
tar -xzf data/dem/sample/rasters_COP30_sample.tar.gz -C data/dem/sample/   # -> data/dem/sample/output_hh.asc
```

## cop30/

Put a larger extract of the area here to plan longer cross-countries (the sample caps triangles at about 55 km)
and to keep more of your IGC tracks when training (only the part of a track inside the DEM is used). Give each
extract its own sub-folder: exports are all called `output_hh.asc`.

- Format: ESRI ASCII grid (`.asc`) in geographic WGS 84 coordinates, with its `.prj`, as the sample. With
  OpenTopography, choose *Copernicus GLO-30* and the *ESRI ASCII* output.
- Size: every command works on the whole grid, so time and memory grow with the number of cells. Cut the area
  you fly (a few tens of km) rather than a whole range.
- Use it as any DEM: `cheminee serve data/dem/cop30/<area>/output_hh.asc`, or
  `cheminee train data/dem/cop30/<area>/output_hh.asc data/igc --out models/local/thermal_net.pt`.
- Licence: an extract of the Copernicus DEM carries the same notice as the sample (main
  [README](../../README.md#licence-and-data)); `cheminee export` writes it into every flight package.
