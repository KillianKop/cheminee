# Models

`thermal_net.pt`: ThermalNet (21 input features) trained on 300 **synthetic** flights (`cheminee synth-igc`,
seed 0) over the sample DEM. It validates the training pipeline; replace it with a model trained on real tracks
before any practical use. Reproduce it with:

```bash
cheminee synth-igc data/output_hh.asc --out data/igc_synthetic -n 300
cheminee train data/output_hh.asc data/igc_synthetic --out models/thermal_net.pt
```
