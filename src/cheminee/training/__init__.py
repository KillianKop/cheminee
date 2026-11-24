"""Training pipeline: IGC tracks -> labelled samples -> ThermalNet checkpoint."""

from .dataset import DatasetConfig, ThermalDataset, build_dataset
from .train import TrainConfig, train

__all__ = ["DatasetConfig", "ThermalDataset", "TrainConfig", "build_dataset", "train"]
