"""Neural lift model: a small MLP predicting the lift index from terrain, sun and weather."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from ..config import Params
from ..domain import Terrain
from . import features as ft


class ThermalNet(nn.Module):
    """MLP: standardised features -> lift index in [0, 1] (sigmoid output)."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 64, 32)):
        super().__init__()
        layers, n_in = [], n_features
        for n_out in hidden:
            layers += [nn.Linear(n_in, n_out), nn.SiLU()]
            n_in = n_out
        layers.append(nn.Linear(n_in, 1))
        self.body = nn.Sequential(*layers)
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (apply a sigmoid to get the lift index)."""
        return self.body((x - self.mean) / self.std).squeeze(-1)


def save_checkpoint(path: str, model: ThermalNet, hidden: tuple[int, ...], meta: dict) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": list(hidden),
            "feature_names": list(ft.FEATURE_NAMES),
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path: str) -> tuple[ThermalNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["feature_names"] != list(ft.FEATURE_NAMES):
        raise ValueError(f"Checkpoint {path} was trained with another feature set: {ckpt['feature_names']}")
    model = ThermalNet(len(ft.FEATURE_NAMES), tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["meta"]


class NeuralLift:
    """Lift model backed by a trained ThermalNet (same interface as HeuristicLift)."""

    name = "neural"

    def __init__(self, terrain: Terrain, params: Params, model: ThermalNet, smooth_m: float = 150.0):
        self.t, self.p, self.model = terrain, params, model
        self.static = ft.static_stack(terrain).reshape(len(ft.STATIC_NAMES), -1).T  # (n_cells, n_static)
        self.sigma = smooth_m / terrain.dy

    @classmethod
    def from_checkpoint(cls, terrain: Terrain, params: Params, path: str) -> NeuralLift:
        return cls(terrain, params, load_checkpoint(path)[0])

    @torch.no_grad()
    def field(self, hour_local: float) -> np.ndarray:
        p, t = self.p, self.t
        x = ft.assemble(self.static, p.day_of_year, p.to_utc(hour_local), t.lat_mid, t.lon_mid, p.weather)
        out = torch.cat([torch.sigmoid(self.model(chunk)) for chunk in torch.from_numpy(x).split(131072)])
        lift = out.numpy().reshape(t.ny, t.nx)
        return np.clip(ndimage.gaussian_filter(lift, self.sigma), 0, 1).astype(np.float32)
