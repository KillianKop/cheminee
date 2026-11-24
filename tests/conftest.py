import numpy as np
import pytest

from cheminee.config import Params
from cheminee.domain import Terrain
from cheminee.weather import Weather


@pytest.fixture(scope="session")
def terrain() -> Terrain:
    """Small analytic DEM (~7.5 x 7.5 km): two ridges and a valley, 600-1900 m."""
    n = 240
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    z = (
        700.0
        + 1100.0 * np.exp(-(((j - 70) / 28.0) ** 2)) * (0.75 + 0.25 * np.cos(i / 25.0))
        + 900.0 * np.exp(-(((j - 170) / 30.0) ** 2 + ((i - 120) / 90.0) ** 2))
        + 60.0 * np.sin(i / 9.0) * np.cos(j / 11.0)
    )
    return Terrain(z, lon0=6.2, lat0=44.0, cellsize_deg=1.0 / 3600.0)


@pytest.fixture
def params() -> Params:
    return Params(n_nodes=30, node_spacing_m=600.0, weather=Weather(breeze_kmh=0.0))
