"""Elevation tiles cut from the DEM for the web map (pure functions, no I/O).

The page draws the relief with MapLibre GL from the same DEM the planner uses, so routes, thermals and the
terrain on screen agree. Tiles are Web Mercator XYZ, 256 px, in the "Mapbox Terrain-RGB" encoding read by
MapLibre's `raster-dem` sources: height = -10000 + (R * 65536 + G * 256 + B) * 0.1 m.

Outside the DEM there is no data: the altitude slopes down from the DEM edge to the lowest DEM altitude over a
few kilometres (an apron, instead of a vertical wall), then stays flat. The page darkens everything outside the
DEM, so nobody mistakes the surroundings for real terrain.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage

from ..domain import Terrain

TILE_PX = 256
DEM_MAX_ZOOM = 13  # ~14 m per pixel at 44° N: finer than the 30 m DEM, MapLibre overzooms beyond
HEIGHT_STEP_M = 1.0  # heights sent to the map are rounded to this: invisible on a 30 m DEM, PNG 40 % smaller
APRON_M = 2500.0  # e-folding distance of the slope outside the DEM
APRON_EXTENT_M = 8 * APRON_M  # beyond this, the apron is flat (within 0.04 % of the edge height)


def tile_lonlat(z: int, x: int, y: int, size: int = TILE_PX) -> tuple[np.ndarray, np.ndarray]:
    """Longitudes of the pixel columns and latitudes of the pixel rows (centres) of an XYZ tile."""
    n = size * 2**z
    px = (x * size + np.arange(size) + 0.5) / n
    py = (y * size + np.arange(size) + 0.5) / n
    lon = px * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * py))))
    return lon, lat


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) of an XYZ tile."""
    n = 2**z
    lon0, lon1 = x / n * 360.0 - 180.0, (x + 1) / n * 360.0 - 180.0
    lat1 = float(np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))))
    lat0 = float(np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n)))))
    return lon0, lat0, lon1, lat1


def encode_terrain_rgb(height_m: np.ndarray) -> np.ndarray:
    """(h, w) heights in metres -> (h, w, 3) uint8 Terrain-RGB (0.1 m steps)."""
    v = np.clip(np.rint((np.asarray(height_m, dtype=np.float64) + 10000.0) * 10.0), 0, 2**24 - 1).astype(np.uint32)
    return np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], axis=-1).astype(np.uint8)


def decode_terrain_rgb(rgb: np.ndarray) -> np.ndarray:
    """Inverse of `encode_terrain_rgb` (used by the tests)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return -10000.0 + (rgb[..., 0] * 65536.0 + rgb[..., 1] * 256.0 + rgb[..., 2]) * 0.1


def has_relief(terrain: Terrain, z: int, x: int, y: int) -> bool:
    """False when the tile is entirely flat floor: farther than the apron from the DEM."""
    lon0, lat0, lon1, lat1 = tile_bounds(z, x, y)
    x0, y0 = terrain.lonlat_to_xy(lon0, lat0)
    x1, y1 = terrain.lonlat_to_xy(lon1, lat1)
    m = APRON_EXTENT_M
    return bool(x1 >= -m and x0 <= terrain.width + m and y1 >= -m and y0 <= terrain.height + m)


def dem_heights(terrain: Terrain, z: int, x: int, y: int, size: int = TILE_PX) -> np.ndarray:
    """(size, size) altitudes of a tile: DEM (bilinear) inside, a sloping apron then a flat floor outside."""
    lon, lat = tile_lonlat(z, x, y, size)
    floor = float(terrain.z.min())
    if not has_relief(terrain, z, x, y):
        return np.full((size, size), floor, dtype=np.float32)
    # The DEM is a regular lon/lat grid: x depends on the column only, y on the row only.
    xs, _ = terrain.lonlat_to_xy(lon, 0.0)
    _, ys = terrain.lonlat_to_xy(0.0, lat)
    xc, yc = np.clip(xs, 0.0, terrain.width), np.clip(ys, 0.0, terrain.height)
    r, _ = terrain.xy_to_rowcol(0.0, yc)
    _, c = terrain.xy_to_rowcol(xc, 0.0)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    edge = ndimage.map_coordinates(terrain.z, [rr, cc], order=1, mode="nearest")
    dist = np.hypot(np.abs(ys - yc)[:, None], np.abs(xs - xc)[None, :])  # metres outside the DEM (0 inside)
    return (floor + (edge - floor) * np.exp(-dist / APRON_M)).astype(np.float32)


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def dem_tile_png(terrain: Terrain, z: int, x: int, y: int) -> bytes:
    """Terrain-RGB PNG of one XYZ tile of the DEM (heights rounded to HEIGHT_STEP_M)."""
    h = dem_heights(terrain, z, x, y)
    return png_bytes(encode_terrain_rgb(np.round(h / HEIGHT_STEP_M) * HEIGHT_STEP_M))
