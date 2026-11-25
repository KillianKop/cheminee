"""Pure helpers for the thermal.kk7.ch raster tiles (no I/O).

Tile maths (Web Mercator, XYZ / TMS rows), colour -> probability decoding, reprojection onto a cheminee
`Terrain` grid, and choice of the kk7 time-filter layers for a day and an hour.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

TILE_PX = 256
MAX_ZOOM = 12  # deepest zoom of the kk7 "thermals" layers

# Colours of the "thermals" tiles, from "no thermal" to the highest probability: a transparent -> dark blue
# fade-in (observed on the tiles: rgb = (0, 0, alpha / 2)) followed by the 16-colour ramp of the kk7 web
# client (`js/main.js`).
RAMP_RGBA = np.array(
    [[0, 0, 0, 0], [0, 0, 127, 255]]
    + [
        [*rgb, 255]
        for rgb in (
            (0, 0, 196),
            (0, 0, 255),
            (0, 60, 255),
            (0, 124, 255),
            (0, 188, 255),
            (19, 252, 228),
            (70, 255, 177),
            (122, 255, 125),
            (173, 255, 73),
            (225, 255, 22),
            (255, 211, 0),
            (255, 152, 0),
            (255, 93, 0),
            (255, 33, 0),
            (200, 0, 0),
            (128, 0, 0),
        )
    ],
    dtype=float,
)
# Thermal probability of the last ramp colour. kk7 does not document the scale of the tiles: this value makes
# the tile colours agree with the probabilities of the kk7 hotspots on the Digne sample (see README).
RAMP_P_MAX = 0.5

# kk7 time filter. Season layer -> centre day of year (each layer spans about four months).
SEASONS = {"jan": 15, "apr": 105, "jul": 196, "oct": 288}
# Time-of-day layer -> centre in hours after sunrise ("04": 0-7 h, "07": 5-11 h, "10": 7 h or more).
TIMES_OF_DAY = {"04": 3.5, "07": 8.0, "10": 11.0}


def _ramp_samples(per_segment: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Dense RGBA samples of the ramp and their position s in [0, 1]."""
    n_seg = len(RAMP_RGBA) - 1
    k = np.arange(n_seg * per_segment + 1) / per_segment
    i = np.minimum(k.astype(int), n_seg - 1)
    f = (k - i)[:, None]
    return RAMP_RGBA[i] * (1.0 - f) + RAMP_RGBA[i + 1] * f, k / n_seg


_RAMP_POINTS, _RAMP_S = _ramp_samples()
_RAMP_TREE = cKDTree(_RAMP_POINTS)


def decode_tile(rgba: np.ndarray) -> np.ndarray:
    """(h, w, 4) uint8 RGBA tile -> (h, w) float32 thermal probability, 0 where the tile is transparent."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    keys = rgba.reshape(-1, 4).view(np.uint32).ravel()
    unique, inverse = np.unique(keys, return_inverse=True)
    colours = unique.view(np.uint8).reshape(-1, 4)
    _, idx = _RAMP_TREE.query(colours.astype(float))
    p = (_RAMP_S[idx] * RAMP_P_MAX).astype(np.float32)
    p[colours[:, 3] == 0] = 0.0
    return p[inverse.ravel()].reshape(rgba.shape[:2])


def encode_tile(p: np.ndarray) -> np.ndarray:
    """Inverse of `decode_tile` (probability -> RGBA), used to build synthetic tiles in tests."""
    n_seg = len(RAMP_RGBA) - 1
    s = np.clip(np.asarray(p, dtype=float) / RAMP_P_MAX, 0.0, 1.0) * n_seg
    i = np.minimum(s.astype(int), n_seg - 1)
    f = (s - i)[..., None]
    return np.rint(RAMP_RGBA[i] * (1.0 - f) + RAMP_RGBA[i + 1] * f).astype(np.uint8)


# -- tile maths ------------------------------------------------------------------ #
def lonlat_to_pixel(lon, lat, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    """Global Web Mercator pixel coordinates at `zoom` (x east, y SOUTH from the north edge, XYZ convention)."""
    n = TILE_PX * 2**zoom
    x = (np.asarray(lon, dtype=float) + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(np.asarray(lat, dtype=float)))) / np.pi) / 2.0 * n
    return x, y


def tms_row(y: int, zoom: int) -> int:
    """XYZ tile row -> TMS tile row (kk7 serves TMS: row 0 at the south)."""
    return 2**zoom - 1 - y


def terrain_bbox(terrain) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) of a cheminee Terrain."""
    lon0, lat0 = terrain.xy_to_lonlat(0.0, 0.0)
    lon1, lat1 = terrain.xy_to_lonlat(terrain.width, terrain.height)
    return float(lon0), float(lat0), float(lon1), float(lat1)


def tiles_covering(lon_min: float, lat_min: float, lon_max: float, lat_max: float, zoom: int) -> list[tuple[int, int]]:
    """XYZ (x, y) of the tiles covering a bounding box, with one pixel of margin for the interpolation."""
    x0, y0 = lonlat_to_pixel(lon_min, lat_max, zoom)
    x1, y1 = lonlat_to_pixel(lon_max, lat_min, zoom)
    cols = range(int((x0 - 1) // TILE_PX), int((x1 + 1) // TILE_PX) + 1)
    rows = range(int((y0 - 1) // TILE_PX), int((y1 + 1) // TILE_PX) + 1)
    return [(x, y) for y in rows for x in cols]


def to_terrain_grid(terrain, tiles: dict[tuple[int, int], np.ndarray | None], zoom: int) -> np.ndarray:
    """Bilinear resampling of decoded tiles {XYZ (x, y): probability} onto the terrain cells (ny, nx).

    Missing tiles (None: no data at kk7) count as probability 0.
    """
    xs, ys = [k[0] for k in tiles], [k[1] for k in tiles]
    tx0, ty0 = min(xs), min(ys)
    mosaic = np.zeros(((max(ys) - ty0 + 1) * TILE_PX, (max(xs) - tx0 + 1) * TILE_PX), dtype=np.float32)
    for (x, y), p in tiles.items():
        if p is not None:
            r, c = (y - ty0) * TILE_PX, (x - tx0) * TILE_PX
            mosaic[r : r + TILE_PX, c : c + TILE_PX] = p

    # Cell centres: longitude depends on the column only, latitude on the row only.
    rows, cols = np.arange(terrain.ny), np.arange(terrain.nx)
    _, y_m = terrain.rowcol_to_xy(rows, np.zeros_like(rows))
    x_m, _ = terrain.rowcol_to_xy(np.zeros_like(cols), cols)
    lon, _ = terrain.xy_to_lonlat(x_m, 0.0)
    _, lat = terrain.xy_to_lonlat(0.0, y_m)
    px, _ = lonlat_to_pixel(lon, 0.0, zoom)
    _, py = lonlat_to_pixel(0.0, lat, zoom)
    # pixel (i, j) covers [j, j + 1) x [i, i + 1): its centre is at +0.5
    rr, cc = np.meshgrid(py - ty0 * TILE_PX - 0.5, px - tx0 * TILE_PX - 0.5, indexing="ij")
    return ndimage.map_coordinates(mosaic, [rr, cc], order=1, mode="nearest").astype(np.float32)


# -- time filter ----------------------------------------------------------------- #
def nearest_season(day_of_year: int) -> str:
    """kk7 season layer whose centre is the closest to the day (circular year)."""

    def dist(day: int) -> float:
        d = abs(day_of_year - day) % 365
        return min(d, 365 - d)

    return min(SEASONS, key=lambda k: dist(SEASONS[k]))


def time_of_day_weights(hours_after_sunrise: float) -> dict[str, float]:
    """Linear interpolation weights between the time-of-day layers (clamped at both ends, sum = 1)."""
    keys = sorted(TIMES_OF_DAY, key=TIMES_OF_DAY.get)
    centres = np.array([TIMES_OF_DAY[k] for k in keys])
    h = float(np.clip(hours_after_sunrise, centres[0], centres[-1]))
    j = int(np.clip(np.searchsorted(centres, h) - 1, 0, len(keys) - 2))
    f = (h - centres[j]) / (centres[j + 1] - centres[j])
    weights = dict.fromkeys(keys, 0.0)
    weights[keys[j]], weights[keys[j + 1]] = 1.0 - f, f
    return weights
