"""Digital elevation model on a regular lon/lat grid, with a local metric frame."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .sun import sun_vector

EARTH_M_PER_DEG = 111_320.0


class Terrain:
    """DEM with a local metric frame: x = east, y = north (m), origin = SW corner.

    The elevation array `z` has row 0 at the NORTH edge (raster convention).
    """

    def __init__(self, z: np.ndarray, lon0: float, lat0: float, cellsize_deg: float):
        self.z = z.astype(np.float32)
        self.ny, self.nx = z.shape
        self.lon0, self.lat0, self.cell_deg = lon0, lat0, cellsize_deg
        self.lat_mid = lat0 + 0.5 * self.ny * cellsize_deg
        self.lon_mid = lon0 + 0.5 * self.nx * cellsize_deg
        self.dy = cellsize_deg * EARTH_M_PER_DEG
        self.dx = self.dy * np.cos(np.radians(self.lat_mid))
        self.width, self.height = self.nx * self.dx, self.ny * self.dy

        # Surface normal (east, north, up). Row index grows southward -> flip dz/dy.
        dz_drow, dz_dcol = np.gradient(self.z)
        dzdx, dzdy = dz_dcol / self.dx, -dz_drow / self.dy
        norm = np.sqrt(dzdx**2 + dzdy**2 + 1.0)
        self.normal = np.stack([-dzdx / norm, -dzdy / norm, 1.0 / norm]).astype(np.float32)
        self.slope = np.arccos(self.normal[2])  # rad
        self._tpi: dict[float, np.ndarray] = {}

    @classmethod
    def from_asc(cls, path: str) -> Terrain:
        """Load an ESRI ASCII grid (.asc, WGS84 lon/lat). Header length is auto-detected."""
        header, n_header = {}, 0
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0][0].isalpha():
                    header[parts[0].lower()] = float(parts[1])
                    n_header += 1
                else:
                    break
        z = np.loadtxt(path, skiprows=n_header, dtype=np.float32)
        nodata = header.get("nodata_value")
        if nodata is not None:
            z[z == nodata] = np.nan
        if np.isnan(z).any():  # fill holes with the lowest altitude
            z = np.where(np.isnan(z), np.nanmin(z), z)
        cell = header["cellsize"]
        lon0 = header.get("xllcorner", header.get("xllcenter", 0.0) - 0.5 * cell)
        lat0 = header.get("yllcorner", header.get("yllcenter", 0.0) - 0.5 * cell)
        print(f"[terrain] {path}: {z.shape[1]}x{z.shape[0]} cells, z in [{z.min():.0f}, {z.max():.0f}] m")
        return cls(z, lon0, lat0, cell)

    # -- coordinate helpers -------------------------------------------------- #
    def xy_to_rowcol(self, x, y):
        return (self.ny - 0.5) - np.asarray(y) / self.dy, np.asarray(x) / self.dx - 0.5

    def rowcol_to_xy(self, row, col):
        return (np.asarray(col) + 0.5) * self.dx, (self.ny - 0.5 - np.asarray(row)) * self.dy

    def xy_to_lonlat(self, x, y):
        return (
            self.lon0 + np.asarray(x) / self.dx * self.cell_deg,
            self.lat0 + np.asarray(y) / self.dy * self.cell_deg,
        )

    def lonlat_to_xy(self, lon, lat):
        return (
            (np.asarray(lon) - self.lon0) / self.cell_deg * self.dx,
            (np.asarray(lat) - self.lat0) / self.cell_deg * self.dy,
        )

    def contains(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        return (x >= 0) & (x <= self.width) & (y >= 0) & (y <= self.height)

    def sample(self, grid: np.ndarray, x, y) -> np.ndarray:
        """Bilinear sampling of a (ny, nx) grid at metric coordinates."""
        r, c = self.xy_to_rowcol(x, y)
        out = ndimage.map_coordinates(grid, [np.atleast_1d(r), np.atleast_1d(c)], order=1, mode="nearest")
        return out.reshape(np.shape(r))

    # -- derived fields -------------------------------------------------------- #
    def tpi(self, radius_m: float = 600.0) -> np.ndarray:
        """Topographic position index: altitude minus the mean altitude around (m).

        Positive on ridges and upper slopes, negative in valleys.
        """
        if radius_m not in self._tpi:
            r = max(1, int(radius_m / self.dy))
            self._tpi[radius_m] = self.z - ndimage.uniform_filter(self.z, 2 * r + 1, mode="nearest")
        return self._tpi[radius_m]

    def uphill(self, smooth_m: float = 500.0) -> tuple[np.ndarray, np.ndarray]:
        """Unit-ish vector field (east, north) pointing up the large-scale slope.

        Its norm goes from 0 on flat ground to 1 for slopes steeper than 10 %: used as
        the direction and relative strength of the up-slope / valley breeze.
        """
        if "uphill" not in self._tpi:
            zs = ndimage.gaussian_filter(self.z, smooth_m / self.dy)
            dz_drow, dz_dcol = np.gradient(zs)
            gx, gy = dz_dcol / self.dx, -dz_drow / self.dy
            norm = np.hypot(gx, gy)
            scale = np.clip(norm / 0.10, 0, 1) / np.maximum(norm, 1e-9)
            self._tpi["uphill"] = ((gx * scale).astype(np.float32), (gy * scale).astype(np.float32))
        return self._tpi["uphill"]

    def shadow_mask(self, az_deg: float, el_deg: float, max_dist_m: float = 4000.0) -> np.ndarray:
        """Boolean (ny, nx) mask of the cells lit by the sun (False = in a cast shadow)."""
        if el_deg <= 0:
            return np.zeros(self.z.shape, dtype=bool)
        az, tan_el = np.radians(az_deg), np.tan(np.radians(el_deg))
        step = 2.0 * self.dy
        lit = np.ones(self.z.shape, dtype=bool)
        for k in range(1, int(max_dist_m / step) + 1):
            d = k * step
            dc, dr = int(round(d * np.sin(az) / self.dx)), int(round(-d * np.cos(az) / self.dy))
            if abs(dr) >= self.ny or abs(dc) >= self.nx:
                break
            # obstacle = terrain at (row + dr, col + dc), compared with the sun ray height
            dst = (slice(max(-dr, 0), self.ny - max(dr, 0)), slice(max(-dc, 0), self.nx - max(dc, 0)))
            src = (slice(max(dr, 0), self.ny - max(-dr, 0)), slice(max(dc, 0), self.nx - max(-dc, 0)))
            lit[dst] &= self.z[src] - self.z[dst] <= d * tan_el
        return lit

    def hillshade(self, az_deg: float = 315.0, el_deg: float = 45.0) -> np.ndarray:
        return np.clip(np.tensordot(sun_vector(az_deg, el_deg), self.normal, 1), 0, 1)
