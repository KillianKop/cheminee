"""Where to take the weather for a planning request: the takeoff and two points of a triangle around it."""

from __future__ import annotations

import numpy as np

from ..domain import Terrain

MAX_SIDE_M = 20000.0  # side of the triangle of weather points (a typical cross triangle has 25-35 km sides)
EDGE_MARGIN_M = 500.0


def weather_sites(
    terrain: Terrain, anchor_xy: tuple[float, float], manual_xy: list | None = None, anchor_role: str = "takeoff"
) -> list[tuple]:
    """Three (x, y, role) sites: the anchor (takeoff, or the DEM centre when the takeoff is automatic), then two
    points forming an equilateral triangle with it, pointing towards the DEM centre and inside the DEM.
    `manual_xy` replaces the two automatic points (positions chosen by the user)."""
    x0, y0 = (float(v) for v in anchor_xy)
    sites = [(x0, y0, anchor_role)]
    cx, cy = 0.5 * terrain.width, 0.5 * terrain.height
    heading = np.arctan2(cx - x0, cy - y0) if np.hypot(cx - x0, cy - y0) > 1000.0 else 0.0  # from north
    side = min(MAX_SIDE_M, 0.6 * min(terrain.width, terrain.height))
    lo = np.array([EDGE_MARGIN_M, EDGE_MARGIN_M])
    hi = np.array([terrain.width - EDGE_MARGIN_M, terrain.height - EDGE_MARGIN_M])
    for _ in range(40):  # shrink until both points are on the DEM
        pts = [np.array([x0 + side * np.sin(heading + a), y0 + side * np.cos(heading + a)]) for a in (-0.5236, 0.5236)]
        if all((p >= lo).all() and (p <= hi).all() for p in pts):
            break
        side *= 0.9
    else:
        pts = [np.clip(p, lo, hi) for p in pts]
    for k, xy in enumerate((manual_xy or [])[:2]):  # points placed by hand replace the automatic ones
        pts[k] = np.asarray(xy, dtype=float)
    return sites + [(float(p[0]), float(p[1]), f"point {k + 2}") for k, p in enumerate(pts)]
