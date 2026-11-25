"""Where to take the weather for a planning request: the takeoff and up to nine other points around it."""

from __future__ import annotations

import numpy as np

from ..domain import Terrain

MAX_POINTS = 10  # weather points of a planning request, the takeoff included (one forecast request each)
MAX_SIDE_M = 20000.0  # side of the triangle of three weather points (cross triangles have 25-35 km sides)
EDGE_MARGIN_M = 500.0


def weather_sites(
    terrain: Terrain,
    anchor_xy: tuple[float, float],
    manual_xy: list | None = None,
    anchor_role: str = "takeoff",
    n: int = 3,
) -> list[tuple]:
    """`n` (x, y, role) sites: the anchor (takeoff, or the DEM centre when the takeoff is automatic), then n - 1
    points that make a regular polygon with it, pointing towards the DEM centre and inside the DEM. With n = 3 it
    is an equilateral triangle of side `MAX_SIDE_M` at most. Its circumradius stays the one of that triangle
    whatever n, so more points cover the same area more densely.

    `manual_xy` lists the points placed by the user in the order of the sites after the anchor; None entries
    (or a shorter list) leave the automatic positions."""
    if not 1 <= n <= MAX_POINTS:
        raise ValueError(f"weather points: 1 to {MAX_POINTS}, got {n}")
    x0, y0 = (float(v) for v in anchor_xy)
    sites = [(x0, y0, anchor_role)]
    if n == 1:
        return sites
    cx, cy = 0.5 * terrain.width, 0.5 * terrain.height
    heading = np.arctan2(cx - x0, cy - y0) if np.hypot(cx - x0, cy - y0) > 1000.0 else 0.0  # from north
    radius = min(MAX_SIDE_M, 0.6 * min(terrain.width, terrain.height)) / np.sqrt(3.0)
    lo = np.array([EDGE_MARGIN_M, EDGE_MARGIN_M])
    hi = np.array([terrain.width - EDGE_MARGIN_M, terrain.height - EDGE_MARGIN_M])
    # polygon vertices seen from its centre; vertex 0 is the anchor
    angles = heading + np.pi + 2.0 * np.pi * np.arange(1, n) / n
    for _ in range(40):  # shrink until every point is on the DEM
        centre = np.array([x0 + radius * np.sin(heading), y0 + radius * np.cos(heading)])
        pts = [centre + radius * np.array([np.sin(a), np.cos(a)]) for a in angles]
        if all((p >= lo).all() and (p <= hi).all() for p in pts):
            break
        radius *= 0.9
    else:
        pts = [np.clip(p, lo, hi) for p in pts]
    for k, xy in enumerate((manual_xy or [])[: n - 1]):  # points placed by hand replace the automatic ones
        if xy is not None:
            pts[k] = np.asarray(xy, dtype=float)
    return sites + [(float(p[0]), float(p[1]), f"point {k + 2}") for k, p in enumerate(pts)]
