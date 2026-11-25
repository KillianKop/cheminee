"""Three points of a flight where its weather is taken: the turnpoints of a triangle, or along a straight cross.

The three fixes forming the largest triangle (the 3-point distance of the competitions) are the turnpoints of a
triangle flight. For a straight line or an out-and-return that triangle is flat: the points are then the takeoff,
the farthest fix from it and the fix closest to the middle of the two.
"""

from __future__ import annotations

import numpy as np

from .parser import IgcTrack

MAX_FIXES = 300  # the search is cubic: fixes are thinned out first
FLAT_RATIO = 0.1  # area / area of the equilateral triangle of the same perimeter, below which a triangle is flat


def _local_xy(track: IgcTrack) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular metres around the first fix (accurate enough over a flight)."""
    lat0 = np.radians(float(track.lat[0]))
    x = np.radians(track.lon - track.lon[0]) * 6371000.0 * np.cos(lat0)
    y = np.radians(track.lat - track.lat[0]) * 6371000.0
    return x, y


def flight_sites(track: IgcTrack) -> list[tuple[int, str]]:
    """Three (fix index, role) of the flight, in flight order: roles "TP1", "TP2", "TP3" for a triangle, else
    "start", "middle", "far"."""
    x, y = _local_xy(track)
    keep = np.unique(np.linspace(0, len(track) - 1, min(len(track), MAX_FIXES)).round().astype(int))
    px, py = x[keep], y[keep]
    d = np.hypot(px[:, None] - px[None], py[:, None] - py[None])
    best, best_ijk = -1.0, (0, 0, 0)
    n = len(keep)
    later = np.triu(np.ones((n, n), dtype=bool), 1)  # j < k
    for i in range(n - 2):
        perim = d[i, :, None] + d[:, :] + d[i, None, :]  # (j, k): d(i, j) + d(j, k) + d(i, k)
        perim = np.where(later & (np.arange(n)[:, None] > i), perim, -1.0)
        j, k = np.unravel_index(int(np.argmax(perim)), perim.shape)
        if perim[j, k] > best:
            best, best_ijk = float(perim[j, k]), (i, int(j), int(k))
    i, j, k = best_ijk
    a, b, c = d[i, j], d[j, k], d[i, k]
    s = 0.5 * (a + b + c)
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
    equilateral = np.sqrt(3.0) / 4.0 * (2.0 * s / 3.0) ** 2
    if best > 0 and area >= FLAT_RATIO * equilateral:
        return [(int(keep[i]), "TP1"), (int(keep[j]), "TP2"), (int(keep[k]), "TP3")]
    far = int(np.argmax(np.hypot(x - x[0], y - y[0])))
    mx, my = 0.5 * (x[0] + x[far]), 0.5 * (y[0] + y[far])
    middle = int(np.argmin(np.hypot(x[: far + 1] - mx, y[: far + 1] - my)))
    return [(0, "start"), (middle, "middle"), (far, "far")]
