"""XC scoring: free distance over 3 turnpoints."""

from __future__ import annotations

import numpy as np


def three_tp_distance(pts_idx: list[int], dist: np.ndarray) -> tuple[float, list[int]]:
    """Best start -> TP1 -> TP2 -> TP3 -> end chain along a track.

    Parameters
    ----------
    pts_idx : node indices of the successive track points.
    dist    : (n_nodes, n_nodes) distance matrix (m).

    Returns
    -------
    (distance in m, positions in the track of the 5 scoring points).
    """
    n = len(pts_idx)
    d = dist[np.ix_(pts_idx, pts_idx)]
    upper = np.tri(n, n, 0, dtype=bool).T  # i <= j: points are taken in track order
    best, back = np.zeros(n), []
    for _ in range(4):
        cand = np.where(upper, best[:, None] + d, -np.inf)
        back.append(cand.argmax(axis=0))
        best = cand.max(axis=0)
    chain = [int(best.argmax())]
    for bp in reversed(back):
        chain.append(int(bp[chain[-1]]))
    return float(best.max()), chain[::-1]
