"""Trajectory optimiser on top of the planner graph.

- `earliest`  : time-dependent Dijkstra (earliest arrival at every node).
- `triangles` : exact scan of turnpoint triples by decreasing score; a triple is kept
                if takeoff -> TP1 -> TP2 -> TP3 -> closing point is flyable in time.
- `to_goal`   : fastest route to a given point.
- `optimise`  : best routes for an objective ("score", "free", "triangle", "fai").
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from ..domain import Route
from .planner import Planner


class RouteOptimizer:
    def __init__(self, planner: Planner):
        self.pl, self.p = planner, planner.p
        self._cache: dict[tuple, tuple] = {}

    def _log(self, msg: str) -> None:
        self.pl._log(msg)

    # -- fastest paths ----------------------------------------------------------- #
    def earliest(self, src: int, t_dep: float, first: bool = False):
        """Earliest arrival (local hours) at every node when leaving `src` at `t_dep`.

        Returns (arrival times (n,), predecessor (n,)). Results are cached per
        (source, departure time bin).
        """
        pl = self.pl
        key = (src, pl.bin_floor(t_dep), first)
        if key not in self._cache:
            arr = np.full(pl.n, np.inf)
            pred = np.full(pl.n, -1, dtype=int)
            arr[src] = t_dep
            heap = [(t_dep, src)]
            while heap:
                t, u = heapq.heappop(heap)
                if t > arr[u]:
                    continue
                b = pl.bin_floor(t)
                if b >= pl.n_bins:
                    continue
                js, leg_h = pl.edges(b, u, first and u == src and t == t_dep)[:2]
                for j, dt in zip(js.tolist(), leg_h.tolist()):
                    ta = t + dt
                    if ta < arr[j] and ta <= pl.t_end:
                        arr[j], pred[j] = ta, u
                        heapq.heappush(heap, (ta, j))
            self._cache[key] = (arr - t_dep, pred)
        offset, pred = self._cache[key]
        return t_dep + offset, pred

    @staticmethod
    def _path(pred: np.ndarray, arr: np.ndarray, dst: int):
        """Node list and times from the source of `pred` to `dst`."""
        nodes = [dst]
        while pred[nodes[-1]] >= 0:
            nodes.append(int(pred[nodes[-1]]))
        nodes.reverse()
        return nodes, [float(arr[n]) for n in nodes]

    # -- goal flight ---------------------------------------------------------------- #
    def to_goal(self, goal_xy: tuple[float, float]) -> list[Route]:
        """Fastest route to the thermal node nearest to `goal_xy` (or the closest reachable one)."""
        pl = self.pl
        t0 = pl.launch_hour()
        if t0 is None:
            return []
        arr, pred = self.earliest(0, t0, first=True)
        d_goal = np.hypot(pl.x - goal_xy[0], pl.y - goal_xy[1])
        d_goal[~np.isfinite(arr)] = np.inf
        d_goal[0] = np.inf
        if not np.isfinite(d_goal).any():
            return []
        target = int(d_goal.argmin())
        nodes, times = self._path(pred, arr, target)
        self._log(f"[goal] node {target} at {d_goal[target] / 1000:.1f} km from the goal, reached at {times[-1]:.2f} h")
        return [pl.make_route(nodes, times, "goal", float(pl.dist[0, target]), [0, len(nodes) - 1])]

    # -- triangles ------------------------------------------------------------------ #
    def triangles(self, fai_only: bool = False) -> list[Route]:
        """Best closed triangles (flat or FAI), by decreasing points."""
        pl, p = self.pl, self.p
        t0 = pl.launch_hour()
        if t0 is None:
            return []
        a0, p0 = self.earliest(0, t0, first=True)
        cand = np.flatnonzero(np.isfinite(a0))
        cand = cand[cand != 0]
        if len(cand) < 3:
            return []
        d = pl.dist
        i, j, k = np.meshgrid(cand, cand, cand, indexing="ij")
        i, j, k = i.ravel(), j.ravel(), k.ravel()
        sides = np.stack([d[i, j], d[j, k], d[k, i]])
        per = sides.sum(axis=0)
        is_fai = sides.min(axis=0) >= p.fai_min_leg * per
        points = per * np.where(is_fai, p.coef_fai, p.coef_triangle)
        keep = (i != j) & (j != k) & (i != k) & (per > 3000.0)
        # necessary condition: the three legs alone cannot be flown faster than the fastest
        # ground pace observed from the takeoff (with a 50 % margin)
        reach = cand[np.isfinite(a0[cand]) & (a0[cand] > t0)]
        pace = 1.5 * (d[0, reach] / (a0[reach] - t0)).max()  # m / h
        keep &= a0[i] + (d[i, j] + d[j, k]) / pace <= pl.t_end
        if fai_only:
            keep &= is_fai
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(-points[idx])][: p.max_triangle_evals]

        routes, kept_tps, kept_sets, n_eval = [], [], [], 0
        for q in idx:
            n_eval += 1
            tps = {int(i[q]), int(j[q]), int(k[q])}
            if any(len(tps & o) >= 2 for o in kept_tps):
                continue
            flown = self._fly_triangle(int(i[q]), int(j[q]), int(k[q]), per[q], a0, p0)
            if flown is None:
                continue
            # Same triangle, other start vertex / direction: keep the one that lands first.
            for perm in itertools.permutations(sorted(tps)):
                other = self._fly_triangle(*perm, per[q], a0, p0)
                if other is not None and other[1][-1] < flown[1][-1]:
                    flown = other
            nodes, times, pos = flown
            node_set = set(nodes)
            if any(len(node_set & o) / len(node_set | o) > p.diversity for o in kept_sets):
                continue
            routes.append(pl.make_route(nodes, times, "fai" if is_fai[q] else "triangle", float(per[q]), pos))
            kept_tps.append(tps)
            kept_sets.append(node_set)
            if len(routes) >= p.n_routes:
                break
        self._log(
            f"[search] triangles: {n_eval} of {len(idx)} candidate triples scanned, "
            f"{len(self._cache)} fastest-path trees, {len(routes)} routes kept"
        )
        return routes

    def _fly_triangle(self, tp1: int, tp2: int, tp3: int, perimeter: float, a0, p0):
        """Fastest takeoff -> TP1 -> TP2 -> TP3 -> closing point. None if not flyable in time."""
        d = self.pl.dist
        if not np.isfinite(a0[tp1]):
            return None
        a1, p1 = self.earliest(tp1, a0[tp1])
        if not np.isfinite(a1[tp2]):
            return None
        a2, p2 = self.earliest(tp2, a1[tp2])
        if not np.isfinite(a2[tp3]):
            return None
        # Closing: come back within 5 % of the perimeter of the takeoff or of TP1.
        radius = self.p.closing_fraction * perimeter
        close = np.flatnonzero((d[0] <= radius) | (d[tp1] <= radius))
        n3, t3 = [tp3], [float(a2[tp3])]
        if tp3 not in close:
            a3, p3 = self.earliest(tp3, a2[tp3])
            if not np.isfinite(a3[close]).any():
                return None
            n3, t3 = self._path(p3, a3, int(close[a3[close].argmin()]))
        n0, t0 = self._path(p0, a0, tp1)
        n1, t1 = self._path(p1, a1, tp2)
        n2, t2 = self._path(p2, a2, tp3)
        pos = [len(n0) - 1, len(n0) + len(n1) - 2, len(n0) + len(n1) + len(n2) - 3]
        return n0 + n1[1:] + n2[1:] + n3[1:], t0 + t1[1:] + t2[1:] + t3[1:], pos

    # -- entry point ---------------------------------------------------------------- #
    def optimise(self, objective: str | None = None) -> list[Route]:
        """Best diverse routes for the objective, sorted by decreasing points."""
        objective = objective or self.p.objective
        if objective == "free":
            return self.pl.search()
        if objective == "triangle":
            return self.triangles()
        if objective == "fai":
            return self.triangles(fai_only=True)
        if objective != "score":
            raise ValueError(f"Unknown objective: {objective}")
        pool = sorted(self.pl.search() + self.triangles(), key=lambda r: -r.points)
        routes, kept = [], []
        for r in pool:
            s = set(r.nodes)
            if any(len(s & o) / len(s | o) > self.p.diversity for o in kept):
                continue
            routes.append(r)
            kept.append(s)
            if len(routes) >= self.p.n_routes:
                break
        return routes
