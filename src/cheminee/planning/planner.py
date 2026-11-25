"""Flight planner: thermal nodes -> glide graph -> time-dependent route search.

The planner does not know how the lift index is produced: it receives any object
implementing `cheminee.lift.LiftModel` (heuristic or neural).

Physics of one leg i -> j (speed-to-fly, MacCready-like):
    time(v) = glide time at airspeed v + time to climb back, in the thermal of j,
              the height lost during that glide
The airspeed v is chosen among the speeds of the glider polar to minimise time(v),
subject to terrain clearance and to the arrival margin above j. Wind (at glide
altitude) and ceiling change from hour to hour ("slots").

The search climbs to the ceiling in every thermal. The climbs of a route are then
shared out like a pilot would (`Planner._partial_climbs`): climb only what is needed to
reach a stronger thermal, cross weak ones without circling, and end with a final glide
that arrives with the safety margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..config import Params
from ..domain import Route, Terrain, three_tp_distance
from ..lift import LiftModel

_EMPTY = np.array([], dtype=int)


@dataclass
class _Slot:
    """Hour-long period with a constant ceiling and glide wind."""

    ceil: float
    wind: np.ndarray  # (2,) m/s at glide altitude
    t_glide: np.ndarray  # (V, n, n) glide duration (s) per airspeed, inf if impossible
    loss: np.ndarray  # (V, n, n) height lost (m) per airspeed
    feas: np.ndarray  # (V, n, n) glide feasible from the ceiling
    nbrs: list  # per node: neighbour indices (sector rule)
    low0_nbrs: np.ndarray  # nodes reachable from the takeoff altitude, at trim speed
    low0_t: np.ndarray  # (n,) glide duration (s) of that direct glide
    low0_loss: np.ndarray  # (n,) height lost (m)


class Planner:
    def __init__(
        self, terrain: Terrain, params: Params, takeoff_xy: tuple[float, float], lift: LiftModel, verbose: bool = True
    ):
        self.t, self.p, self.lift, self.verbose = terrain, params, lift, verbose
        p, w = params, params.weather
        t_end = p.t_end
        if w.overdev_hour is not None:  # storms: everybody must be on the ground
            t_end = min(t_end, p.from_solar(w.overdev_hour, terrain.lon_mid))
        self.t_end = max(t_end, p.t_start + p.dt_min / 60.0)
        self.n_bins = int(np.ceil((self.t_end - p.t_start) * 60.0 / p.dt_min))
        self.hours = p.t_start + np.arange(self.n_bins + 1) * p.dt_min / 60.0
        self.n_slots = max(1, int(np.ceil(self.t_end - p.t_start)))
        self.slot_hours = p.t_start + np.arange(self.n_slots) + 0.5
        self.ceil = np.array([w.ceiling(self.solar(h)) for h in self.slot_hours])
        self._leg_cache: dict[int, tuple] = {}
        self._build_nodes(takeoff_xy)
        self._build_slots()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def solar(self, hour_local: float) -> float:
        return self.p.to_solar(hour_local, self.t.lon_mid)

    def bin_floor(self, hour: float) -> int:
        return int(np.clip((hour - self.p.t_start) * 60.0 / self.p.dt_min + 1e-9, 0, self.n_bins))

    def bin_of(self, hour: float) -> int:
        return int(np.clip(round((hour - self.p.t_start) * 60.0 / self.p.dt_min), 0, self.n_bins))

    def slot_of(self, b: int) -> int:
        return min(int(b * self.p.dt_min / 60.0), self.n_slots - 1)

    # -- thermal nodes -------------------------------------------------------- #
    def _build_nodes(self, takeoff_xy) -> None:
        t, p = self.t, self.p
        sample_hours = np.arange(p.t_start, self.t_end + 1e-6, 0.5)
        fields = np.stack([self.lift.field(h) for h in sample_hours])
        self.sample_hours, self.fields = sample_hours, fields
        potential = fields.max(axis=0)
        potential += 1e-6 * np.random.default_rng(0).random(potential.shape)  # break ties
        potential[t.z > self.ceil.max() - p.arrival_margin_m - 50.0] = 0.0

        # Local maxima of the daily potential, with a minimum spacing.
        size = 2 * int(p.node_spacing_m / t.dy / 2) + 1
        peaks = (potential == ndimage.maximum_filter(potential, size)) & (potential > 0.5 * p.lift_min)
        rows, cols = np.nonzero(peaks)
        order = np.argsort(-potential[rows, cols])[: p.n_nodes]
        nx_, ny_ = t.rowcol_to_xy(rows[order], cols[order])

        self.x = np.concatenate([[takeoff_xy[0]], nx_])
        self.y = np.concatenate([[takeoff_xy[1]], ny_])
        self.zg = t.sample(t.z, self.x, self.y)  # ground altitude of the nodes
        self.n = len(self.x)

        # Lift index of each node at each sample hour. For the takeoff we take the best
        # value within ~400 m (the pilot soars in front of the launch).
        r, c = t.xy_to_rowcol(self.x, self.y)
        node_lift = np.stack([ndimage.map_coordinates(f, [r, c], order=1, mode="nearest") for f in fields])
        k = int(400.0 / t.dy)
        r0, c0 = int(round(float(r[0]))), int(round(float(c[0])))
        window = (slice(max(r0 - k, 0), r0 + k + 1), slice(max(c0 - k, 0), c0 + k + 1))
        node_lift[:, 0] = [f[window].max() for f in fields]
        # (n_bins + 1, n_nodes): linear interpolation in time
        self.node_lift = np.stack([np.interp(self.hours, sample_hours, node_lift[:, i]) for i in range(self.n)], axis=1)
        # climb rate (m/s) of every node at every time bin
        self.climb = p.w_ref_ms * self.node_lift - p.polar.circling_sink_ms
        self._log(
            f"[nodes] {self.n - 1} thermal nodes ({self.lift.name} lift), takeoff at "
            f"{self.zg[0]:.0f} m, ceiling {self.ceil.min():.0f}-{self.ceil.max():.0f} m"
        )

    # -- glide graph ----------------------------------------------------------- #
    def _build_slots(self) -> None:
        t, p, w = self.t, self.p, self.p.weather
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        self.dist = dist = np.hypot(dx, dy)
        safe = np.where(dist > 0, dist, 1.0)
        ex, ey = dx / safe, dy / safe
        off_diag = ~np.eye(self.n, dtype=bool)

        # Altitude needed along every leg (terrain + clearance): (n, n, S - 1)
        s = np.linspace(0.0, 1.0, 64)[1:]
        px = self.x[:, None, None] + dx[:, :, None] * s
        py = self.y[:, None, None] + dy[:, :, None] * s
        along = np.maximum(dist[:, :, None] * s, 1e-6)
        need = t.sample(t.z, px.ravel(), py.ravel()).reshape(self.n, self.n, -1)
        need += np.minimum(p.clearance_m, 0.2 * along)  # clearance ramps up right after the start
        arrival_need = self.zg[None, :] + p.arrival_margin_m

        def max_slope(h_start):
            """Largest height loss per metre allowed on each leg when starting at h_start (n, 1, 1)."""
            slope = ((h_start - need) / along).min(axis=2)
            return np.minimum(slope, (h_start[:, :, 0] - arrival_need) / safe)

        def neighbours(mask_row, i):
            """Keep the nearest reachable nodes per azimuth sector (`Params.neighbours_per_sector`)."""
            js = np.nonzero(mask_row)[0]
            if len(js) == 0:
                return _EMPTY
            sector = ((np.arctan2(dx[i, js], dy[i, js]) % (2 * np.pi)) / (2 * np.pi) * p.n_sectors).astype(int)
            keep = []
            for sec in np.unique(sector):
                cand = js[sector == sec]
                keep.extend(cand[np.argsort(dist[i, cand])[: p.neighbours_per_sector]])
            return np.array(sorted(keep), dtype=int)

        va = p.polar.speeds[:, None, None]
        sink = p.polar.sink(p.polar.speeds)[:, None, None]
        vt = p.polar.trim_index
        low_start = np.full((self.n, 1, 1), -1e9)
        low_start[0] = self.zg[0] + 5.0  # only the takeoff may start without climbing
        self.slots: list[_Slot] = []
        for k in range(self.n_slots):
            ceil = float(self.ceil[k])
            wind = w.wind_vec(0.5 * (ceil + float(self.zg.mean())), self.solar(self.slot_hours[k]))
            w_par, w_perp = ex * wind[0] + ey * wind[1], ex * wind[1] - ey * wind[0]
            root2 = va**2 - w_perp**2
            vg = w_par + np.sqrt(np.clip(root2, 0, None))  # wind triangle, per airspeed
            valid = (root2 > 0) & (vg > 1.0) & off_diag
            vg = np.where(valid, vg, 1.0)
            t_glide = np.where(valid, dist / vg, np.inf)
            loss = np.where(valid, dist * sink / vg, np.inf)
            with np.errstate(invalid="ignore"):
                loss_per_m = loss / safe
            feas = valid & (loss_per_m <= max_slope(np.full((self.n, 1, 1), ceil))[None])
            low_ok = valid[vt, 0] & (loss_per_m[vt, 0] <= max_slope(low_start)[0])
            self.slots.append(
                _Slot(
                    ceil,
                    wind,
                    t_glide,
                    loss,
                    feas,
                    [neighbours(row, i) for i, row in enumerate(feas.any(axis=0))],
                    neighbours(low_ok, 0),
                    t_glide[vt, 0],
                    loss[vt, 0],
                )
            )
        n_glides = int(self.slots[self.n_slots // 2].feas.any(axis=0).sum())
        self._log(
            f"[graph] {self.n_slots} hourly slots, {n_glides} feasible glides at mid-day, "
            f"glide wind {np.hypot(*self.slots[self.n_slots // 2].wind) * 3.6:.0f} km/h"
        )

    def _legs(self, b: int):
        """Best airspeed per leg for departures during time bin b (cached)."""
        if b not in self._leg_cache:
            s = self.slots[self.slot_of(b)]
            climb = self.climb[b]
            ok = climb >= self.p.min_climb_ms
            total = np.where(
                s.feas & ok[None, None, :], s.t_glide + np.where(s.feas, s.loss, 0.0) / np.where(ok, climb, 1.0), np.inf
            )
            vi = total.argmin(axis=0)[None]
            self._leg_cache[b] = (
                np.take_along_axis(total, vi, 0)[0] / 3600.0,
                np.take_along_axis(s.t_glide, vi, 0)[0] / 3600.0,
                s.ceil - np.take_along_axis(s.loss, vi, 0)[0],
                ok,
                vi[0],  # airspeed index of each leg
            )
        return self._leg_cache[b]

    def edges(self, b: int, i: int, first: bool = False):
        """Legs leaving node i during time bin b.

        Returns (neighbours, leg duration h, glide duration h, arrival altitude, start altitude).
        `first` marks the very first leg of the flight: the pilot is still at takeoff height
        and either climbs in the house thermal or glides directly to a nearby one.
        """
        leg_h, glide_h, arr_alt, ok, _ = self._legs(b)
        s = self.slots[self.slot_of(b)]
        if i == 0 and first and not ok[0]:
            js = s.low0_nbrs[ok[s.low0_nbrs]]
            arrival = self.zg[0] - s.low0_loss[js]
            total = s.low0_t[js] + (s.ceil - arrival) / self.climb[b, js]
            return js, total / 3600.0, s.low0_t[js] / 3600.0, arrival, float(self.zg[0])
        if not ok[i]:
            return _EMPTY, _EMPTY, _EMPTY, _EMPTY, s.ceil
        js = s.nbrs[i]
        js = js[np.isfinite(leg_h[i, js])]
        extra = (s.ceil - self.zg[0]) / self.climb[b, 0] / 3600.0 if (i == 0 and first) else 0.0
        return js, leg_h[i, js] + extra, glide_h[i, js], arr_alt[i, js], s.ceil

    def lift_at(self, x, y, hour: float) -> np.ndarray:
        """Lift index at metric coordinates, from the nearest pre-computed half-hour field."""
        k = int(np.abs(self.sample_hours - hour).argmin())
        return self.t.sample(self.fields[k], x, y)

    def launch_hour(self) -> float | None:
        """First time the pilot can leave the takeoff (house thermal or direct glide)."""
        for b in range(self.n_bins):
            if len(self.edges(b, 0, first=True)[0]):
                return float(self.hours[b])
        return None

    # -- free distance search ---------------------------------------------------- #
    def search(self) -> list[Route]:
        """Time-dependent label-setting search maximising the 3-TP free distance.

        A label is (score, times, path, A) where A[k, m] is the best (k+1)-leg chain
        ending at the m-th point of the path (incremental 3-TP scoring).
        """
        p = self.p
        dt_h = p.dt_min / 60.0
        buckets = [dict() for _ in range(self.n_bins + 1)]
        buckets[0][0] = [(0.0, (p.t_start,), (0,), np.zeros((4, 1)))]
        all_labels = []

        for b in range(self.n_bins):
            # The pilot may wait on the takeoff for better conditions.
            if 0 in buckets[b] and not any(len(lab[2]) == 1 for lab in buckets[b + 1].get(0, [])):
                t_wait = float(self.hours[b + 1])
                buckets[b + 1].setdefault(0, []).append((0.0, (t_wait,), (0,), np.zeros((4, 1))))
            for i, labels in list(buckets[b].items()):
                for score, times, path, A in labels:
                    nbrs, leg_h = self.edges(b, i, first=len(path) == 1)[:2]
                    if len(nbrs) == 0:
                        continue  # stuck: the thermal died, the flight ends here
                    d = self.dist[np.ix_(np.array(path), nbrs)]  # (len(path), K)
                    prev = np.vstack([np.zeros((1, len(path))), A[:3]])  # chains of 0..3 legs
                    new = (prev[:, :, None] + d[None]).max(axis=1)  # (4, K)
                    new_score = np.maximum(score, new[3])
                    for k, j in enumerate(nbrs.tolist()):
                        t_arr = times[-1] + leg_h[k]
                        if t_arr > self.t_end or j in path[-2:] or path.count(j) >= p.max_visits:
                            continue
                        nb = max(b + 1, int((t_arr - p.t_start) / dt_h))
                        bucket = buckets[nb].setdefault(j, [])
                        if len(bucket) >= p.labels_per_state and new_score[k] <= bucket[-1][0]:
                            continue
                        bucket.append(
                            (
                                float(new_score[k]),
                                times + (float(t_arr),),
                                path + (j,),
                                np.hstack([A, new[:, k : k + 1]]),
                            )
                        )
                        bucket.sort(key=lambda lab: -lab[0])
                        del bucket[p.labels_per_state :]
            for labels in buckets[b].values():
                all_labels.extend(labels)
        for labels in buckets[self.n_bins].values():
            all_labels.extend(labels)

        # Keep the best routes, enforcing diversity (Jaccard similarity of node sets).
        all_labels.sort(key=lambda lab: -lab[0])
        routes, kept_sets = [], []
        for _, times, path, _ in all_labels:
            if len(path) < 2:
                continue
            s = set(path)
            if any(len(s & o) / len(s | o) > p.diversity for o in kept_sets):
                continue
            kept_sets.append(s)
            routes.append(self.make_route(list(path), list(times)))
            if len(routes) >= p.n_routes:
                break
        self._log(f"[search] free distance: {len(all_labels)} labels explored, {len(routes)} routes kept")
        return routes

    # -- route construction -------------------------------------------------------- #
    def make_route(
        self,
        path: list[int],
        times: list[float],
        kind: str = "free",
        distance_m: float | None = None,
        turnpoints: list[int] | None = None,
    ) -> Route:
        """Attach the score and the altitude profile (climb / glide sawtooth) to a path.

        `times` come from the search, which climbs to the ceiling in every thermal. With `Params.partial_climbs`
        the climbs are then shared out along the path like a pilot would (see `_partial_climbs`).
        """
        p = self.p
        if distance_m is None:
            distance_m, turnpoints = three_tp_distance(path, self.dist)
        coef = {"free": p.coef_free, "triangle": p.coef_triangle, "fai": p.coef_fai}.get(kind, 1.0)
        legs = [self._leg(k, i, j, times) for k, (i, j) in enumerate(zip(path[:-1], path[1:]))]
        prof, out_times = self._full_climbs(path, times, legs)
        if p.partial_climbs and len(path) >= 2 and all(leg is not None for leg in legs):
            better = self._partial_climbs(path, times, legs, prof)
            if better is not None and better[1][-1] <= out_times[-1] + 1e-9:
                prof, out_times = better
        return Route(
            path,
            out_times,
            distance_m / 1000.0,
            turnpoints,
            kind,
            coef * distance_m / 1000.0,
            np.column_stack([self.x[path], self.y[path], self.zg[path]]),
            np.array(prof),
        )

    def _leg(self, k: int, i: int, j: int, times: list[float]) -> dict | None:
        """The leg i -> j of the search, around its departure time: speed, height loss, glide time, start height."""
        first = k == 0
        for b in (self.bin_floor(times[k]), self.bin_floor(times[k]) + 1, self.bin_floor(times[k]) - 1):
            if not 0 <= b < self.n_bins:
                continue
            js, _, glide_h, arrival, start_alt = self.edges(b, i, first)
            hit = np.flatnonzero(js == j)
            if len(hit):
                s = self.slots[self.slot_of(b)]
                low = first and i == 0 and not self._legs(b)[3][0]  # direct glide from the takeoff height
                v = None if low else int(self._legs(b)[4][i, j])
                return dict(
                    b=b,
                    v=v,
                    glide_h=float(glide_h[hit[0]]),
                    arrival=float(arrival[hit[0]]),
                    start=float(start_alt),
                    loss=float(start_alt - arrival[hit[0]]),
                    ceil=s.ceil,
                )
        return None

    def _full_climbs(self, path: list[int], times: list[float], legs: list[dict | None]):
        """Profile of the search: climb to the ceiling in every thermal; the flight ends on arrival."""
        p = self.p
        prof = [(self.x[0], self.y[0], self.zg[0], times[0])]
        for k, (i, j) in enumerate(zip(path[:-1], path[1:])):
            leg = legs[k]
            if leg is None:  # should not happen; keep a plausible profile anyway
                glide_h = 0.5 * (times[k + 1] - times[k])
                arrival = float(self.zg[j] + p.arrival_margin_m)
                start_alt = float(self.ceil[self.slot_of(self.bin_floor(times[k]))])
            else:
                glide_h, arrival, start_alt = leg["glide_h"], leg["arrival"], leg["start"]
            if k == 0 and start_alt > self.zg[0] + 1.0:  # initial climb in the house thermal
                climb0 = self.climb[self.bin_floor(times[0]), 0]
                prof.append(
                    (self.x[0], self.y[0], start_alt, times[0] + (start_alt - self.zg[0]) / max(climb0, 0.1) / 3600.0)
                )
            elif k > 0:  # top of the previous climb, reached when leaving node i
                prof.append((self.x[i], self.y[i], start_alt, times[k]))
            t_arrive = min(prof[-1][3] + glide_h, times[k + 1])
            prof.append((self.x[j], self.y[j], arrival, t_arrive))
        return prof, list(times[:-1]) + [prof[-1][3]]

    def start_need(self, i: int, j: int, loss: float) -> float:
        """Lowest altitude to leave i and glide to j losing `loss` metres: arrival margin above j and terrain
        clearance all along (at fraction f of the leg the glider has lost f * loss)."""
        p = self.p
        dist = float(self.dist[i, j])
        f = np.linspace(0.0, 1.0, 64)[1:]
        along = np.maximum(dist * f, 1e-6)
        xs, ys = self.x[i] + (self.x[j] - self.x[i]) * f, self.y[i] + (self.y[j] - self.y[i]) * f
        need = self.t.sample(self.t.z, xs, ys) + np.minimum(p.clearance_m, 0.2 * along)
        return float(max(self.zg[j] + p.arrival_margin_m + loss, (need + loss * f).max()))

    def _partial_climbs(self, path: list[int], times: list[float], legs: list[dict], full_prof: list):
        """Share the climbs out along the path like a pilot: altitude is fuel, a metre costs 1 / climb rate.

        Greedy rule of the "gas station problem", optimal when climb rates are constant: in a thermal, if a
        stronger one (or the goal) can be reached without climbing more than the ceiling allows, climb just
        enough to reach it; otherwise climb to the ceiling. A thermal reached before it works is waited for
        (the pilot hangs on in weak lift). A thermal that is not worth circling is crossed in a straight line,
        which still gains a little height (`dolphin_core_m` of rising air at glide speed). The last climb
        therefore stops at the height the final glide needs. Returns (profile, times) or None if the plan cannot
        be kept (then the full-climb profile is used).
        """
        p = self.p
        m = len(path) - 1
        loss = [leg["loss"] for leg in legs]
        glide_s = [leg["glide_h"] * 3600.0 for leg in legs]
        need = [
            self.start_need(path[k], path[k + 1], loss[k]) if legs[k]["v"] is not None else legs[k]["start"]
            for k in range(m)
        ]
        # free height gained crossing node k straight (k >= 1), with the lift at the time of the search
        gain = [0.0] * (m + 1)
        for k in range(1, m):
            vg = self.dist[path[k - 1], path[k]] / max(glide_s[k - 1], 1.0)
            w_air = p.w_ref_ms * float(self.node_lift[min(self.bin_floor(times[k]), self.n_bins), path[k]])
            gain[k] = max(w_air, 0.0) * p.dolphin_core_m / max(vg, 1.0)

        def price(k: int, hour: float) -> float:
            rate = float(self.climb[min(self.bin_floor(hour), self.n_bins), path[k]])
            return 1.0 / rate if rate >= p.min_climb_ms else np.inf

        def reach(k: int, j: int) -> float:
            """Lowest departure height at k to arrive at j without circling in between."""
            r = need[j - 1]
            for q in range(j - 2, k - 1, -1):
                r = max(need[q], r - gain[q + 1] + loss[q])
            return r

        # the first leg is kept when it starts low from the takeoff; otherwise the house thermal is a station
        if legs[0]["v"] is None:
            prof, k0 = list(full_prof[:2]), 1
            alt, t = full_prof[1][2], full_prof[1][3]
        else:
            prof, k0 = [full_prof[0]], 0
            alt, t = float(self.zg[0]), times[0]
        out_times = [times[0]]
        for k in range(k0, m):
            cap = float(self.ceil[self.slot_of(self.bin_floor(t))])
            own = price(k, t)
            target = None
            for j in range(k + 1, m + 1):
                r = reach(k, j)
                if r > cap + 1e-6:
                    break
                if j == m or price(j, times[j]) < own:
                    target = r
                    break
            depart = max(alt, target) if target is not None else (max(alt, cap) if np.isfinite(own) else alt)
            if depart < need[k] - 1e-6:
                return None
            if depart > alt + 1e-6 and not np.isfinite(own):
                # arrived before the thermal works: hang on in weak lift until it does, then climb
                b0 = self.bin_floor(t)
                ok = np.flatnonzero(self.climb[b0 : self.n_bins + 1, path[k]] >= p.min_climb_ms)
                if not len(ok):
                    return None  # it never works again: keep the full-climb plan
                t = max(t, float(self.hours[b0 + ok[0]]))
                own = price(k, t)
            if depart > alt + 1.0:
                t += (depart - alt) * own / 3600.0
                prof.append((self.x[path[k]], self.y[path[k]], depart, t))
            if k > 0:
                out_times.append(t)
            t += glide_s[k] / 3600.0
            alt = depart - loss[k]
            if k + 1 < m:  # height gained crossing the core of the next thermal, counted on arrival there
                alt = min(alt + gain[k + 1], float(self.ceil[self.slot_of(self.bin_floor(t))]))
            prof.append((self.x[path[k + 1]], self.y[path[k + 1]], alt, t))
        out_times.append(t)
        if t > self.t_end + 1e-9 or np.any(np.diff(out_times) <= 0):
            return None
        return prof, out_times


def default_takeoff(terrain: Terrain, params: Params, lift: LiftModel) -> tuple[float, float]:
    """Pick a plausible takeoff: best lift one hour after the start among high, flyable ground."""
    f = lift.field(params.t_start + 1.0).copy()
    ceiling = params.weather.ceiling(params.to_solar(params.t_start + 1.0, terrain.lon_mid))
    f[terrain.z < np.percentile(terrain.z, 85)] = 0
    f[terrain.z > ceiling - 300] = 0
    r, c = np.unravel_index(f.argmax(), f.shape)
    x, y = terrain.rowcol_to_xy(r, c)
    return float(x), float(y)
