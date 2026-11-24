"""Thermal (climb) detection in a track, in a local metric frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalConfig:
    window_s: float = 20.0  # averaging window of the vertical speed
    min_climb_ms: float = 0.5  # smoothed vertical speed above which the pilot "climbs"
    max_gap_s: float = 15.0  # climbing phases closer than this are merged
    min_duration_s: float = 60.0  # shorter climbs are ignored
    min_gain_m: float = 50.0  # smaller height gains are ignored
    entry_window_s: float = 40.0  # window used to locate the entry / exit of the thermal
    min_turn_rate_deg_s: float = 3.0  # mean rotation of the ground track: rejects straight climbs


@dataclass
class Thermal:
    i0: int  # first / last fix index
    i1: int
    t0: float  # s
    t1: float
    x: float  # entry point (m): proxy of the trigger point on the ground
    y: float
    climb_ms: float  # mean climb rate
    drift: np.ndarray  # (2,) horizontal drift of the thermal column (m/s) ~ wind

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


def vertical_speed(t: np.ndarray, alt: np.ndarray, window_s: float) -> np.ndarray:
    """Vertical speed averaged over a centred window (m/s)."""
    half = 0.5 * window_s
    return (np.interp(t + half, t, alt) - np.interp(t - half, t, alt)) / window_s


def detect_thermals(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, alt: np.ndarray, cfg: ThermalConfig | None = None
) -> list[Thermal]:
    """Detect sustained climbs. `x`, `y` are metric coordinates of the fixes."""
    cfg = cfg or ThermalConfig()
    climbing = vertical_speed(t, alt, cfg.window_s) > cfg.min_climb_ms
    edges = np.flatnonzero(np.diff(np.r_[0, climbing.astype(int), 0]))
    starts, ends = list(edges[::2]), list(edges[1::2] - 1)
    # merge phases separated by short gaps
    merged = []
    for i0, i1 in zip(starts, ends):
        if merged and t[i0] - t[merged[-1][1]] <= cfg.max_gap_s:
            merged[-1][1] = i1
        else:
            merged.append([i0, i1])

    thermals = []
    for i0, i1 in merged:
        duration, gain = t[i1] - t[i0], alt[i1] - alt[i0]
        if duration < cfg.min_duration_s or gain < cfg.min_gain_m:
            continue
        # A thermal is circled: the ground track must keep turning (a straight glide through
        # rising air is not a thermal; it is used as a glide sample instead).
        heading = np.unwrap(np.arctan2(np.diff(y[i0 : i1 + 1]), np.diff(x[i0 : i1 + 1])))
        if abs(np.degrees(heading[-1] - heading[0])) / duration < cfg.min_turn_rate_deg_s:
            continue
        win = min(cfg.entry_window_s, 0.5 * duration)
        head = (t >= t[i0]) & (t <= t[i0] + win)
        tail = (t >= t[i1] - win) & (t <= t[i1])
        c0 = np.array([x[head].mean(), y[head].mean()])
        c1 = np.array([x[tail].mean(), y[tail].mean()])
        drift = (c1 - c0) / max(t[tail].mean() - t[head].mean(), 1.0)
        thermals.append(
            Thermal(
                int(i0), int(i1), float(t[i0]), float(t[i1]), float(c0[0]), float(c0[1]), float(gain / duration), drift
            )
        )
    return thermals


def estimate_wind(thermals: list[Thermal]) -> tuple[float, float]:
    """Mean wind of a flight from the drift of its thermals.

    Returns (direction the wind comes FROM in degrees, speed in m/s).
    """
    if not thermals:
        return 0.0, 0.0
    weights = np.array([th.duration_s for th in thermals])
    v = (np.stack([th.drift for th in thermals]) * weights[:, None]).sum(axis=0) / weights.sum()
    return float(np.degrees(np.arctan2(-v[0], -v[1])) % 360.0), float(np.hypot(*v))
