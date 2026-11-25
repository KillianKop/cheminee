"""Minimal IGC reader / writer (date header + B fix records)."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_B_RECORD = re.compile(
    r"^B(\d{2})(\d{2})(\d{2})(\d{2})(\d{5})([NS])(\d{3})(\d{5})([EW])([AV])"
    r"(-\d{4}|\d{5})(-\d{4}|\d{5})"
)
_DATE = re.compile(r"^HFDTE(?:DATE:)?(\d{2})(\d{2})(\d{2})")


def find_igc(directory: str | Path) -> list[Path]:
    """IGC files under `directory` (sub-folders included, `.igc` in any case), sorted, each listed once.

    Globbing `*.igc` then `*.IGC` lists every file twice on case-insensitive file systems (Windows, macOS).
    """
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".igc" and p.is_file())


@dataclass
class IgcTrack:
    date: dt.date
    t: np.ndarray  # seconds since 00:00 UTC of `date` (monotonic, may exceed 86400)
    lat: np.ndarray  # degrees
    lon: np.ndarray  # degrees
    alt: np.ndarray  # metres AMSL (GPS altitude, barometric as a fallback)
    name: str = ""

    @property
    def day_of_year(self) -> int:
        return self.date.timetuple().tm_yday

    def __len__(self) -> int:
        return len(self.t)


def read_igc(path: str) -> IgcTrack:
    date, rows = None, []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("B"):
                m = _B_RECORD.match(line)
                if not m:
                    continue
                hh, mm, ss, lat_d, lat_m, ns, lon_d, lon_m, ew, _, baro, gps = m.groups()
                lat = (int(lat_d) + int(lat_m) / 60000.0) * (1 if ns == "N" else -1)
                lon = (int(lon_d) + int(lon_m) / 60000.0) * (1 if ew == "E" else -1)
                rows.append((int(hh) * 3600 + int(mm) * 60 + int(ss), lat, lon, int(baro), int(gps)))
            elif date is None and line.startswith("HFDTE"):
                m = _DATE.match(line)
                if m:
                    d, mo, y = (int(v) for v in m.groups())
                    date = dt.date(2000 + y, mo, d)
    if date is None or len(rows) < 2:
        raise ValueError(f"{path}: not a usable IGC file (missing date header or fixes)")
    a = np.array(rows, dtype=float)
    t = a[:, 0] + 86400.0 * np.cumsum(np.r_[0, np.diff(a[:, 0]) < -43200])  # midnight rollover
    alt = a[:, 4] if np.any(a[:, 4] != 0) else a[:, 3]
    keep = np.r_[True, np.diff(t) > 0]  # drop duplicated fixes
    return IgcTrack(date, t[keep], a[keep, 1], a[keep, 2], alt[keep], name=str(path))


def _coord(value: float, deg_digits: int, pos: str, neg: str) -> str:
    hemi = pos if value >= 0 else neg
    value = abs(value)
    deg = int(value)
    return f"{deg:0{deg_digits}d}{int(round((value - deg) * 60000)):05d}{hemi}"


def write_igc(path: str, track: IgcTrack, pilot: str = "SYNTHETIC") -> None:
    lines = [
        "AXXXcheminee",
        f"HFDTE{track.date:%d%m%y}",
        f"HFPLTPILOTINCHARGE:{pilot}",
        "HFGTYGLIDERTYPE:PARAGLIDER",
        "HFDTM100GPSDATUM:WGS-1984",
    ]
    for t, lat, lon, alt in zip(track.t, track.lat, track.lon, track.alt):
        s = int(round(t)) % 86400
        a = int(round(alt))
        lat_s, lon_s = _coord(lat, 2, "N", "S"), _coord(lon, 3, "E", "W")
        if lat_s[2:7] == "60000" or lon_s[3:8] == "60000":  # rounding to a full degree
            lat_s, lon_s = _coord(round(lat, 6) + 1e-9, 2, "N", "S"), _coord(round(lon, 6) + 1e-9, 3, "E", "W")
        lines.append(f"B{s // 3600:02d}{s % 3600 // 60:02d}{s % 60:02d}{lat_s}{lon_s}A{a:05d}{a:05d}")
    with open(path, "w") as f:
        f.write("\r\n".join(lines) + "\r\n")
