"""Network access to thermal.kk7.ch: the only module of cheminee_kk7 that does HTTP.

kk7 usage policy (https://thermal.kk7.ch, "Usage Policy"): identify the client in the User-Agent, add `?src=`
to tile requests, fetch only the area in use, cache (tiles live 30 days), at most 10 requests/s, no bulk
prefetching. Everything downloaded is cached under `data/kk7_cache/` (git-ignored, never committed).
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from .tiles import decode_tile, tms_row

BASE_URL = "https://thermal.kk7.ch"
USER_AGENT = "cheminee-kk7/0.1 (paragliding flight planner prototype)"
SRC = "cheminee"
MIN_INTERVAL_S = 0.25  # at most 4 requests/s (kk7 allows 10)
CACHE_MAX_AGE_S = 30 * 24 * 3600.0
DEFAULT_CACHE = os.path.join("data", "kk7_cache")

_last_request = 0.0


def _get(url: str, accept: str | None = None) -> bytes | None:
    """GET with rate limiting. Returns None when kk7 has no data (204 / 404)."""
    global _last_request
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as answer:
            return None if answer.status == 204 else answer.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    finally:
        _last_request = time.monotonic()


def _usable(path: str, offline: bool) -> bool:
    return os.path.exists(path) and (offline or time.time() - os.path.getmtime(path) < CACHE_MAX_AGE_S)


def tile_path(cache_dir: str, key: str, zoom: int, x: int, y: int) -> str:
    """Cache file of the XYZ tile (x, y), stored under its TMS row like the kk7 URL."""
    return os.path.join(cache_dir, f"thermals_{key}", str(zoom), str(x), f"{tms_row(y, zoom)}.png")


def fetch_tile(
    key: str, zoom: int, x: int, y: int, cache_dir: str = DEFAULT_CACHE, offline: bool = False
) -> tuple[np.ndarray | None, str]:
    """Decoded probability of one tile (None = no data) and where it came from: cache, download, missing."""
    path = tile_path(cache_dir, key, zoom, x, y)
    empty = path + ".empty"  # kk7 answered "no data": do not ask again
    if _usable(path, offline):
        with open(path, "rb") as f:
            return decode_tile(np.asarray(Image.open(io.BytesIO(f.read())).convert("RGBA"))), "cache"
    if _usable(empty, offline) or offline:
        return None, "missing"
    data = _get(f"{BASE_URL}/tiles/thermals_{key}/{zoom}/{x}/{tms_row(y, zoom)}.png?src={SRC}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not data:
        open(empty, "w").close()
        return None, "missing"
    with open(path, "wb") as f:
        f.write(data)
    return decode_tile(np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))), "download"


def fetch_layer(
    tiles: list[tuple[int, int]], key: str, zoom: int, cache_dir: str = DEFAULT_CACHE, offline: bool = False
) -> dict[tuple[int, int], np.ndarray | None]:
    """Decoded tiles of one kk7 layer (`<season>_<time>`, e.g. `jul_07`) over a list of XYZ tiles."""
    t0 = time.perf_counter()
    out, sources = {}, {"cache": 0, "download": 0, "missing": 0}
    for x, y in tiles:
        out[(x, y)], source = fetch_tile(key, zoom, x, y, cache_dir, offline)
        sources[source] += 1
    print(
        f"[kk7] layer {key} z{zoom}: {len(tiles)} tiles ({sources['cache']} cached, {sources['download']} "
        f"downloaded, {sources['missing']} without data) in {time.perf_counter() - t0:.1f} s"
    )
    return out


def fetch_hotspots(
    bbox: tuple[float, float, float, float],
    key: str,
    limit: int = 250,
    cache_dir: str = DEFAULT_CACHE,
    offline: bool = False,
) -> dict | None:
    """kk7 hotspots (GeoJSON points with a `probability` property) inside (lon_min, lat_min, lon_max, lat_max)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    s, w = np.floor(lat_min * 1000) / 1000, np.floor(lon_min * 1000) / 1000
    n, e = np.ceil(lat_max * 1000) / 1000, np.ceil(lon_max * 1000) / 1000
    area = f"{s:.3f},{w:.3f},{n:.3f},{e:.3f}"
    path = os.path.join(cache_dir, f"hotspots_{key}_{area.replace(',', '_')}_{limit}.json")
    if _usable(path, offline):
        with open(path) as f:
            return json.load(f)
    if offline:
        return None
    # the hotspot API refuses unknown query parameters (`src` included)
    data = _get(f"{BASE_URL}/api/hotspots/geojson/{key}/{area}?limit={limit}", accept="application/geo+json")
    if not data:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    hotspots = json.loads(data)
    print(f"[kk7] hotspots {key}: {len(hotspots.get('features', []))} points downloaded")
    return hotspots
