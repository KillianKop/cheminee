// cheminee web interface: MapLibre map (relief from the planner's DEM), takeoff picking, weather, planning jobs.
"use strict";

const $ = (id) => document.getElementById(id);
const LIFT_STOPS = [[0, [141, 156, 138]], [0.2, [201, 199, 154]], [0.45, [242, 193, 78]], [0.7, [240, 127, 60]], [1, [200, 30, 30]]];
const LIFT_MAX = 0.8; // lift index shown with the last colour
const BASEMAPS = {
  topo: {
    tiles: ["a", "b", "c"].map((s) => `https://${s}.tile.opentopomap.org/{z}/{x}/{y}.png`),
    maxzoom: 17,
    attribution: 'Map data © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, SRTM · ' +
      'style © <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)',
  },
  osm: {
    tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    maxzoom: 19,
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
};

const state = {
  cfg: null,
  map: null,
  takeoff: null, // {lon, lat, z}
  takeoffMarker: null,
  result: null,
  hourIdx: 0,
  shown: new Set(),
  run: 0, // id of the latest planning request
  frameQueued: false,
  liftLut: null,
  overlays: false, // map layers of the computed area, routes and thermals added
};

// -- helpers ---------------------------------------------------------------------------- //
async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const answer = await fetch(path, options);
  const data = await answer.json().catch(() => ({ error: `HTTP ${answer.status}` }));
  if (!answer.ok) throw new Error(data.error || `HTTP ${answer.status}`);
  return data;
}

function fmtHour(h) {
  const m = Math.round(h * 60);
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

function setStatus(text, error = false) {
  const el = $("status");
  el.hidden = !text;
  el.textContent = text || "";
  el.classList.toggle("error", error);
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") Object.assign(node.style, v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

function insideDem(lon, lat) {
  const b = state.cfg.dem.bounds;
  return lon >= b.lon_min && lon <= b.lon_max && lat >= b.lat_min && lat <= b.lat_max;
}

// -- form -------------------------------------------------------------------------------- //
function buildForm() {
  const cfg = state.cfg, d = cfg.defaults;
  $("dem-name").textContent = `${cfg.dem.name} · ${cfg.dem.width_km.toFixed(1)} × ${cfg.dem.height_km.toFixed(1)} km`;
  $("default-day").textContent = d.day_of_year;
  for (const name of Object.keys(cfg.presets)) $("preset").append(el("option", { value: name }, name.replaceAll("_", " ")));
  $("preset").value = d.preset;
  for (const id of ["t_start", "t_end", "utc_offset", "n_routes", "objective", "forecast_model"]) $(id).value = d[id];
  if (!cfg.forecast) {
    $("forecast").disabled = true;
    $("forecast").parentElement.title = "forecasts are disabled on this server";
  }
  for (const [name, f] of Object.entries(cfg.weather_fields)) {
    $("weather-fields").append(el("label", {}, `${f.label} (${f.unit})`,
      el("input", { type: "number", id: `w_${name}`, min: f.min, max: f.max, step: f.step, "data-field": name })));
  }
  updatePlaceholders();
  $("preset").addEventListener("change", updatePlaceholders);
  $("forecast").addEventListener("change", () => {
    if ($("forecast").checked && !$("date").value) {
      $("date").value = new Date(Date.now() + 86400e3).toISOString().slice(0, 10);
    }
    updatePlaceholders();
  });
  $("btn-reset-weather").addEventListener("click", () => {
    document.querySelectorAll("#weather-fields input").forEach((i) => (i.value = ""));
  });
  $("btn-weather").addEventListener("click", previewWeather);
  $("btn-plan").addEventListener("click", plan);
  $("btn-clear-takeoff").addEventListener("click", () => setTakeoff(null));
  $("basemap").addEventListener("change", setBasemap);
  $("exaggeration").addEventListener("change", setRelief);
  $("show-lift").addEventListener("change", showLift);
  $("hour").addEventListener("input", () => {
    state.hourIdx = +$("hour").value;
    if (state.frameQueued) return; // at most one repaint per animation frame while the slider moves
    state.frameQueued = true;
    requestAnimationFrame(() => { state.frameQueued = false; showLift(); });
  });
}

function updatePlaceholders() {
  const preset = state.cfg.presets[$("preset").value];
  const forecast = $("forecast").checked;
  document.querySelectorAll("#weather-fields input").forEach((input) => {
    const v = preset[input.dataset.field];
    input.placeholder = forecast ? "forecast" : v === null || v === undefined ? "auto" : String(v);
  });
}

function request() {
  const num = (id) => ($(id).value === "" ? undefined : Number($(id).value));
  const req = {
    preset: $("preset").value,
    objective: $("objective").value,
    n_routes: num("n_routes"),
    t_start: num("t_start"),
    t_end: num("t_end"),
    utc_offset: num("utc_offset"),
    shadows: $("shadows").checked,
    forecast: $("forecast").checked,
    forecast_model: $("forecast_model").value,
    weather: {},
  };
  if ($("date").value) req.date = $("date").value;
  if (state.takeoff) req.takeoff = { lon: state.takeoff.lon, lat: state.takeoff.lat };
  document.querySelectorAll("#weather-fields input").forEach((input) => {
    if (input.value !== "") req.weather[input.dataset.field] = Number(input.value);
  });
  return req;
}

// -- map ------------------------------------------------------------------------------------ //
function initMap() {
  const cfg = state.cfg, b = cfg.dem.bounds;
  const dem = {
    type: "raster-dem",
    tiles: [location.origin + cfg.dem.tiles.url],
    tileSize: 256,
    encoding: cfg.dem.tiles.encoding,
    maxzoom: cfg.dem.tiles.max_zoom,
    attribution: cfg.dem.attribution,
  };
  const { attribution, ...shade } = dem; // the notice is shown once, by the terrain source
  const raster = (name) => ({ type: "raster", tileSize: 256, ...BASEMAPS[name] });
  const map = new maplibregl.Map({
    container: "map",
    style: {
      version: 8,
      sources: { topo: raster("topo"), osm: raster("osm"), "dem-terrain": { ...dem, attribution }, "dem-shade": shade },
      layers: [
        { id: "background", type: "background", paint: { "background-color": "#e9e6df" } },
        { id: "topo", type: "raster", source: "topo" },
        { id: "osm", type: "raster", source: "osm", layout: { visibility: "none" } },
        { id: "hillshade", type: "hillshade", source: "dem-shade",
          paint: { "hillshade-exaggeration": 0.35, "hillshade-shadow-color": "#2f2f2f" } },
      ],
      terrain: { source: "dem-terrain", exaggeration: 1.5 },
    },
    bounds: [[b.lon_min, b.lat_min], [b.lon_max, b.lat_max]],
    fitBoundsOptions: { padding: 30 },
    pitch: 55,
    bearing: -15,
    maxPitch: 80,
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
  // "style.load", not "load": "load" waits for every visible tile, the far background included
  map.once("style.load", () => {
    addOverlays(map);
    state.overlays = true;
    if (state.result) drawResult();
  });
  map.on("click", onMapClick);
  map.on("mousemove", (ev) => {
    const { lng, lat } = ev.lngLat;
    $("cursor").textContent = `${lat.toFixed(4)}° N ${lng.toFixed(4)}° E` + (insideDem(lng, lat) ? "" : " · outside the computed area");
  });
  state.map = map;
}

function addOverlays(map) {
  const b = state.cfg.dem.bounds;
  const box = [[b.lon_min, b.lat_min], [b.lon_max, b.lat_min], [b.lon_max, b.lat_max], [b.lon_min, b.lat_max], [b.lon_min, b.lat_min]];
  // Darken what lies outside the DEM with four bands around it (a world-sized polygon with a hole does not
  // render reliably on terrain).
  const M = 3; // degrees around the DEM
  const rect = (w, s, e, n) => ({ type: "Feature", properties: { kind: "mask" },
    geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } });
  map.addSource("dem-area", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.getSource("dem-area").setData({ type: "FeatureCollection", features: [
      rect(b.lon_min - M, b.lat_max, b.lon_max + M, b.lat_max + M),
      rect(b.lon_min - M, b.lat_min - M, b.lon_max + M, b.lat_min),
      rect(b.lon_min - M, b.lat_min, b.lon_min, b.lat_max),
      rect(b.lon_max, b.lat_min, b.lon_max + M, b.lat_max),
      { type: "Feature", properties: { kind: "outline" }, geometry: { type: "LineString", coordinates: box } },
  ] });
  map.addLayer({ id: "dem-mask", type: "fill", source: "dem-area", filter: ["==", ["get", "kind"], "mask"],
    paint: { "fill-color": "#1d232a", "fill-opacity": 0.35, "fill-antialias": false } });
  map.addLayer({ id: "dem-outline", type: "line", source: "dem-area", filter: ["==", ["get", "kind"], "outline"],
    paint: { "line-color": "#1d232a", "line-width": 1.5, "line-dasharray": [3, 2] } });

  const empty = { type: "FeatureCollection", features: [] };
  for (const id of ["routes", "turnpoints", "nodes"]) map.addSource(id, { type: "geojson", data: empty });
  map.addLayer({ id: "route-casing", type: "line", source: "routes",
    layout: { "line-join": "round", "line-cap": "round" }, paint: { "line-color": "#ffffff", "line-width": 7, "line-opacity": 0.85 } });
  map.addLayer({ id: "route-line", type: "line", source: "routes",
    layout: { "line-join": "round", "line-cap": "round" }, paint: { "line-color": ["get", "color"], "line-width": 4 } });
  map.addLayer({ id: "route-legs", type: "line", source: "turnpoints", filter: ["==", ["geometry-type"], "LineString"],
    paint: { "line-color": ["get", "color"], "line-width": 2, "line-dasharray": [2, 2] } });
  map.addLayer({ id: "nodes", type: "circle", source: "nodes", paint: {
    "circle-radius": 4,
    "circle-color": ["interpolate", ["linear"], ["get", "climb"], 0, "#8d9c8a", 1.5, "#f2c14e", 3, "#c81e1e"],
    "circle-stroke-color": "#111", "circle-stroke-width": 1 } });
  map.addLayer({ id: "turnpoints", type: "circle", source: "turnpoints", filter: ["==", ["geometry-type"], "Point"],
    paint: { "circle-radius": 7, "circle-color": ["get", "color"], "circle-stroke-color": "#fff", "circle-stroke-width": 2 } });

  const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
  map.on("mouseenter", "nodes", (ev) => {
    const f = ev.features[0];
    map.getCanvas().style.cursor = "pointer";
    popup.setLngLat(f.geometry.coordinates)
      .setHTML(`thermal · ground ${f.properties.z} m · best climb ${f.properties.climb.toFixed(1)} m/s`).addTo(map);
  });
  map.on("mouseleave", "nodes", () => { map.getCanvas().style.cursor = ""; popup.remove(); });
}

function setBasemap() {
  const map = state.map, choice = $("basemap").value;
  for (const id of Object.keys(BASEMAPS)) map.setLayoutProperty(id, "visibility", id === choice ? "visible" : "none");
}

function setRelief() {
  const map = state.map, ex = Number($("exaggeration").value);
  if (ex === 0) {
    map.setTerrain(null);
    map.easeTo({ pitch: 0, bearing: 0 });
  } else {
    map.setTerrain({ source: "dem-terrain", exaggeration: ex });
    if (map.getPitch() < 20) map.easeTo({ pitch: 55 });
  }
}

async function onMapClick(ev) {
  const { lng, lat } = ev.lngLat;
  if (!insideDem(lng, lat)) {
    setStatus("The takeoff must be inside the computed area (dashed outline).", true);
    return;
  }
  setStatus("");
  try {
    const point = await api(`/api/elevation?lon=${lng.toFixed(6)}&lat=${lat.toFixed(6)}`);
    setTakeoff({ lon: lng, lat, z: point.z });
  } catch (err) {
    setStatus(err.message, true);
  }
}

function setTakeoff(point) {
  state.takeoff = point;
  const info = $("takeoff-info");
  if (state.takeoffMarker) state.takeoffMarker.remove();
  state.takeoffMarker = null;
  if (point) {
    info.innerHTML = `<b>${point.z} m</b> · ${point.lat.toFixed(4)}° N, ${point.lon.toFixed(4)}° E<br>` +
      '<span class="hint">Click elsewhere to move it.</span>';
    state.takeoffMarker = new maplibregl.Marker({ element: el("div", { class: "marker-takeoff", title: "takeoff" }) })
      .setLngLat([point.lon, point.lat]).addTo(state.map);
  } else {
    info.textContent = "Click on the map, inside the computed area, to place the takeoff. Without one, the planner " +
      "picks a high spot with good lift.";
  }
  info.classList.toggle("hint", !point);
  $("btn-clear-takeoff").hidden = !point;
}

// -- lift overlay -------------------------------------------------------------------------- //
function liftLut() {
  if (state.liftLut) return state.liftLut;
  const lut = new Uint8ClampedArray(101 * 4);
  for (let i = 0; i <= 100; i++) {
    const v = Math.min(i / 100 / LIFT_MAX, 1);
    let k = 0;
    while (k < LIFT_STOPS.length - 2 && v > LIFT_STOPS[k + 1][0]) k++;
    const [v0, c0] = LIFT_STOPS[k], [v1, c1] = LIFT_STOPS[k + 1];
    const f = Math.min(Math.max((v - v0) / (v1 - v0), 0), 1);
    for (let c = 0; c < 3; c++) lut[i * 4 + c] = c0[c] + f * (c1[c] - c0[c]);
    lut[i * 4 + 3] = 255 * Math.min(Math.max((i / 100 - 0.08) / 0.35, 0), 0.8); // weak lift stays transparent
  }
  return (state.liftLut = lut);
}

function liftImage(frame) {
  const g = state.cfg.lift_grid, lut = liftLut();
  const canvas = el("canvas", { width: g.nx, height: g.ny });
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(g.nx, g.ny);
  let p = 0;
  for (let r = 0; r < g.ny; r++) {
    const row = frame[r];
    for (let c = 0; c < g.nx; c++, p += 4) {
      const i = Math.min(Math.max(row[c], 0), 100) * 4;
      img.data[p] = lut[i]; img.data[p + 1] = lut[i + 1]; img.data[p + 2] = lut[i + 2]; img.data[p + 3] = lut[i + 3];
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

function showLift() {
  const map = state.map, res = state.result;
  if (!map || !res || !state.overlays) return;
  $("hour-label").textContent = fmtHour(res.lift.hours[state.hourIdx]);
  const visible = $("show-lift").checked;
  if (map.getLayer("lift")) map.setLayoutProperty("lift", "visibility", visible ? "visible" : "none");
  if (!visible) return;
  const url = liftImage(res.lift.frames[state.hourIdx]);
  const [w, n, e, s] = state.cfg.lift_grid.bounds;
  const coordinates = [[w, n], [e, n], [e, s], [w, s]];
  const source = map.getSource("lift");
  if (source) {
    source.updateImage({ url, coordinates });
  } else {
    map.addSource("lift", { type: "image", url, coordinates });
    map.addLayer({ id: "lift", type: "raster", source: "lift",
      paint: { "raster-fade-duration": 0, "raster-resampling": "linear" } }, "dem-mask");
  }
}

// -- weather preview --------------------------------------------------------------------- //
async function previewWeather() {
  const out = $("weather-out");
  out.textContent = "Loading…";
  try {
    const w = await api("/api/weather", request());
    const arrow = (dir) => `<span class="arrow" style="transform: rotate(${dir}deg)">↓</span>`;
    const rows = w.hours.map((h) => `<tr><td>${fmtHour(h.hour)}</td><td>${h.ceiling_m}</td><td>${h.cloud_base_m}</td>` +
      `<td>${arrow(h.wind_dir_deg)} ${h.wind_kmh}</td><td>${arrow(h.wind_aloft_dir_deg)} ${h.wind_aloft_kmh}</td>` +
      `<td>${Math.round(100 * h.sun)} %</td><td>${h.strength.toFixed(2)}</td></tr>`).join("");
    out.innerHTML = `<table><thead><tr><th>local</th><th>ceiling</th><th>base</th><th>wind</th><th>3000 m</th>` +
      `<th>sun</th><th>instab.</th></tr></thead><tbody>${rows}</tbody></table>` +
      `<div class="sources">m, km/h · arrows point where the wind blows · ${w.sources.join(" + ")}</div>`;
  } catch (err) {
    out.innerHTML = "";
    out.append(el("div", { class: "status error" }, err.message));
  }
}

// -- planning ------------------------------------------------------------------------------ //
async function plan() {
  const run = (state.run += 1); // a newer click abandons the older job's follow-up
  $("btn-plan").disabled = true;
  $("log").hidden = false;
  $("log").textContent = "";
  setStatus("Starting…");
  const t0 = performance.now();
  try {
    const { job } = await api("/api/plan", request());
    await follow(job, run, t0);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    if (run === state.run) $("btn-plan").disabled = false;
  }
}

// Long polling: the server answers as soon as the job logs a line or finishes (no fixed polling delay).
async function follow(id, run, t0) {
  let since = 0;
  for (;;) {
    const job = await api(`/api/jobs/${id}?since=${since}&wait=15`);
    if (run !== state.run) return;
    since = job.log.length;
    $("log").textContent = job.log.join("\n");
    $("log").scrollTop = $("log").scrollHeight;
    if (job.status === "error") throw new Error(job.error);
    if (job.status === "done") {
      showResult(job.result, job.elapsed_s, (performance.now() - t0) / 1000);
      return;
    }
    setStatus(job.status === "queued" ? "Waiting for the previous computation…" : "Computing…");
  }
}

function showResult(result, elapsed, total) {
  state.result = result;
  state.shown = new Set(result.routes.length ? [0] : []);
  const hours = result.lift.hours;
  let best = 0;
  hours.forEach((h, i) => { if (Math.abs(h - 14) < Math.abs(hours[best] - 14)) best = i; });
  state.hourIdx = best;
  $("hour").max = hours.length - 1;
  $("hour").value = best;
  $("lift-box").hidden = false;
  if (!state.takeoff) setTakeoff({ lon: result.takeoff.lon, lat: result.takeoff.lat, z: result.takeoff.z });
  const timing = `${elapsed} s server, ${total.toFixed(1)} s in all`;
  setStatus(result.routes.length
    ? `${result.routes.length} routes (${timing}) · ${result.lift_name} lift. Click a route to show it.`
    : `No flight found with this weather (${timing}). Try a later takeoff, another takeoff or the free distance.`,
    !result.routes.length);
  listRoutes();
  drawResult();
}

function drawResult() {
  const map = state.map, res = state.result;
  if (!map || !res || !state.overlays) return;
  const shown = res.routes.map((r, k) => [r, k]).filter(([, k]) => state.shown.has(k));
  map.getSource("routes").setData({ type: "FeatureCollection", features: shown.map(([r, k]) => ({
    type: "Feature", properties: { color: r.color, route: k },
    geometry: { type: "LineString", coordinates: r.path.map((p) => [p[0], p[1]]) },
  })) });
  map.getSource("turnpoints").setData({ type: "FeatureCollection", features: shown.flatMap(([r]) => [
    { type: "Feature", properties: { color: r.color }, geometry: { type: "LineString", coordinates: r.turnpoints_lonlat } },
    ...r.turnpoints_lonlat.map((c) => ({ type: "Feature", properties: { color: r.color }, geometry: { type: "Point", coordinates: c } })),
  ]) });
  map.getSource("nodes").setData({ type: "FeatureCollection", features: res.nodes.lonlat.map((c, i) => ({
    type: "Feature", properties: { z: res.nodes.z[i], climb: res.nodes.climb_ms[i] },
    geometry: { type: "Point", coordinates: c },
  })) });
  showLift();
}

// -- route list and altitude profile ------------------------------------------------------- //
function profileSvg(r, ceiling) {
  const W = 320, H = 120, L = 30, B = 14;
  const path = r.path, t0 = path[0][3], t1 = path[path.length - 1][3];
  const zMin = Math.min(...path.map((p) => p[4])) - 100;
  const zMax = Math.max(ceiling, ...path.map((p) => p[2])) + 100;
  const x = (t) => L + ((t - t0) / Math.max(t1 - t0, 1e-6)) * (W - L - 4);
  const y = (z) => 4 + (1 - (z - zMin) / (zMax - zMin)) * (H - B - 4);
  const flight = path.map((p) => `${x(p[3]).toFixed(1)},${y(p[2]).toFixed(1)}`).join(" ");
  const ground = `${x(t0)},${y(zMin)} ` + path.map((p) => `${x(p[3]).toFixed(1)},${y(p[4]).toFixed(1)}`).join(" ") + ` ${x(t1)},${y(zMin)}`;
  const ticks = [zMin + 100, (zMin + zMax) / 2, zMax - 100].map((z) => Math.round(z / 100) * 100);
  return `<svg class="profile" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="altitude profile">` +
    `<polygon class="ground" points="${ground}"/>` +
    `<line class="ceiling" x1="${L}" x2="${W - 4}" y1="${y(ceiling)}" y2="${y(ceiling)}"/>` +
    `<polyline class="flight" stroke="${r.color}" points="${flight}"/>` +
    ticks.map((z) => `<text x="2" y="${y(z) + 3}">${z}</text>`).join("") +
    `<text x="${L}" y="${H - 2}">${fmtHour(t0)}</text><text x="${W - 34}" y="${H - 2}">${fmtHour(t1)}</text></svg>`;
}

function listRoutes() {
  const list = $("routes"), res = state.result;
  const ceiling = Math.max(...res.ceiling.m);
  list.innerHTML = "";
  res.routes.forEach((r, k) => {
    const segs = r.segments.map((s) => s.type === "climb"
      ? `<tr><td>${fmtHour(s.t0)}</td><td>climb</td><td>${s.a0} → ${s.a1} m</td><td>${s.rate.toFixed(1)} m/s</td></tr>`
      : `<tr><td>${fmtHour(s.t0)}</td><td>glide ${(s.dist / 1000).toFixed(1)} km</td><td>${s.a0} → ${s.a1} m</td>` +
        `<td>${s.airspeed_kmh} km/h</td></tr>`).join("");
    const item = el("li", { style: { "--c": r.color } });
    item.innerHTML = `<div class="title"><span>${k + 1}. ${r.label}</span><span>${r.distance_km} km · ${r.points} pts</span></div>` +
      `<div class="meta">${fmtHour(r.t_start)}–${fmtHour(r.t_end)} · ${r.duration_h.toFixed(1)} h · ${r.speed_kmh} km/h ·` +
      ` <a href="#" class="details">profile & segments</a></div>` +
      `<div class="more">${profileSvg(r, ceiling)}<table><thead><tr><th>time</th><th>phase</th><th>altitude</th>` +
      `<th>rate / airspeed</th></tr></thead><tbody>${segs}</tbody></table></div>`;
    item.classList.toggle("off", !state.shown.has(k));
    item.addEventListener("click", (ev) => {
      if (ev.target.classList.contains("details")) {
        ev.preventDefault();
        item.classList.toggle("open");
        return;
      }
      if (ev.target.closest(".more")) return;
      if (state.shown.has(k)) state.shown.delete(k); else state.shown.add(k);
      item.classList.toggle("off", !state.shown.has(k));
      drawResult();
    });
    list.append(item);
  });
}

// -- start ---------------------------------------------------------------------------------- //
(async function main() {
  try {
    state.cfg = await api("/api/config");
  } catch (err) {
    document.body.textContent = `Cannot reach the cheminee server: ${err.message}`;
    return;
  }
  buildForm();
  initMap();
})();
