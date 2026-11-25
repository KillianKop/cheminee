// cheminee web interface: 3D terrain (plotly.js), takeoff picking, weather, planning jobs.
"use strict";

const $ = (id) => document.getElementById(id);
const TERRAIN_SCALE = [[0, "#5b7f4a"], [0.3, "#9fae73"], [0.55, "#cbb98b"], [0.8, "#a68c74"], [1, "#f4f1ec"]];
const LIFT_SCALE = [[0, "#8d9c8a"], [0.2, "#c9c79a"], [0.45, "#f2c14e"], [0.7, "#f07f3c"], [1, "#c81e1e"]];

const state = {
  cfg: null,
  takeoff: null, // {x_km, y_km, z}
  result: null,
  hourIdx: 0,
  shown: new Set(),
  run: 0, // id of the latest planning request
  frameQueued: false,
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

function lonLat(xKm, yKm) {
  const d = state.cfg.dem, b = d.bounds;
  return [b.lon_min + (xKm / d.width_km) * (b.lon_max - b.lon_min), b.lat_min + (yKm / d.height_km) * (b.lat_max - b.lat_min)];
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
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
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
      const tomorrow = new Date(Date.now() + 86400e3);
      $("date").value = tomorrow.toISOString().slice(0, 10);
    }
    updatePlaceholders();
  });
  $("btn-reset-weather").addEventListener("click", () => {
    document.querySelectorAll("#weather-fields input").forEach((i) => (i.value = ""));
  });
  $("btn-weather").addEventListener("click", previewWeather);
  $("btn-plan").addEventListener("click", plan);
  $("btn-clear-takeoff").addEventListener("click", () => setTakeoff(null));
  $("colour").addEventListener("change", draw);
  $("exaggeration").addEventListener("change", draw);
  $("hour").addEventListener("input", () => {
    state.hourIdx = +$("hour").value;
    if (state.frameQueued) return; // at most one repaint per animation frame while the slider moves
    state.frameQueued = true;
    requestAnimationFrame(() => { state.frameQueued = false; recolour(); });
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
  if (state.takeoff) req.takeoff = { x_km: state.takeoff.x_km, y_km: state.takeoff.y_km };
  document.querySelectorAll("#weather-fields input").forEach((input) => {
    if (input.value !== "") req.weather[input.dataset.field] = Number(input.value);
  });
  return req;
}

// -- takeoff ------------------------------------------------------------------------------ //
function setTakeoff(point) {
  state.takeoff = point;
  const info = $("takeoff-info");
  if (point) {
    const [lon, lat] = lonLat(point.x_km, point.y_km);
    info.innerHTML = `<b>${Math.round(point.z)} m</b> · ${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E<br>` +
      `<span class="hint">x ${point.x_km.toFixed(2)} km, y ${point.y_km.toFixed(2)} km. Click elsewhere to move it.</span>`;
  } else {
    info.textContent = "Click on the terrain to place the takeoff. Without one, the planner picks a high spot with good lift.";
  }
  info.classList.toggle("hint", !point);
  $("btn-clear-takeoff").hidden = !point;
  draw();
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
  const run = (state.run = (state.run || 0) + 1); // a newer click abandons the older job's follow-up
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
  const scale = result.lift.scale;
  result.lift.frames = result.lift.frames.map((f) => f.map((row) => row.map((v) => v * scale)));
  state.result = result;
  state.shown = new Set(result.routes.length ? [0] : []);
  const hours = result.lift.hours;
  let best = 0;
  hours.forEach((h, i) => { if (Math.abs(h - 14) < Math.abs(hours[best] - 14)) best = i; });
  state.hourIdx = best;
  $("hour").max = hours.length - 1;
  $("hour").value = best;
  $("hour-box").hidden = false;
  $("colour").value = "lift";
  if (!state.takeoff) {
    const t = result.takeoff;
    state.takeoff = { x_km: t.x_km, y_km: t.y_km, z: t.z };
    setTakeoff(state.takeoff);
  }
  const timing = `${elapsed} s server, ${total.toFixed(1)} s in all`;
  setStatus(result.routes.length
    ? `${result.routes.length} routes (${timing}) · ${result.lift_name} lift. Click a route to show it.`
    : `No flight found with this weather (${timing}). Try a later takeoff, another takeoff or the free distance.`,
    !result.routes.length);
  listRoutes();
  draw();
}

function listRoutes() {
  const list = $("routes");
  list.innerHTML = "";
  state.result.routes.forEach((r, k) => {
    const segs = r.segments.map((s) => s.type === "climb"
      ? `<tr><td>${fmtHour(s.t0)}</td><td>climb</td><td>${s.a0} → ${s.a1} m</td><td>${s.rate.toFixed(1)} m/s</td></tr>`
      : `<tr><td>${fmtHour(s.t0)}</td><td>glide ${(s.dist / 1000).toFixed(1)} km</td><td>${s.a0} → ${s.a1} m</td>` +
        `<td>${s.airspeed_kmh} km/h</td></tr>`).join("");
    const item = el("li", { style: { "--c": r.color } });
    item.innerHTML = `<div class="title"><span>${k + 1}. ${r.label}</span><span>${r.distance_km} km · ${r.points} pts</span></div>` +
      `<div class="meta">${fmtHour(r.t_start)}–${fmtHour(r.t_end)} · ${r.duration_h.toFixed(1)} h · ${r.speed_kmh} km/h ·` +
      ` <a href="#" class="details">segments</a></div>` +
      `<table><thead><tr><th>time</th><th>phase</th><th>altitude</th><th>rate / airspeed</th></tr></thead><tbody>${segs}</tbody></table>`;
    item.classList.toggle("off", !state.shown.has(k));
    item.addEventListener("click", (ev) => {
      if (ev.target.classList.contains("details")) {
        ev.preventDefault();
        item.classList.toggle("open");
        return;
      }
      if (state.shown.has(k)) state.shown.delete(k); else state.shown.add(k);
      item.classList.toggle("off", !state.shown.has(k));
      draw();
    });
    list.append(item);
  });
}

// -- 3D scene ------------------------------------------------------------------------------- //
function draw() {
  const cfg = state.cfg, g = cfg.grid, res = state.result;
  const liftMode = $("colour").value === "lift" && res;
  const frame = liftMode ? res.lift.frames[state.hourIdx] : null;
  $("hour-label").textContent = res ? fmtHour(res.lift.hours[state.hourIdx]) : "";

  const traces = [{
    type: "surface",
    x: g.x_km,
    y: g.y_km,
    z: g.z,
    surfacecolor: frame || g.z,
    colorscale: frame ? LIFT_SCALE : TERRAIN_SCALE,
    cmin: frame ? 0 : cfg.dem.z_min,
    cmax: frame ? 0.8 : cfg.dem.z_max,
    colorbar: { title: { text: frame ? "lift" : "m" }, len: 0.4, x: 0.0, thickness: 12 },
    lighting: { ambient: 0.55, diffuse: 0.9, specular: 0.05, roughness: 0.9 },
    lightposition: { x: -2000, y: 2000, z: 6000 },
    hovertemplate: frame
      ? "%{z:.0f} m · lift %{surfacecolor:.2f}<extra>click: takeoff here</extra>"
      : "%{z:.0f} m<extra>click: takeoff here</extra>",
  }];

  let zTop = cfg.dem.z_max + 100;
  if (res) {
    const ceiling = Math.max(...res.ceiling.m);
    zTop = Math.max(zTop, ceiling + 100);
    traces.push({
      type: "scatter3d", mode: "markers", x: res.nodes.x_km, y: res.nodes.y_km, z: res.nodes.z.map((z) => z + 30),
      marker: { size: 2.5, color: "#111" }, hoverinfo: "skip",
    });
    res.routes.forEach((r, k) => {
      if (!state.shown.has(k)) return;
      traces.push({
        type: "scatter3d", mode: "lines",
        x: r.profile.map((p) => p[0]), y: r.profile.map((p) => p[1]), z: r.profile.map((p) => p[2]),
        text: r.profile.map((p) => fmtHour(p[3])),
        line: { color: r.color, width: 6 },
        hovertemplate: `%{text} · %{z:.0f} m<extra>route ${k + 1}</extra>`,
      });
      traces.push({
        type: "scatter3d", mode: "lines+markers",
        x: r.turnpoints_km.map((p) => p[0]), y: r.turnpoints_km.map((p) => p[1]),
        z: r.turnpoints_km.map(() => ceiling),
        line: { color: r.color, width: 2, dash: "dash" },
        marker: { size: 4, color: r.color, symbol: "diamond" },
        hoverinfo: "skip",
      });
    });
  }
  if (state.takeoff) {
    traces.push({
      type: "scatter3d", mode: "markers+text", x: [state.takeoff.x_km], y: [state.takeoff.y_km], z: [state.takeoff.z + 40],
      text: ["takeoff"], textposition: "top center",
      marker: { size: 7, color: "#ffd400", line: { color: "#111", width: 2 } }, hoverinfo: "skip",
    });
  }

  const exaggeration = Number($("exaggeration").value);
  const zMin = cfg.dem.z_min - 50;
  const layout = {
    uirevision: "keep",
    showlegend: false,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: getComputedStyle(document.body).backgroundColor,
    scene: {
      xaxis: { title: { text: "x east (km)" } },
      yaxis: { title: { text: "y north (km)" } },
      zaxis: { title: { text: "alt (m)" }, range: [zMin, zTop], nticks: 4 },
      aspectmode: "manual",
      aspectratio: {
        x: 1,
        y: cfg.dem.height_km / cfg.dem.width_km,
        z: (exaggeration * (zTop - zMin)) / (cfg.dem.width_km * 1000),
      },
      camera: { eye: { x: -0.7, y: -1.0, z: 0.75 } },
    },
  };
  Plotly.react("plot", traces, layout, { responsive: true, displaylogo: false });
}

// Hour slider: only the colours of the surface change, the rest of the scene is left alone.
function recolour() {
  const res = state.result;
  $("hour-label").textContent = res ? fmtHour(res.lift.hours[state.hourIdx]) : "";
  if (!res || $("colour").value !== "lift") return;
  Plotly.restyle("plot", { surfacecolor: [res.lift.frames[state.hourIdx]] }, [0]);
}

function bindClicks() {
  const plot = $("plot");
  plot.on("plotly_click", (ev) => {
    const p = ev.points.find((q) => q.curveNumber === 0);
    if (p) setTakeoff({ x_km: p.x, y_km: p.y, z: p.z });
  });
  plot.on("plotly_hover", (ev) => {
    const p = ev.points.find((q) => q.curveNumber === 0);
    if (!p) return;
    const [lon, lat] = lonLat(p.x, p.y);
    $("cursor").textContent = `${lat.toFixed(4)}° N ${lon.toFixed(4)}° E · ${Math.round(p.z)} m`;
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
  draw();
  bindClicks();
})();
