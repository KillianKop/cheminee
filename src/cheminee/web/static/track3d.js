// 3D flight track for MapLibre: a WebGL custom layer drawing smooth lines at the flight altitude.
//
// - The line has a constant width on screen (extruded in the vertex shader), with a dark outline.
// - Climbs in thermals (track points at the same position, rising altitude) are drawn as spirals around the
//   thermal, like a recorded flight, instead of a vertical bar.
// - A translucent curtain hangs from the glides down to the ground.
// - The ceiling of the selected hour is a translucent plane over the computed area, framed by a line: a visual
//   reference to judge the flight altitudes.
// - Positions use the matrix MapLibre draws the terrain with (`modelViewProjectionMatrix`: world pixels in x/y,
//   metres in z, terrain elevation included), so the track and the relief line up exactly. Coordinates are
//   relative to a local origin so that float32 keeps centimetre precision.
"use strict";

const TRACK_LINE_PX = 3.5; // coloured line width
const TRACK_OUTLINE_PX = 5.5; // dark outline under it
const SPIRAL_RADIUS_M = 45;
const SPIRAL_GAIN_PER_TURN_M = 70;
const SPIRAL_POINTS_PER_TURN = 24;
const CURTAIN_ALPHA = 0.15;
const CEILING_COLOR = "#2b6fd6";
const CEILING_ALPHA = 0.1;
const CEILING_LINE_PX = 2;

const LINE_VS = `
attribute vec3 a_pos;
attribute vec3 a_prev;
attribute vec3 a_next;
attribute float a_side;
uniform mat4 u_matrix;
uniform float u_ex;
uniform vec2 u_viewport;
uniform float u_width;
vec4 project(vec3 p) { return u_matrix * vec4(p.xy, p.z * u_ex, 1.0); }
vec2 unit(vec2 v) { float l = length(v); return l > 1e-6 ? v / l : vec2(0.0); }
void main() {
  vec4 c = project(a_pos);
  vec2 s = c.xy / c.w * u_viewport * 0.5;
  vec4 cp = project(a_prev);
  vec4 cn = project(a_next);
  vec2 d = unit(s - cp.xy / cp.w * u_viewport * 0.5) + unit(cn.xy / cn.w * u_viewport * 0.5 - s);
  vec2 dir = length(d) > 1e-6 ? normalize(d) : vec2(1.0, 0.0);
  c.xy += vec2(-dir.y, dir.x) * a_side * u_width / u_viewport * c.w;
  gl_Position = c;
}`;
const LINE_FS = `
precision mediump float;
uniform vec4 u_color;
void main() { gl_FragColor = vec4(u_color.rgb * u_color.a, u_color.a); }`;
const CURTAIN_VS = `
attribute vec3 a_pos;
uniform mat4 u_matrix;
uniform float u_ex;
void main() { gl_Position = u_matrix * vec4(a_pos.xy, a_pos.z * u_ex, 1.0); }`;

function compile(gl, vs, fs) {
  const program = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

// Column-major 4x4 product a * b, in double precision.
function mat4Mul(a, b) {
  const o = new Float64Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
      o[i * 4 + j] = s;
    }
  }
  return o;
}

function hexColor(hex, alpha) {
  const v = parseInt(hex.slice(1), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255, alpha];
}

// Track rows [lon, lat, flight altitude, ground altitude] -> points {x, y (mercator), z, ground, glide}, with
// spirals in place of the climbs.
function trackPoints(track) {
  const pts = [];
  const merc = (lon, lat) => maplibregl.MercatorCoordinate.fromLngLat({ lng: lon, lat });
  const push = (lon, lat, z, ground, glide) => {
    const m = merc(lon, lat);
    pts.push({ x: m.x, y: m.y, z, ground, glide });
  };
  for (let i = 0; i < track.length; i++) {
    const [lon, lat, alt, ground] = track[i];
    const next = track[i + 1];
    const mLat = 111320, mLon = 111320 * Math.cos((lat * Math.PI) / 180);
    const climbing = next && Math.hypot((next[0] - lon) * mLon, (next[1] - lat) * mLat) < 1 && next[2] > alt + 1;
    push(lon, lat, alt, ground, true); // track points carry the curtain, spiral points do not
    if (!climbing) continue;
    const turns = (next[2] - alt) / SPIRAL_GAIN_PER_TURN_M;
    const n = Math.max(SPIRAL_POINTS_PER_TURN, Math.round(turns * SPIRAL_POINTS_PER_TURN));
    for (let k = 1; k < n; k++) {
      const f = k / n, a = 2 * Math.PI * turns * f;
      const r = SPIRAL_RADIUS_M * Math.min(1, 4 * Math.min(f, 1 - f) * turns); // ease in and out of the spiral
      push(lon + (r * Math.cos(a)) / mLon, lat + (r * Math.sin(a)) / mLat, alt + (next[2] - alt) * f, ground, false);
    }
  }
  return pts;
}

class TrackLayer {
  // origin: [lon, lat] near the data (the DEM corner): positions are stored relative to it
  constructor(origin) {
    this.id = "track3d";
    this.type = "custom";
    this.renderingMode = "3d";
    this.routes = [];
    this.ceiling = null; // {bounds, alt}
    this.exaggeration = 1;
    this.meshes = [];
    this.ceilingMesh = null;
    const o = maplibregl.MercatorCoordinate.fromLngLat({ lng: origin[0], lat: origin[1] });
    this.origin = [o.x, o.y];
  }

  onAdd(map, gl) {
    this.map = map;
    this.gl = gl;
    this.line = compile(gl, LINE_VS, LINE_FS);
    this.surface = compile(gl, CURTAIN_VS, LINE_FS);
    this.upload();
    this.uploadCeiling();
  }

  onRemove(map, gl) {
    this.free();
    this.freeCeiling();
    gl.deleteProgram(this.line);
    gl.deleteProgram(this.surface);
  }

  // routes: [{color, track}] of the routes to show
  setRoutes(routes) {
    this.routes = routes;
    if (this.gl) this.upload();
    this.repaint();
  }

  // bounds: {lon_min, lat_min, lon_max, lat_max}; alt in metres, or null to hide the ceiling
  setCeiling(bounds, alt) {
    this.ceiling = alt === null ? null : { bounds, alt };
    if (this.gl) this.uploadCeiling();
    this.repaint();
  }

  setExaggeration(ex) {
    this.exaggeration = ex;
    this.repaint();
  }

  repaint() {
    if (this.map) this.map.triggerRepaint();
  }

  rel(p) {
    return [p.x - this.origin[0], p.y - this.origin[1], p.z];
  }

  buffer(target, data) {
    const gl = this.gl, b = gl.createBuffer();
    gl.bindBuffer(target, b);
    gl.bufferData(target, data, gl.STATIC_DRAW);
    return b;
  }

  // Screen-space line through points [x, y, z] (relative mercator x/y, metres z): 2 vertices per point.
  lineMesh(rel) {
    const n = Math.min(rel.length, 32000); // 16-bit indices
    const data = new Float32Array(n * 2 * 10);
    for (let i = 0; i < n; i++) {
      const prev = rel[Math.max(i - 1, 0)], next = rel[Math.min(i + 1, n - 1)];
      for (let s = 0; s < 2; s++) data.set([...rel[i], ...prev, ...next, s ? 1 : -1], (i * 2 + s) * 10);
    }
    const idx = [];
    for (let i = 0; i < n - 1; i++) idx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
    const gl = this.gl;
    return { buf: this.buffer(gl.ARRAY_BUFFER, data), idx: this.buffer(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx)), count: idx.length };
  }

  // Triangles over vertices [x, y, z].
  surfaceMesh(vertices, idx) {
    const gl = this.gl;
    return {
      buf: this.buffer(gl.ARRAY_BUFFER, new Float32Array(vertices.flat())),
      idx: this.buffer(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx)),
      count: idx.length,
    };
  }

  deleteMesh(m) {
    if (m) for (const b of [m.buf, m.idx]) this.gl.deleteBuffer(b);
  }

  free() {
    for (const m of this.meshes) {
      this.deleteMesh(m.line);
      this.deleteMesh(m.curtain);
    }
    this.meshes = [];
  }

  freeCeiling() {
    if (!this.ceilingMesh) return;
    this.deleteMesh(this.ceilingMesh.plane);
    this.deleteMesh(this.ceilingMesh.frame);
    this.ceilingMesh = null;
  }

  upload() {
    this.free();
    for (const r of this.routes) {
      const pts = trackPoints(r.track).slice(0, 32000);
      if (pts.length < 2) continue;
      const rel = pts.map((p) => this.rel(p));
      const vertices = [], idx = [];
      pts.forEach((p, i) => vertices.push(rel[i], [rel[i][0], rel[i][1], p.ground]));
      for (let i = 0; i < pts.length - 1; i++) {
        if (pts[i].glide && pts[i + 1].glide) idx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
      }
      this.meshes.push({ color: r.color, line: this.lineMesh(rel), curtain: this.surfaceMesh(vertices, idx) });
    }
  }

  uploadCeiling() {
    this.freeCeiling();
    if (!this.ceiling) return;
    const { bounds: b, alt } = this.ceiling;
    const corner = (lon, lat) => {
      const m = maplibregl.MercatorCoordinate.fromLngLat({ lng: lon, lat });
      return this.rel({ x: m.x, y: m.y, z: alt });
    };
    const c = [corner(b.lon_min, b.lat_min), corner(b.lon_max, b.lat_min), corner(b.lon_max, b.lat_max), corner(b.lon_min, b.lat_max)];
    this.ceilingMesh = { plane: this.surfaceMesh(c, [0, 1, 2, 0, 2, 3]), frame: this.lineMesh([...c, c[0], c[1]]) };
  }

  drawSurface(mesh, color) {
    const gl = this.gl, pos = gl.getAttribLocation(this.surface, "a_pos");
    if (!mesh || !mesh.count) return;
    gl.uniform4fv(gl.getUniformLocation(this.surface, "u_color"), color);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.buf);
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 3, gl.FLOAT, false, 12, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.idx);
    gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
    gl.disableVertexAttribArray(pos);
  }

  drawLine(mesh, width, color) {
    const gl = this.gl, loc = (name) => gl.getAttribLocation(this.line, name);
    const attrs = [[loc("a_pos"), 3, 0], [loc("a_prev"), 3, 12], [loc("a_next"), 3, 24], [loc("a_side"), 1, 36]];
    gl.uniform1f(gl.getUniformLocation(this.line, "u_width"), width * (window.devicePixelRatio || 1));
    gl.uniform4fv(gl.getUniformLocation(this.line, "u_color"), color);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.buf);
    for (const [l, size, offset] of attrs) {
      gl.enableVertexAttribArray(l);
      gl.vertexAttribPointer(l, size, gl.FLOAT, false, 40, offset);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.idx);
    gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
    for (const [l] of attrs) gl.disableVertexAttribArray(l);
  }

  render(gl, _matrix, options) {
    if ((!this.meshes.length && !this.ceilingMesh) || !options || !options.modelViewProjectionMatrix) return;
    // world pixels = mercator * worldSize (tile size 512): local origin + scale, then MapLibre's terrain matrix
    const ws = 512 * Math.pow(2, this.map.getZoom());
    const [ox, oy] = this.origin;
    const local = [ws, 0, 0, 0, 0, ws, 0, 0, 0, 0, 1, 0, ox * ws, oy * ws, 0, 1];
    const matrix = new Float32Array(mat4Mul(options.modelViewProjectionMatrix, local));

    // translucent surfaces first, without writing depth: the lines behind them stay visible
    gl.useProgram(this.surface);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.surface, "u_matrix"), false, matrix);
    gl.uniform1f(gl.getUniformLocation(this.surface, "u_ex"), this.exaggeration);
    gl.depthMask(false);
    if (this.ceilingMesh) this.drawSurface(this.ceilingMesh.plane, hexColor(CEILING_COLOR, CEILING_ALPHA));
    for (const m of this.meshes) this.drawSurface(m.curtain, hexColor(m.color, CURTAIN_ALPHA));
    gl.depthMask(true);

    gl.useProgram(this.line);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.line, "u_matrix"), false, matrix);
    gl.uniform1f(gl.getUniformLocation(this.line, "u_ex"), this.exaggeration);
    gl.uniform2fv(gl.getUniformLocation(this.line, "u_viewport"), [gl.drawingBufferWidth, gl.drawingBufferHeight]);
    if (this.ceilingMesh) this.drawLine(this.ceilingMesh.frame, CEILING_LINE_PX, hexColor(CEILING_COLOR, 0.8));
    for (const m of this.meshes) this.drawLine(m.line, TRACK_OUTLINE_PX, [0.07, 0.08, 0.1, 0.5]);
    for (const m of this.meshes) this.drawLine(m.line, TRACK_LINE_PX, hexColor(m.color, 1));
  }
}
