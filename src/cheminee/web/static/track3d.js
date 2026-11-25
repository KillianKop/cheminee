// 3D flight track for MapLibre: a WebGL custom layer drawing smooth lines at the flight altitude.
//
// - The line has a constant width on screen (extruded in the vertex shader), with a dark outline.
// - Climbs in thermals (track points at the same position, rising altitude) are drawn as spirals around the
//   thermal, like a recorded flight, instead of a vertical bar.
// - A translucent curtain hangs from the glides down to the ground.
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
  constructor() {
    this.id = "track3d";
    this.type = "custom";
    this.renderingMode = "3d";
    this.routes = [];
    this.exaggeration = 1.5;
    this.meshes = [];
  }

  onAdd(map, gl) {
    this.map = map;
    this.gl = gl;
    this.line = compile(gl, LINE_VS, LINE_FS);
    this.curtain = compile(gl, CURTAIN_VS, LINE_FS);
    this.upload();
  }

  onRemove(map, gl) {
    this.free();
    gl.deleteProgram(this.line);
    gl.deleteProgram(this.curtain);
  }

  // routes: [{color, track}] of the routes to show
  setRoutes(routes) {
    this.routes = routes;
    if (this.gl) this.upload();
    if (this.map) this.map.triggerRepaint();
  }

  setExaggeration(ex) {
    this.exaggeration = ex;
    if (this.map) this.map.triggerRepaint();
  }

  free() {
    for (const m of this.meshes) for (const b of [m.lineBuf, m.lineIdx, m.curtainBuf, m.curtainIdx]) this.gl.deleteBuffer(b);
    this.meshes = [];
  }

  upload() {
    const gl = this.gl;
    this.free();
    const all = this.routes.map((r) => ({ color: r.color, pts: trackPoints(r.track) })).filter((r) => r.pts.length > 1);
    if (!all.length) return;
    const o = all[0].pts[0];
    this.origin = [o.x, o.y];
    for (const { color, pts } of all) {
      const n = Math.min(pts.length, 32000); // 2 vertices per point, 16-bit indices
      const rel = pts.slice(0, n).map((p) => [p.x - o.x, p.y - o.y, p.z]);
      const line = new Float32Array(n * 2 * 10);
      for (let i = 0; i < n; i++) {
        const prev = rel[Math.max(i - 1, 0)], next = rel[Math.min(i + 1, n - 1)];
        for (let s = 0; s < 2; s++) {
          line.set([...rel[i], ...prev, ...next, s ? 1 : -1], (i * 2 + s) * 10);
        }
      }
      const lineIdx = [];
      for (let i = 0; i < n - 1; i++) lineIdx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
      const curtain = new Float32Array(n * 2 * 3);
      for (let i = 0; i < n; i++) {
        curtain.set(rel[i], i * 6);
        curtain.set([rel[i][0], rel[i][1], pts[i].ground], i * 6 + 3);
      }
      const curtainIdx = [];
      for (let i = 0; i < n - 1; i++) {
        if (pts[i].glide && pts[i + 1].glide) curtainIdx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
      }
      const buffer = (target, data) => {
        const b = gl.createBuffer();
        gl.bindBuffer(target, b);
        gl.bufferData(target, data, gl.STATIC_DRAW);
        return b;
      };
      this.meshes.push({
        color,
        lineBuf: buffer(gl.ARRAY_BUFFER, line),
        lineIdx: buffer(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(lineIdx)),
        lineCount: lineIdx.length,
        curtainBuf: buffer(gl.ARRAY_BUFFER, curtain),
        curtainIdx: buffer(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(curtainIdx)),
        curtainCount: curtainIdx.length,
      });
    }
  }

  render(gl, _matrix, options) {
    if (!this.meshes.length || !options || !options.modelViewProjectionMatrix) return;
    // world pixels = mercator * worldSize (tile size 512): local origin + scale, then MapLibre's terrain matrix
    const ws = 512 * Math.pow(2, this.map.getZoom());
    const [ox, oy] = this.origin;
    const local = [ws, 0, 0, 0, 0, ws, 0, 0, 0, 0, 1, 0, ox * ws, oy * ws, 0, 1];
    const matrix = new Float32Array(mat4Mul(options.modelViewProjectionMatrix, local));
    const viewport = [gl.drawingBufferWidth, gl.drawingBufferHeight];
    const ratio = window.devicePixelRatio || 1;
    const ex = this.exaggeration;

    // curtains first, translucent, without writing depth so the lines behind stay visible
    gl.useProgram(this.curtain);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.curtain, "u_matrix"), false, matrix);
    gl.uniform1f(gl.getUniformLocation(this.curtain, "u_ex"), ex);
    const cPos = gl.getAttribLocation(this.curtain, "a_pos");
    gl.depthMask(false);
    for (const m of this.meshes) {
      if (!m.curtainCount) continue;
      gl.uniform4fv(gl.getUniformLocation(this.curtain, "u_color"), hexColor(m.color, CURTAIN_ALPHA));
      gl.bindBuffer(gl.ARRAY_BUFFER, m.curtainBuf);
      gl.enableVertexAttribArray(cPos);
      gl.vertexAttribPointer(cPos, 3, gl.FLOAT, false, 12, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.curtainIdx);
      gl.drawElements(gl.TRIANGLES, m.curtainCount, gl.UNSIGNED_SHORT, 0);
    }
    gl.disableVertexAttribArray(cPos);
    gl.depthMask(true);

    gl.useProgram(this.line);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.line, "u_matrix"), false, matrix);
    gl.uniform1f(gl.getUniformLocation(this.line, "u_ex"), ex);
    gl.uniform2fv(gl.getUniformLocation(this.line, "u_viewport"), viewport);
    const loc = (name) => gl.getAttribLocation(this.line, name);
    const attrs = [[loc("a_pos"), 3, 0], [loc("a_prev"), 3, 12], [loc("a_next"), 3, 24], [loc("a_side"), 1, 36]];
    for (const [width, color] of [[TRACK_OUTLINE_PX, null], [TRACK_LINE_PX, "route"]]) {
      gl.uniform1f(gl.getUniformLocation(this.line, "u_width"), width * ratio);
      for (const m of this.meshes) {
        gl.uniform4fv(gl.getUniformLocation(this.line, "u_color"), color ? hexColor(m.color, 1) : [0.07, 0.08, 0.1, 0.5]);
        gl.bindBuffer(gl.ARRAY_BUFFER, m.lineBuf);
        for (const [l, size, offset] of attrs) {
          gl.enableVertexAttribArray(l);
          gl.vertexAttribPointer(l, size, gl.FLOAT, false, 40, offset);
        }
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.lineIdx);
        gl.drawElements(gl.TRIANGLES, m.lineCount, gl.UNSIGNED_SHORT, 0);
      }
    }
    for (const [l] of attrs) gl.disableVertexAttribArray(l);
  }
}
