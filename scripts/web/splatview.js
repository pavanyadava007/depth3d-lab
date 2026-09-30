// Minimal 3D Gaussian Splat viewer for .splat files (32 bytes per Gaussian: xyz f32, scale f32, rgba u8, quat u8).
// WebGL2 only, no WebAssembly, no external libraries. The depth sort worker, texture packing and shaders are
// adapted from Kevin Kwok's "splat" viewer (https://github.com/antimatter15/splat, MIT License, Copyright (c) 2023
// Kevin Kwok). The camera, controls and module wrapper are ours. Coordinate convention of the file: x right, y down,
// z forward (OpenCV), as written by scripts/build_space.py.

function createWorker(self) {
  let buffer;
  let vertexCount = 0;
  let viewProj;
  let lastProj = [];
  let depthIndex = new Uint32Array();
  let lastVertexCount = 0;
  const _floatView = new Float32Array(1);
  const _int32View = new Int32Array(_floatView.buffer);
  function floatToHalf(float) {
    _floatView[0] = float;
    const f = _int32View[0];
    const sign = (f >> 31) & 0x0001;
    const exp = (f >> 23) & 0x00ff;
    let frac = f & 0x007fffff;
    let newExp;
    if (exp == 0) { newExp = 0; }
    else if (exp < 113) { newExp = 0; frac |= 0x00800000; frac = frac >> (113 - exp); if (frac & 0x01000000) { newExp = 1; frac = 0; } }
    else if (exp < 142) { newExp = exp - 112; }
    else { newExp = 31; frac = 0; }
    return (sign << 15) | (newExp << 10) | (frac >> 13);
  }
  function packHalf2x16(x, y) { return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0; }
  function generateTexture() {
    if (!buffer) return;
    const f_buffer = new Float32Array(buffer);
    const u_buffer = new Uint8Array(buffer);
    const texwidth = 1024 * 2;
    const texheight = Math.ceil((2 * vertexCount) / texwidth);
    const texdata = new Uint32Array(texwidth * texheight * 4);
    const texdata_c = new Uint8Array(texdata.buffer);
    const texdata_f = new Float32Array(texdata.buffer);
    for (let i = 0; i < vertexCount; i++) {
      texdata_f[8 * i + 0] = f_buffer[8 * i + 0];
      texdata_f[8 * i + 1] = f_buffer[8 * i + 1];
      texdata_f[8 * i + 2] = f_buffer[8 * i + 2];
      texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
      texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
      texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
      texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3];
      const scale = [f_buffer[8 * i + 3 + 0], f_buffer[8 * i + 3 + 1], f_buffer[8 * i + 3 + 2]];
      const rot = [(u_buffer[32 * i + 28 + 0] - 128) / 128, (u_buffer[32 * i + 28 + 1] - 128) / 128,
                   (u_buffer[32 * i + 28 + 2] - 128) / 128, (u_buffer[32 * i + 28 + 3] - 128) / 128];
      const M = [
        1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]), 2.0 * (rot[1] * rot[2] + rot[0] * rot[3]), 2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),
        2.0 * (rot[1] * rot[2] - rot[0] * rot[3]), 1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]), 2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),
        2.0 * (rot[1] * rot[3] + rot[0] * rot[2]), 2.0 * (rot[2] * rot[3] - rot[0] * rot[1]), 1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
      ].map((k, j) => k * scale[Math.floor(j / 3)]);
      const sigma = [
        M[0] * M[0] + M[3] * M[3] + M[6] * M[6], M[0] * M[1] + M[3] * M[4] + M[6] * M[7], M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
        M[1] * M[1] + M[4] * M[4] + M[7] * M[7], M[1] * M[2] + M[4] * M[5] + M[7] * M[8], M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
      ];
      texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
      texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
      texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
    }
    self.postMessage({ texdata, texwidth, texheight }, [texdata.buffer]);
  }
  function runSort(viewProj) {
    if (!buffer) return;
    const f_buffer = new Float32Array(buffer);
    if (lastVertexCount == vertexCount) {
      const dot = lastProj[2] * viewProj[2] + lastProj[6] * viewProj[6] + lastProj[10] * viewProj[10];
      if (Math.abs(dot - 1) < 0.01) return;
    } else {
      generateTexture();
      lastVertexCount = vertexCount;
    }
    let maxDepth = -Infinity;
    let minDepth = Infinity;
    const sizeList = new Int32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) {
      const depth = ((viewProj[2] * f_buffer[8 * i + 0] + viewProj[6] * f_buffer[8 * i + 1] + viewProj[10] * f_buffer[8 * i + 2]) * 4096) | 0;
      sizeList[i] = depth;
      if (depth > maxDepth) maxDepth = depth;
      if (depth < minDepth) minDepth = depth;
    }
    const depthInv = (256 * 256 - 1) / (maxDepth - minDepth);
    const counts0 = new Uint32Array(256 * 256);
    for (let i = 0; i < vertexCount; i++) { sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0; counts0[sizeList[i]]++; }
    const starts0 = new Uint32Array(256 * 256);
    for (let i = 1; i < 256 * 256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];
    depthIndex = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) depthIndex[starts0[sizeList[i]]++] = i;
    lastProj = viewProj;
    self.postMessage({ depthIndex, viewProj, vertexCount }, [depthIndex.buffer]);
  }
  let sortRunning;
  const throttledSort = () => {
    if (!sortRunning) {
      sortRunning = true;
      const lastView = viewProj;
      runSort(lastView);
      setTimeout(() => { sortRunning = false; if (lastView !== viewProj) throttledSort(); }, 0);
    }
  };
  self.onmessage = (e) => {
    if (e.data.buffer) { buffer = e.data.buffer; vertexCount = e.data.vertexCount; }
    else if (e.data.view) { viewProj = e.data.view; throttledSort(); }
  };
}

const vertexShaderSource = `#version 300 es
precision highp float;
precision highp int;
uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;
in vec2 position;
in int index;
out vec4 vColor;
out vec2 vPosition;
void main () {
  uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
  vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
  vec4 pos2d = projection * cam;
  float clip = 1.2 * pos2d.w;
  if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
  vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
  mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);
  mat3 J = mat3(focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z), 0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z), 0., 0., 0.);
  mat3 T = transpose(mat3(view)) * J;
  mat3 cov2d = transpose(T) * Vrk * T;
  float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
  float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
  float lambda1 = mid + radius, lambda2 = mid - radius;
  if (lambda2 < 0.0) return;
  vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
  vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
  vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);
  vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
  vPosition = position;
  vec2 vCenter = vec2(pos2d) / pos2d.w;
  gl_Position = vec4(vCenter + position.x * majorAxis / viewport + position.y * minorAxis / viewport, 0.0, 1.0);
}`;

const fragmentShaderSource = `#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vPosition;
out vec4 fragColor;
void main () {
  float A = -dot(vPosition, vPosition);
  if (A < -4.0) discard;
  float B = exp(A) * vColor.a;
  fragColor = vec4(B * vColor.rgb, B);
}`;

function projectionMatrix(fx, fy, width, height) {
  const znear = 0.2, zfar = 500;
  return [(2 * fx) / width, 0, 0, 0, 0, -(2 * fy) / height, 0, 0, 0, 0, zfar / (zfar - znear), 1, 0, 0, -(zfar * znear) / (zfar - znear), 0];
}
function multiply4(a, b) {
  const o = new Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o[4 * c + r] = b[4 * c] * a[r] + b[4 * c + 1] * a[4 + r] + b[4 * c + 2] * a[8 + r] + b[4 * c + 3] * a[12 + r];
  return o;
}
const norm = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
// world-to-camera matrix (column-major) for an OpenCV camera (x right, y down, z forward) in a y-down world
function lookAt(eye, target) {
  const f = norm([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  const r = norm(cross([0, 1, 0], f));
  const d = cross(f, r);
  return [r[0], d[0], f[0], 0, r[1], d[1], f[1], 0, r[2], d[2], f[2], 0, -dot(r, eye), -dot(d, eye), -dot(f, eye), 1];
}

export async function mountSplatViewer(el, url, opts = {}) {
  const canvas = document.createElement('canvas');
  canvas.style.width = '100%'; canvas.style.height = '100%'; canvas.style.display = 'block';
  el.appendChild(canvas);
  const status = document.createElement('div');
  status.style.cssText = 'position:absolute;left:8px;top:8px;color:#ddd;font:12px sans-serif;background:rgba(0,0,0,.5);padding:2px 6px;border-radius:4px';
  el.style.position = 'relative'; el.appendChild(status);
  const gl = canvas.getContext('webgl2', { antialias: false });
  if (!gl) { status.textContent = 'WebGL2 is not available in this browser.'; return; }
  const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const program = gl.createProgram();
  gl.attachShader(program, mk(gl.VERTEX_SHADER, vertexShaderSource));
  gl.attachShader(program, mk(gl.FRAGMENT_SHADER, fragmentShaderSource));
  gl.linkProgram(program); gl.useProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
  gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
  const u_projection = gl.getUniformLocation(program, 'projection');
  const u_viewport = gl.getUniformLocation(program, 'viewport');
  const u_focal = gl.getUniformLocation(program, 'focal');
  const u_view = gl.getUniformLocation(program, 'view');
  gl.uniform1i(gl.getUniformLocation(program, 'u_texture'), 0);
  const vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
  const a_position = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(a_position);
  gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  const indexBuffer = gl.createBuffer();
  const a_index = gl.getAttribLocation(program, 'index');
  gl.enableVertexAttribArray(a_index);
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
  gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
  gl.vertexAttribDivisor(a_index, 1);

  // orbit camera
  const target = opts.target ? opts.target.slice() : [0, 0, 20];
  let dist = opts.distance || 30, yaw = opts.yaw || 0, pitch = opts.pitch || 0.12;
  let proj, W = 1, H = 1;
  const resize = () => {
    W = Math.max(1, el.clientWidth); H = Math.max(1, el.clientHeight || 500);
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    const f = 0.5 * canvas.height / Math.tan((opts.fovDeg || 60) * Math.PI / 360);
    proj = projectionMatrix(f, f, canvas.width, canvas.height);
    gl.uniform2fv(u_focal, new Float32Array([f, f]));
    gl.uniform2fv(u_viewport, new Float32Array([canvas.width, canvas.height]));
    gl.uniformMatrix4fv(u_projection, false, proj);
  };
  window.addEventListener('resize', resize); resize();
  const eye = () => [target[0] + dist * Math.cos(pitch) * Math.sin(yaw), target[1] - dist * Math.sin(pitch), target[2] - dist * Math.cos(pitch) * Math.cos(yaw)];
  let drag = null;
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, b: e.button }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
    if (drag.b === 2 || e.shiftKey) {
      const v = lookAt(eye(), target); const s = dist * 0.0015;
      target[0] -= (v[0] * dx + v[1] * dy) * s; target[1] -= (v[4] * dx + v[5] * dy) * s; target[2] -= (v[8] * dx + v[9] * dy) * s;
    } else { yaw -= dx * 0.005; pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.005)); }
  });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); dist *= Math.exp(e.deltaY * 0.001); dist = Math.max(1, Math.min(300, dist)); }, { passive: false });

  const worker = new Worker(URL.createObjectURL(new Blob(['(', createWorker.toString(), ')(self)'], { type: 'application/javascript' })));
  let vertexCount = 0, indexReady = false;
  worker.onmessage = (e) => {
    if (e.data.texdata) {
      const { texdata, texwidth, texheight } = e.data;
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, texwidth, texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texture);
    } else if (e.data.depthIndex) {
      gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, e.data.depthIndex, gl.DYNAMIC_DRAW);
      vertexCount = e.data.vertexCount; indexReady = true;
    }
  };
  status.textContent = 'loading splats ...';
  const resp = await fetch(url);
  if (!resp.ok) { status.textContent = 'could not load ' + url + ' (' + resp.status + ')'; return; }
  const buf = await resp.arrayBuffer();
  const n = Math.floor(buf.byteLength / 32);
  worker.postMessage({ buffer: buf, vertexCount: n }, [buf]);
  status.textContent = n.toLocaleString() + ' Gaussians. Drag to orbit, right-drag or shift-drag to pan, wheel to zoom.';
  el.__splatState = () => ({ vertexCount, indexReady });
  (function frame() {
    const view = lookAt(eye(), target);
    worker.postMessage({ view: multiply4(proj, view) });
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (indexReady && vertexCount > 0) {
      gl.uniformMatrix4fv(u_view, false, view);
      gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
    }
    requestAnimationFrame(frame);
  })();
}
