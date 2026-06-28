#!/usr/bin/env python3
from __future__ import annotations

"""Build a self-contained click-to-similarity HTML visualizer for UMAP XYZ images.

Usage
-----
.. code-block:: python

    from decomposition_umap.interactive import (
        build_interactive_umap,
        build_interactive_umap_from_embed_map,
        open_interactive_umap,
    )

    # From a saved (3, H, W) or (H, W, 3) .npy file
    build_interactive_umap("umap_xyz.npy", output_html="visualizer.html")

    # Directly from an embed_map list returned by decompose_and_embed()
    build_interactive_umap_from_embed_map(embed_map, output_html="visualizer.html")

    # Same but auto-opens in browser
    open_interactive_umap(embed_map)
"""

import base64
import html
import io
import json
import os
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image


def _as_hwc_xyz(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D XYZ array, got shape {arr.shape}")
    if arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr[:3], 0, -1)
    elif arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
    else:
        raise ValueError(
            "expected shape (3,H,W), (4,H,W), (H,W,3), or (H,W,4); "
            f"got {arr.shape}"
        )
    return arr


def _robust_rgb(xyz: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(xyz).all(axis=-1)
    rgb = np.zeros((*xyz.shape[:2], 4), dtype=np.uint8)
    rgb[..., 3] = np.where(valid, 255, 0).astype(np.uint8)

    stats = {
        "shape": list(xyz.shape),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
        "channels": [],
    }
    if not valid.any():
        return rgb, stats

    for channel in range(3):
        values = xyz[..., channel][valid]
        lo, hi = np.nanpercentile(values, [lo_pct, hi_pct])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            rgb[..., channel] = 127
            lo = hi = float(values[0]) if values.size else 0.0
        else:
            scaled = (xyz[..., channel] - lo) / (hi - lo)
            scaled = np.clip(scaled, 0.0, 1.0)
            rgb[..., channel] = np.where(valid, np.round(scaled * 255.0), 0).astype(np.uint8)
        stats["channels"].append(
            {
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "lo_pct": float(lo),
                "hi_pct": float(hi),
            }
        )
    return rgb, stats


def _png_data_url(rgba: np.ndarray) -> str:
    image = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _float32_base64(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr.astype("<f4", copy=False))
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def _build_html_content(
    input_path: Path,
    xyz: np.ndarray,
    lo_pct: float,
    hi_pct: float,
) -> str:
    rgba, stats = _robust_rgb(xyz, lo_pct=lo_pct, hi_pct=hi_pct)
    data_url = _png_data_url(rgba)
    xyz_b64 = _float32_base64(xyz)

    title = f"{input_path.name} visualizer"
    stats_json = json.dumps(stats, separators=(",", ":"))
    source_text = html.escape(str(input_path))
    title_text = html.escape(title)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_text}</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #111317;
    --panel: #1d222a;
    --text: #e9edf4;
    --muted: #9aa6b8;
    --line: #374151;
    --accent: #67d4ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    min-height: 100vh;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }}
  h1 {{
    margin: 0;
    font-size: 16px;
    font-weight: 650;
  }}
  .meta {{
    color: var(--muted);
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .toolbar {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .hint {{
    color: var(--muted);
    font-size: 12px;
  }}
  button {{
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #252c36;
    color: var(--text);
    padding: 6px 10px;
    cursor: pointer;
  }}
  button:hover {{ border-color: var(--accent); }}
  label.control {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 12px;
    white-space: nowrap;
  }}
  input[type="range"] {{
    width: 130px;
    accent-color: var(--accent);
  }}
  select {{
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #252c36;
    color: var(--text);
    padding: 5px 8px;
  }}
  main {{
    height: calc(100vh - 58px);
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
  }}
  .stage {{
    position: relative;
    overflow: hidden;
    background:
      linear-gradient(45deg, #0d0f13 25%, transparent 25%),
      linear-gradient(-45deg, #0d0f13 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #0d0f13 75%),
      linear-gradient(-45deg, transparent 75%, #0d0f13 75%);
    background-size: 24px 24px;
    background-position: 0 0, 0 12px, 12px -12px, -12px 0;
  }}
  canvas {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    image-rendering: pixelated;
    cursor: grab;
  }}
  canvas.dragging {{ cursor: grabbing; }}
  aside {{
    border-left: 1px solid var(--line);
    background: var(--panel);
    padding: 14px;
    overflow: auto;
  }}
  dl {{ margin: 0; }}
  dt {{ color: var(--muted); font-size: 12px; margin-top: 12px; }}
  dd {{ margin: 3px 0 0; overflow-wrap: anywhere; }}
  .channel {{
    display: grid;
    grid-template-columns: 20px 1fr;
    gap: 8px;
    align-items: center;
    margin-top: 10px;
  }}
  .swatch {{ width: 16px; height: 16px; border-radius: 4px; }}
  .r {{ background: #ff5b6b; }}
  .g {{ background: #34d399; }}
  .b {{ background: #60a5fa; }}
  @media (max-width: 860px) {{
    main {{ grid-template-columns: 1fr; grid-template-rows: minmax(380px, 1fr) auto; }}
    aside {{ border-left: 0; border-top: 1px solid var(--line); }}
  }}
</style>
</head>
<body>
<header>
  <div>
    <h1>{title_text}</h1>
    <div class="meta">{source_text}</div>
  </div>
  <div class="toolbar">
    <button id="fit" type="button">Fit</button>
    <button id="one" type="button">1:1</button>
    <button id="rgb" type="button">RGB</button>
    <button id="flipY" type="button">Flip Display Y</button>
    <button id="saveNpy" type="button">Save NPY</button>
    <label class="control">Gamma <input id="gamma" type="range" min="0.05" max="8" step="0.05" value="0.65"></label>
    <label class="control">Map <select id="colormap">
      <option value="turbo">Turbo</option>
      <option value="viridis">Viridis</option>
      <option value="plasma">Plasma</option>
      <option value="inferno">Inferno</option>
      <option value="magma">Magma</option>
      <option value="cividis">Cividis</option>
      <option value="cubehelix">Cubehelix</option>
      <option value="icefire">Ice/Fire</option>
      <option value="fire">Fire</option>
      <option value="cyanmagenta">Cyan/Magenta</option>
      <option value="redblue">Red/Blue</option>
      <option value="gray">Gray</option>
    </select></label>
  </div>
</header>
<main>
  <section class="stage"><canvas id="view"></canvas></section>
  <aside>
    <dl>
      <dt>Array</dt><dd id="shape"></dd>
      <dt>Valid pixels</dt><dd id="valid"></dd>
      <dt>Cursor</dt><dd id="cursor">-</dd>
      <dt>Selected</dt><dd id="selected">click a valid pixel</dd>
      <dt>View</dt><dd id="mode">RGB projection</dd>
      <dt>Similarity scale</dt><dd id="window">-</dd>
      <dt>Save orientation</dt><dd>original array shape/orientation</dd>
      <dt>Zoom</dt><dd id="zoom">-</dd>
    </dl>
    <p class="hint">Click a valid pixel to draw a cosine-similarity map against the selected XYZ vector.</p>
    <div class="channel"><span class="swatch r"></span><span id="ch0"></span></div>
    <div class="channel"><span class="swatch g"></span><span id="ch1"></span></div>
    <div class="channel"><span class="swatch b"></span><span id="ch2"></span></div>
  </aside>
</main>
<script>
const DATA_URL = {json.dumps(data_url)};
const XYZ_B64 = {json.dumps(xyz_b64)};
const STATS = {stats_json};
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const img = new Image();
let xyz = null;
let simCanvas = null;
let simValues = null;
let simMin = 0;
let simMax = 1;
let selectedPx = null;
let selectedPy = null;
let mode = "rgb";
let flipY = false;
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let dragging = false;
let moved = false;
let downX = 0;
let downY = 0;
let lastX = 0;
let lastY = 0;

function decodeFloat32(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}

function finite3(index) {{
  return Number.isFinite(xyz[index]) && Number.isFinite(xyz[index + 1]) && Number.isFinite(xyz[index + 2]);
}}

function fmt(v) {{
  return Number.isFinite(v) ? v.toFixed(4) : String(v);
}}

function lerp(a, b, t) {{
  return a + (b - a) * t;
}}

function lerpColor(a, b, t) {{
  return [
    Math.round(lerp(a[0], b[0], t)),
    Math.round(lerp(a[1], b[1], t)),
    Math.round(lerp(a[2], b[2], t))
  ];
}}

function rampColor(stops, t) {{
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {{
    const left = stops[i];
    const right = stops[i + 1];
    if (t <= right[0]) {{
      const local = (t - left[0]) / Math.max(1e-12, right[0] - left[0]);
      return lerpColor(left[1], right[1], local);
    }}
  }}
  return stops[stops.length - 1][1];
}}

function resize() {{
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  draw();
}}

function fit() {{
  if (!img.width || !img.height) return;
  scale = Math.min(canvas.width / img.width, canvas.height / img.height);
  offsetX = (canvas.width - img.width * scale) / 2;
  offsetY = (canvas.height - img.height * scale) / 2;
  draw();
}}

function oneToOne() {{
  const dpr = window.devicePixelRatio || 1;
  scale = dpr;
  offsetX = (canvas.width - img.width * scale) / 2;
  offsetY = (canvas.height - img.height * scale) / 2;
  draw();
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  const source = mode === "sim" && simCanvas ? simCanvas : img;
  if (source && (source.complete || source.width)) {{
    ctx.save();
    if (flipY) {{
      ctx.translate(0, canvas.height);
      ctx.scale(1, -1);
      ctx.drawImage(source, offsetX, canvas.height - offsetY - img.height * scale, img.width * scale, img.height * scale);
    }} else {{
      ctx.drawImage(source, offsetX, offsetY, img.width * scale, img.height * scale);
    }}
    ctx.restore();
  }}
  document.getElementById("zoom").textContent = `${{(scale / (window.devicePixelRatio || 1) * 100).toFixed(1)}}%`;
}}

function canvasToPixel(clientX, clientY) {{
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const x = (clientX - rect.left) * dpr;
  const y = (clientY - rect.top) * dpr;
  return {{
    px: Math.floor((x - offsetX) / scale),
    py: Math.floor(((flipY ? canvas.height - y : y) - offsetY) / scale)
  }};
}}

function turboColor(t) {{
  t = Math.max(0, Math.min(1, t));
  const r = 34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05))));
  const g = 23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (1073.77 + t * 707.56))));
  const b = 27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66))));
  return [
    Math.max(0, Math.min(255, Math.round(r))),
    Math.max(0, Math.min(255, Math.round(g))),
    Math.max(0, Math.min(255, Math.round(b)))
  ];
}}

function colorFor(t) {{
  const map = document.getElementById("colormap").value;
  if (map === "turbo") return turboColor(t);
  if (map === "viridis") return rampColor([
    [0.00, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.50, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.00, [253, 231, 37]]
  ], t);
  if (map === "plasma") return rampColor([
    [0.00, [13, 8, 135]],
    [0.25, [126, 3, 168]],
    [0.50, [204, 71, 120]],
    [0.75, [248, 149, 64]],
    [1.00, [240, 249, 33]]
  ], t);
  if (map === "inferno") return rampColor([
    [0.00, [0, 0, 4]],
    [0.25, [87, 15, 109]],
    [0.50, [187, 55, 84]],
    [0.75, [249, 142, 8]],
    [1.00, [252, 255, 164]]
  ], t);
  if (map === "magma") return rampColor([
    [0.00, [0, 0, 4]],
    [0.25, [80, 18, 123]],
    [0.50, [182, 54, 121]],
    [0.75, [251, 136, 97]],
    [1.00, [252, 253, 191]]
  ], t);
  if (map === "cividis") return rampColor([
    [0.00, [0, 34, 77]],
    [0.25, [39, 75, 108]],
    [0.50, [104, 113, 111]],
    [0.75, [177, 155, 84]],
    [1.00, [253, 231, 55]]
  ], t);
  if (map === "cubehelix") return rampColor([
    [0.00, [0, 0, 0]],
    [0.20, [29, 43, 80]],
    [0.40, [27, 102, 91]],
    [0.60, [126, 121, 70]],
    [0.80, [199, 148, 194]],
    [1.00, [255, 255, 255]]
  ], t);
  if (map === "icefire") return rampColor([
    [0.00, [0, 18, 97]],
    [0.25, [0, 157, 196]],
    [0.48, [236, 248, 255]],
    [0.52, [255, 243, 224]],
    [0.75, [226, 84, 35]],
    [1.00, [90, 0, 0]]
  ], t);
  if (map === "fire") return rampColor([
    [0.00, [0, 0, 0]],
    [0.25, [92, 0, 80]],
    [0.50, [210, 40, 40]],
    [0.75, [255, 169, 36]],
    [1.00, [255, 255, 210]]
  ], t);
  if (map === "cyanmagenta") return rampColor([
    [0.00, [0, 0, 0]],
    [0.25, [0, 178, 210]],
    [0.50, [235, 235, 235]],
    [0.75, [228, 44, 180]],
    [1.00, [255, 235, 255]]
  ], t);
  if (map === "redblue") return rampColor([
    [0.00, [49, 54, 149]],
    [0.45, [116, 173, 209]],
    [0.50, [247, 247, 247]],
    [0.55, [244, 109, 67]],
    [1.00, [165, 0, 38]]
  ], t);
  const g = Math.round(t * 255);
  return [g, g, g];
}}

function renderSimilarity() {{
  if (!simValues) return;
  const gamma = Number(document.getElementById("gamma").value);
  const lo = simMin;
  const hi = simMax;
  const out = new ImageData(img.width, img.height);
  for (let i = 0, o = 0; i < simValues.length; i++, o += 4) {{
    const sim = simValues[i];
    if (!Number.isFinite(sim)) {{
      out.data[o + 3] = 0;
      continue;
    }}
    const t = Math.max(0, Math.min(1, (sim - lo) / Math.max(1e-12, hi - lo)));
    const color = colorFor(Math.pow(t, gamma));
    out.data[o] = color[0];
    out.data[o + 1] = color[1];
    out.data[o + 2] = color[2];
    out.data[o + 3] = 255;
  }}
  simCanvas = document.createElement("canvas");
  simCanvas.width = img.width;
  simCanvas.height = img.height;
  simCanvas.getContext("2d").putImageData(out, 0, 0);
  document.getElementById("window").textContent = `${{fmt(lo)}} to ${{fmt(hi)}}, gamma=${{fmt(gamma)}}`;
  if (mode === "sim") draw();
}}

function npyFloat32Blob(values, shape) {{
  const magic = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]);
  const version = new Uint8Array([1, 0]);
  let shapeText = shape.join(", ");
  if (shape.length === 1) shapeText += ",";
  let header = `{{'descr': '<f4', 'fortran_order': False, 'shape': (${{shapeText}}), }}`;
  const encoder = new TextEncoder();
  let headerLen = encoder.encode(header + "\\n").length;
  const preambleLen = magic.length + version.length + 2;
  const padLen = (16 - ((preambleLen + headerLen) % 16)) % 16;
  header = header + " ".repeat(padLen) + "\\n";
  const headerBytes = encoder.encode(header);
  const headerSize = new Uint8Array(2);
  headerSize[0] = headerBytes.length & 0xff;
  headerSize[1] = (headerBytes.length >> 8) & 0xff;
  const dataBytes = new Uint8Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
  return new Blob([magic, version, headerSize, headerBytes, dataBytes], {{ type: "application/octet-stream" }});
}}

function downloadBlob(blob, filename) {{
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}}

function saveDisplayedNpy() {{
  if (mode !== "sim" || !simValues) {{
    document.getElementById("window").textContent = "click a pixel first, then save the similarity map";
    return;
  }}
  const blob = npyFloat32Blob(simValues, [img.height, img.width]);
  const filename = `similarity_x${{selectedPx}}_y${{selectedPy}}.npy`;
  downloadBlob(blob, filename);
}}

function computeSimilarity(px, py) {{
  if (!xyz || px < 0 || py < 0 || px >= img.width || py >= img.height) return;
  const seedIndex = (py * img.width + px) * 3;
  if (!finite3(seedIndex)) return;

  const sx = xyz[seedIndex];
  const sy = xyz[seedIndex + 1];
  const sz = xyz[seedIndex + 2];
  const seedNorm = Math.hypot(sx, sy, sz);
  if (!Number.isFinite(seedNorm) || seedNorm <= 1e-12) return;

  simValues = new Float32Array(img.width * img.height);
  simValues.fill(Number.NaN);
  let minSim = Infinity;
  let maxSim = -Infinity;
  for (let y = 0; y < img.height; y++) {{
    for (let x = 0; x < img.width; x++) {{
      const i = (y * img.width + x) * 3;
      if (!finite3(i)) {{
        continue;
      }}
      const vx = xyz[i];
      const vy = xyz[i + 1];
      const vz = xyz[i + 2];
      const norm = Math.hypot(vx, vy, vz);
      if (!Number.isFinite(norm) || norm <= 1e-12) {{
        continue;
      }}
      const sim = (sx * vx + sy * vy + sz * vz) / (seedNorm * norm);
      minSim = Math.min(minSim, sim);
      maxSim = Math.max(maxSim, sim);
      simValues[y * img.width + x] = sim;
    }}
  }}

  simMin = minSim;
  simMax = maxSim;
  selectedPx = px;
  selectedPy = py;
  renderSimilarity();
  mode = "sim";
  document.getElementById("mode").textContent = `similarity map, cosine ${{fmt(minSim)}} to ${{fmt(maxSim)}}`;
  document.getElementById("selected").textContent = `x=${{px}}, y=${{py}}, vector=(${{fmt(sx)}}, ${{fmt(sy)}}, ${{fmt(sz)}})`;
  draw();
}}

function showRgb() {{
  mode = "rgb";
  document.getElementById("mode").textContent = "RGB projection";
  document.getElementById("window").textContent = "-";
  draw();
}}

function toggleFlipY() {{
  flipY = !flipY;
  document.getElementById("flipY").textContent = flipY ? "Unflip Display Y" : "Flip Display Y";
  draw();
}}

canvas.addEventListener("wheel", (event) => {{
  event.preventDefault();
  const before = canvasToPixel(event.clientX, event.clientY);
  const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
  scale = Math.min(80, Math.max(0.02, scale * factor));
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const mx = (event.clientX - rect.left) * dpr;
  const my = (event.clientY - rect.top) * dpr;
  offsetX = mx - before.px * scale;
  offsetY = (flipY ? canvas.height - my : my) - before.py * scale;
  draw();
}}, {{ passive: false }});

canvas.addEventListener("pointerdown", (event) => {{
  dragging = true;
  moved = false;
  downX = event.clientX;
  downY = event.clientY;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointermove", (event) => {{
  const pix = canvasToPixel(event.clientX, event.clientY);
  document.getElementById("cursor").textContent =
    pix.px >= 0 && pix.py >= 0 && pix.px < img.width && pix.py < img.height
      ? `x=${{pix.px}}, y=${{pix.py}}`
      : "-";
  if (!dragging) return;
  const dpr = window.devicePixelRatio || 1;
  if (Math.hypot(event.clientX - downX, event.clientY - downY) > 3) moved = true;
  offsetX += (event.clientX - lastX) * dpr;
  offsetY += (event.clientY - lastY) * dpr;
  lastX = event.clientX;
  lastY = event.clientY;
  draw();
}});
canvas.addEventListener("pointerup", (event) => {{
  if (!moved) {{
    const pix = canvasToPixel(event.clientX, event.clientY);
    computeSimilarity(pix.px, pix.py);
  }}
  dragging = false;
  canvas.classList.remove("dragging");
}});

document.getElementById("fit").addEventListener("click", fit);
document.getElementById("one").addEventListener("click", oneToOne);
document.getElementById("rgb").addEventListener("click", showRgb);
document.getElementById("flipY").addEventListener("click", toggleFlipY);
document.getElementById("saveNpy").addEventListener("click", saveDisplayedNpy);
document.getElementById("gamma").addEventListener("input", renderSimilarity);
document.getElementById("colormap").addEventListener("change", renderSimilarity);
window.addEventListener("resize", resize);

document.getElementById("shape").textContent = STATS.shape.join(" x ");
document.getElementById("valid").textContent = `${{STATS.valid_pixels}} / ${{STATS.total_pixels}}`;
for (let i = 0; i < 3; i++) {{
  const ch = STATS.channels[i] || {{}};
  document.getElementById(`ch${{i}}`).textContent =
    `raw ${{fmt(ch.min)}} to ${{fmt(ch.max)}}, scaled ${{fmt(ch.lo_pct)}} to ${{fmt(ch.hi_pct)}}`;
}}

xyz = decodeFloat32(XYZ_B64);
img.onload = () => {{ resize(); fit(); }};
img.src = DATA_URL;
</script>
</body>
</html>
"""


# ── public API ──────────────────────────────────────────────────────

def build_interactive_umap(
    xyz_npy: str | Path,
    output_html: str | Path | None = None,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> str:
    """Build a self-contained click-to-similarity HTML visualizer from a UMAP XYZ .npy.

    Args:
        xyz_npy: Path to a (3,H,W) or (H,W,3) UMAP embedding .npy file.
        output_html: Output .html path (default: ``xyz_npy`` with ``.html`` suffix).
        lo_pct: Lower percentile for RGB scaling (default 1.0).
        hi_pct: Upper percentile for RGB scaling (default 99.0).

    Returns:
        Path to the generated HTML file.
    """
    input_path = Path(xyz_npy).expanduser().resolve()
    if output_html is None:
        output_path = input_path.with_suffix(".html")
    else:
        output_path = Path(output_html).expanduser().resolve()

    xyz = _as_hwc_xyz(np.load(input_path))
    html_content = _build_html_content(input_path, xyz, lo_pct, hi_pct)
    output_path.write_text(html_content, encoding="utf-8")
    return str(output_path)


def build_interactive_umap_from_embed_map(
    embed_map: list[np.ndarray],
    output_html: str | Path | None = None,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> str:
    """Build an interactive HTML visualizer from an ``embed_map`` list.

    The ``embed_map`` is the list of 2D arrays returned by
    ``decompose_and_embed()`` (e.g. ``[x_map, y_map]`` or ``[x_map, y_map, z_map]``).

    Args:
        embed_map: List of 2D ndarrays, one per UMAP dimension (2 or 3 channels).
        output_html: Output .html path. Defaults to a temp file.
        lo_pct: Lower percentile for RGB scaling.
        hi_pct: Upper percentile for RGB scaling.

    Returns:
        Path to the generated HTML file.
    """
    if len(embed_map) not in (2, 3):
        raise ValueError(
            f"embed_map must have 2 or 3 channels, got {len(embed_map)}"
        )

    # Stack into (C, H, W) array
    if len(embed_map) == 2:
        # 2-channel: duplicate one channel to make 3 for RGB visualisation
        stacked = np.stack([embed_map[0], embed_map[1], embed_map[0]], axis=0)
    else:
        stacked = np.stack(embed_map, axis=0)

    if output_html is None:
        output_html = Path(tempfile.mktemp(suffix=".html"))
    else:
        output_html = Path(output_html)

    # Save to a temporary .npy and call build_interactive_umap
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        np.save(tmp_path, stacked.astype(np.float32))
        return build_interactive_umap(tmp_path, output_html, lo_pct, hi_pct)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def open_interactive_umap(
    embed_map: list[np.ndarray],
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> str:
    """Same as ``build_interactive_umap_from_embed_map`` but opens in browser.

    Args:
        embed_map: List of 2D ndarrays from ``decompose_and_embed()``.
        lo_pct: Lower percentile for RGB scaling.
        hi_pct: Upper percentile for RGB scaling.

    Returns:
        Path to the generated HTML file.
    """
    html_path = build_interactive_umap_from_embed_map(embed_map, lo_pct=lo_pct, hi_pct=hi_pct)
    webbrowser.open(f"file://{html_path}")
    return html_path


# ── CLI entry point ─────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to a UMAP XYZ .npy file")
    parser.add_argument("--output", type=Path, default=None, help="HTML output path")
    parser.add_argument("--lo-pct", type=float, default=1.0, help="lower percentile for RGB scaling")
    parser.add_argument("--hi-pct", type=float, default=99.0, help="upper percentile for RGB scaling")
    args = parser.parse_args()
    out = build_interactive_umap(args.input, args.output, args.lo_pct, args.hi_pct)
    print(f"html_saved={out}")
    print(f"open {out}")


if __name__ == "__main__":
    main()
