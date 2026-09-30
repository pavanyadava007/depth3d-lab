"""Build the static site (site/index.html + site/assets) for a Hugging Face static Space.

Everything on the page is derived from results/*.json, docs/RESULTS.md and the small renders produced by
the scripts. External libraries: plotly.js (jsdelivr), three.js (unpkg) and an own wasm-free WebGL2 splat viewer (assets/splatview.js, adapted from antimatter15/splat, MIT). Nothing is
uploaded by this script.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import struct
from pathlib import Path

import numpy as np

RES = Path("results")
SITE = Path("site")
ASSETS = SITE / "assets"
MAX_SPLAT_BYTES = 15 * 1024 * 1024


def load(name: str):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ markdown -> html (subset)


def md_to_html(md: str) -> str:
    out = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = [[html.escape(c.strip()) for c in r.strip().strip("|").split("|")] for r in rows]
            head, body = cells[0], cells[2:]
            t = ["<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead><tbody>"]
            for r in body:
                t.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue
        m = re.match(r"^(#+)\s+(.*)", ln)
        if m:
            lvl = min(len(m.group(1)) + 1, 4)
            out.append(f"<h{lvl}>{html.escape(m.group(2))}</h{lvl}>")
        elif ln.strip():
            txt = html.escape(ln)
            txt = re.sub(r"`([^`]+)`", r"<code>\1</code>", txt)
            txt = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", txt)
            out.append(f"<p>{txt}</p>")
        i += 1
    return "\n".join(out)


# ------------------------------------------------------------------ charts

COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def charts() -> list[dict]:
    ch = []
    monos = [json.loads(p.read_text()) for p in sorted(RES.glob("mono_depth_*.json"))]
    order = ["depth_pro", "depth_pro_gtfocal", "dav2_metric_outdoor_large", "moge2", "moge2_gtfocal"]
    monos.sort(key=lambda r: order.index(r["variant"]) if r["variant"] in order else 99)
    if monos:
        names = [m["variant"] for m in monos]
        ch.append({
            "id": "chart_a_overall", "title": "Track A: AbsRel, raw metric output vs per-image median scaling",
            "data": [
                {"type": "bar", "name": "raw", "x": names, "y": [m["raw"]["abs_rel"] for m in monos], "marker": {"color": COLORS[0]}},
                {"type": "bar", "name": "median-scaled", "x": names, "y": [m["scaled"]["abs_rel"] for m in monos], "marker": {"color": COLORS[1]}},
            ],
            "layout": {"barmode": "group", "yaxis": {"title": "AbsRel (lower is better)"}},
        })
        cams = list(monos[0]["per_camera_raw"].keys())
        ch.append({
            "id": "chart_a_cam", "title": "Track A: AbsRel per camera (raw metric output)",
            "data": [
                {"type": "bar", "name": m["variant"], "x": cams, "y": [m["per_camera_raw"][c]["abs_rel"] for c in cams], "marker": {"color": COLORS[i % 6]}}
                for i, m in enumerate(monos)
            ],
            "layout": {"barmode": "group", "yaxis": {"title": "AbsRel raw"}},
        })
        ch.append({
            "id": "chart_a_cam_scaled", "title": "Track A: AbsRel per camera (median-scaled = shape error)",
            "data": [
                {"type": "bar", "name": m["variant"], "x": cams, "y": [m["per_camera_scaled"][c]["abs_rel"] for c in cams], "marker": {"color": COLORS[i % 6]}}
                for i, m in enumerate(monos)
            ],
            "layout": {"barmode": "group", "yaxis": {"title": "AbsRel scaled"}},
        })
        bins = list(monos[0]["per_bin_raw"].keys())
        ch.append({
            "id": "chart_a_bins", "title": "Track A: AbsRel per depth bin (raw / scaled)",
            "data": [
                {"type": "bar", "name": f"{m['variant']} {k}", "x": bins, "y": [m[f"per_bin_{k}"][b]["abs_rel"] for b in bins],
                 "marker": {"color": COLORS[i % 6], "opacity": 1.0 if k == "raw" else 0.5}}
                for i, m in enumerate(monos) for k in ("raw", "scaled")
            ],
            "layout": {"barmode": "group", "yaxis": {"title": "AbsRel"}},
        })
        sc = [m for m in monos if m.get("scale_factor_by_camera")]
        ch.append({
            "id": "chart_a_scale", "title": "Track A: median scale factor GT/pred per camera (1.0 = metric scale correct)",
            "data": [
                {"type": "bar", "name": m["variant"], "x": cams, "y": [m["scale_factor_by_camera"][c]["median"] for c in cams], "marker": {"color": COLORS[i % 6]}}
                for i, m in enumerate(sc)
            ],
            "layout": {"barmode": "group", "yaxis": {"title": "median(GT)/median(pred)"}, "shapes": [{"type": "line", "x0": -0.5, "x1": len(cams) - 0.5, "y0": 1, "y1": 1, "line": {"dash": "dot", "color": "#888"}}]},
        })
    gss = [json.loads(p.read_text()) for p in sorted(RES.glob("gs_*.json"))]
    gorder = ["random", "lidar", "lidar_depth0.05", "lidar_depth0.2", "random_15k", "lidar_15k", "lidar_depth0.05_15k", "lidar_depth0.2_15k"]
    gss.sort(key=lambda r: gorder.index(r["name"]) if r["name"] in gorder else 99)
    if gss:
        names = [g["name"] for g in gss]
        for key, title in (("psnr", "PSNR (higher is better)"), ("ssim", "SSIM (higher is better)"), ("lpips_alex", "LPIPS alex (lower is better)")):
            ch.append({
                "id": f"chart_c_{key}", "title": f"Track C: held-out {title}",
                "data": [{"type": "bar", "x": names, "y": [g["test"][key]["mean"] for g in gss],
                          "error_y": {"type": "data", "array": [g["test"][key]["std"] for g in gss]}, "marker": {"color": COLORS[2]}}],
                "layout": {"yaxis": {"title": key}},
            })
        ch.append({
            "id": "chart_c_curve", "title": "Track C: training loss and number of Gaussians",
            "data": [{"type": "scatter", "mode": "lines", "name": g["name"], "x": [r[0] for r in g["loss_curve"]], "y": [r[1] for r in g["loss_curve"]]} for g in gss]
            + [{"type": "scatter", "mode": "lines", "name": f"{g['name']} gaussians", "x": [r[0] for r in g["loss_curve"]], "y": [r[5] for r in g["loss_curve"]], "yaxis": "y2", "line": {"dash": "dot"}} for g in gss],
            "layout": {"xaxis": {"title": "step"}, "yaxis": {"title": "loss"}, "yaxis2": {"title": "gaussians", "overlaying": "y", "side": "right"}},
        })
    mv = load("multiview_vggt.json")
    if mv:
        keys = ["vggt_seq_scaled", "vggt_sim3_scaled", "vggt_img_scaled", "depth_pro_img_scaled", "depth_pro_raw"]
        ch.append({
            "id": "chart_b", "title": "Track B: AbsRel on identical reference pixels, mean with 95% bootstrap CI",
            "data": [
                {"type": "bar", "name": s, "x": keys, "y": [mv["summary"][s]["abs_rel"][k]["mean"] for k in keys],
                 "error_y": {"type": "data", "symmetric": False,
                             "array": [mv["summary"][s]["abs_rel"][k]["hi"] - mv["summary"][s]["abs_rel"][k]["mean"] for k in keys],
                             "arrayminus": [mv["summary"][s]["abs_rel"][k]["mean"] - mv["summary"][s]["abs_rel"][k]["lo"] for k in keys]},
                 "marker": {"color": COLORS[i]}}
                for i, s in enumerate(mv["summary"])
            ],
            "layout": {"barmode": "group", "yaxis": {"title": "AbsRel"}},
        })
        pe = {s: [e["rot_err_deg"] for r in mv["sequences"][s] for e in r["pose_errors"]] for s in mv["sequences"]}
        te = {s: [e["trans_dir_err_deg"] for r in mv["sequences"][s] for e in r["pose_errors"]] for s in mv["sequences"]}
        ch.append({
            "id": "chart_b_pose", "title": "Track B: VGGT relative pose errors vs nuScenes GT (per view pair)",
            "data": [{"type": "box", "name": f"{s} rotation [deg]", "y": pe[s]} for s in pe] + [{"type": "box", "name": f"{s} translation direction [deg]", "y": te[s]} for s in te],
            "layout": {"yaxis": {"title": "degrees", "type": "log"}},
        })
    nf = load("synthesis_nextframe.json")
    if nf:
        ms = list(nf["methods"].keys())
        for key, title in (("psnr", "PSNR vs real next frame"), ("lpips_alex", "LPIPS (alex) vs real next frame")):
            ch.append({
                "id": f"chart_e_{key}", "title": f"Track E: {title}, 95% CI over held-out pairs",
                "data": [{"type": "bar", "x": ms, "y": [nf["methods"][m][key]["mean"] for m in ms],
                          "error_y": {"type": "data", "symmetric": False,
                                      "array": [nf["methods"][m][key]["hi"] - nf["methods"][m][key]["mean"] for m in ms],
                                      "arrayminus": [nf["methods"][m][key]["mean"] - nf["methods"][m][key]["lo"] for m in ms]},
                          "marker": {"color": COLORS[3]}}],
                "layout": {"yaxis": {"title": key}},
            })
        curves = []
        for mode in ("fm", "regression"):
            t = load(f"nextframe_train_{mode}.json")
            if t:
                curves.append({"type": "scatter", "mode": "lines", "name": mode, "x": [r[0] for r in t["curve"]], "y": [r[1] for r in t["curve"]]})
        if curves:
            ch.append({"id": "chart_e_curve", "title": "Track E: training loss", "data": curves, "layout": {"xaxis": {"title": "step"}, "yaxis": {"title": "loss", "type": "log"}}})
    sh = load("sharp_nvs.json")
    if sh and sh.get("status") == "ok":
        ks = list(sh["summary"].keys())
        ch.append({
            "id": "chart_d", "title": "Track D: SHARP render vs real frame, PSNR and LPIPS",
            "data": [
                {"type": "bar", "name": "PSNR", "x": ks, "y": [sh["summary"][k]["psnr"]["mean"] for k in ks], "marker": {"color": COLORS[4]}},
                {"type": "bar", "name": "LPIPS x 10", "x": ks, "y": [10 * sh["summary"][k]["lpips_alex"]["mean"] for k in ks], "marker": {"color": COLORS[5]}},
            ],
            "layout": {"barmode": "group"},
        })
    return ch


# ------------------------------------------------------------------ exports


def export_pointcloud() -> dict | None:
    src = Path("data/viewer/vggt_vs_lidar.npz")
    if not src.exists():
        return None
    d = np.load(src)
    vx = d["vggt_xyz"].astype(np.float32)
    vc = d["vggt_rgb"].astype(np.uint8)
    lx = d["lidar_xyz"].astype(np.float32)
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "vggt_points.bin").write_bytes(vx.tobytes())
    (ASSETS / "vggt_colors.bin").write_bytes(vc.tobytes())
    (ASSETS / "lidar_points.bin").write_bytes(lx.tobytes())
    return {"n_vggt": int(len(vx)), "n_lidar": int(len(lx)), "scene": str(d["scene"]), "camera": str(d["camera"])}


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], -1)


def export_splat(name: str) -> dict | None:
    """Write site/assets/scene.splat (antimatter15 layout) from data/gs/<name>.pt, y-down frame, <= 15 MB."""
    import torch

    src = Path(f"data/gs/{name}.pt")
    if not src.exists():
        return None
    p = torch.load(src, map_location="cpu")
    means = p["means"].numpy().astype(np.float32)
    scales = np.exp(p["scales"].numpy()).astype(np.float32)
    quats = p["quats"].numpy().astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9
    opac = 1 / (1 + np.exp(-p["opacities"].numpy()))
    rgb = np.clip(p["sh0"].numpy()[:, 0] * 0.28209479177387814 + 0.5, 0, 1)
    # rotate z-up world into y-down viewer frame: x' = x, y' = -z, z' = y (rotation of -90 deg about x)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    means = means @ R.T
    c = np.cos(-np.pi / 4)
    s = np.sin(-np.pi / 4)
    qR = np.array([c, s, 0, 0], dtype=np.float32)  # rotation by -90 deg about x, (w, x, y, z)
    quats = quat_mul(np.broadcast_to(qR, quats.shape), quats)
    keep = opac > 0.02
    idx = np.nonzero(keep)[0]
    max_n = MAX_SPLAT_BYTES // 32
    if len(idx) > max_n:
        rng = np.random.default_rng(0)
        # keep the most opaque / largest first so the scene stays readable
        weight = opac[idx] * scales[idx].mean(1)
        idx = idx[np.argsort(-weight)[:max_n]]
        rng.shuffle(idx)
    buf = bytearray()
    for i in idx:
        buf += struct.pack("<3f", *means[i])
        buf += struct.pack("<3f", *scales[i])
        buf += struct.pack("<4B", *(rgb[i] * 255).astype(np.uint8), int(opac[i] * 255))
        q = quats[i]
        buf += struct.pack("<4B", *np.clip(q * 128 + 128, 0, 255).astype(np.uint8))
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "scene.splat").write_bytes(bytes(buf))
    return {"n_gaussians_exported": int(len(idx)), "n_gaussians_total": int(len(means)), "bytes": len(buf), "source": name}


def copy_media() -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path("scripts/web/splatview.js"), ASSETS / "splatview.js")
    media = {"gs_previews": [], "videos": [], "grid": None, "sharp": []}
    for p in sorted(Path("docs/img").glob("gs_*.jpg")):
        shutil.copy(p, ASSETS / p.name)
        media["gs_previews"].append(p.name)
    for p in sorted(Path("docs/img").glob("*.mp4")):
        shutil.copy(p, ASSETS / p.name)
        media["videos"].append(p.name)
    g = Path("docs/img/nextframe_grid.jpg")
    if g.exists():
        shutil.copy(g, ASSETS / g.name)
        media["grid"] = g.name
    for p in sorted(Path("docs/img").glob("sharp_*.jpg")):
        shutil.copy(p, ASSETS / p.name)
        media["sharp"].append(p.name)
    return media


# ------------------------------------------------------------------ page

CSS = """
:root{--bg:#fafafa;--fg:#1c1c1c;--muted:#666;--card:#fff;--line:#ddd;--acc:#4C78A8}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:var(--bg);color:var(--fg);line-height:1.45}
header{background:#1f2937;color:#fff;padding:22px 16px}header h1{margin:0;font-size:1.5rem}header p{margin:6px 0 0;color:#cbd5e1}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:14px;flex-wrap:wrap;z-index:5}
nav a{color:var(--acc);text-decoration:none;font-weight:600}
main{max-width:1180px;margin:0 auto;padding:16px}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:16px 0}
table{border-collapse:collapse;font-size:0.85rem;margin:8px 0;display:block;overflow-x:auto;white-space:nowrap}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
.chart{width:100%;height:380px}
.tabs button{border:1px solid var(--line);background:#fff;padding:6px 10px;margin:2px;border-radius:6px;cursor:pointer}
.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.viewer{width:100%;height:520px;background:#111;border-radius:6px}
.row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1 1 320px}
img{max-width:100%}code{background:#f1f5f9;padding:1px 4px;border-radius:4px}
.small{color:var(--muted);font-size:0.85rem}
video{max-width:100%;border-radius:6px}
"""


def build_page(results_html: str, chart_specs: list[dict], pc: dict | None, splat: dict | None, media: dict, depth_meta: list | None) -> str:
    chart_divs = "\n".join(f'<h4>{html.escape(c["title"])}</h4><div id="{c["id"]}" class="chart"></div>' for c in chart_specs)
    chart_json = json.dumps([{"id": c["id"], "data": c["data"], "layout": {**c["layout"], "margin": {"t": 20, "r": 20, "b": 60, "l": 60}}} for c in chart_specs])

    depth_html = "<p class='small'>Depth previews not rendered (run scripts/render_depth_previews.py).</p>"
    if depth_meta:
        depth_html = "<div class='tabs' id='depth_frames'>" + "".join(
            f"<button data-k='{i}' class='{'on' if i == 0 else ''}'>{html.escape(m['scene'])} {html.escape(m['camera'])}</button>" for i, m in enumerate(depth_meta)
        ) + "</div><div class='tabs' id='depth_layers'>" + "".join(
            f"<button data-l='{layer}' class='{'on' if layer == 'image' else ''}'>{label}</button>"
            for layer, label in (("image", "image"), ("depth_pro", "Depth Pro"), ("dav2", "Depth Anything V2 metric"), ("lidar", "LiDAR GT overlay"))
        ) + "</div><div class='row'><div><img id='depth_img' src='assets/depth/" + depth_meta[0]["layers"]["image"] + "' alt='depth preview'></div>" \
            "<div id='depth_info' class='small'></div></div>" \
            "<p class='small'>Colour scale: log depth 1-80 m (turbo, red = near). AbsRel on this frame's LiDAR pixels is shown on the right.</p>"

    pc_html = "<p class='small'>Point cloud export missing (run scripts/eval_multiview.py).</p>"
    if pc:
        pc_html = f"<div id='pc_viewer' class='viewer'></div><p class='small'>VGGT point map of {html.escape(pc['camera'])} in {html.escape(pc['scene'])} ({pc['n_vggt']:,} points, coloured by the image, scaled to LiDAR with one sequence scale) and the LiDAR return of the same keyframe ({pc['n_lidar']:,} points, white). Drag to orbit, wheel to zoom. Toggle: <button id='pc_toggle_vggt'>VGGT</button> <button id='pc_toggle_lidar'>LiDAR</button></p>"

    splat_html = "<p class='small'>Splat export missing (train Track C first).</p>"
    if splat:
        splat_html = f"<div id='splat_viewer' class='viewer'></div><p class='small'>{splat['n_gaussians_exported']:,} of {splat['n_gaussians_total']:,} Gaussians from the '{html.escape(splat['source'])}' variant ({splat['bytes'] / 1e6:.1f} MB .splat). Drag to orbit, right-drag to pan, wheel to zoom. Moving cars and pedestrians appear as ghosts because the scene is modelled as static.</p>"

    prev_html = "".join(f"<div><img src='assets/{p}' alt='{p}'><div class='small'>{p} (left: held-out photo, right: render)</div></div>" for p in media["gs_previews"])
    vid_html = "".join(f"<div><video controls loop muted playsinline src='assets/{v}'></video><div class='small'>{v}: CAM_FRONT path rendered by the 3DGS model (train and held-out timestamps)</div></div>" for v in media["videos"])
    grid_html = f"<img src='assets/{media['grid']}' alt='next-frame grid'><div class='small'>Rows: current frame t, real frame t+1, flow-matching sample of t+1 (held-out scenes, 128x64 upsampled 2x).</div>" if media["grid"] else "<p class='small'>Grid missing.</p>"
    sharp_html = "".join(f"<div><img src='assets/{p}' alt='{p}'><div class='small'>{p}</div></div>" for p in media["sharp"]) or "<p class='small'>No SHARP renders (see STATUS.md).</p>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>depth3d-lab</title>
<style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
</head><body>
<header><h1>depth3d-lab: depth, 3D reconstruction and view synthesis on nuScenes-mini</h1>
<p>Zero-shot evaluation of public models plus small trained models, all measured on one NVIDIA L4. Every number on this page is generated from results/*.json. Author: Pavan Yadav Annappa. MIT licence for the code; nuScenes-mini is CC BY-NC-SA 4.0 and is not redistributed here (only tokens, metrics and small renders).</p></header>
<nav><a href="#charts">Charts</a><a href="#depth">Depth viewer</a><a href="#pointcloud">Point cloud</a><a href="#splat">Gaussian splats</a><a href="#video">Videos and grids</a><a href="#tables">All tables</a></nav>
<main>
<section id="charts"><h2>Charts</h2>{chart_divs}</section>
<section id="depth"><h2>Depth comparison viewer</h2>{depth_html}</section>
<section id="pointcloud"><h2>3D point cloud: VGGT vs LiDAR</h2>{pc_html}</section>
<section id="splat"><h2>Trained 3D Gaussian Splatting scene</h2>{splat_html}</section>
<section id="video"><h2>Renders</h2><h3>3DGS held-out previews</h3><div class="row">{prev_html}</div><h3>3DGS camera path</h3><div class="row">{vid_html}</div><h3>Track E: real vs generated next frames</h3>{grid_html}<h3>Track D: SHARP renders</h3><div class="row">{sharp_html}</div></section>
<section id="tables"><h2>All result tables (docs/RESULTS.md)</h2>{results_html}</section>
</main>
<script>
const CHARTS = {chart_json};
for (const c of CHARTS) {{ Plotly.newPlot(c.id, c.data, c.layout, {{responsive: true, displaylogo: false}}); }}
const DEPTH = {json.dumps(depth_meta or [])};
let dk = 0, dl = 'image';
function updDepth() {{
  if (!DEPTH.length) return;
  const m = DEPTH[dk];
  document.getElementById('depth_img').src = 'assets/depth/' + m.layers[dl];
  document.getElementById('depth_info').innerHTML = `<b>${{m.scene}} ${{m.camera}}</b> (keyframe ${{m.t_index}}, ${{m.n_lidar_points}} LiDAR points)<br>` +
    `Depth Pro AbsRel raw ${{m.depth_pro_abs_rel_raw.toFixed(3)}}, scaled ${{m.depth_pro_abs_rel_scaled.toFixed(3)}}, median scale ${{m.depth_pro_median_scale.toFixed(2)}}<br>` +
    `DA-V2 metric AbsRel raw ${{m.dav2_abs_rel_raw.toFixed(3)}}, scaled ${{m.dav2_abs_rel_scaled.toFixed(3)}}, median scale ${{m.dav2_median_scale.toFixed(2)}}`;
  document.querySelectorAll('#depth_frames button').forEach(b => b.classList.toggle('on', +b.dataset.k === dk));
  document.querySelectorAll('#depth_layers button').forEach(b => b.classList.toggle('on', b.dataset.l === dl));
}}
document.querySelectorAll('#depth_frames button').forEach(b => b.onclick = () => {{ dk = +b.dataset.k; updDepth(); }});
document.querySelectorAll('#depth_layers button').forEach(b => b.onclick = () => {{ dl = b.dataset.l; updDepth(); }});
updDepth();
</script>
<script type="importmap">{{"imports": {{"three": "https://unpkg.com/three@0.160.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
const el = document.getElementById('pc_viewer');
if (el) {{
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(60, el.clientWidth / el.clientHeight, 0.1, 500);
  cam.position.set(0, -8, -12);
  const renderer = new THREE.WebGLRenderer({{antialias: true}});
  renderer.setSize(el.clientWidth, el.clientHeight); el.appendChild(renderer.domElement);
  const ctrl = new OrbitControls(cam, renderer.domElement); ctrl.target.set(0, 0, 20);
  async function loadBin(url) {{ const r = await fetch(url); return await r.arrayBuffer(); }}
  const [vp, vc, lp] = await Promise.all([loadBin('assets/vggt_points.bin'), loadBin('assets/vggt_colors.bin'), loadBin('assets/lidar_points.bin')]);
  // camera frame is OpenCV (y down): flip y so that up is up in the viewer
  const flip = (buf) => {{ const a = new Float32Array(buf); for (let i = 1; i < a.length; i += 3) a[i] = -a[i]; return a; }};
  const g1 = new THREE.BufferGeometry(); g1.setAttribute('position', new THREE.BufferAttribute(flip(vp), 3));
  g1.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(vc), 3, true));
  const m1 = new THREE.Points(g1, new THREE.PointsMaterial({{size: 0.12, vertexColors: true}})); scene.add(m1);
  const g2 = new THREE.BufferGeometry(); g2.setAttribute('position', new THREE.BufferAttribute(flip(lp), 3));
  const m2 = new THREE.Points(g2, new THREE.PointsMaterial({{size: 0.15, color: 0xffffff}})); scene.add(m2);
  document.getElementById('pc_toggle_vggt').onclick = () => m1.visible = !m1.visible;
  document.getElementById('pc_toggle_lidar').onclick = () => m2.visible = !m2.visible;
  scene.add(new THREE.AxesHelper(2));
  (function anim() {{ requestAnimationFrame(anim); ctrl.update(); renderer.render(scene, cam); }})();
}}
</script>
<script type="module">
import {{ mountSplatViewer }} from './assets/splatview.js';
const el = document.getElementById('splat_viewer');
if (el) {{ mountSplatViewer(el, 'assets/scene.splat', {{ target: [0, 0, 20], distance: 20, pitch: 0.08 }}).catch(e => {{ el.textContent = 'viewer error: ' + e; }}); }}
</script>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat-variant", default="")
    args = ap.parse_args()
    SITE.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    results_md = Path("docs/RESULTS.md")
    results_html = md_to_html(results_md.read_text()) if results_md.exists() else "<p>RESULTS.md missing</p>"
    pc = export_pointcloud()
    variant = args.splat_variant
    if not variant:
        gss = [json.loads(p.read_text()) for p in RES.glob("gs_*.json")]
        variant = max(gss, key=lambda g: g["test"]["psnr"]["mean"])["name"] if gss else "lidar"
    splat = export_splat(variant)
    media = copy_media()
    depth_meta = None
    dm = ASSETS / "depth" / "index.json"
    if dm.exists():
        depth_meta = json.loads(dm.read_text())
    page = build_page(results_html, charts(), pc, splat, media, depth_meta)
    if any(ch in page for ch in "–—"):
        raise SystemExit("long dash in page")
    (SITE / "index.html").write_text(page)
    total = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    manifest = {str(p.relative_to(SITE)): p.stat().st_size for p in sorted(SITE.rglob("*")) if p.is_file()}
    (SITE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"site: {len(manifest)} files, {total / 1e6:.1f} MB total; splat={splat}; pointcloud={pc}")


if __name__ == "__main__":
    main()
