"""Render small PNGs for the site's depth comparison viewer: image, Depth Pro, DA-V2, LiDAR overlay.

Four fixed frames (one per camera group and one night scene). Outputs go to site/assets/depth/ at
<= 400 px wide. These are our own renders derived from nuScenes-mini, kept small on purpose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from depth3d import nusc as N
from depth3d.lidar_gt import build_gt
from depth3d.metrics import median_scale
from depth3d.mono import load_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FRAMES = [
    ("scene-0061", 10, "CAM_FRONT"),
    ("scene-0916", 20, "CAM_BACK"),
    ("scene-0553", 15, "CAM_FRONT_LEFT"),
    ("scene-1094", 12, "CAM_FRONT"),
]


def colorize(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    d = np.clip((np.log(np.clip(depth, vmin, vmax)) - np.log(vmin)) / (np.log(vmax) - np.log(vmin)), 0, 1)
    return (plt.get_cmap("turbo")(1 - d)[..., :3] * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/assets/depth")
    ap.add_argument("--width", type=int, default=400)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nusc = N.load_nusc()
    models = {"depth_pro": load_model("depth_pro"), "dav2": load_model("dav2_metric_outdoor_large")}
    meta = []
    for scene, ti, cam in FRAMES:
        tok = N.scene_sample_tokens(nusc, scene)[ti]
        frame = N.cam_frame(nusc, tok, cam)
        img = np.asarray(Image.open(frame.path).convert("RGB"))
        gt = build_gt(nusc, tok, cam, cache_dir=Path("data/gt"))
        w = args.width
        h = int(img.shape[0] * w / img.shape[1])
        key = f"{scene}_{ti}_{cam.lower()}"
        Image.fromarray(img).resize((w, h), Image.BILINEAR).save(out / f"{key}_image.png", optimize=True)
        vmin, vmax = 1.0, 80.0
        entry = {"key": key, "scene": scene, "camera": cam, "t_index": ti, "layers": {"image": f"{key}_image.png"}}
        for name, m in models.items():
            pred = m.predict(img, float(frame.K[0, 0]))
            depth = pred["depth_pro"] if name == "depth_pro" else pred["dav2_metric_outdoor_large"]
            pz = depth[gt["v"], gt["u"]]
            s = median_scale(pz, gt["z"])
            entry[f"{name}_abs_rel_raw"] = float(np.mean(np.abs(pz - gt["z"]) / gt["z"]))
            entry[f"{name}_abs_rel_scaled"] = float(np.mean(np.abs(pz * s - gt["z"]) / gt["z"]))
            entry[f"{name}_median_scale"] = float(s)
            col = colorize(depth, vmin, vmax)
            Image.fromarray(col).resize((w, h), Image.BILINEAR).save(out / f"{key}_{name}.png", optimize=True)
            entry["layers"][name] = f"{key}_{name}.png"
        # LiDAR overlay: image darkened with coloured dots
        over = (img * 0.45).astype(np.uint8).copy()
        cols = colorize(gt["z"], vmin, vmax)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                uu = np.clip(gt["u"] + du, 0, img.shape[1] - 1)
                vv = np.clip(gt["v"] + dv, 0, img.shape[0] - 1)
                over[vv, uu] = cols
        Image.fromarray(over).resize((w, h), Image.BILINEAR).save(out / f"{key}_lidar.png", optimize=True)
        entry["layers"]["lidar"] = f"{key}_lidar.png"
        entry["n_lidar_points"] = int(len(gt["z"]))
        meta.append(entry)
    (out / "index.json").write_text(json.dumps(meta, indent=1))
    print("wrote", len(meta), "frames to", out)


if __name__ == "__main__":
    main()
