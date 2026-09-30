"""Cache sparse LiDAR depth GT for every keyframe x camera under data/gt/ (gitignored).

Also writes results/lidar_gt_stats.json with point counts and depth statistics per camera.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from depth3d import nusc as N
from depth3d.lidar_gt import build_gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/gt")
    ap.add_argument("--out", default="results/lidar_gt_stats.json")
    args = ap.parse_args()
    nusc = N.load_nusc()
    frames = N.all_keyframes(nusc)
    t0 = time.time()
    per_cam: dict[str, dict] = {c: {"n_images": 0, "n_points": 0, "depths": []} for c in N.CAMERAS}
    for _scene, tok in tqdm(frames, desc="lidar gt"):
        for cam in N.CAMERAS:
            gt = build_gt(nusc, tok, cam, cache_dir=Path(args.cache))
            per_cam[cam]["n_images"] += 1
            per_cam[cam]["n_points"] += int(len(gt["z"]))
            per_cam[cam]["depths"].append(gt["z"])
    stats = {}
    for cam, d in per_cam.items():
        z = np.concatenate(d["depths"]) if d["depths"] else np.zeros(0)
        stats[cam] = {
            "n_images": d["n_images"],
            "n_points": d["n_points"],
            "points_per_image": d["n_points"] / max(d["n_images"], 1),
            "median_depth_m": float(np.median(z)) if z.size else None,
            "frac_1_20m": float(np.mean(z < 20)) if z.size else None,
            "frac_20_40m": float(np.mean((z >= 20) & (z < 40))) if z.size else None,
            "frac_40_80m": float(np.mean(z >= 40)) if z.size else None,
        }
    out = {
        "n_keyframes": len(frames),
        "n_images": len(frames) * len(N.CAMERAS),
        "depth_range_m": [1.0, 80.0],
        "wall_time_s": time.time() - t0,
        "per_camera": stats,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_camera"}))
    print("DONE")


if __name__ == "__main__":
    main()
