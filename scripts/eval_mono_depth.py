"""Track A: zero-shot monocular metric depth vs sparse LiDAR on nuScenes-mini.

For every keyframe x 6 cameras (or every k-th keyframe with --stride), runs one model and evaluates
its depth at LiDAR pixels in [1, 80] m, both raw (metric) and per-image median-scaled. Breaks down by
camera, camera group (front / side / back) and depth bin. Writes results/mono_depth_<variant>.json.
Optionally caches the predicted depth maps of one scene (for 3DGS depth supervision).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from depth3d import nusc as N
from depth3d.lidar_gt import build_gt
from depth3d.metrics import DEPTH_BINS, MetricAccumulator, bin_masks, bootstrap_mean_ci, median_scale
from depth3d.mono import load_model, pred_cache_path

PER_IMAGE_COLUMNS = [
    "sample_token", "camera", "n_points", "abs_rel_raw", "abs_rel_scaled", "d1_raw", "d1_scaled",
    "scale", "rmse_raw", "rmse_scaled", "pred_focal_px", "gt_focal_px",
]  # fmt: skip


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["depth_pro", "dav2_metric_outdoor_large", "moge2"])
    ap.add_argument("--stride", type=int, default=1, help="use every k-th keyframe")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after this many images")
    ap.add_argument("--gt-cache", default="data/gt")
    ap.add_argument("--save-pred-scene", default="", help="scene name whose predictions are cached to data/pred")
    ap.add_argument("--pred-root", default="data/pred")
    ap.add_argument("--pred-downscale", type=int, default=2)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nusc = N.load_nusc()
    model = load_model(args.model, device=device)
    frames = N.all_keyframes(nusc)[:: args.stride]
    jobs = [(scene, tok, cam) for scene, tok in frames for cam in N.CAMERAS]
    if args.limit:
        jobs = jobs[: args.limit]

    def prepare(job):
        scene, tok, cam = job
        frame = N.cam_frame(nusc, tok, cam)
        gt = build_gt(nusc, tok, cam, cache_dir=Path(args.gt_cache))
        return scene, frame, gt, _load_image(frame.path)

    variants = list(model.variants)
    acc = {
        v: {
            "all_raw": MetricAccumulator(),
            "all_scaled": MetricAccumulator(),
            "cam_raw": {c: MetricAccumulator() for c in N.CAMERAS},
            "cam_scaled": {c: MetricAccumulator() for c in N.CAMERAS},
            "group_raw": {g: MetricAccumulator() for g in ("front", "side", "back")},
            "group_scaled": {g: MetricAccumulator() for g in ("front", "side", "back")},
            "bin_raw": {f"{int(lo)}-{int(hi)}m": MetricAccumulator() for lo, hi in DEPTH_BINS},
            "bin_scaled": {f"{int(lo)}-{int(hi)}m": MetricAccumulator() for lo, hi in DEPTH_BINS},
            "per_image": [],
        }
        for v in variants
    }
    times_ms: list[float] = []
    n_images = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for scene, frame, gt, image in tqdm(pool.map(prepare, jobs), total=len(jobs), desc=args.model):
            f_px = float(frame.K[0, 0])
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds = model.predict(image, f_px)
            if device == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            n_images += 1
            gtz = gt["z"].astype(np.float64)
            u, v = gt["u"], gt["v"]
            group = N.CAMERA_GROUP[frame.channel]
            extra = model.extra
            for var, depth in preds.items():
                pz = depth[v, u].astype(np.float64)
                ok = np.isfinite(pz) & (pz > 0)
                pz, gz = pz[ok], gtz[ok]
                s = median_scale(pz, gz)
                pzs = pz * s
                a = acc[var]
                a["all_raw"].add(pz, gz)
                a["all_scaled"].add(pzs, gz)
                a["cam_raw"][frame.channel].add(pz, gz)
                a["cam_scaled"][frame.channel].add(pzs, gz)
                a["group_raw"][group].add(pz, gz)
                a["group_scaled"][group].add(pzs, gz)
                for b, m in bin_masks(gz).items():
                    a["bin_raw"][b].add(pz[m], gz[m])
                    a["bin_scaled"][b].add(pzs[m], gz[m])
                img_raw = MetricAccumulator()
                img_raw.add(pz, gz)
                img_sc = MetricAccumulator()
                img_sc.add(pzs, gz)
                r, sc = img_raw.result(), img_sc.result()
                a["per_image"].append(
                    [
                        frame.sample_token, frame.channel, int(gz.size), r.get("abs_rel"), sc.get("abs_rel"),
                        r.get("d1"), sc.get("d1"), s, r.get("rmse"), sc.get("rmse"),
                        extra.get("pred_focal_px"), f_px,
                    ]  # fmt: skip
                )
                if args.save_pred_scene and scene == args.save_pred_scene:
                    p = pred_cache_path(Path(args.pred_root), var, frame.sd_token)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    k = args.pred_downscale
                    small = depth[::k, ::k].astype(np.float16) if k > 1 else depth.astype(np.float16)
                    np.savez_compressed(p, depth=small, downscale=k, scale_to_lidar=s)
    wall = time.time() - t_start
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for var in variants:
        a = acc[var]
        per_img = np.array([[row[3], row[4], row[7]] for row in a["per_image"]], dtype=np.float64)
        focals = {}
        for cam in N.CAMERAS:
            rows = [row for row in a["per_image"] if row[1] == cam and row[10] is not None]
            if rows:
                pf = np.array([row[10] for row in rows], dtype=np.float64)
                focals[cam] = {
                    "gt_focal_px": rows[0][11],
                    "pred_focal_px_median": float(np.median(pf)),
                    "pred_focal_px_p10": float(np.quantile(pf, 0.1)),
                    "pred_focal_px_p90": float(np.quantile(pf, 0.9)),
                    "median_ratio_pred_over_gt": float(np.median(pf) / rows[0][11]),
                }
        scale_by_cam = {}
        for cam in N.CAMERAS:
            sv = np.array([row[7] for row in a["per_image"] if row[1] == cam], dtype=np.float64)
            sv = sv[np.isfinite(sv)]
            if sv.size:
                scale_by_cam[cam] = {
                    "median": float(np.median(sv)),
                    "p10": float(np.quantile(sv, 0.1)),
                    "p90": float(np.quantile(sv, 0.9)),
                }
        result = {
            "variant": var,
            "model": args.model,
            "dataset": "nuScenes v1.0-mini",
            "gpu": "NVIDIA L4",
            "n_images": n_images,
            "n_keyframes": len(frames),
            "stride": args.stride,
            "depth_range_m": [1.0, 80.0],
            "timing_ms_per_image": {
                "mean": float(np.mean(times_ms)),
                "median": float(np.median(times_ms)),
                "note": "model forward incl. pre/post-processing and both variants when shared; excludes JPEG decode",
            },
            "wall_time_s": wall,
            "raw": a["all_raw"].result(),
            "scaled": a["all_scaled"].result(),
            "per_camera_raw": {c: m.result() for c, m in a["cam_raw"].items()},
            "per_camera_scaled": {c: m.result() for c, m in a["cam_scaled"].items()},
            "per_group_raw": {g: m.result() for g, m in a["group_raw"].items()},
            "per_group_scaled": {g: m.result() for g, m in a["group_scaled"].items()},
            "per_bin_raw": {b: m.result() for b, m in a["bin_raw"].items()},
            "per_bin_scaled": {b: m.result() for b, m in a["bin_scaled"].items()},
            "per_image_abs_rel_raw_ci": bootstrap_mean_ci(per_img[:, 0]) if per_img.size else None,
            "per_image_abs_rel_scaled_ci": bootstrap_mean_ci(per_img[:, 1]) if per_img.size else None,
            "scale_factor_by_camera": scale_by_cam,
            "focal_by_camera": focals,
            "per_image_columns": PER_IMAGE_COLUMNS,
            "per_image": a["per_image"],
        }
        out = Path(args.out_dir) / f"mono_depth_{var}.json"
        out.write_text(json.dumps(result, indent=1))
        print(var, "raw", {k: round(v, 4) for k, v in result["raw"].items() if k != "n_points"})
        print(var, "scaled", {k: round(v, 4) for k, v in result["scaled"].items() if k != "n_points"})
    print(f"images {n_images} wall {wall:.1f}s mean {np.mean(times_ms):.1f} ms/image (NVIDIA L4)")
    print("DONE")


if __name__ == "__main__":
    main()
