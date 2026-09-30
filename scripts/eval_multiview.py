"""Track B: feed-forward multi-view reconstruction (VGGT) vs LiDAR on nuScenes-mini.

Setting (i): the 6-camera rig at one timestamp (wide baseline, low overlap), reference = CAM_FRONT.
Setting (ii): 5 consecutive CAM_FRONT keyframes (forward motion, 2 Hz), reference = centre frame.
Depth on the reference view is aligned with ONE scale per sequence (median(GT)/median(pred) over all views'
LiDAR pixels, the same estimator that is used per image), with a sim(3) (Umeyama on camera centres) scale,
and with per-image median scaling.
Relative camera poses are compared with nuScenes ego poses + calibration. Depth Pro is run on the same
reference images so that the mono-vs-multi-view comparison uses identical LiDAR pixels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from depth3d import nusc as N
from depth3d.lidar_gt import build_gt
from depth3d.metrics import (
    MetricAccumulator,
    bootstrap_mean_ci,
    median_scale,
    paired_bootstrap,
    relative_pose,
    rotation_error_deg,
    translation_direction_error_deg,
    umeyama,
)
from depth3d.mono import load_model
from depth3d.multiview import VGGTModel


def upsample(depth: np.ndarray, h: int, w: int) -> np.ndarray:
    t = torch.from_numpy(depth)[None, None]
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()


def metrics_of(pred: np.ndarray, gt: np.ndarray) -> dict:
    acc = MetricAccumulator()
    acc.add(pred, gt)
    return acc.result()


def eval_sequence(vggt, mono, nusc, frames: list, ref: int, gt_cache: Path, viewer_out: Path | None = None) -> dict:
    out = vggt.infer([f.path for f in frames])
    S = len(frames)
    gts = [build_gt(nusc, f.sample_token, f.channel, cache_dir=gt_cache) for f in frames]
    preds_full = [upsample(out["depth"][i], frames[i].height, frames[i].width) for i in range(S)]
    # one scale per sequence from all views: median(GT) / median(pred), the same estimator as per image
    all_pz, all_gz = [], []
    for i in range(S):
        pz = preds_full[i][gts[i]["v"], gts[i]["u"]]
        ok = pz > 1e-3
        all_pz.append(pz[ok])
        all_gz.append(gts[i]["z"][ok])
    s_seq = median_scale(np.concatenate(all_pz), np.concatenate(all_gz))
    # camera poses: predicted cam-from-world -> cam-to-world (pred world = first camera)
    T_pred_c2w = []
    for i in range(S):
        T = np.eye(4)
        T[:3] = out["extrinsics"][i]
        T_pred_c2w.append(np.linalg.inv(T))
    T_gt_c2w = [f.T_cam_to_world for f in frames]
    c_pred = np.stack([T[:3, 3] for T in T_pred_c2w])
    c_gt = np.stack([T[:3, 3] for T in T_gt_c2w])
    try:
        s_sim, _R, _t = umeyama(c_pred, c_gt)
    except ValueError:
        s_sim = float("nan")
    pose_errors = []
    for j in range(S):
        if j == ref:
            continue
        rel_p = relative_pose(T_pred_c2w[ref], T_pred_c2w[j])
        rel_g = relative_pose(T_gt_c2w[ref], T_gt_c2w[j])
        pose_errors.append(
            {
                "view": frames[j].channel if S == 6 else f"t{j - ref:+d}",
                "rot_err_deg": rotation_error_deg(rel_p[:3, :3], rel_g[:3, :3]),
                "trans_dir_err_deg": translation_direction_error_deg(rel_p[:3, 3], rel_g[:3, 3]),
                "gt_baseline_m": float(np.linalg.norm(rel_g[:3, 3])),
                "pred_baseline_scaled_m": float(np.linalg.norm(rel_p[:3, 3]) * s_seq),
            }
        )
    # reference view depth metrics
    g = gts[ref]
    pz = preds_full[ref][g["v"], g["u"]].astype(np.float64)
    gz = g["z"].astype(np.float64)
    ok = pz > 1e-3
    pz, gz = pz[ok], gz[ok]
    s_img = median_scale(pz, gz)
    mono_pred = mono.predict(np.asarray(Image.open(frames[ref].path).convert("RGB")), float(frames[ref].K[0, 0]))
    mono_z = {k: v[g["v"], g["u"]].astype(np.float64)[ok] for k, v in mono_pred.items()}
    res = {
        "scene": frames[ref].scene_name,
        "ref_sample_token": frames[ref].sample_token,
        "ref_camera": frames[ref].channel,
        "n_ref_points": int(gz.size),
        "scale_seq": s_seq,
        "scale_sim3": s_sim,
        "scale_img": s_img,
        "vggt_raw": metrics_of(pz, gz),
        "vggt_seq_scaled": metrics_of(pz * s_seq, gz),
        "vggt_sim3_scaled": metrics_of(pz * s_sim, gz) if np.isfinite(s_sim) else None,
        "vggt_img_scaled": metrics_of(pz * s_img, gz),
        "depth_pro_raw": metrics_of(mono_z["depth_pro"], gz),
        "depth_pro_img_scaled": metrics_of(mono_z["depth_pro"] * median_scale(mono_z["depth_pro"], gz), gz),
        "depth_pro_gtfocal_raw": metrics_of(mono_z["depth_pro_gtfocal"], gz),
        "pose_errors": pose_errors,
        "all_views_seq_scaled_abs_rel": [
            float(np.mean(np.abs(preds_full[i][gts[i]["v"], gts[i]["u"]] * s_seq - gts[i]["z"]) / gts[i]["z"]))
            for i in range(S)
        ],
    }
    if viewer_out is not None:
        # point cloud of the reference view in its camera frame (metres after sequence scaling) + LiDAR
        Kp = out["intrinsics"][ref]
        d = out["depth"][ref] * s_seq
        conf = out["depth_conf"][ref]
        hh, ww = d.shape
        us, vs = np.meshgrid(np.arange(ww), np.arange(hh))
        x = (us - Kp[0, 2]) / Kp[0, 0] * d
        y = (vs - Kp[1, 2]) / Kp[1, 1] * d
        pts = np.stack([x, y, d], -1).reshape(-1, 3)
        cols = (out["images"][ref].transpose(1, 2, 0).reshape(-1, 3) * 255).astype(np.uint8)
        keep = (conf.reshape(-1) > np.quantile(conf, 0.2)) & (d.reshape(-1) < 80) & (d.reshape(-1) > 0.5)
        pts, cols = pts[keep], cols[keep]
        step = max(1, len(pts) // 60000)
        pts, cols = pts[::step], cols[::step]
        lid = N.lidar_points(nusc, frames[ref].sample_token)
        T = N.lidar_to_cam_matrix(nusc, frames[ref].sample_token, frames[ref].channel)
        lid_c = (T[:3, :3] @ lid.T).T + T[:3, 3]
        lid_c = lid_c[(lid_c[:, 2] > 0.5) & (lid_c[:, 2] < 80)]
        viewer_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            viewer_out, vggt_xyz=pts.astype(np.float16), vggt_rgb=cols, lidar_xyz=lid_c.astype(np.float16),
            scene=frames[ref].scene_name, camera=frames[ref].channel, sample_token=frames[ref].sample_token,
        )  # fmt: skip
    return res


def summarize(rows: list[dict], key: str, metric: str = "abs_rel") -> dict:
    vals = np.array([r[key][metric] for r in rows if r.get(key)], dtype=np.float64)
    return bootstrap_mean_ci(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=4, help="sequences per scene per setting")
    ap.add_argument("--gt-cache", default="data/gt")
    ap.add_argument("--out", default="results/multiview_vggt.json")
    args = ap.parse_args()
    nusc = N.load_nusc()
    t_load = time.time()
    vggt = VGGTModel()
    mono = load_model("depth_pro")
    load_time = time.time() - t_load
    results = {"rig6": [], "temporal5": []}
    times = {"rig6": [], "temporal5": []}
    viewer_done = False
    for scene in tqdm(nusc.scene, desc="scenes"):
        toks = N.scene_sample_tokens(nusc, scene["name"])
        for ti in np.linspace(0, len(toks) - 1, args.per_scene).astype(int):
            frames = [N.cam_frame(nusc, toks[ti], c) for c in N.CAMERAS]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = eval_sequence(vggt, mono, nusc, frames, ref=0, gt_cache=Path(args.gt_cache))
            torch.cuda.synchronize()
            times["rig6"].append((time.perf_counter() - t0) * 1000)
            r["t_index"] = int(ti)
            results["rig6"].append(r)
        for start in np.linspace(0, len(toks) - 5, args.per_scene).astype(int):
            frames = [N.cam_frame(nusc, toks[start + k], "CAM_FRONT") for k in range(5)]
            vo = None
            if not viewer_done:
                vo = Path("data/viewer/vggt_vs_lidar.npz")
                viewer_done = True
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = eval_sequence(vggt, mono, nusc, frames, ref=2, gt_cache=Path(args.gt_cache), viewer_out=vo)
            torch.cuda.synchronize()
            times["temporal5"].append((time.perf_counter() - t0) * 1000)
            r["t_start"] = int(start)
            results["temporal5"].append(r)
    summary = {}
    cmp_keys = ["vggt_seq_scaled", "vggt_img_scaled", "depth_pro_img_scaled"]
    for setting, rows in results.items():
        pe = [e for r in rows for e in r["pose_errors"]]
        summary[setting] = {
            "n_sequences": len(rows),
            "n_ref_points_total": int(sum(r["n_ref_points"] for r in rows)),
            "abs_rel": {
                k: summarize(rows, k)
                for k in [
                    "vggt_raw", "vggt_seq_scaled", "vggt_sim3_scaled", "vggt_img_scaled",
                    "depth_pro_raw", "depth_pro_img_scaled", "depth_pro_gtfocal_raw",
                ]  # fmt: skip
            },
            "d1": {k: summarize(rows, k, "d1") for k in cmp_keys},
            "rmse": {k: summarize(rows, k, "rmse") for k in cmp_keys},
            "paired_abs_rel_vggt_seq_minus_depthpro_img": paired_bootstrap(
                np.array([r["vggt_seq_scaled"]["abs_rel"] for r in rows]),
                np.array([r["depth_pro_img_scaled"]["abs_rel"] for r in rows]),
            ),
            "paired_abs_rel_vggt_img_minus_depthpro_img": paired_bootstrap(
                np.array([r["vggt_img_scaled"]["abs_rel"] for r in rows]),
                np.array([r["depth_pro_img_scaled"]["abs_rel"] for r in rows]),
            ),
            "pose": {
                "rot_err_deg_mean": float(np.mean([e["rot_err_deg"] for e in pe])),
                "rot_err_deg_median": float(np.median([e["rot_err_deg"] for e in pe])),
                "trans_dir_err_deg_mean": float(np.nanmean([e["trans_dir_err_deg"] for e in pe])),
                "trans_dir_err_deg_median": float(np.nanmedian([e["trans_dir_err_deg"] for e in pe])),
                "gt_baseline_m_mean": float(np.mean([e["gt_baseline_m"] for e in pe])),
                "n_pairs": len(pe),
            },
            "scale_seq_median": float(np.median([r["scale_seq"] for r in rows])),
            "timing_ms_per_sequence": {
                "mean": float(np.mean(times[setting])),
                "note": "VGGT forward + Depth Pro on the reference view + metric computation, NVIDIA L4",
            },
        }
    out = {
        "model": "VGGT-1B (facebook/VGGT-1B), bf16 autocast, input 518x294",
        "mono_reference": "Depth Pro on the same reference image and LiDAR pixels",
        "gpu": "NVIDIA L4",
        "model_load_s": load_time,
        "summary": summary,
        "sequences": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for k, v in summary.items():
        print(k, {m: round(c["mean"], 4) for m, c in v["abs_rel"].items() if c["n"]}, v["pose"])
    print("DONE")


if __name__ == "__main__":
    main()
