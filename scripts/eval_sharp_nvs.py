"""Track D: Apple SHARP single-image 3D Gaussians rendered at the true next camera pose.

For one CAM_FRONT keyframe per scene: predict metric Gaussians from frame t (SHARP, known nuScenes focal),
render them (a) at the identity pose, (b) at the true relative pose of keyframe t+1 (0.5 s later) and
(c) at the next 12 Hz sweep (about 1/12 s later), using nuScenes ego poses + calibrated_sensor (OpenCV
convention, metres). Compare each render with the real frame: PSNR on covered pixels (alpha > 0.5),
SSIM averaged over covered pixels, LPIPS (alex) with uncovered pixels filled from the real frame.
Writes results/sharp_nvs.json and two side-by-side JPGs in docs/img/. On failure writes status != ok.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from depth3d import nusc as N
from depth3d.gs import _gaussian_window
from depth3d.metrics import bootstrap_mean_ci

CKPT = Path.home() / ".cache/torch/hub/checkpoints/sharp_2572gikvuh.pt"


def ssim_map(a: torch.Tensor, b: torch.Tensor, window: int = 11) -> torch.Tensor:
    C = a.shape[1]
    w = _gaussian_window(window, 1.5, a.device).repeat(C, 1, 1, 1)
    mu1 = F.conv2d(a, w, padding=window // 2, groups=C)
    mu2 = F.conv2d(b, w, padding=window // 2, groups=C)
    s11 = F.conv2d(a * a, w, padding=window // 2, groups=C) - mu1**2
    s22 = F.conv2d(b * b, w, padding=window // 2, groups=C) - mu2**2
    s12 = F.conv2d(a * b, w, padding=window // 2, groups=C) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s11 + s22 + c2))


def load_img(path, size) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].cuda()


def compare(render: torch.Tensor, alpha: torch.Tensor, real: torch.Tensor, lp) -> dict:
    """render, real: (1,3,H,W) in [0,1]; alpha: (1,1,H,W)."""
    mask = (alpha > 0.5).float()
    cov = float(mask.mean())
    if cov < 1e-3:
        return {"psnr": float("nan"), "ssim": float("nan"), "lpips_alex": float("nan"), "coverage": cov}
    mse = float((((render - real) ** 2) * mask).sum() / (mask.sum() * 3))
    filled = render * mask + real * (1 - mask)
    sm = ssim_map(filled, real)
    return {
        "psnr": float(10 * np.log10(1.0 / max(mse, 1e-10))),
        "ssim": float((sm * mask).sum() / (mask.sum() * 3)),
        "lpips_alex": float(lp(filled * 2 - 1, real * 2 - 1)),
        "coverage": cov,
    }


def side_by_side(real: torch.Tensor, render: torch.Tensor, path: Path, width: int = 400) -> None:
    def to_im(t):
        a = (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        im = Image.fromarray(a)
        return im.resize((width, int(im.height * width / im.width)), Image.BILINEAR)

    a, b = to_im(real), to_im(render)
    c = Image.new("RGB", (a.width * 2 + 4, a.height), (255, 255, 255))
    c.paste(a, (0, 0))
    c.paste(b, (a.width + 4, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    c.save(path, "JPEG", quality=80, optimize=True)


def run(args) -> dict:
    import lpips
    from sharp.cli.predict import predict_image
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import gsplat as sharp_gsplat

    device = torch.device("cuda")
    t0 = time.time()
    state = torch.load(CKPT, weights_only=True, map_location="cpu")
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state)
    predictor.eval().to(device)
    renderer = sharp_gsplat.GSplatRenderer(color_space="linearRGB")
    lp = lpips.LPIPS(net="alex", verbose=False).to(device)
    load_time = time.time() - t0
    nusc = N.load_nusc()
    rows = []
    pred_ms, render_ms = [], []
    eval_size = (800, 450)
    scenes = nusc.scene[: args.limit] if args.limit else nusc.scene
    for si, scene in enumerate(scenes):
        toks = N.scene_sample_tokens(nusc, scene["name"])
        ti = min(args.t_index, len(toks) - 2)
        f_t = N.cam_frame(nusc, toks[ti], "CAM_FRONT")
        f_n = N.cam_frame(nusc, toks[ti + 1], "CAM_FRONT")
        sd_t = nusc.get("sample_data", f_t.sd_token)
        sweep = nusc.get("sample_data", sd_t["next"]) if sd_t["next"] else None
        image = np.asarray(Image.open(f_t.path).convert("RGB"))
        H, W = image.shape[:2]
        f_px = float(f_t.K[0, 0])
        torch.cuda.synchronize()
        tp = time.perf_counter()
        gaussians = predict_image(predictor, image, f_px, device)
        torch.cuda.synchronize()
        pred_ms.append((time.perf_counter() - tp) * 1000)
        n_g = int(gaussians.mean_vectors.shape[1])
        K = torch.tensor([[f_px, 0, W / 2, 0], [0, f_px, H / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=device)[None]
        targets = {"identity": (np.eye(4), f_t.path)}
        T_next = np.linalg.inv(f_n.T_cam_to_world) @ f_t.T_cam_to_world  # points in cam_t -> cam_{t+1}
        targets["next_keyframe"] = (T_next, f_n.path)
        sweep_T = None
        if sweep is not None:
            T_sw_c2w = N.sensor_to_world(nusc, sweep["token"])
            sweep_T = np.linalg.inv(T_sw_c2w) @ f_t.T_cam_to_world
            targets["next_sweep"] = (sweep_T, N.data_root() / sweep["filename"])
        row = {
            "scene": scene["name"],
            "sample_token": f_t.sample_token,
            "next_sample_token": f_n.sample_token,
            "n_gaussians": n_g,
            "translation_m_next_keyframe": float(np.linalg.norm(T_next[:3, 3])),
            "rotation_deg_next_keyframe": float(
                np.degrees(np.arccos(np.clip((np.trace(T_next[:3, :3]) - 1) / 2, -1, 1)))
            ),
            "translation_m_next_sweep": float(np.linalg.norm(sweep_T[:3, 3])) if sweep_T is not None else None,
            "dt_s_next_keyframe": (f_n.timestamp - f_t.timestamp) / 1e6,
            "dt_s_next_sweep": (sweep["timestamp"] - f_t.timestamp) / 1e6 if sweep else None,
        }
        for name, (T, path) in targets.items():
            ext = torch.from_numpy(T).float().to(device)[None]
            torch.cuda.synchronize()
            tr = time.perf_counter()
            with torch.no_grad():
                out = renderer(gaussians, extrinsics=ext, intrinsics=K, image_width=W, image_height=H)
            torch.cuda.synchronize()
            render_ms.append((time.perf_counter() - tr) * 1000)
            color = F.interpolate(out.color, size=(eval_size[1], eval_size[0]), mode="area")
            alpha = F.interpolate(out.alpha, size=(eval_size[1], eval_size[0]), mode="area")
            real = load_img(path, eval_size)
            row[name] = compare(color.clamp(0, 1), alpha, real, lp)
            if name == "next_keyframe" and si in (0, 3):
                out_jpg = Path(args.img_dir) / f"sharp_{scene['name']}_next_keyframe.jpg"
                side_by_side(real, color * (alpha > 0.5).float(), out_jpg)
        rows.append(row)
        print(scene["name"], {k: round(row[k]["psnr"], 2) for k in targets}, flush=True)
    summary = {}
    for name in ("identity", "next_sweep", "next_keyframe"):
        vals = [r[name] for r in rows if name in r]
        if not vals:
            continue
        trans = [r[f"translation_m_{name}"] for r in rows if r.get(f"translation_m_{name}") is not None]
        summary[name] = {
            "psnr": bootstrap_mean_ci(np.array([v["psnr"] for v in vals])),
            "ssim": bootstrap_mean_ci(np.array([v["ssim"] for v in vals])),
            "lpips_alex": bootstrap_mean_ci(np.array([v["lpips_alex"] for v in vals])),
            "coverage_mean": float(np.mean([v["coverage"] for v in vals])),
            "translation_m_mean": float(np.mean(trans)) if trans else 0.0,
        }
    return {
        "status": "ok",
        "model": "SHARP (apple/ml-sharp, checkpoint sharp_2572gikvuh.pt), known nuScenes focal, "
        "principal point assumed at the image centre",
        "gpu": "NVIDIA L4",
        "n_images": len(rows),
        "eval_resolution": list(eval_size),
        "model_load_s": load_time,
        "predict_ms_mean": float(np.mean(pred_ms)),
        "render_ms_mean": float(np.mean(render_ms)),
        "summary": summary,
        "per_image": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-index", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N scenes")
    ap.add_argument("--img-dir", default="docs/img")
    ap.add_argument("--out", default="results/sharp_nvs.json")
    args = ap.parse_args()
    try:
        res = run(args)
    except Exception as e:  # noqa: BLE001
        res = {"status": "failed", "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
        print(res["traceback"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(res["status"], res.get("reason", ""))
    print("DONE")


if __name__ == "__main__":
    main()
