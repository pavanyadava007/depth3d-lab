"""Track C: 3D Gaussian Splatting on one nuScenes-mini scene with gsplat and an own training loop.

Variants (the ablation):
  random       - SfM-free init: uniform points in a box around the trajectory, random colours
  lidar        - aggregated train-timestamp LiDAR in the world frame, coloured by projection
  lidar_depth  - lidar init + monocular depth supervision (rendered expected depth vs the cached,
                 per-image LiDAR-scale-aligned Track A prediction), --depth-weight ablated

Writes results/gs_<name>.json, two held-out preview JPGs to docs/img/, a CAM_FRONT path GIF/MP4
and the final Gaussians (data/gs/<name>.pt, gitignored) for the web viewer export.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from depth3d import gs as G
from depth3d import nusc as N


def save_side_by_side(gt: torch.Tensor, pred: torch.Tensor, path: Path, width: int = 400) -> None:
    def to_img(t):
        a = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        im = Image.fromarray(a)
        h = int(im.height * width / im.width)
        return im.resize((width, h), Image.BILINEAR)

    a, b = to_img(gt), to_img(pred)
    canvas = Image.new("RGB", (a.width + b.width + 4, a.height), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + 4, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "JPEG", quality=80, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="")
    ap.add_argument("--variant", required=True, choices=["random", "lidar", "lidar_depth"])
    ap.add_argument("--name", default="")
    ap.add_argument("--depth-weight", type=float, default=0.0)
    ap.add_argument("--depth-variant", default="depth_pro", help="Track A variant whose cached maps supervise")
    ap.add_argument("--pred-root", default="data/pred")
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=200_000)
    ap.add_argument("--max-lidar-points", type=int, default=300_000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--max-gaussians", type=int, default=3_000_000)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--img-dir", default="docs/img")
    ap.add_argument("--save-dir", default="data/gs")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    nusc = N.load_nusc()
    scene = args.scene or json.loads(Path("results/scene_dynamics.json").read_text())["selected_scene"]
    name = args.name or (args.variant if args.variant != "lidar_depth" else f"lidar_depth{args.depth_weight:g}")
    data = G.load_scene(nusc, scene, downscale=args.downscale)
    train_views, test_views = data.train_views, data.test_views
    extent = G.scene_scale(data)
    print(f"scene {scene}: {len(train_views)} train / {len(test_views)} test views, extent {extent:.1f} m")

    t_init = time.time()
    if args.variant == "random":
        pts, cols = G.init_random(data, args.n_random, seed=args.seed)
    else:
        pts, cols = G.init_lidar(nusc, data, args.max_lidar_points, seed=args.seed)
    init_time = time.time() - t_init
    params = G.create_gaussians(pts, cols, args.sh_degree, device)
    n_init = len(pts)
    print(f"init {args.variant}: {n_init} points in {init_time:.1f}s")

    optimizers = G.make_optimizers(params, extent)
    from gsplat import DefaultStrategy

    strategy = DefaultStrategy(
        refine_start_iter=500,
        refine_stop_iter=args.steps // 2,
        reset_every=3000,
        refine_every=100,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=extent)
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / args.steps))

    images = {v.sd_token: G.load_image_tensor(v, device) for v in data.views}
    mono = {}
    if args.variant == "lidar_depth":
        for v in train_views:
            d = G.load_mono_depth(Path(args.pred_root), args.depth_variant, v, device)
            if d is not None:
                mono[v.sd_token] = d
        print(f"mono depth maps for {len(mono)}/{len(train_views)} train views ({args.depth_variant})")
        if not mono:
            raise SystemExit("no cached monocular depth found; run eval_mono_depth.py --save-pred-scene first")

    rng = np.random.default_rng(args.seed)
    losses = []
    t0 = time.time()
    torch.cuda.synchronize()
    pbar = tqdm(range(args.steps), desc=name)
    for step in pbar:
        view = train_views[rng.integers(len(train_views))]
        sh_deg = min(step // 1000, args.sh_degree)
        out, alpha, info = G.render(params, view, device, sh_deg, render_mode="RGB+ED")
        rgb = out[0, ..., :3]
        depth = out[0, ..., 3]
        gt = images[view.sd_token]
        l1 = F.l1_loss(rgb, gt)
        ssim_val = G.ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim_val)
        dl = torch.zeros((), device=device)
        if args.variant == "lidar_depth" and view.sd_token in mono:
            dl = G.depth_loss(depth, alpha[0, ..., 0], mono[view.sd_token])
            loss = loss + args.depth_weight * dl
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        if len(params["means"]) < args.max_gaussians:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 50 == 0:
            losses.append([step, float(loss), float(l1), float(ssim_val), float(dl), int(len(params["means"]))])
            pbar.set_postfix(loss=f"{float(loss):.4f}", n=len(params["means"]))
    torch.cuda.synchronize()
    train_time = time.time() - t0
    n_final = int(len(params["means"]))
    print(f"trained {args.steps} steps in {train_time:.1f}s, {n_final} gaussians (NVIDIA L4)")

    # ------------------------------------------------------------ evaluation on held-out views
    import lpips

    lp = lpips.LPIPS(net="alex", verbose=False).to(device)
    per_view = []
    render_times = []
    with torch.no_grad():
        for v in test_views:
            torch.cuda.synchronize()
            tr = time.perf_counter()
            out, alpha, _ = G.render(params, v, device, args.sh_degree, render_mode="RGB+ED")
            torch.cuda.synchronize()
            render_times.append((time.perf_counter() - tr) * 1000)
            rgb = out[0, ..., :3].clamp(0, 1)
            gt = images[v.sd_token]
            a = rgb.permute(2, 0, 1)[None]
            b = gt.permute(2, 0, 1)[None]
            per_view.append(
                {
                    "sd_token": v.sd_token,
                    "t_index": v.t_index,
                    "camera": v.channel,
                    "psnr": G.psnr(rgb, gt),
                    "ssim": float(G.ssim(a, b)),
                    "lpips_alex": float(lp(a * 2 - 1, b * 2 - 1)),
                }
            )
        train_psnr = []
        for v in train_views[:: max(1, len(train_views) // 12)]:
            out, _, _ = G.render(params, v, device, args.sh_degree, render_mode="RGB")
            train_psnr.append(G.psnr(out[0].clamp(0, 1), images[v.sd_token]))
        # previews: first test timestamp, CAM_FRONT and CAM_BACK_LEFT
        t_first = test_views[0].t_index
        for cam in ("CAM_FRONT", "CAM_BACK_LEFT"):
            v = next(x for x in test_views if x.t_index == t_first and x.channel == cam)
            out, _, _ = G.render(params, v, device, args.sh_degree, render_mode="RGB")
            save_side_by_side(images[v.sd_token], out[0], Path(args.img_dir) / f"gs_{name}_{cam.lower()}.jpg")
        if not args.no_video:
            frames_dir = Path(args.save_dir) / f"frames_{name}"
            frames_dir.mkdir(parents=True, exist_ok=True)
            front = sorted([x for x in data.views if x.channel == "CAM_FRONT"], key=lambda x: x.t_index)
            for i, v in enumerate(front):
                out, _, _ = G.render(params, v, device, args.sh_degree, render_mode="RGB")
                arr = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                im = Image.fromarray(arr).resize((480, 270), Image.BILINEAR)
                im.save(frames_dir / f"{i:03d}.png")
            mp4 = Path(args.img_dir) / f"gs_{name}_cam_front_path.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "6", "-i", str(frames_dir / "%03d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", str(mp4)],
                check=False,
            )  # fmt: skip

    def agg(key):
        vals = np.array([p[key] for p in per_view])
        return {
            "mean": float(vals.mean()), "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max())
        }  # fmt: skip

    result = {
        "name": name,
        "variant": args.variant,
        "scene": scene,
        "gpu": "NVIDIA L4",
        "seed": args.seed,
        "steps": args.steps,
        "downscale": args.downscale,
        "image_size": [test_views[0].width, test_views[0].height],
        "n_train_views": len(train_views),
        "n_test_views": len(test_views),
        "holdout": "every 8th timestamp (offset 4), all 6 cameras",
        "depth_weight": args.depth_weight if args.variant == "lidar_depth" else 0.0,
        "depth_variant": args.depth_variant if args.variant == "lidar_depth" else None,
        "n_gaussians_init": n_init,
        "n_gaussians_final": n_final,
        "init_time_s": init_time,
        "train_time_s": train_time,
        "steps_per_s": args.steps / train_time,
        "render_ms_per_test_view": {"mean": float(np.mean(render_times)), "median": float(np.median(render_times))},
        "test": {"psnr": agg("psnr"), "ssim": agg("ssim"), "lpips_alex": agg("lpips_alex")},
        "train_psnr_subset_mean": float(np.mean(train_psnr)),
        "per_view": per_view,
        "loss_curve_columns": ["step", "loss", "l1", "ssim", "depth_loss", "n_gaussians"],
        "loss_curve": losses,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / f"gs_{name}.json").write_text(json.dumps(result, indent=1))
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in params.items()}, Path(args.save_dir) / f"{name}.pt")
    print(json.dumps(result["test"]))
    print("DONE")


if __name__ == "__main__":
    main()
