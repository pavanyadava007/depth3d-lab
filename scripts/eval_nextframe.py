"""Track E evaluation on the two held-out scenes: flow-matching sample vs regression UNet vs copy baseline.

Metrics per test pair: PSNR and LPIPS (alex) of the predicted next frame against the real next frame.
Optionally FID of unconditional samples (condition zeroed, as in training with condition dropout) against
the held-out real frames, via torchmetrics if it is installed. Writes results/synthesis_nextframe.json and
docs/img/nextframe_grid.jpg (2 x 4 real vs generated examples at 128x64).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from depth3d.metrics import bootstrap_mean_ci, paired_bootstrap
from depth3d.nextframe import HOLDOUT_SCENES, UNet, fm_sample, load_pairs, psnr_batch, regression_predict, to_tensor


def load(path: str, device: str) -> UNet:
    ck = torch.load(path, map_location=device)
    m = UNet(base=ck["args"]["base"]).to(device).eval()
    m.load_state_dict(ck["ema"])
    return m


def to_u8(x: torch.Tensor) -> np.ndarray:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nextframe/frames.npz")
    ap.add_argument("--fm", default="data/nextframe/fm.pt")
    ap.add_argument("--reg", default="data/nextframe/regression.pt")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fid", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    frames, _, test_idx = load_pairs(args.data)
    frames_t = to_tensor(frames, device)
    fm = load(args.fm, device)
    reg = load(args.reg, device) if Path(args.reg).exists() else None
    import lpips

    lp = lpips.LPIPS(net="alex", verbose=False).to(device)
    res = {k: {"psnr": [], "lpips": []} for k in ("fm_sample", "regression", "copy_current")}
    t_sample = 0.0
    grid_real, grid_gen, grid_cur = [], [], []
    for i in range(0, len(test_idx), args.batch):
        b = test_idx[i : i + args.batch]
        cond = frames_t[b]
        real = frames_t[b + 1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sample = fm_sample(fm, cond, steps=args.steps, seed=args.seed + i)
        torch.cuda.synchronize()
        t_sample += time.perf_counter() - t0
        preds = {"fm_sample": sample, "copy_current": cond}
        if reg is not None:
            preds["regression"] = regression_predict(reg, cond)
        for k, p in preds.items():
            res[k]["psnr"].extend(psnr_batch(p, real).tolist())
            res[k]["lpips"].extend(lp(p, real).flatten().tolist())
        if i == 0:
            sel = np.linspace(0, len(b) - 1, 4).astype(int)
            grid_real, grid_gen, grid_cur = to_u8(real[sel]), to_u8(sample[sel]), to_u8(cond[sel])
    if reg is None:
        del res["regression"]
    out = {
        "dataset": "nuScenes v1.0-mini CAM_FRONT keyframes + sweeps, 128x64, consecutive pairs (0.1 s)",
        "holdout_scenes": list(HOLDOUT_SCENES),
        "n_test_pairs": int(len(test_idx)),
        "gpu": "NVIDIA L4",
        "sampler": f"Euler, {args.steps} steps",
        "sampling_ms_per_frame": 1000.0 * t_sample / len(test_idx),
        "methods": {},
    }
    for k, d in res.items():
        out["methods"][k] = {
            "psnr": bootstrap_mean_ci(np.array(d["psnr"])),
            "lpips_alex": bootstrap_mean_ci(np.array(d["lpips"])),
        }
    if "regression" in res:
        out["paired_lpips_fm_minus_regression"] = paired_bootstrap(
            np.array(res["fm_sample"]["lpips"]), np.array(res["regression"]["lpips"])
        )
    out["paired_lpips_fm_minus_copy"] = paired_bootstrap(
        np.array(res["fm_sample"]["lpips"]), np.array(res["copy_current"]["lpips"])
    )
    out["paired_psnr_fm_minus_copy"] = paired_bootstrap(
        np.array(res["fm_sample"]["psnr"]), np.array(res["copy_current"]["psnr"])
    )
    if args.fid:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance

            fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            real_all = frames_t[test_idx + 1]
            for i in range(0, len(real_all), 128):
                fid.update((real_all[i : i + 128] + 1) / 2, real=True)
            n_gen = len(test_idx)
            for i in range(0, n_gen, 128):
                nb = min(128, n_gen - i)
                zeros = torch.zeros(nb, 3, 64, 128, device=device)
                g = fm_sample(fm, zeros, steps=args.steps, seed=1000 + i)
                fid.update((g + 1) / 2, real=False)
            out["fid_unconditional"] = {
                "value": float(fid.compute()),
                "n_real": int(len(real_all)),
                "n_generated": int(n_gen),
                "note": "unconditional samples (condition zeroed) vs held-out real frames, 128x64 upsampled "
                "to 299 by torchmetrics; small n, treat as indicative only",
            }
        except Exception as e:  # noqa: BLE001
            out["fid_unconditional"] = {"value": None, "note": f"not measured: {type(e).__name__}: {e}"}
    else:
        out["fid_unconditional"] = {"value": None, "note": "not measured (run with --fid)"}
    Path("results").mkdir(exist_ok=True)
    Path("results/synthesis_nextframe.json").write_text(json.dumps(out, indent=1))
    # grid: row 1 real next frames, row 2 generated (with the current frame on the left as reference)
    rows = []
    for imgs in (grid_real, grid_gen):
        rows.append(np.concatenate(list(imgs), axis=1))
    grid = np.concatenate(rows, axis=0)
    cur = np.concatenate(list(grid_cur), axis=1)
    full = np.concatenate([cur, grid], axis=0)
    Path("docs/img").mkdir(parents=True, exist_ok=True)
    Image.fromarray(full).resize((full.shape[1] * 2, full.shape[0] * 2), Image.NEAREST).save(
        "docs/img/nextframe_grid.jpg", "JPEG", quality=88
    )
    print(json.dumps({k: {m: round(v["mean"], 4) for m, v in d.items()} for k, d in out["methods"].items()}))
    print("DONE")


if __name__ == "__main__":
    main()
