"""Sanity check of the MoGe-2 wrapper: our one-forward two-variant predict() against the library's own
model.infer(x) and model.infer(x, fov_x=...) (apply_mask=False) on one keyframe of CAM_FRONT and CAM_BACK.
Also logs the focal estimate. Writes results/moge2_focal_check.json. Runs on the GPU (fp16 autocast forward,
fp32 post-processing, exactly as the library's infer does).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from depth3d import nusc as N
from depth3d.mono import MoGe2Model


def main() -> None:
    nusc = N.load_nusc()
    tok = N.scene_sample_tokens(nusc, "scene-0061")[5]
    model = MoGe2Model(device="cuda")
    out = {"sample_token": tok, "device": "NVIDIA L4, fp16 autocast forward, fp32 post-processing", "cameras": {}}
    for cam in ("CAM_FRONT", "CAM_BACK"):
        fr = N.cam_frame(nusc, tok, cam)
        img = np.asarray(Image.open(fr.path).convert("RGB"))
        W = img.shape[1]
        f_true = float(fr.K[0, 0])
        ours = model.predict(img, f_true)
        x = torch.from_numpy(img).to("cuda").permute(2, 0, 1).float() / 255.0
        ref_est = model.model.infer(x, apply_mask=False)
        fov_x = float(np.degrees(2 * np.arctan(0.5 * W / f_true)))
        ref_gt = model.model.infer(x, fov_x=fov_x, apply_mask=False)
        d_est = np.abs(ours["moge2"] - ref_est["depth"].float().cpu().numpy()).max()
        d_gt = np.abs(ours["moge2_gtfocal"] - ref_gt["depth"].float().cpu().numpy()).max()
        lib_f = float(ref_est["intrinsics"][0, 0]) * W
        out["cameras"][cam] = {
            "image_width_px": int(W),
            "true_focal_px": f_true,
            "library_estimated_focal_px": lib_f,
            "ours_estimated_focal_px": model.last_focal_px,
            "max_abs_diff_estimated_focal_variant_m": float(d_est),
            "max_abs_diff_known_focal_variant_m": float(d_gt),
            "median_depth_estimated_focal_m": float(np.median(ours["moge2"])),
            "median_depth_known_focal_m": float(np.median(ours["moge2_gtfocal"])),
            "true_over_estimated_focal": f_true / lib_f,
            "mask_frac_valid": float(ref_est["mask"].float().mean()) if "mask" in ref_est else None,
        }
    Path("results").mkdir(exist_ok=True)
    Path("results/moge2_focal_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
