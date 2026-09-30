"""Sanity check of the Depth Pro known-focal path (CPU): compare our two-variant predict() with the library's
model.infer(x) and model.infer(x, f_px) on one keyframe of CAM_FRONT and CAM_BACK, and log focal estimates.

f_px must be in pixels of the image passed to the model (nuScenes: 1600 px wide), because Depth Pro's infer
takes W from the input before resizing to its 1536x1536 network resolution. Writes results/depth_pro_focal_check.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from depth3d import nusc as N
from depth3d.mono import DepthProModel


def main() -> None:
    nusc = N.load_nusc()
    tok = N.scene_sample_tokens(nusc, "scene-0061")[5]
    model = DepthProModel(device="cpu", precision=torch.float32)
    out = {"sample_token": tok, "device": "cpu fp32", "cameras": {}}
    for cam in ("CAM_FRONT", "CAM_BACK"):
        fr = N.cam_frame(nusc, tok, cam)
        img = np.asarray(Image.open(fr.path).convert("RGB"))
        f_true = float(fr.K[0, 0])
        ours = model.predict(img, f_true)
        x = model.transform(img)
        ref_est = model.model.infer(x)
        ref_gt = model.model.infer(x, f_px=torch.tensor(f_true))
        d_est = np.abs(ours["depth_pro"] - ref_est["depth"].squeeze().numpy()).max()
        d_gt = np.abs(ours["depth_pro_gtfocal"] - ref_gt["depth"].squeeze().numpy()).max()
        out["cameras"][cam] = {
            "image_width_px": int(img.shape[1]),
            "true_focal_px": f_true,
            "library_estimated_focal_px": float(ref_est["focallength_px"]),
            "ours_estimated_focal_px": model.last_focal_px,
            "max_abs_diff_estimated_focal_variant_m": float(d_est),
            "max_abs_diff_known_focal_variant_m": float(d_gt),
            "median_depth_estimated_focal_m": float(np.median(ours["depth_pro"])),
            "median_depth_known_focal_m": float(np.median(ours["depth_pro_gtfocal"])),
            "true_over_estimated_focal": f_true / float(ref_est["focallength_px"]),
        }
    Path("results").mkdir(exist_ok=True)
    Path("results/depth_pro_focal_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
