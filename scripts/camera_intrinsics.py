"""Principal point offset and focal length of the nuScenes-mini cameras.

Depth Pro and SHARP assume the principal point at the image centre. This script measures how far the
calibrated principal points are from the centre of the 1600x900 images, per camera channel and overall.
Output: results/camera_intrinsics.json (CPU only, no images read).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth3d import nusc as N

W, H = 1600, 900


def main() -> None:
    nusc = N.load_nusc()
    per: dict[str, list[tuple[float, float, float]]] = {}
    for cs in nusc.calibrated_sensor:
        K = np.array(cs["camera_intrinsic"], dtype=float)
        if K.size == 0:
            continue
        ch = nusc.get("sensor", cs["sensor_token"])["channel"]
        per.setdefault(ch, []).append((K[0, 2] - W / 2, K[1, 2] - H / 2, K[0, 0]))
    out: dict = {"image_size": [W, H], "per_camera": {}}
    allv = []
    for ch in sorted(per):
        v = np.array(per[ch])
        allv.append(v)
        out["per_camera"][ch] = {
            "n_calibrations": int(len(v)),
            "dx_mean_px": float(v[:, 0].mean()),
            "dy_mean_px": float(v[:, 1].mean()),
            "focal_mean_px": float(v[:, 2].mean()),
        }
    a = np.concatenate(allv)
    out["overall"] = {
        "n_calibrations": int(len(a)),
        "abs_dx_mean_px": float(np.abs(a[:, 0]).mean()),
        "abs_dy_mean_px": float(np.abs(a[:, 1]).mean()),
        "abs_dx_max_px": float(np.abs(a[:, 0]).max()),
        "abs_dy_max_px": float(np.abs(a[:, 1]).max()),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/camera_intrinsics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["overall"]))


if __name__ == "__main__":
    main()
