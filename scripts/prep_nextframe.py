"""Track E data: CAM_FRONT keyframes + sweeps of nuScenes-mini, resized to 128x64, as consecutive pairs.

Writes data/nextframe/frames.npz (uint8 frames, scene ids, timestamps) - gitignored. The two held-out
scenes are fixed here so that every script agrees.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from depth3d import nusc as N
from depth3d.nextframe import HOLDOUT_SCENES

W, H = 128, 64


def main() -> None:
    nusc = N.load_nusc()
    root = N.data_root()
    scenes = {s["token"]: s["name"] for s in nusc.scene}
    frames = []
    for sd in nusc.sample_data:
        if sd["channel"] != "CAM_FRONT":
            continue
        sample = nusc.get("sample", sd["sample_token"])
        frames.append((scenes[sample["scene_token"]], sd["timestamp"], sd["filename"], sd["is_key_frame"]))
    frames.sort()
    t0 = time.time()
    imgs = np.zeros((len(frames), H, W, 3), dtype=np.uint8)
    for i, (_, _, fn, _) in enumerate(tqdm(frames, desc="resize")):
        imgs[i] = np.asarray(Image.open(root / fn).convert("RGB").resize((W, H), Image.BOX))
    scene_names = np.array([f[0] for f in frames])
    ts = np.array([f[1] for f in frames], dtype=np.int64)
    is_kf = np.array([f[3] for f in frames])
    out = Path("data/nextframe")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "frames.npz", frames=imgs, scene=scene_names, timestamp=ts, is_keyframe=is_kf)
    # consecutive pairs inside a scene
    same = scene_names[1:] == scene_names[:-1]
    dt = np.diff(ts) / 1e6
    n_pairs = int(same.sum())
    holdout = np.isin(scene_names[:-1], HOLDOUT_SCENES) & same
    meta = {
        "n_frames": len(frames),
        "n_keyframes": int(is_kf.sum()),
        "size_wh": [W, H],
        "n_pairs": n_pairs,
        "n_train_pairs": int((same & ~np.isin(scene_names[:-1], HOLDOUT_SCENES)).sum()),
        "n_test_pairs": int(holdout.sum()),
        "holdout_scenes": list(HOLDOUT_SCENES),
        "pair_dt_s_median": float(np.median(dt[same])),
        "pair_dt_s_p90": float(np.quantile(dt[same], 0.9)),
        "prep_time_s": time.time() - t0,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/nextframe_data.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
