"""Count moving objects per nuScenes-mini scene and pick the most static scene for 3DGS.

An annotation counts as moving when the devkit box velocity (finite differences over neighbouring
keyframes) has a norm above 0.5 m/s. The attribute-based count (vehicle.moving, pedestrian.moving,
cycle.with_rider) is reported alongside. Output: results/scene_dynamics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth3d import nusc as N

VEL_THRESH = 0.5
MOVING_ATTRS = {"vehicle.moving", "pedestrian.moving", "cycle.with_rider"}


def main() -> None:
    nusc = N.load_nusc()
    out = {}
    for scene in nusc.scene:
        toks = N.scene_sample_tokens(nusc, scene["name"])
        n_ann = n_moving = n_attr_moving = 0
        moving_instances = set()
        for tok in toks:
            sample = nusc.get("sample", tok)
            for ann_tok in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_tok)
                n_ann += 1
                v = nusc.box_velocity(ann_tok)
                if np.all(np.isfinite(v)) and float(np.linalg.norm(v)) > VEL_THRESH:
                    n_moving += 1
                    moving_instances.add(ann["instance_token"])
                attrs = {nusc.get("attribute", a)["name"] for a in ann["attribute_tokens"]}
                if attrs & MOVING_ATTRS:
                    n_attr_moving += 1
        out[scene["name"]] = {
            "n_keyframes": len(toks),
            "n_annotations": n_ann,
            "n_moving_annotations_vel": n_moving,
            "n_moving_instances_vel": len(moving_instances),
            "frac_moving_vel": n_moving / max(n_ann, 1),
            "n_moving_annotations_attr": n_attr_moving,
            "description": scene["description"],
        }
    ranked = sorted(out, key=lambda s: (out[s]["n_moving_annotations_vel"], out[s]["n_annotations"]))
    result = {
        "velocity_threshold_mps": VEL_THRESH,
        "moving_attributes": sorted(MOVING_ATTRS),
        "scenes": out,
        "ranking_fewest_moving_first": ranked,
        "selected_scene": ranked[0],
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/scene_dynamics.json").write_text(json.dumps(result, indent=2))
    for s in ranked:
        d = out[s]
        print(s, d["n_moving_annotations_vel"], d["n_annotations"], d["description"][:60])
    print("selected", ranked[0])


if __name__ == "__main__":
    main()
