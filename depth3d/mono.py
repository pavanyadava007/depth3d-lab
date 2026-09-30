"""Zero-shot monocular metric depth models behind one small interface.

Every model takes an RGB uint8 image (H, W, 3) and returns a dict of variant name -> metric depth map
(H, W) float32 in metres at the input resolution. A model may return several variants from one forward
pass (Depth Pro: depth from its own focal estimate and depth from the known nuScenes focal).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

DEPTH_PRO_REPO = "apple/DepthPro"
DAV2_REPO = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
MOGE2_REPO = "Ruicheng/moge-2-vitl-normal"


class MonoDepthModel:
    name: str = "base"
    variants: tuple[str, ...] = ()

    def predict(self, image: np.ndarray, f_px: float | None) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @property
    def extra(self) -> dict:
        return {}


class DepthProModel(MonoDepthModel):
    """Apple Depth Pro (metric depth + focal estimate). Two variants from one forward pass."""

    name = "depth_pro"
    variants = ("depth_pro", "depth_pro_gtfocal")

    def __init__(self, device: str = "cuda", precision: torch.dtype = torch.half):
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(DEPTH_PRO_REPO, "depth_pro.pt")
        config = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=ckpt)
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=config, device=torch.device(device), precision=precision
        )
        self.model.eval()
        self.device = device
        self.last_focal_px: float | None = None

    @torch.no_grad()
    def predict(self, image: np.ndarray, f_px: float | None) -> dict[str, np.ndarray]:
        x = self.transform(image)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        _, _, H, W = x.shape
        size = self.model.img_size
        xr = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        canonical_inv, fov_deg = self.model.forward(xr)
        f_est = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        self.last_focal_px = float(f_est.squeeze())
        out = {}
        focals = {"depth_pro": f_est}
        if f_px is not None:
            focals["depth_pro_gtfocal"] = torch.tensor(float(f_px), device=canonical_inv.device)
        for key, f in focals.items():
            inv = canonical_inv.float() * (W / f)
            inv = torch.nn.functional.interpolate(inv, size=(H, W), mode="bilinear", align_corners=False)
            depth = 1.0 / torch.clamp(inv, min=1e-4, max=1e4)
            out[key] = depth.squeeze().float().cpu().numpy()
        return out

    @property
    def extra(self) -> dict:
        return {"pred_focal_px": self.last_focal_px}


class DepthAnythingV2Metric(MonoDepthModel):
    """Depth Anything V2 metric, outdoor (VKITTI fine-tuned), Large, via transformers."""

    name = "dav2_metric_outdoor_large"
    variants = ("dav2_metric_outdoor_large",)

    def __init__(self, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(DAV2_REPO)
        self.model = AutoModelForDepthEstimation.from_pretrained(DAV2_REPO).to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, image: np.ndarray, f_px: float | None) -> dict[str, np.ndarray]:
        H, W = image.shape[:2]
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        post = self.processor.post_process_depth_estimation(outputs, target_sizes=[(H, W)])
        depth = post[0]["predicted_depth"].float().cpu().numpy()
        return {self.name: depth}


class MoGe2Model(MonoDepthModel):
    """Microsoft MoGe-2 (metric). Two variants: own focal estimate and known focal (fov_x)."""

    name = "moge2"
    variants = ("moge2", "moge2_gtfocal")

    def __init__(self, device: str = "cuda"):
        from moge.model.v2 import MoGeModel

        self.model = MoGeModel.from_pretrained(MOGE2_REPO).to(device).eval()
        self.device = device
        self.last_focal_px: float | None = None

    @torch.no_grad()
    def predict(self, image: np.ndarray, f_px: float | None) -> dict[str, np.ndarray]:
        H, W = image.shape[:2]
        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float() / 255.0
        out = {}
        res = self.model.infer(x)
        out["moge2"] = res["depth"].float().cpu().numpy()
        fx = float(res["intrinsics"][0, 0]) * W
        self.last_focal_px = fx
        if f_px is not None:
            fov_x = float(np.degrees(2 * np.arctan(0.5 * W / f_px)))
            res2 = self.model.infer(x, fov_x=fov_x)
            out["moge2_gtfocal"] = res2["depth"].float().cpu().numpy()
        return out

    @property
    def extra(self) -> dict:
        return {"pred_focal_px": self.last_focal_px}


REGISTRY = {
    "depth_pro": DepthProModel,
    "dav2_metric_outdoor_large": DepthAnythingV2Metric,
    "moge2": MoGe2Model,
}


def load_model(name: str, device: str = "cuda") -> MonoDepthModel:
    return REGISTRY[name](device=device)


def pred_cache_path(root: Path, variant: str, sd_token: str) -> Path:
    return Path(root) / variant / f"{sd_token}.npz"
