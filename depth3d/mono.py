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
    """Microsoft MoGe-2 (metric). Two variants from ONE forward pass: own focal estimate and known focal (fov_x).

    The post-processing mirrors ``MoGeModel.infer`` (focal / shift recovery from the affine point map, metric
    scale) but is applied twice to the same network output, like the Depth Pro wrapper, so that the timing is
    one forward per image. No validity mask is applied (``apply_mask=False`` behaviour): sky-like pixels keep the
    model's raw depth and are clipped to the scoring range like for the other models (KITTI protocol).
    scripts/check_moge2_focal.py compares both variants against the library's own ``infer``.
    """

    name = "moge2"
    variants = ("moge2", "moge2_gtfocal")

    def __init__(self, device: str = "cuda", resolution_level: int = 9):
        from moge.model.v2 import MoGeModel

        self.model = MoGeModel.from_pretrained(MOGE2_REPO).to(device).eval()
        self.device = device
        lo, hi = self.model.num_tokens_range
        self.num_tokens = int(lo + (resolution_level / 9) * (hi - lo))
        self.last_focal_px: float | None = None

    @torch.no_grad()
    def _forward(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(dtype=self.model.dtype)
        use_fp16 = self.model.dtype != torch.float16
        with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=use_fp16):
            out = self.model.forward(x, num_tokens=self.num_tokens)
        points = out["points"].float()
        mask = out["mask"].float() > 0.5 if out.get("mask") is not None else None
        return points, mask, out["metric_scale"].float()

    @torch.no_grad()
    def _depth(self, points: torch.Tensor, mask: torch.Tensor | None, metric_scale: torch.Tensor, aspect: float,
               fov_x: float | None) -> tuple[torch.Tensor, float]:
        """Depth (1, H, W) in metres and normalised fx (relative to the image width), as in MoGeModel.infer."""
        from moge.utils.geometry_torch import recover_focal_shift

        with torch.autocast(device_type=self.model.device.type, dtype=torch.float32):
            if fov_x is None:
                focal, shift = recover_focal_shift(points, mask)
            else:
                focal = aspect / (1 + aspect**2) ** 0.5 / torch.tan(
                    torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2)
                )
                if focal.ndim == 0:
                    focal = focal[None].expand(points.shape[0])
                _, shift = recover_focal_shift(points, mask, focal=focal)
            fx = focal / 2 * (1 + aspect**2) ** 0.5 / aspect
            depth = (points[..., 2] + shift[..., None, None]) * metric_scale[:, None, None]
        return depth, float(fx.squeeze())

    @torch.no_grad()
    def predict(self, image: np.ndarray, f_px: float | None) -> dict[str, np.ndarray]:
        H, W = image.shape[:2]
        aspect = W / H
        points, mask, metric_scale = self._forward(image)
        depth, fx = self._depth(points, mask, metric_scale, aspect, None)
        self.last_focal_px = fx * W
        out = {"moge2": depth.squeeze(0).float().cpu().numpy()}
        if f_px is not None:
            fov_x = float(np.degrees(2 * np.arctan(0.5 * W / f_px)))
            depth2, _ = self._depth(points, mask, metric_scale, aspect, fov_x)
            out["moge2_gtfocal"] = depth2.squeeze(0).float().cpu().numpy()
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
