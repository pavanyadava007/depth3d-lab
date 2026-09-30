"""VGGT wrapper: feed-forward multi-view depth + cameras from a list of images."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

VGGT_REPO = "facebook/VGGT-1B"
VGGT_W, VGGT_H = 518, 294  # 1600x900 -> 518 wide, height rounded to a multiple of 14


class VGGTModel:
    def __init__(self, device: str = "cuda"):
        from vggt.models.vggt import VGGT

        try:
            self.model = VGGT.from_pretrained(VGGT_REPO)
        except Exception:  # noqa: BLE001 - fall back to the raw checkpoint file
            from huggingface_hub import hf_hub_download

            self.model = VGGT()
            self.model.load_state_dict(torch.load(hf_hub_download(VGGT_REPO, "model.pt"), map_location="cpu"))
        self.model = self.model.to(device).eval()
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    @staticmethod
    def preprocess(paths: list) -> torch.Tensor:
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB").resize((VGGT_W, VGGT_H), Image.BICUBIC)
            imgs.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1))
        return torch.stack(imgs)

    @torch.no_grad()
    def infer(self, paths: list) -> dict:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = self.preprocess(paths).to(self.device)
        with torch.autocast("cuda", dtype=self.dtype):
            preds = self.model(images[None])
        extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
        return {
            "depth": preds["depth"][0, ..., 0].float().cpu().numpy(),  # (S, H, W), first camera frame scale
            "depth_conf": preds["depth_conf"][0].float().cpu().numpy(),
            "extrinsics": extri[0].float().cpu().numpy(),  # (S, 3, 4) camera-from-world, OpenCV
            "intrinsics": intri[0].float().cpu().numpy(),  # (S, 3, 3) at 518x294
            "images": images.cpu().numpy(),
        }
