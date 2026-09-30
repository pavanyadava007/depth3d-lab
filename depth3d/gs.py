"""3D Gaussian Splatting on one nuScenes scene with gsplat: data, initialisation, losses, evaluation.

World frame: nuScenes global frame re-centred at the mean camera position so that coordinates stay
small. Cameras follow the OpenCV convention (x right, y down, z forward), which is what gsplat expects
for its world-to-camera ``viewmats``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from depth3d import nusc as N
from depth3d.lidar_gt import project_points


@dataclass
class View:
    sd_token: str
    sample_token: str
    channel: str
    t_index: int
    K: np.ndarray  # 3x3 at the training resolution
    T_world_to_cam: np.ndarray  # 4x4 (re-centred world)
    T_cam_to_world: np.ndarray
    image_path: Path
    width: int
    height: int
    is_test: bool


@dataclass
class SceneData:
    scene_name: str
    views: list[View]
    center: np.ndarray  # world offset subtracted from all positions
    sample_tokens: list[str]
    downscale: int

    @property
    def train_views(self) -> list[View]:
        return [v for v in self.views if not v.is_test]

    @property
    def test_views(self) -> list[View]:
        return [v for v in self.views if v.is_test]


def load_scene(nusc, scene_name: str, downscale: int = 2, holdout_every: int = 8, holdout_offset: int = 4) -> SceneData:
    tokens = N.scene_sample_tokens(nusc, scene_name)
    frames = [(i, tok, N.cam_frame(nusc, tok, cam)) for i, tok in enumerate(tokens) for cam in N.CAMERAS]
    positions = np.stack([f.T_cam_to_world[:3, 3] for _, _, f in frames])
    center = positions.mean(0)
    views = []
    for i, tok, f in frames:
        T_c2w = f.T_cam_to_world.copy()
        T_c2w[:3, 3] -= center
        K = f.K.copy()
        K[:2] /= downscale
        views.append(
            View(
                sd_token=f.sd_token,
                sample_token=tok,
                channel=f.channel,
                t_index=i,
                K=K,
                T_world_to_cam=np.linalg.inv(T_c2w),
                T_cam_to_world=T_c2w,
                image_path=f.path,
                width=f.width // downscale,
                height=f.height // downscale,
                is_test=(i % holdout_every) == holdout_offset,
            )
        )
    return SceneData(scene_name, views, center, tokens, downscale)


def load_image_tensor(view: View, device: str) -> torch.Tensor:
    img = Image.open(view.image_path).convert("RGB")
    if img.size != (view.width, view.height):
        img = img.resize((view.width, view.height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def scene_scale(data: SceneData) -> float:
    pos = np.stack([v.T_cam_to_world[:3, 3] for v in data.views])
    return float(np.max(np.linalg.norm(pos - pos.mean(0), axis=1)) * 1.1)


# ---------------------------------------------------------------- initialisation


def init_random(
    data: SceneData, n_points: int, seed: int = 0, margin_xy: float = 40.0
) -> tuple[np.ndarray, np.ndarray]:
    """SfM-free init: uniform points in a box around the driven trajectory (the 'driving frustum')."""
    rng = np.random.default_rng(seed)
    pos = np.stack([v.T_cam_to_world[:3, 3] for v in data.views])
    lo = pos.min(0) - np.array([margin_xy, margin_xy, 3.0])
    hi = pos.max(0) + np.array([margin_xy, margin_xy, 12.0])
    pts = rng.uniform(lo, hi, size=(n_points, 3))
    cols = rng.uniform(0.2, 0.8, size=(n_points, 3))
    return pts.astype(np.float32), cols.astype(np.float32)


def init_lidar(
    nusc, data: SceneData, max_points: int, voxel: float = 0.15, seed: int = 0, zmax: float = 80.0
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate keyframe LiDAR of the TRAIN timestamps in the re-centred world frame, coloured by projection."""
    rng = np.random.default_rng(seed)
    train_t = sorted({v.t_index for v in data.train_views})
    all_pts, all_cols = [], []
    for ti in train_t:
        tok = data.sample_tokens[ti]
        pts_l = N.lidar_points(nusc, tok)
        T_l2w = N.sensor_to_world(nusc, nusc.get("sample", tok)["data"]["LIDAR_TOP"])
        rng_ok = np.linalg.norm(pts_l, axis=1)
        pts_l = pts_l[(rng_ok > 1.0) & (rng_ok < zmax)]
        pts_w = (T_l2w[:3, :3] @ pts_l.T).T + T_l2w[:3, 3] - data.center
        cols = np.full((len(pts_w), 3), np.nan, dtype=np.float32)
        views_t = [v for v in data.views if v.t_index == ti]
        for v in views_t:
            img = np.asarray(Image.open(v.image_path).convert("RGB").resize((v.width, v.height)), dtype=np.float32)
            hom = np.concatenate([pts_w, np.ones((len(pts_w), 1))], 1)
            cam = (v.T_world_to_cam @ hom.T)[:3]
            z = cam[2]
            uv = v.K @ cam
            u = uv[0] / np.where(z > 0, z, 1)
            vv = uv[1] / np.where(z > 0, z, 1)
            ok = (z > 0.5) & (u >= 0) & (u < v.width - 1) & (vv >= 0) & (vv < v.height - 1) & np.isnan(cols[:, 0])
            cols[ok] = img[vv[ok].astype(int), u[ok].astype(int)] / 255.0
        keep = ~np.isnan(cols[:, 0])
        all_pts.append(pts_w[keep].astype(np.float32))
        all_cols.append(cols[keep])
    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)
    # voxel downsample: keep one random point per voxel
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    pts, cols = pts[idx], cols[idx]
    if len(pts) > max_points:
        sel = rng.choice(len(pts), max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    return pts, cols


# ---------------------------------------------------------------- gaussians


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / 0.28209479177387814


def knn_mean_dist(points: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    """Mean distance to the k nearest neighbours (excluding self), computed in chunks on the GPU."""
    out = torch.empty(len(points), device=points.device)
    for i in range(0, len(points), chunk):
        d = torch.cdist(points[i : i + chunk], points)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        out[i : i + chunk] = d.topk(k, largest=False).values.mean(1)
    return out


def create_gaussians(points: np.ndarray, colors: np.ndarray, sh_degree: int, device: str, init_opacity: float = 0.1):
    means = torch.from_numpy(points).float().to(device)
    rgb = torch.from_numpy(colors).float().to(device)
    dist = knn_mean_dist(means).clamp(min=1e-3, max=5.0)
    scales = torch.log(dist).unsqueeze(-1).repeat(1, 3)
    n = len(means)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), init_opacity, device=device))
    colors_sh = torch.zeros(n, (sh_degree + 1) ** 2, 3, device=device)
    colors_sh[:, 0] = rgb_to_sh0(rgb)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(colors_sh[:, :1]),
            "shN": torch.nn.Parameter(colors_sh[:, 1:]),
        }
    )
    return params


def make_optimizers(params, scene_extent: float, batch_size: int = 1) -> dict[str, torch.optim.Optimizer]:
    lrs = {
        "means": 1.6e-4 * scene_extent,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    scale = math.sqrt(batch_size)
    return {
        name: torch.optim.Adam([{"params": params[name], "lr": lr * scale, "name": name}], eps=1e-15 / scale)
        for name, lr in lrs.items()
    }


def render(params, view: View, device: str, sh_degree: int, render_mode: str = "RGB+ED", bg: float = 0.0):
    from gsplat import rasterization

    viewmat = torch.from_numpy(view.T_world_to_cam).float().to(device)[None]
    K = torch.from_numpy(view.K).float().to(device)[None]
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=view.width,
        height=view.height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        packed=False,
        absgrad=False,
        rasterize_mode="classic",
        backgrounds=torch.full((1, 3), bg, device=device) if render_mode.startswith("RGB") else None,
    )
    return out, alpha, info


# ---------------------------------------------------------------- losses / metrics


def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(img1: torch.Tensor, img2: torch.Tensor, window: int = 11) -> torch.Tensor:
    """SSIM for (1, C, H, W) tensors in [0, 1] with an 11x11 Gaussian window, sigma 1.5."""
    C = img1.shape[1]
    w = _gaussian_window(window, 1.5, img1.device).repeat(C, 1, 1, 1)
    mu1 = F.conv2d(img1, w, padding=window // 2, groups=C)
    mu2 = F.conv2d(img2, w, padding=window // 2, groups=C)
    s11 = F.conv2d(img1 * img1, w, padding=window // 2, groups=C) - mu1**2
    s22 = F.conv2d(img2 * img2, w, padding=window // 2, groups=C) - mu2**2
    s12 = F.conv2d(img1 * img2, w, padding=window // 2, groups=C) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s11 + s22 + c2))
    return m.mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return float(10 * math.log10(1.0 / max(mse, 1e-10)))


def depth_loss(rendered_depth: torch.Tensor, alpha: torch.Tensor, target: torch.Tensor, max_depth: float = 80.0):
    """Relative L1 between rendered expected depth and a (scale-aligned) monocular target on valid pixels."""
    valid = (alpha > 0.5) & torch.isfinite(target) & (target > 0.5) & (target < max_depth)
    if valid.sum() == 0:
        return rendered_depth.sum() * 0.0
    return (torch.abs(rendered_depth[valid] - target[valid]) / target[valid]).mean()


def load_mono_depth(pred_root: Path, variant: str, view: View, device: str) -> torch.Tensor | None:
    p = Path(pred_root) / variant / f"{view.sd_token}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    depth = d["depth"].astype(np.float32) * float(d["scale_to_lidar"])
    t = torch.from_numpy(depth).to(device)
    if t.shape != (view.height, view.width):
        t = F.interpolate(t[None, None], size=(view.height, view.width), mode="nearest")[0, 0]
    return t


def lidar_depth_for_view(nusc, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse LiDAR (u, v, z) at the training resolution for one view (for reporting only)."""
    tok = view.sample_token
    pts_l = N.lidar_points(nusc, tok)
    T = N.lidar_to_cam_matrix(nusc, tok, view.channel)
    return project_points(pts_l, T, view.K, view.width, view.height)
