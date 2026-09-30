"""Sparse LiDAR depth ground truth by projecting LIDAR_TOP into each camera.

The projection chain is lidar -> ego(lidar ts) -> world -> ego(cam ts) -> cam, using the
nuScenes calibrated_sensor and ego_pose records. Points with ``zmin < z < zmax`` that land
inside the image are kept; when several points hit the same pixel the smallest depth wins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ZMIN = 1.0
ZMAX = 80.0


def project_points(
    pts: np.ndarray,
    T_src_to_cam: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    zmin: float = ZMIN,
    zmax: float = ZMAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (N, 3) points into the image; return integer pixel u, v and depth z with z-buffering.

    Returns arrays sorted by pixel index so that each (v, u) appears once with the minimum depth.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {pts.shape}")
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = (T_src_to_cam @ hom.T)[:3]
    z = cam[2]
    keep = (z > zmin) & (z < zmax)
    cam = cam[:, keep]
    z = z[keep]
    if z.size == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    uv = K @ cam
    u = uv[0] / z
    v = uv[1] / z
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui, vi, z = ui[inside], vi[inside], z[inside]
    # z-buffer: sort by (pixel, depth) and keep the first of each pixel
    pix = vi * width + ui
    order = np.lexsort((z, pix))
    pix_s, z_s = pix[order], z[order]
    first = np.ones(len(pix_s), dtype=bool)
    first[1:] = pix_s[1:] != pix_s[:-1]
    pix_u, z_u = pix_s[first], z_s[first]
    return (pix_u % width).astype(np.int32), (pix_u // width).astype(np.int32), z_u.astype(np.float32)


def gt_cache_path(cache_dir: Path, sample_token: str, channel: str) -> Path:
    return Path(cache_dir) / f"{sample_token}_{channel}.npz"


def build_gt(nusc, sample_token: str, channel: str, cache_dir: Path | None = None) -> dict:
    """Project the keyframe LiDAR into ``channel``; optionally cache as npz. Returns dict(u, v, z, width, height)."""
    from depth3d.nusc import cam_frame, lidar_points, lidar_to_cam_matrix

    if cache_dir is not None:
        p = gt_cache_path(cache_dir, sample_token, channel)
        if p.exists():
            d = np.load(p)
            return {k: d[k] for k in d.files}
    frame = cam_frame(nusc, sample_token, channel)
    pts = lidar_points(nusc, sample_token)
    T = lidar_to_cam_matrix(nusc, sample_token, channel)
    u, v, z = project_points(pts, T, frame.K, frame.width, frame.height)
    out = {"u": u, "v": v, "z": z, "width": np.int32(frame.width), "height": np.int32(frame.height)}
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(gt_cache_path(cache_dir, sample_token, channel), **out)
    return out


def dense_from_sparse(gt: dict) -> np.ndarray:
    """Dense (H, W) float32 depth image with 0 where there is no LiDAR return."""
    h, w = int(gt["height"]), int(gt["width"])
    img = np.zeros((h, w), dtype=np.float32)
    img[gt["v"], gt["u"]] = gt["z"]
    return img
