"""Depth metrics, scale alignment, pose errors and bootstrap confidence intervals (numpy only)."""

from __future__ import annotations

import numpy as np

DEPTH_BINS = [(1.0, 20.0), (20.0, 40.0), (40.0, 80.0)]


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Standard KITTI-style metrics on paired 1-D arrays of positive depths."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 1:
        raise ValueError("pred and gt must be 1-D arrays of equal length")
    if gt.size == 0:
        return {k: float("nan") for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]}
    pred = np.clip(pred, 1e-3, None)
    thresh = np.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
        "sq_rel": float(np.mean((pred - gt) ** 2 / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "d1": float(np.mean(thresh < 1.25)),
        "d2": float(np.mean(thresh < 1.25**2)),
        "d3": float(np.mean(thresh < 1.25**3)),
    }


def median_scale(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-image scalar s = median(gt) / median(pred) (the usual 'median scaling')."""
    pred = np.asarray(pred, dtype=np.float64)
    m = np.median(pred)
    if not np.isfinite(m) or m <= 0:
        return float("nan")
    return float(np.median(gt) / m)


def bin_masks(gt: np.ndarray, bins=DEPTH_BINS) -> dict[str, np.ndarray]:
    return {f"{int(lo)}-{int(hi)}m": (gt >= lo) & (gt < hi) for lo, hi in bins}


class MetricAccumulator:
    """Accumulates sums so that metrics over all points (not mean-of-image-means) can be computed."""

    KEYS = ["abs_rel", "sq_rel", "sq_err", "sq_log_err", "d1", "d2", "d3"]

    def __init__(self) -> None:
        self.sums = dict.fromkeys(self.KEYS, 0.0)
        self.n = 0

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred = np.clip(np.asarray(pred, dtype=np.float64), 1e-3, None)
        gt = np.asarray(gt, dtype=np.float64)
        if gt.size == 0:
            return
        thresh = np.maximum(pred / gt, gt / pred)
        self.sums["abs_rel"] += float(np.sum(np.abs(pred - gt) / gt))
        self.sums["sq_rel"] += float(np.sum((pred - gt) ** 2 / gt))
        self.sums["sq_err"] += float(np.sum((pred - gt) ** 2))
        self.sums["sq_log_err"] += float(np.sum((np.log(pred) - np.log(gt)) ** 2))
        self.sums["d1"] += float(np.sum(thresh < 1.25))
        self.sums["d2"] += float(np.sum(thresh < 1.25**2))
        self.sums["d3"] += float(np.sum(thresh < 1.25**3))
        self.n += int(gt.size)

    def result(self) -> dict[str, float]:
        if self.n == 0:
            return {"n_points": 0}
        n = self.n
        return {
            "n_points": n,
            "abs_rel": self.sums["abs_rel"] / n,
            "sq_rel": self.sums["sq_rel"] / n,
            "rmse": float(np.sqrt(self.sums["sq_err"] / n)),
            "rmse_log": float(np.sqrt(self.sums["sq_log_err"] / n)),
            "d1": self.sums["d1"] / n,
            "d2": self.sums["d2"] / n,
            "d3": self.sums["d3"] / n,
        }


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI of the mean over images."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(values.size),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap CI over paired per-image differences a - b (positive means a is larger)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    ci = bootstrap_mean_ci(d, n_boot=n_boot, seed=seed)
    ci["frac_a_lower"] = float(np.mean(d < 0)) if d.size else float("nan")
    return ci


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R = R_pred.T @ R_gt
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def translation_direction_error_deg(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Angle between translation directions (scale-free)."""
    a = np.asarray(t_pred, dtype=np.float64)
    b = np.asarray(t_gt, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    cos = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def relative_pose(T_a_to_world: np.ndarray, T_b_to_world: np.ndarray) -> np.ndarray:
    """T_b_to_a: pose of camera b expressed in camera a."""
    return np.linalg.inv(T_a_to_world) @ T_b_to_world


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (s, R, t) minimising ||s R src + t - dst||; src, dst are (N, 3)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need at least 3 paired 3-D points")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t
