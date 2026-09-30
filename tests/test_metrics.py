import numpy as np
import pytest

from depth3d.metrics import (
    MetricAccumulator,
    depth_metrics,
    median_scale,
    paired_bootstrap,
    relative_pose,
    rotation_error_deg,
    translation_direction_error_deg,
    umeyama,
)


def test_perfect_prediction():
    gt = np.linspace(1, 80, 100)
    m = depth_metrics(gt, gt)
    assert m["abs_rel"] == 0 and m["rmse"] == 0 and m["d1"] == 1.0


def test_scale_error_is_fixed_by_median_scaling():
    gt = np.linspace(2, 60, 200)
    pred = 0.5 * gt
    raw = depth_metrics(pred, gt)
    s = median_scale(pred, gt)
    assert s == pytest.approx(2.0)
    fixed = depth_metrics(pred * s, gt)
    assert raw["abs_rel"] == pytest.approx(0.5)
    assert fixed["abs_rel"] == pytest.approx(0.0, abs=1e-9)
    assert raw["d1"] == 0.0 and fixed["d1"] == 1.0


def test_accumulator_matches_pointwise_metrics():
    rng = np.random.default_rng(1)
    gt = rng.uniform(1, 80, 1000)
    pred = gt * rng.uniform(0.8, 1.2, 1000)
    acc = MetricAccumulator()
    acc.add(pred[:300], gt[:300])
    acc.add(pred[300:], gt[300:])
    a = acc.result()
    b = depth_metrics(pred, gt)
    for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]:
        assert a[k] == pytest.approx(b[k])
    assert a["n_points"] == 1000


def test_paired_bootstrap_sign():
    a = np.full(50, 0.10)
    b = np.full(50, 0.20)
    ci = paired_bootstrap(a, b)
    assert ci["mean"] == pytest.approx(-0.10)
    assert ci["hi"] < 0


def test_pose_errors_and_umeyama():
    from scipy.spatial.transform import Rotation as Rot

    R = Rot.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    assert rotation_error_deg(R, R) == pytest.approx(0.0, abs=1e-4)
    assert rotation_error_deg(np.eye(3), Rot.from_euler("z", 15, degrees=True).as_matrix()) == pytest.approx(15.0)
    assert translation_direction_error_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    rng = np.random.default_rng(2)
    src = rng.normal(size=(40, 3))
    s, t = 2.5, np.array([1.0, -2.0, 3.0])
    dst = s * (R @ src.T).T + t
    s2, R2, t2 = umeyama(src, dst)
    assert s2 == pytest.approx(s)
    assert np.allclose(R2, R, atol=1e-8) and np.allclose(t2, t, atol=1e-8)
    Ta = np.eye(4)
    Tb = np.eye(4)
    Tb[:3, 3] = [0, 0, 5]
    rel = relative_pose(Ta, Tb)
    assert np.allclose(rel[:3, 3], [0, 0, 5])


def test_predictions_are_clipped_to_range():
    gt = np.array([10.0, 20.0, 30.0])
    pred = np.array([10.0, 20.0, 1e4])  # sky-like outlier
    m = depth_metrics(pred, gt)
    assert m["rmse"] == pytest.approx(np.sqrt((80.0 - 30.0) ** 2 / 3))
    acc = MetricAccumulator()
    acc.add(pred, gt)
    r = acc.result()
    assert r["frac_pred_above_max"] == pytest.approx(1 / 3)
    assert r["rmse"] == pytest.approx(m["rmse"])
