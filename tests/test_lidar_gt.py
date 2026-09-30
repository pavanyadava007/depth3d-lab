import numpy as np
import pytest

from depth3d import nusc as N
from depth3d.lidar_gt import dense_from_sparse, project_points


def _K():
    return np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])


def test_synthetic_frustum_roundtrip():
    rng = np.random.default_rng(0)
    K = _K()
    W, H = 640, 480
    z = rng.uniform(2.0, 60.0, 500)
    u = rng.uniform(0, W - 1e-3, 500)
    v = rng.uniform(0, H - 1e-3, 500)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], 1)
    ui, vi, zi = project_points(pts, np.eye(4), K, W, H)
    assert len(zi) == len(np.unique(np.floor(v).astype(int) * W + np.floor(u).astype(int)))
    assert zi.min() > 1.0 and zi.max() < 80.0
    # every kept depth is one of the inputs at the same pixel
    dense = dense_from_sparse({"u": ui, "v": vi, "z": zi, "width": W, "height": H})
    for k in range(50):
        pu, pv = int(np.floor(u[k])), int(np.floor(v[k]))
        assert dense[pv, pu] <= z[k] + 1e-4 and dense[pv, pu] > 0


def test_zbuffer_keeps_nearest_and_range_filter():
    K = _K()
    pts = np.array([[0, 0, 10.0], [0, 0, 5.0], [0, 0, 30.0], [0, 0, 0.5], [0, 0, 100.0], [1000.0, 0, 10.0]])
    u, v, z = project_points(pts, np.eye(4), K, 640, 480)
    assert len(z) == 1
    assert z[0] == pytest.approx(5.0)
    assert u[0] == 320 and v[0] == 240


def test_extrinsic_transform_applied():
    K = _K()
    T = np.eye(4)
    T[:3, 3] = [0, 0, 7.0]  # source frame is 7 m behind the camera
    u, v, z = project_points(np.array([[0.0, 0.0, 3.0]]), T, K, 640, 480)
    assert z[0] == pytest.approx(10.0)


@pytest.mark.skipif(not N.data_available(), reason="nuScenes-mini not on disk")
def test_real_frame_sanity():
    from depth3d.lidar_gt import build_gt

    nusc = N.load_nusc()
    tok = nusc.scene[0]["first_sample_token"]
    gt = build_gt(nusc, tok, "CAM_FRONT")
    assert 1000 < len(gt["z"]) < 40000
    assert (gt["u"] < gt["width"]).all() and (gt["v"] < gt["height"]).all()
    assert 3.0 < np.median(gt["z"]) < 40.0
