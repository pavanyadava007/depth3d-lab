import numpy as np
import torch

from depth3d.nextframe import HOLDOUT_SCENES, UNet, fm_loss, fm_sample, load_pairs, psnr_batch, regression_loss


def test_unet_shapes_and_losses_finite():
    torch.manual_seed(0)
    m = UNet(base=16)
    x = torch.randn(2, 3, 64, 128)
    c = torch.randn(2, 3, 64, 128)
    out = m(x, c, torch.rand(2))
    assert out.shape == (2, 3, 64, 128)
    assert torch.isfinite(fm_loss(m, x, c))
    assert torch.isfinite(regression_loss(m, x, c))
    s = fm_sample(m, c, steps=2, seed=0)
    assert s.shape == c.shape and s.min() >= -1 and s.max() <= 1


def test_psnr_batch_identity():
    a = torch.rand(3, 3, 8, 8) * 2 - 1
    p = psnr_batch(a, a)
    assert (p > 90).all()


def test_load_pairs_splits_scenes(tmp_path):
    frames = np.zeros((6, 64, 128, 3), dtype=np.uint8)
    scene = np.array(["scene-0001", "scene-0001", "scene-0001", HOLDOUT_SCENES[0], HOLDOUT_SCENES[0], "scene-0002"])
    ts = np.arange(6)
    np.savez(tmp_path / "f.npz", frames=frames, scene=scene, timestamp=ts, is_keyframe=np.ones(6, bool))
    fr, tr, te = load_pairs(str(tmp_path / "f.npz"))
    assert fr.shape[0] == 6
    assert list(tr) == [0, 1]  # pairs inside scene-0001 only; (2,3) and (4,5) cross scene borders
    assert list(te) == [3]
