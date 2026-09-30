import numpy as np
import torch

from depth3d.gs import psnr, rgb_to_sh0, ssim


def test_ssim_and_psnr_identity():
    img = torch.rand(1, 3, 32, 48)
    assert float(ssim(img, img)) > 0.999
    assert psnr(img[0].permute(1, 2, 0), img[0].permute(1, 2, 0)) > 90
    noisy = (img + 0.1 * torch.randn_like(img)).clamp(0, 1)
    assert float(ssim(img, noisy)) < 0.99


def test_sh0_roundtrip():
    rgb = torch.tensor([[0.2, 0.5, 0.9]])
    back = rgb_to_sh0(rgb) * 0.28209479177387814 + 0.5
    assert torch.allclose(back, rgb, atol=1e-6)


def test_splat_quaternion_rotation():
    from scripts.build_space import quat_mul

    # rotating by q then by its conjugate gives identity
    q = np.array([[0.7071068, 0.7071068, 0, 0]], dtype=np.float32)
    qc = q * np.array([1, -1, -1, -1], dtype=np.float32)
    r = quat_mul(q, qc)
    assert np.allclose(r, [[1, 0, 0, 0]], atol=1e-6)
