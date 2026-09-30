"""Track E: minimal conditional next-frame model (flow matching) in plain PyTorch.

The UNet takes the noisy target frame and the current frame concatenated (6 channels) plus a time
embedding and predicts the flow-matching velocity. The same network with a fixed t and zero noise input
is the deterministic regression baseline (trained with L2 on the next frame).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.n = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.out = nn.Conv2d(ch, ch, 1)
        self.heads = heads

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.out(a.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    """Small UNet for 64x128 frames: channels (64, 128, 256), attention at the 16x32 level."""

    def __init__(self, in_ch: int = 6, out_ch: int = 3, base: int = 64, mults=(1, 2, 4), tdim: int = 256):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        chs = [base * m for m in mults]
        self.down = nn.ModuleList()
        c = base
        for i, co in enumerate(chs):
            blocks = nn.ModuleList([ResBlock(c, co, tdim), ResBlock(co, co, tdim)])
            attn = Attention(co) if i == len(chs) - 1 else nn.Identity()
            pool = nn.Conv2d(co, co, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity()
            self.down.append(nn.ModuleList([blocks, attn, pool]))
            c = co
        self.mid = nn.ModuleList([ResBlock(c, c, tdim), Attention(c), ResBlock(c, c, tdim)])
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(chs))):
            blocks = nn.ModuleList([ResBlock(c + co, co, tdim), ResBlock(co, co, tdim)])
            attn = Attention(co) if i == len(chs) - 1 else nn.Identity()
            ups = nn.ConvTranspose2d(co, co, 4, stride=2, padding=1) if i > 0 else nn.Identity()
            self.up.append(nn.ModuleList([blocks, attn, ups]))
            c = co
        self.outc = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.temb(timestep_embedding(t, self.tdim))
        h = self.inc(torch.cat([x, cond], 1))
        skips = []
        for blocks, attn, pool in self.down:
            for b in blocks:
                h = b(h, temb)
            h = attn(h)
            skips.append(h)
            h = pool(h)
        h = self.mid[0](h, temb)
        h = self.mid[1](h)
        h = self.mid[2](h, temb)
        for blocks, attn, upsample in self.up:
            h = torch.cat([h, skips.pop()], 1)
            for b in blocks:
                h = b(h, temb)
            h = attn(h)
            h = upsample(h)
        return self.outc(h)


def to_tensor(frames_u8: np.ndarray, device) -> torch.Tensor:
    """uint8 (N, H, W, 3) -> float (N, 3, H, W) in [-1, 1]."""
    return torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0


def fm_loss(model: nn.Module, x1: torch.Tensor, cond: torch.Tensor, cond_drop: float = 0.1) -> torch.Tensor:
    """Conditional flow matching: x_t = (1 - t) x0 + t x1, target velocity x1 - x0."""
    b = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(b, device=x1.device)
    xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
    if cond_drop > 0:
        keep = (torch.rand(b, device=x1.device) > cond_drop).float()[:, None, None, None]
        cond = cond * keep
    v = model(xt, cond, t)
    return F.mse_loss(v, x1 - x0)


@torch.no_grad()
def fm_sample(model: nn.Module, cond: torch.Tensor, steps: int = 50, seed: int | None = None) -> torch.Tensor:
    """Euler integration of the learned velocity field from noise (t=0) to data (t=1)."""
    g = torch.Generator(device=cond.device)
    if seed is not None:
        g.manual_seed(seed)
    x = torch.randn(cond.shape, device=cond.device, generator=g)
    ts = torch.linspace(0, 1, steps + 1, device=cond.device)
    for i in range(steps):
        t = ts[i].expand(cond.shape[0])
        v = model(x, cond, t)
        x = x + (ts[i + 1] - ts[i]) * v
    return x.clamp(-1, 1)


def regression_loss(model: nn.Module, x1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    pred = model(torch.zeros_like(x1), cond, torch.zeros(x1.shape[0], device=x1.device))
    return F.mse_loss(pred, x1)


@torch.no_grad()
def regression_predict(model: nn.Module, cond: torch.Tensor) -> torch.Tensor:
    return model(torch.zeros_like(cond), cond, torch.zeros(cond.shape[0], device=cond.device)).clamp(-1, 1)


def psnr_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample PSNR for tensors in [-1, 1]."""
    mse = ((a - b) / 2.0).pow(2).flatten(1).mean(1).clamp(min=1e-10)
    return 10 * torch.log10(1.0 / mse)
