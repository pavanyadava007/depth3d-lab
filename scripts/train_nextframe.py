"""Track E training: conditional flow-matching next-frame model, or the L2 regression baseline.

DDP-capable: when WORLD_SIZE > 1 (torchrun) the process group is initialised, the data is sharded by rank
and the model is wrapped in DistributedDataParallel. Backend from DDP_BACKEND (default nccl).
Tested on one NVIDIA L4 only (single process, and 2 processes with gloo on the same GPU as a smoke test).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from depth3d.nextframe import UNet, fm_loss, load_pairs, regression_loss, to_tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fm", "regression"], default="fm")
    ap.add_argument("--data", default="data/nextframe/frames.npz")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--max-minutes", type=float, default=40.0, help="wall-clock cap; steps actually done are recorded")
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        torch.distributed.init_process_group(backend=os.environ.get("DDP_BACKEND", "nccl"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)

    frames, train_idx, _ = load_pairs(args.data)
    frames_t = to_tensor(frames, device)
    model = UNet(base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if distributed else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
    rng = np.random.default_rng(args.seed + rank)
    shard = train_idx[rank::world]
    curve = []
    t0 = time.time()
    steps_done = 0
    for step in range(args.steps):
        if time.time() - t0 > args.max_minutes * 60:
            break
        steps_done = step + 1
        b = rng.choice(shard, args.batch // world)
        cond = frames_t[b]
        x1 = frames_t[b + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss = fm_loss(net, x1, cond) if args.mode == "fm" else regression_loss(net, x1, cond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
        if step % args.log_every == 0 and rank == 0:
            curve.append([step, float(loss), time.time() - t0])
            print(f"step {step} loss {float(loss):.4f} {time.time() - t0:.0f}s", flush=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    train_time = time.time() - t0
    if rank == 0:
        out = Path(args.out or f"data/nextframe/{args.mode}.pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ema": ema.state_dict(), "model": model.state_dict(), "args": vars(args)}, out)
        info = {
            "mode": args.mode,
            "gpu": "NVIDIA L4",
            "world_size": world,
            "n_params": n_params,
            "steps": steps_done,
            "steps_requested": args.steps,
            "max_minutes": args.max_minutes,
            "batch": args.batch,
            "lr": args.lr,
            "seed": args.seed,
            "n_train_pairs": int(len(train_idx)),
            "train_time_s": train_time,
            "steps_per_s": steps_done / train_time,
            "curve_columns": ["step", "loss", "elapsed_s"],
            "curve": curve,
        }
        Path("results").mkdir(exist_ok=True)
        Path(f"results/nextframe_train_{args.mode}.json").write_text(json.dumps(info, indent=1))
        print(f"trained {steps_done} steps in {train_time:.0f}s, {n_params / 1e6:.1f}M params (NVIDIA L4)")
        print("DONE")
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
