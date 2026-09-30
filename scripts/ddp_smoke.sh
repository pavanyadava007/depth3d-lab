#!/usr/bin/env bash
# Smoke test of the DDP code path in scripts/train_nextframe.py with 2 processes on ONE GPU (gloo backend).
# This only checks that init_process_group, sharding and DistributedDataParallel run; it is not a speed test.
cd /home/ec2-user/workspace/depth3d-lab || exit 1
DDP_BACKEND=gloo .venv/bin/torchrun --standalone --nproc_per_node=2 scripts/train_nextframe.py \
  --mode fm --steps 20 --batch 16 --log-every 10 --out logs/ddp_smoke.pt
