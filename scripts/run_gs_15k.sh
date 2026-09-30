#!/usr/bin/env bash
# Second 3DGS ablation set with a 15k-step budget (the 7k set is produced by run_gpu_queue.sh).
cd /home/ec2-user/workspace/depth3d-lab || exit 1
PY=.venv/bin/python
export PATH=/home/ec2-user/workspace/depth3d-lab/.venv/bin:$PATH TORCH_CUDA_ARCH_LIST="8.9"
BEST=$(cat logs/best_mono_variant.txt 2>/dev/null || echo depth_pro)
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
for V in random lidar; do
  log "== 3DGS $V 15k"; $PY scripts/train_gs.py --variant $V --steps 15000 --name ${V}_15k > logs/gs_${V}_15k.log 2>&1; tail -1 logs/gs_${V}_15k.log
done
for W in 0.05 0.2; do
  log "== 3DGS lidar_depth w=$W 15k"; $PY scripts/train_gs.py --variant lidar_depth --depth-weight $W --depth-variant $BEST --steps 15000 --name lidar_depth${W}_15k > logs/gs_lidar_depth${W}_15k.log 2>&1; tail -1 logs/gs_lidar_depth${W}_15k.log
done
log "GS15K_DONE"
