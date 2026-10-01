#!/usr/bin/env bash
# Sequential GPU queue (2026-10-01): MoGe-2 Track A, then seeds 1-2 for 3DGS (7k) and for Track E.
# Every job has the GPU to itself, except the two Track E trainings of one seed, which share it on purpose (as seed 0).
cd /home/ec2-user/workspace/depth3d-lab || exit 1
PY=.venv/bin/python
export PATH=/home/ec2-user/workspace/depth3d-lab/.venv/bin:$PATH
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export TORCH_CUDA_ARCH_LIST="8.9"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
BEST=$(cat logs/best_mono_variant.txt)

log "== MoGe-2 eval (all 2,424 images, clipped scoring)"
$PY scripts/eval_mono_depth.py --model moge2 --save-pred-scene scene-0916 > logs/eval_moge2.log 2>&1
tail -3 logs/eval_moge2.log

for S in 1 2; do
  for V in random lidar; do
    log "== 3DGS $V seed $S"
    $PY scripts/train_gs.py --variant $V --steps 7000 --seed $S --name ${V}_s$S --no-video --img-dir logs/img_seeds --save-dir logs/gs_seeds > logs/gs_${V}_s$S.log 2>&1
    tail -1 logs/gs_${V}_s$S.log
  done
  for W in 0.05 0.2; do
    log "== 3DGS lidar_depth w=$W ($BEST) seed $S"
    $PY scripts/train_gs.py --variant lidar_depth --depth-weight $W --depth-variant $BEST --steps 7000 --seed $S --name lidar_depth${W}_s$S --no-video --img-dir logs/img_seeds --save-dir logs/gs_seeds > logs/gs_lidar_depth${W}_s$S.log 2>&1
    tail -1 logs/gs_lidar_depth${W}_s$S.log
  done
done

for S in 1 2; do
  log "== Track E seed $S: fm + regression concurrently (GPU shared between the two, 40 min cap)"
  $PY scripts/train_nextframe.py --mode fm --steps 12000 --seed $S --out data/nextframe/fm_s$S.pt --out-json results/nextframe_train_fm_s$S.json > logs/train_fm_s$S.log 2>&1 &
  $PY scripts/train_nextframe.py --mode regression --steps 12000 --seed $S --out data/nextframe/regression_s$S.pt --out-json results/nextframe_train_regression_s$S.json > logs/train_reg_s$S.log 2>&1 &
  wait
  tail -n 2 logs/train_fm_s$S.log logs/train_reg_s$S.log
  log "== Track E eval seed $S"
  $PY scripts/eval_nextframe.py --fid --seed $S --fm data/nextframe/fm_s$S.pt --reg data/nextframe/regression_s$S.pt --out-json results/synthesis_nextframe_s$S.json --grid logs/img_seeds/nextframe_grid_s$S.jpg > logs/eval_nextframe_s$S.log 2>&1
  tail -3 logs/eval_nextframe_s$S.log
done
log "SEEDS_QUEUE_DONE"
