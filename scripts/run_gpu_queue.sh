#!/usr/bin/env bash
# Sequential GPU queue so that every reported timing is measured on an otherwise idle NVIDIA L4.
cd /home/ec2-user/workspace/depth3d-lab || exit 1
PY=.venv/bin/python
export PATH=/home/ec2-user/workspace/depth3d-lab/.venv/bin:$PATH
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export TORCH_CUDA_ARCH_LIST="8.9"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "== Depth Pro eval (clipped scoring)"
$PY scripts/eval_mono_depth.py --model depth_pro --save-pred-scene scene-0916 > logs/eval_depth_pro.log 2>&1
tail -3 logs/eval_depth_pro.log

log "== DA-V2 metric outdoor eval"
$PY scripts/eval_mono_depth.py --model dav2_metric_outdoor_large --save-pred-scene scene-0916 > logs/eval_dav2.log 2>&1
tail -3 logs/eval_dav2.log

BEST=$($PY scripts/pick_best_mono.py)
log "best mono variant for depth supervision: $BEST"
echo "$BEST" > logs/best_mono_variant.txt

log "== 3DGS smoke"
$PY scripts/train_gs.py --variant lidar --name smoke --steps 300 --no-video --out-dir logs/smoke --img-dir logs/smoke --save-dir logs/smoke > logs/gs_smoke.log 2>&1
tail -2 logs/gs_smoke.log
SPS=$($PY -c "import json;print(json.load(open('logs/smoke/gs_smoke.json'))['steps_per_s'])" 2>/dev/null || echo 0)
STEPS=$($PY -c "s=float('$SPS'); print(7000 if s>8 else 4000)")
log "smoke steps/s=$SPS -> using $STEPS steps per variant"

for V in random lidar; do
  log "== 3DGS $V"
  $PY scripts/train_gs.py --variant $V --steps $STEPS > logs/gs_$V.log 2>&1
  tail -2 logs/gs_$V.log
done
for W in 0.05 0.2; do
  log "== 3DGS lidar_depth w=$W ($BEST)"
  $PY scripts/train_gs.py --variant lidar_depth --depth-weight $W --depth-variant $BEST --steps $STEPS > logs/gs_lidar_depth$W.log 2>&1
  tail -2 logs/gs_lidar_depth$W.log
done

log "== VGGT multi-view eval"
$PY scripts/eval_multiview.py > logs/eval_multiview.log 2>&1
tail -3 logs/eval_multiview.log

log "== Track E: flow matching + regression baseline (run concurrently, GPU shared between the two)"
$PY scripts/train_nextframe.py --mode fm --steps 12000 > logs/train_fm.log 2>&1 &
$PY scripts/train_nextframe.py --mode regression --steps 12000 > logs/train_reg.log 2>&1 &
wait
tail -2 logs/train_fm.log logs/train_reg.log
log "== Track E eval"
$PY scripts/eval_nextframe.py --fid > logs/eval_nextframe.log 2>&1
tail -3 logs/eval_nextframe.log

log "== depth previews for the site"
$PY scripts/render_depth_previews.py > logs/depth_previews.log 2>&1
tail -1 logs/depth_previews.log
log "QUEUE_DONE"
