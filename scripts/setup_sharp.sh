#!/usr/bin/env bash
# Track D needs Apple's SHARP (https://github.com/apple/ml-sharp). Its source is not redistributed here:
# this script clones it at the commit used in the study into third_party/ (git-ignored), installs it into the
# venv WITHOUT its pinned requirements (so torch 2.7.1+cu126 stays), and fetches the public checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
COMMIT=aed6527499ef91cba3b54c18d49a870f25947190
mkdir -p third_party
if [ ! -d third_party/ml-sharp ]; then
  git clone https://github.com/apple/ml-sharp.git third_party/ml-sharp
fi
git -C third_party/ml-sharp checkout -q "$COMMIT"
"$PY" -m pip install --no-deps -e third_party/ml-sharp
CK=~/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt
if [ ! -f "$CK" ]; then
  mkdir -p "$(dirname "$CK")"
  curl -L -o "$CK" https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
fi
"$PY" -c "import sharp; print('sharp ok')"
echo "SHARP_SETUP_DONE"
