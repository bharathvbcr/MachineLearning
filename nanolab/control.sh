#!/usr/bin/env bash
# E16 seq-63 CONTROL. swa_w64 and attention are the same function here (a 64-wide
# window spans a 63-token sequence), but measurement says their BACKWARD passes
# differ by 3.6e-4 in bf16, so training can diverge and saturation is bistable.
# This run therefore measures the board's RESOLUTION: the spread between two
# provably identical arms is the floor below which a seq-255 gap means nothing.
#
# batch 256, not the board's 64: at seq 63 / bs 64 the recalibration saw zero
# variance in 12 runs (all ~0.24), and a regime with no variance cannot detect
# whether the harness separates identical arms. bs 256 did vary (1/12 solved).
set -u
cd "$HOME/MLSystemsLab" || exit 1
echo "control waiting for board $(date -u +%FT%TZ)"
deadline=$(( $(date +%s) + 28800 ))
while ! grep -q "^board exit=" nanolab/out/board.log 2>/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "board overran 8h; control ABORTED"; exit 1; }
  sleep 60
done
echo "board done, control start $(date -u +%FT%TZ)"

python3 -u -m nanolab.mqar_suite \
  --out nanolab/out/mqar_e16_control --device cuda \
  --cells 16 --batch 256 \
  --arms attention,swa_w64 \
  --seeds 15 --steps 3000 --lr-rule sqrt \
  --workers 4 --gpus 1
echo "control exit=$? $(date -u +%FT%TZ)"
