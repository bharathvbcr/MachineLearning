#!/usr/bin/env bash
# Chain: wait out the n=15 rate probe, then run the E16 board at the ONE batch
# with a demonstrated non-zero solve rate at this cell (bs 64: 2/3 then 7/10).
# Deliberately ONE cell. seq 63 and seq 511 are NOT included: no tested batch
# gives seq 63 a workable rate (1/12 across the whole recalibration), so a board
# there would floor every arm at 0 and separate nothing.
set -u
cd "$HOME/MLSystemsLab" || exit 1
echo "chain start $(date -u +%FT%TZ)"

deadline=$(( $(date +%s) + 7200 ))
while pgrep -f "mqar_suite .*mqar_e16_probe" >/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "probe overran 2h; starting board anyway"; break; }
  sleep 60
done
echo "probe exited $(date -u +%FT%TZ)"

python3 -u -m nanolab.mqar_suite \
  --out nanolab/out/mqar_e16_board --device cuda \
  --cells 64 --batch 64 \
  --arms attention,gdn,mingru,swa_w64,swa_w64_nosink \
  --seeds 15 --steps 3000 --lr-rule sqrt \
  --workers 4 --gpus 1
echo "board exit=$? $(date -u +%FT%TZ)"
