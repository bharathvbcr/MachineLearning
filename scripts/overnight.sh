#!/usr/bin/env bash
# Chain: wait out the n=15 rate probe, then run the E16 board at the ONE batch
# with a demonstrated non-zero solve rate at this cell (bs 64: 2/3 then 7/10).
# Deliberately ONE cell. seq 63 and seq 511 are NOT included: no tested batch
# gives seq 63 a workable rate (1/12 across the whole recalibration), so a board
# there would floor every arm at 0 and separate nothing.
set -u
cd "$HOME/MLSystemsLab" || exit 1
echo "chain start $(date -u +%FT%TZ)"

# Waits on the probe's own LOG, not on pgrep. The pgrep form this replaced
# cost 79 minutes of idle GPU: the probe was launched as
#   tmux new-session -d -s probe "cd ... && python3 -m nanolab.mqar_suite ..."
# and tmux's OWN command line carries that whole inner string, so
# `pgrep -f "mqar_suite .*mqar_e16_probe"` kept matching the tmux process long
# after the python exited. The wait only ended when the 2h deadline fired.
# The same pattern also self-matches when the waiter is an inline `bash -c`,
# because then the pattern is in the waiter's own argv too. A log marker
# written by the work itself cannot match anything but the work.
deadline=$(( $(date +%s) + 7200 ))
while ! grep -q "^probe exit=" nanolab/out/probe.log 2>/dev/null; do
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
