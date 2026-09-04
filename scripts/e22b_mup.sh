#!/usr/bin/env bash
# The two muP cells, after the MoE board drains.
#
# They were refused on the first attempt: e22_queue.sh passed --workers 2, and
# gpu_bundle declines to co-locate two of its jobs on one device. The exit codes
# propagated and the MoE stage still ran, so this picks up exactly the two
# stages that did not.
#
# 20 jobs, one at a time, ~8 min each -> ~2.7h.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"

# Wait on the LOG MARKER, never pgrep -- tmux's own argv carries the whole inner
# command, so a pgrep pattern outlives the work it is watching.
deadline=$(( $(date +%s) + 8*3600 ))
while ! grep -q "^e22 exit=" nanolab/out/e22.log 2>/dev/null; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "e22 overran 8h; starting anyway $(date -u +%FT%TZ)"
    break
  fi
  sleep 120
done
echo "e22b start $(date -u +%FT%TZ)"

rc_any=0
for suite in e1_mup_tuned_spattn e1_sp_coldattn; do
  echo "--- $suite start $(date -u +%FT%TZ)"
  python3 -u scripts/gpu_bundle.py --only "$suite" --workers 1
  rc=$?
  echo "--- $suite exit=$rc $(date -u +%FT%TZ)"
  [ "$rc" -ne 0 ] && rc_any=$rc
done

python3 -u scripts/gpu_bundle.py --analyse 2>&1 | tail -60
echo "e22b exit=$rc_any $(date -u +%FT%TZ)"
