#!/usr/bin/env bash
# E16: MQAR recall across sequence lengths that straddle the window.
#
# The last unrun item of the SWA set. E12/E15 already answered the CE half --
# the window costs quality at ctx 512 (w64 +0.0986 nats vs dense, 5/5) and is
# free at ctx 2048 for w512 -- but CE at a fixed length cannot see whether a
# window breaks RECALL specifically, which is the axis the arms exist for.
# Straddling the window is the case that separates them.
#
# Idle guard uses nvidia-smi's compute-app list, NOT pgrep. `pgrep -f` matches
# whole command lines, so a shell whose argv CONTAINS the pattern -- the bash -c
# that writes this script, or any status check I run -- matches it. That cost
# ~48 min of idle GPU on 2026-09-06. nvidia-smi answers the question actually
# being asked: is anything using the device.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
LOG=nanolab/out/_tune/e16_mqar.log

gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }

quiet=0
for _ in $(seq 1 90); do
  if gpu_busy; then quiet=0; else
    quiet=$((quiet+1)); [ "$quiet" -ge 3 ] && break; fi
  sleep 10
done

{
  echo "=== E16: MQAR across window-straddling lengths $(date -u +%FT%TZ) ==="
  s=$(date +%s)
  python3 -u -m nanolab.crossover_replicate swaboard --only mqar-calibrate --workers 1 2>&1
  echo "e16 calibrate exit=$? elapsed=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)"
  s=$(date +%s)
  python3 -u -m nanolab.crossover_replicate swaboard --only mqar-grid --workers 4 2>&1
  echo "e16 grid exit=$? elapsed=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)"
  echo "e16 exit=0 $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
