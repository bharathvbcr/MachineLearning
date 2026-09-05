#!/usr/bin/env bash
# Stage 3, re-run: tenancy WITH CUDA MPS.
#
# Without MPS, concurrent processes time-slice the GPU: measured stage 2, that
# is a flat ~9% loss for arms that already saturate (attention, hybrid) and a
# 1.53x GAIN for the dispatch-bound one (gdn), because time-slicing recovers a
# job's idle launch gaps. MPS lets clients share SMs instead of alternating, so
# it should help the saturating arms too -- if it helps at all on this box.
#
# The first attempt at this stage recorded "MPS: FAILED TO START" while the
# daemon was in fact up: `pgrep` raced the daemon's fork. That is the exact
# shape of a check reporting a result it did not measure, so here the gate is
# `get_server_list` returning a server PID, and the sweep does NOT run unless it
# does. It also re-checks after the sweep, so a daemon that died mid-run cannot
# be reported as an MPS result.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
OUT=nanolab/out/_tune

mps_servers() { echo get_server_list | nvidia-cuda-mps-control 2>/dev/null | tr -d '[:space:]'; }

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if [ -z "$(mps_servers)" ]; then
  nvidia-cuda-mps-control -d || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -n "$(mps_servers)" ] && break; sleep 1; done
fi
BEFORE="$(mps_servers)"
if [ -z "$BEFORE" ]; then
  echo "MPS: NOT SERVING -- refusing to run a sweep that would be labelled 'mps'"
  echo "sprint exit=1 $(date -u +%FT%TZ)"
  exit 1
fi
echo "MPS: serving, server(s)=$BEFORE"

CLIENTS_BEFORE=$(grep -c "NEW CLIENT" "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null || echo 0)

for arm in attention hybrid_mingru8_attn4 gdn; do
  ten="1,2,3,4,6"; [ "$arm" = gdn ] && ten="1,2,3"
  echo "--- tenancy[mps] $arm ($ten) ---"
  python3 -u scripts/tune_tenancy.py --arm "$arm" --tenancies "$ten" \
    --seconds 45 --out "$OUT/tenancy_mps_${arm}.json"
done

AFTER="$(mps_servers)"
CLIENTS_AFTER=$(grep -c "NEW CLIENT" "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null || echo 0)
echo "MPS: after sweep server(s)=$AFTER ; clients connected during sweep=$((CLIENTS_AFTER-CLIENTS_BEFORE))"
[ -z "$AFTER" ] && echo "MPS: WARNING daemon died mid-sweep -- rows above are NOT trustworthy"
[ "$CLIENTS_AFTER" -eq "$CLIENTS_BEFORE" ] && echo "MPS: WARNING no clients connected -- workers did not use MPS"

echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 2
[ -z "$(mps_servers)" ] && echo "MPS: down" || echo "MPS: STILL UP"
echo "stage3 exit=0 $(date -u +%FT%TZ)"
echo "sprint exit=0 $(date -u +%FT%TZ)"
