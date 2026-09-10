#!/usr/bin/env bash
# The MPS numerical-identity condition, third attempt -- with the gate fixed.
#
# The first attempt gated on `pgrep`, which raced the daemon's fork and reported
# "FAILED TO START" while it was up. The second gated on `get_server_list`
# returning a server pid, which is stricter than reality: MPS spawns a *server*
# only when a CUDA client first connects, so a freshly started control daemon
# legitimately reports an EMPTY server list and the gate refused it twice.
#
# The precondition is that the control daemon answers at all; the proof that the
# run actually used MPS is that clients connected during it, which the control
# log records. So: gate on the daemon responding, and verify after the fact by
# counting NEW CLIENT lines. A run with zero new clients is reported as not an
# MPS result, rather than being quietly filed as one.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
OUT=nanolab/out/_tune
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

daemon_up() { echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; }

if ! daemon_up; then
  nvidia-cuda-mps-control -d || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do daemon_up && break; sleep 1; done
fi
if ! daemon_up; then
  echo "MPS: control daemon not answering -- skipping m3 rather than mislabelling it"
  tail -5 "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null
  echo "m3b exit=1 $(date -u +%FT%TZ)"
  exit 1
fi
echo "MPS: control daemon answering (server list may be empty until a client attaches)"
BEFORE=$(grep -c "NEW CLIENT" "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null || echo 0)

rm -rf "$OUT/val_m3"
s=$(date +%s)
CROSSOVER_ARMS=attention,hybrid_mingru8_attn4 CROSSOVER_JOB_PREFIX=valm3 \
CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=5000000 \
python3 -u -m nanolab.crossover_replicate launch --out "$OUT/val_m3" --workers 3 2>&1 | tail -3
echo "WALLCLOCK cond=m3 workers=3 elapsed=$(( $(date +%s) - s ))s"

AFTER=$(grep -c "NEW CLIENT" "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null || echo 0)
N=$((AFTER-BEFORE))
echo "MPS: clients connected during m3 = $N ; servers now = $(echo get_server_list | nvidia-cuda-mps-control 2>/dev/null | tr -d '[:space:]')"
if [ "$N" -le 0 ]; then
  echo "MPS: WARNING no clients connected -- m3 is NOT an MPS result"
fi
echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 2
echo "m3b exit=0 $(date -u +%FT%TZ)"
