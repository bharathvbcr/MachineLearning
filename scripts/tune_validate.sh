#!/usr/bin/env bash
# Does changing tenancy (or enabling MPS) change what a run MEASURES?
#
# Every board in this repo assumes it does not ("tenancy only changes
# throughput, never the loss curve"). That assumption has never been tested, and
# data.should_gpu_resident picks the sampler from FREE VRAM at construction time
# -- with a CUDA generator on one path and a CPU generator on the other -- so a
# late-starting worker at high tenancy could silently train on different tokens.
#
# Four conditions, same arms and seeds, one output dir each (a changed `workers`
# is a changed recipe, so it must not reuse a directory):
#   a1  workers 1, no MPS      -- reference
#   a2  workers 1, no MPS      -- CONTROL: run-to-run noise floor at fixed config
#   t3  workers 3, no MPS
#   m3  workers 3, MPS on
# a1 vs a2 bounds what "identical" can mean; only differences larger than that
# are attributable to tenancy or MPS.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
OUT=nanolab/out/_tune
BUDGET=5000000            # ~305 steps at 16384 tok/step: past warmup, 6 evals
ARMS=attention,hybrid_mingru8_attn4

run_cond() {   # $1=tag  $2=workers
  local dir="$OUT/val_$1"
  rm -rf "$dir"
  echo "--- condition $1 (workers=$2) $(date -u +%FT%TZ) ---"
  local s=$(date +%s)
  CROSSOVER_ARMS=$ARMS CROSSOVER_JOB_PREFIX="val$1" \
  CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=$BUDGET \
  python3 -u -m nanolab.crossover_replicate launch --out "$dir" --workers "$2" \
    2>&1 | tail -5
  local e=$(date +%s)
  # REAL wall clock, unlike tune_tenancy.py's synthetic window: these jobs also
  # evaluate, checkpoint and load data, and a co-resident job can fill those
  # gaps. If tenancy ever pays on this box, it pays here.
  echo "WALLCLOCK cond=$1 workers=$2 elapsed=$((e-s))s"
}

echo "=== VALIDATION $(date -u +%FT%TZ) ==="
run_cond a1 1
run_cond a2 1
run_cond t3 3

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
mps_servers() { echo get_server_list | nvidia-cuda-mps-control 2>/dev/null | tr -d '[:space:]'; }
if [ -z "$(mps_servers)" ]; then
  nvidia-cuda-mps-control -d || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -n "$(mps_servers)" ] && break; sleep 1; done
fi
# Gate on the daemon actually serving, not on `pgrep` winning a race against its
# fork -- that race already mislabelled one stage of this sprint. A condition
# named m3 must not be a second copy of t3.
if [ -z "$(mps_servers)" ]; then
  echo "MPS: NOT SERVING -- skipping condition m3 rather than mislabelling it"
else
  echo "MPS: serving, server(s)=$(mps_servers)"
  run_cond m3 3
  echo "MPS: after m3, server(s)=$(mps_servers)"
fi
echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 2
[ -z "$(mps_servers)" ] && echo "MPS: down" || echo "MPS: STILL UP"

echo "validate exit=$? $(date -u +%FT%TZ)"
