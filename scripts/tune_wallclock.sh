#!/usr/bin/env bash
# End-to-end board wall clock at a realistic budget: workers 1 vs 3 vs 3+MPS.
#
# Everything else in this sprint measures throughput synthetically -- one arm,
# lockstep processes, no evaluation, no checkpointing, no data loading. That
# biases the answer AGAINST tenancy, because a real job leaves GPU-idle gaps a
# co-resident job can fill. `tune_validate.sh` runs real jobs but at 5M tokens,
# where start-up is ~78% of compute and tenancy therefore looks far better than
# it can at 50M (start-up ~8%).
#
# This is the honest middle: real suite jobs, real evals and checkpoints, at 20M
# tokens, so the start-up fraction is within a factor of ~2.5 of a 50M board
# rather than a factor of ten. Three arms x 2 seeds = 6 jobs per condition, one
# fresh directory each because `workers` is a recipe field.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
OUT=nanolab/out/_tune
BUDGET=20000000
ARMS=attention,hybrid_mingru8_attn4

mps_servers() { echo get_server_list | nvidia-cuda-mps-control 2>/dev/null | tr -d '[:space:]'; }

run_cond() {   # $1=tag  $2=workers
  local dir="$OUT/wc_$1"
  rm -rf "$dir"
  echo "--- wallclock $1 (workers=$2) $(date -u +%FT%TZ) ---"
  local s=$(date +%s)
  CROSSOVER_ARMS=$ARMS CROSSOVER_JOB_PREFIX="wc$1" \
  CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=$BUDGET \
  python3 -u -m nanolab.crossover_replicate launch --out "$dir" --workers "$2" \
    --seed 1337 2>&1 | tail -3
  CROSSOVER_ARMS=$ARMS CROSSOVER_JOB_PREFIX="wc$1" \
  CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=$BUDGET \
  python3 -u -m nanolab.crossover_replicate launch --out "$dir" --workers "$2" \
    --seed 42 2>&1 | tail -3
  local e=$(date +%s)
  echo "WALLCLOCK cond=$1 workers=$2 budget=$BUDGET elapsed=$((e-s))s"
}

echo "=== END-TO-END WALL CLOCK $(date -u +%FT%TZ) ==="
run_cond t1 1
run_cond t3 3

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if [ -z "$(mps_servers)" ]; then
  nvidia-cuda-mps-control -d || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -n "$(mps_servers)" ] && break; sleep 1; done
fi
if [ -z "$(mps_servers)" ]; then
  echo "MPS: NOT SERVING -- skipping m3 rather than mislabelling it as an MPS result"
else
  echo "MPS: serving, server(s)=$(mps_servers)"
  run_cond m3 3
  echo "MPS: after m3, server(s)=$(mps_servers)"
fi
echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 2
[ -z "$(mps_servers)" ] && echo "MPS: down" || echo "MPS: STILL UP"

echo "wallclock exit=0 $(date -u +%FT%TZ)"
