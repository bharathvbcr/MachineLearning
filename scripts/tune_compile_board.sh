#!/usr/bin/env bash
# What does torch.compile cost, in the only units that matter here?
#
# Measured 2026-09-05 on this box: 1.94x on attention, 1.96x on minGRU -- by far
# the largest lever in the sprint, and the repo disables it everywhere
# (`compile=False` hardcoded in job_config and current_recipe) on a note about
# Inductor stalling on aarch64 that torch 2.7.0 no longer justifies: it compiles
# in ~30 s.
#
# Speed alone cannot decide it. Inductor fuses and re-associates, so a compiled
# run's loss moves in the last places, and every board that pools with an
# existing directory would be comparing two different numerics. This runs the
# SAME arms and seeds eager and compiled, at a budget long enough for any
# divergence to compound, so the price is a measured number of nats rather than
# an assumption. `compile` is now recorded in the recipe, so the two directories
# can never be merged by accident.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
OUT=nanolab/out/_tune
BUDGET=20000000
ARMS=${ARMS:-attention}

run_cond() {   # $1=tag  $2=CROSSOVER_COMPILE
  local dir="$OUT/cmp_$1"
  rm -rf "$dir"
  echo "--- compile-board $1 (CROSSOVER_COMPILE=$2, arms=$ARMS) $(date -u +%FT%TZ) ---"
  local s=$(date +%s)
  CROSSOVER_COMPILE=$2 CROSSOVER_ARMS=$ARMS CROSSOVER_JOB_PREFIX="cmp$1" \
  CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=$BUDGET \
  python3 -u -m nanolab.crossover_replicate launch --out "$dir" --workers 1 2>&1 | tail -4
  local e=$(date +%s)
  echo "WALLCLOCK cond=$1 compile=$2 arms=$ARMS elapsed=$((e-s))s"
}

echo "=== COMPILE BOARD $(date -u +%FT%TZ) ==="
run_cond eager 0
run_cond on 1
echo "compile_board exit=0 $(date -u +%FT%TZ)"
