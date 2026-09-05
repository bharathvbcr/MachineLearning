#!/usr/bin/env bash
# GH200 tuning sprint (2026-09-05): measure the knobs that set how long the
# E28-E35 program takes, WITHOUT changing what any board measures.
#
# Stage 2/3 answer the question the repo has never asked: `workers` (tenancy) is
# a recipe field every board sets by hand, but with no MPS daemon running,
# concurrent processes time-slice the GPU rather than sharing SMs -- so tenancy
# may be buying far less than the boards assume. Stage 3 repeats stage 2 with
# MPS on; the pair is the whole point.
#
# Each stage appends a log marker so the next can be chained (repo rule 4).
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
OUT=nanolab/out/_tune
mkdir -p "$OUT"

mps_up() {
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d && sleep 3
  pgrep -a nvidia-cuda-mps-control >/dev/null && echo "MPS: up" || echo "MPS: FAILED TO START"
}
mps_down() {
  echo quit | nvidia-cuda-mps-control 2>/dev/null
  sleep 2
  pgrep -a nvidia-cuda-mps-control >/dev/null && echo "MPS: STILL UP" || echo "MPS: down"
}

stage_tenancy() {  # $1=tag  $2=extra label
  for arm in attention hybrid_mingru8_attn4 gdn; do
    ten="1,2,3,4,6"
    [ "$arm" = gdn ] && ten="1,2,3"          # 25.6GB/job: 3 is the VRAM ceiling
    echo "--- tenancy[$1] $arm ($ten) ---"
    python3 -u scripts/tune_tenancy.py --arm "$arm" --tenancies "$ten" \
      --seconds 45 --out "$OUT/tenancy_${1}_${arm}.json"
  done
}

echo "=== STAGE 2: tenancy WITHOUT MPS $(date -u +%FT%TZ) ==="
stage_tenancy nomps
echo "stage2 exit=$? $(date -u +%FT%TZ)"

echo "=== STAGE 3: tenancy WITH MPS $(date -u +%FT%TZ) ==="
mps_up
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
stage_tenancy mps
echo "stage3 exit=$? $(date -u +%FT%TZ)"
mps_down

echo "sprint exit=0 $(date -u +%FT%TZ)"
