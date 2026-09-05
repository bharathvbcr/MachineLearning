#!/usr/bin/env bash
# Phase 2: close the gaps phase 1 left open, in value order.
#
#  1 gdnchunk2  the chunk-width speed-up (1.98x) is unusable until its numerics
#               are checked, and phase 1's check was vacuous (zero-init output
#               projection made both sides all zeros)
#  2 drift      phase 1's later probes read ~12% below its first sweep on the
#               same code path; either the box slowed or the harnesses differ,
#               and every absolute number depends on which
#  3 m3         the MPS numerical-identity condition, skipped twice because the
#               daemon was not serving -- correctly skipped rather than
#               mislabelled, but still missing
#  4 mingru     compile end-to-end on the arm that gained most (1.96x), now that
#               train.py's gate is "one mixer kind" rather than "attention"
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
OUT=nanolab/out/_tune

step() { local n="$1" l="$2"; shift 2; echo "===== $n $(date -u +%FT%TZ) ====="; "$@" 2>&1 | tee "$l"; echo "$n exit=${PIPESTATUS[0]} $(date -u +%FT%TZ)"; }

step gdnchunk2 "$OUT/gdnchunk2.log" python3 -u scripts/tune_gdnchunk2.py

step drift "$OUT/drift.log" \
  python3 -u -m nanolab.sweep_gpu arm --arms attention,mingru,hybrid_mingru8_attn4,gdn,moe_e8k1 \
    --batch_size 32 --block_size 512 --peak_flops 752.8e12 --iters 20 \
    --mem_fraction 0 --grad_checkpoint false --out "$OUT/arm_cost_drift.json"

echo "===== m3 $(date -u +%FT%TZ) ====="
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
mps_servers() { echo get_server_list | nvidia-cuda-mps-control 2>/dev/null | tr -d '[:space:]'; }
if [ -z "$(mps_servers)" ]; then nvidia-cuda-mps-control -d || true; for _ in $(seq 12); do [ -n "$(mps_servers)" ] && break; sleep 1; done; fi
if [ -z "$(mps_servers)" ]; then
  echo "MPS: NOT SERVING -- m3 skipped again; see $CUDA_MPS_LOG_DIRECTORY/control.log"
  tail -5 "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null
else
  echo "MPS: serving, server(s)=$(mps_servers)"
  rm -rf "$OUT/val_m3"
  s=$(date +%s)
  CROSSOVER_ARMS=attention,hybrid_mingru8_attn4 CROSSOVER_JOB_PREFIX=valm3 \
  CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=5000000 \
  python3 -u -m nanolab.crossover_replicate launch --out "$OUT/val_m3" --workers 3 2>&1 | tail -3
  echo "WALLCLOCK cond=m3 workers=3 elapsed=$(( $(date +%s) - s ))s"
  echo "MPS: after m3 server(s)=$(mps_servers)"
fi
echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 2
echo "m3 exit=0 $(date -u +%FT%TZ)"

step mingru_compile "$OUT/compile_board_mingru.log" \
  env ARMS=mingru bash scripts/tune_compile_board.sh

echo "phase2 exit=0 $(date -u +%FT%TZ)"
