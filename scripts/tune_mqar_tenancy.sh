#!/usr/bin/env bash
# Does tenancy pay off at the RECALL shape, where the 124M board says it does not?
#
# At the 50M board shape (124M params, batch 32 x ctx 512) aggregate throughput
# PEAKS at workers=1 and loses ~9% at workers>=2 -- there is no MPS daemon, so
# concurrent processes time-slice instead of sharing SMs. The recall grid is a
# different regime: ~9.5M params, d_model 256, seq 31 at batch 256, and
# mqar_suite's own --workers help asserts "a single run leaves the GPU mostly
# idle". That assertion has never been measured. E28 is 120 runs and E32 is 60,
# so the answer sets their wall clock.
#
# Same cell, same arms and seeds, one fresh --out per tenancy (the ledger skips
# completed runs, so sharing a directory would time nothing). Steps are cut to
# 1000 because only the RATIO between tenancies is wanted.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
OUT=nanolab/out/_tune
STEPS=1000

for W in 1 2 4 8; do
  d="$OUT/mqar_t$W"
  rm -rf "$d"
  echo "--- mqar workers=$W $(date -u +%FT%TZ) ---"
  s=$(date +%s)
  python3 -u -m nanolab.mqar_suite --out "$d" --device cuda \
    --arms attention,gdn --pairs 8 --steps $STEPS --batch 256 \
    --seeds 4 --lr-rule sqrt --workers "$W" --gpus 1 >"$OUT/mqar_t$W.log" 2>&1
  rc=$?
  e=$(date +%s)
  echo "mqar workers=$W elapsed=$((e-s))s rc=$rc"
done
echo "mqar_tenancy exit=0 $(date -u +%FT%TZ)"
