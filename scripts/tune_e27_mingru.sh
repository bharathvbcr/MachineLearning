#!/usr/bin/env bash
# E27, final piece: the width-1536 minGRU arm at the learning rate the REPAIRED
# probe picks, in its own directory at workers 1.
#
# Why a new directory rather than unholding the jobs already queued: those live
# in `crossover_ladder1536`, whose recipe records `workers: 2`, and two w1536
# minGRU jobs (~45 GiB each, measured today) do not fit on a 94.5 GiB card -- that
# is exactly what OOMed the probe. `lock_recipe` refuses a different tenancy in
# the same directory, so the arm gets its own.
#
# Reading the ladder across two directories is sound, and today it is measured
# rather than assumed: `--workers` 1 against 3 moves `final_val` by 0.0012 nats
# on average, against a 0.0014 nat floor between two runs of the SAME
# configuration. Tenancy changes the reported number less than rerunning does.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
ARGMIN_FILE=nanolab/out/_tune/e27_argmin.txt

if [ ! -s "$ARGMIN_FILE" ]; then
  echo "no argmin recorded: the probe did not settle lr20/lr40/lr80."
  echo "Refusing to run five 50M jobs at a learning rate nothing selected."
  echo "e27_mingru exit=1 $(date -u +%FT%TZ)"
  exit 1
fi
LR=$(cat "$ARGMIN_FILE")
ARM="w1536_mingru_${LR}"
echo "=== E27 minGRU arm: $ARM (argmin of the repaired probe) $(date -u +%FT%TZ) ==="

s=$(date +%s)
CROSSOVER_ARMS=$ARM CROSSOVER_JOB_PREFIX=cx32lad1536m \
CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=50000000 \
python3 -u -m nanolab.crossover_replicate launch \
  --out nanolab/out/crossover_ladder1536_mingru --workers 1 2>&1 | tail -6
echo "e27_mingru elapsed=$(( $(date +%s) - s ))s"
echo "e27_mingru exit=0 $(date -u +%FT%TZ)"
