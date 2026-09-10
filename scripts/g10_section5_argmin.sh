#!/usr/bin/env bash
# G10: section 5's hybrid-vs-attention rows at each arm's own measured 50M argmin.
#
# G7 re-ran them at 1x/4x/8x and found all three arms preferring the same multiplier,
# which retires the confound E35 inferred. This closes the bracket underneath: the
# probe carries both hybrid shapes and attention at w768 down to 0.25x, so the argmin
# is located rather than assumed to be the lowest point anyone happened to run.
#
# Grows `crossover50m_ratioplace32` -- the board the confounded rows came from -- one
# arm at a time. Arms already at n=5 are skipped, so if the argmin turns out to be 1x
# this stage costs nothing and says so.
#
# Pre-registered reading: whether the three arms share a multiplier is the whole
# question. Sharing one means a shared-LR board was never reading them from different
# distances and section 5 stands as measured. Differing means section 5's rows are
# re-read at the per-arm optima and restated, whichever way they then fall.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g10
stage_start g10
python3 scripts/g9_plan.py --hybrids
PLAN=$(python3 scripts/g9_plan.py --hybrids --arms)
rc=$?
if [ "$rc" -ne 0 ] || [ -z "$PLAN" ]; then
  echo "g10: planner refused; section 5 stays at the LRs already measured"
  stage_end g10 2
fi
echo "g10 plan: $PLAN"
BOARD=$(python3 -c "import json;print(','.join(json.load(open('nanolab/out/crossover50m_ratioplace32/recipe.json'))['arms']))")
export CROSSOVER_ARMS="$BOARD,$PLAN"
export CROSSOVER_JOB_PREFIX=cx32p
rc=0
for a in ${PLAN//,/ }; do
  python3 -u -m nanolab.crossover_replicate launch \
    --out nanolab/out/crossover50m_ratioplace32 --workers 3 --arm "$a" || rc=$?
done
echo "g10 launch exit=$rc $(date -u +%FT%TZ)"
B=crossover50m_ratioplace32
REF=$(echo "$PLAN" | tr ',' '\n' | grep attention | head -1)
for a in $(echo "$PLAN" | tr ',' '\n' | grep -v attention); do
  python3 scripts/paired_board.py "$a@$B" --ref "$REF@$B"
done
echo "REMINDER: bash scripts/pull_artifacts.sh $B"
stage_end g10 "$rc"
