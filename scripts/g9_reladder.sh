#!/usr/bin/env bash
# G9: the width ladder re-run at the learning rate the 50M probe actually measured.
#
# The ladder's rungs were spent at LRs probed over 610 steps. G7 and G3 showed every
# arm at every width wants less at 3051. This re-runs them at the measured 50M argmin,
# n=5, so the width claim can be restated as a comparison of arms at their own optima
# instead of at a 10M-derived protocol.
#
# The multiplier is NOT written here. `scripts/g9_plan.py` derives it from the probe
# boards and refuses rather than guesses: a cell whose minimum sits at either end of
# its grid refuses (an argmin at the edge is not an argmin -- and this runner's own
# VRAM guard exists because a stage script once read an argmin off four OOM survivors,
# one of them an edge, and spent ten 50M jobs on it), and w1536, which has no probe of
# its own, inherits a multiplier only when all three probed widths agree.
#
# Routing mirrors how the original rung was built, because the tenancy is not free:
#   w384/w768/w1152 -> grow `crossover_ladder50m`   (workers 2)
#   w1536, both arms -> a NEW board                  (workers 1)
# Neither is a preference. `crossover_ladder1536` lists `w1536_mingru_lr20` in its arms
# with no such run on disk: it OOMed two-to-a-device on 2026-09-05. This script first
# said "grow that board at workers 2" for the ATTENTION arm, copying the tenancy it
# recorded -- and G8c proved that wrong by trying it: the VRAM guard refuses two d1536
# attention jobs at 40.2 GiB each on a 94.5 GiB device. That board ran at 2 before the
# guard existed. Dropping it to workers 1 is not available either, because
# `lock_recipe` refuses a tenancy change. So w1536 goes to a new board at workers 1,
# where both its arms can sit together at one recipe.
#
# One arm at a time: `crossover_ladder50m` also holds G8b's n=1 probe arms, and an
# unfiltered launch would queue four more seeds for each of them.
#
# Pre-registered reading: the crossing is re-estimated per width at the new LR. If the
# monotone width trend survives, section 3's result stands and is finally described
# correctly. If it flattens or reverses, the width claim was an artifact of the
# 10M-derived protocol and is withdrawn -- that outcome is as publishable as the other
# and must not be softened.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g9
stage_start g9
PLAN=$(python3 scripts/g9_plan.py --arms)
rc=$?
if [ "$rc" -ne 0 ] || [ -z "$PLAN" ]; then
  echo "g9: planner refused; not spending n=5 on an unlocated argmin"
  python3 scripts/g9_plan.py
  stage_end g9 2
fi
echo "g9 plan: $PLAN"
rc=0
for a in ${PLAN//,/ }; do
  case "$a" in
    w1536_*)           OUT=nanolab/out/crossover_ladder1536_argmin;   W=1; PFX=cx32lad1536a ;;
    *)                 OUT=nanolab/out/crossover_ladder50m;           W=2; PFX=cx32lad50 ;;
  esac
  # The arm list a board is launched with must contain everything already recorded in
  # it, or `lock_recipe` sees the list SHRINK and refuses; `--arm` then filters which
  # of them is actually queued.
  CROSSOVER_ARMS=$(python3 - "$OUT" "$a" <<'PYEOF'
import json, os, sys
out, arm = sys.argv[1], sys.argv[2]
rec = os.path.join(out, "recipe.json")
have = json.load(open(rec))["arms"] if os.path.exists(rec) else []
print(",".join(dict.fromkeys(list(have) + [arm])))
PYEOF
)
  export CROSSOVER_ARMS CROSSOVER_JOB_PREFIX="$PFX"
  echo "g9: $a -> $OUT (workers $W)"
  python3 -u -m nanolab.crossover_replicate launch --out "$OUT" --workers "$W" --arm "$a" || rc=$?
done
echo "g9 launch exit=$rc $(date -u +%FT%TZ)"
python3 scripts/lr_argmin.py crossover_ladder50m --compare crossover_ladder_probe
for b in crossover_ladder50m crossover_ladder1536_argmin; do
  echo "REMINDER: bash scripts/pull_artifacts.sh $b"
done
stage_end g9 "$rc"
