#!/usr/bin/env bash
# G6: a fifth ladder rung at d_model 1920, probe then rung.
#
# Four points cannot distinguish a crossing token that keeps climbing with width from
# one that saturates, which is why the paper declines to claim a functional form. Five
# cannot fit one either -- but five can say whether the fourth-to-fifth step is the same
# size as the third-to-fourth, which is the question a reviewer actually asks.
#
# This runs AFTER G9 has re-measured the first four rungs at their own 50M argmins, so
# the fifth point is comparable to them rather than to the 10M-derived protocol they
# were originally spent at. Running it earlier would have produced a fifth point in the
# old protocol and answered nothing.
#
# Tenancy 1 throughout. 1920 is the widest shape this program has trained and w1536
# minGRU already OOMed two-to-a-device; a refused plan costs a worker slot, an OOM at
# this width costs the board. head_dim stays 64, so width still moves through n_head
# alone (30 heads) -- the invariant the ladder test enforces.
#
# Pre-registered reading: the rung's crossing against the re-measured d1536 one. A step
# of the same size as 1152->1536 means the trend has not saturated by 417M parameters;
# a materially smaller one means it is flattening and the paper says so. Either way the
# paper still declines to fit a form to five points.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g6
stage_start g6
PROBE=nanolab/out/crossover_probe50m_w1920
export CROSSOVER_ARMS=$(python3 -c "from nanolab.crossover_replicate import G6_PROBE_ARMS; print(','.join(G6_PROBE_ARMS))")
export CROSSOVER_JOB_PREFIX=cx32p1920
python3 -u -m nanolab.crossover_replicate launch --out "$PROBE" --workers 1 --seed 1337
rc=$?; echo "g6 probe exit=$rc $(date -u +%FT%TZ)"
python3 scripts/lr_argmin.py crossover_probe50m_w1920
PLAN=$(python3 scripts/g9_plan.py --width 1920 --board crossover_probe50m_w1920 --arms)
prc=$?
if [ "$prc" -ne 0 ] || [ -z "$PLAN" ]; then
  echo "g6: planner refused the d1920 rung; the probe stands, the rung does not run"
  python3 scripts/g9_plan.py --width 1920 --board crossover_probe50m_w1920
  stage_end g6 2
fi
echo "g6 plan: $PLAN"
export CROSSOVER_ARMS="$PLAN"
export CROSSOVER_JOB_PREFIX=cx32lad1920
for a in ${PLAN//,/ }; do
  python3 -u -m nanolab.crossover_replicate launch \
    --out nanolab/out/crossover_ladder1920 --workers 1 --arm "$a" || rc=$?
done
echo "g6 rung exit=$rc $(date -u +%FT%TZ)"
echo "REMINDER: bash scripts/pull_artifacts.sh crossover_probe50m_w1920 && bash scripts/pull_artifacts.sh crossover_ladder1920"
stage_end g6 "$rc"
