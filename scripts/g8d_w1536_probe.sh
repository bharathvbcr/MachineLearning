#!/usr/bin/env bash
# G8d: w1536's own 50M LR probe. Added after G8a came back, not planned with it.
#
# G9 was written to let w1536 inherit a multiplier when the probed widths agreed, on
# the grounds that the 10M probe found the argmin width-invariant (8x for attention and
# 4x for minGRU at every width). G8a then measured the 50M grid and found **d384 at 2x
# and d768 at 1x**. Width-invariance holds at a short horizon and breaks at a long one,
# so there is nothing for w1536 to inherit and the inheritance path has been removed
# from the planner entirely: every width is now planned from its own probe.
#
# Without this stage the re-tuned width claim would span three rungs instead of four,
# and the top rung -- the 13.55M endpoint of "+27% across 10.3x params" -- would be the
# one missing. Ten jobs to keep it.
#
# Tenancy 1: w1536 minGRU OOMed two-to-a-device on 2026-09-05, which is why that arm
# lives on its own board. Five LR points, 0.25x..4x; 8x is rejected decisively in every
# 50M cell measured so far and a w1536 job is expensive.
#
# Pre-registered reading: the argmin per mixer, interior or refused. Combined with d384
# 2x and d768 1x this is also the first read on whether the 50M argmin falls MONOTONE
# with width -- which, if it does, is a result about muP transfer at long horizon and
# not merely a correction to this paper's tuning.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g8d
stage_start g8d
CROSSOVER_ARMS=$(python3 -c "from nanolab.crossover_replicate import G8D_PROBE_ARMS; print(','.join(G8D_PROBE_ARMS))") || exit 1
export CROSSOVER_ARMS
export CROSSOVER_JOB_PREFIX=cx32p1536
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover_probe50m_w1536 --workers 1 --seed 1337
rc=$?; echo "g8d launch exit=$rc $(date -u +%FT%TZ)"
python3 scripts/lr_argmin.py crossover_probe50m_w1536
echo "=== the 50M argmin across every probed width ==="
for b in crossover_probe50m crossover_ladder50m crossover_probe50m_w1536; do
  python3 scripts/lr_argmin.py "$b" 2>/dev/null | grep -E "^  d|argmin"
done
echo "REMINDER: bash scripts/pull_artifacts.sh crossover_probe50m_w1536"
stage_end g8d "$rc"
