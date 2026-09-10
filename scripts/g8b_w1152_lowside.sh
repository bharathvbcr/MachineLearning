#!/usr/bin/env bash
# G8b: the w1152 low side, so the 50M argmin's width-invariance is checked over
# 3x in width rather than 2x.
#
# G3 already put lr20/40/80 on the ladder board for minGRU at w1152 and
# lr40/80/160 for attention; only the bottom half of the grid is missing. This
# grows `crossover_ladder50m` through the arms-may-grow seam, at n=1 and seed
# 1337 to match the probe, so the three widths are read the same way. Tenancy 2
# and eager are fixed by the lock, not chosen.
#
# Pre-registered reading: the same three tests as G8a, with w1152 added to the
# width-invariance one. If w384, w768 and w1152 agree, one number re-tunes the
# whole ladder; if they disagree, the paper's width result needs a per-rung probe
# before it can be restated at all.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g8b
stage_start g8b
CROSSOVER_ARMS=$(python3 -c "from nanolab.crossover_replicate import G8B_ARMS; print(','.join(G8B_ARMS))") || exit 1
export CROSSOVER_ARMS
export CROSSOVER_JOB_PREFIX=cx32lad50
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover_ladder50m --workers 2 --seed 1337
rc=$?; echo "g8b launch exit=$rc $(date -u +%FT%TZ)"
python3 scripts/lr_argmin.py crossover_ladder50m --compare crossover_ladder_probe
echo "REMINDER: bash scripts/pull_artifacts.sh crossover_ladder50m"
stage_end g8b "$rc"
