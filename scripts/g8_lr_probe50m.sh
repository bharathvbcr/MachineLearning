#!/usr/bin/env bash
# G8a: where is the LR argmin at the budget the boards actually spend?
#
# `scripts/e21_ladder_probe.sh` recorded this as an open assumption when it was
# written: "the best LR at 10M is not guaranteed to be the best at 50M, so phase
# 2 inherits an assumption rather than a measurement." G7 broke it. At w768/50M
# loss rises monotonically with LR for attention (4.2181 -> 4.3487 -> 4.4732 over
# 1x/4x/8x, 0/5 seeds at both steps) and for both hybrid shapes, while the 10M
# probe had attention's minimum AT 8x. The ladder therefore spent 50M at an LR
# chosen from a 610-step run, and at w768 that is 0.255 nats off the best LR
# measured at 3051 steps.
#
# This mirrors the 10M probe exactly -- same grid shape, same n=1 at seed 1337,
# same eager recipe, batch 32 / block 512 / eval_iters 20 -- so the only thing
# that differs between the two boards is the token budget and the comparison is
# one subtraction. It is its own directory rather than a growth of an existing
# board because it is its own artifact: the 50M twin of `crossover_ladder_probe`.
#
# The grid runs DOWN to 0.25x. E21's first sweep ran off the top edge and had
# 4.0/8.0 appended; every 50M grid to date runs off the bottom edge instead. An
# argmin at the edge is not an argmin, in either direction.
#
# Pre-registered reading, written before the runs:
#   * INTERIOR OR NOTHING. If any arm's minimum sits at 0.25x, that arm's grid
#     extends again and no argmin is claimed for it -- the same rule that added
#     4.0 and 8.0 at 10M.
#   * Width-invariance: w384 and w768 picking the same multiplier for a mixer
#     means the 50M argmin transfers across width as the 10M one did, and one
#     probe re-tunes the whole ladder. Different multipliers means every rung
#     needs its own probe and the ladder's cost triples.
#   * Arm-invariance: attention, minGRU and both hybrid shapes picking the same
#     multiplier RETIRES the confound E35 inferred -- a shared-LR board would
#     then not be reading arms from different distances, and section 5's rows
#     stand as measured. Different multipliers means section 5 is re-run per arm.
#   * Horizon shift: 50M argmin minus 10M argmin, same width, mixer and seed, is
#     the number sections 3 and 4 need in order to say how far off they were.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g8a
stage_start g8a
CROSSOVER_ARMS=$(python3 -c "from nanolab.crossover_replicate import PROBE50M_ARMS; print(','.join(PROBE50M_ARMS))") || exit 1
export CROSSOVER_ARMS
export CROSSOVER_JOB_PREFIX=cx32p50
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover_probe50m --workers 3 --seed 1337
rc=$?; echo "g8a launch exit=$rc $(date -u +%FT%TZ)"
python3 scripts/lr_argmin.py crossover_probe50m --compare crossover_ladder_probe
echo "REMINDER: bash scripts/pull_artifacts.sh crossover_probe50m"
stage_end g8a "$rc"
