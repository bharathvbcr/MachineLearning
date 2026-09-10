#!/usr/bin/env bash
# G3: does the LR argmin move with the HORIZON?
#
# Every argmin in the ladder was located at 10M tokens on ONE seed and then spent
# at 50M. Argmin-invariance across width is measured; across horizon it is not,
# and it is load-bearing for every crossing in sections 3 and 4. Two of the six
# probe curves are also nearly flat at the bottom: minGRU's 4x beats 8x by only
# 0.0121 at w384 and beats 2x by 0.0055 at w1152, against a 0.0031-nat rerun
# floor -- at n=1 those are not resolved argmins.
#
# This GROWS `crossover_ladder50m` rather than opening a board: that directory
# already holds every argmin arm at n=5 under the identical recipe, so adding the
# neighbours puts the whole LR grid in one suite at 50M with no pooling argument.
# Eager at tenancy 2, both fixed by the lock. 30 new jobs; the six arms already on
# disk are skipped. Reallocating those six saved jobs buys the w384 rung, so both
# thin minGRU cases are covered rather than one.
#
# Pre-registered reading, written before the runs:
#   * An argmin has MOVED when a neighbour beats the incumbent on >=4/5 seeds AND
#     the paired mean gap exceeds the 0.0031-nat floor. Anything less is a tie and
#     is reported as a tie, not as invariance.
#   * No argmin moves -> the ladder's tuning is horizon-invariant over 5x in
#     budget and sections 3 and 4 stand as written, caveat removed.
#   * Any argmin moves -> the affected rungs were measured at a mistuned point and
#     must be re-run at the 50M argmin before the width claim can be stated. The
#     paper's width result is withdrawn to a caveat until they are.
#   * minGRU at w1152 is expected to be the ambiguous one (2x vs 4x differ by
#     0.0055 at 10M). A tie there is a real outcome, not a failed measurement, and
#     it bounds how precisely the ladder's minGRU arms were ever tuned.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g3
stage_start g3
CROSSOVER_ARMS=$(python3 -c "from nanolab.crossover_replicate import G3_ARMS; print(','.join(G3_ARMS))") || exit 1
export CROSSOVER_ARMS
export CROSSOVER_JOB_PREFIX=cx32lad50
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover_ladder50m --workers 2
rc=$?; echo "g3 launch exit=$rc $(date -u +%FT%TZ)"
B=crossover_ladder50m
echo "=== attention at w1152: is 8x still the argmin at 50M? (ref = lr80) ==="
python3 scripts/paired_board.py w1152_attention_lr40@$B w1152_attention_lr160@$B --ref w1152_attention_lr80@$B
echo "=== minGRU at w1152: is 4x still the argmin at 50M? (ref = lr40) ==="
python3 scripts/paired_board.py w1152_mingru_lr20@$B w1152_mingru_lr80@$B --ref w1152_mingru_lr40@$B
echo "=== minGRU at w384: is 4x still the argmin at 50M? (ref = lr40) ==="
python3 scripts/paired_board.py w384_mingru_lr20@$B w384_mingru_lr80@$B --ref w384_mingru_lr40@$B
echo "REMINDER: bash scripts/pull_artifacts.sh $B"
stage_end g3 "$rc"
