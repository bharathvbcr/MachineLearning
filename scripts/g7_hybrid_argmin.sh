#!/usr/bin/env bash
# G7: the hybrid-vs-attention rows re-measured at each arm's own LR argmin.
#
# E35 put the 8+4 hybrid's argmin at 4x base -- the RECURRENT arm's, not
# attention's 8x. `crossover50m_ratioplace32` ran at 1x, so its beat-attention
# rows read attention from 8x away and the hybrids from 4x away. That is the
# confound; the placement contrasts (hybrid-vs-hybrid at one LR) are common-mode
# and unaffected.
#
# This GROWS ratioplace32 rather than opening a board. `lock_recipe` allows the
# arm list to grow and refuses every field that decides how a job trains, so the
# 1x baseline and the re-tuned arms end up in one directory at one recipe: the
# contrast that answers the objection is within-suite, and no part of it crosses
# the compile boundary. Eager at tenancy 3, both fixed by the lock, not by choice.
#
# Both 4x and 8x are carried for each hybrid shape. E35 measured the 4x argmin at
# w384 and 200M tokens; assuming it transfers to w768 at 50M would repeat exactly
# the error this board exists to correct.
#
# Pre-registered reading, written before the runs:
#   * hybrid's own argmin: lr40 - lr80 < 0 on >=4/5 seeds confirms E35's 4x at
#     this width and budget. If lr80 wins instead, the "majority mixer sets the
#     optimum" claim is width- or horizon-local and says so.
#   * vs attention at each arm's argmin (hybrid lr40 vs attention lr80, whichever
#     multiplier each actually wins at): paired mean < 0 on >=4/5 seeds and
#     |gap| > 0.0031 nats -> the beat-attention rows stand, restated at the argmin.
#     Interval spanning 0 -> rows withdrawn, replaced by "no separation once both
#     arms are tuned". Sign reversal -> section 5 rewritten as attention wins.
#   * Prediction on record from E35's w384/200M mitigation: the hybrid's margin
#     WIDENS relative to the 1x rows already on this board, not vanishes.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g7
stage_start g7
# From the registry, so a launcher cannot list a subset (the RATIO_ARMS rule).
# The six 1x arms are already on disk and are skipped as done.
CROSSOVER_ARMS=$(python3 -c "from nanolab.crossover_replicate import G7_ARMS; print(','.join(G7_ARMS))") || exit 1
export CROSSOVER_ARMS
export CROSSOVER_JOB_PREFIX=cx32p
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover50m_ratioplace32 --workers 3
rc=$?; echo "g7 launch exit=$rc $(date -u +%FT%TZ)"
B=crossover50m_ratioplace32
echo "=== does the hybrid win at 4x here?  (ref = same shape at 8x) ==="
python3 scripts/paired_board.py w768_hybrid_mingru8_attn4_lr40@$B --ref w768_hybrid_mingru8_attn4_lr80@$B
python3 scripts/paired_board.py w768_hybrid_mingru_periodic_lr40@$B --ref w768_hybrid_mingru_periodic_lr80@$B
python3 scripts/paired_board.py w768_attention_lr80@$B --ref w768_attention_lr40@$B
echo "=== hybrids vs attention, each at its argmin (ref = attention 8x) ==="
python3 scripts/paired_board.py w768_hybrid_mingru8_attn4_lr40@$B w768_hybrid_mingru8_attn4_lr80@$B \
  w768_hybrid_mingru_periodic_lr40@$B w768_hybrid_mingru_periodic_lr80@$B --ref w768_attention_lr80@$B
echo "=== the 1x rows this board already carries, for the same contrast ==="
python3 scripts/paired_board.py hybrid_mingru8_attn4@$B hybrid_mingru_periodic@$B --ref attention@$B
echo "REMINDER: bash scripts/pull_artifacts.sh $B"
stage_end g7 "$rc"
