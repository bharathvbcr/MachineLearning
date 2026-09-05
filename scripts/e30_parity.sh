#!/usr/bin/env bash
# E30: the two closers on the 2026-09-04 no-regret hybrid result.
#
#  (a) `attention` INSIDE crossover50m_ratioplace32, so the 8+4-vs-attention
#      curves become a within-suite comparison. The recipe's arm list grows
#      (lock_recipe allows exactly that); tenancy must stay 3, the suite's own.
#  (b) The same hybrids at minGRU expansion 1 (Config.mingru_expand), which
#      takes 8+4 from 147.2M parameters (+19% over attention's 123.7M) to
#      128.3M (+3.8%). Exact parity would need the value-residual projections
#      removed, which is a different arm; say "+3.8%" in the writeup.
#
# Pre-registered: (a) attention per seed within +-0.005 of suite 22's at 50M
# keeps the cross-suite pairing valid. (b) x1 hybrids paired 5/5 vs attention at
# 50M AND no re-crossing on the mean curves -> the no-regret claim survives near
# parity; otherwise the margin was parameters.
source "$(dirname "$0")/_stage_common.sh"
stage_wait; stage_start e30
export CROSSOVER_ARMS=hybrid_mingru11_attn1,hybrid_mingru_periodic,hybrid_mingru_bookend,hybrid_mingru8_attn4,hybrid_mingru10_attn2,attention
export CROSSOVER_JOB_PREFIX=cx32p
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover50m_ratioplace32 --workers 3
rc_a=$?; echo "e30a launch exit=$rc_a $(date -u +%FT%TZ)"
export CROSSOVER_ARMS=attention,hybrid_mingru8_attn4_x1,hybrid_mingru_periodic_x1
export CROSSOVER_JOB_PREFIX=cx32par
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover50m_parity32 --workers 3
rc_b=$?; echo "e30b launch exit=$rc_b $(date -u +%FT%TZ)"
echo "--- (a) the hybrid family against attention, now within-suite"
python3 scripts/paired_board.py hybrid_mingru8_attn4@crossover50m_ratioplace32 hybrid_mingru_periodic@crossover50m_ratioplace32 --ref attention@crossover50m_ratioplace32
python3 scripts/paired_board.py attention@crossover50m_ratioplace32 --ref attention@crossover50m
echo "--- (b) near-parity hybrids against attention, within-suite"
python3 scripts/paired_board.py hybrid_mingru8_attn4_x1@crossover50m_parity32 hybrid_mingru_periodic_x1@crossover50m_parity32 --ref attention@crossover50m_parity32
echo "REMINDER: bash scripts/pull_artifacts.sh crossover50m_ratioplace32 ; bash scripts/pull_artifacts.sh crossover50m_parity32"
rc=$rc_a; [ "$rc_b" -ne 0 ] && rc=$rc_b
stage_end e30 "$rc"
