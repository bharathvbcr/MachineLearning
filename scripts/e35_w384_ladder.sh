#!/usr/bin/env bash
# E35 rung 1: the token ladder at width 384, 200M tokens = ~5 tokens/param on
# the 40.6M attention shape (0.4 epochs of the 497.5M corpus). Every 50M board
# sits at 0.4 tokens/param; this is the first point outside that regime.
# Arms at the LRs E21 measured (attention x8, minGRU x4) and the 8+4 hybrid at
# both, since it has no measured optimum. Rung 2 (800M = 1.6 epochs, revisits
# data) is NOT launched here: change the budget and dir deliberately, and record
# the epoch count in the writeup.
# Pre-registered: does the 50M crossing token scale with the budget, and what
# is the ordering at ~5 tokens/param.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused e35
stage_start e35
export CROSSOVER_TOKEN_BUDGET=200000000
export CROSSOVER_ARMS=w384_attention_lr80,w384_mingru_lr40,w384_hybrid_mingru8_attn4_lr40,w384_hybrid_mingru8_attn4_lr80
export CROSSOVER_JOB_PREFIX=cx32w384x4
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover200m_w384 --workers 2
rc=$?; echo "e35 launch exit=$rc $(date -u +%FT%TZ)"
python3 scripts/paired_board.py w384_mingru_lr40@crossover200m_w384 w384_hybrid_mingru8_attn4_lr40@crossover200m_w384 w384_hybrid_mingru8_attn4_lr80@crossover200m_w384 --ref w384_attention_lr80@crossover200m_w384 --markers 12.3e6,25e6,50e6,100e6,150e6
echo "REMINDER: bash scripts/pull_artifacts.sh crossover200m_w384"
stage_end e35 "$rc"
