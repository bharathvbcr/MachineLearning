#!/usr/bin/env bash
# E33: depth for width at ~21M non-embedding parameters, the shape axis a
# latency budget cares about (launches per token) and no board has run.
# attn3 (3L x 768) is 21.3M non-embedding and was measured in crossover50m_loop32
# at this recipe and tenancy; 12L x 384 is 21.2M (`w384_attention_lr10` = the
# shared base LR); 6L x 512 = 18.9M and 6L x 576 = 23.9M bracket it (head_dim
# stays 64, so 544 is unreachable). Shared LR is a known limitation: E21
# measured different LR optima per shape; report, do not tune.
# Pre-registered: 6L within 0.05 of 12L -> 6 layers is the latency shape;
# 3L within 0.05 -> 3.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused e33
stage_start e33
export CROSSOVER_ARMS=attention,attn6_w512,attn6_w576,w384_attention_lr10
export CROSSOVER_JOB_PREFIX=cx32shape
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover50m_shape32 --workers 3
rc=$?; echo "e33 launch exit=$rc $(date -u +%FT%TZ)"
python3 scripts/paired_board.py attn6_w512@crossover50m_shape32 attn6_w576@crossover50m_shape32 w384_attention_lr10@crossover50m_shape32 attn3@crossover50m_loop32 --ref attention@crossover50m_shape32
echo "REMINDER: bash scripts/pull_artifacts.sh crossover50m_shape32"
stage_end e33 "$rc"
