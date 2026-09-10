#!/usr/bin/env bash
# E29: E19's parameter board with a router that can learn to route.
#
# Every committed moe_e*k1 run renormalised its top-1 weight to exactly 1, so
# the gate never saw a task gradient (nanolab.tests::moe_top1_router_task_gradient
# reproduces it). `*_raw` arms use the Switch weighting (Config.moe_router_weight).
# NEW out dir: pre-fix runs must never be pooled with these by a resume.
#
# The control decides readability, exactly as in e19b: moe_e1k1_raw is a dense
# SwiGLU through the MoE path and must land inside `attention`'s interval.
# Pre-registered: moe_e8k1_raw minus attention (paired) at or above -0.016 ->
# parameters still not the bottleneck at 0.4 tokens/param; disjointly better ->
# E19's conclusion was a router artifact.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused e29
stage_start e29
export CROSSOVER_ARMS=attention,moe_e1k1_raw,moe_e4k1_raw,moe_e8k1_raw
export CROSSOVER_JOB_PREFIX=cx32moed
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover50m_moe32d --workers 2
rc=$?; echo "e29 launch exit=$rc $(date -u +%FT%TZ)"
python3 -u -m nanolab.crossover_replicate table --out nanolab/out/crossover50m_moe32d 2>&1 | tail -20
python3 scripts/paired_board.py moe_e1k1_raw@crossover50m_moe32d moe_e4k1_raw@crossover50m_moe32d moe_e8k1_raw@crossover50m_moe32d --ref attention@crossover50m_moe32d
echo "val_aux is logged per eval, not subtracted: 0.12 at perfect balance (0.01 x 12 layers); above that the router is collapsing."
echo "REMINDER: bash scripts/pull_artifacts.sh crossover50m_moe32d  (from the laptop)"
stage_end e29 "$rc"
