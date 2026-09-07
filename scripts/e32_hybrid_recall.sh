#!/usr/bin/env bash
# E32: the no-regret hybrid on the recall grid it was never measured on.
# hybrid_mingru10_attn2 reached 13/15 at p=8/9k (attention 12/15); 8+4 has no
# recall cell. Same ledger and cells as E8.
# Pre-registered: rate inside attention's Wilson interval at every cell ->
# no-regret on both metrics.
source "$(dirname "$0")/_stage_common.sh"
stage_wait; stage_start e32
rc_any=0
for STEPS in 3000 9000; do for PAIRS in 4 8; do
  python3 -u -m nanolab.mqar_suite --out nanolab/out/mqar_e8 --device cuda \
    --arms attention,hybrid_mingru10_attn2,hybrid_mingru8_attn4 \
    --pairs "$PAIRS" --steps "$STEPS" --batch 256 --seeds 15 --lr-rule sqrt --workers 4 --gpus 1
  rc=$?; echo "--- cell p=$PAIRS steps=$STEPS exit=$rc $(date -u +%FT%TZ)"; [ "$rc" -ne 0 ] && rc_any=$rc
done; done
echo "REMINDER: bash scripts/pull_artifacts.sh mqar_e8"
stage_end e32 "$rc_any"
