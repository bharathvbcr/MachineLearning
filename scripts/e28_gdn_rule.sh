#!/usr/bin/env bash
# E28: is the repo's GDN hard-recall failure the operator variant?
#
# nanolab's gated delta rule reads the UNDECAYED state in its correction
# (S <- aS + b(v - Sk)k^T); the published rule (arXiv:2412.06464 eq. 8) reads
# the decayed one. `gdn_pub` / `hybrid_gdn_periodic_pub` are the same blocks on
# the published rule (Config.gdn_rule). Same E8 grid, same ledger: the existing
# arms are skipped off nanolab/out/mqar_e8/runs.jsonl and listed only so the
# report prints the comparison. Then the E16 seq-255 cell, where every
# recurrent arm scored 0/15.
#
# Pre-registered: gdn_pub >= gdn + 4/15 at p=8/9k, or any solved seed at
# seq 255 -> the variant was the defect. Otherwise capacity/optimisation, and
# the fast-weight line proceeds on the published rule regardless.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused e28
stage_start e28
rc_any=0
for STEPS in 3000 9000; do for PAIRS in 4 8; do
  echo "--- cell p=$PAIRS steps=$STEPS start $(date -u +%FT%TZ)"
  python3 -u -m nanolab.mqar_suite --out nanolab/out/mqar_e8 --device cuda \
    --arms gdn,gdn_pub,hybrid_gdn_periodic,hybrid_gdn_periodic_pub \
    --pairs "$PAIRS" --steps "$STEPS" --batch 256 --seeds 15 --lr-rule sqrt \
    --workers 4 --gpus 1
  rc=$?; echo "--- cell p=$PAIRS steps=$STEPS exit=$rc $(date -u +%FT%TZ)"
  [ "$rc" -ne 0 ] && rc_any=$rc
done; done
echo "--- seq-255 cell start $(date -u +%FT%TZ)"
python3 -u -m nanolab.mqar_suite --out nanolab/out/mqar_e16_board --device cuda \
  --cells 64 --batch 64 --arms attention,gdn,gdn_pub \
  --seeds 15 --steps 3000 --lr-rule sqrt --workers 4 --gpus 1
rc=$?; echo "--- seq-255 cell exit=$rc $(date -u +%FT%TZ)"; [ "$rc" -ne 0 ] && rc_any=$rc
echo "REMINDER: bash scripts/pull_artifacts.sh mqar_e8 ; bash scripts/pull_artifacts.sh mqar_e16_board  (from the laptop)"
stage_end e28 "$rc_any"
