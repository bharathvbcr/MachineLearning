#!/usr/bin/env bash
# E28 (memo item 14): does the GDN rule defect change recall?
#
# `nanolab/mixers.py` implements the gated delta rule with the correction read
# off the UNDECAYED state; the published operator (arXiv:2412.06464 eq. 8) reads
# the decayed one. Every GDN number in this repo is about the repo's variant.
# `gdn_pub` and `hybrid_gdn_periodic_pub` carry the published rule behind
# `gdn_rule="published"`, at zero throughput cost (measured today: both variants
# run 27.8K tok/s at the 50M board shape, identical to four significant figures).
#
# Tenancy here is the OPPOSITE of the 50M boards. At ~9.5M parameters and
# sequence 31 a single run does leave the GPU idle, exactly as mqar_suite's
# --workers help claims: measured today, the same cell takes 613 s at workers 1
# and 229 s at 4, a 2.68x gain. So this runs at 4, unlike anything at 124M.
#
# The two 3000-step cells only. The 9000-step cells are 3x the cost and belong in
# a window that can finish them; the ledger is cumulative, so they append later
# without rerunning these.
#
# PRE-REGISTERED READOUT (handoff section 5.2): if `gdn_pub` solves at least 4/15
# more seeds than `gdn` at p=8/9k, or any seed at seq 255, the variant was the
# defect and every downstream GDN claim is re-based on the published rule. If not,
# the hard-recall failure is capacity or optimisation, and the fast-weight line
# proceeds on the published rule anyway. Either outcome is a result.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
ARMS=gdn,gdn_pub,hybrid_gdn_periodic,hybrid_gdn_periodic_pub

echo "=== E28 smoke: the new arms must build and step before the board $(date -u +%FT%TZ) ==="
python3 -u -m nanolab.crossover_replicate smoke --arms gdn_pub,hybrid_gdn_periodic_pub 2>&1 | tail -8
rc=${PIPESTATUS[0]}
if [ "$rc" != "0" ]; then
  echo "smoke FAILED rc=$rc -- refusing to launch the board"
  echo "e28 exit=1 $(date -u +%FT%TZ)"
  exit 1
fi

for PAIRS in 4 8; do
  echo "--- E28 cell: pairs=$PAIRS steps=3000 batch=256 workers=4 $(date -u +%FT%TZ) ---"
  s=$(date +%s)
  # One directory per cell. Both cells previously wrote to nanolab/out/mqar_e8,
  # which put two `pairs` recipes in one place AND appended into a results file
  # that already held earlier MQAR work under the same arm names. Nothing was
  # lost -- runs.jsonl is append-only and the run name carries pairs/batch/steps
  # and the lr rule -- but `pairs` is not a field on the record, so separating
  # the cells afterwards means parsing run names. Separate them up front.
  python3 -u -m nanolab.mqar_suite --out "nanolab/out/mqar_e28_p$PAIRS" --device cuda \
    --arms "$ARMS" --pairs "$PAIRS" --steps 3000 --batch 256 \
    --seeds 15 --lr-rule sqrt --workers 4 --gpus 1 2>&1 | tail -25
  echo "e28 cell pairs=$PAIRS elapsed=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)"
done

echo "e28 exit=0 $(date -u +%FT%TZ)"
