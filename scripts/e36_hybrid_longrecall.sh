#!/usr/bin/env bash
# E36: the hybrid on the long-sequence recall cell it has never been run on.
#
# THE GAP. Of 922 MQAR runs in this repo, zero are a hybrid at block_size >= 255.
# Hybrid recall coverage stops at seq 31. Meanwhile the seq-255 cell is where the
# recurrent arms do not merely degrade, they return nothing: attention 11/15,
# minGRU 0/15 [0.00, 0.20], GDN 0/15 -- 80 recurrent runs at seq >= 255 and not
# one solve. The paper's best arm on cross-entropy (`hybrid_mingru8_attn4` beats
# dense attention by 0.0165 nats, and by 0.0053 at parameter parity) is therefore
# unmeasured on the one axis where its recurrent half fails categorically. That
# is the first question a reader asks and the repo cannot answer it.
#
# E32 does not close this. E32 runs hybrids at pairs 4 and 8 -- seq 15 and 31 --
# which is the regime where the hybrids already look good (12/15 at p=8/3k against
# attention's 2/15). Short-cell strength is exactly what must not be extrapolated
# to seq 255, because that is the extrapolation the pure arms fail.
#
# THE CELL. Recipe-identical to the existing seq-255 cell in mqar_e16 so the new
# arms pool with the attention/minGRU/GDN rows already there: pairs 64
# (block_size = 4*pairs-1 = 255), batch 64, 3000 steps, 15 seeds, sqrt LR rule.
# Batch 64 is not a default -- E16's calibration measured larger batches strictly
# worse at this cell (solves by batch: 64->3, 128->2, 256->2, 512->0) and refused
# to name a saturating batch at all. `attention` is listed in --arms on purpose:
# its 15 runs are already done, so the suite skips them by name and costs nothing,
# and the printed board then carries the control beside the new arms instead of
# requiring a join after the fact.
#
# PRE-REGISTERED READING, written before the runs:
#   * solve rate inside attention's Wilson interval [0.48, 0.89] at 15 seeds
#     -> the hybrid inherits attention's recall; "no regret" holds on both metrics
#        and the CE win in E10/E30 is not bought with a capability.
#   * solve rate inside the recurrent arms' [0.00, 0.20]
#     -> the CE win IS bought with the capability, and every held-out-loss
#        comparison in this repo is blind to what the arm gave up.
#   * anything strictly between the two intervals
#     -> a graded result, reportable as such, and the cell needs more seeds
#        before it carries a claim. Do not round it toward either pre-registration.
# Three arms: the CE winner (`periodic`, 3 interleaved), the parity winner
# (`8+4`, 4 stacked), and `10+2`, which is the only hybrid with existing short-cell
# recall data and so the only one whose short-to-long transfer can be checked.
#
# WAITING. One waiter, on `tune_followup`, which is itself the single waiter on
# `tune_program`. Chaining here rather than on the program's own marker keeps that
# invariant: two chains keyed to one log string is what once fired E27's probe and
# three val_m3 workers onto a 94.5 GiB card at the same instant. `workers` is a
# recorded recipe field, so a fourth process on the device would silently make
# every running board a 4-tenancy board whose recipe.json says otherwise.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
LOG=nanolab/out/_followup.log

if [ "${E36_NOWAIT:-0}" != "1" ]; then
  echo "e36 waiting for tune_followup ($(date -u +%FT%TZ))"
  deadline=$(( $(date +%s) + 24*3600 ))
  while ! grep -q "^tune_followup exit=" "$LOG" 2>/dev/null; do
    [ "$(date +%s)" -gt "$deadline" ] && { echo "gave up waiting at $(date -u +%FT%TZ)"; exit 1; }
    sleep 120
  done
fi

quiet=0
while true; do
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
    quiet=0
  else
    quiet=$((quiet+1)); [ "$quiet" -ge 3 ] && break
  fi
  sleep 20
done

echo "e36 start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
s=$(date +%s)
python3 -u -m nanolab.mqar_suite \
  --out nanolab/out/mqar_e16 --device cuda \
  --cells 64 --batch 64 --steps 3000 --seeds 15 --lr-rule sqrt \
  --arms attention,hybrid_mingru8_attn4,hybrid_mingru_periodic,hybrid_mingru10_attn2 \
  --workers 4 --gpus 1
rc=$?
echo "REMINDER: bash scripts/pull_artifacts.sh mqar_e16"
echo "e36 exit=$rc elapsed=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)"
exit "$rc"
