#!/usr/bin/env bash
# The open half of the September program, in one serial driver.
#
# Boards: E31, E29, E33, E30(a+b), E34, E32, E35. Every one of them was written
# on 2026-09-05 and has never run.
#
# Three things this driver does that the chain of independent waiters did not:
#
#   * ONE stage at a time. `launch` blocks until its queue drains, so the
#     sequencing is just sequence -- no markers, no pgrep, no waiters firing
#     together on a shared string. Two of those once fired at once and OOMed
#     E27's probe.
#   * REAL exit codes. queue_rest.sh echoed a hardcoded "exit=0" after each
#     stage; that turned two crashes into a clean-looking log and left the card
#     idle for 54 minutes. Every stage here reports what actually happened, and
#     a failure does not stop the queue -- the next board is independent of it.
#   * compile ON, which is the whole reason this is worth running now rather
#     than last week: 1.94x on attention, 1.96x on minGRU, 1.95x on both
#     hybrids. It is a recorded recipe field, so no compiled run can ever pool
#     with an eager one. E30's phase (a) sets it back to 0 for itself; see the
#     comment in that script.
#
# The idle guard asks nvidia-smi, not pgrep: `pgrep -f` matches whole command
# lines, so the very shell running this script matches a pattern containing its
# own text. That cost 48 minutes on 2026-09-06.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
OUT=nanolab/out
mkdir -p "$OUT"

wait_for_idle() {
  local quiet=0
  echo "waiting for an idle device ($(date -u +%FT%TZ))"
  while true; do
    if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
      quiet=0
    else
      quiet=$((quiet+1))
      [ "$quiet" -ge 3 ] && break
    fi
    sleep 20
  done
  echo "device idle at $(date -u +%FT%TZ)"
}

stage() {                       # stage <name> <script>
  local name="$1" script="$2" t0 rc
  echo ""
  echo "################################################################"
  echo "##  $name  $(date -u +%FT%TZ)"
  echo "################################################################"
  t0=$(date +%s)
  bash "$script" 2>&1 | tee "$OUT/${name}.log"
  rc=${PIPESTATUS[0]}
  echo "$name exit=$rc elapsed=$(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)"
  return 0
}

wait_for_idle
export CROSSOVER_COMPILE=1

# Cheapest and cleanest first, so a late failure cannot cost the whole queue.
stage e31 scripts/e31_tie.sh          # tied/untied x value residual, 2x2
stage e29 scripts/e29_moe_raw.sh      # E19's board with a router that can learn
stage e33 scripts/e33_shape.sh        # depth for width at ~21M non-embedding
stage e30 scripts/e30_parity.sh       # within-suite hybrids, then parity at x1
stage e34 scripts/e34_copy.sh         # is the crossing the induction-head bump
stage e32 scripts/e32_hybrid_recall.sh  # the no-regret hybrid on MQAR
stage e35 scripts/e35_w384_ladder.sh  # 200M tokens at width 384 -- the long one

echo ""
echo "===== program summary $(date -u +%FT%TZ) ====="
for s in e31 e29 e33 e30 e34 e32 e35; do
  printf '  %-5s %s\n' "$s" \
    "$(grep -h "^$s exit=" "$OUT/${s}.log" 2>/dev/null | tail -1 || echo 'no marker')"
done
echo "tune_program exit=0 $(date -u +%FT%TZ)"
