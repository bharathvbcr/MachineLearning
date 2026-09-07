#!/usr/bin/env bash
# The two follow-ups that must not run while a board does, in ONE chain.
#
# Both need the program finished, and there must be exactly ONE waiter on that
# marker. Two chains keyed to the same log string is not a hypothetical failure
# here -- it is what fired three val_m3 workers and E27's width-1536 probe at the
# same instant and OOMed both on a 94.5 GiB card. So the wait happens once, in
# this driver, and each stage is invoked with FOLLOWUP_NOWAIT=1.
#
# Why they wait at all: `workers` is a recorded recipe field. A fourth process on
# the device would make every running board a 4-tenancy board whose elapsed_s and
# tok_s describe a recipe its own recipe.json does not.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
LOG=nanolab/out/_program.log

echo "followup waiting for tune_program ($(date -u +%FT%TZ))"
deadline=$(( $(date +%s) + 20*3600 ))
while ! grep -q "^tune_program exit=" "$LOG" 2>/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "gave up waiting at $(date -u +%FT%TZ)"; exit 1; }
  sleep 120
done
quiet=0
while true; do
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
    quiet=0
  else
    quiet=$((quiet+1)); [ "$quiet" -ge 3 ] && break
  fi
  sleep 20
done
echo "device idle; follow-ups start $(date -u +%FT%TZ)"
export FOLLOWUP_NOWAIT=1

# The diagnostic first: it decides whether E31's untied half is salvageable by
# re-evaluation or needs 10 jobs of GPU, and it costs a few forward passes.
bash scripts/tune_e31_untied_diag.sh 2>&1 | tee nanolab/out/_e31diag.log
echo "followup e31diag exit=${PIPESTATUS[0]} $(date -u +%FT%TZ)"

bash scripts/tune_e17_boardshape.sh 2>&1 | tee nanolab/out/_e17bs.log
echo "followup e17 exit=${PIPESTATUS[0]} $(date -u +%FT%TZ)"

echo "tune_followup exit=0 $(date -u +%FT%TZ)"
