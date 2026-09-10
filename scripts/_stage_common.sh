# Sourced by the e28..e35 and g* stage scripts. Not executable on its own.
#
# WAIT: optional log-marker chaining. `PREV=e30 bash scripts/e31_parity.sh`
# waits until nanolab/out/e30.log carries "e30 exit=" -- the marker the work
# itself writes -- and never on pgrep (see scripts/overnight.sh for why).
#
# The marker is a HINT. The device is the truth, and the two are checked
# separately, because on 2026-09-09 they disagreed and the cost was two stages:
#
#   `stage_wait` used to give up after PREV_HOURS and print "starting anyway".
#   g10's marker never came (its predecessor g11b runs ~5 h and g10 had been
#   queued far longer), so g4 started anyway -- onto a GH200 already carrying
#   g11b's workers. All four of g4's MQAR workers died of CUDA OOM. g4's own
#   exit marker then released g5, which launched into the same contention and
#   lost 12 of its 18 family arms the same way. One timeout, two stages.
#
#   "Starting anyway" is the failure mode the repo's own rule names: a check
#   that could not run must never behave like one that ran and passed. A
#   predecessor that has not finished is not a predecessor that finished.
#
# So: an overrun on the MARKER is now loud but non-fatal -- the predecessor may
# genuinely be wedged, and hanging forever is its own failure -- while the
# DEVICE check that follows it is what actually gates the launch, and that one
# refuses. A stage that cannot get an idle device writes its own exit marker and
# stops, so the chain drains as refusals instead of piling workers onto one GPU.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"

# Compute processes currently resident on the GPU, one PID per line.
device_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9]+$' || true
}

stage_wait() {
  local prev="${PREV:-}"
  if [ -n "$prev" ]; then
    local deadline=$(( $(date +%s) + ${PREV_HOURS:-12}*3600 ))
    while ! grep -q "^${prev} exit=" "nanolab/out/${prev}.log" 2>/dev/null; do
      if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "$prev has not written its exit marker after ${PREV_HOURS:-12}h" \
             "$(date -u +%FT%TZ) -- falling through to the device check, which" \
             "will refuse if $prev is still holding the GPU"
        break
      fi
      sleep 120
    done
  fi
  # Whether or not a marker arrived, do not launch onto an occupied device.
  local settle=$(( $(date +%s) + ${DEVICE_WAIT_HOURS:-24}*3600 ))
  local pids
  while :; do
    pids=$(device_pids)
    [ -z "$pids" ] && return 0
    if [ "$(date +%s)" -ge "$settle" ]; then
      echo "REFUSING to start: the device has been busy for" \
           "${DEVICE_WAIT_HOURS:-24}h and still carries PIDs" \
           "[$(echo $pids | tr '\n' ' ')] $(date -u +%FT%TZ)"
      return 75
    fi
    sleep 120
  done
}

# `stage_wait || stage_refused NAME` is the calling convention; a stage that
# skips the guard is one bad night away from repeating 2026-09-09.
stage_refused() { echo "$1 exit=75 $(date -u +%FT%TZ)"; exit 75; }

stage_start() { echo "$1 start $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true; }
stage_end() { echo "$1 exit=$2 $(date -u +%FT%TZ)"; exit "$2"; }
# Every crossover board in this program shares the 50M recipe unless a script
# overrides a field AFTER sourcing this file.
export CROSSOVER_BATCH=32
export CROSSOVER_EVAL_ITERS=20
export CROSSOVER_TOKEN_BUDGET=50000000
