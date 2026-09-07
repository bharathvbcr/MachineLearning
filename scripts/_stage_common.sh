# Sourced by the e28..e35 stage scripts. Not executable on its own.
#
# WAIT: optional log-marker chaining. `PREV=e29 bash scripts/e30_parity.sh`
# waits until nanolab/out/e29.log carries "e29 exit=" -- the marker the work
# itself writes -- and never on pgrep (see scripts/overnight.sh for why).
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
stage_wait() {
  local prev="${PREV:-}"
  [ -z "$prev" ] && return 0
  local deadline=$(( $(date +%s) + ${PREV_HOURS:-12}*3600 ))
  while ! grep -q "^${prev} exit=" "nanolab/out/${prev}.log" 2>/dev/null; do
    [ "$(date +%s)" -gt "$deadline" ] && { echo "$prev overran; starting anyway $(date -u +%FT%TZ)"; break; }
    sleep 120
  done
}
stage_start() { echo "$1 start $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true; }
stage_end() { echo "$1 exit=$2 $(date -u +%FT%TZ)"; exit "$2"; }
# Every crossover board in this program shares the 50M recipe unless a script
# overrides a field AFTER sourcing this file.
export CROSSOVER_BATCH=32
export CROSSOVER_EVAL_ITERS=20
export CROSSOVER_TOKEN_BUDGET=50000000
