#!/usr/bin/env bash
# E19: parameters at fixed compute, from the OTHER side of E18.
#
# E18's looped arms hold compute and REMOVE parameters (123.7M -> 59.9M).
# A top-1 MoE holds active compute and ADDS them (123.7M -> 520.1M): only
# `moe_top_k` experts run per token however many exist. Together the two boards
# span 59.9M..520.1M parameters at ONE compute budget, which makes them a
# parameter axis rather than two unrelated boards.
#
# top_k=1, not the Config default of 2. Two experts per token would run twice
# dense's FFN work and confound the parameter effect with a compute effect.
#
# moe_e1k1 is the control: 1 expert top-1 IS a dense SwiGLU routed through the
# MoE code path. If it does not match `attention`, the gap is the router and the
# aux loss rather than the extra parameters, and the whole board reads
# differently. Same role attn3/attn6 play in E18.
#
# 2 workers, not 3: moe_e8k1 is 520M parameters, and this session has already
# been wrong about GPU memory by 4x once. Fewer workers is the cheap side of
# that error.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"

# Wait on E18's LOG MARKER, never on pgrep. See scripts/overnight.sh for the 79
# minutes of idle GPU that idiom cost: tmux's own argv carries the whole inner
# command, so the pattern outlives the work.
deadline=$(( $(date +%s) + 8*3600 ))
while ! grep -q "^e18 exit=" nanolab/out/e18.log 2>/dev/null; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "e18 overran 8h; starting anyway"
    break
  fi
  sleep 120
done
echo "e18 done, e19 start $(date -u +%FT%TZ)"

export CROSSOVER_ARMS=attention,moe_e1k1,moe_e4k1,moe_e8k1
export CROSSOVER_JOB_PREFIX=cx32moe
export CROSSOVER_BATCH=32
export CROSSOVER_EVAL_ITERS=20
export CROSSOVER_TOKEN_BUDGET=50000000
python3 -u -m nanolab.crossover_replicate launch \
  --out nanolab/out/crossover50m_moe32 --workers 2
echo "e19 launch exit=$? $(date -u +%FT%TZ)"
sleep 60
while pgrep -f "crossover_replicate worker" >/dev/null; do sleep 60; done
echo "e19 workers drained $(date -u +%FT%TZ)"
python3 -u -m nanolab.crossover_replicate table \
  --out nanolab/out/crossover50m_moe32 2>&1 | tail -25
echo "e19 exit=0 $(date -u +%FT%TZ)"
