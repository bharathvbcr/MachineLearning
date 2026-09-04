#!/usr/bin/env bash
# E18: the depth axis. No board in this repo has ever varied n_layer -- every
# arm is 12 layers, so params and compute move together and the two matching
# rules (equal tokens, equal wall clock) have never been able to disagree.
#
# looped_attn3x4 and looped_attn6x2 have the PARAMS of 3 and 6 layers and the
# COMPUTE of 12 (identical flops/token to `attention`, checked before launch).
# attn3 and attn6 are the same parameter counts with the loop removed -- without
# them "looping buys depth" is unfalsifiable.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
export CROSSOVER_ARMS=attention,looped_attn3x4,looped_attn6x2,attn3,attn6
export CROSSOVER_JOB_PREFIX=cx32loop
export CROSSOVER_BATCH=32
export CROSSOVER_EVAL_ITERS=20
export CROSSOVER_TOKEN_BUDGET=50000000
echo "e18 start $(date -u +%FT%TZ)"
python3 -u -m nanolab.crossover_replicate launch \
  --out nanolab/out/crossover50m_loop32 --workers 3
echo "e18 launch exit=$? $(date -u +%FT%TZ)"
sleep 60
while pgrep -f "crossover_replicate worker" >/dev/null; do sleep 60; done
echo "e18 workers drained $(date -u +%FT%TZ)"
python3 -u -m nanolab.crossover_replicate table --out nanolab/out/crossover50m_loop32 2>&1 | tail -25
echo "e18 exit=0 $(date -u +%FT%TZ)"
