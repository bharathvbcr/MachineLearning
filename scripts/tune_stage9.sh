#!/usr/bin/env bash
# Eval-noise probe: how much of a board's reported number is eval_iters noise.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
python3 -u scripts/tune_evaliters.py \
  --suite nanolab/out/crossover50m_loop32 \
  --a cx32loop_attention_s1337 --b cx32loop_attn6_s1337 --batches 200
echo "stage9 exit=$? $(date -u +%FT%TZ)"
