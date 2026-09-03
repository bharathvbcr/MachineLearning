#!/usr/bin/env bash
# Everything left on the backlog that is specified enough to run unattended,
# chained behind the seq-511 window sweep.
#
#   E17  the CUDA flash-attention kernel has never been compiled or executed.
#        Two checks in nanolab/tests.py are SKIPPED for want of a CUDA device;
#        this is the device. Minutes, no jobs, no budget.
#   E10  hybrid_mingru10_attn2 inside crossover50m_ratio32, so bookend-vs-10+2
#        is a within-suite final_val comparison instead of a cross-suite one.
#        5 runs -- claim_job() skips the 20 already done, verified at line 1228.
#
# The scale ladder is NOT here: it needs presets that do not exist and a
# width-vs-LR rule, which is the E13 confound. It is a design decision, not a
# queue item.
#
# Waits on the LOG, not on pgrep: a stale tmux cmdline matching the pattern
# once cost 79 minutes of idle GPU on this box.
set -u
cd "$HOME/MLSystemsLab" || exit 1
echo "queue start $(date -u +%FT%TZ)"

deadline=$(( $(date +%s) + 6*3600 ))
while ! grep -q "^window exit=" nanolab/out/window.log 2>/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "sweep overran 6h; going anyway"; break; }
  sleep 60
done
echo "sweep done $(date -u +%FT%TZ)"

echo "########## E17: CUDA flash-attention kernel ##########"
python3 -u -m nanolab.tests 2>&1 | tail -25
echo "e17 tests exit=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
for hd in 32 64; do
  echo "--- head_dim $hd, bf16 ---"
  python3 -u -m nanolab.flash_cuda --batch 8 --seq 1024 --heads 12 \
    --kv_heads 4 --head_dim "$hd" --dtype bf16
  echo "e17 bench hd$hd exit=$?"
done
echo "e17 exit=0 $(date -u +%FT%TZ)"

echo "########## E10: ratio32 + the co-leader ##########"
python3 -u -m nanolab.crossover_replicate ratio32 --gpus 1
sleep 30
while pgrep -f "crossover_replicate worker" >/dev/null; do sleep 60; done
echo "e10 exit=0 $(date -u +%FT%TZ)"
python3 -u -m nanolab.crossover_replicate status 2>&1 | tail -20
echo "queue exit=0 $(date -u +%FT%TZ)"
