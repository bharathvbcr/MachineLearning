#!/usr/bin/env bash
# E16 window sweep at seq 511. The seq-511 board left ONE alternative
# explanation standing: swa_w64 scored 0/10, but a 64-token window provably
# cannot reach a key 128 pairs back, so the result is arithmetic, not evidence
# about the SWA mechanism. These three arms vary ONLY the window.
#
# block_size = 2*(128+128)-1 = 511, so swa_w512 has a window >= the whole
# sequence: it IS full causal attention wearing the SWA code path. If it does
# not reproduce the attention arms 8/10, the SWA implementation is the
# confound and the whole board needs re-reading. If it does, the 0/10 at w64
# is the window and nothing else, and w128/w256 give the curve between.
#
# No GDN arm here, so the 28.9 GiB worst case does not apply: 3 workers at
# ~16.5 GiB is ~50 of 95.6 GiB.
set -u
cd "$HOME/MLSystemsLab" || exit 1
echo "window sweep start $(date -u +%FT%TZ)"
python3 -u -m nanolab.mqar_suite \
  --out nanolab/out/mqar_e16_window511 --device cuda \
  --cells 128 --batch 128 \
  --arms swa_w128,swa_w256,swa_w512 \
  --seeds 10 --steps 3000 --lr-rule sqrt \
  --workers 3 --gpus 1
echo "window exit=$? $(date -u +%FT%TZ)"
