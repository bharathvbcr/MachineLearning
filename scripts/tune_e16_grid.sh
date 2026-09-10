#!/usr/bin/env bash
# E16 grid at a FIXED batch, measuring solve RATE rather than requiring saturation.
#
# Why this bypasses `swaboard --only mqar-grid`: the calibrate phase demands a
# batch that saturates attention on 3/3 seeds, found none at any cell, wrote
# {"16":null,"64":null,"128":null} and correctly refused. That refusal is right
# for the question it asks and wrong for the question E16 asks.
#
# The calibration's own 36 runs show why. Every run landed either at a floor
# (~0.24 / ~0.086 / ~0.048 by cell) or at exactly 1.000 -- a solve/no-solve
# phase transition, not a difficulty gradient, which is the bimodality E8 warned
# about. With that shape the measurable quantity is the SOLVE RATE per arm, and
# 3 seeds cannot distinguish 1/3 from 2/3. Fifteen can.
#
# Batch is pinned to 64 because the calibration measured larger batches to be
# strictly worse -- solves by batch: 64->3, 128->2, 256->2, 512->0 -- and 64 is
# also the cheapest. This is a deliberate override of a refusal, so it is done
# HERE rather than by writing a batch into calibration.json: that file should
# keep saying null, because null is what the calibration actually found.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
LOG=nanolab/out/_tune/e16_grid.log

gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
quiet=0
for _ in $(seq 1 90); do
  if gpu_busy; then quiet=0; else
    quiet=$((quiet+1)); [ "$quiet" -ge 3 ] && break; fi
  sleep 10
done

{
  echo "=== E16 grid, fixed bs=64, 15 seeds $(date -u +%FT%TZ) ==="
  echo "    arms: attention,gdn,mingru,swa_w64,swa_w64_nosink   cells: 16,64,128"
  echo "    calibration.json stays null on purpose; the override lives in this script"
  s=$(date +%s)
  python3 -u -m nanolab.mqar_suite \
    --out nanolab/out/mqar_e16 --device cuda \
    --cells 16,64,128 \
    --batch-by-cell '{"16":64,"64":64,"128":64}' \
    --arms attention,gdn,mingru,swa_w64,swa_w64_nosink \
    --seeds 15 --workers 4 --steps 3000 2>&1
  echo "e16grid exit=$? elapsed=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
