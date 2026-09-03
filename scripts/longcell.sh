#!/usr/bin/env bash
# E16 stage 4: the seq-511 cell -- the one that actually tests the paper's claim,
# where a 64-wide window sees 1/8 of the context and only the sinks reach back.
#
# batch 128: the only batch that saturated this cell (3/3 under sqrt; bs 64 was
# 1/3). n=3 makes that a crude rate estimate, so if every arm floors at ~0.05 the
# cell was mispriced, not the architectures -- read the attention arm first.
#
# 2 workers, NOT 4: gdn peaks at 28.9 GiB here (attention 15.6, swa_w64 16.5) and
# shard() round-robins by job index, so every worker enters the gdn block at the
# same time. 3 workers = 86.7 GiB of 95.6 with no headroom; 4 OOMs.
#
# 10 seeds, not 15: seeds 1-10 are a prefix of E8_SEEDS, so extending to 15 later
# resumes off the ledger instead of re-running.
set -u
cd "$HOME/MLSystemsLab" || exit 1
echo "longcell waiting for control $(date -u +%FT%TZ)"
deadline=$(( $(date +%s) + 21600 ))
while ! grep -q "^control exit=" nanolab/out/control.log 2>/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "control overran 6h; longcell ABORTED"; exit 1; }
  sleep 60
done
echo "control done, longcell start $(date -u +%FT%TZ)"

python3 -u -m nanolab.mqar_suite \
  --out nanolab/out/mqar_e16_seq511 --device cuda \
  --cells 128 --batch 128 \
  --arms attention,gdn,mingru,swa_w64,swa_w64_nosink \
  --seeds 10 --steps 3000 --lr-rule sqrt \
  --workers 2 --gpus 1
echo "longcell exit=$? $(date -u +%FT%TZ)"
