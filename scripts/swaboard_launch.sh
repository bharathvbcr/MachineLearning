#!/usr/bin/env bash
# Deploy and run the SWA board (E12/E15/E16) on a fresh GH200.
#
# Usage:  BOX=ubuntu@<instance-ip> bash scripts/swaboard_launch.sh
#
# Idempotent: every phase is resumable, so re-running after a drop picks up
# where it stopped. Nothing here bills until the preflight passes.
set -euo pipefail
: "${BOX:?set BOX=user@host for the GH200 instance}"
# Relative to the REMOTE home. Not "~/..." -- an unquoted tilde in an ssh or
# rsync argument expands on the CLIENT, which would have targeted this Mac's
# home directory instead of the box's.
REMOTE="${REMOTE:-MLSystemsLab}"

echo "==> 1/5  code"
git -C . rev-parse --short HEAD
rsync -az --delete --exclude '.git' --exclude 'nanolab/out' --exclude 'nanolab/data' \
      --exclude '__pycache__' --exclude 'target' ./ "$BOX:$REMOTE/"

echo "==> 2/5  reference corpus (497,500,000 tokens; NOT re-tokenized)"
# The Batcher samples with replacement, so corpus SIZE is part of the recipe:
# 50M tokens over 497.5M is 0.04 epochs, over 50M it is 0.4. Suites 26 and E9
# were measured against this exact corpus, so it is copied, not regenerated --
# a fresh tokenization of the same nominal size is not byte-identical.
rsync -az --info=progress2 nanolab/data/HuggingFaceFW_fineweb-edu/ \
      "$BOX:$REMOTE/nanolab/data/HuggingFaceFW_fineweb-edu/"

echo "==> 3/5  gate the box BEFORE it bills"
ssh "$BOX" "cd $REMOTE && python3 -c \"
import os
p='nanolab/data/HuggingFaceFW_fineweb-edu/train.bin'
n=os.path.getsize(p)//2
assert n==497_500_000, f'corpus is {n:,} tokens, reference is 497,500,000'
print(f'corpus OK: {n:,} tokens')\" \
 && python3 -m nanolab.tests \
 && python3 -m nanolab.crossover_replicate measure-peak"

echo "==> 4/5  smoke every arm the board will run (40 steps each, isolated subtree)"
ssh "$BOX" "cd $REMOTE && python3 -m nanolab.crossover_replicate smoke \
      --arms attention,gdn,swa_w64,swa_w128,swa_w256,swa_w64_nosink,hybrid_mingru10_swa2"

echo "==> 5/5  the board, detached under tmux so an ssh drop cannot kill it"
ssh "$BOX" "cd $REMOTE && tmux new-session -d -s swaboard \
      'python3 -u -m nanolab.crossover_replicate swaboard 2>&1 | tee nanolab/out/swaboard.log'"
echo
echo "running. follow it with:"
echo "  ssh $BOX -t 'tmux attach -t swaboard'"
echo "  ssh $BOX 'cd $REMOTE && python3 -m nanolab.crossover_replicate status'"
