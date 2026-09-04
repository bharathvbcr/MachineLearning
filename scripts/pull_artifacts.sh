#!/usr/bin/env bash
# Pull the PUBLISHED run artifacts off the rented box. Runs on the laptop.
#
# Why this exists. On 2026-09-04 the E21 phase-2 board finished 30/30, reported
# cleanly, and then the instance became unreachable before anything was synced.
# Its 30 run directories are gone. The stage log had been captured, so the
# result survived as a table in docs/LADDER_BOARD_2026-09-04.md -- but the
# per-seed metrics.jsonl that PAPER section 10 lists as a required reproduction
# artifact did not.
#
# Nothing about that was inevitable. `.gitignore` already carries explicit
# exceptions for exactly these filenames (added for gap B3), and 2272 of them
# are tracked in this repo. The failure was purely that nobody copied them
# while the box was up.
#
# So: run this DURING a board, not after it. It is incremental and cheap --
# metrics.jsonl + config.json across all of nanolab/out is ~16 MB, against 467
# GB of .pt checkpoints that are deliberately never pulled.
#
#   bash scripts/pull_artifacts.sh                    # everything
#   bash scripts/pull_artifacts.sh crossover_ladder50m  # one suite
#
# HOST is the ssh alias. If the box was recycled onto a new instance, repoint
# the alias in ~/.ssh/config -- do not accept a changed host key to make this
# script work.
set -u
HOST="${HOST:-lambda-gpu}"
REMOTE="${REMOTE:-MLSystemsLab/nanolab/out}"
LOCAL="${LOCAL:-nanolab/out}"
SUITE="${1:-}"

src="$HOST:$REMOTE/${SUITE:+$SUITE/}"
dst="$LOCAL/${SUITE:+$SUITE/}"
mkdir -p "$dst"

echo "pull $src -> $dst"
# Include only what .gitignore publishes. --prune-empty-dirs keeps the tree from
# filling with empty checkpoint-only directories.
rsync -az --prune-empty-dirs \
  --include='*/' \
  --include='metrics.jsonl' --include='config.json' --include='queue.json' \
  --include='recipe.json'   --include='ledger.json'  --include='summary.json' \
  --include='runs.jsonl'    --include='final_val.json' \
  --include='*.log' \
  --exclude='*' \
  "$src" "$dst"
rc=$?
echo "pull exit=$rc $(date -u +%FT%TZ)"
[ "$rc" -ne 0 ] && exit "$rc"

n=$(find "$dst" -name metrics.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "$n metrics.jsonl present under $dst"
echo
echo "Commit them -- a file on a rented box is not an artifact:"
echo "  git add $dst && git commit -m 'artifacts: <suite>'"
