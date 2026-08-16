#!/usr/bin/env bash
#
# cleanup.sh — reclaim disk in the parameter_golf workspace.
#
# Two tiers:
#   1. SAFE   (always)  — regenerable junk: python caches, lint/test caches,
#                         editor swap files, .DS_Store, *.tmp/*.bak/*.orig, etc.
#   2. CKPT   (--checkpoints) — redundant resumable training snapshots
#                         (ckpt.pt) that are >AGE days old AND whose run dir
#                         still has best.pt or final.pt. Frees ~GBs.
#
# NEVER touched: anything under records/ (experiment results), *.log training
# logs, .git/, best.pt / final.pt, dataset .bin files.
#
# Dry-run by default. Pass --apply to actually delete.
#
# Usage:
#   ./cleanup.sh                      # safe tier, dry-run (shows what it would do)
#   ./cleanup.sh --apply              # safe tier, delete
#   ./cleanup.sh --checkpoints        # safe + ckpt sweep, dry-run
#   ./cleanup.sh --apply --checkpoints --age 14
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPLY=0
DO_CKPT=0
AGE=7
LOG="$ROOT/.cleanup.log"

while [ $# -gt 0 ]; do
  case "$1" in
    --apply)        APPLY=1 ;;
    --checkpoints)  DO_CKPT=1 ;;
    --age)          AGE="$2"; shift ;;
    -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

ts() { date '+%Y-%m-%d %H:%M:%S'; }
mode() { [ "$APPLY" -eq 1 ] && echo "APPLY" || echo "DRY-RUN"; }

freed_bytes=0
log() { echo "$*"; echo "[$(ts)] $*" >> "$LOG"; }

# Sum size (bytes) of a list of paths on stdin, then optionally delete them.
# $1 = human label
sweep() {
  local label="$1" n=0 bytes=0 p sz
  while IFS= read -r p; do
    [ -e "$p" ] || continue
    sz=$(du -sk "$p" 2>/dev/null | awk '{print $1}'); sz=${sz:-0}
    bytes=$(( bytes + sz * 1024 ))
    n=$(( n + 1 ))
    if [ "$APPLY" -eq 1 ]; then rm -rf "$p"; fi
  done
  if [ "$n" -gt 0 ]; then
    freed_bytes=$(( freed_bytes + bytes ))
    log "  $label: $n item(s), $(numfmt --to=iec --suffix=B "$bytes" 2>/dev/null || echo "${bytes}B")"
  fi
}

log "=== cleanup [$(mode)] root=$ROOT age=${AGE}d ==="

# ---- Tier 1: safe regenerable junk ----------------------------------------
# Cache directories.
sweep "cache dirs" < <(find "$ROOT" -type d \( \
      -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \
   -o -name '.ruff_cache' -o -name '.ipynb_checkpoints' \) \
   -not -path '*/.git/*' -prune -print 2>/dev/null)

# Junk files (skip .git and records/).
sweep "junk files" < <(find "$ROOT" -type f \( \
      -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \
   -o -name '*.swp' -o -name '*.swo' -o -name '*~' \
   -o -name '*.tmp' -o -name '*.temp' -o -name '*.bak' -o -name '*.orig' \
   -o -name 'core.*' -o -name 'nohup.out' \) \
   -not -path '*/.git/*' -not -path '*/records/*' -print 2>/dev/null)

# ---- Tier 2: redundant resumable checkpoints (opt-in) ---------------------
if [ "$DO_CKPT" -eq 1 ]; then
  log "  -- checkpoint sweep (ckpt.pt >${AGE}d with surviving best/final) --"
  while IFS= read -r ckpt; do
    [ -e "$ckpt" ] || continue
    d="$(dirname "$ckpt")"
    if [ -f "$d/final.pt" ] || [ -f "$d/best.pt" ]; then
      sweep "    $(basename "$d")/ckpt.pt" < <(printf '%s\n' "$ckpt")
    else
      log "    skip $(basename "$d")/ckpt.pt (no best/final to fall back on)"
    fi
  done < <(find "$ROOT" -type f -name 'ckpt.pt' -mtime +"$AGE" -not -path '*/.git/*' 2>/dev/null)
fi

# ---- Summary --------------------------------------------------------------
human=$(numfmt --to=iec --suffix=B "$freed_bytes" 2>/dev/null || echo "${freed_bytes}B")
if [ "$APPLY" -eq 1 ]; then
  log "=== DONE: reclaimed $human ==="
else
  log "=== DRY-RUN: would reclaim $human (re-run with --apply to delete) ==="
fi
