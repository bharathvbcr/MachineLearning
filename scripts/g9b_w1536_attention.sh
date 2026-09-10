#!/usr/bin/env bash
# G9b: the one rung G9 dropped -- w1536 attention at its measured 50M argmin (0.5x).
#
# G9 delivered 7 of its 8 arms. It refused the eighth because it routed w1536 attention
# to `crossover_ladder1536` at workers 2 and the VRAM guard blocks that (40.2 GiB x 2 of
# a 94.5 GiB device). That routing had already been FIXED on disk before the arm ran --
# but the fix never reached the running process: **bash reads a script incrementally by
# byte offset, so overwriting a script while it executes does not update the running
# instance.** The check made at the time ("g9 has not launched anything yet") grepped for
# a w1536 launch, which was true and irrelevant; the script was already running.
#
# So this is a new file rather than an edit, and it targets the board the minGRU half
# actually landed in -- `crossover_ladder1536m_argmin`, workers 1, every other recipe
# field matching -- so both re-tuned w1536 arms end up in one directory at one tenancy,
# which is what the fixed routing intended under a different name.
#
# Pre-registered reading: this completes the re-tuned ladder at four widths. The crossing
# is then re-estimated per width against the arms G9 already placed, and section 3's
# width trend either survives re-tuning or is withdrawn -- unchanged from G9's header.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g9b
stage_start g9b
OUT=nanolab/out/crossover_ladder1536m_argmin
ARM=w1536_attention_lr05
CROSSOVER_ARMS=$(python3 - "$OUT" "$ARM" <<'PYEOF'
import json, os, sys
out, arm = sys.argv[1], sys.argv[2]
rec = os.path.join(out, "recipe.json")
have = json.load(open(rec))["arms"] if os.path.exists(rec) else []
print(",".join(dict.fromkeys(list(have) + [arm])))
PYEOF
)
export CROSSOVER_ARMS
export CROSSOVER_JOB_PREFIX=cx32lad1536m2
python3 -u -m nanolab.crossover_replicate launch --out "$OUT" --workers 1 --arm "$ARM"
rc=$?; echo "g9b launch exit=$rc $(date -u +%FT%TZ)"
echo "=== the re-tuned ladder, all four widths ==="
python3 scripts/lr_argmin.py crossover_ladder50m 2>/dev/null | grep -E "^  d|argmin"
python3 scripts/lr_argmin.py crossover_ladder1536m_argmin 2>/dev/null | grep -E "^  d|argmin"
echo "REMINDER: bash scripts/pull_artifacts.sh"
stage_end g9b "$rc"
