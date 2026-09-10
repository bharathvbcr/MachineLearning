#!/usr/bin/env bash
# G8c: retire the last unreadable records in the paper -- two runs, about a dollar.
#
# `cx32lad1536_w1536_attention_lr80_s1337` and `_s42` are the only 2 of 1151
# committed eval curves whose token axis folds back on itself: both were resumed
# before D1, so `tokens` restarted at 0 while `step` carried on. They sit at the
# TOP rung of the width ladder -- the 13.55M endpoint of the paper's "+27% across
# 10.3x params" -- so 2 of that rung's 5 seeds were recoverable only through a
# step-keyed reading, and `paired_board.py` now refuses them outright.
#
# Rerunning them on the D1/D3-fixed harness into a FRESH directory, rather than
# forcing over the originals: the corrupted records are the evidence for a defect
# section 8 reports, and overwriting evidence to make a caveat go away is not the
# same as measuring it away. Same recipe, same two seeds, nothing destroyed.
#
# Tenancy 1, not 2. The first attempt copied `crossover_ladder1536`'s recorded
# workers=2 and the VRAM guard refused it: two d1536 attention jobs need 80.4 GiB of a
# 94.5 GiB device. That board ran at 2 before the guard existed. `lock_recipe` runs
# BEFORE the guard, so the refused launch still left a stub recording workers=2, which
# then had to be cleared -- it held no runs.
#
# Pre-registered reading: `final_val` is a scalar written at the end and is NOT
# touched by the token fold, so the two boards must agree on it. Agreement within
# the 0.0031-nat floor means the fold cost the paper nothing but the token axis,
# the endpoint rows were always sound, and the step-keyed crossing recovered the
# rung correctly. Disagreement beyond it means the resume corrupted training
# itself, not just the counter, and the w1536 rung comes out of the paper.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g8c
stage_start g8c
export CROSSOVER_ARMS=w1536_attention_lr80
export CROSSOVER_JOB_PREFIX=cx32lad1536redo
rc=0
for s in 1337 42; do
  python3 -u -m nanolab.crossover_replicate launch \
    --out nanolab/out/crossover_ladder1536_redo --workers 1 --seed "$s" || rc=$?
done
echo "g8c launch exit=$rc $(date -u +%FT%TZ)"
python3 - <<'PYEOF'
import json, os
print("\n=== final_val: folded originals vs clean reruns ===")
def fin(p):
    v = None
    if not os.path.exists(p): return None
    for line in open(p):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("event") == "done": v = r.get("final_val")
    return v
worst, missing = 0.0, []
for s in (1337, 42):
    a = fin(f"nanolab/out/crossover_ladder1536/cx32lad1536_w1536_attention_lr80_s{s}/metrics.jsonl")
    b = fin(f"nanolab/out/crossover_ladder1536_redo/cx32lad1536redo_w1536_attention_lr80_s{s}/metrics.jsonl")
    d = None if (a is None or b is None) else b - a
    if d is None:
        missing.append(s)
    else:
        worst = max(worst, abs(d))
    print(f"  seed {s:5d}  folded={a}  rerun={b}  delta={d if d is None else f'{d:+.4f}'}")
# A comparison that could not run must never print what a passing one prints. The first
# attempt of this stage did exactly that: both reruns were absent, `worst` stayed at its
# 0.0 initial value, and it reported AGREES on no data at all.
if missing:
    raise SystemExit(
        f"  INCOMPLETE: no rerun for seed(s) {missing} -- nothing was compared. "
        "This is NOT agreement; the caveat on the w1536 rung stands.")
print(f"  worst |delta| = {worst:.4f}  vs the 0.0031 rerun floor -> "
      f"{'AGREES: the fold cost only the token axis' if worst <= 0.0031 else 'EXCEEDS THE FLOOR: resume corrupted training, not just the counter'}")
PYEOF
echo "REMINDER: bash scripts/pull_artifacts.sh crossover_ladder1536_redo"
stage_end g8c "$rc"
