#!/usr/bin/env bash
# E21 phase 2: the width ladder at each cell's OWN best learning rate.
#
# Phase 1 swept 0.5x .. 32x base at three widths and every cell now has an
# INTERIOR optimum, so these six LRs are argmins rather than edges:
#
#   width  attention          minGRU
#    384   lr80  5.2287       lr40  5.1651
#    768   lr80  5.1663       lr40  5.0494
#   1152   lr80  5.1541       lr40  4.9947
#
# attention peaks at 8x base at every width; minGRU peaks at 4x at every width.
# A consistent 2x separation, stable across a 3x span in width. That asymmetry
# is the reason this phase exists: a board that hands both arms one shared LR
# reads them at different distances from their own optima, which is a mechanism
# for rank movement rather than another instance of one.
#
# 50M tokens, n=5, so the cells are directly comparable to every other board in
# this repo. The comparison this answers: does the attention-vs-minGRU ranking
# move with WIDTH once each arm is tuned at each width?
#
# The shared-LR condition already exists at width 768 -- crossover50m carries
# both arms at base LR, 50M, n=5 -- so that contrast is available without
# re-running it here.
#
# Standing caveat for the writeup: phase 1 picked these LRs at 10M tokens, not
# 50M. Shorter runs favour higher LRs, so phase 2 inherits an assumption. If a
# result turns on it, carrying all of lr40/lr80/lr160 at one width would test
# it directly.
#
# 2 workers: the widest arm is 328.7M parameters.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
echo "e21 phase2 start $(date -u +%FT%TZ)"

export CROSSOVER_ARMS=w384_attention_lr80,w384_mingru_lr40,w768_attention_lr80,w768_mingru_lr40,w1152_attention_lr80,w1152_mingru_lr40
export CROSSOVER_JOB_PREFIX=cx32lad50
export CROSSOVER_BATCH=32
export CROSSOVER_EVAL_ITERS=20
export CROSSOVER_TOKEN_BUDGET=50000000
python3 -u -m nanolab.crossover_replicate launch \
  --out nanolab/out/crossover_ladder50m --workers 2
rc=$?
echo "e21p2 launch exit=$rc $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "e21p2 exit=$rc $(date -u +%FT%TZ)  LAUNCH FAILED, no jobs ran"
  exit "$rc"
fi
sleep 60
while pgrep -f "crossover_replicate worker" >/dev/null; do sleep 60; done
echo "e21p2 workers drained $(date -u +%FT%TZ)"

python3 - <<'PYEOF'
import json, math, statistics
from collections import Counter
from pathlib import Path
from nanolab.crossover_replicate import _arm_from_run_dir
root = Path("nanolab/out/crossover_ladder50m")
q = json.load(open(root / "queue.json"))
print("e21p2 queue:", dict(Counter(j["status"] for j in q["jobs"])),
      "of", len(q["jobs"]))
by = {}
for d in sorted(root.iterdir()):
    m = d / "metrics.jsonl"
    if not d.is_dir() or not m.exists():
        continue
    fin = None
    for line in m.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "done":
            fin = r.get("final_val")
    if fin is not None:
        by.setdefault(_arm_from_run_dir(d), []).append(fin)
cells = {}
print("\n  %-24s %3s %-26s" % ("arm", "n", "final_val [95% t]"))
for arm, f in sorted(by.items()):
    mu = statistics.mean(f)
    sd = statistics.stdev(f) if len(f) > 1 else 0.0
    hw = 2.776 * sd / math.sqrt(len(f)) if len(f) > 1 else 0.0
    print("  %-24s %3d %.4f [%.4f, %.4f]" % (arm, len(f), mu, mu - hw, mu + hw))
    w = arm.split("_")[0]
    mx = "attention" if "attention" in arm else "mingru"
    cells[(w, mx)] = (mu, mu - hw, mu + hw)

# The question this board exists to answer, stated rather than left to the eye.
print("\n  ranking by width (attention vs minGRU, each at its own best LR):")
def disjoint(a, b):
    return a[2] < b[1] or b[2] < a[1]
for w in ("w384", "w768", "w1152"):
    a, m = cells.get((w, "attention")), cells.get((w, "mingru"))
    if not (a and m):
        continue
    win = "attention" if a[0] < m[0] else "mingru"
    print("    %-6s attention %.4f  minGRU %.4f  -> %s by %.4f%s"
          % (w, a[0], m[0], win, abs(a[0] - m[0]),
             "  (DISJOINT)" if disjoint(a, m) else "  (intervals overlap)"))
winners = {w: ("attention" if cells[(w, "attention")][0] < cells[(w, "mingru")][0]
               else "mingru")
           for w in ("w384", "w768", "w1152")
           if (w, "attention") in cells and (w, "mingru") in cells}
if len(set(winners.values())) > 1:
    print("\n  RANKING MOVES WITH WIDTH: %s" % winners)
else:
    print("\n  ranking is STABLE across width (%s wins at every width)"
          % next(iter(set(winners.values())), "?"))
PYEOF
echo "e21p2 exit=$? $(date -u +%FT%TZ)"
