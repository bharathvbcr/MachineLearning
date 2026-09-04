#!/usr/bin/env bash
# E21 phase 1: the LR probe for the width ladder.
#
# The backlog's loudest reviewer objection is that everything in this repo is
# one small scale -- every board is d_model 768. The reason the ladder stayed
# "unpriced" was never the compute, it was the learning rate: E13 established
# that this repo's muP cells measured a broken attention temperature, so
# transferring one LR across widths by muP would reintroduce the exact confound
# the paper already carries a caveat about.
#
# So measure instead of parametrize. Three widths (384/768/1152, head_dim
# pinned at 64 so width moves through n_head alone), two mixers (attention and
# minGRU -- the pair whose ranking the paper is about), FIVE LRs each. Phase 2
# then re-runs each width at ITS OWN best LR with n=5.
#
# Five, not three, because the first sweep (0.5/1.0/2.0) was entirely on the
# wrong side: all six cells picked 2.0, the top edge, loss monotone decreasing
# in LR. An argmin at the edge is not an argmin. 4.0 and 8.0 extend it; the 18
# runs already on disk are skipped on relaunch.
#
# 10M tokens, not 50M. This phase only has to RANK three learning rates within
# a width, and at 50M it would cost ~6h to do it. The caveat is real and worth
# stating in the writeup: the best LR at 10M is not guaranteed to be the best
# at 50M, so phase 2 inherits an assumption rather than a measurement. Phase 2
# carrying all three LRs at one width would test it, if it turns out to matter.
#
# 2 workers: the widest arm is 328.7M parameters, and this session has already
# been wrong about GPU memory by 4x once.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"

echo "e21 probe start $(date -u +%FT%TZ)"

export CROSSOVER_ARMS=w384_attention_lr05,w384_attention_lr10,w384_attention_lr20,w384_attention_lr40,w384_attention_lr80,w384_mingru_lr05,w384_mingru_lr10,w384_mingru_lr20,w384_mingru_lr40,w384_mingru_lr80,w768_attention_lr05,w768_attention_lr10,w768_attention_lr20,w768_attention_lr40,w768_attention_lr80,w768_mingru_lr05,w768_mingru_lr10,w768_mingru_lr20,w768_mingru_lr40,w768_mingru_lr80,w1152_attention_lr05,w1152_attention_lr10,w1152_attention_lr20,w1152_attention_lr40,w1152_attention_lr80,w1152_mingru_lr05,w1152_mingru_lr10,w1152_mingru_lr20,w1152_mingru_lr40,w1152_mingru_lr80
export CROSSOVER_JOB_PREFIX=cx32lad
export CROSSOVER_BATCH=32
export CROSSOVER_EVAL_ITERS=20
export CROSSOVER_TOKEN_BUDGET=10000000
python3 -u -m nanolab.crossover_replicate launch \
  --out nanolab/out/crossover_ladder_probe --workers 2 --seed 1337
rc=$?
echo "e21probe launch exit=$rc $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "e21probe exit=$rc $(date -u +%FT%TZ)  LAUNCH FAILED, no jobs ran"
  exit "$rc"
fi
sleep 60
while pgrep -f "crossover_replicate worker" >/dev/null; do sleep 60; done
echo "e21probe workers drained $(date -u +%FT%TZ)"

python3 - <<'PYEOF'
import json
from collections import Counter
from pathlib import Path
from nanolab.crossover_replicate import _arm_from_run_dir
root = Path("nanolab/out/crossover_ladder_probe")
q = json.load(open(root / "queue.json"))
print("e21probe queue:", dict(Counter(j["status"] for j in q["jobs"])),
      "of", len(q["jobs"]))
vals = {}
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
        vals[_arm_from_run_dir(d)] = fin
print("\n  %-24s %10s" % ("arm", "final_val"))
for k in sorted(vals):
    print("  %-24s %10.4f" % (k, vals[k]))
# Phase 2 needs one LR per (width, mixer): report the argmin explicitly rather
# than leaving it to be eyeballed off the list.
best = {}
for k, v in vals.items():
    w, mx, lr = k.split("_", 2)
    cur = best.get((w, mx))
    if cur is None or v < cur[1]:
        best[(w, mx)] = (lr, v)
print("\n  best LR per cell (feeds phase 2):")
for (w, mx), (lr, v) in sorted(best.items()):
    print("    %-8s %-10s -> %-6s  final_val %.4f" % (w, mx, lr, v))
# The edges are the FIRST and LAST multipliers actually swept, read from the
# arm table rather than hardcoded -- the previous version still named lr20 as
# the top edge after the sweep had been extended to lr80, which would have
# reported "no edge picked" for exactly the failure it exists to catch.
from nanolab.crossover_replicate import LADDER_LR_MULTS
_tags = [f"lr{str(m).replace('.', '')}" for m in sorted(LADDER_LR_MULTS)]
edge = [f"{w}/{mx}={lr}" for (w, mx), (lr, _) in best.items()
        if lr in (_tags[0], _tags[-1])]
if edge:
    print("\n  WARNING: these cells picked an EDGE of the sweep, so the true "
          "optimum may lie outside it: " + ", ".join(sorted(edge)))
PYEOF
echo "e21probe exit=$? $(date -u +%FT%TZ)"
