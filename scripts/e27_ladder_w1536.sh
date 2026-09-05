#!/usr/bin/env bash
# E21 extended: a fourth width, taking the ladder's span from 3x to 4x.
#
# Phase 2 found the attention-vs-minGRU ranking stable over 384..1152 with a
# flat margin (0.1471 / 0.1585 / 0.1517). That is a claim about a 3x span, and
# the honest limit of it is that nobody has looked outside. 1536 tests both of
# the ladder's findings at a new point: the ranking, and the LR invariant
# (attention peaked at 8x base and minGRU at 4x at every width so far).
#
# Two stages in one script, because stage 2's arms are chosen by stage 1's
# result rather than assumed: probe at 10M / n=1, read the argmin per mixer,
# then run those two cells at 50M / n=5 like every other board in this repo.
#
# NOT a wall-clock board. That was the first idea and the measured rates killed
# it: minGRU runs at 0.77-0.79x attention's throughput at every width on this
# box (ctx 512), so matching wall clock hands minGRU FEWER tokens and can only
# widen a gap that is already disjoint. Nothing to learn there.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"

# Chain on the log marker, never pgrep.
deadline=$(( $(date +%s) + 5*3600 ))
while ! grep -q "^e26 exit=" nanolab/out/e26.log 2>/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "e26 overran; starting anyway"; break; }
  sleep 120
done
echo "e27 start $(date -u +%FT%TZ)"

export CROSSOVER_BATCH=32
export CROSSOVER_EVAL_ITERS=20

# ---- stage 1: LR probe, 10M tokens, 1 seed, bracketing the known optima ----
export CROSSOVER_ARMS=w1536_attention_lr40,w1536_attention_lr80,w1536_attention_lr160,w1536_mingru_lr20,w1536_mingru_lr40,w1536_mingru_lr80
export CROSSOVER_JOB_PREFIX=cx32lad1536p
export CROSSOVER_TOKEN_BUDGET=10000000
python3 -u -m nanolab.crossover_replicate launch \
  --out nanolab/out/crossover_ladder_probe1536 --workers 1 --seed 1337
rc=$?
echo "e27 probe exit=$rc $(date -u +%FT%TZ)"
[ "$rc" -ne 0 ] && { echo "e27 exit=$rc PROBE FAILED"; exit "$rc"; }

# The first run of this script proceeded on FOUR of six probe jobs: two minGRU
# cells OOMed, `launch` exited 0 anyway, and stage 2 then read minGRU's argmin
# off the single survivor -- which was the bottom edge of the swept range, and
# contradicted the ladder's own invariant. Ten 50M-token jobs went to the wrong
# learning rate. `launch` now exits non-zero when the queue holds failures, but
# a stage that reads a queue must check the queue rather than trust an exit
# code it did not write.
python3 - <<'PYEOF' || exit 1
import json, sys
from collections import Counter
q = json.load(open("nanolab/out/crossover_ladder_probe1536/queue.json"))
c = Counter(j["status"] for j in q["jobs"])
print("e27 probe queue:", dict(c), "of", len(q["jobs"]))
bad = [j for j in q["jobs"] if j["status"] != "done"]
for j in bad:
    print("  NOT DONE:", j["id"], j["status"], (j.get("detail") or "")[:90])
if bad:
    print("REFUSING stage 2: an argmin over a partial probe is not an argmin.")
    sys.exit(1)
PYEOF

BEST=$(python3 - <<'PYEOF'
import json
from pathlib import Path
from nanolab.crossover_replicate import _arm_from_run_dir
root=Path("nanolab/out/crossover_ladder_probe1536")
vals={}
for d in sorted(root.iterdir()):
    m=d/"metrics.jsonl"
    if not d.is_dir() or not m.exists(): continue
    fin=None
    for line in m.read_text().splitlines():
        try: r=json.loads(line)
        except Exception: continue
        if r.get("event")=="done": fin=r.get("final_val")
    if fin is not None: vals[_arm_from_run_dir(d)]=fin
best={}
for k,v in vals.items():
    mx="attention" if "attention" in k else "mingru"
    if mx not in best or v<best[mx][1]: best[mx]=(k,v)
import sys
for mx,(k,v) in sorted(best.items()):
    print("%s %.4f" % (k,v), file=sys.stderr)
# an argmin at an edge is not an argmin -- say so loudly, still proceed
edges={"w1536_attention_lr40","w1536_attention_lr160",
       "w1536_mingru_lr20","w1536_mingru_lr80"}
hit=[k for k,_ in best.values() if k in edges]
if hit: print("EDGE:"+",".join(hit), file=sys.stderr)
print(",".join(k for k,_ in sorted(best.values())))
PYEOF
)
echo "e27 stage2 arms: $BEST"
case "$BEST" in
  *lr40|*lr40,*|*lr160*|*lr20,*|*_mingru_lr20|*lr80,*_mingru_lr80)
    # An argmin on the edge of the swept range is not an argmin. Previously this
    # was a stderr note nobody gated on.
    echo "e27: an argmin landed on a SWEEP EDGE -- widen the probe before"
    echo "     spending 50M-token jobs on it. Refusing stage 2."
    echo "e27 exit=1 $(date -u +%FT%TZ)"
    exit 1;;
esac

# ---- stage 2: the two winning cells at 50M, n=5 ----
export CROSSOVER_ARMS="$BEST"
export CROSSOVER_JOB_PREFIX=cx32lad1536
export CROSSOVER_TOKEN_BUDGET=50000000
python3 -u -m nanolab.crossover_replicate launch \
  --out nanolab/out/crossover_ladder1536 --workers 1
rc=$?
echo "e27 board exit=$rc $(date -u +%FT%TZ)"

python3 - <<'PYEOF'
import json, math, statistics
from collections import Counter
from pathlib import Path
from nanolab.crossover_replicate import _arm_from_run_dir
root=Path("nanolab/out/crossover_ladder1536")
if not (root/"queue.json").exists():
    print("e27: no queue"); raise SystemExit(0)
q=json.load(open(root/"queue.json"))
print("e27 queue:", dict(Counter(j["status"] for j in q["jobs"])), "of", len(q["jobs"]))
by={}
for d in sorted(root.iterdir()):
    m=d/"metrics.jsonl"
    if not d.is_dir() or not m.exists(): continue
    fin=None
    for line in m.read_text().splitlines():
        try: r=json.loads(line)
        except Exception: continue
        if r.get("event")=="done": fin=r.get("final_val")
    if fin is not None: by.setdefault(_arm_from_run_dir(d),[]).append(fin)
cells={}
for arm,f in sorted(by.items()):
    mu=statistics.mean(f); sd=statistics.stdev(f) if len(f)>1 else 0.0
    hw=2.776*sd/math.sqrt(len(f)) if len(f)>1 else 0.0
    cells["attention" if "attention" in arm else "mingru"]=(mu,mu-hw,mu+hw)
    print("  %-26s n=%d %.4f [%.4f, %.4f]" % (arm,len(f),mu,mu-hw,mu+hw))
a,m=cells.get("attention"),cells.get("mingru")
if a and m:
    dis = a[2]<m[1] or m[2]<a[1]
    print("\n  w1536: attention %.4f  minGRU %.4f -> %s by %.4f%s"
          % (a[0],m[0],"attention" if a[0]<m[0] else "minGRU",abs(a[0]-m[0]),
             "  (DISJOINT)" if dis else "  (intervals overlap)"))
    print("  prior widths, attention margin: 384 +0.1471  768 +0.1585  1152 +0.1517")
    print("  ranking %s at 4x span" % ("HOLDS" if a[0]<m[0] else "MOVES"))
PYEOF
echo "e27 exit=$rc $(date -u +%FT%TZ)"
