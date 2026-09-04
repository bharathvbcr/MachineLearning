#!/usr/bin/env bash
# E19b: E19 re-run on the fixed zero-init.
#
# The first E19 is void. Its 1-expert control -- which is a dense SwiGLU wearing
# the MoE code path, so it HAD to match `attention` -- lost by 0.127 nats with
# disjoint intervals, because `_zero_init_output_projections` tested `ffn.down`
# and a MoE keeps its projections on ffn.experts[i].down. Every MoE run in this
# repo trained without the identity-init stabilizer every dense arm got.
#
# So the old board's attention-vs-MoE gap measures a missing init, not
# parameters, and its "4.2x the parameters bought nothing" is unsupported.
# A NEW output directory on purpose: the old runs are not comparable to these
# and must not be pooled with them by a resume.
#
# The control is the readout. If moe_e1k1 now lands inside `attention`'s
# interval, the board can be read as a parameter experiment. If it still does
# not, something else differs between the two paths and the MoE arms stay
# uninterpretable -- which is a reportable result, not a failure.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"

# Wait on E20's LOG MARKER, never pgrep -- see scripts/overnight.sh.
deadline=$(( $(date +%s) + 10*3600 ))
while ! grep -q "^e20 exit=" nanolab/out/e20.log 2>/dev/null; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "e20 overran 10h; starting anyway"
    break
  fi
  sleep 120
done
echo "e20 done, e19b start $(date -u +%FT%TZ)"

export CROSSOVER_ARMS=attention,moe_e1k1,moe_e4k1,moe_e8k1
export CROSSOVER_JOB_PREFIX=cx32moeb
export CROSSOVER_BATCH=32
export CROSSOVER_EVAL_ITERS=20
export CROSSOVER_TOKEN_BUDGET=50000000
python3 -u -m nanolab.crossover_replicate launch \
  --out nanolab/out/crossover50m_moe32b --workers 2
rc=$?
echo "e19b launch exit=$rc $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "e19b exit=$rc $(date -u +%FT%TZ)  LAUNCH FAILED, no jobs ran"
  exit "$rc"
fi
sleep 60
while pgrep -f "crossover_replicate worker" >/dev/null; do sleep 60; done
echo "e19b workers drained $(date -u +%FT%TZ)"

python3 - <<'PYEOF'
import json, math, statistics
from collections import Counter
from pathlib import Path
from nanolab.crossover_replicate import _arm_from_run_dir
root = Path("nanolab/out/crossover50m_moe32b")
q = json.load(open(root / "queue.json"))
print("e19b queue:", dict(Counter(j["status"] for j in q["jobs"])),
      "of", len(q["jobs"]))
by = {}
for d in sorted(root.iterdir()):
    m = d / "metrics.jsonl"
    if not d.is_dir() or not m.exists():
        continue
    fin = tk = None
    for line in m.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "done":
            fin, tk = r.get("final_val"), r.get("mean_tok_s")
    if fin is not None:
        by.setdefault(_arm_from_run_dir(d), []).append((fin, tk))
print("  %-12s %3s %-26s %9s" % ("arm", "n", "final_val [95% t]", "tok/s"))
rows = {}
for arm, v in sorted(by.items(), key=lambda kv: statistics.mean(x[0] for x in kv[1])):
    f = [x[0] for x in v]
    t = [x[1] for x in v if x[1]]
    mu = statistics.mean(f)
    sd = statistics.stdev(f) if len(f) > 1 else 0.0
    hw = 2.776 * sd / math.sqrt(len(f)) if len(f) > 1 else 0.0
    rows[arm] = (mu, mu - hw, mu + hw)
    print("  %-12s %3d %.4f [%.4f, %.4f] %9.0f"
          % (arm, len(f), mu, mu - hw, mu + hw,
             statistics.median(t) if t else 0))
# The control decides whether the rest of the board can be read at all.
a, c = rows.get("attention"), rows.get("moe_e1k1")
if a and c:
    overlap = c[1] <= a[2] and a[1] <= c[2]
    print("\nCONTROL: moe_e1k1 vs attention intervals %s"
          % ("OVERLAP -- board is readable as a parameter experiment"
             if overlap else
             "still DISJOINT -- MoE path differs from dense for some other "
             "reason; the MoE arms remain uninterpretable"))
PYEOF
echo "e19b exit=$? $(date -u +%FT%TZ)"
