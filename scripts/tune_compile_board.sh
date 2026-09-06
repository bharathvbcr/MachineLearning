#!/usr/bin/env bash
# Does compile's per-arm speed-up survive a real board?
#
# Three numbers in docs/GPU_TUNING_2026-09-05.md are step-loop only: gdn 3.17x,
# hybrid_mingru8_attn4 1.95x, and attention 1.94x -- the last of which has no
# board at all (the "961 s -> 495 s" this report used to cite is not produced by
# any directory in _tune/; the only compile board is cmp_eager/cmp_on, and it
# runs mingru).
#
# That matters because the same reasoning already failed once. The fused_ce step
# loop predicted 1.55x and the board delivered 1.14x -- over-promised by 36%,
# because a step loop times training steps while a run also evaluates 61 times,
# loads data and checkpoints. Compile touches the eval path too, so it should
# transfer better than fused_ce did. "Should" is the reason to measure it.
#
# gdn is the one to watch: 3.17x is kernel-level and its compile costs 603 s
# cold. At 10M tokens per seed that is a large fraction of the run, so the board
# number can legitimately come in far under 3.17x without anything being wrong.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
OUT=nanolab/out/_tune
ARMS=attention,hybrid_mingru8_attn4,gdn

wait_for_idle() {
  local quiet=0
  echo "waiting for the GPU to go idle ($(date -u +%FT%TZ))"
  for _ in $(seq 1 150); do
    # nvidia-smi, NOT pgrep. `pgrep -f` matches whole command lines, so any
    # shell whose argv CONTAINS this pattern matches it -- the `bash -c` that
    # writes a chain script via heredoc, and every status check typed with the
    # same string. That held one stage off an idle card for 48 min on
    # 2026-09-06. The compute-app list answers the question actually being
    # asked: is anything using the device.
    if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
      quiet=0
    else
      quiet=$((quiet+1))
      [ "$quiet" -ge 3 ] && { echo "GPU idle at $(date -u +%FT%TZ)"; return 0; }
    fi
    sleep 20
  done
  echo "GPU still busy after 50 min; refusing to launch into a contended card"
  return 1
}
wait_for_idle || { echo "compile_board exit=1 $(date -u +%FT%TZ)"; exit 1; }

run_cell() {   # name, CROSSOVER_COMPILE value
  local name="$1" comp="$2"
  rm -rf "$OUT/$name"
  local s=$(date +%s)
  CROSSOVER_ARMS="$ARMS" CROSSOVER_JOB_PREFIX="$name" \
  CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=10000000 \
  CROSSOVER_COMPILE="$comp" \
  python3 -u -m nanolab.crossover_replicate launch \
    --out "$OUT/$name" --workers 1 2>&1 | tail -4
  echo "WALLCLOCK cond=$name compile=$comp elapsed=$(( $(date +%s) - s ))s"
  python3 -c "
import json,sys
r=json.load(open('$OUT/$name/recipe.json'))
want = ('$comp' == '1')
print('  recipe recorded: compile=%s fused_ce=%s' % (r.get('compile'), r.get('fused_ce')))
sys.exit(0 if r.get('compile') is want else 1)" || {
    echo "  recipe did NOT record compile=$comp; this cell is not what it claims"
    echo "compile_board exit=1 $(date -u +%FT%TZ)"; exit 1; }
}

echo "=== compile on a real board, 3 arms x 5 seeds x 10M $(date -u +%FT%TZ) ==="
run_cell cmpval_eager 0
run_cell cmpval_on    1

echo
echo "=== per-arm readout ==="
python3 - <<'PY'
import json, glob, os, statistics
from collections import defaultdict

def board(root):
    out = defaultdict(dict)
    for f in sorted(glob.glob(root + "/*/metrics.jsonl")):
        done = None
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get("event") == "done":
                    done = r                    # last: append-mode safe
        if not done:
            continue
        name = os.path.basename(os.path.dirname(f))
        body, seed = name.rsplit("_s", 1)
        arm = body.split("_", 1)[1]
        out[arm][int(seed)] = (done.get("final_val"), done.get("elapsed_s"))
    return out

a = board("nanolab/out/_tune/cmpval_eager")
b = board("nanolab/out/_tune/cmpval_on")
STEP_LOOP = {"attention": 1.94, "hybrid_mingru8_attn4": 1.95, "gdn": 3.17}

for arm in sorted(set(a) & set(b)):
    seeds = sorted(set(a[arm]) & set(b[arm]))
    if not seeds:
        continue
    sp = [a[arm][s][1] / b[arm][s][1] for s in seeds]
    dv = [b[arm][s][0] - a[arm][s][0] for s in seeds]
    et, ec = sum(a[arm][s][1] for s in seeds), sum(b[arm][s][1] for s in seeds)
    print("\n  %s  (n=%d)" % (arm, len(seeds)))
    print("    eager %6.0fs -> compiled %6.0fs   board %.2fx   step loop %.2fx  %s"
          % (et, ec, et / ec, STEP_LOOP.get(arm, 0),
             "AGREES" if abs(et / ec - STEP_LOOP.get(arm, 0)) < 0.25 else "<-- DIVERGES"))
    print("    per-seed speedups: " + " ".join("%.2f" % x for x in sp))
    print("    final_val: mean %+.4f  mean ABS %.4f  max ABS %.4f  (floor 0.0014/0.0031)"
          % (statistics.mean(dv), statistics.mean(map(abs, dv)), max(map(abs, dv))))
PY
echo "compile_board exit=0 $(date -u +%FT%TZ)"
