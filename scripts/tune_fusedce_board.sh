#!/usr/bin/env bash
# Does the 3.02x survive a real board, or is it a step-loop artefact?
#
# The 2x2 in section 1b was one fwd+bwd on fixed weights. Its compile half
# reproduces the five-seed end-to-end result (1.94x both) and its eager half
# reproduces appendix F (1.29x both), so the unfused-compiled cell is the only
# claim on that page resting on a microbenchmark alone. This runs it.
#
# Paired design: same arm, same seeds, same budget, compile ON in both cells,
# fused_ce the only variable. Two directories because fused_ce is now a recorded
# recipe field -- which is the point of recording it, and lock_recipe would
# refuse to mix them in one.
#
# Expect ~455 s fused and ~292 s unfused from the step-loop rates. A board that
# lands far off those is the interesting outcome, not a failure.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
OUT=nanolab/out/_tune

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
wait_for_idle || { echo "fusedce_board exit=1 $(date -u +%FT%TZ)"; exit 1; }

run_cell() {   # name, CROSSOVER_FUSED_CE value
  local name="$1" fce="$2"
  rm -rf "$OUT/$name"
  local s=$(date +%s)
  CROSSOVER_ARMS=attention CROSSOVER_JOB_PREFIX="$name" \
  CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=20000000 \
  CROSSOVER_COMPILE=1 CROSSOVER_FUSED_CE="$fce" \
  python3 -u -m nanolab.crossover_replicate launch \
    --out "$OUT/$name" --workers 1 2>&1 | tail -4
  echo "WALLCLOCK cond=$name fused_ce=$fce elapsed=$(( $(date +%s) - s ))s"
  # Prove the knob actually landed, rather than trusting the env var.
  python3 -c "
import json,sys
r=json.load(open('$OUT/$name/recipe.json'))
print('  recipe recorded: compile=%s fused_ce=%s' % (r.get('compile'), r.get('fused_ce')))
sys.exit(0 if r.get('compile') is True else 1)" || {
    echo "  recipe did NOT record compile=True; this cell is not what it claims"; return 1; }
}

echo "=== fused_ce on a real board, compile ON in both cells $(date -u +%FT%TZ) ==="
run_cell ce_fused   1
run_cell ce_unfused 0

echo
echo "=== paired readout ==="
python3 - <<'PY'
import json, glob, os, statistics
def board(root):
    out = {}
    for f in sorted(glob.glob(root + "/*/metrics.jsonl")):
        done = None
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get("event") == "done":
                    done = r                       # last: append-mode safe
        if done:
            seed = os.path.basename(os.path.dirname(f)).rsplit("_s", 1)[-1]
            out[seed] = (done.get("final_val"), done.get("elapsed_s"))
    return out
a = board("nanolab/out/_tune/ce_fused")
b = board("nanolab/out/_tune/ce_unfused")
seeds = sorted(set(a) & set(b), key=int)
if not seeds:
    print("  no paired seeds; nothing to report")
else:
    print("  seed      fused final_val   unfused    diff |  fused s  unfused s  speedup")
    ds, sp = [], []
    for s in seeds:
        fv, fe = a[s]; uv, ue = b[s]
        ds.append(uv - fv); sp.append(fe / ue)
        print("  %-8s %13.4f %10.4f %7.4f | %7.0f %9.0f %8.2fx"
              % (s, fv, uv, uv - fv, fe, ue, fe / ue))
    print()
    print("  final_val: mean diff %+.4f  mean ABS %.4f  max ABS %.4f  (rerun floor 0.0014/0.0031)"
          % (statistics.mean(ds), statistics.mean(map(abs, ds)), max(map(abs, ds))))
    print("  wall clock: mean speedup %.2fx  (step loop predicted 1.55x)" % statistics.mean(sp))
PY
echo "fusedce_board exit=0 $(date -u +%FT%TZ)"
