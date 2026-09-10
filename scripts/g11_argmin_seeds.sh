#!/usr/bin/env bash
# G11: put n=3 under every 50M argmin, and test the 1/width law where it can fail.
#
# Both of those turned out to be one experiment. The law says the d1152 optimum sits at
# lr 0.0004 (= 0.4608/1152, two-thirds of base); the ladder grid is 2x-spaced and has
# only 0.5x (4.1659) and 1x (4.1595), straddling it 0.0064 apart. Fitting the d1536
# basin -- the one measured on both sides -- puts the gain of the true argmin over 1x at
# ~0.0115, so the 0.667x-vs-1x gap at d1152 should land around 0.002-0.006 nats. That is
# AT OR BELOW the 0.0031 rerun floor, so the two-job version of this test cannot resolve
# it at n=1 no matter what it returns. The law's point and the neighbours both need n=3,
# which is the same 25 arm-cells either way.
#
# So: the argmin and its two neighbours, at n=3, at all four widths -- plus the new
# 0.667x point at d1152. Every cell currently on disk is n=1 at seed 1337 (or n=5 where
# G3 ran), so this adds seeds 42 and 100.
#
# Routing is forced by three separate tenancy locks, none of them chosen here:
#   d384, d768  -> crossover_probe50m        workers 3
#   d1152       -> crossover_ladder50m       workers 2
#   d1536       -> crossover_probe50m_w1536  workers 1   (w1536 cannot share a device)
#
# Pre-registered reading:
#   * The law holds at d1152 if 0.667x beats BOTH 0.5x and 1x on the n=3 mean by more
#     than the 0.0031 floor. If it lands inside the floor of 1x, the law is consistent
#     with the data but not established by it, and must be reported that way -- a flat
#     basin is not a confirmed prediction.
#   * If 0.667x is WORSE than 1x beyond the floor, `width x lr = 0.4608` is falsified at
#     the one width that could falsify it, and the 1/width claim comes out of the writeup
#     entirely -- three points on a line through two free parameters is not a law.
#   * Each width's argmin is re-read at n=3. Any argmin that moves under the extra seeds
#     was never located, and G9's re-tuned ladder rung for that width is retired with it.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g11
stage_start g11
rc=0
launch_cell() {   # board workers prefix seed arm
  local out="$1" w="$2" pfx="$3" seed="$4" arm="$5"
  CROSSOVER_ARMS=$(python3 - "$out" "$arm" <<'PYEOF'
import json, os, sys
out, arm = sys.argv[1], sys.argv[2]
rec = os.path.join(out, "recipe.json")
have = json.load(open(rec))["arms"] if os.path.exists(rec) else []
print(",".join(dict.fromkeys(list(have) + [arm])))
PYEOF
)
  export CROSSOVER_ARMS CROSSOVER_JOB_PREFIX="$pfx"
  python3 -u -m nanolab.crossover_replicate launch --out "$out" --workers "$w" \
    --arm "$arm" --seed "$seed" || rc=$?
}

# width : board : workers : prefix : the argmin and its two neighbours
CELLS="
384:nanolab/out/crossover_probe50m:3:cx32p50:lr10 lr20 lr40
768:nanolab/out/crossover_probe50m:3:cx32p50:lr05 lr10 lr20
1152:nanolab/out/crossover_ladder50m:2:cx32lad50:lr05 lr0667 lr10
1536:nanolab/out/crossover_probe50m_w1536:1:cx32p1536:lr025 lr05 lr10
"
for row in $CELLS; do
  W=${row%%:*}; rest=${row#*:}
  OUT=${rest%%:*}; rest=${rest#*:}
  NW=${rest%%:*}; rest=${rest#*:}
  PFX=${rest%%:*}; MULTS=${rest#*:}
  for mx in attention mingru; do
    for m in $MULTS; do
      for seed in 1337 42 100; do
        echo "g11: w${W} ${mx} ${m} seed ${seed} -> ${OUT} (workers ${NW})"
        launch_cell "$OUT" "$NW" "$PFX" "$seed" "w${W}_${mx}_${m}"
      done
    done
  done
done
echo "g11 launch exit=$rc $(date -u +%FT%TZ)"
echo "=== argmins re-read at n=3 ==="
for b in crossover_probe50m crossover_ladder50m crossover_probe50m_w1536; do
  python3 scripts/lr_argmin.py "$b" 2>/dev/null | grep -E "^  d|argmin"
done
echo "=== does width x lr hold? ==="
python3 - <<'PYEOF'
import importlib.util, pathlib
sp = importlib.util.spec_from_file_location("lr_argmin", "scripts/lr_argmin.py")
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
BOARD = {384: "crossover_probe50m", 768: "crossover_probe50m",
         1152: "crossover_ladder50m", 1536: "crossover_probe50m_w1536"}
print("  width  mixer       argmin      lr        width x lr   verdict")
for w, b in BOARD.items():
    try:
        cells = m.cells(b)
    except SystemExit:
        continue
    for mx in ("attention", "mingru"):
        c = cells.get((w, mx))
        if not c:
            continue
        best, verdict = m.argmin(c)
        lr = best * 6e-4
        print(f"  {w:<6d} {mx:<11s} {best:<11g} {lr:<9.6f} {w*lr:<12.4f} {verdict}")
print("  the law predicts width x lr = 0.4608 at every row")
PYEOF
echo "REMINDER: bash scripts/pull_artifacts.sh"
stage_end g11 "$rc"
