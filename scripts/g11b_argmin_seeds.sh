#!/usr/bin/env bash
# G11b: the two experiments G11 was written to run and did not.
#
# G11 exited 2 having launched a quarter of its plan and none of the point that
# mattered. Its cell table was a here-string iterated with `for row in $CELLS`, and
# word splitting does not respect lines: each row's trailing multiplier list
# `lr05 lr0667 lr10` split into three separate "rows", of which the second and third
# carried no colons, so `${row%%:*}` returned the whole token for every field and the
# launcher was invoked as `--workers lr0667`. argparse refused it 48 times. What DID
# run was the first multiplier of each line -- one NEIGHBOUR per width, never an argmin,
# and never 0.667x. This file reads its cells with `while IFS= read -r`, one line at a
# time, which is the fix.
#
# That accident then produced a false finding, and the false finding produced a real
# one. G11's readout announced that two argmins had moved. Neither had: it had enriched
# one point per curve to n=3 and left its neighbours at n=1, and `lr_argmin.py` was
# taking the plain mean of whatever seeds each point carried. Seed 100 is the easiest
# seed at every cell on these boards and 777 the hardest -- 4.1229 against 4.1803 at
# d1152 attention 1x, a 0.0574 spread, eighteen times the 0.0031 rerun floor, where the
# argmin margins being decided are 0.005 to 0.05. `lr_argmin.py` now pairs by seed and
# refuses a curve with no seed in common, and the four argmins read exactly as before.
#
# So this stage is the same two experiments, re-scoped by what that reading showed:
#
#   1. d1152 at 0.667x -- the 1/width law's one unverified point, and the only place it
#      can fail. The law says the d1152 optimum is lr 0.0004; the grid has 0.5x (0.0003)
#      and 1x (0.0006) and 1x wins by 0.0047 paired on 3 seeds -- above the floor, but
#      the closest margin on the whole ladder, and 0.667x lies between them. Three seeds
#      here pair against 0.5x's three and against 1x's five.
#
#   2. n=3 under each width's argmin. G11 left every width with n=3 at a neighbour and
#      n=1 at the argmin itself, which is the wrong way round. Adding seeds 42 and 100
#      at the argmin makes each width's tightest comparison a 3-seed paired one:
#        d384  2x vs 1x   (margin 0.0167 at n=1 today)
#        d768  1x vs 0.5x (0.0393)
#        d1536 0.5x vs 1x (0.0085) -- both points, since 1x is also n=1 and it is the
#              tight side; 0.25x already carries three seeds.
#      d1152's argmin already has n=5, so it appears here only through 0.667x.
#
# Routing is forced by three tenancy locks, none of them chosen here:
#   d384, d768  -> crossover_probe50m        workers 3
#   d1152       -> crossover_ladder50m       workers 2
#   d1536       -> crossover_probe50m_w1536  workers 1   (w1536 cannot share a device)
#
# ~30 runs, about 4.9 h of GH200 at the tenancies above, ~$11.2 at $2.29/h.
#
# Pre-registered reading:
#   * The law holds at d1152 if 0.667x beats BOTH 0.5x and 1x paired, by more than the
#     0.0031 floor. If it lands inside the floor of 1x, the law is CONSISTENT with the
#     data and not established by it, and must be written up that way -- a flat basin is
#     not a confirmed prediction.
#   * If 0.667x is WORSE than 1x beyond the floor, `width x lr = 0.4608` is falsified at
#     the one width that could falsify it, and the 1/width claim comes out of the writeup
#     entirely. Three points on a line through two free parameters is not a law.
#   * Any argmin that moves once its own point carries three seeds was never located, and
#     G9's re-tuned ladder rung for that width is retired with it. This is now a real
#     test rather than G11's: the comparison is paired, so a move means the ordering
#     flipped on shared seeds, not that the seed sets differed.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g11b
stage_start g11b
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

# width : board : workers : prefix : multipliers : seeds
# Read one LINE at a time. `for row in $CELLS` splits on every space, which is the
# bug this file exists to repair.
CELLS="384:nanolab/out/crossover_probe50m:3:cx32p50:lr20:42 100
768:nanolab/out/crossover_probe50m:3:cx32p50:lr10:42 100
1152:nanolab/out/crossover_ladder50m:2:cx32lad50:lr0667:1337 42 100
1536:nanolab/out/crossover_probe50m_w1536:1:cx32p1536:lr05 lr10:42 100"

while IFS= read -r row; do
  [ -z "$row" ] && continue
  W=${row%%:*};    rest=${row#*:}
  OUT=${rest%%:*}; rest=${rest#*:}
  NW=${rest%%:*};  rest=${rest#*:}
  PFX=${rest%%:*}; rest=${rest#*:}
  MULTS=${rest%%:*}
  SEEDS=${rest#*:}
  case "$NW" in ''|*[!0-9]*) echo "g11b: REFUSING row '$row' -- workers '$NW' is not a number"; rc=3; continue;; esac
  for mx in attention mingru; do
    for m in $MULTS; do
      for seed in $SEEDS; do
        echo "g11b: w${W} ${mx} ${m} seed ${seed} -> ${OUT} (workers ${NW})"
        launch_cell "$OUT" "$NW" "$PFX" "$seed" "w${W}_${mx}_${m}"
      done
    done
  done
done <<< "$CELLS"

echo "g11b launch exit=$rc $(date -u +%FT%TZ)"
echo "=== argmins, paired by seed ==="
for b in crossover_probe50m crossover_ladder50m crossover_probe50m_w1536; do
  python3 scripts/lr_argmin.py "$b"
done
echo "=== the 1/width law at every width ==="
python3 - <<'PYEOF'
import importlib.util
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
        print(f"  {w:<6d} {mx:<11s} {best:<11g} {lr:<9.6f} {w*lr:<12.4f} {verdict[:64]}")
print("  the law predicts width x lr = 0.4608 at every row")
print("  a row whose verdict is not 'interior' is NOT evidence for or against it")
PYEOF
echo "=== d1152: the law's point against both its neighbours, paired ==="
python3 - <<'PYEOF'
import importlib.util, statistics
sp = importlib.util.spec_from_file_location("lr_argmin", "scripts/lr_argmin.py")
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
cells = m.cells("crossover_ladder50m")
FLOOR = m.RERUN_FLOOR
for mx in ("attention", "mingru"):
    curve = cells.get((1152, mx)) or {}
    law = curve.get(0.6667) or curve.get(0.667)
    if not law:
        print(f"  d1152 {mx}: 0.667x has no finished run; the law is untested here")
        continue
    for nb in (0.5, 1.0):
        other = curve.get(nb)
        if not other:
            print(f"  d1152 {mx}: no {nb:g}x to compare against")
            continue
        shared = sorted(set(law) & set(other))
        if not shared:
            print(f"  d1152 {mx}: 0.667x and {nb:g}x share no seed; not comparable")
            continue
        d = [other[s] - law[s] for s in shared]
        mean = statistics.mean(d)
        wins = sum(1 for x in d if x > 0)
        if mean > FLOOR:
            verdict = f"0.667x WINS by {mean:.4f} ({wins}/{len(d)}), above the {FLOOR} floor"
        elif mean < -FLOOR:
            verdict = f"0.667x LOSES by {-mean:.4f} ({len(d)-wins}/{len(d)}), above the {FLOOR} floor"
        else:
            verdict = f"tied within the floor ({mean:+.4f}); consistent, not established"
        print(f"  d1152 {mx}: 0.667x vs {nb:g}x on seeds {shared} -- {verdict}")
PYEOF
echo "REMINDER: bash scripts/pull_artifacts.sh"
stage_end g11b "$rc"
