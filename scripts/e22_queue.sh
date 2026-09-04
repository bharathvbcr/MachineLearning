#!/usr/bin/env bash
# The queue that was waiting on GPU access, in dependency order.
#
# Three items, 40 jobs. Written 2026-09-04 while the box at 192.222.51.171 was
# presenting a host key that did not match known_hosts, so none of it has run.
# Nothing here is speculative: every arm is registered, unblocked, and its config
# verified by `--dry-run`.
#
#   1. e1_mup_tuned_spattn  10 jobs  closes PAPER 8.4 rows 1-3
#   2. e1_sp_coldattn       10 jobs  separates muP from the temperature
#   3. crossover50m_moe32c  20 jobs  MoE arms with the aux fix, + val_aux
#
# NOTE ON HARDWARE. Items 1-2 are in the bundle's same-box list: they join a
# board measured on the GH200 that ran suites 22-26, and PAPER 7.1 refuses a
# cross-hardware join. If the instance was recycled onto different silicon the
# preflight will REFUSE, and that refusal is correct -- do not reach for
# --allow-cross-hardware-board to get past it. Item 3 is self-contained and runs
# anywhere.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
echo "e22 queue start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

rc_any=0
run_stage() {   # name, then the command
  local name="$1"; shift
  echo "--- $name start $(date -u +%FT%TZ)"
  "$@"
  local rc=$?
  # A hardcoded exit=0 at the end of a stage is what turned two crashes into a
  # clean-looking log earlier in this line of work and idled the GPU for 54
  # minutes. Propagate, and keep going: a refusal on stage 1 must not silently
  # cancel stage 3, which does not depend on it.
  echo "--- $name exit=$rc $(date -u +%FT%TZ)"
  [ "$rc" -ne 0 ] && rc_any=$rc
  return 0
}

# --- 1 + 2: the muP cells. Both are unblocked -- the proxy and the basin anchor
# already exist on disk, so neither waits on anything.
run_stage e1_mup_tuned_spattn \
  python3 -u scripts/gpu_bundle.py --only e1_mup_tuned_spattn --workers 2
run_stage e1_sp_coldattn \
  python3 -u scripts/gpu_bundle.py --only e1_sp_coldattn --workers 2

python3 -u scripts/gpu_bundle.py --analyse 2>&1 | tail -40

# --- 3: the MoE board on the fixed reporting path. A NEW out dir on purpose:
# pre-fix runs carry the load-balancing aux inside final_val and must never be
# pooled with these by a resume.
export CROSSOVER_ARMS=attention,moe_e1k1,moe_e4k1,moe_e8k1
export CROSSOVER_JOB_PREFIX=cx32moec
export CROSSOVER_BATCH=32
export CROSSOVER_EVAL_ITERS=20
export CROSSOVER_TOKEN_BUDGET=50000000
run_stage crossover50m_moe32c \
  python3 -u -m nanolab.crossover_replicate launch \
    --out nanolab/out/crossover50m_moe32c --workers 2

python3 - <<'PYEOF'
import json, math, statistics
from collections import Counter
from pathlib import Path
from nanolab.crossover_replicate import _arm_from_run_dir
root = Path("nanolab/out/crossover50m_moe32c")
if not (root / "queue.json").exists():
    print("moe32c: no queue -- stage did not launch")
    raise SystemExit(0)
q = json.load(open(root / "queue.json"))
print("moe32c queue:", dict(Counter(j["status"] for j in q["jobs"])),
      "of", len(q["jobs"]))
by, aux = {}, {}
for d in sorted(root.iterdir()):
    m = d / "metrics.jsonl"
    if not d.is_dir() or not m.exists():
        continue
    fin = va = None
    for line in m.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "eval" and r.get("val_aux") is not None:
            va = r["val_aux"]
        if r.get("event") == "done":
            fin = r.get("final_val")
    if fin is not None:
        arm = _arm_from_run_dir(d)
        by.setdefault(arm, []).append(fin)
        if va is not None:
            aux.setdefault(arm, []).append(va)
print("\n  %-12s %3s %-26s %s" % ("arm", "n", "final_val [95% t]", "val_aux"))
rows = {}
for arm, f in sorted(by.items(), key=lambda kv: statistics.mean(kv[1])):
    mu = statistics.mean(f)
    sd = statistics.stdev(f) if len(f) > 1 else 0.0
    hw = 2.776 * sd / math.sqrt(len(f)) if len(f) > 1 else 0.0
    rows[arm] = (mu, mu - hw, mu + hw)
    a = aux.get(arm)
    print("  %-12s %3d %.4f [%.4f, %.4f] %s"
          % (arm, len(f), mu, mu - hw, mu + hw,
             ("%.4f" % statistics.median(a)) if a else "-"))
# final_val is now pure CE, so the control must simply MATCH -- no subtraction.
a, c = rows.get("attention"), rows.get("moe_e1k1")
if a and c:
    ok = c[1] <= a[2] and a[1] <= c[2]
    print("\nCONTROL: moe_e1k1 vs attention %s (gap %+.4f)"
          % ("OVERLAP -- the aux fix holds and the board is readable"
             if ok else
             "DISJOINT -- something OTHER than the aux still differs; do not "
             "read the parameter arms", c[0] - a[0]))
    print("val_aux is logged, not subtracted: at perfect balance it is "
          "moe_aux_weight * n_layer = 0.12; above that the router is collapsing.")
PYEOF
echo "e22 exit=$rc_any $(date -u +%FT%TZ)"
exit "$rc_any"
