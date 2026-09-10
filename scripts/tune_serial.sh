#!/usr/bin/env bash
# One serial driver for the remaining work.
#
# The chain of independent tmux/setsid waiters this replaces had two of them
# waiting on the SAME marker ("phase2 exit="), so they fired together: three
# val_m3 workers holding 51 GiB and E27's width-1536 probe wanting ~45 GiB on a
# 94.5 GiB card. Both probe cells OOMed in 20 s. They failed loudly rather than
# falling back to the memmap sampler, which is the good failure mode -- but the
# scheduling was mine to get right, and a single script that runs one stage at a
# time is the fix, not a longer chain.
#
# Waits for the GPU to be genuinely free before starting, since the killed
# stage's launch process outlives its parent script.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
OUT=nanolab/out/_tune

wait_for_idle() {
  local quiet=0
  echo "waiting for the GPU to go idle before starting ($(date -u +%FT%TZ))"
  while true; do
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
      [ "$quiet" -ge 3 ] && break     # three consecutive clear checks
    fi
    sleep 20
  done
  echo "GPU idle at $(date -u +%FT%TZ)"
}

step() { local n="$1" l="$2"; shift 2; echo "===== $n $(date -u +%FT%TZ) ====="; "$@" 2>&1 | tee "$l"; echo "$n exit=${PIPESTATUS[0]} $(date -u +%FT%TZ)"; }

wait_for_idle

# ---- 1. E27 probe rerun, alone on the device this time ----
echo "===== e27_probe $(date -u +%FT%TZ) ====="
python3 - <<'PY'
import json
from pathlib import Path
q = Path("nanolab/out/crossover_ladder_probe1536b/queue.json")
if q.exists():
    from nanolab.crossover_replicate import _lock_load, _lock_save
    fh, state = _lock_load(q)
    n = 0
    for j in state["jobs"]:
        if j["status"] == "failed":
            j["status"] = "pending"; n += 1
    _lock_save(fh, state)
    print(f"  reset {n} OOMed job(s) to pending")
else:
    print("  no probe queue yet; launch will create it")
PY
CROSSOVER_ARMS=w1536_mingru_lr40,w1536_mingru_lr80 CROSSOVER_JOB_PREFIX=cx32lad1536pb \
CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=10000000 \
python3 -u -m nanolab.crossover_replicate launch \
  --out nanolab/out/crossover_ladder_probe1536b --workers 1 --seed 1337 2>&1 | tail -5
echo "e27_probe exit=$? $(date -u +%FT%TZ)"

# ---- 2. argmin over the three cells, then the minGRU arm ----
echo "===== e27_argmin $(date -u +%FT%TZ) ====="
python3 - <<'PY'
import json
from pathlib import Path

def final_val(d: Path):
    f = d / "metrics.jsonl"
    if not f.exists():
        return None
    last = None
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("event") == "eval":
                last = r          # final_val, never best_val (repo rule 7)
    return last and last.get("val_loss")

vals = {}
for root in ("nanolab/out/crossover_ladder_probe1536",
             "nanolab/out/crossover_ladder_probe1536b"):
    p = Path(root)
    if not p.exists():
        continue
    for d in sorted(p.iterdir()):
        if d.is_dir():
            for lr in ("lr20", "lr40", "lr80"):
                if f"mingru_{lr}" in d.name:
                    v = final_val(d)
                    if v is not None:
                        vals[lr] = v
for lr in ("lr20", "lr40", "lr80"):
    print(f"  w1536_mingru_{lr}: " + (f"{vals[lr]:.4f}" if lr in vals else "MISSING"))
out = Path("nanolab/out/_tune/e27_argmin.txt")
if len(vals) == 3:
    best = min(vals, key=vals.get)
    print(f"  ARGMIN={best}" + ("  (EDGE of the swept range)" if best in ("lr20", "lr80")
                                else "  (interior)"))
    out.write_text(best)
else:
    print("  argmin NOT decidable; the minGRU arm stays held")
    out.unlink(missing_ok=True)
PY
step e27_mingru "$OUT/e27_mingru.log" bash scripts/tune_e27_mingru.sh

# ---- 3. remaining measurements ----
step ce_numerics "$OUT/ce_numerics.log" python3 -u scripts/tune_ce_numerics.py
step e28 "$OUT/e28.log" bash scripts/tune_e28_recall.sh
step compile_hybrid "$OUT/compile_hybrid.log" python3 -u scripts/tune_compile_hybrid.py

echo "tune_serial exit=0 $(date -u +%FT%TZ)"
