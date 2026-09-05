#!/usr/bin/env bash
# E27 (width-1536 ladder) repair, per the handoff's step 0, with today's numbers.
#
# State, re-verified 2026-09-05: `crossover_ladder1536` holds 2 w1536_attention_lr80
# jobs stuck at `running` (orphaned 06:05Z, both with ckpt.pt so they resume),
# 3 more attention pending and 5 w1536_mingru_lr20 pending; the probe directory
# has 4 done and 2 CUDA-OOM failures (w1536_mingru lr40 and lr80, which were
# 45.9 + 48.4 GiB co-resident at workers 2).
#
# Order matters: the minGRU arm's learning rate was picked from the SURVIVORS of
# that OOM and landed on lr20, the edge of the swept range (the script itself
# printed EDGE:). So the probe cells are rerun FIRST, at workers 1 in a fresh
# directory (the old one records workers 2 and lock_recipe refuses a different
# tenancy), and the minGRU arm stays held until its argmin is known.
#
# Tenancy, measured today: at width 1536 one minGRU job is ~45 GiB real and two
# do not fit -- the guard and the measurement agree on workers 1 there. The
# attention half must run at workers 2 because that is what its recipe records,
# and 2 x ~31.6 GiB = 63 GiB of 94.5 fits; the guard's refusal quotes 40.2 GiB,
# which its own message flags as "SCALED from the 768 cell, not measured", so
# --ignore-vram is used with a measured figure behind it rather than a hope.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
LAD=nanolab/out/crossover_ladder1536
PROBE_B=nanolab/out/crossover_ladder_probe1536b

echo "=== E27 REPAIR $(date -u +%FT%TZ) ==="

# ---- 1. queue hygiene: claim_job only claims `pending`, so the orphans are
#         invisible to any relaunch until they are reset; hold the minGRU arm
#         so the attention half cannot drag it along at the wrong LR.
python3 - <<'PY'
from pathlib import Path
from nanolab.crossover_replicate import _lock_load, _lock_save
q = Path("nanolab/out/crossover_ladder1536/queue.json")
fh, state = _lock_load(q)
changed = {}
for j in state["jobs"]:
    if j["status"] == "running":
        j["status"] = "pending"; changed[j["id"]] = "running->pending (has ckpt, resumes)"
    elif j["arm"] == "w1536_mingru_lr20" and j["status"] == "pending":
        j["status"] = "held";    changed[j["id"]] = "pending->held (LR unsettled)"
_lock_save(fh, state)
for k, v in changed.items():
    print(f"  {k}: {v}")
from collections import Counter
print("  queue now:", dict(Counter(j["status"] for j in state["jobs"])))
PY

# ---- 2. rerun the two OOMed probe cells, alone on the device
echo "--- probe rerun (w1536_mingru lr40, lr80) at workers 1 $(date -u +%FT%TZ) ---"
s=$(date +%s)
CROSSOVER_ARMS=w1536_mingru_lr40,w1536_mingru_lr80 CROSSOVER_JOB_PREFIX=cx32lad1536pb \
CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=10000000 \
python3 -u -m nanolab.crossover_replicate launch --out "$PROBE_B" --workers 1 --seed 1337 2>&1 | tail -4
echo "probe_rerun elapsed=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)"

# ---- 3. the minGRU argmin, read from the run records of BOTH probe directories
echo "--- minGRU argmin over lr20 / lr40 / lr80 ---"
python3 - <<'PY'
import json
from pathlib import Path

def final_val(run_dir: Path):
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return None
    last = None
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
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
        if not d.is_dir():
            continue
        for lr in ("lr20", "lr40", "lr80"):
            if f"mingru_{lr}" in d.name:
                v = final_val(d)
                if v is not None:
                    vals[lr] = v
for lr in ("lr20", "lr40", "lr80"):
    print(f"  w1536_mingru_{lr}: {vals.get(lr, float('nan')):.4f}"
          if lr in vals else f"  w1536_mingru_{lr}: MISSING")
if len(vals) == 3:
    best = min(vals, key=vals.get)
    edge = best in ("lr20", "lr80")
    print(f"  ARGMIN={best}" + ("  (EDGE of the swept range)" if edge else "  (interior)"))
    Path("nanolab/out/_tune/e27_argmin.txt").write_text(best)
else:
    print("  argmin NOT decidable: a cell is missing; minGRU arm stays held")
PY

# ---- 4. the attention half, at the tenancy its recipe records
echo "--- attention half at workers 2 (recipe-locked) $(date -u +%FT%TZ) ---"
s=$(date +%s)
CROSSOVER_ARMS=w1536_attention_lr80,w1536_mingru_lr20 CROSSOVER_JOB_PREFIX=cx32lad1536 \
CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=50000000 \
python3 -u -m nanolab.crossover_replicate launch --out "$LAD" --workers 2 --ignore-vram 2>&1 | tail -5
echo "attention_half elapsed=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)"

echo "e27_repair exit=0 $(date -u +%FT%TZ)"
