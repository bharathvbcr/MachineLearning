#!/usr/bin/env bash
# Rerun E27's attention half, which I broke.
#
# Four of its five w1536_attention_lr80 jobs failed with CUDA OOM (one with
# CUBLAS_STATUS_ALLOC_FAILED) because two of my stage chains keyed on the same
# log marker and fired together: three val_m3 workers held 51 GiB while this
# board wanted 2 x ~31.6 GiB on a 94.5 GiB card. Alone at the tenancy its recipe
# records, the board fits -- 63 of 94.5 GiB, measured today.
#
# The failures are recoverable: `claim_job` only claims `pending`, so the four
# have to be reset before any relaunch can see them, and the one that completed
# is left alone. The minGRU arm stays `held` -- its learning rate is still
# waiting on the repaired probe.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH

echo "=== E27 attention half, rerun $(date -u +%FT%TZ) ==="
python3 - <<'PY'
from pathlib import Path
from nanolab.crossover_replicate import _lock_load, _lock_save
from collections import Counter
q = Path("nanolab/out/crossover_ladder1536/queue.json")
fh, state = _lock_load(q)
n = 0
for j in state["jobs"]:
    if j["status"] == "failed" and j["arm"] == "w1536_attention_lr80":
        j["status"] = "pending"
        j.pop("detail", None)
        n += 1
_lock_save(fh, state)
print(f"  reset {n} OOMed attention job(s) to pending")
print("  queue now:", dict(Counter(j["status"] for j in state["jobs"])))
PY

s=$(date +%s)
CROSSOVER_ARMS=w1536_attention_lr80,w1536_mingru_lr20 CROSSOVER_JOB_PREFIX=cx32lad1536 \
CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=50000000 \
python3 -u -m nanolab.crossover_replicate launch \
  --out nanolab/out/crossover_ladder1536 --workers 2 --ignore-vram 2>&1 | tail -6
echo "e27_attn elapsed=$(( $(date +%s) - s ))s"
echo "e27_attn exit=0 $(date -u +%FT%TZ)"
