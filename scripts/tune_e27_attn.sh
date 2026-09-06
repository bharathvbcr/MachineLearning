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
#
# Tenancy is NOT a free choice here. recipe.json for this directory locks
# workers=2, and lock_recipe refuses any mismatch on a shared field, so the four
# reruns must land at the tenancy s777 already ran at or they cannot pool with
# it. --ignore-vram is required for the same reason: the planner sizes this cell
# by SCALING the 768 measurement (40.2 GiB/job -> refuses 2), and its scaling is
# demonstrably conservative -- it predicts 54.0 GiB for mingru w1536, which
# measured 46.7 GB on this box today. Two attention jobs leave >30 GiB free,
# which is well clear of the 9.27 GiB where the sampler silently switches paths,
# so the pessimistic case here OOMs loudly rather than training on other tokens.
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH

# The first attempt died because a chain fired on a log marker while another
# stage's workers still held 51 GiB. A marker says a script returned, not that
# the card is free -- launch processes outlive the parent that printed it. Wait
# for three consecutive clear checks before claiming the device.
wait_for_idle() {
  local quiet=0
  echo "waiting for the GPU to go idle ($(date -u +%FT%TZ))"
  for _ in $(seq 1 120); do
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
  echo "GPU still busy after 40 min; refusing to launch into a contended card"
  return 1
}
wait_for_idle || { echo "e27_attn exit=1 $(date -u +%FT%TZ)"; exit 1; }

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
