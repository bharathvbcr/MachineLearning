#!/usr/bin/env bash
# G12: settle three w1536 runs whose sampler path cannot be established.
#
# `cx32p1536_w1536_attention_lr05_s42`, `_lr05_s100` and `_lr10_s42` completed during the
# 2026-09-09 stage cascade at 33%, 44% and 54% of their cohort's median `mean_tok_s`.
# That degradation is equally consistent with co-tenancy and with the memmap fallback, and
# throughput cannot tell them apart -- so these runs are neither cleared nor condemned by
# it, they are simply unreadable.
#
# Why the fallback matters: this repo's train split is 497.5M tokens, above
# `data.py:_GPU_RESIDENT_MAX_TOKENS` (150M), so `should_gpu_resident` consults free VRAM
# and needs `free > 9.27 GiB`. Under that, the Batcher samples with a CPU generator
# instead of a CUDA one -- the same seed, a different token stream -- and it needs LESS
# memory, so it turns an OOM into a run that finishes and looks ordinary.
#
# Into a FRESH directory, never over the originals: the originals are the evidence for
# the defect this repairs, and overwriting evidence to make a caveat go away is not the
# same as measuring it away. Same rule G8c follows.
#
# Both copies now carry `"sampler"` in their start record (added 2026-09-10), so the
# comparison is direct. The determinism reading is the independent one.
#
# Pre-registered reading:
#   * A rerun whose `final_val` agrees within the 0.0031 rerun floor means the original
#     drew the same token stream: the original stands and the d1536 argmin is unchanged.
#   * A gap far above the floor means the original trained on different tokens. It is
#     replaced by the rerun and the d1536 argmin is re-read without it -- and since 0.5x
#     is the argmin and 1x its tight-side neighbour (0.0085 apart), that re-read can move
#     the top rung of the width ladder.
#   * Any run that reports `sampler: memmap` is condemned on that alone, whatever its loss.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g12
stage_start g12
rc=0
OUT=nanolab/out/crossover_p1536_samplercheck
for spec in "w1536_attention_lr05:42" "w1536_attention_lr05:100" "w1536_attention_lr10:42"; do
  arm=${spec%%:*}; seed=${spec#*:}
  CROSSOVER_ARMS=$(python3 - "$OUT" "$arm" <<'PYEOF'
import json, os, sys
out, arm = sys.argv[1], sys.argv[2]
rec = os.path.join(out, "recipe.json")
have = json.load(open(rec))["arms"] if os.path.exists(rec) else []
print(",".join(dict.fromkeys(list(have) + [arm])))
PYEOF
)
  export CROSSOVER_ARMS CROSSOVER_JOB_PREFIX=cx32sc1536
  echo "g12: $arm seed $seed -> $OUT (workers 1)"
  python3 -u -m nanolab.crossover_replicate launch --out "$OUT" --workers 1 \
    --arm "$arm" --seed "$seed" || rc=$?
done
echo "g12 launch exit=$rc $(date -u +%FT%TZ)"
echo "=== original vs rerun, against the 0.0031 floor ==="
python3 - <<'PYEOF'
import json, os
FLOOR = 0.0031
def read(d):
    m, c = os.path.join(d, "metrics.jsonl"), os.path.join(d, "config.json")
    if not (os.path.exists(m) and os.path.exists(c)):
        return None, None
    fin, samp = None, None
    for line in open(m):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "done":
            fin = r.get("final_val")
        if r.get("event") == "start":
            samp = r.get("sampler", "unrecorded")
    return fin, samp
PAIRS = [("nanolab/out/crossover_probe50m_w1536/cx32p1536_w1536_attention_lr05_s42",
          "nanolab/out/crossover_p1536_samplercheck/cx32sc1536_w1536_attention_lr05_s42"),
         ("nanolab/out/crossover_probe50m_w1536/cx32p1536_w1536_attention_lr05_s100",
          "nanolab/out/crossover_p1536_samplercheck/cx32sc1536_w1536_attention_lr05_s100"),
         ("nanolab/out/crossover_probe50m_w1536/cx32p1536_w1536_attention_lr10_s42",
          "nanolab/out/crossover_p1536_samplercheck/cx32sc1536_w1536_attention_lr10_s42")]
bad = 0
for orig, new in PAIRS:
    fo, so = read(orig)
    fn, sn = read(new)
    name = os.path.basename(orig)
    if fo is None or fn is None:
        print("  %-46s INCOMPLETE (orig=%s rerun=%s) -- this is NOT agreement"
              % (name, fo, fn))
        bad += 1
        continue
    d = abs(fo - fn)
    verdict = "AGREES" if d <= FLOOR else "DIFFERS -- original trained on other tokens"
    if d > FLOOR:
        bad += 1
    print("  %-46s orig=%.4f (%s)  rerun=%.4f (%s)  |d|=%.4f  %s"
          % (name, fo, so, fn, sn, d, verdict))
print("  %d of %d unresolved or condemned" % (bad, len(PAIRS)))
PYEOF
echo "REMINDER: bash scripts/pull_artifacts.sh"
stage_end g12 "$rc"
