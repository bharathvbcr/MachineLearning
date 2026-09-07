#!/usr/bin/env bash
# Smoke every arm the E29-E35 program needs, compiled, before any board runs.
#
# Two things are being checked, and neither is the science:
#   1. Every new arm builds, trains and writes a `done` record. Sixteen of them
#      have never executed -- they were written on 2026-09-05 and the box was
#      reset before anything ran.
#   2. `torch.compile` survives each one. The gate was widened to every arm on
#      the strength of a 5-seed board that covered attention/mingru/gdn/hybrid/
#      moe -- NOT untied embeddings, not expansion 1, not 6x512, not width 384.
#      A graph break is fine; a crash 40 minutes into a board is not.
#
# Throwaway directories, tiny budget. The side effect is the point of doing it
# first: Inductor's on-disk cache comes out warm, and the boards that follow pay
# 244 s instead of 603 s for the compiles they share.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"

export CROSSOVER_BATCH=32 CROSSOVER_BLOCK=512 CROSSOVER_EVAL_ITERS=20
export CROSSOVER_TOKEN_BUDGET=2000000
export CROSSOVER_COMPILE=1

fail=0
run() {                      # run <tag> <dir> <arms...>
  local tag="$1" dir="$2"; shift 2
  echo "===== smoke $tag  $(date -u +%FT%TZ) ====="
  rm -rf "nanolab/out/$dir"
  CROSSOVER_ARMS="$1" CROSSOVER_JOB_PREFIX="smk${tag}" \
    python3 -u -m nanolab.crossover_replicate launch \
      --out "nanolab/out/$dir" --workers 3 --seed 1337 2>&1 | tail -4
  local rc=${PIPESTATUS[0]}
  echo "smoke $tag exit=$rc"
  [ "$rc" -ne 0 ] && fail=1
  return 0
}

run e29 _smoke_e29 "attention,moe_e1k1_raw,moe_e4k1_raw,moe_e8k1_raw"
run e30b _smoke_e30b "hybrid_mingru8_attn4_x1,hybrid_mingru_periodic_x1,mingru_x1"
run e31 _smoke_e31 "attention_untied,attention_novr,attention_untied_novr"
run e33 _smoke_e33 "attn6_w512,attn6_w576,w384_attention_lr10"
run e35 _smoke_e35 "w384_attention_lr80,w384_mingru_lr40,w384_hybrid_mingru8_attn4_lr40,w384_hybrid_mingru8_attn4_lr80"

# E34 varies copy_probe, which is a recipe field: its own directory.
echo "===== smoke e34 (copy probe) $(date -u +%FT%TZ) ====="
rm -rf nanolab/out/_smoke_e34
CROSSOVER_COPY_PROBE=1 CROSSOVER_ARMS="attention,mingru,hybrid_mingru8_attn4" \
CROSSOVER_JOB_PREFIX=smke34 \
  python3 -u -m nanolab.crossover_replicate launch \
    --out nanolab/out/_smoke_e34 --workers 3 --seed 1337 2>&1 | tail -4
rc=$?; echo "smoke e34 exit=$rc"; [ "$rc" -ne 0 ] && fail=1

echo "===== readout $(date -u +%FT%TZ) ====="
python3 - <<'PYEOF'
import json
from pathlib import Path
bad = []
for root in sorted(Path("nanolab/out").glob("_smoke_e*")):
    for d in sorted(root.iterdir()):
        m = d / "metrics.jsonl"
        if not d.is_dir() or not m.exists():
            continue
        done = comp = None
        for line in m.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") == "done":
                done = r
            if r.get("event") == "compile":
                comp = r
        tag = f"{root.name}/{d.name}"
        if done is None:
            bad.append(f"{tag}: no done record")
            continue
        fv = done.get("final_val")
        extra = ""
        if (d / "metrics.jsonl").read_text().find('"copy_loss"') >= 0:
            extra = "  copy_loss logged"
        ok = fv is not None and fv == fv and fv < 20
        if not ok:
            bad.append(f"{tag}: final_val={fv}")
        print(f"  {'ok ' if ok else 'BAD'} {tag:52s} final_val={fv}"
              f"  {done.get('elapsed_s', '?')}s{extra}")
print()
if bad:
    print("SMOKE FAILURES:")
    for b in bad:
        print("  " + b)
else:
    print("every smoked arm trained and wrote a finite final_val")
PYEOF
echo "smoke_program exit=$fail $(date -u +%FT%TZ)"
