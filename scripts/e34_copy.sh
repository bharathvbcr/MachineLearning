#!/usr/bin/env bash
# E34: is the attention/minGRU crossing the induction-head bump?
# Config.copy_probe logs `copy_loss` (CE on the second occurrence of a repeated
# 32-token span, fixed probe batches) beside val_loss at every eval. Training
# is untouched; the recipe records the flag, so this is its own directory.
# Pre-registered: per seed, the copy-loss drop within +-1M tokens of the
# attention-over-minGRU crossing on 4/5 seeds -> predictive; else withdrawn.
source "$(dirname "$0")/_stage_common.sh"
stage_wait; stage_start e34
export CROSSOVER_COPY_PROBE=1
export CROSSOVER_ARMS=attention,mingru,hybrid_mingru8_attn4
export CROSSOVER_JOB_PREFIX=cx32copy
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover50m_copy32 --workers 3
rc=$?; echo "e34 launch exit=$rc $(date -u +%FT%TZ)"
python3 scripts/paired_board.py mingru@crossover50m_copy32 hybrid_mingru8_attn4@crossover50m_copy32 --ref attention@crossover50m_copy32
python3 - <<'PYEOF'
import json
from pathlib import Path
from nanolab.crossover_replicate import _arm_from_run_dir
root = Path("nanolab/out/crossover50m_copy32"); runs = {}
for d in sorted(root.iterdir()):
    m = d / "metrics.jsonl"
    if not d.is_dir() or not m.exists(): continue
    seed = json.loads((d / "config.json").read_text())["seed"]; ev = []
    for line in m.read_text().splitlines():
        try: r = json.loads(line)
        except Exception: continue
        if r.get("event") == "eval" and r.get("copy_loss") is not None:
            ev.append((r["tokens"], r["val_loss"], r["copy_loss"]))
    if ev: runs.setdefault(_arm_from_run_dir(d), {})[seed] = ev
att, mg = runs.get("attention", {}), runs.get("mingru", {})
print("per seed: token where attention's copy_loss first drops below the midpoint of its range, and the token where attention overtakes minGRU")
for s in sorted(set(att) & set(mg)):
    cl = [c for _, _, c in att[s]]; mid = (max(cl) + min(cl)) / 2
    drop = next((t for t, _, c in att[s] if c < mid), None)
    a = {t: v for t, v, _ in att[s]}; g = {t: v for t, v, _ in mg[s]}
    cross = next((t for t in sorted(set(a) & set(g)) if t > 2e6 and a[t] < g[t]), None)
    print(f"  seed {s}: copy drop {drop/1e6 if drop else float('nan'):.2f}M   crossing {cross/1e6 if cross else float('nan'):.2f}M   "
          f"copy_loss max {max(cl):.3f} min {min(cl):.3f}")
PYEOF
echo "REMINDER: bash scripts/pull_artifacts.sh crossover50m_copy32"
stage_end e34 "$rc"
