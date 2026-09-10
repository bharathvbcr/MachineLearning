#!/usr/bin/env bash
# E31: tied/untied embeddings x value residual, 2x2 on the attention arm.
#
# Every CE board is tied, and the MQAR probe found tied embeddings cap recall at
# 0.555; value residual is the largest single lever the sprint ladder found.
# The hypothesis is that part of VR's win is a workaround for tying (it gives
# every layer a non-embedding token representation). `attention` is the
# (tied, VR) cell. Untying adds 38,633,472 parameters at width 768: report it.
#
# Pre-registered: VR gain untied < half of VR gain tied -> interaction;
# unchanged -> independent.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused e31
stage_start e31
export CROSSOVER_ARMS=attention,attention_untied,attention_novr,attention_untied_novr
export CROSSOVER_JOB_PREFIX=cx32tie
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover50m_tie32 --workers 3
rc=$?; echo "e31 launch exit=$rc $(date -u +%FT%TZ)"
python3 scripts/paired_board.py attention_untied@crossover50m_tie32 attention_novr@crossover50m_tie32 attention_untied_novr@crossover50m_tie32 --ref attention@crossover50m_tie32
python3 - <<'PYEOF'
import json, statistics
from pathlib import Path
from nanolab.crossover_replicate import _arm_from_run_dir
root = Path("nanolab/out/crossover50m_tie32"); fin = {}
for d in sorted(root.iterdir()):
    m = d / "metrics.jsonl"
    if not d.is_dir() or not m.exists(): continue
    seed = json.loads((d / "config.json").read_text())["seed"]
    for line in m.read_text().splitlines():
        try: r = json.loads(line)
        except Exception: continue
        if r.get("event") == "done" and r.get("final_val") is not None:
            fin.setdefault(_arm_from_run_dir(d), {})[seed] = r["final_val"]
need = ("attention", "attention_untied", "attention_novr", "attention_untied_novr")
if all(a in fin for a in need):
    seeds = sorted(set.intersection(*(set(fin[a]) for a in need)))
    tied = [fin["attention_novr"][s] - fin["attention"][s] for s in seeds]          # VR gain, tied
    unt = [fin["attention_untied_novr"][s] - fin["attention_untied"][s] for s in seeds]  # VR gain, untied
    d = [t - u for t, u in zip(tied, unt)]
    se = statistics.stdev(d) / len(d) ** 0.5
    print(f"VR gain tied   {statistics.mean(tied):+.4f}  untied {statistics.mean(unt):+.4f}  "
          f"difference {statistics.mean(d):+.4f} 95% [{statistics.mean(d)-2.776*se:+.4f},{statistics.mean(d)+2.776*se:+.4f}]  n={len(d)}")
    print("pre-registered: untied gain < half of tied gain -> interaction; else independent")
else:
    print("E31 readout: incomplete arms", {a: len(fin.get(a, {})) for a in need})
PYEOF
echo "REMINDER: bash scripts/pull_artifacts.sh crossover50m_tie32"
stage_end e31 "$rc"
