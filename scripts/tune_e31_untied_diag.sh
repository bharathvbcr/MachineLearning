#!/usr/bin/env bash
# Why does E31's untied+no-VR cell report val 11.15 while training to 4.16?
#
# What is established: all five seeds do it; train loss descends normally and
# ends BELOW attention_untied's (4.156 vs 4.209) with grad norms ~0.4; val rises
# monotonically 10.67 -> 11.15, which is above uniform (ln 50257 = 10.82). At 50M
# tokens over a 497.5M-token corpus the model sees each token once, so train and
# val cannot legitimately separate by 7 nats. The val number is an artifact, and
# E31's pre-registered 2x2 readout cannot be computed until it is explained.
#
# Three tests, cheapest first, each one able to kill a hypothesis:
#
#  1. Re-evaluate the SAVED final.pt eagerly. This separates "the weights are
#     bad" from "the evaluation was wrong" and costs one forward pass. If eager
#     eval returns ~4.2, the weights are fine and the recorded val came from the
#     compiled eval graph -- which `evaluate` builds separately, since it runs
#     under model.eval() + no_grad.
#  2. The same checkpoint, evaluated COMPILED, on the same batches. If (1) is
#     fine and (2) reproduces 11.15, that is the compiled eval graph, isolated.
#  3. A short eager training run of the same arm. Confirms the cell is
#     measurable at all once compile is off, which is what a rerun would need.
#
# Chained behind the program: `workers` is a recorded recipe field and a fourth
# process would retag every running board's tenancy.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
LOG=nanolab/out/_program.log
echo "e31diag waiting for tune_program ($(date -u +%FT%TZ))"
if [ "${FOLLOWUP_NOWAIT:-0}" = "1" ]; then
  echo "  (driver already waited)"
else
deadline=$(( $(date +%s) + 20*3600 ))
while ! grep -q "^tune_program exit=" "$LOG" 2>/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "gave up waiting"; exit 1; }
  sleep 120
done
quiet=0
while true; do
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
    quiet=0
  else
    quiet=$((quiet+1)); [ "$quiet" -ge 3 ] && break
  fi
  sleep 20
done
fi
echo "e31diag start $(date -u +%FT%TZ)"

python3 -u - <<'PYEOF'
import json, torch
from pathlib import Path
from nanolab.config import build_config
from nanolab.model import build_model
from nanolab.data import Batcher
from nanolab.train import evaluate
import contextlib

root = Path("nanolab/out/crossover50m_tie32")
for arm in ("attention_untied_novr", "attention_untied"):
    d = root / f"cx32tie_{arm}_s1337"
    cfg = build_config(None, json.loads((d / "config.json").read_text()))
    ckpt = torch.load(d / "final.pt", map_location="cuda", weights_only=False)
    sd = ckpt.get("model", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    print(f"\n=== {arm} ===")
    for tag, use_compile in (("eager", False), ("compiled", True)):
        torch.manual_seed(0)
        m = build_model(cfg).to("cuda")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [{tag}] state_dict missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("     missing:", sorted(missing)[:6])
        if use_compile:
            import torch._dynamo as dynamo
            dynamo.reset()
            m = torch.compile(m)
        vb = Batcher(Path(cfg.data_dir), "val", cfg, "cuda")
        tb = Batcher(Path(cfg.data_dir), "train", cfg, "cuda")
        v = evaluate(m, vb, cfg, ctx)
        t = evaluate(m, tb, cfg, ctx)
        print(f"  [{tag:8s}] val {v:8.4f}   train-split {t:8.4f}   gap {v - t:+.4f}")
PYEOF
echo "e31diag exit=$? $(date -u +%FT%TZ)"
