"""Can the hybrids be compiled too, or is 1.00x really their ceiling?

Compile is worth 1.94x on attention and 1.96x on a pure minGRU stack, but the
hybrid measured exactly 1.00x -- because Dynamo gave up:

    torch._dynamo hit config.recompile_limit (8)
    function: 'forward' (nanolab/model.py:410)
    last reason: 0/7: GLOBAL_STATE changed: grad_mode

That is not the mixer graph-breaking. `grad_mode` is the train/eval switch: the
suite evaluates 61 times per 50M run inside `torch.no_grad()`, and each flip is a
new guard. A stack with two mixer kinds has more variants to begin with, so it
exhausts the budget first and falls back to eager permanently.

This matters more than the single arm suggests: E30, E32 and E34 are hybrid
boards, so a hybrid stuck at 1.00x while attention runs 1.94x would make the
compiled program lopsided.

Tests raising the limit, and reports whether the hybrid then reaches the ~1.9x
the pure stacks get. Also re-checks GDN and MoE, which were measured at 0.97x and
1.08x with the default limit and may simply have been paying the same tax.

    python scripts/tune_compile_hybrid.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

OUT = REPO / "nanolab/out/_tune"
ARMS = ["hybrid_mingru8_attn4", "hybrid_mingru_periodic", "gdn", "moe_e8k1"]


def build(arm_name):
    import os
    from nanolab.crossover_replicate import ARMS as REG, job_config
    from nanolab.model import build_model
    from nanolab.optim import build_optimizers
    arm = next(a for a in REG if a.name == arm_name)
    os.environ["CROSSOVER_BATCH"] = "32"
    os.environ["CROSSOVER_BLOCK"] = "512"
    os.environ["CROSSOVER_TOKEN_BUDGET"] = "50000000"
    cfg = job_config({"id": "ch", "arm": arm.name, "mixer": arm.mixer,
                      "layer_mixers": arm.layer_mixers, "seed": 1337}, OUT)
    torch.manual_seed(1337)
    model = build_model(cfg).to("cuda")
    return cfg, model, build_optimizers(model, cfg)


def timed(model, opts, cfg, x, y, iters=12, warm=5, evals=True):
    ac = torch.autocast("cuda", dtype=torch.bfloat16)

    def step():
        with ac:
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)

    def evaluate():
        # The grad_mode flip that generates the recompiles in the first place.
        model.eval()
        with torch.no_grad(), ac:
            model(x, y)
        model.train()

    for _ in range(warm):
        step()
        if evals:
            evaluate()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.time() - t) / iters * 1000
    return {"ms_per_step": ms, "tok_s": cfg.batch_size * cfg.block_size / (ms / 1000)}


def run(arm, limit) -> dict:
    import torch._dynamo as dynamo
    dynamo.reset()
    if limit is not None:
        # Renamed from cache_size_limit; torch 2.7 names it recompile_limit and
        # the warning quotes that name.
        for attr in ("recompile_limit", "cache_size_limit"):
            if hasattr(dynamo.config, attr):
                setattr(dynamo.config, attr, limit)
    cfg, model, opts = build(arm)
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size),
                      generator=g).cuda()
    y = torch.roll(x, -1, dims=1)
    eager = timed(model, opts, cfg, x, y)
    del model, opts
    torch.cuda.empty_cache()

    cfg, model, opts = build(arm)
    t0 = time.time()
    cm = torch.compile(model)
    try:
        comp = timed(cm, opts, cfg, x, y)
    except Exception as e:
        return {"eager": eager, "err": f"{type(e).__name__}: {str(e)[:200]}"}
    out = {"eager": eager, "compiled": comp, "compile_s": time.time() - t0,
           "speedup": comp["tok_s"] / eager["tok_s"], "limit": limit}
    del model, cm, opts
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    res = {}
    print(f"torch {torch.__version__}")
    for arm in ARMS:
        res[arm] = {}
        for limit in (None, 64):
            tag = "default(8)" if limit is None else str(limit)
            try:
                r = run(arm, limit)
                res[arm][tag] = r
                if "err" in r:
                    print(f"  {arm:<24} limit {tag:<10} ERROR {r['err'][:70]}")
                else:
                    print(f"  {arm:<24} limit {tag:<10} "
                          f"{r['eager']['tok_s']/1e3:6.1f}K -> "
                          f"{r['compiled']['tok_s']/1e3:6.1f}K  "
                          f"**{r['speedup']:.2f}x**  (compile {r['compile_s']:.0f}s)",
                          flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                res[arm][tag] = {"ERROR": str(e)[:200]}
            torch.cuda.empty_cache()
    (OUT / "compile_hybrid.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {OUT/'compile_hybrid.json'}")
