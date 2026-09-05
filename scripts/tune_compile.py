"""Is torch.compile still unusable on this box, and what would it buy?

Two repo facts make this worth re-testing rather than assuming:
  * `compile=False` is hardcoded in crossover_replicate.job_config and in
    current_recipe, and train.py additionally refuses to compile anything that
    is not a pure-attention stack ("recurrent loops / expert dispatch graph-break
    badly"). Both predate torch 2.7 on aarch64.
  * At the board shape a 124M model reaches ~14% MFU with the GPU busy ~100% of
    the wall clock -- i.e. the kernels are too small, which is exactly what
    Inductor fusion addresses. GDN is worst (2.7% MFU, 4.7x slower than
    attention) and is the program's bottleneck arm.

For each arm: compile latency, steady-state throughput vs eager, and max|dy|
between compiled and eager outputs on identical inputs -- because adopting
compile is a NUMERICS change (new recipe, new directory), not a free win, and
the size of that change should be a measured number.

    python scripts/tune_compile.py attention mingru hybrid_mingru8_attn4 gdn
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
ARMS_DEFAULT = ["attention", "mingru", "hybrid_mingru8_attn4", "gdn", "moe_e8k1"]


def build(arm_name):
    import os
    from nanolab.crossover_replicate import ARMS, job_config
    from nanolab.model import build_model
    from nanolab.optim import build_optimizers
    arm = next(a for a in ARMS if a.name == arm_name)
    os.environ["CROSSOVER_BATCH"] = "32"
    os.environ["CROSSOVER_BLOCK"] = "512"
    os.environ["CROSSOVER_TOKEN_BUDGET"] = "50000000"
    cfg = job_config({"id": "cmp", "arm": arm.name, "mixer": arm.mixer,
                      "layer_mixers": arm.layer_mixers, "seed": 1337}, OUT)
    torch.manual_seed(1337)
    model = build_model(cfg).to("cuda")
    return cfg, model, build_optimizers(model, cfg)


def timed(model, opts, cfg, x, y, iters):
    ac = torch.autocast("cuda", dtype=torch.bfloat16)

    def step():
        with ac:
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.time() - t) / iters * 1000
    return {"ms_per_step": ms,
            "tok_s": cfg.batch_size * cfg.block_size / (ms / 1000),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9}


def run(arm_name, iters=15) -> dict:
    res = {"arm": arm_name}
    cfg, model, opts = build(arm_name)
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size), generator=g).cuda()
    y = torch.roll(x, -1, dims=1)          # next-token, not synth_batch's y==x

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_logits, _ = model(x, y)
        ref = ref_logits.float().clone() if ref_logits is not None else None
    res["eager"] = timed(model, opts, cfg, x, y, iters)
    print(f"  {arm_name:<24} eager    {res['eager']['tok_s']/1e3:7.1f}K tok/s  "
          f"{res['eager']['ms_per_step']:7.1f} ms", flush=True)

    del model, opts
    torch.cuda.empty_cache()
    cfg, model, opts = build(arm_name)
    t0 = time.time()
    try:
        cmodel = torch.compile(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = cmodel(x, y)
        loss.backward()
        for o in opts:
            o.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        res["compile_s"] = time.time() - t0
    except Exception as e:
        res["compiled"] = {"err": f"{type(e).__name__}: {str(e)[:300]}"}
        res["compile_s"] = time.time() - t0
        print(f"  {arm_name:<24} compile  FAILED after {res['compile_s']:.0f}s: "
              f"{type(e).__name__}: {str(e)[:160]}", flush=True)
        return res

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cl, _ = cmodel(x, y)
        res["max_abs_logit_diff"] = (
            float((cl.float() - ref).abs().max()) if ref is not None else None)
    res["compiled"] = timed(cmodel, opts, cfg, x, y, iters)
    res["speedup"] = res["compiled"]["tok_s"] / res["eager"]["tok_s"]
    print(f"  {arm_name:<24} compiled {res['compiled']['tok_s']/1e3:7.1f}K tok/s  "
          f"{res['compiled']['ms_per_step']:7.1f} ms  "
          f"-> {res['speedup']:.2f}x  (compile {res['compile_s']:.0f}s, "
          f"max|dlogit| {res.get('max_abs_logit_diff')})", flush=True)
    del model, cmodel, opts
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    import importlib.util
    arms = sys.argv[1:] or ARMS_DEFAULT
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"triton available: {importlib.util.find_spec('triton') is not None}  "
          f"torch {torch.__version__}")
    out = {}
    for a in arms:
        try:
            out[a] = run(a)
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[a] = {"ERROR": f"{type(e).__name__}: {e}"}
        torch.cuda.empty_cache()
    (OUT / "compile.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT/'compile.json'}")
