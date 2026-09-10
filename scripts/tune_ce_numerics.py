"""What does turning fused cross-entropy OFF cost in numerics, and does it
compose with torch.compile?

Unfused CE measured 1.29x on attention and 1.24x on the hybrid, for +10.8 GB --
affordable on a 94.5 GiB card, and the chunking was sized for the 8 GB card this
repo started on. But `fused_ce` is a Config field, so flipping it is a numerics
change, and the throughput table cannot price that: its `loss` column is read
after five optimizer steps, so the two settings have already diverged onto
different trajectories by then.

This isolates it: ONE forward pass, identical weights, identical batch, fused
against unfused. Any difference is the chunked reduction's association order and
nothing else. Then it times fused/unfused x eager/compiled together, because two
speed-ups that each move ~1.3-1.9x are only worth stacking if they compose.

    python scripts/tune_ce_numerics.py
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


def cfg_for(arm_name, **extra):
    import os
    from nanolab.crossover_replicate import ARMS, job_config
    arm = next(a for a in ARMS if a.name == arm_name)
    os.environ["CROSSOVER_BATCH"] = "32"
    os.environ["CROSSOVER_BLOCK"] = "512"
    os.environ["CROSSOVER_TOKEN_BUDGET"] = "50000000"
    cfg = job_config({"id": "ce", "arm": arm.name, "mixer": arm.mixer,
                      "layer_mixers": arm.layer_mixers, "seed": 1337}, OUT)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def main() -> None:
    from nanolab.model import build_model
    from nanolab.optim import build_optimizers
    res = {}

    # ---- 1. numerics: same weights, same batch, one forward ----
    print("=== fused vs unfused CE: one forward, identical weights ===")
    for arm in ("attention", "hybrid_mingru8_attn4"):
        torch.manual_seed(1337)
        cfg = cfg_for(arm)
        model = build_model(cfg).to("cuda").eval()
        g = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size),
                          generator=g).cuda()
        y = torch.roll(x, -1, dims=1)
        losses = {}
        with torch.no_grad():
            for label, fused, chunks in (("fused/16", True, 16), ("fused/4", True, 4),
                                         ("unfused", False, 0)):
                # Model.forward reads cfg.fused_ce at call time off the Config
                # it was built with, so mutating the object switches the path
                # without rebuilding -- which is the point: identical weights.
                cfg.fused_ce, cfg.fused_ce_chunks = fused, (chunks or 16)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                losses[label] = float(loss)
        base = losses["fused/16"]
        res[arm] = {"losses": losses,
                    "abs_diff_unfused_vs_fused16": abs(losses["unfused"] - base),
                    "rel_diff": abs(losses["unfused"] - base) / abs(base)}
        print(f"  {arm}")
        for k, v in losses.items():
            print(f"    {k:<10} loss {v:.10f}")
        print(f"    unfused - fused/16 = {losses['unfused']-base:+.3e} "
              f"({res[arm]['rel_diff']:.2e} relative)")
        del model
        torch.cuda.empty_cache()

    # ---- 2. do the two speed-ups compose? ----
    print("\n=== throughput: fused x compiled (attention) ===")
    res["compose"] = {}
    for fused in (True, False):
        for comp in (False, True):
            label = f"{'fused/16' if fused else 'unfused':<9} {'compiled' if comp else 'eager':<8}"
            torch.manual_seed(1337)
            cfg = cfg_for("attention", fused_ce=fused, fused_ce_chunks=16)
            model = build_model(cfg).to("cuda")
            opts = build_optimizers(model, cfg)
            if comp:
                model = torch.compile(model)
            g = torch.Generator(device="cpu").manual_seed(0)
            x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size),
                              generator=g).cuda()
            y = torch.roll(x, -1, dims=1)
            ac = torch.autocast("cuda", dtype=torch.bfloat16)

            def step():
                with ac:
                    _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                for o in opts:
                    o.step()
                    o.zero_grad(set_to_none=True)

            try:
                for _ in range(6):
                    step()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                for _ in range(15):
                    step()
                torch.cuda.synchronize()
                ms = (time.time() - t) / 15 * 1000
                tok = cfg.batch_size * cfg.block_size / (ms / 1000)
                gb = torch.cuda.max_memory_reserved() / 1e9
                res["compose"][label.strip()] = {"tok_s": tok, "ms": ms, "resv_gb": gb}
                print(f"  {label} {tok/1e3:7.1f}K tok/s  {ms:7.1f} ms  {gb:5.1f} GB")
            except torch.cuda.OutOfMemoryError:
                res["compose"][label.strip()] = {"err": "OOM"}
                print(f"  {label} OOM")
            del model, opts
            torch.cuda.empty_cache()

    ok = {k: v for k, v in res["compose"].items() if "err" not in v}
    if "fused/16 eager" in ok:
        b = ok["fused/16 eager"]["tok_s"]
        print(f"\n  relative to the configured fused/16 eager ({b/1e3:.1f}K tok/s):")
        for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["tok_s"]):
            print(f"    {k:<20} {v['tok_s']/b:5.2f}x")
    (OUT / "ce_numerics.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT/'ce_numerics.json'}")


if __name__ == "__main__":
    main()
