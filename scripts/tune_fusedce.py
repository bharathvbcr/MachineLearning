"""Is the fused cross-entropy chunking helping or hurting on a 94 GiB card?

`crossover50m` sets `fused_ce=True, fused_ce_chunks=16`. Fused/chunked CE exists
to avoid materialising the full logits tensor -- at batch 32 x ctx 512 over a
50304 vocab that is 16384 x 50304 x 4 B = 3.3 GB in fp32, which mattered on the
8 GB card this repo started on. On a GH200 with 94.5 GiB it may simply be
splitting one efficient GEMM into 16 inefficient ones.

Sweeps chunk count and the unfused path, reporting throughput, peak memory, and
the loss each configuration computes on identical inputs -- because the chunk
count changes the reduction order, so adopting a different one is a numerics
change (new recipe, new directory), and the size of that change belongs in the
decision.

    python scripts/tune_fusedce.py
"""
from __future__ import annotations

import json
import sys
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
    cfg = job_config({"id": "fce", "arm": arm.name, "mixer": arm.mixer,
                      "layer_mixers": arm.layer_mixers, "seed": 1337}, OUT)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def main() -> None:
    from nanolab.bench_gpu import bench
    res = {}
    settings = [("fused/4", True, 4), ("fused/8", True, 8), ("fused/16", True, 16),
                ("fused/32", True, 32), ("unfused", False, 0)]
    for arm in ("attention", "hybrid_mingru8_attn4"):
        res[arm] = {}
        print(f"\n=== {arm} ===")
        for label, fused, chunks in settings:
            kw = {"fused_ce": fused}
            if fused:
                kw["fused_ce_chunks"] = chunks
            cfg = cfg_for(arm, **kw)
            try:
                r = bench(cfg, 752.8e12, iters=15)
                res[arm][label] = {k: r[k] for k in
                                   ("tok_s", "ms_per_step", "peak_mem_gb",
                                    "reserved_gb", "loss")}
                print(f"  {label:<10} {r['tok_s']/1e3:7.1f}K tok/s  "
                      f"{r['ms_per_step']:7.1f} ms  {r['peak_mem_gb']:5.2f}GB alloc / "
                      f"{r['reserved_gb']:5.2f}GB resv  loss {r['loss']:.6f}")
            except torch.cuda.OutOfMemoryError:
                res[arm][label] = {"err": "OOM"}
                print(f"  {label:<10} OOM")
            torch.cuda.empty_cache()
        ok = {k: v for k, v in res[arm].items() if "err" not in v}
        if ok:
            base = ok.get("fused/16")
            best = max(ok.items(), key=lambda kv: kv[1]["tok_s"])
            if base:
                print(f"  -> best {best[0]} at {best[1]['tok_s']/1e3:.1f}K tok/s = "
                      f"{best[1]['tok_s']/base['tok_s']:.2f}x the current "
                      f"fused/16 setting; extra VRAM "
                      f"{best[1]['reserved_gb'] - base['reserved_gb']:+.2f} GB")
    (OUT / "fusedce.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT/'fusedce.json'}")


if __name__ == "__main__":
    main()
