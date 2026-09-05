"""
nanolab.sweep_gpu — run EVERY model variant on the GPU and rank them (guide §7).

``bench_gpu`` measures one config; this drives the same ``bench()`` across the
whole registry so you get one apples-to-apples table per axis:

  * mixers      — attention / mingru / mamba2 / gdn / mla  (optimizer fixed)
  * optimizers  — muon / adamw / sgd / lion / schedulefree / sophia / prodigy
                  (mixer fixed)
  * ffn         — swiglu / relu2 / gelu / moe
  * arm         — the crossover_replicate ARM registry (hybrids and per-arm
                  overrides included), configured through `job_config` so a row
                  is the cost of the job the suite runner would actually launch

Everything else (params budget, context, batch, dtype, seed) is held fixed, so
the only thing moving between rows is the one axis named in the header. Each row
reports throughput (tok/s), MFU, peak memory, and the fwd/bwd/opt phase split —
the numbers you actually optimize against. OOM rows are reported, not fatal.

    python -m nanolab.sweep_gpu all      --batch_size 8 --block_size 1024
    python -m nanolab.sweep_gpu mixer    --batch_size 8
    python -m nanolab.sweep_gpu optimizer
    python -m nanolab.sweep_gpu ffn
    python -m nanolab.sweep_gpu arm --arms attention,mingru,hybrid_mingru8_attn4 \
        --batch_size 32 --block_size 512

Use a fast, GPU-resident synthetic batch (no dataloader), so the ranking
reflects the compute ceiling, not I/O.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .bench_gpu import DEFAULT_PEAK_FLOPS, bench
from .config import MIXERS, OPTIMIZERS, build_config
from .utils import format_count

FFNS = ("swiglu", "relu2", "gelu", "moe")
# each optimizer gets a sensible peak LR (matches experiments.optimizer_bakeoff)
# so a row is "this optimizer's compute cost", not a mistuned LR. LRs don't
# affect throughput, but they keep the loss column honest.
_LR_FOR = {"sgd": 0.1, "lion": 1.2e-4, "prodigy": 1.0}


def _enable_perf_flags(tf32: bool, flash: bool, mem_fraction: float):
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if flash:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    # CRITICAL on 8 GB Windows/WDDM: cap the allocator so an over-budget config
    # raises a clean OutOfMemoryError instead of silently spilling to host RAM
    # over PCIe (the "sysmem fallback" — ~25x slower, looks like a multi-minute
    # hang at 100% util / low power). With the cap, the sweep reports "OOM" for
    # that row in milliseconds and moves on.
    if mem_fraction and mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)


def _bench_one(base_over, peak_flops, iters):
    cfg = build_config("phase1", base_over)
    try:
        r = bench(cfg, peak_flops, iters=iters)
        r["ok"] = True
        return r
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"ok": False, "err": "OOM"}
    except Exception as e:                       # keep the sweep going
        torch.cuda.empty_cache()
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}


def _row(label, r):
    if not r["ok"]:
        return f"  {label:<14} *** {r['err']} ***"
    return (f"  {label:<14} {format_count(r['tok_s']):>9} tok/s | "
            f"MFU {r['mfu']*100:5.1f}% | {r['peak_mem_gb']:4.2f}GB | "
            f"fwd {r['fwd_ms']:5.1f} bwd {r['bwd_ms']:5.1f} opt {r['opt_ms']:5.1f} ms | "
            f"loss {r['loss']:.2f}")


def _bench_arm(name, args):
    """Cost of one ARM at the board shape, built the way the runner builds it.

    Goes through `crossover_replicate.job_config` rather than `build_config` so
    the row carries the arm's `layer_mixers` and its own overrides (SWA window,
    gdn_rule, mingru_expand, ...). Those are what make it that arm, and a bench
    that dropped them would price a different model than the suite runs.
    """
    import os
    from .crossover_replicate import job_config
    # job_config reads the recipe from the environment; pin it to this sweep's
    # shape so the row is the board's cost, not the module defaults.
    os.environ["CROSSOVER_BATCH"] = str(args.batch_size)
    os.environ["CROSSOVER_BLOCK"] = str(args.block_size)
    os.environ["CROSSOVER_TOKEN_BUDGET"] = str(args.token_budget)
    job = {"id": f"bench_{name}", "arm": name, "mixer": _arm_mixer(name),
           "layer_mixers": _arm_layers(name), "seed": 1337}
    cfg = job_config(job, Path("nanolab/out/_bench"))
    try:
        r = bench(cfg, args.peak_flops, iters=args.iters)
        r["ok"] = True
        r["params_total"] = cfg_params(cfg)
        return r
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"ok": False, "err": "OOM"}
    except Exception as e:
        torch.cuda.empty_cache()
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}


def cfg_params(cfg) -> int:
    from .model import build_model
    return build_model(cfg).num_params()


def _arm_mixer(name: str) -> str:
    from .crossover_replicate import ARMS
    for a in ARMS:
        if a.name == name:
            return a.mixer
    raise SystemExit(f"unknown arm {name!r}")


def _arm_layers(name: str) -> str:
    from .crossover_replicate import ARMS
    for a in ARMS:
        if a.name == name:
            return a.layer_mixers
    raise SystemExit(f"unknown arm {name!r}")


def _sweep_arms(names, args):
    print(f"\n########## GPU SWEEP: arm "
          f"(bs{args.batch_size} ctx{args.block_size} {args.dtype}) ##########")
    print("  each row is one suite arm at the board shape, via job_config\n")
    rows = []
    for name in names:
        r = _bench_arm(name, args)
        print(_row(name, r))
        rows.append((name, r))
        # 20+ rows in one process: without this the freed model/optimizer blocks
        # stay in the caching allocator and the next row's peak_mem reads high.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    ok = [(v, r) for v, r in rows if r["ok"]]
    ok.sort(key=lambda vr: -vr[1]["tok_s"])
    if ok:
        base = ok[-1][1]["tok_s"]
        print("\n  --- arm: ranked by throughput ---")
        for rank, (v, r) in enumerate(ok, 1):
            print(f"   {rank}. {v:<32} {format_count(r['tok_s'])} tok/s  "
                  f"{r['ms_per_step']:7.1f} ms/step  {r['peak_mem_gb']:5.2f}GB  "
                  f"{r['tok_s']/base:5.2f}x slowest")
    return {v: r for v, r in rows}


def _sweep(axis, values, fixed, args):
    print(f"\n########## GPU SWEEP: {axis} "
          f"(bs{args.batch_size} ctx{args.block_size} {args.dtype}) ##########")
    print(f"  fixed: {fixed}  | params held ~constant; only '{axis}' varies\n")
    rows = []
    for v in values:
        over = dict(run_name=f"sweep_{axis}_{v}", batch_size=args.batch_size,
                    block_size=args.block_size, grad_accum=1, dtype=args.dtype,
                    fused_ce=args.fused_ce, fused_ce_chunks=args.fused_ce_chunks,
                    grad_checkpoint=args.grad_checkpoint, compile=False)
        over.update(fixed)
        over[axis] = v
        if axis == "optimizer" and v in _LR_FOR:
            over["lr"] = _LR_FOR[v]
        r = _bench_one(over, args.peak_flops, args.iters)
        print(_row(v, r))
        rows.append((v, r))
    # rank the ones that ran, by throughput
    ok = [(v, r) for v, r in rows if r["ok"]]
    ok.sort(key=lambda vr: -vr[1]["tok_s"])
    print(f"\n  --- {axis}: ranked by throughput ---")
    for rank, (v, r) in enumerate(ok, 1):
        print(f"   {rank}. {v:<14} {format_count(r['tok_s'])} tok/s  "
              f"MFU {r['mfu']*100:.1f}%  {r['peak_mem_gb']:.2f}GB")
    return {v: r for v, r in rows}


def main():
    p = argparse.ArgumentParser(description="GPU sweep over the nanolab registry")
    p.add_argument("axis", choices=["all", "mixer", "optimizer", "ffn", "arm"])
    p.add_argument("--arms", default="",
                   help="comma-separated ARM names for the `arm` axis")
    p.add_argument("--token_budget", type=int, default=50_000_000,
                   help="token budget job_config scales the `arm` axis to")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--mixer", default="attention", help="fixed mixer for non-mixer sweeps")
    p.add_argument("--optimizer", default="muon", help="fixed optimizer for non-optimizer sweeps")
    p.add_argument("--fused_ce", type=_b, default=True)
    p.add_argument("--fused_ce_chunks", type=int, default=8)
    # grad-checkpoint ON by default: at ctx1024/124M on 8 GB it is the difference
    # between a 2.9 GB step and a >8 GB sysmem-fallback thrash (see probe_perf).
    p.add_argument("--grad_checkpoint", type=_b, default=True)
    p.add_argument("--tf32", type=_b, default=True)
    p.add_argument("--sdp_flash", type=_b, default=True)
    p.add_argument("--mem_fraction", type=float, default=0.92,
                   help="cap VRAM so over-budget configs OOM cleanly instead of "
                        "thrashing host RAM (0 disables the cap)")
    p.add_argument("--peak_flops", type=float, default=DEFAULT_PEAK_FLOPS)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--out", default="", help="optional JSON path to dump all rows")
    args = p.parse_args()

    assert torch.cuda.is_available(), "no CUDA device"
    _enable_perf_flags(args.tf32, args.sdp_flash, args.mem_fraction)
    name = torch.cuda.get_device_name(0)
    print(f"device: {name}  | tf32={args.tf32} flash={args.sdp_flash} "
          f"fused_ce={args.fused_ce}/{args.fused_ce_chunks} gckpt={args.grad_checkpoint} "
          f"mem_cap={args.mem_fraction}")

    t0 = time.time()
    results = {}
    if args.axis == "arm":
        if not args.arms:
            raise SystemExit("--arms is required for the `arm` axis")
        results["arm"] = _sweep_arms(args.arms.split(","), args)
    if args.axis in ("all", "mixer"):
        results["mixer"] = _sweep("mixer", list(MIXERS),
                                  {"optimizer": args.optimizer}, args)
    if args.axis in ("all", "optimizer"):
        results["optimizer"] = _sweep("optimizer", list(OPTIMIZERS),
                                      {"mixer": args.mixer}, args)
    if args.axis in ("all", "ffn"):
        results["ffn"] = _sweep("ffn", list(FFNS),
                                {"mixer": args.mixer, "optimizer": args.optimizer}, args)
    print(f"\nsweep done in {time.time()-t0:.0f}s")

    if args.out:
        # strip non-serializable, keep the scalar metrics
        dump = {ax: {v: {k: x for k, x in r.items() if isinstance(x, (int, float, bool, str))}
                     for v, r in rows.items()} for ax, rows in results.items()}
        Path(args.out).write_text(json.dumps(dump, indent=2))
        print(f"wrote {args.out}")


def _b(v):
    return str(v).lower() in ("1", "true", "yes", "y", "on")


if __name__ == "__main__":
    main()
