"""
nanolab.sweep_gpu — run EVERY model variant on the GPU and rank them (guide §7).

``bench_gpu`` measures one config; this drives the same ``bench()`` across the
whole registry so you get one apples-to-apples table per axis:

  * mixers      — attention / mingru / mamba2 / gdn / mla  (optimizer fixed)
  * optimizers  — muon / adamw / sgd / lion / schedulefree / sophia / prodigy
                  (mixer fixed)
  * ffn         — swiglu / relu2 / gelu / moe

Everything else (params budget, context, batch, dtype, seed) is held fixed, so
the only thing moving between rows is the one axis named in the header. Each row
reports throughput (tok/s), MFU, peak memory, and the fwd/bwd/opt phase split —
the numbers you actually optimize against. OOM rows are reported, not fatal.

    python -m nanolab.sweep_gpu all      --batch_size 8 --block_size 1024
    python -m nanolab.sweep_gpu mixer    --batch_size 8
    python -m nanolab.sweep_gpu optimizer
    python -m nanolab.sweep_gpu ffn

Use a fast, GPU-resident synthetic batch (no dataloader), so the ranking
reflects the compute ceiling, not I/O.
"""

from __future__ import annotations

import argparse
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
    p.add_argument("axis", choices=["all", "mixer", "optimizer", "ffn"])
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
