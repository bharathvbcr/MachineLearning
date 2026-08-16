"""
nanolab.bench_gpu — measure (and maximize) GPU utilization (guide §7).

The training-loop logger reports MFU, but for *optimization* we want a tight,
repeatable micro-benchmark that isolates GPU compute from data I/O and prints a
per-phase breakdown (forward / backward / optimizer) plus peak memory and the
achieved vs theoretical FLOPs.

    python -m nanolab.bench_gpu --batch_size 8 --block_size 1024
    python -m nanolab.bench_gpu --batch_size 24 --fused_ce true --tf32 true

Uses synthetic in-GPU tokens by default so the number reflects the GPU ceiling,
not the CPU dataloader (which is profiled separately). The enemy is idle tensor
cores (§7): watch tok/s and MFU climb as you toggle the levers below.
"""

from __future__ import annotations

import argparse
import time

import torch

from .config import build_config
from .model import build_model
from .optim import build_optimizers
from .utils import format_count


# 3070 Ti Laptop bf16 tensor-core peak (FP16/BF16 w/ FP32 accumulate), dense.
# ~5120 CUDA / 160 tensor cores @ ~1.785 GHz -> ~46 TFLOP/s realistic dense bf16.
DEFAULT_PEAK_FLOPS = 46e12


def synth_batch(cfg, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size), generator=g)
    return x.to(device), x.to(device)


def bench(cfg, peak_flops, warmup=5, iters=20):
    assert torch.cuda.is_available(), "no CUDA device"
    device = "cuda"
    torch.manual_seed(cfg.seed)

    model = build_model(cfg).to(device)
    optimizers = build_optimizers(model, cfg)
    flops_per_tok = model.flops_per_token()
    x, y = synth_batch(cfg, device)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[cfg.dtype]
    autocast = (torch.autocast("cuda", dtype=dtype)
                if cfg.dtype != "fp32" else torch.autocast("cuda", enabled=False))

    def one_step(measure=False):
        phases = {}
        t = {k: torch.cuda.Event(enable_timing=True) for k in
             ("s", "fwd", "bwd", "opt")}
        if measure:
            t["s"].record()
        with autocast:
            _, loss = model(x, y)
            loss = loss / cfg.grad_accum
        if measure:
            t["fwd"].record()
        loss.backward()
        if measure:
            t["bwd"].record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if measure:
            t["opt"].record()
            torch.cuda.synchronize()
            phases["fwd"] = t["s"].elapsed_time(t["fwd"])
            phases["bwd"] = t["fwd"].elapsed_time(t["bwd"])
            phases["opt"] = t["bwd"].elapsed_time(t["opt"])
        return loss, phases

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    agg = {"fwd": 0.0, "bwd": 0.0, "opt": 0.0}
    last_loss = None
    for _ in range(iters):
        last_loss, ph = one_step(measure=True)
        for k in agg:
            agg[k] += ph[k]
    torch.cuda.synchronize()
    wall = time.time() - t0

    tokens = cfg.batch_size * cfg.block_size * iters
    tok_s = tokens / wall
    achieved = flops_per_tok * tokens / wall
    mfu = achieved / peak_flops
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9

    return {
        "tok_s": tok_s, "mfu": mfu, "peak_mem_gb": peak_mem,
        "reserved_gb": reserved, "ms_per_step": wall / iters * 1000,
        "fwd_ms": agg["fwd"] / iters, "bwd_ms": agg["bwd"] / iters,
        "opt_ms": agg["opt"] / iters, "loss": last_loss.item() * cfg.grad_accum,
        "params_m": model.num_params() / 1e6, "flops_per_tok": flops_per_tok,
    }


def _apply_perf_flags(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.sdp_flash:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="phase1")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--mixer", default="attention")
    p.add_argument("--optimizer", default="muon")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--fused_ce", type=_b, default=None,
                   help="chunked fused linear cross-entropy (saves logits memory)")
    p.add_argument("--fused_ce_chunks", type=int, default=4)
    p.add_argument("--grad_checkpoint", type=_b, default=False)
    p.add_argument("--tf32", type=_b, default=True)
    p.add_argument("--sdp_flash", type=_b, default=True)
    p.add_argument("--peak_flops", type=float, default=DEFAULT_PEAK_FLOPS)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    _apply_perf_flags(args)

    overrides = dict(run_name="bench", batch_size=args.batch_size,
                     block_size=args.block_size, grad_accum=args.grad_accum,
                     mixer=args.mixer, optimizer=args.optimizer, dtype=args.dtype,
                     grad_checkpoint=args.grad_checkpoint, compile=False)
    if args.fused_ce is not None:
        overrides["fused_ce"] = args.fused_ce
        overrides["fused_ce_chunks"] = args.fused_ce_chunks
    cfg = build_config(args.preset, overrides)

    print(f"\n=== bench: {cfg.mixer} {cfg.optimizer} bs{cfg.batch_size} "
          f"ctx{cfg.block_size} {cfg.dtype} | tf32={args.tf32} "
          f"fused_ce={getattr(cfg,'fused_ce',False)} gckpt={cfg.grad_checkpoint} ===")
    try:
        r = bench(cfg, args.peak_flops, iters=args.iters)
    except torch.cuda.OutOfMemoryError:
        print("  *** OOM at this config ***")
        return
    print(f"  params      : {r['params_m']:.1f}M")
    print(f"  throughput  : {format_count(r['tok_s'])} tok/s  ({r['ms_per_step']:.1f} ms/step)")
    print(f"  MFU         : {r['mfu']*100:.1f}%  (peak {args.peak_flops/1e12:.0f} TFLOP/s)")
    print(f"  peak memory : {r['peak_mem_gb']:.2f} GB alloc / {r['reserved_gb']:.2f} GB reserved (8 GB card)")
    print(f"  phase ms    : fwd {r['fwd_ms']:.1f} | bwd {r['bwd_ms']:.1f} | opt {r['opt_ms']:.1f}")
    print(f"  loss        : {r['loss']:.3f}")


def _b(v):
    return str(v).lower() in ("1", "true", "yes", "y", "on")


if __name__ == "__main__":
    main()
