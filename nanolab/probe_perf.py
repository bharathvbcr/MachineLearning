"""Per-phase GPU timing + peak-memory probe to localize throughput regressions.

The key failure mode on an 8 GB Windows/WDDM card is the *sysmem fallback*: when
allocations exceed VRAM the driver silently spills to host RAM over PCIe (~25x
slower) instead of OOMing, so a too-big batch reads as "100% util, low power,
seconds per step". This probe times a real fwd+bwd+opt step and prints peak
memory so that thrash is visible as a number, not a mystery hang.

    python -u -m nanolab.probe_perf --batch_size 8 --block_size 1024
    python -u -m nanolab.probe_perf --batch_size 8 --block_size 1024 --grad_checkpoint true --fused_ce true
"""
from __future__ import annotations

import argparse
import time

import torch

from .config import build_config
from .model import build_model
from .optim import build_optimizers


def cuda_time(fn, n=4, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000  # ms


def _b(v):
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--grad_checkpoint", type=_b, default=False)
    p.add_argument("--fused_ce", type=_b, default=False)
    p.add_argument("--fused_ce_chunks", type=int, default=8)
    p.add_argument("--mixer", default="attention")
    p.add_argument("--ffn", default="swiglu")
    p.add_argument("--mixer_chunk", type=int, default=0, help="0 -> Config default")
    p.add_argument("--optimizer", default="muon")
    args = p.parse_args()

    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    dev = "cuda"

    over = dict(
        batch_size=args.batch_size, block_size=args.block_size, grad_accum=1,
        mixer=args.mixer, ffn=args.ffn, optimizer=args.optimizer, dtype="bf16",
        compile=False, fused_ce=args.fused_ce, fused_ce_chunks=args.fused_ce_chunks,
        grad_checkpoint=args.grad_checkpoint)
    if args.mixer_chunk:
        over["mixer_chunk"] = args.mixer_chunk
    cfg = build_config("phase1", over)
    ce = f"fused_ce={cfg.fused_ce}" + (f"/{cfg.fused_ce_chunks}ch" if cfg.fused_ce else "")
    print(f"cfg: bs{cfg.batch_size} ctx{cfg.block_size} {cfg.mixer} ffn={cfg.ffn} "
          f"{cfg.optimizer} {ce} gckpt={cfg.grad_checkpoint}", flush=True)
    model = build_model(cfg).to(dev)
    model.train()
    opts = build_optimizers(model, cfg)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size), generator=g).to(dev)
    y = x.clone()
    ac = torch.autocast("cuda", dtype=torch.bfloat16)

    def full_step():
        for o in opts:
            o.zero_grad(set_to_none=True)
        with ac:
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for o in opts:
            o.step()
        return loss

    # warmup once, then reset peak-mem so the number reflects steady state
    full_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ms = cuda_time(full_step, n=4)
    tok_s = cfg.batch_size * cfg.block_size / (ms / 1000)
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    print(f"  full step: {ms:7.1f} ms  | {tok_s:8.0f} tok/s  | "
          f"peak {peak:.2f} GB alloc / {reserved:.2f} GB reserved", flush=True)


if __name__ == "__main__":
    main()
