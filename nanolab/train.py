"""
nanolab.train — the training loop & instrumentation (guide §4.1, §6).

    python -m nanolab.train --preset phase0 --mixer attention
    python -m nanolab.train --preset phase2 --optimizer muon --schedule cosine
    python -m nanolab.train --preset cpu_smoke           # CPU sanity check

Everything the guide says to watch is logged from run one (§6.1):
    train loss · val loss · learning rate · grad norm · tokens/s · MFU

The loop is exactly the §4.1 skeleton: gradient accumulation, bf16 autocast,
global-norm gradient clipping, a schedule, and an optimizer step. Checkpoints
are written and resume is supported (§6.4 hygiene).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .config import add_config_args, config_from_args, parse_layer_mixers
from .data import Batcher, get_dataset
from .model import build_model
from .optim import build_optimizers, is_lr_free, is_schedule_free
from .schedules import apply_lr, make_schedule
from .utils import (Logger, pick_device, resolve_dtype,
                    set_seed, human_time)


def require_finite(step: int, **metrics: float) -> None:
    """Fail closed on NaN/Inf. A check that did not run must not look like a pass."""
    bad = [f"{name}={value}" for name, value in metrics.items()
           if not math.isfinite(float(value))]
    if bad:
        raise RuntimeError(f"non-finite at step {step}: {', '.join(bad)}")


def evaluate(model, batcher, cfg, ctx, optimizers=None):
    """Estimate loss over ``eval_iters`` batches. For Schedule-Free we switch to
    the averaged (eval) iterate first, then back (guide §4.5)."""
    sf = optimizers is not None and is_schedule_free(cfg)
    if sf:
        optimizers[0].eval()
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    with torch.no_grad():
        for i in range(cfg.eval_iters):
            x, y = batcher.batch()
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
    model.train()
    if sf:
        optimizers[0].train()
    val = losses.mean().item()
    require_finite(-1, val_loss=val)
    return val


def train(cfg, overfit: int = 0):
    set_seed(cfg.seed)

    if cfg.diffusion_mode != "none":
        from . import diffusion as D
        return D.train(
            cfg,
            init_ckpt=cfg.diffusion_init_ckpt,
            anneal_steps=cfg.diffusion_anneal_steps,
            complementary=cfg.diffusion_complementary,
            block_len=cfg.diffusion_block_len if cfg.diffusion_mode == "block" else 0,
        )

    device = pick_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    if device.startswith("cuda") and cfg.tf32:
        # free throughput on Ampere+: TF32 matmul/cudnn (guide §7.3)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    # cap the allocator so an over-budget step OOMs cleanly instead of spilling to
    # host RAM (the WDDM sysmem-fallback thrash, guide §7 / bench_gpu notes)
    if device.startswith("cuda") and 0 < getattr(cfg, "mem_fraction", 0) < 1.0:
        torch.cuda.set_per_process_memory_fraction(cfg.mem_fraction, 0)
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir, cfg)

    # ---- data (guide §3) ----
    data_dir, vocab_size, _ = get_dataset(cfg)
    if cfg.vocab_size == 0:           # char datasets set vocab dynamically
        cfg.vocab_size = vocab_size
    train_batcher = Batcher(data_dir, "train", cfg, device, overfit=overfit)
    val_batcher = Batcher(data_dir, "val", cfg, device)

    # ---- model (guide §2) ----
    model = build_model(cfg).to(device)
    log.banner(model)

    # torch.compile needs Triton (Inductor backend); it's absent on Windows, and
    # the failure is lazy (fires on the first forward, uncatchable here), so skip
    # compile unless Triton is actually importable. Also skip for non-attention/
    # MoE paths (recurrent loops / expert dispatch graph-break badly).
    import importlib.util
    has_triton = importlib.util.find_spec("triton") is not None
    if (cfg.compile and has_triton and device.startswith("cuda")
            and all(k == "attention" for k in parse_layer_mixers(cfg))
            and cfg.ffn != "moe"):
        model = torch.compile(model)
    elif cfg.compile and not has_triton:
        log.info("torch.compile requested but Triton not available -> running eager")
    elif cfg.compile and not all(k == "attention" for k in parse_layer_mixers(cfg)):
        log.info("torch.compile skipped: non-attention mixer in the stack")

    # ---- optimizer + schedule (guide §4, §5) ----
    optimizers = build_optimizers(model, cfg)
    schedule = make_schedule(cfg)
    sf = is_schedule_free(cfg)        # LR is internal; schedule is a no-op

    autocast = (torch.autocast(device_type="cuda", dtype=dtype)
                if device.startswith("cuda") and dtype != torch.float32
                else nullcontext())

    # ---- resume (guide §6.4) ----
    start_step = 0
    ckpt = out_dir / "ckpt.pt"
    if ckpt.exists() and os.environ.get("RESUME", "0") == "1":
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if state.get("optimizers"):          # light checkpoints omit these
            for opt, sd in zip(optimizers, state["optimizers"]):
                opt.load_state_dict(sd)
        else:
            log.info("checkpoint has no optimizer state; resuming weights only")
        start_step = state["step"]
        log.info(f"resumed from step {start_step}")
    elif cfg.init_ckpt:
        state = torch.load(cfg.init_ckpt, map_location=device, weights_only=False)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.load_state_dict(state["model"])
        log.info(f"initialized weights from {cfg.init_ckpt}")

    flops_per_tok = model.flops_per_token() if hasattr(model, "flops_per_token") \
        else model._orig_mod.flops_per_token()
    best_val = math.inf
    t0 = time.time()
    tokens_seen = 0
    lr_free = is_lr_free(cfg)            # Schedule-Free / Prodigy set their own LR
    is_sophia = cfg.optimizer == "sophia"
    model.train()
    if sf:
        optimizers[0].train()

    for step in range(start_step, cfg.max_steps):
        # ---- LR schedule (guide §5) ----
        lr = schedule(step)
        if not lr_free:
            apply_lr(optimizers, lr, cfg)

        # ---- curriculum (§3): grow context (seqlen) and/or the easy→hard data
        # frontier (difficulty, with difficulty-sorted data) over training ----
        cur_len = _curriculum_len(cfg, step)
        frontier = _curriculum_frontier(cfg, step)

        # ---- gradient accumulation (guide §4.1) ----
        # Host syncs (loss.item, cuda.synchronize) only happen on log steps so
        # the GPU can run ahead. clip_grad_norm_ stays every step (that's the
        # clip); we just avoid pulling scalars to CPU.
        log_now = (step % cfg.log_interval == 0)
        if device.startswith("cuda"):
            if log_now:
                torch.cuda.synchronize()
                t_step = time.time()
        else:
            t_step = time.time()
        for micro in range(cfg.grad_accum):
            x, y = train_batcher.batch(cur_len, frontier)
            with autocast:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            loss.backward()
        step_tokens = cfg.batch_size * cfg.grad_accum * cur_len

        # ---- gradient clipping (guide §5.4) ----
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # Sophia: refresh the diagonal-curvature estimate every k steps (§4.2)
        if is_sophia and step % cfg.sophia_hess_interval == 0:
            optimizers[0].update_hessian()

        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        tokens_seen += step_tokens

        loss_v = float(loss.detach()) * cfg.grad_accum
        gnorm = float(grad_norm)
        require_finite(step, loss=loss_v, gnorm=gnorm)

        # ---- logging (guide §6.1) ----
        if log_now:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                dt = time.time() - t_step
            else:
                dt = time.time() - t_step
            tok_s = step_tokens / max(dt, 1e-9)
            mfu = _mfu(flops_per_tok, step_tokens, dt, device)
            log.step(step, loss=loss_v, lr=lr,
                     grad_norm=gnorm, tok_s=tok_s, mfu=mfu,
                     tokens=tokens_seen)

        # ---- periodic eval + checkpoint (guide §6.1, §6.4) ----
        if step > 0 and step % cfg.eval_interval == 0:
            val = evaluate(model, val_batcher, cfg, autocast, optimizers)
            extra = {}
            if cfg.eval_train:
                extra["train_loss"] = evaluate(
                    model, train_batcher, cfg, autocast, optimizers)
            if getattr(schedule, "reactive", False):
                schedule.observe(val)          # ReduceLROnPlateau
            if cfg.tokenizer == "char":        # char models: bits-per-char (§3)
                extra["bpc"] = val / math.log(2)
            log.eval(step, val_loss=val,
                     val_ppl=math.exp(min(val, 20)), tokens=tokens_seen, **extra)
            if val < best_val:
                best_val = val
                _save(out_dir / "best.pt", model, optimizers, step, cfg, val, light=True)

        if step > 0 and step % cfg.ckpt_interval == 0:
            _save(ckpt, model, optimizers, step, cfg, best_val)   # full: resume

    # ---- final eval ----
    val = evaluate(model, val_batcher, cfg, autocast, optimizers)
    _save(out_dir / "final.pt", model, optimizers, cfg.max_steps, cfg, val, light=True)
    # ensure best.pt always exists (e.g. when eval_interval >= max_steps, no
    # periodic eval fired and best.pt was never written).
    if val <= best_val or not (out_dir / "best.pt").exists():
        _save(out_dir / "best.pt", model, optimizers, cfg.max_steps, cfg, val, light=True)
    best_val = min(best_val, val)
    elapsed_s = time.time() - t0
    log.done(best_val, human_time(elapsed_s), tokens_seen, elapsed_s=elapsed_s)
    return best_val


def _curriculum_len(cfg, step):
    """Sequence-length curriculum (§3): linearly grow context from
    ``curriculum_start_len`` to ``block_size`` over ``curriculum_frac`` of the
    run, then hold. Returns ``block_size`` unless the seqlen curriculum is on."""
    if cfg.curriculum != "seqlen":
        return cfg.block_size
    start = cfg.curriculum_start_len or max(8, cfg.block_size // 4)
    frac = max(1, int(cfg.curriculum_frac * cfg.max_steps))
    p = min(1.0, step / frac)
    cur = int(start + p * (cfg.block_size - start))
    return max(8, min(cfg.block_size, (cur // 8) * 8))   # multiple of 8


def _curriculum_frontier(cfg, step):
    """Difficulty curriculum (§3): grow the reachable-data window from a small
    fraction to the whole corpus over ``curriculum_frac`` of the run. With
    difficulty-sorted data (easy first, `prep_fineweb --sort_difficulty`) this
    presents easy→hard. Returns 1.0 unless the difficulty curriculum is on."""
    if cfg.curriculum != "difficulty":
        return 1.0
    start = 0.1                                  # begin on the easiest 10%
    frac = max(1, int(cfg.curriculum_frac * cfg.max_steps))
    return min(1.0, start + (1 - start) * min(1.0, step / frac))


def _mfu(flops_per_tok, tokens, dt, device):
    """Model-FLOPs utilization (guide §6.1, §7). Peak FLOPs is hardware-specific;
    default is a 3070 Ti Laptop BF16 dense peak (~46 TFLOP/s); override with the
    PEAK_FLOPS env var for other cards. On CPU this is meaningless -> 0."""
    if not device.startswith("cuda"):
        return 0.0
    achieved = flops_per_tok * tokens / max(dt, 1e-9)
    peak = float(os.environ.get("PEAK_FLOPS", 46e12))
    return achieved / peak


def _save(path, model, optimizers, step, cfg, val, light=False):
    """Save a checkpoint. ``light=True`` (best.pt / final.pt) stores weights
    only — for inference; ``light=False`` (resume ckpt) also stores optimizer
    state. Optimizer state (Muon momentum + Adam m/v) roughly doubles the file,
    so the inference checkpoints stay small."""
    sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    blob = {"model": sd, "step": step, "cfg": cfg.to_dict(), "val_loss": val}
    if not light:
        blob["optimizers"] = [o.state_dict() for o in optimizers]
    torch.save(blob, path)


def main():
    p = argparse.ArgumentParser(description="nanolab trainer (modern small-LM guide)")
    add_config_args(p)
    p.add_argument("--overfit", type=int, default=0,
                   help="restrict train data to N tokens (overfitting demo, §6.3)")
    args = p.parse_args()
    cfg = config_from_args(args)
    train(cfg, overfit=args.overfit)


if __name__ == "__main__":
    main()
