"""
nanolab.sft — Phase-2 supervised fine-tuning: teach the 128M base model to think
then answer, with prompt-masked loss.

    # 1) build data (torch-free, run first):
    python -m nanolab.sft_data --dataset gsm8k
    # 2) fine-tune from the base checkpoint:
    python -m nanolab.sft --base_run run128m_fineweb_2k --run sft128m_gsm8k --steps 800

Reuses the base model's architecture and checkpoint. The model's cross-entropy
already honours ``ignore_index=-1``, so we simply set the target to -1 on prompt
tokens — the model trains only on the completion (think span + JSON answer + eot).
The ``<think>`` / ``</think>`` / ``<|answer|>`` tokens live on the padded vocab
region, so no embedding resize is needed (see nanolab/special_tokens.py).

Output checkpoint (best.pt) + config.json are written in the same format the rest
of nanolab uses, so ``python -m nanolab.reason --run <sft_run>`` loads it directly.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import build_config
from .model import build_model
from .utils import pick_device, resolve_dtype, set_seed


class SFTBatcher:
    """Random-window sampler over the packed SFT stream. Targets are masked to -1
    wherever the per-token loss mask is 0 (prompt tokens), so cross-entropy skips
    them. Windows may straddle examples — the eot separators and the mask keep
    that harmless for a short fine-tune."""

    def __init__(self, data_dir: Path, split: str, block_size: int,
                 batch_size: int, device: str, seed: int):
        self.tokens = np.memmap(data_dir / f"{split}_tokens.bin", dtype=np.uint16, mode="r")
        self.loss = np.memmap(data_dir / f"{split}_loss.bin", dtype=np.uint8, mode="r")
        if len(self.tokens) != len(self.loss):
            raise SystemExit(f"{split}: tokens/loss length mismatch")
        # cap the window to the stream — STaR's early rounds collect few traces, so
        # a tiny split must not slice past its end (mismatched x/y lengths).
        self.bs = max(1, min(block_size, len(self.tokens) - 1))
        self.B = batch_size
        self.device = device
        self.gen = torch.Generator().manual_seed(seed + (0 if split == "train" else 1))

    def __len__(self):
        return len(self.tokens)

    def batch(self):
        n = max(1, len(self.tokens) - self.bs - 1)
        ix = torch.randint(n, (self.B,), generator=self.gen)
        x = torch.stack([torch.from_numpy(self.tokens[i:i + self.bs].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.tokens[i + 1:i + 1 + self.bs].astype(np.int64)) for i in ix])
        m = torch.stack([torch.from_numpy(self.loss[i + 1:i + 1 + self.bs].astype(np.int64)) for i in ix])
        y = torch.where(m.bool(), y, torch.full_like(y, -1))      # mask prompt tokens
        if self.device.startswith("cuda"):
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)


def _cosine_lr(step, total, base_lr, warmup, floor_frac=0.1):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


@torch.no_grad()
def _eval(model, batcher, ctx, iters):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = batcher.batch()
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def run_sft(base_run, run, out_dir, data_dir, steps, lr, batch_size,
            block_size, eval_interval, weight_decay, seed):
    base_dir = Path(out_dir) / base_run
    cfg = build_config(overrides=json.loads((base_dir / "config.json").read_text()))
    if cfg.tokenizer != "gpt2":
        raise SystemExit("SFT pipeline requires the gpt2 BPE tokenizer")
    if block_size:
        cfg.block_size = block_size
    cfg.batch_size = batch_size
    cfg.run_name = run
    set_seed(seed)

    device = pick_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    ctx = (torch.autocast(device_type="cuda", dtype=dtype)
           if device.startswith("cuda") and dtype != torch.float32 else nullcontext())

    model = build_model(cfg).to(device)
    state = torch.load(base_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sft] loaded base {base_run} ({n_params/1e6:.1f}M params) on {device}")

    data_dir = Path(data_dir)
    tb = SFTBatcher(data_dir, "train", cfg.block_size, batch_size, device, seed)
    vb = SFTBatcher(data_dir, "val", cfg.block_size, batch_size, device, seed)
    print(f"[sft] train_tokens={len(tb)} val_tokens={len(vb)} "
          f"block={cfg.block_size} batch={batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=weight_decay)
    warmup = max(2, min(steps // 10, 50))

    out_run = Path(out_dir) / run
    out_run.mkdir(parents=True, exist_ok=True)
    (out_run / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    can_eval = len(vb) >= 2
    best = math.inf
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, steps, lr, warmup)
        x, y = tb.batch()
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 20 == 0 or step == steps - 1:
            print(f"step {step:4d}/{steps}  loss {loss.item():.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  gnorm {float(gnorm):.2f}")
        if can_eval and eval_interval and (step % eval_interval == 0 and step > 0 or step == steps - 1):
            vloss = _eval(model, vb, ctx, iters=10)
            print(f"  [eval] step {step}  val_loss {vloss:.4f}  ppl {math.exp(min(vloss,20)):.2f}")
            if vloss < best:
                best = vloss
                _save(out_run / "best.pt", model, step, cfg, vloss)

    final = _eval(model, vb, ctx, iters=10) if can_eval else float(loss.item())
    _save(out_run / "final.pt", model, steps, cfg, final)
    if final <= best or not (out_run / "best.pt").exists():
        _save(out_run / "best.pt", model, steps, cfg, final)
    print(f"[sft] done in {time.time()-t0:.1f}s  best_val={min(best, final):.4f}  -> {out_run}")
    return min(best, final)


def _save(path, model, step, cfg, val):
    sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({"model": sd, "step": step, "cfg": cfg.to_dict(), "val_loss": val}, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run", default="run128m_fineweb_2k")
    p.add_argument("--run", default="sft128m_gsm8k")
    p.add_argument("--out_dir", default="nanolab/out")
    p.add_argument("--data_dir", default="nanolab/data/sft")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_size", type=int, default=0, help="0 -> keep base config")
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    run_sft(args.base_run, args.run, args.out_dir, args.data_dir, args.steps,
            args.lr, args.batch_size, args.block_size, args.eval_interval,
            args.weight_decay, args.seed)


if __name__ == "__main__":
    main()
