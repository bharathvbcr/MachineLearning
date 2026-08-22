"""
nanolab.utils — seeding, device/dtype, and the run logger (guide §6.1).

The logger writes a human-readable console line AND a JSONL metrics file
(``metrics.jsonl``) so curves can be plotted later. "Logging is not optional"
(guide §0). If ``wandb`` is installed and ``WANDB=1``, metrics also stream there.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(name: str, device: str):
    if not device.startswith("cuda"):
        return torch.float32          # bf16 autocast is a CUDA path here
    return {"bf16": torch.bfloat16, "fp16": torch.float16,
            "fp32": torch.float32}[name]


def format_count(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def human_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class Logger:
    def __init__(self, out_dir: Path, cfg):
        self.out_dir = Path(out_dir)
        self.cfg = cfg
        self.f = open(self.out_dir / "metrics.jsonl", "a", encoding="utf-8")
        (self.out_dir / "config.json").write_text(
            json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        self.wandb = None
        if os.environ.get("WANDB", "0") == "1":
            try:
                import wandb
                wandb.init(project="nanolab", name=cfg.run_name, config=cfg.to_dict())
                self.wandb = wandb
            except Exception as e:
                print(f"[wandb disabled: {e}]")

    def _emit(self, record):
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()
        if self.wandb:
            self.wandb.log(record)

    def banner(self, model):
        n = model.num_params(non_embedding=False)
        nemb = model.num_params(non_embedding=True)
        print("=" * 70)
        print(f"  run        : {self.cfg.run_name}")
        print(f"  mixer      : {self.cfg.mixer}    optimizer: {self.cfg.optimizer}"
              f"    schedule: {self.cfg.schedule}")
        if getattr(self.cfg, "layer_mixers", ""):
            print(f"  layers     : {self.cfg.layer_mixers}")
        print(f"  params     : {format_count(n)} total ({format_count(nemb)} non-embed)")
        print(f"  arch       : L{self.cfg.n_layer} d{self.cfg.d_model} "
              f"h{self.cfg.n_head}/{self.cfg.n_kv_head}kv  ctx{self.cfg.block_size}  "
              f"ffn={self.cfg.ffn} qk_norm={self.cfg.qk_norm}")
        print(f"  tokens/step: {format_count(self.cfg.tokens_per_step)}  "
              f"(bs{self.cfg.batch_size} x ga{self.cfg.grad_accum} x ctx{self.cfg.block_size})")
        print("=" * 70)
        self._emit({"event": "start", "params": n, "params_non_embed": nemb})

    def info(self, msg):
        print(f"  [info] {msg}")

    def step(self, step, **kw):
        rec = {"event": "train", "step": step, **kw}
        self._emit(rec)
        print(f"  step {step:>6} | loss {kw['loss']:.4f} | lr {kw['lr']:.2e} | "
              f"gnorm {kw['grad_norm']:.2f} | {format_count(kw['tok_s'])} tok/s | "
              f"mfu {kw['mfu']*100:.1f}%")

    def eval(self, step, **kw):
        rec = {"event": "eval", "step": step, **kw}
        self._emit(rec)
        bpc = f"  bpc {kw['bpc']:.3f}" if "bpc" in kw else ""
        train_bit = (f"train {kw['train_loss']:.4f}  " if "train_loss" in kw else "")
        print(f"  -------- eval @ {step}: {train_bit}"
              f"val {kw['val_loss']:.4f}  ppl {kw['val_ppl']:.2f}{bpc} --------")

    def done(self, best_val, elapsed, tokens, elapsed_s=None):
        # elapsed is a human string for the console; elapsed_s is the machine
        # readable wall clock.  Emitting only the former is why no suite before
        # 2026-08-22 has a recoverable run time.
        record = {"event": "done", "best_val": best_val, "tokens": tokens}
        if elapsed_s is not None:
            record["elapsed_s"] = float(elapsed_s)
            record["mean_tok_s"] = float(tokens) / max(float(elapsed_s), 1e-9)
        self._emit(record)
        print("=" * 70)
        print(f"  DONE  best_val_loss={best_val:.4f}  "
              f"tokens={format_count(tokens)}  time={elapsed}")
        print("=" * 70)
        self.f.close()
