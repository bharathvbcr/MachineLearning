"""
nanolab.bench_modes — wall-clock tok/s for the tri-mode diffusion samplers.

Loads a (diffusion-adapted or AR) checkpoint and times generation under each
decoding mode the package supports, so the cached vs. uncached speedups proven
exact in tests.py can be confirmed as *real* wall-clock wins on the GPU:

  ar             — greedy autoregressive (token-by-token), reference baseline
  diffusion      — full bidirectional parallel denoise
  block          — semi-AR block diffusion          (+ KV cache)
  selfspec       — diffusion draft + AR verify       (+ KV cache, lossless)

Quality needs the diffusion conversion (``diffusion.py train``); for raw timing
any checkpoint of the right shape works. Token ids are synthetic (no tokenizer
dependency) — we measure throughput, not text.

    python -m nanolab.bench_modes --ckpt nanolab/out/run128m_fineweb_2k/best.pt \
        --gen_len 256 --block_len 32 --steps 8
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from .config import build_config
from .model import build_model
from . import diffusion as D


class _Enc:
    """Tokenizer stand-in: a fixed prompt of ``P`` ids, identity decode. Lets the
    samplers run without tiktoken since we only care about wall-clock."""

    def __init__(self, P):
        self.P = P

    def encode_ordinary(self, _s):
        return list(range(1, self.P + 1))

    def decode(self, ids):
        return list(ids)


def _load(ckpt, device):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = build_config(None, sd["cfg"])
    cfg.device = device
    cfg.compile = False
    m = build_model(cfg).to(device).eval()
    m.load_state_dict(sd["model"])
    return m, cfg


@torch.no_grad()
def _ar_greedy(m, P, gen_len, device, autocast):
    """Reference AR baseline: greedy, token-by-token, NO KV cache (each step
    re-encodes the prefix — same primitive the diffusion reference samplers use,
    so the comparison is apples-to-apples)."""
    m.set_block_attention(0)
    m.set_causal(True)
    ids = list(range(1, P + 1))
    for _ in range(gen_len):
        with autocast:
            h = m.forward_hidden(torch.tensor([ids], device=device))
            lg = F.linear(h, m.lm_head.weight)[0].float()
        lg[:, D.MASK_ID] = -float("inf")
        ids.append(int(lg[-1].argmax()))
    return ids


@torch.no_grad()
def measure_acceptance(m, gen_len, block_len, draft_steps, prompt_len, device, autocast):
    """Average self-speculation acceptance length = generated tokens / verify
    forwards. This is the number that decides whether self-spec is a win: each
    verify forward commits (accepted prefix + 1) tokens, so acceptance ~= 1 means
    no speedup (an un-adapted draft), while acceptance >> 1 means the diffusion
    draft is good and many tokens clear per AR verify. Counts verify forwards by
    wrapping the model's windowed forward (the causal=True calls are the verifies)."""
    verifies = {"n": 0}
    orig = m.forward_hidden_window

    def wrapped(x, abs_start, caches, commit, causal=False):
        if causal and not commit:           # the verify forward (not prime/commit)
            verifies["n"] += 1
        return orig(x, abs_start, caches, commit, causal)

    m.forward_hidden_window = wrapped
    try:
        enc = _Enc(prompt_len)
        D.sample_selfspec_cached(m, enc, "", gen_len, block_len, draft_steps,
                                 device, autocast, temperature=0.0)
    finally:
        m.forward_hidden_window = orig
    return gen_len / max(1, verifies["n"])


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time(fn, device, warmup=1, reps=2):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / reps


def run(ckpt, gen_len, block_len, steps, prompt_len, device, dtype, modes=None, accept=False):
    m, cfg = _load(ckpt, device)
    enc = _Enc(prompt_len)
    autocast = (torch.autocast("cuda", dtype=dtype)
                if device.startswith("cuda") and dtype != torch.float32
                else _nullctx())
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"model {n:.1f}M | {cfg.n_layer}L d{cfg.d_model} | gen_len={gen_len} "
          f"block_len={block_len} steps={steps} prompt={prompt_len} | {device}/{dtype}")
    print("-" * 64)

    jobs = [
        ("ar", "ar (token-by-token)",
         lambda: _ar_greedy(m, prompt_len, gen_len, device, autocast)),
        ("diffusion", "diffusion (full)",
         lambda: D.sample(m, enc, "", gen_len, steps, device, autocast, temperature=0.0)),
        ("block", "block  uncached", lambda: D.sample_blockwise(
            m, enc, "", gen_len, block_len, steps, device, autocast, temperature=0.0)),
        ("block", "block  cached", lambda: D.sample_blockwise_cached(
            m, enc, "", gen_len, block_len, steps, device, autocast, temperature=0.0)),
        ("selfspec", "selfspec uncached", lambda: D.sample_selfspec(
            m, enc, "", gen_len, block_len, steps, device, autocast, temperature=0.0)),
        ("selfspec", "selfspec cached", lambda: D.sample_selfspec_cached(
            m, enc, "", gen_len, block_len, steps, device, autocast, temperature=0.0)),
    ]
    if modes:
        keep = set(modes.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    results = {}
    for _key, name, fn in jobs:
        dt = _time(fn, device)
        toks = gen_len / dt
        results[name] = toks
        print(f"  {name:<22} {dt*1e3:8.1f} ms   {toks:8.1f} tok/s")
    print("-" * 64)
    base = results.get("ar (token-by-token)", None)
    if base:
        for k in ("block  cached", "selfspec cached", "diffusion (full)"):
            if k in results:
                print(f"  {k:<22} speedup vs AR: {results[k]/base:5.2f}x")
    if accept:
        a = measure_acceptance(m, gen_len, block_len, steps, prompt_len, device, autocast)
        print(f"  self-spec acceptance:  {a:5.2f} tokens/verify "
              f"(~1 = un-adapted draft, >1 = real speedup)")
        results["acceptance"] = a
    return results


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    p = argparse.ArgumentParser(description="tri-mode sampler wall-clock benchmark")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gen_len", type=int, default=256)
    p.add_argument("--block_len", type=int, default=32)
    p.add_argument("--steps", type=int, default=8,
                   help="denoise rounds (per block for block/selfspec)")
    p.add_argument("--prompt_len", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--modes", default="",
                   help="comma list to restrict: ar,diffusion,block,selfspec (default all)")
    p.add_argument("--accept", action="store_true",
                   help="also report self-speculation acceptance (tokens/verify)")
    args = p.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    run(args.ckpt, args.gen_len, args.block_len, args.steps, args.prompt_len,
        args.device, dtype, args.modes or None, args.accept)


if __name__ == "__main__":
    main()
