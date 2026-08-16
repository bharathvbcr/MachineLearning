"""
nanolab.diffusion — Phase 3: convert the AR model to a diffusion LM (guide §9).

This is the optional Phase 3 of the guide: take the *same* 128M (here 30M)
decoder you already trained autoregressively and adapt it into a **masked /
absorbing-state diffusion** language model, DiffuGPT/DiffuLLaMA-style. You only
learn the diffusion part — the base is the AR checkpoint you understand.

The recipe (guide §9):
  1. Attention-mask annealing — gradually switch causal -> bidirectional
     (``--anneal_steps``). A diffusion model denoises in parallel with
     bidirectional attention; annealing from the AR model's causal mask is what
     makes the adaptation cheap.
  2. Masked / diffusion objective — each step pick a noise level t~U(0,1), mask
     that fraction of tokens (absorbing [MASK] state), predict the originals,
     loss only on masked positions, reweighted by 1/t (the MDLM/LLaDA NLL bound).
  3. LLaDA-2.0 add-ons — ``--complementary`` masking (two opposite masked views
     per sequence, so every position is supervised once) and confidence-based
     parallel decoding at sampling time.

  4. Block diffusion / tri-mode (``--block_len``, Nemotron-Labs-Diffusion-style).
     Train with *block-causal* attention (causal across blocks, bidirectional
     within) instead of annealing to fully bidirectional. One weight set then
     serves three decoding modes, all selectable at ``sample`` time via ``--mode``
     (the attention pattern is the only thing that changes — GPT.set_block_attention
     / set_causal toggle it, no separate checkpoints):
       * ``diffusion`` — full parallel denoise (block_len >= T).
       * ``block``     — semi-AR: decode block by block, parallel within a block.
                         Throughput path; ``--cached`` keeps a KV cache of finalized
                         blocks (never recomputed; identical output, ~10x fewer
                         positions processed) — sample_blockwise_cached.
       * ``selfspec``  — diffusion DRAFTS a block, a single causal AR forward
                         VERIFIES it; accept the longest greedy-matching prefix +1.
                         Output is identical to greedy AR (lossless), just faster.
                         ``--cached`` keeps a persistent causal committed cache
                         (textbook speculative decoding; ~7x fewer positions,
                         same output) — sample_selfspec_cached.

    # adapt the trained AR checkpoint into a diffusion model (full diffusion)
    python -m nanolab.diffusion train --preset phase0 --mixer attention \
        --init nanolab/out/phase0_tinystories/best.pt --max_steps 2000

    # OR block diffusion (semi-AR), the basis for block/selfspec sampling.
    # NB: use --optimizer adamw for the conversion. Muon's Newton-Schulz step
    # natively crashes (CUDA abort, no Python traceback) on the large early grads
    # of the cold-start adaptation; AdamW is stable. (Muon is fine for the AR
    # pre-train — only the diffusion cold-start trips it.)
    python -m nanolab.diffusion train --preset phase0 --mixer attention --optimizer adamw \
        --init nanolab/out/phase0_tinystories/best.pt --block_len 32 --max_steps 2000

    # generate by iterative denoising (full / semi-AR block / self-speculation)
    python -m nanolab.diffusion sample --preset phase0 \
        --ckpt nanolab/out/diffusion_phase0/final.pt --prompt "Once upon a time,"
    python -m nanolab.diffusion sample --preset phase0 --mode block --block_len 32 ...
    python -m nanolab.diffusion sample --preset phase0 --mode selfspec --block_len 32 ...
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import add_config_args, config_from_args
from .data import Batcher, get_dataset
from .model import build_model
from .optim import build_optimizers
from .schedules import apply_lr, make_schedule
from .utils import Logger, human_time, pick_device, resolve_dtype, set_seed

# Absorbing [MASK] state. GPT-2 BPE uses ids 0..50256; vocab is padded to 50304,
# so 50257 is a free, dedicated mask id that needs no tokenizer change.
MASK_ID = 50257
EPS = 1e-3


def mask_tokens(x, t, mask_id=MASK_ID):
    """Absorbing-state forward process: independently replace each token with
    [MASK] with probability ``t`` (per-sequence noise level). Returns the noised
    input and the boolean mask of corrupted positions."""
    B, T = x.shape
    probs = t.view(B, 1).expand(B, T)
    m = torch.rand(B, T, device=x.device) < probs
    xm = torch.where(m, torch.full_like(x, mask_id), x)
    return xm, m


def diffusion_loss(model, x_noised, x_clean, t, mask, lm_head_weight):
    """Masked cross-entropy at the corrupted positions, reweighted by 1/t — the
    MDLM/LLaDA per-token NLL bound (guide §9). ``t`` is per-sequence.

    The model sees ``x_noised`` (with [MASK] at corrupted positions) and must
    predict the ORIGINAL ``x_clean`` there. Targets MUST be the clean tokens, not
    the masked input — otherwise the model trivially learns to echo [MASK] and
    the loss collapses to 0."""
    hidden = model.forward_hidden(x_noised)           # bidirectional (B,T,d)
    logits = F.linear(hidden, lm_head_weight)         # all positions
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)),
                         x_clean.view(-1), reduction="none").view(x_clean.shape)
    ce = ce * mask                                    # supervise only masked positions
    per_seq = ce.sum(dim=1) / x_clean.shape[1]        # mean over length
    weight = 1.0 / t.clamp_min(EPS)                   # 1/t reweighting
    return (weight * per_seq).mean()


def train(cfg, init_ckpt, anneal_steps, complementary, block_len=0, mask_id=MASK_ID):
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    if device.startswith("cuda") and cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Cap VRAM so an over-budget config raises a clean OutOfMemoryError instead of
    # silently spilling to host RAM over PCIe (~25x slower — looks like a hang).
    # The masked objective materializes a full (B,T,vocab) logits tensor (no
    # fused-CE on this path), so this guard matters more here than in train.py.
    if device.startswith("cuda") and getattr(cfg, "mem_fraction", 0):
        torch.cuda.set_per_process_memory_fraction(cfg.mem_fraction, 0)
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir, cfg)

    data_dir, vocab, _ = get_dataset(cfg)
    if cfg.vocab_size == 0:
        cfg.vocab_size = vocab
    train_b = Batcher(data_dir, "train", cfg, device)
    val_b = Batcher(data_dir, "val", cfg, device)

    model = build_model(cfg).to(device)
    if init_ckpt:
        sd = torch.load(init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        log.info(f"initialized from AR checkpoint {init_ckpt}")
    log.banner(model)

    optimizers = build_optimizers(model, cfg)
    schedule = make_schedule(cfg)
    autocast = (torch.autocast("cuda", dtype=dtype)
                if device.startswith("cuda") and dtype != torch.float32
                else _nullctx())
    lm_w = model.lm_head.weight

    # Block diffusion (semi-AR, Nemotron-Labs-Diffusion-style): instead of
    # annealing to FULL bidirectional attention, attend block-causally — causal
    # across blocks of `block_len`, bidirectional within. This keeps an AR-style
    # left-to-right context (so the model can decode block by block and reuse the
    # earlier blocks as a KV cache) while denoising each block in parallel. It is
    # also the prerequisite for self-speculation sampling. block_len==0 -> the
    # original full-diffusion conversion (anneal causal -> bidirectional).
    if block_len > 0:
        model.set_block_attention(block_len)
        log.info(f"block diffusion: block-causal attention, block_len={block_len}")

    t0 = time.time()
    model.train()
    best_val = math.inf
    for step in range(cfg.max_steps):
        lr = schedule(step)
        apply_lr(optimizers, lr, cfg)

        if block_len == 0:
            # attention-mask annealing: p(causal) decays 1 -> 0 over anneal_steps.
            p_causal = max(0.0, 1.0 - step / max(1, anneal_steps))
            model.set_causal(torch.rand(1).item() < p_causal)

        for _ in range(cfg.grad_accum):
            x, _ = train_b.batch()
            tt = torch.rand(x.shape[0], device=device) * (1 - EPS) + EPS
            with autocast:
                xm, m = mask_tokens(x, tt, mask_id)
                loss = diffusion_loss(model, xm, x, tt, m, lm_w)
                if complementary:                     # LLaDA-2.0 complementary view
                    xc = torch.where(~m, torch.full_like(x, mask_id), x)
                    mc = ~m
                    # complement masks the (1-t) fraction left clean by the first view
                    tc = (1 - tt).clamp_min(EPS)
                    loss = 0.5 * (loss + diffusion_loss(model, xc, x, tc, mc, lm_w))
                loss = loss / cfg.grad_accum
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % cfg.log_interval == 0:
            log.step(step, loss=loss.item() * cfg.grad_accum, lr=lr,
                     grad_norm=float(gn), tok_s=0.0, mfu=0.0)
        if step > 0 and step % cfg.eval_interval == 0:
            val = evaluate(model, val_b, cfg, autocast, lm_w, mask_id)
            log.eval(step, train_loss=val, val_loss=val, val_ppl=math.exp(min(val, 20)))
            if val < best_val:
                best_val = val
                _save(out_dir / "best.pt", model, step, cfg, val)

    model.set_causal(False)          # final model is fully bidirectional
    val = evaluate(model, val_b, cfg, autocast, lm_w, mask_id)
    _save(out_dir / "final.pt", model, cfg.max_steps, cfg, min(best_val, val))
    log.done(min(best_val, val), human_time(time.time() - t0), 0)
    return min(best_val, val)


@torch.no_grad()
def evaluate(model, batcher, cfg, autocast, lm_w, mask_id, iters=40):
    """Diffusion ELBO-style val loss at a fixed mid-noise level for comparability."""
    model.eval()
    was_causal = model.blocks[0].mixer.causal if hasattr(model.blocks[0].mixer, "causal") else None
    model.set_causal(False)
    device = next(model.parameters()).device
    losses = []
    for _ in range(iters):
        x, _ = batcher.batch()
        tt = torch.full((x.shape[0],), 0.5, device=device)
        with autocast:
            xm, m = mask_tokens(x, tt, mask_id)
            losses.append(diffusion_loss(model, xm, x, tt, m, lm_w).item())
    model.train()
    if was_causal is not None:
        model.set_causal(was_causal)
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, enc, prompt, gen_len, steps, device, autocast, mask_id=MASK_ID,
           temperature=0.7):
    """Confidence-based parallel denoising (LLaDA-style, guide §9): start from an
    all-[MASK] continuation and, over ``steps`` rounds, reveal a growing set of
    positions until none remain.

    Which positions to reveal is chosen by model confidence (max prob); the token
    placed there is *sampled* from the per-position distribution at ``temperature``
    (>0) — so temperature actually affects diversity. temperature==0 => greedy."""
    model.eval()
    model.set_causal(False)
    prompt_ids = enc.encode_ordinary(prompt)
    P = len(prompt_ids)
    x = torch.full((1, P + gen_len), mask_id, device=device, dtype=torch.long)
    x[0, :P] = torch.tensor(prompt_ids, device=device)

    for s in range(steps):
        with autocast:
            hidden = model.forward_hidden(x)
            logits = F.linear(hidden, model.lm_head.weight)[0].float()   # (T, V)
        logits[:, mask_id] = -float("inf")             # never predict [MASK]
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)   # sampled token per pos
            conf = probs.max(dim=-1).values                 # confidence for ranking
        else:
            conf, tok = F.softmax(logits, dim=-1).max(dim=-1)
        masked = (x[0] == mask_id)
        if not masked.any():
            break
        # reveal a growing fraction; the last round fills everything left
        n_left = int(masked.sum().item())
        reveal = max(1, math.ceil(n_left / (steps - s)))
        conf_masked = conf.masked_fill(~masked, -1.0)
        idx = torch.topk(conf_masked, reveal).indices
        x[0, idx] = tok[idx]
    return enc.decode(x[0].tolist())


@torch.no_grad()
def _denoise_region(model, x, lo, hi, steps, autocast, mask_id, temperature):
    """Confidence-based parallel denoising of ``x[0, lo:hi]`` in place (the inner
    loop shared by the block-wise and self-speculation samplers). The attention
    mode (block-causal / bidirectional) is configured by the caller. Reveals the
    highest-confidence masked positions inside [lo,hi) over ``steps`` rounds."""
    for s in range(steps):
        masked = (x[0, lo:hi] == mask_id)
        if not masked.any():
            break
        with autocast:
            hidden = model.forward_hidden(x)
            logits = F.linear(hidden, model.lm_head.weight)[0].float()
        logits[:, mask_id] = -float("inf")
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.max(dim=-1).values
        else:
            conf, tok = F.softmax(logits, dim=-1).max(dim=-1)
        region = torch.zeros_like(conf, dtype=torch.bool)
        region[lo:hi] = masked                          # only this region's masked slots
        n_left = int(masked.sum().item())
        reveal = min(max(1, math.ceil(n_left / (steps - s))), n_left)
        conf_r = conf.masked_fill(~region, -1.0)
        idx = torch.topk(conf_r, reveal).indices
        x[0, idx] = tok[idx]
    return x


@torch.no_grad()
def sample_blockwise(model, enc, prompt, gen_len, block_len, steps_per_block, device,
                     autocast, mask_id=MASK_ID, temperature=0.7):
    """Semi-autoregressive block diffusion (guide §9 / Nemotron-Labs-Diffusion):
    decode the continuation block by block, left to right. Within each block tokens
    are denoised bidirectionally in parallel; across blocks attention is causal, so
    a finalized block becomes fixed left-context for the next one. This is the
    throughput path — `block_len` tokens emerge per `steps_per_block` forwards
    instead of one token per forward. (A production impl also caches the finalized
    blocks' KV; this reference recomputes the trunk each round, so it demonstrates
    the *pattern*, not the wall-clock win.)"""
    model.eval()
    model.set_block_attention(block_len)
    prompt_ids = enc.encode_ordinary(prompt)
    P = len(prompt_ids)
    T = P + gen_len
    x = torch.full((1, T), mask_id, device=device, dtype=torch.long)
    x[0, :P] = torch.tensor(prompt_ids, device=device)
    start = P
    while start < T:
        end = min(T, ((start // block_len) + 1) * block_len)   # end of block holding `start`
        _denoise_region(model, x, start, end, steps_per_block, autocast, mask_id, temperature)
        start = end
    return enc.decode(x[0, P:].tolist())


@torch.no_grad()
def sample_blockwise_cached(model, enc, prompt, gen_len, block_len, steps_per_block,
                           device, autocast, mask_id=MASK_ID, temperature=0.7):
    """KV-cached semi-AR block diffusion — same outputs as ``sample_blockwise`` but
    finalized blocks are cached and never recomputed (the real throughput win). The
    sequence is processed strictly one absolute block at a time; each block's window
    forward attends to the cached K,V of all earlier blocks (no full-trunk recompute)
    and, once denoised, commits its own K,V to the cache.

    Correctness rests on the cache invariant (mixers.forward_cached): block-causal
    attention == unmasked attention over [cache | this block], so this reproduces
    the masked full-sequence sampler exactly (see the regression test)."""
    model.eval()
    caches = [dict() for _ in range(len(model.blocks))]
    prompt_ids = enc.encode_ordinary(prompt)
    P = len(prompt_ids)
    T = P + gen_len
    full = torch.full((1, T), mask_id, device=device, dtype=torch.long)
    full[0, :P] = torch.tensor(prompt_ids, device=device)

    def _window_logits(lo, hi, commit):
        hidden = model.forward_hidden_window(full[:, lo:hi], lo, caches, commit)
        return F.linear(hidden, model.lm_head.weight)[0].float()      # (T_win, V)

    pos = 0
    while pos < T:
        lo = (pos // block_len) * block_len           # absolute block boundaries
        hi = min(T, lo + block_len)
        if not (full[0, lo:hi] == mask_id).any():     # clean (prompt) block -> prime cache
            _window_logits(lo, hi, commit=True)
            pos = hi
            continue
        rounds = max(1, steps_per_block)
        for s in range(rounds):
            masked = (full[0, lo:hi] == mask_id)
            if not masked.any():
                break
            logits = _window_logits(lo, hi, commit=False)
            logits[:, mask_id] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                tok = torch.multinomial(probs, 1).squeeze(-1)
                conf = probs.max(dim=-1).values
            else:
                conf, tok = F.softmax(logits, dim=-1).max(dim=-1)
            n_left = int(masked.sum().item())
            reveal = min(max(1, math.ceil(n_left / (rounds - s))), n_left)
            conf_r = conf.masked_fill(~masked, -1.0)   # rank only masked, in-window idx
            idx = torch.topk(conf_r, reveal).indices
            full[0, lo + idx] = tok[idx]
        _window_logits(lo, hi, commit=True)            # commit finalized block to cache
        pos = hi
    return enc.decode(full[0, P:].tolist())


@torch.no_grad()
def sample_selfspec(model, enc, prompt, gen_len, block_len, draft_steps, device,
                    autocast, mask_id=MASK_ID, temperature=0.7):
    """Self-speculation (Nemotron-Labs-Diffusion's third mode): diffusion DRAFTS a
    block of tokens in parallel, then a single CAUSAL (AR) forward VERIFIES them and
    accepts the longest prefix that matches greedy AR decoding — plus one bonus/
    correction token. The committed text is identical to pure greedy AR (lossless),
    but many tokens are confirmed per AR forward when the draft is good. One weight
    set plays both roles: block-causal attention to draft, causal attention to
    verify."""
    model.eval()
    prompt_ids = enc.encode_ordinary(prompt)
    P = len(prompt_ids)
    committed = list(prompt_ids)
    while len(committed) - P < gen_len:
        ctx = len(committed)
        dbl = min(block_len, gen_len - (len(committed) - P))

        # --- draft: block-causal diffusion fills the next `dbl` masked slots ---
        model.set_block_attention(block_len)
        x = torch.full((1, ctx + dbl), mask_id, device=device, dtype=torch.long)
        x[0, :ctx] = torch.tensor(committed, device=device)
        _denoise_region(model, x, ctx, ctx + dbl, draft_steps, autocast, mask_id, temperature)
        draft = x[0, ctx:ctx + dbl]

        # --- verify: one causal AR forward over [committed | draft] ---
        model.set_block_attention(0)
        model.set_causal(True)
        with autocast:
            hidden = model.forward_hidden(x)               # x already = [committed | draft]
            logits = F.linear(hidden, model.lm_head.weight)[0].float()
        logits[:, mask_id] = -float("inf")
        ar_pred = logits.argmax(dim=-1)                    # ar_pred[p] => greedy token at p+1

        # accept the longest prefix where the draft matches greedy AR
        accept = 0
        for j in range(dbl):
            if int(draft[j]) == int(ar_pred[ctx - 1 + j]):
                accept += 1
            else:
                break
        committed.extend(int(t) for t in draft[:accept])
        # one guaranteed token: AR's correction at the first mismatch, or the
        # next-token continuation when the whole draft was accepted.
        committed.append(int(ar_pred[ctx - 1 + accept]))
    return enc.decode(committed[P:P + gen_len])


@torch.no_grad()
def sample_selfspec_cached(model, enc, prompt, gen_len, block_len, draft_steps, device,
                           autocast, mask_id=MASK_ID, temperature=0.7):
    """KV-cached self-speculation — same lossless output as ``sample_selfspec`` but
    the committed prefix lives in a persistent CAUSAL KV cache instead of being
    re-encoded on every draft and verify. This is textbook speculative decoding:
    keep the target (AR) cache + the next-token distribution, let diffusion draft a
    block against that cache, verify the block in one causal cached forward, accept
    the longest greedy-matching prefix, then commit accepted+bonus into the cache.

    Drafting reuses the cache with bidirectional-in-block attention (causal=False);
    verifying reuses it with causal-in-block attention (causal=True). Neither
    recomputes the committed prefix."""
    model.eval()
    caches = [dict() for _ in range(len(model.blocks))]      # persistent causal cache
    prompt_ids = enc.encode_ordinary(prompt)
    P = len(prompt_ids)
    committed = list(prompt_ids)
    lm_w = model.lm_head.weight

    def _logits(tokens, abs_start, commit, causal):
        hidden = model.forward_hidden_window(tokens, abs_start, caches, commit, causal)
        lg = F.linear(hidden, lm_w)[0].float()
        lg[:, mask_id] = -float("inf")
        return lg

    # prime the causal cache over the prompt; keep its last-token next-distribution.
    ptok = torch.tensor([prompt_ids], device=device)
    next_logits = _logits(ptok, 0, commit=True, causal=True)[-1]      # dist for position P
    n = P                                                            # == len(committed)

    while len(committed) - P < gen_len:
        dbl = min(block_len, gen_len - (len(committed) - P))

        # --- draft: diffusion fills `dbl` masked slots, bidirectional in-block ---
        draft = torch.full((1, dbl), mask_id, device=device, dtype=torch.long)
        for s in range(max(1, draft_steps)):
            masked = (draft[0] == mask_id)
            if not masked.any():
                break
            logits = _logits(draft, n, commit=False, causal=False)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                tok = torch.multinomial(probs, 1).squeeze(-1)
                conf = probs.max(dim=-1).values
            else:
                conf, tok = F.softmax(logits, dim=-1).max(dim=-1)
            n_left = int(masked.sum().item())
            reveal = min(max(1, math.ceil(n_left / (draft_steps - s))), n_left)
            conf_r = conf.masked_fill(~masked, -1.0)
            idx = torch.topk(conf_r, reveal).indices
            draft[0, idx] = tok[idx]

        # --- verify: one causal cached forward over the draft block ---
        vlog = _logits(draft, n, commit=False, causal=True)          # vlog[j] => dist at n+j+1
        # greedy AR distribution for position n+j: next_logits (j=0) else vlog[j-1]
        accept = 0
        for j in range(dbl):
            dist = next_logits if j == 0 else vlog[j - 1]
            if int(draft[0, j]) == int(dist.argmax()):
                accept += 1
            else:
                break
        bonus_dist = next_logits if accept == 0 else vlog[accept - 1]
        bonus = int(bonus_dist.argmax())

        # --- commit accepted draft + bonus into the causal cache (one fwd) ---
        commit_ids = [int(t) for t in draft[0, :accept]] + [bonus]
        ctok = torch.tensor([commit_ids], device=device)
        next_logits = _logits(ctok, n, commit=True, causal=True)[-1]  # dist for next position
        committed.extend(commit_ids)
        n += len(commit_ids)

    return enc.decode(committed[P:P + gen_len])


# ---------------------------------------------------------------------------
def _save(path, model, step, cfg, val):
    sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({"model": sd, "step": step, "cfg": cfg.to_dict(), "val_loss": val,
                "diffusion": True, "mask_id": MASK_ID}, path)


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    p = argparse.ArgumentParser(description="nanolab diffusion conversion (guide §9)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    add_config_args(pt)
    pt.add_argument("--init", default="", help="AR checkpoint to adapt from")
    pt.add_argument("--anneal_steps", type=int, default=400,
                    help="steps to anneal causal -> bidirectional (block_len=0 only)")
    pt.add_argument("--complementary", type=_b, default=True,
                    help="LLaDA-2.0 complementary masking (two opposite views)")
    pt.add_argument("--block_len", type=int, default=0,
                    help="0 = full diffusion (anneal to bidirectional); >0 = block "
                         "diffusion with block-causal attention of this block length "
                         "(semi-AR, enables block-wise + self-speculation sampling)")

    ps = sub.add_parser("sample")
    add_config_args(ps)
    ps.add_argument("--ckpt", required=True)
    ps.add_argument("--prompt", default="Once upon a time,")
    ps.add_argument("--gen_len", type=int, default=64)
    ps.add_argument("--steps", type=int, default=32,
                    help="denoise rounds (per block for mode=block/selfspec)")
    ps.add_argument("--mode", choices=["diffusion", "block", "selfspec"],
                    default="diffusion",
                    help="diffusion = full parallel denoise; block = semi-AR block "
                         "diffusion; selfspec = diffusion-draft + AR-verify (lossless)")
    ps.add_argument("--block_len", type=int, default=32,
                    help="block size for mode=block/selfspec")
    ps.add_argument("--cached", type=_b, default=True,
                    help="mode=block/selfspec: use the KV-cached sampler (the "
                         "committed prefix is not recomputed). Same outputs, faster.")
    ps.add_argument("--temperature", type=float, default=0.7)

    args = p.parse_args()
    cfg = config_from_args(args)

    if args.cmd == "train":
        # default the run name to diffusion_<preset>, but respect an explicit
        # --run_name (only present in args when the user passed it).
        if getattr(args, "run_name", None) is None:
            cfg.run_name = "diffusion_" + (args.preset or "run")
        train(cfg, args.init, args.anneal_steps, args.complementary, args.block_len)
    else:
        import tiktoken
        device = pick_device(cfg.device)
        dtype = resolve_dtype(cfg.dtype, device)
        if cfg.vocab_size == 0:
            cfg.vocab_size = 50304
        model = build_model(cfg).to(device)
        sd = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        enc = tiktoken.get_encoding("gpt2")
        autocast = (torch.autocast("cuda", dtype=dtype)
                    if device.startswith("cuda") and dtype != torch.float32 else _nullctx())
        if args.mode == "block":
            fn = sample_blockwise_cached if args.cached else sample_blockwise
            txt = fn(model, enc, args.prompt, args.gen_len, args.block_len,
                     args.steps, device, autocast, temperature=args.temperature)
            txt = args.prompt + txt
        elif args.mode == "selfspec":
            fn = sample_selfspec_cached if args.cached else sample_selfspec
            txt = fn(model, enc, args.prompt, args.gen_len, args.block_len,
                     args.steps, device, autocast, temperature=args.temperature)
            txt = args.prompt + " " + txt
        else:
            txt = sample(model, enc, args.prompt, args.gen_len, args.steps, device, autocast,
                         temperature=args.temperature)
        print("=" * 70)
        print(txt)
        print("=" * 70)


def _b(v):
    return str(v).lower() in ("1", "true", "yes", "y", "on")


if __name__ == "__main__":
    main()
