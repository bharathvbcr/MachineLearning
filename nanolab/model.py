"""
nanolab.model — a ~128M decoder-only LM with the 2025-2026 modern stack.

Maps directly onto the guide's "modern stack vs GPT-2 defaults" table (§2):
  RoPE · RMSNorm (pre-norm) · QK-Norm · SwiGLU · tied embeddings ·
  zero-init output projections · optional GQA · μP-ready · pluggable mixer.

The block is mixer-agnostic (guide §2.5): attention / mingru / mamba2 / gdn all
slot into the same residual stream, so the only thing that changes across an
A/B is ``cfg.mixer``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mixers import RMSNorm, build_mixer, build_rope_cache
from .config import parse_layer_mixers


def _norm(cfg, dim):
    return RMSNorm(dim) if cfg.norm == "rmsnorm" else nn.LayerNorm(dim)


class SwiGLU(nn.Module):
    """Gated FFN: down(SiLU(gate(x)) * up(x)). Hidden ~= 2/3*4*d so the param
    count matches a plain 4x GELU MLP (guide §2.1)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        from .config import _swiglu_hidden
        h = _swiglu_hidden(d)
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ReLU2MLP(nn.Module):
    """ReLU² FFN — the modded-nanoGPT speedrun activation (guide §2.3)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


class GELUMLP(nn.Module):
    """GPT-2 baseline FFN (guide §2 teaching baseline)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MoE(nn.Module):
    """Sparse mixture-of-experts FFN (guide §2.1): ``moe_experts`` SwiGLU experts,
    ``moe_top_k`` active per token — big capacity at low active compute. Adds a
    Switch-Transformer load-balancing auxiliary loss (exposed via ``self.aux``)
    so the router doesn't collapse onto a few experts."""

    def __init__(self, cfg):
        super().__init__()
        self.n_exp = cfg.moe_experts
        self.k = cfg.moe_top_k
        self.gate = nn.Linear(cfg.d_model, self.n_exp, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg) for _ in range(self.n_exp)])
        self.aux = None                 # load-balancing loss, set each forward

    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(-1, d)                                # (N, d)
        probs = F.softmax(self.gate(xf), dim=-1)            # (N, E)
        weights, idx = torch.topk(probs, self.k, dim=-1)    # (N, k)
        weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        for e in range(self.n_exp):
            sel_mask, slot = (idx == e).max(dim=-1)          # token routed to e?
            sel = sel_mask.nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            w = weights[sel, slot[sel]].unsqueeze(-1)
            out[sel] += w * self.experts[e](xf[sel])
        # Switch load-balancing aux loss: E * sum_e (frac_to_e * mean_prob_e).
        # frac (token counts) is a constant; the gradient flows through the mean
        # router probability, pushing the gate toward uniform expert usage.
        frac = torch.zeros(self.n_exp, device=x.device, dtype=probs.dtype)
        frac.scatter_add_(0, idx[:, 0], torch.ones_like(idx[:, 0], dtype=probs.dtype))
        frac = frac / xf.shape[0]
        self.aux = self.n_exp * (frac * probs.mean(dim=0)).sum()
        return out.view(B, T, d)


def build_ffn(cfg):
    return {"swiglu": SwiGLU, "relu2": ReLU2MLP, "gelu": GELUMLP,
            "moe": MoE}[cfg.ffn](cfg)


class FusedLinearCrossEntropy(torch.autograd.Function):
    """Memory-efficient ``cross_entropy(x @ W.T, target)`` (guide §7.1).

    The full logits tensor is ``(N, vocab)`` — for B·T=8192, vocab=50304 that is
    ~1.6 GB in fp32 and the ``cross_entropy`` upcast doubles it. We never hold it
    all at once: tile over tokens, and compute the input/weight gradients *during
    the forward pass* (one matmul per chunk, recomputing nothing in backward), so
    peak logit memory is ``chunk·vocab`` instead of ``N·vocab``. That freed VRAM
    buys a much larger batch — i.e. more work per kernel launch, higher MFU.

    This is the pure-PyTorch equivalent of LinkedIn's Liger fused linear CE; it
    needs no Triton/CUDA toolchain, so it runs on Windows/Ampere as-is.
    """

    @staticmethod
    def forward(ctx, x, weight, target, n_chunks, ignore_index):
        N, D = x.shape
        x = x.contiguous()
        n_valid = (target != ignore_index).sum().clamp(min=1).float()
        total = x.new_zeros((), dtype=torch.float32)
        grad_x = torch.empty_like(x, dtype=torch.float32)
        grad_w = torch.zeros_like(weight, dtype=torch.float32)
        chunk = (N + n_chunks - 1) // n_chunks
        wf = weight.float()
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            xc = x[i:j].float()                       # (Nc, D)
            tc = target[i:j]                          # (Nc,)
            logits = xc @ wf.T                        # (Nc, V) fp32 — the only big tensor
            logp = torch.log_softmax(logits, dim=-1)
            valid = (tc != ignore_index)
            tc_safe = tc.clamp(min=0)
            nll = -logp.gather(1, tc_safe[:, None]).squeeze(1)
            total += torch.where(valid, nll, torch.zeros_like(nll)).sum()
            # dL/dlogits = softmax - onehot, masked, averaged over valid tokens
            g = logp.exp()
            g.scatter_add_(1, tc_safe[:, None], -valid.float()[:, None])
            g.mul_(valid.float()[:, None]).div_(n_valid)
            grad_x[i:j] = (g @ wf).to(grad_x.dtype)
            grad_w.addmm_(g.T, xc)
        ctx.save_for_backward(grad_x.to(x.dtype), grad_w.to(weight.dtype))
        return total / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        grad_x, grad_w = ctx.saved_tensors
        return grad_x * grad_out, grad_w * grad_out, None, None, None


def fused_linear_cross_entropy(x, weight, target, n_chunks=8, ignore_index=-1):
    return FusedLinearCrossEntropy.apply(
        x.reshape(-1, x.shape[-1]), weight, target.reshape(-1), n_chunks, ignore_index)


class Block(nn.Module):
    """Pre-norm residual block: x += mixer(norm(x)); x += ffn(norm(x))."""

    def __init__(self, cfg, mixer: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.norm1 = _norm(cfg, cfg.d_model)
        self.mixer = build_mixer(cfg, mixer)
        self.norm2 = _norm(cfg, cfg.d_model)
        self.ffn = build_ffn(cfg)

    def forward(self, x, cos, sin, v0):
        mixed, raw_v = self.mixer(self.norm1(x), cos, sin, v0)
        x = x + mixed
        x = x + self.ffn(self.norm2(x))
        return x, raw_v

    def forward_cached(self, x, cos, sin, v0, cache, commit, causal=False):
        """KV-cached counterpart of ``forward`` for one block window (attention
        mixers only — recurrent mixers would need their own state carry)."""
        mixed, raw_v = self.mixer.forward_cached(
            self.norm1(x), cos, sin, v0, cache, commit, causal)
        x = x + mixed
        x = x + self.ffn(self.norm2(x))
        return x, raw_v


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layer_kinds = parse_layer_mixers(cfg)
        self.blocks = nn.ModuleList(
            [Block(cfg, kind) for kind in self.layer_kinds]
        )
        self.norm_f = _norm(cfg, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight     # weight tying

        # learned absolute positions only if RoPE is off
        self.pos_emb = None
        if cfg.pos != "rope":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)

        # μP (§10): tune HPs at base width, transfer to larger width. Width factor
        # drives the output multiplier, hidden-LR (in optim) and μP init below.
        self.width_mult = (cfg.d_model / cfg.mup_base_width) if cfg.mup else 1.0
        self.output_mult = 1.0 / self.width_mult        # logit scale (= 1 if no μP)

        self._rope = {}   # cached per (seq_len, device, dtype)
        self._moe_aux = 0.0
        self.apply(self._init_weights)
        if cfg.mup and self.width_mult != 1.0:
            self._mup_init()
        if cfg.zero_init_proj:
            self._zero_init_output_projections()
        # tied weights are deduped by parameters(), so this counts them once
        self.n_params = sum(p.numel() for p in self.parameters())

    # -- init (guide §2.1 zero-init; standard GPT-2-style for the rest) --------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _mup_init(self):
        """μP init: hidden-layer weights scale as 1/sqrt(width_mult) so the
        forward activation scale is width-invariant (Tensor Programs V). Input
        embeddings and the LM head keep their base init."""
        s = 1.0 / math.sqrt(self.width_mult)
        for blk in self.blocks:
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.mul_(s)

    def _zero_init_output_projections(self):
        """Zero the residual-path output projections so each block starts as
        identity — a cheap, strong stabilizer (guide §2.1, §2.3)."""
        for blk in self.blocks:
            mixer = blk.mixer
            for attr in ("o_proj", "out_proj", "out"):
                proj = getattr(mixer, attr, None)
                if isinstance(proj, nn.Linear):
                    nn.init.zeros_(proj.weight)
            ffn = blk.ffn
            if isinstance(getattr(ffn, "down", None), nn.Linear):
                nn.init.zeros_(ffn.down.weight)

    def set_causal(self, causal: bool):
        """Toggle causal vs bidirectional attention across all layers. Used by
        the diffusion conversion (§9): a masked-diffusion model attends
        bidirectionally, so generation sees both sides of a masked token."""
        for blk in self.blocks:
            if hasattr(blk.mixer, "causal"):
                blk.mixer.causal = causal

    def set_block_attention(self, block_len: int):
        """Set a block-causal (semi-autoregressive) attention mask on every
        attention layer: causal ACROSS blocks of ``block_len`` tokens,
        bidirectional WITHIN each block. This is the attention pattern behind
        block diffusion / Nemotron-Labs-Diffusion-style tri-mode decoding — one
        weight set then spans AR (block_len=1), block diffusion (1<block_len<T)
        and full diffusion (block_len>=T). ``block_len<=0`` disables it and
        restores the plain causal/bidirectional ``set_causal`` behaviour."""
        for blk in self.blocks:
            if hasattr(blk.mixer, "block_attn"):
                blk.mixer.block_attn = max(0, block_len)

    def _rope_cache(self, T, device, dtype):
        key = (T, device, dtype)
        if key not in self._rope:
            self._rope[key] = build_rope_cache(
                T, self.cfg.head_dim, self.cfg.rope_base, device, dtype)
        return self._rope[key]

    def _rope_slice(self, start, length, device, dtype):
        """RoPE cos/sin for absolute positions [start, start+length) — what a KV-
        cached window needs (its queries/keys sit at non-zero absolute offsets)."""
        cos, sin = self._rope_cache(start + length, device, dtype)
        return cos[start:start + length], sin[start:start + length]

    @torch.no_grad()
    def forward_hidden_window(self, x, abs_start, caches, commit, causal=False):
        """Run the trunk over ONE block window ``x`` (B,T) whose first token is at
        absolute position ``abs_start``, threading a per-layer KV ``caches`` list
        (one dict per block/layer). Returns hidden states for the window only.
        Used by the cached semi-AR / self-speculation samplers so finalized blocks
        are never recomputed. ``causal`` makes the active block attend causally
        within itself (lossless self-spec verify). Attention mixers only."""
        cfg = self.cfg
        h = self.tok_emb(x)
        if self.pos_emb is not None:
            pos = torch.arange(abs_start, abs_start + x.shape[1], device=x.device)
            h = h + self.pos_emb(pos)[None]
        h = self.drop(h)
        if cfg.pos == "rope":
            cos, sin = self._rope_slice(abs_start, x.shape[1], x.device, h.dtype)
        else:
            cos = sin = None
        v0 = None
        for blk, cache in zip(self.blocks, caches):
            h, raw_v = blk.forward_cached(h, cos, sin, v0, cache, commit, causal)
            if v0 is None and raw_v is not None:
                v0 = raw_v
        return self.norm_f(h)

    def forward_hidden(self, idx):
        """Run the trunk (embeddings -> blocks -> final norm) and return the
        per-position hidden states (B, T, d). Used by the diffusion objective
        (§9), which needs logits at EVERY position, not just the last."""
        cfg = self.cfg
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(idx.shape[1], device=idx.device)
            x = x + self.pos_emb(pos)[None]
        x = self.drop(x)
        if cfg.pos == "rope":
            cos, sin = self._rope_cache(idx.shape[1], idx.device, x.dtype)
        else:
            cos = sin = None
        v0 = None
        for blk in self.blocks:
            if cfg.grad_checkpoint and self.training:
                x, raw_v = checkpoint(blk, x, cos, sin, v0, use_reentrant=False)
            else:
                x, raw_v = blk(x, cos, sin, v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v          # capture layer-0 values for value residual
        # collect MoE load-balancing aux loss (0 if no MoE layers)
        if cfg.ffn == "moe":
            self._moe_aux = sum(b.ffn.aux for b in self.blocks if b.ffn.aux is not None)
        return self.norm_f(x)

    def forward(self, idx, targets=None):
        x = self.forward_hidden(idx)
        cfg = self.cfg
        # μP output multiplier (1.0 without μP). Scaling the head input is
        # equivalent to scaling the logits and keeps fused-CE/grad correct.
        if self.output_mult != 1.0:
            x = x * self.output_mult
        aux = (cfg.moe_aux_weight * self._moe_aux) if cfg.ffn == "moe" else 0.0
        if targets is not None:
            if cfg.fused_ce:
                # never materialize full logits (guide §7.1) — returns loss only
                loss = fused_linear_cross_entropy(
                    x, self.lm_head.weight, targets, cfg.fused_ce_chunks)
                return None, loss + aux
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss + aux
        # inference: only compute the last position's logits
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    # -- muP-aware MFU estimate (guide §6.1) ---------------------------------
    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_emb is not None:
            n -= self.pos_emb.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """nanoGPT-style estimate: 6*N + 12*L*H*Q*T (attention path). For MoE the
        non-embedding param count is reduced to the *active* params (only top_k of
        the experts run per token) so MFU isn't overcounted."""
        cfg = self.cfg
        N = self.num_params(non_embedding=True)
        if cfg.ffn == "moe" and cfg.moe_experts > cfg.moe_top_k:
            # discount the inactive experts: each block has moe_experts FFNs but
            # only moe_top_k execute per token.
            ffn_params = sum(p.numel() for b in self.blocks for p in b.ffn.experts.parameters())
            inactive = ffn_params * (1 - cfg.moe_top_k / cfg.moe_experts)
            N -= int(inactive)
        return 6 * N + mixer_flops_per_token(cfg)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


def mixer_flops_per_token(cfg) -> int:
    """The sequence-mixer's own FLOPs term.

    Sole owner of this formula. `GPT.flops_per_token` and the crossover probe's
    MFU estimate both call it: they used to carry separate copies, and the copy
    in the probe silently charged an unrecognised mixer ZERO attention FLOPs --
    so a new mixer's MFU came out too low there and too high nowhere, with
    nothing to reveal the disagreement.

    Convention: T keys per query, with the causal factor of 2 left out (as the
    original did). A windowed query can never see more than `swa_window` keys,
    so SWA's span is capped -- charging it the dense T would inflate its MFU by
    T/window, 8x at SWA(64,4) on a 512 context.
    """
    kinds = parse_layer_mixers(cfg)
    H, Q, T = cfg.n_head, cfg.head_dim, cfg.block_size
    n_attn = sum(1 for k in kinds if k in ("attention", "mla"))
    n_swa = sum(1 for k in kinds if k == "swa")
    return 12 * H * Q * (n_attn * T + n_swa * min(cfg.swa_window, T))


def build_model(cfg) -> GPT:
    return GPT(cfg)
