"""
nanolab.mixers — the pluggable sequence mixer (guide §2.5).

Everything else in the model (embeddings, MLP, optimizer, seed, tokens) is held
fixed; only THIS module changes when you pass ``--mixer``. That is what makes
the Transformer-vs-SSM A/B honest (guide §2.5, §8).

| mixer      | what it is                 | cost in T | deps                         |
|------------|----------------------------|-----------|------------------------------|
| attention  | full pairwise mixing       | quadratic | none (SDPA / FlashAttention) |
| mingru     | minimal parallel RNN       | linear    | none (pure torch reference)  |
| mamba2     | selective state-space (SSD)| linear    | none (pure torch, slow)      |
| gdn        | gated linear attention     | linear    | none (pure torch, slow)      |

The recurrent mixers here are *correctness-first, pure-PyTorch* references — no
mamba-ssm / flash-linear-attention CUDA kernels required (guide notes those as
optional pip installs). They run on the 3070 Ti and on CPU; they are slower than
the fused kernels but produce identical math. The repo's competition trainers
(train_hypercascade.py / train_rada.py) hold the chunk-parallel fast paths,
verified to 1e-5 by verify_scan.py / verify_gdn.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square norm (no mean subtraction) — cheaper than LayerNorm and
    just as stable in pre-norm (guide §2.1)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    """Precompute cos/sin for rotary embeddings (guide §2.1)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)             # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)      # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, T, H, D)  ;  cos/sin: (T, D)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + rot * sin


def block_causal_mask(T: int, block_len: int, device, cache: dict) -> torch.Tensor:
    """Block-causal (semi-autoregressive) attention mask, cached per (T, block_len).

    A query may attend to every position in its OWN block (bidirectional, like
    diffusion) and in all EARLIER blocks (causal, across blocks). The two extremes
    fall out for free: ``block_len==1`` is a plain causal mask, ``block_len>=T`` is
    fully bidirectional — so a single integer spans AR, block-diffusion and full
    diffusion attention (the Nemotron-Labs-Diffusion tri-mode idea, see
    diffusion.py). Returns a (1,1,T,T) boolean keep-mask for SDPA's ``attn_mask``."""
    key = (T, block_len, device)
    m = cache.get(key)
    if m is None:
        blk = torch.arange(T, device=device) // block_len      # block id per position
        m = (blk[:, None] >= blk[None, :])[None, None]          # query-block >= key-block
        cache[key] = m
    return m


# ---------------------------------------------------------------------------
# 1. Attention — the default, strongest sub-200M mixer (guide §2.5)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    """GQA + RoPE + QK-norm, with optional gated-output and value-residual.

    Gated attention + value residual were the *champion* combination in this
    repo's own architecture ladder (logs/ablations champion.json), so they are
    on by default for the attention path.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.mup = cfg.mup
        # muP prescribes 1/d attention logits; mup_sqrt_attn_scale keeps SP's
        # 1/sqrt(d) so the term can be ablated on its own.
        self.attn_scale = (1.0 / cfg.head_dim
                           if cfg.mup and not cfg.mup_sqrt_attn_scale
                           else 1.0 / math.sqrt(cfg.head_dim))
        d = cfg.d_model
        self.q_proj = nn.Linear(d, cfg.n_head * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(d, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.head_dim, d, bias=False)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim)
            self.k_norm = RMSNorm(cfg.head_dim)

        self.gated = cfg.gated_attention
        if self.gated:
            # per-head sigmoid gate on the attention output
            self.gate = nn.Linear(d, cfg.n_head, bias=True)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)  # start gate at 0.5 (neutral)

        self.value_residual = cfg.value_residual
        if self.value_residual:
            # learnable per-layer blend lambda; 0 => pure current value
            self.vr_lambda = nn.Parameter(torch.zeros(1))

        # causal by default (AR). The diffusion path (§9) flips this to False for
        # bidirectional attention; GPT.set_causal() toggles every layer at once.
        self.causal = True
        # >0 => block-causal (semi-AR) mask of this block length, set by
        # GPT.set_block_attention(); overrides ``causal`` when active.
        self.block_attn = 0
        self._mask_cache = {}

    def forward(self, x, cos, sin, v0=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        raw_v = v

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # value residual: blend layer-0 values (guide notes value-embeds/residual)
        if self.value_residual and v0 is not None:
            lam = torch.sigmoid(self.vr_lambda)
            v = (1 - lam) * v + lam * v0

        # (B, H, T, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # GQA: repeat KV heads to match Q heads
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scale = self.attn_scale
        if self.block_attn > 0:                  # semi-AR: causal across blocks, dense within
            mask = block_causal_mask(T, self.block_attn, x.device, self._mask_cache)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=scale)
        y = y.transpose(1, 2).contiguous()       # (B, T, H, D)

        if self.gated:
            g = torch.sigmoid(self.gate(x)).unsqueeze(-1)   # (B,T,H,1)
            y = y * g

        y = y.view(B, T, self.n_head * self.head_dim)
        return self.o_proj(y), raw_v

    def forward_cached(self, x, cos, sin, v0, cache, commit, causal=False):
        """Incremental forward over ONE block window for semi-AR / self-spec
        sampling. ``x`` is the active block; ``cos``/``sin`` are RoPE for its
        ABSOLUTE positions; ``cache`` is a per-layer dict holding the K,V of all
        previously-finalized blocks (pre-GQA, (B, Hkv, L, D)).

        The active queries always attend in FULL to the cache (strictly-earlier
        context). Within the active block: ``causal=False`` (block diffusion) is
        bidirectional, ``causal=True`` is lower-triangular — the latter makes the
        window forward equivalent to a plain causal pass over [cache | block], which
        is what the lossless self-speculation VERIFY needs. ``commit=True`` writes
        this block's K,V into the cache (once finalized); ``commit=False`` (denoise/
        draft rounds, whose tokens still change) leaves the cache untouched. Either
        way the cached context is never recomputed — the throughput win."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        raw_v = v
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if self.value_residual and v0 is not None:
            lam = torch.sigmoid(self.vr_lambda)
            v = (1 - lam) * v + lam * v0

        q = q.transpose(1, 2)                         # (B,H,T,D)
        k_t, v_t = k.transpose(1, 2), v.transpose(1, 2)   # (B,Hkv,T,D)
        ck, cv = cache.get("k"), cache.get("v")
        k_full = k_t if ck is None else torch.cat([ck, k_t], dim=2)
        v_full = v_t if cv is None else torch.cat([cv, v_t], dim=2)
        if commit:
            cache["k"], cache["v"] = k_full.detach(), v_full.detach()

        kk, vv = k_full, v_full
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            kk = kk.repeat_interleave(rep, dim=1)
            vv = vv.repeat_interleave(rep, dim=1)
        scale = self.attn_scale
        attn_mask = None
        if causal:                                # full to cache, causal within block
            Lk = k_full.shape[2]
            left = torch.ones(T, Lk - T, dtype=torch.bool, device=x.device)
            right = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            attn_mask = torch.cat([left, right], dim=1)[None, None]
        y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=attn_mask,
                                           is_causal=False, scale=scale)
        y = y.transpose(1, 2).contiguous()
        if self.gated:
            g = torch.sigmoid(self.gate(x)).unsqueeze(-1)
            y = y * g
        y = y.view(B, T, self.n_head * self.head_dim)
        return self.o_proj(y), raw_v


# ---------------------------------------------------------------------------
# 2. minGRU — minimal parallel RNN, zero-dependency recurrent reference
#    ("Were RNNs All We Needed?", arXiv:2410.01201) (guide §2.5)
# ---------------------------------------------------------------------------
class MinGRU(nn.Module):
    """Parallel-scan minGRU. Linear in T, fully parallel via a log-space
    associative scan, no external deps. The pedagogy baseline for "is the mixer
    or the recipe what matters?" (guide §2.5).

    Optional value-residual (``cfg.value_residual``): layer 0 exposes ``raw_v``
    via ``W_v``; deeper layers blend ``v0 @ W_v0_up`` into the scan input
    ``h_pre`` with metal-compatible 2-weight ``vr_lambda``."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.expand = 2
        hid = self.expand * d
        self.to_z = nn.Linear(d, hid, bias=False)
        self.to_h = nn.Linear(d, hid, bias=False)
        self.out = nn.Linear(hid, d, bias=False)

        self.value_residual = cfg.value_residual
        if self.value_residual:
            self.n_kv_head = cfg.n_kv_head
            self.head_dim = cfg.head_dim
            kv = self.n_kv_head * self.head_dim
            self.v_proj = nn.Linear(d, kv, bias=False)       # W_v  -> [B,T,KV]
            self.v0_up = nn.Linear(kv, hid, bias=False)     # W_v0_up [KV,2C]
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5]))

    @staticmethod
    def _g(x):  # continuous, positive activation from the paper
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

    @staticmethod
    def _log_g(x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def forward(self, x, cos=None, sin=None, v0=None):
        B, T, _ = x.shape
        # log-space parallel scan for numerical stability
        log_z = -F.softplus(-self.to_z(x))            # log(sigmoid(z))
        log_coeff = -F.softplus(self.to_z(x))         # log(1 - sigmoid(z))
        h_pre = self.to_h(x)                          # [B,T,2C]
        raw_v = None
        if self.value_residual:
            raw_v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
            if v0 is not None:
                v0_flat = v0.reshape(B, T, -1)
                h_pre = (self.vr_lambda[0] * self.v0_up(v0_flat)
                         + self.vr_lambda[1] * h_pre)
        log_h_tilde = self._log_g(h_pre)
        log_zh = log_z + log_h_tilde
        h = _parallel_scan_log(log_coeff, log_zh)
        return self.out(h), raw_v


def _parallel_scan_log(log_coeffs, log_values):
    """Heinsen associative scan in log space: h_t = a_t * h_{t-1} + b_t,
    with a_t = exp(log_coeffs), b_t = exp(log_values). O(T) sequential here for
    clarity & correctness; vectorized cumsum keeps it fast enough for reference
    use. (B, T, H)."""
    a_star = torch.cumsum(log_coeffs, dim=1)
    log_h0_plus_b = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b
    return log_h.exp()


# ---------------------------------------------------------------------------
# 3. Mamba-2 (SSD) — selective state space, pure-PyTorch sequential reference
#    "Transformers are SSMs", arXiv:2405.21060 (guide §2.5)
# ---------------------------------------------------------------------------
def ssd_chunk_parallel(x_scaled, B_h, C_h, log_dA, chunk):
    """Chunk-parallel SSD scan (guide §7) — the genuinely-vectorized one.

    Inputs (all fp32): x_scaled (B,L,H,P) = dt-scaled input, B_h/C_h (B,L,N),
    log_dA (B,L,H) <= 0. Returns y (B,L,H,P). The O(L) sequential recurrence is
    replaced by 2·K Python steps (K=ceil(L/C)): pass 1 solves each C×C chunk in
    parallel via matrix products (a causal decay-weighted C·Bᵀ attention), pass 2
    carries the chunk-final state across chunks. At L=1024/C=32 that's 64 steps,
    not 1024 — and it is plain autograd-differentiable (no custom Function needed,
    the graph is only 2K deep). Ported from the repo's verified
    parameter-golf/train_hypercascade.py::_ssd_chunk_parallel (verify_scan.py, 1e-5).
    """
    with torch.autocast(device_type=x_scaled.device.type, enabled=False):
        x_scaled, B_h, C_h, log_dA = (t.float() for t in (x_scaled, B_h, C_h, log_dA))
        Bsz, L, H, P = x_scaled.shape
        N = B_h.shape[-1]
        C = chunk
        L_pad = (L + C - 1) // C * C
        K = L_pad // C
        if L_pad > L:
            pad = L_pad - L
            x_scaled = F.pad(x_scaled, (0, 0, 0, 0, 0, pad))
            B_h = F.pad(B_h, (0, 0, 0, pad))
            C_h = F.pad(C_h, (0, 0, 0, pad))
            log_dA = F.pad(log_dA, (0, 0, 0, pad))     # 0 -> no decay on pad
        x_c = x_scaled.view(Bsz, K, C, H, P)
        B_c = B_h.view(Bsz, K, C, N)
        C_c = C_h.view(Bsz, K, C, N)
        ldA_c = log_dA.view(Bsz, K, C, H)
        cmask = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x_scaled.device))

        y_local = x_scaled.new_zeros(Bsz, K, C, H, P)
        h_chunk = x_scaled.new_zeros(Bsz, K, H, P, N)   # final state per chunk
        for k in range(K):                              # pass 1: local O(C^2) SSD
            cl = torch.cumsum(ldA_c[:, k], dim=1)       # (B,C,H) cumulative log-decay
            diff = cl[:, :, None, :] - cl[:, None, :, :]
            M_c = torch.exp(diff.clamp(max=0.0)) * cmask[None, :, :, None]
            CB_c = torch.matmul(C_c[:, k], B_c[:, k].transpose(1, 2))   # (B,C,C)
            W_c = M_c * CB_c[:, :, :, None]
            y_local[:, k] = torch.einsum("bcsh,bshp->bchp", W_c, x_c[:, k])
            decay_to_end = torch.exp((cl[:, -1:, :] - cl).clamp(max=0.0))
            h_chunk[:, k] = torch.einsum("bch,bchp,bcn->bhpn",
                                         decay_to_end, x_c[:, k], B_c[:, k])

        y_carry = x_scaled.new_zeros(Bsz, K, C, H, P)
        h_carry = x_scaled.new_zeros(Bsz, H, P, N)
        for k in range(K):                              # pass 2: carry propagation
            if k > 0:
                cl = torch.cumsum(ldA_c[:, k], dim=1)
                M_cum = torch.exp(cl)
                y_carry[:, k] = torch.einsum("bch,bhpn,bcn->bchp",
                                             M_cum, h_carry, C_c[:, k])
            M_tot = torch.exp(ldA_c[:, k].sum(dim=1))   # (B,H)
            h_carry = M_tot[:, :, None, None] * h_carry + h_chunk[:, k]
        return (y_local + y_carry).reshape(Bsz, L_pad, H, P)[:, :L]


class Mamba2(nn.Module):
    """Minimal Mamba-2/SSD mixer in pure PyTorch.

    The scan uses the chunk-parallel ``ssd_chunk_parallel`` kernel above; the
    O(L) sequential recurrence the repo's verify_scan.py validates to 1e-5 is
    kept as ``_sequential`` for the regression test.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.d_inner = 2 * d
        self.n_head = max(1, self.d_inner // 64)
        self.head_dim = self.d_inner // self.n_head
        self.d_state = cfg.d_state
        self.d_conv = 4

        self.in_proj = nn.Linear(d, 2 * self.d_inner + 2 * self.d_state + self.n_head, bias=False)
        conv_dim = self.d_inner + 2 * self.d_state   # conv spans x, B, C together
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, self.d_conv,
                                groups=conv_dim, padding=self.d_conv - 1, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.n_head + 1).float()))
        self.D = nn.Parameter(torch.ones(self.n_head))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_head))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)
        self.chunk = min(cfg.mixer_chunk, cfg.block_size) or 32   # SSD chunk size

    def _project(self, x):
        """Shared front-end: returns z (gate), and the SSD scan inputs in fp32:
        x_scaled = dt*x, log_dA = dt*A, Bm, Cm, plus the raw xs for the D skip."""
        B, T, _ = x.shape
        zxbcdt = self.in_proj(x)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.n_head], dim=-1)
        xBC = self.conv1d(xBC.transpose(1, 2))[..., :T].transpose(1, 2)
        xBC = F.silu(xBC)
        xs, Bm, Cm = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias).float()    # (B,T,H)
        A = -torch.exp(self.A_log.float())            # (H,)
        xs = xs.view(B, T, self.n_head, self.head_dim).float()
        x_scaled = dt.unsqueeze(-1) * xs              # (B,T,H,P)  dt-scaled input
        log_dA = dt * A                               # (B,T,H)    <= 0
        return z, xs, x_scaled, Bm.float(), Cm.float(), log_dA

    def _finish(self, x, z, xs, y):
        B, T = x.shape[0], x.shape[1]
        y = y + xs * self.D.view(1, 1, self.n_head, 1)
        y = y.reshape(B, T, self.d_inner).to(x.dtype)
        return self.out_proj(self.norm(y) * F.silu(z)), None

    def forward(self, x, *args, **kwargs):
        z, xs, x_scaled, Bm, Cm, log_dA = self._project(x)
        y = ssd_chunk_parallel(x_scaled, Bm, Cm, log_dA, self.chunk)   # (B,T,H,P)
        return self._finish(x, z, xs, y)

    def _sequential(self, x):
        """O(T) reference recurrence over the SSD state h (B,H,P,N) — kept for the
        chunk-kernel regression test (verify_scan.py validates it to 1e-5)."""
        B, T, _ = x.shape
        z, xs, x_scaled, Bm, Cm, log_dA = self._project(x)
        dA = torch.exp(log_dA)
        h = x.new_zeros(B, self.n_head, self.head_dim, self.d_state, dtype=torch.float32)
        ys = []
        for t in range(T):
            dBx = x_scaled[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1).unsqueeze(1)
            h = dA[:, t].unsqueeze(-1).unsqueeze(-1) * h + dBx
            ys.append(torch.einsum("bhpn,bn->bhp", h, Cm[:, t]))
        return self._finish(x, z, xs, torch.stack(ys, dim=1))


# ---------------------------------------------------------------------------
# 4. Gated DeltaNet (GDN) — gated linear attention with delta-rule recall
#    arXiv:2412.06464 (guide §2.5)
# ---------------------------------------------------------------------------
# Chunk-parallel kernel (guide §7): the per-timestep recurrence is O(T) sequential
# matmuls — brutally slow on a GPU (the pure-PyTorch sequential ref benched at
# ~240 tok/s, ~33x slower than attention). This processes the scan in chunks: the
# inter-chunk state carry is the only thing that stays sequential (O(T/chunk)
# steps), so the GPU sees far fewer, larger kernels. The forward/backward are
# written by hand (one autograd.Function) so we never build a T-deep autograd
# graph — backward only stores the N=T/chunk small (D,D) state carries. Ported and
# adapted from the repo's verified parameter-golf/verify_gdn_wy.py (matches the
# sequential reference to <1e-9 in fp64; see the regression test).
def gdn_chunked(q, k, v, alpha, beta, chunk=32):
    """Chunk-parallel gated delta rule via the WY / UT-transform (guide §7).

    q,k,v: [B,H,L,D]; alpha,beta: [B,H,L]. The O(T) recurrence
        e_t = v_t - S_{t-1} k_t ;  S_t = a_t S_{t-1} + b_t e_t k_tᵀ ;  y_t = S_t q_t
    is solved in 2·(T/chunk) steps with NO per-timestep loop. Unrolling within a
    chunk, the write vectors u_t = b_t e_t satisfy a unit-lower-triangular system
        (I + M) U = R,   M[t,j] = b_t (A_{t-1}/A_j)(k_t·k_j) (j<t),
                          R_t   = b_t v_t − b_t A_{t-1} (S_in k_t),
    where A_t = ∏_{i≤t} a_i is the cumulative gate. U = (I+M)⁻¹R is one batched
    triangular solve; outputs and the chunk-final state then fall out of two
    decay-weighted matmuls. Only the chunk carry S_in stays sequential. Pure
    autograd (no manual backward); fp32 to match — verified exact vs the O(T)
    reference (`_sequential`, test `gdn_chunked_matches_sequential`). This is the
    fully-vectorised analogue of `ssd_chunk_parallel` for the delta rule."""
    with torch.autocast(device_type=k.device.type, enabled=False):
        q, k, v = q.float(), k.float(), v.float()
        # log(alpha) in the WY kernel: sigmoid gates can be 0 after a large
        # update, then backward of log is 1/alpha = inf and the run NaNs
        # (loss stayed finite through step 50 on GH200; gnorm blew at 55).
        alpha = alpha.float().clamp(min=1e-4, max=1.0)
        beta = beta.float().clamp(min=0.0, max=1.0)
        B, H, L, D = k.shape
        C = chunk
        pad = (-L) % C
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            alpha = F.pad(alpha, (0, pad), value=1.0)     # identity: keep state
            beta = F.pad(beta, (0, pad), value=0.0)       # no write
        N = (L + pad) // C
        qc, kc, vc = (t.view(B, H, N, C, D) for t in (q, k, v))
        bc = beta.view(B, H, N, C)
        cum = torch.cumsum(torch.log(alpha.view(B, H, N, C)), dim=-1)   # A_t = exp(cum_t)
        cum_prev = cum - torch.log(alpha.view(B, H, N, C))              # A_{t-1}
        # (I+M) U = R, M strictly lower; U = (I+M)^{-1} R via one batched solve
        # Exponentiate only the triangular part we use. The full CxC ratio
        # A_{t-1}/A_j is >>1 for j>t and overflowed to inf; autograd then NaN'd
        # even though .tril() hid those entries in the forward.
        logM = (cum_prev[..., :, None] - cum[..., None, :]).tril(-1)
        decM = torch.exp(logM.clamp(max=0.0))
        M = (bc[..., :, None] * decM * (kc @ kc.transpose(-1, -2))).tril(-1)
        eye = torch.eye(C, device=k.device, dtype=k.dtype).expand(B, H, N, C, C)
        Tinv = torch.linalg.solve_triangular(eye + M, eye, upper=False, unitriangular=True)
        logY = (cum[..., :, None] - cum[..., None, :]).tril(0)
        decY = torch.exp(logY.clamp(max=0.0))                 # A_t/A_j
        Wy = (decY * (qc @ kc.transpose(-1, -2))).tril(0)     # output weights
        A_t = torch.exp(cum)[..., None]                       # (B,H,N,C,1)
        A_prev = torch.exp(cum_prev)
        dec_jC = torch.exp((cum[..., -1:] - cum).clamp(max=0.0))[..., None]  # A_C/A_j
        A_C = torch.exp(cum[..., -1])[..., None, None]        # (B,H,N,1,1)

        S = torch.zeros(B, H, D, D, device=k.device, dtype=k.dtype)     # state k->v
        ys = []
        for m in range(N):
            km, vm, qm = kc[:, :, m], vc[:, :, m], qc[:, :, m]          # (B,H,C,D)
            sk = torch.einsum("bhde,bhce->bhcd", S, km)                 # S k_t
            R = bc[:, :, m][..., None] * vm - (bc[:, :, m] * A_prev[:, :, m])[..., None] * sk
            u = torch.einsum("bhtj,bhjd->bhtd", Tinv[:, :, m], R)       # u = (I+M)^-1 R
            sq = torch.einsum("bhde,bhce->bhcd", S, qm)                 # S q_t
            y = A_t[:, :, m] * sq + torch.einsum("bhtj,bhjd->bhtd", Wy[:, :, m], u)
            ys.append(y)
            dS = torch.einsum("bhcd,bhce->bhde", u * dec_jC[:, :, m], km)   # Σ (A_C/A_j) u_j k_jᵀ
            S = A_C[:, :, m] * S + dS
        return torch.cat(ys, dim=2)[:, :, :L]


class GatedDeltaNet(nn.Module):
    """Gated delta rule, computed by the chunk-parallel kernel above (the original
    O(T) sequential recurrence is kept as ``_sequential`` for the regression test).

    State update per step (FP32):
        S <- alpha * S + beta * k (v - S^T k)^T   (delta correction)
        y  = S^T q
    Verified against this exact recurrence by the repo's verify_gdn.py (1e-4).
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_head = max(1, d // 64)
        self.head_dim = d // self.n_head
        self.in_proj = nn.Linear(d, 3 * d + 2 * self.n_head, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.decay_bias = nn.Parameter(torch.zeros(self.n_head))
        self.update_bias = nn.Parameter(torch.zeros(self.n_head))
        self.out_norm = RMSNorm(d)
        self.out_proj = nn.Linear(d, d, bias=False)
        # chunk size for the parallel scan; smaller = less intra-chunk work but
        # more sequential carries. 32 is a good default at head_dim ~64.
        self.chunk = min(cfg.mixer_chunk, cfg.block_size) or 32

    def _project(self, x):
        """Shared input projection -> (q, k, v, alpha, beta) in [B,H,T,P]/[B,H,T]."""
        B, T, D = x.shape
        H, P = self.n_head, self.head_dim
        q, k, v, a_gate, b_gate = torch.split(
            self.in_proj(x), [D, D, D, self.n_head, self.n_head], dim=-1)
        q = self.q_norm(q.view(B, T, H, P))
        k = self.k_norm(k.view(B, T, H, P))
        k = F.normalize(k, dim=-1)                    # L2-normalize keys
        v = v.view(B, T, H, P)
        alpha = torch.sigmoid(a_gate + self.decay_bias).clamp(1e-4, 1.0)
        beta = torch.sigmoid(b_gate + self.update_bias).clamp(0.0, 1.0)
        # -> [B,H,T,P] / [B,H,T], fp32 for the scan (state accumulation is touchy)
        q, k, v = (t.transpose(1, 2).float() for t in (q, k, v))
        return q, k, v, alpha.transpose(1, 2).float(), beta.transpose(1, 2).float()

    def forward(self, x, *args, **kwargs):
        B, T, D = x.shape
        q, k, v, alpha, beta = self._project(x)
        y = gdn_chunked(q, k, v, alpha, beta, self.chunk)   # [B,H,T,P]
        y = y.transpose(1, 2).reshape(B, T, D).to(x.dtype)
        return self.out_proj(self.out_norm(y)), None

    def _sequential(self, x):
        """O(T) reference recurrence — kept for the chunked-kernel regression test
        (``S`` here is the transpose of the kernel's state; outputs are identical)."""
        B, T, D = x.shape
        H, P = self.n_head, self.head_dim
        q, k, v, alpha, beta = self._project(x)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # back to [B,T,H,P]
        alpha, beta = alpha.transpose(1, 2), beta.transpose(1, 2)
        S = x.new_zeros(B, H, P, P, dtype=torch.float32)
        ys = []
        for t in range(T):
            kt, vt = k[:, t], v[:, t]
            at = alpha[:, t].unsqueeze(-1).unsqueeze(-1)
            bt = beta[:, t].unsqueeze(-1)
            pred = torch.einsum("bhpn,bhp->bhn", S, kt)
            delta = (vt - pred) * bt
            S = at * S + torch.einsum("bhp,bhn->bhpn", kt, delta)
            ys.append(torch.einsum("bhpn,bhp->bhn", S, q[:, t]))
        y = torch.stack(ys, dim=1).reshape(B, T, D).to(x.dtype)
        return self.out_proj(self.out_norm(y)), None


# ---------------------------------------------------------------------------
# 5. MLA — Multi-head Latent Attention (DeepSeek-V2, guide §2.1)
# ---------------------------------------------------------------------------
class MLA(nn.Module):
    """Low-rank KV compression + decoupled RoPE. Q and K/V are projected through
    small latent bottlenecks (``q_lora_rank`` / ``kv_lora_rank``), and position
    is carried by a separate ``rope_head_dim`` slice that is shared across heads
    for K — which is what shrinks the KV cache at decode time. The non-positional
    ("nope") slice is content-only. (See the §2.4 note: MLA is an AR-decode
    optimization; re-evaluate for the bidirectional diffusion path.)"""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.nope = cfg.head_dim                       # content head dim
        self.rope = cfg.rope_head_dim or (cfg.head_dim // 2)
        self.kv_rank = cfg.kv_lora_rank or (d // 4)
        self.q_rank = cfg.q_lora_rank or (d // 2)
        self.rope_base = cfg.rope_base
        H, nope, rope = self.n_head, self.nope, self.rope

        # Q path: x -> q_rank -> [nope + rope] per head
        self.q_a = nn.Linear(d, self.q_rank, bias=False)
        self.q_a_norm = RMSNorm(self.q_rank)
        self.q_b = nn.Linear(self.q_rank, H * (nope + rope), bias=False)
        # KV path: x -> [kv_rank latent | shared k_rope]; latent -> per-head k_nope + v
        self.kv_a = nn.Linear(d, self.kv_rank + rope, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_rank)
        self.kv_b = nn.Linear(self.kv_rank, H * (nope + nope), bias=False)  # k_nope + v
        self.o_proj = nn.Linear(H * nope, d, bias=False)
        self._rope = {}
        self.causal = True
        self.block_attn = 0
        self._mask_cache = {}

    def _rope_cache(self, T, device, dtype):
        key = (T, device, dtype)
        if key not in self._rope:
            self._rope[key] = build_rope_cache(T, self.rope, self.rope_base, device, dtype)
        return self._rope[key]

    def forward(self, x, cos, sin, v0=None):
        B, T, _ = x.shape
        H, nope, rope = self.n_head, self.nope, self.rope
        cos, sin = self._rope_cache(T, x.device, x.dtype)   # MLA owns its RoPE dim

        q = self.q_b(self.q_a_norm(self.q_a(x))).view(B, T, H, nope + rope)
        q_nope, q_rope = q.split([nope, rope], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)

        kv = self.kv_a(x)
        kv_lat, k_rope = kv.split([self.kv_rank, rope], dim=-1)
        kv = self.kv_b(self.kv_a_norm(kv_lat)).view(B, T, H, nope + nope)
        k_nope, v = kv.split([nope, nope], dim=-1)
        # k_rope is shared across heads (the MLA cache-shrink trick) -> broadcast
        k_rope = apply_rope(k_rope.view(B, T, 1, rope), cos, sin).expand(B, T, H, rope)

        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)   # (B,H,T,nope+rope)
        k = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(nope + rope)
        if self.block_attn > 0:
            mask = block_causal_mask(T, self.block_attn, x.device, self._mask_cache)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=scale)
        y = y.transpose(1, 2).contiguous().view(B, T, H * nope)
        return self.o_proj(y), None


# ---------------------------------------------------------------------------
MIXER_REGISTRY = {
    "attention": Attention,
    "mingru": MinGRU,
    "mamba2": Mamba2,
    "gdn": GatedDeltaNet,
    "mla": MLA,
}


def build_mixer(cfg, mixer: str | None = None) -> nn.Module:
    kind = mixer if mixer is not None else cfg.mixer
    if kind not in MIXER_REGISTRY:
        raise KeyError(f"unknown mixer {kind!r}; known: {list(MIXER_REGISTRY)}")
    return MIXER_REGISTRY[kind](cfg)


def mixer_needs_eager(mixer: str) -> bool:
    """Recurrent reference mixers use Python loops; torch.compile gains little
    and graph-breaks a lot, so we keep them eager (matches the repo's
    SKIP_COMPILE-when-SSM rule)."""
    return mixer in ("mamba2", "gdn")
