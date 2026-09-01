"""A CPU mirror of ``flash_attn_cuda.cu``'s index arithmetic.

Why this exists: the kernel cannot be run without an NVIDIA GPU, and suite 19's
lesson was that *a custom kernel is worthless until verified against a
brute-force reference*. This module walks the same blocks, tiles and key
sub-tiles the kernel walks, with the same bounds and the same online-softmax
recurrence, in plain torch -- so the parts that are easiest to get wrong and
hardest to eyeball (causal cut per row, tile counts, the sub-tile rescale, the
log2/natural-log conversion on the tape, the GQA head map, the backward's
``j_lo``) can be checked on any machine.

It is a mirror, not a second implementation: **if the .cu changes, change this
too.** It deliberately does not model anything about performance -- no
occupancy, no memory system, no launch geometry beyond the tile shapes.

Accumulation follows the kernel's order per row (sub-tile by sub-tile, key by
key within a sub-tile) but is vectorised across the rows of a block, which do
not interact.
"""

from __future__ import annotations

import math

import torch

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
KSUB = 8      # keys per online-softmax rescale -- flash_attn_cuda.cu's KSUB


def tile_shape(head_dim: int, itemsize: int) -> tuple[int, int]:
    """``(BR, BC)`` for a (head_dim, dtype) pair -- mirrors ``struct Tiles``."""
    raw = 8192 // (2 * head_dim * itemsize)
    bc = 32 if raw < 32 else (64 if raw > 64 else (raw // KSUB) * KSUB)
    return 64, bc


def simulate_fwd(q, k, v, scale, itemsize: int | None = None):
    """Mirror of ``flash_fwd_kernel``. Returns ``(O [B,T,H,D], L [B,H,T])``.

    ``itemsize`` overrides the dtype the *tile geometry* is chosen for, so a
    float64 run (tight tolerances) can still exercise the BC the bf16 kernel
    will use. It does not change the arithmetic precision.
    """
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    group = H // Hkv
    BR, BC = tile_shape(D, itemsize or q.element_size())
    qk_scale = scale * LOG2E

    o = torch.zeros_like(q)
    lse = torch.zeros(B, H, T, dtype=q.dtype)
    neg_inf = torch.tensor(-math.inf, dtype=q.dtype)

    for b in range(B):
        for h in range(H):
            hkv = h // group
            for t_q0 in range(0, T, BR):                      # blockIdx.x
                rows = torch.arange(t_q0, min(t_q0 + BR, T))  # threads, row_valid
                q_reg = q[b, rows, h]                         # [R, D]
                acc = torch.zeros(len(rows), D, dtype=q.dtype)
                m_i = torch.full((len(rows),), -math.inf, dtype=q.dtype)
                l_i = torch.zeros(len(rows), dtype=q.dtype)

                t_q_max = min(t_q0 + BR, T) - 1
                n_k_blocks = (t_q_max // BC) + 1
                for kb in range(n_k_blocks):
                    t_k0 = kb * BC
                    n_k = min(BC, T - t_k0)
                    # staged tile, zero-filled past the tail exactly as the
                    # kernel does (phase 2 multiplies those rows by p == 0).
                    Ks = torch.zeros(BC, D, dtype=q.dtype)
                    Vs = torch.zeros(BC, D, dtype=q.dtype)
                    Ks[:n_k] = k[b, t_k0:t_k0 + n_k, hkv]
                    Vs[:n_k] = v[b, t_k0:t_k0 + n_k, hkv]

                    k_hi = torch.clamp(rows - t_k0 + 1, max=n_k)     # per row
                    for sub in range(0, BC, KSUB):
                        active = sub < k_hi                          # [R]
                        if not bool(active.any()):
                            continue
                        s = torch.full((len(rows), KSUB), -math.inf, dtype=q.dtype)
                        for j in range(KSUB):
                            live = (sub + j) < k_hi
                            dot = q_reg @ Ks[sub + j]
                            s[:, j] = torch.where(live, dot * qk_scale, neg_inf)
                        m_sub = s.max(dim=1).values
                        m_new = torch.where(active, torch.maximum(m_i, m_sub), m_i)
                        alpha = torch.where(active, torch.exp2(m_i - m_new),
                                            torch.ones_like(m_i))
                        l_i = l_i * alpha
                        acc = acc * alpha[:, None]
                        for j in range(KSUB):
                            p = torch.exp2(s[:, j] - m_new)
                            l_i = l_i + p
                            acc = acc + p[:, None] * Vs[sub + j]
                        m_i = m_new

                o[b, rows, h] = acc / l_i[:, None]
                lse[b, h, rows] = m_i * LN2 + torch.log(l_i)
    return o, lse


def simulate_delta(o, do):
    """Mirror of ``flash_delta_kernel``: ``Delta[b,h,t] = sum_d dO*O``."""
    return (do * o).sum(dim=-1).permute(0, 2, 1).contiguous()   # [B,T,H] -> [B,H,T]


def simulate_bwd(q, k, v, o, do, lse, scale, itemsize: int | None = None):
    """Mirror of ``flash_bwd_dq_kernel`` + ``flash_bwd_dkv_kernel``.

    Returns ``(dQ, dK, dV)`` in the input layouts. Both passes recompute the
    scores from the taped LSE, which is what makes them independent of each
    other -- and what makes the pair deterministic without atomics.
    """
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    group = H // Hkv
    BR, BC = tile_shape(D, itemsize or q.element_size())
    BQ = BC
    qk_scale = scale * LOG2E
    delta = simulate_delta(o, do)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # ---- dQ: block per (q_block, b*h), one thread per query row ----
    for b in range(B):
        for h in range(H):
            hkv = h // group
            for t_q0 in range(0, T, BR):
                rows = torch.arange(t_q0, min(t_q0 + BR, T))
                q_reg = q[b, rows, h]
                do_reg = do[b, rows, h]
                Li2 = lse[b, h, rows] * LOG2E
                Di = delta[b, h, rows]
                acc = torch.zeros(len(rows), D, dtype=q.dtype)

                t_q_max = min(t_q0 + BR, T) - 1
                for kb in range((t_q_max // BC) + 1):
                    t_k0 = kb * BC
                    n_k = min(BC, T - t_k0)
                    Ks = k[b, t_k0:t_k0 + n_k, hkv]
                    Vs = v[b, t_k0:t_k0 + n_k, hkv]
                    k_hi = torch.clamp(rows - t_k0 + 1, max=n_k)
                    for tk in range(n_k):
                        live = tk < k_hi
                        score = q_reg @ Ks[tk]
                        dp = do_reg @ Vs[tk]
                        p = torch.exp2(score * qk_scale - Li2)
                        dss = torch.where(live, p * (dp - Di) * scale,
                                          torch.zeros_like(p))
                        acc = acc + dss[:, None] * Ks[tk]
                dq[b, rows, h] = acc

    # ---- dK/dV: block per (k_block, b*hkv), one thread per key row ----
    for b in range(B):
        for hkv in range(Hkv):
            for t_k0 in range(0, T, BC):
                keys = torch.arange(t_k0, min(t_k0 + BC, T))
                k_reg = k[b, keys, hkv]
                v_reg = v[b, keys, hkv]
                acc_k = torch.zeros(len(keys), D, dtype=q.dtype)
                acc_v = torch.zeros(len(keys), D, dtype=q.dtype)

                for g in range(group):
                    h = hkv * group + g
                    for qb in range(t_k0 // BQ, (T + BQ - 1) // BQ):
                        t_q0 = qb * BQ
                        n_q = min(BQ, T - t_q0)
                        Qs = q[b, t_q0:t_q0 + n_q, h]
                        dOs = do[b, t_q0:t_q0 + n_q, h]
                        Ls = lse[b, h, t_q0:t_q0 + n_q] * LOG2E
                        Ds = delta[b, h, t_q0:t_q0 + n_q]
                        j_lo = torch.clamp(keys - t_q0, min=0)     # per key row
                        for j in range(n_q):
                            live = j >= j_lo
                            score = k_reg @ Qs[j]
                            dp = v_reg @ dOs[j]
                            p = torch.exp2(score * qk_scale - Ls[j])
                            p = torch.where(live, p, torch.zeros_like(p))
                            dss = p * (dp - Ds[j]) * scale
                            acc_k = acc_k + dss[:, None] * Qs[j]
                            acc_v = acc_v + p[:, None] * dOs[j]
                dk[b, keys, hkv] = acc_k
                dv[b, keys, hkv] = acc_v

    return dq, dk, dv


def reference_attention(q, k, v, scale):
    """Brute-force causal GQA attention on [B,T,H,D]. Returns ``(O, LSE)``.

    Deliberately the dumbest correct thing: materialise the full [T,T] score
    matrix, mask it, softmax it. This is what the kernel is checked against.
    """
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    group = H // Hkv
    kx = k.repeat_interleave(group, dim=2)
    vx = v.repeat_interleave(group, dim=2)
    qs = q.permute(0, 2, 1, 3)                       # [B,H,T,D]
    ks = kx.permute(0, 2, 1, 3)
    vs = vx.permute(0, 2, 1, 3)
    scores = (qs @ ks.transpose(-1, -2)) * scale     # [B,H,T,T]
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, -math.inf)
    lse = torch.logsumexp(scores, dim=-1)            # [B,H,T]
    o = torch.softmax(scores, dim=-1) @ vs           # [B,H,T,D]
    return o.permute(0, 2, 1, 3).contiguous(), lse.contiguous()


# ---------------------------------------------------------------------------
# tensor-core (mma.sync) forward
# ---------------------------------------------------------------------------
#
# Mirror of ``flash_fwd_mma_kernel``, at lane granularity: 32 lanes per warp,
# each holding exactly the fragment elements the PTX layout gives it. An mma is
# modelled by reconstructing A and B from the fragments, multiplying, and
# scattering the result back per the C layout.
#
# What this verifies and what it does not, stated plainly. It verifies that the
# kernel *uses* the layout self-consistently -- the K and V^T addressing, the
# S -> P repack, which lanes hold which rows, the cross-lane row reductions, the
# per-element causal mask, the accumulator rescale, the epilogue. It cannot
# verify that the layout table itself matches the hardware, because both this
# file and the kernel read the table from the same place. That one fact is
# checked by ``mma_probe`` on a real device, which is why that exists.
#
# Precision is not modelled either: the kernel rounds the P fragment to
# f16/bf16, this runs in the caller's dtype. Run it in float64 and what is left
# is the algorithm.

MMA_BR = 64      # query rows per block   (flash_attn_cuda.cu: MmaTiles::BR)
MMA_BC = 32      # key tile               (MmaTiles::BC)
MMA_BQ = 32      # query tile in dK/dV    (MmaTiles::BQ)
MMA_WARPS = 4    # warps per block        (MmaTiles::kWarps)

_LANE = torch.arange(32)
_GID = _LANE >> 2          # 0..7, the two rows a lane owns: gid and gid+8
_TIG = _LANE & 3           # 0..3, which column pair within the group


def _frag_to_a(a):
    """[32,4,2] A-operand registers -> the 16x16 matrix they encode."""
    A = torch.zeros(16, 16, dtype=a.dtype)
    A[_GID, 2 * _TIG] = a[:, 0, 0]
    A[_GID, 2 * _TIG + 1] = a[:, 0, 1]
    A[_GID + 8, 2 * _TIG] = a[:, 1, 0]
    A[_GID + 8, 2 * _TIG + 1] = a[:, 1, 1]
    A[_GID, 2 * _TIG + 8] = a[:, 2, 0]
    A[_GID, 2 * _TIG + 9] = a[:, 2, 1]
    A[_GID + 8, 2 * _TIG + 8] = a[:, 3, 0]
    A[_GID + 8, 2 * _TIG + 9] = a[:, 3, 1]
    return A


def _frag_to_b(bf):
    """[32,2,2] B-operand registers -> the 16x8 matrix they encode."""
    Bm = torch.zeros(16, 8, dtype=bf.dtype)
    Bm[2 * _TIG, _GID] = bf[:, 0, 0]
    Bm[2 * _TIG + 1, _GID] = bf[:, 0, 1]
    Bm[2 * _TIG + 8, _GID] = bf[:, 1, 0]
    Bm[2 * _TIG + 9, _GID] = bf[:, 1, 1]
    return Bm


def mma_m16n8k16(a, bf, c):
    """c[32,4] += A @ B, with every operand in its per-lane fragment form."""
    D = _frag_to_a(a) @ _frag_to_b(bf)
    c[:, 0] += D[_GID, 2 * _TIG]
    c[:, 1] += D[_GID, 2 * _TIG + 1]
    c[:, 2] += D[_GID + 8, 2 * _TIG]
    c[:, 3] += D[_GID + 8, 2 * _TIG + 1]
    return c


def _group_reduce(x, op):
    """The two ``__shfl_xor_sync`` steps (masks 1 and 2): reduce over the 4
    lanes of each group, which hold different columns of the same two rows."""
    v = x.view(8, 4)
    r = v.amax(dim=1) if op == "max" else v.sum(dim=1)
    return r[:, None].expand(8, 4).reshape(32).clone()


def _load_b(mat, ntile, k0, dt):
    """Mirror of ``load_b``: the n index picks a row of the staged matrix, the
    k index runs along it, and each half is a contiguous pair."""
    row = ntile * 8 + _GID
    bf = torch.zeros(32, 2, 2, dtype=dt)
    bf[:, 0, 0] = mat[row, k0 + 2 * _TIG]
    bf[:, 0, 1] = mat[row, k0 + 2 * _TIG + 1]
    bf[:, 1, 0] = mat[row, k0 + 2 * _TIG + 8]
    bf[:, 1, 1] = mat[row, k0 + 2 * _TIG + 9]
    return bf


def _load_b_strided(mat, ntile, k0, dt):
    """Mirror of ``load_b_strided``: the same matrix read down a column, so a
    register's two elements are a row apart rather than adjacent. This is what
    replaced the transposed shared-memory copies."""
    col = ntile * 8 + _GID
    bf = torch.zeros(32, 2, 2, dtype=dt)
    bf[:, 0, 0] = mat[k0 + 2 * _TIG, col]
    bf[:, 0, 1] = mat[k0 + 2 * _TIG + 1, col]
    bf[:, 1, 0] = mat[k0 + 2 * _TIG + 8, col]
    bf[:, 1, 1] = mat[k0 + 2 * _TIG + 9, col]
    return bf


def _stage(src, t0, n_live, tile):
    """Mirror of ``stage_tile``: rows past the tail clamp to the last live row
    rather than zero-filling, because cp.async copies bytes and cannot
    synthesise zeros. Everything downstream masks those columns anyway -- that
    the mirror still matches the reference is the check that it really does."""
    idx = torch.arange(tile).clamp(max=n_live - 1) + t0
    return src[idx]


def _load_a_rows(mat, ra, rb, ok_a, ok_b, kchunks, T, dt):
    """Mirror of ``load_a_rows``: two [T,D] rows per lane, over all k-chunks."""
    a = torch.zeros(32, kchunks, 4, 2, dtype=dt)
    ma = mat[ra.clamp(max=T - 1)]
    mb = mat[rb.clamp(max=T - 1)]
    z = torch.zeros(32, dtype=dt)
    for kc in range(kchunks):
        c0 = kc * 16 + 2 * _TIG
        a[:, kc, 0, 0] = torch.where(ok_a, ma[_LANE, c0], z)
        a[:, kc, 0, 1] = torch.where(ok_a, ma[_LANE, c0 + 1], z)
        a[:, kc, 1, 0] = torch.where(ok_b, mb[_LANE, c0], z)
        a[:, kc, 1, 1] = torch.where(ok_b, mb[_LANE, c0 + 1], z)
        a[:, kc, 2, 0] = torch.where(ok_a, ma[_LANE, c0 + 8], z)
        a[:, kc, 2, 1] = torch.where(ok_a, ma[_LANE, c0 + 9], z)
        a[:, kc, 3, 0] = torch.where(ok_b, mb[_LANE, c0 + 8], z)
        a[:, kc, 3, 1] = torch.where(ok_b, mb[_LANE, c0 + 9], z)
    return a


def _pack_a_from_acc(lo, hi, dt):
    """Mirror of ``pack_a_from_acc``: two 16x8 accumulators -> one 16x16 A."""
    a = torch.zeros(32, 4, 2, dtype=dt)
    a[:, 0, 0], a[:, 0, 1] = lo[:, 0], lo[:, 1]
    a[:, 1, 0], a[:, 1, 1] = lo[:, 2], lo[:, 3]
    a[:, 2, 0], a[:, 2, 1] = hi[:, 0], hi[:, 1]
    a[:, 3, 0], a[:, 3, 1] = hi[:, 2], hi[:, 3]
    return a


def simulate_fwd_mma(q, k, v, scale):
    """Mirror of ``flash_fwd_mma_kernel``. Returns ``(O [B,T,H,D], L [B,H,T])``."""
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    group = H // Hkv
    assert D % 16 == 0, "head_dim must be a whole number of mma k-chunks"
    KCHUNKS, NS, NO, PCHUNKS = D // 16, MMA_BC // 8, D // 8, MMA_BC // 16
    qk_scale = scale * LOG2E
    dt = q.dtype

    o = torch.zeros_like(q)
    lse = torch.zeros(B, H, T, dtype=dt)

    for b in range(B):
        for h in range(H):
            hkv = h // group
            for t_q0 in range(0, T, MMA_BR):                  # blockIdx.x
                for wid in range(MMA_WARPS):                  # one warp, 16 rows
                    q_base = t_q0 + wid * 16
                    if q_base >= T:
                        continue          # the kernel runs it and stores nothing
                    ra = q_base + _GID
                    rb = q_base + _GID + 8
                    oka, okb = ra < T, rb < T

                    qf = _load_a_rows(q[b, :, h], ra, rb, oka, okb,
                                      KCHUNKS, T, dt)

                    acc_o = torch.zeros(32, NO, 4, dtype=dt)
                    m = torch.full((32, 2), -math.inf, dtype=dt)
                    l = torch.zeros(32, 2, dtype=dt)

                    t_q_max = min(t_q0 + MMA_BR, T) - 1
                    for kb in range((t_q_max // MMA_BC) + 1):
                        t_k0 = kb * MMA_BC
                        n_k = min(MMA_BC, T - t_k0)
                        Ks = _stage(k[b, :, hkv], t_k0, n_k, MMA_BC)
                        Vs = _stage(v[b, :, hkv], t_k0, n_k, MMA_BC)

                        # ---- S = Q K^T ----
                        s_acc = torch.zeros(32, NS, 4, dtype=dt)
                        for n in range(NS):
                            for kc in range(KCHUNKS):
                                kf = _load_b(Ks, n, kc * 16, dt)
                                mma_m16n8k16(qf[:, kc], kf, s_acc[:, n])

                        # ---- mask + online softmax, in registers ----
                        m_sub = torch.full((32, 2), -math.inf, dtype=dt)
                        for n in range(NS):
                            col0 = t_k0 + n * 8 + 2 * _TIG
                            for e in range(4):
                                row = ra if e < 2 else rb
                                col = col0 + (e & 1)
                                keep = (col < T) & (col <= row)
                                s_acc[:, n, e] = torch.where(
                                    keep, s_acc[:, n, e] * qk_scale,
                                    torch.full((32,), -math.inf, dtype=dt))
                            m_sub[:, 0] = torch.maximum(
                                m_sub[:, 0], torch.maximum(s_acc[:, n, 0], s_acc[:, n, 1]))
                            m_sub[:, 1] = torch.maximum(
                                m_sub[:, 1], torch.maximum(s_acc[:, n, 2], s_acc[:, n, 3]))
                        m_sub[:, 0] = _group_reduce(m_sub[:, 0], "max")
                        m_sub[:, 1] = _group_reduce(m_sub[:, 1], "max")

                        m_new = torch.maximum(m, m_sub)
                        alpha = torch.where(m_new == -math.inf,
                                            torch.zeros_like(m_new),
                                            torch.exp2(m - m_new))

                        s = torch.zeros(32, 2, dtype=dt)
                        for n in range(NS):
                            s_acc[:, n, 0] = torch.exp2(s_acc[:, n, 0] - m_new[:, 0])
                            s_acc[:, n, 1] = torch.exp2(s_acc[:, n, 1] - m_new[:, 0])
                            s_acc[:, n, 2] = torch.exp2(s_acc[:, n, 2] - m_new[:, 1])
                            s_acc[:, n, 3] = torch.exp2(s_acc[:, n, 3] - m_new[:, 1])
                            s[:, 0] += s_acc[:, n, 0] + s_acc[:, n, 1]
                            s[:, 1] += s_acc[:, n, 2] + s_acc[:, n, 3]
                        s[:, 0] = _group_reduce(s[:, 0], "sum")
                        s[:, 1] = _group_reduce(s[:, 1], "sum")
                        l = l * alpha + s
                        m = m_new

                        acc_o[:, :, 0] *= alpha[:, 0:1]
                        acc_o[:, :, 1] *= alpha[:, 0:1]
                        acc_o[:, :, 2] *= alpha[:, 1:2]
                        acc_o[:, :, 3] *= alpha[:, 1:2]

                        # ---- O += P V ----
                        for pc in range(PCHUNKS):
                            pf = _pack_a_from_acc(s_acc[:, pc * 2],
                                                  s_acc[:, pc * 2 + 1], dt)
                            for n in range(NO):
                                vf = _load_b_strided(Vs, n, pc * 16, dt)
                                mma_m16n8k16(pf, vf, acc_o[:, n])

                    inv = 1.0 / l
                    for n in range(NO):
                        d0 = n * 8 + 2 * _TIG
                        for lane in range(32):
                            if oka[lane]:
                                o[b, ra[lane], h, d0[lane]] = acc_o[lane, n, 0] * inv[lane, 0]
                                o[b, ra[lane], h, d0[lane] + 1] = acc_o[lane, n, 1] * inv[lane, 0]
                            if okb[lane]:
                                o[b, rb[lane], h, d0[lane]] = acc_o[lane, n, 2] * inv[lane, 1]
                                o[b, rb[lane], h, d0[lane] + 1] = acc_o[lane, n, 3] * inv[lane, 1]
                    for lane in range(32):
                        if _TIG[lane] != 0:
                            continue
                        if oka[lane]:
                            lse[b, h, ra[lane]] = m[lane, 0] * LN2 + torch.log(l[lane, 0])
                        if okb[lane]:
                            lse[b, h, rb[lane]] = m[lane, 1] * LN2 + torch.log(l[lane, 1])
    return o, lse


def simulate_bwd_mma(q, k, v, o, do, lse, scale):
    """Mirror of ``flash_bwd_dq_mma_kernel`` + ``flash_bwd_dkv_mma_kernel``.

    The two passes are mirror images of each other and that is the point: the dQ
    pass makes queries the m dimension and indexes the tape by ROW, the dK/dV
    pass makes keys the m dimension -- computing S^T rather than S -- and indexes
    the tape by COLUMN. That second choice is what removes the register
    transpose the backward would otherwise need, and the column-indexed tape is
    the easiest thing in either file to get backwards, so it is worth checking
    against a brute-force gradient rather than reading twice.
    """
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    group = H // Hkv
    assert D % 16 == 0
    dt = q.dtype
    qk_scale = scale * LOG2E
    delta = simulate_delta(o, do)

    KCHUNKS = D // 16
    NO = D // 8
    NS, PCHUNKS = MMA_BC // 8, MMA_BC // 16      # dQ pass: n and k are keys
    NQ, QCHUNKS = MMA_BQ // 8, MMA_BQ // 16      # dK/dV pass: n and k are queries

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # ---- dQ: block per (q_block, b*h), warp owns 16 query rows ----
    for b in range(B):
        for h in range(H):
            hkv = h // group
            for t_q0 in range(0, T, MMA_BR):
                for wid in range(MMA_WARPS):
                    q_base = t_q0 + wid * 16
                    if q_base >= T:
                        continue
                    ra, rb = q_base + _GID, q_base + _GID + 8
                    oka, okb = ra < T, rb < T
                    z = torch.zeros(32, dtype=dt)
                    qf = _load_a_rows(q[b, :, h], ra, rb, oka, okb, KCHUNKS, T, dt)
                    dof = _load_a_rows(do[b, :, h], ra, rb, oka, okb, KCHUNKS, T, dt)
                    Li2 = torch.stack([
                        torch.where(oka, lse[b, h, ra.clamp(max=T - 1)] * LOG2E, z),
                        torch.where(okb, lse[b, h, rb.clamp(max=T - 1)] * LOG2E, z)], 1)
                    Di = torch.stack([
                        torch.where(oka, delta[b, h, ra.clamp(max=T - 1)], z),
                        torch.where(okb, delta[b, h, rb.clamp(max=T - 1)], z)], 1)

                    acc_dq = torch.zeros(32, NO, 4, dtype=dt)
                    t_q_max = min(t_q0 + MMA_BR, T) - 1
                    for kb in range((t_q_max // MMA_BC) + 1):
                        t_k0 = kb * MMA_BC
                        n_k = min(MMA_BC, T - t_k0)
                        Ks = _stage(k[b, :, hkv], t_k0, n_k, MMA_BC)
                        Vs = _stage(v[b, :, hkv], t_k0, n_k, MMA_BC)

                        s_acc = torch.zeros(32, NS, 4, dtype=dt)
                        dp_acc = torch.zeros(32, NS, 4, dtype=dt)
                        for n in range(NS):
                            for kc in range(KCHUNKS):
                                mma_m16n8k16(qf[:, kc], _load_b(Ks, n, kc * 16, dt),
                                             s_acc[:, n])
                                mma_m16n8k16(dof[:, kc], _load_b(Vs, n, kc * 16, dt),
                                             dp_acc[:, n])

                        for n in range(NS):
                            col0 = t_k0 + n * 8 + 2 * _TIG
                            for e in range(4):
                                lo = e < 2
                                row = ra if lo else rb
                                col = col0 + (e & 1)
                                keep = (oka if lo else okb) & (col < T) & (col <= row)
                                p = torch.where(keep,
                                                torch.exp2(s_acc[:, n, e] * qk_scale
                                                           - Li2[:, 0 if lo else 1]), z)
                                s_acc[:, n, e] = p * (dp_acc[:, n, e]
                                                      - Di[:, 0 if lo else 1]) * scale

                        for pc in range(PCHUNKS):
                            dsf = _pack_a_from_acc(s_acc[:, pc * 2],
                                                   s_acc[:, pc * 2 + 1], dt)
                            for n in range(NO):
                                mma_m16n8k16(dsf,
                                             _load_b_strided(Ks, n, pc * 16, dt),
                                             acc_dq[:, n])

                    for n in range(NO):
                        d0 = n * 8 + 2 * _TIG
                        for lane in range(32):
                            if oka[lane]:
                                dq[b, ra[lane], h, d0[lane]] = acc_dq[lane, n, 0]
                                dq[b, ra[lane], h, d0[lane] + 1] = acc_dq[lane, n, 1]
                            if okb[lane]:
                                dq[b, rb[lane], h, d0[lane]] = acc_dq[lane, n, 2]
                                dq[b, rb[lane], h, d0[lane] + 1] = acc_dq[lane, n, 3]

    # ---- dK/dV: block per (k_block, b*hkv), warp owns 16 KEY rows ----
    for b in range(B):
        for hkv in range(Hkv):
            for t_k0 in range(0, T, MMA_BR):
                for wid in range(MMA_WARPS):
                    k_base = t_k0 + wid * 16
                    if k_base >= T:
                        continue
                    ka, kbb = k_base + _GID, k_base + _GID + 8
                    oka, okb = ka < T, kbb < T
                    z = torch.zeros(32, dtype=dt)
                    kf = _load_a_rows(k[b, :, hkv], ka, kbb, oka, okb, KCHUNKS, T, dt)
                    vf = _load_a_rows(v[b, :, hkv], ka, kbb, oka, okb, KCHUNKS, T, dt)

                    acc_dk = torch.zeros(32, NO, 4, dtype=dt)
                    acc_dv = torch.zeros(32, NO, 4, dtype=dt)
                    for g in range(group):
                        h = hkv * group + g
                        for qb in range(t_k0 // MMA_BQ, (T + MMA_BQ - 1) // MMA_BQ):
                            t_q0 = qb * MMA_BQ
                            n_q = min(MMA_BQ, T - t_q0)
                            Qs = _stage(q[b, :, h], t_q0, n_q, MMA_BQ)
                            dOs = _stage(do[b, :, h], t_q0, n_q, MMA_BQ)
                            tape = torch.arange(MMA_BQ).clamp(max=n_q - 1) + t_q0
                            Ls = lse[b, h, tape] * LOG2E
                            Ds = delta[b, h, tape]

                            st = torch.zeros(32, NQ, 4, dtype=dt)
                            dpt = torch.zeros(32, NQ, 4, dtype=dt)
                            for n in range(NQ):
                                for kc in range(KCHUNKS):
                                    mma_m16n8k16(kf[:, kc], _load_b(Qs, n, kc * 16, dt),
                                                 st[:, n])
                                    mma_m16n8k16(vf[:, kc], _load_b(dOs, n, kc * 16, dt),
                                                 dpt[:, n])

                            p_t = torch.zeros(32, NQ, 4, dtype=dt)
                            for n in range(NQ):
                                qcol = n * 8 + 2 * _TIG      # index WITHIN the tile
                                for e in range(4):
                                    lo = e < 2
                                    row = ka if lo else kbb          # the key
                                    cl = qcol + (e & 1)
                                    col = t_q0 + cl                  # the query
                                    keep = (oka if lo else okb) & (col < T) & (col >= row)
                                    p = torch.where(
                                        keep,
                                        torch.exp2(st[:, n, e] * qk_scale - Ls[cl]), z)
                                    p_t[:, n, e] = p
                                    st[:, n, e] = p * (dpt[:, n, e] - Ds[cl]) * scale

                            for pc in range(QCHUNKS):
                                dsf = _pack_a_from_acc(st[:, pc * 2], st[:, pc * 2 + 1], dt)
                                pf = _pack_a_from_acc(p_t[:, pc * 2], p_t[:, pc * 2 + 1], dt)
                                for n in range(NO):
                                    mma_m16n8k16(
                                        dsf, _load_b_strided(Qs, n, pc * 16, dt),
                                        acc_dk[:, n])
                                    mma_m16n8k16(
                                        pf, _load_b_strided(dOs, n, pc * 16, dt),
                                        acc_dv[:, n])

                    for n in range(NO):
                        d0 = n * 8 + 2 * _TIG
                        for lane in range(32):
                            if oka[lane]:
                                dk[b, ka[lane], hkv, d0[lane]] = acc_dk[lane, n, 0]
                                dk[b, ka[lane], hkv, d0[lane] + 1] = acc_dk[lane, n, 1]
                                dv[b, ka[lane], hkv, d0[lane]] = acc_dv[lane, n, 0]
                                dv[b, ka[lane], hkv, d0[lane] + 1] = acc_dv[lane, n, 1]
                            if okb[lane]:
                                dk[b, kbb[lane], hkv, d0[lane]] = acc_dk[lane, n, 2]
                                dk[b, kbb[lane], hkv, d0[lane] + 1] = acc_dk[lane, n, 3]
                                dv[b, kbb[lane], hkv, d0[lane]] = acc_dv[lane, n, 2]
                                dv[b, kbb[lane], hkv, d0[lane] + 1] = acc_dv[lane, n, 3]

    return dq, dk, dv
