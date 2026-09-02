// FlashDecoding: single-query attention, split over the KV sequence.
//
// The general FA-2 kernels tile over query rows: the grid is
// `ceil(Tq/BR) x B*H` and only `lid < BR` lanes are row-valid. At Tq=1 that is
// one live lane in 32, and the grid collapses to B*H threadgroups -- 8 for an
// 8-head decode. Measured against MLX on an M5 Pro, `global512_decode_4k` ran
// 271x slower, and the slowdown tracked threadgroup count almost monotonically
// (8 -> 271x, 16 -> 47x, 32 -> 27x, 256 -> 10x). The kernels' own comment said
// "decode Tq=1 wastes lanes but FA is tiny vs GEMV"; the benchmark disagreed.
//
// This splits the other way. One simdgroup owns one KV chunk, so the grid is
// `n_chunks x B*H` and every lane is live:
//
//   * lane L owns head dims L, L+32, L+64, ... -- DPL = HEAD_DIM/32 of them,
//     held in registers, so the K/V reads are coalesced across the simdgroup
//     and the P@V accumulate needs no cross-lane communication at all;
//   * a score is one `simd_sum` of the per-lane partial dot, which every lane
//     then holds, so the online-softmax state (m, l) is uniform and the whole
//     kernel is branch-divergence free;
//   * each chunk emits (m, l, acc[D]) and a second pass combines them with the
//     standard rescale, which is exact rather than an approximation.
//
// Masking is transcribed from the general kernels: q_abs = q_pos_offset + t_q,
// k_abs = kv_pos_offset + t_k, keep k_abs <= q_abs, and additionally
// k_abs >= max(0, q_abs - window + 1) when `window` is nonzero. `window == 0`
// is the global rule. A query with nothing unmasked yields zeros, not NaN.
#include <metal_stdlib>
using namespace metal;

constant uint SIMD_W = 32;

// Keys per chunk. Larger amortises the partial write; smaller buys grid
// parallelism, which is the entire point of this kernel. 256 keeps a 4096-key
// history at 16 chunks per head.
constant uint KV_CHUNK = 256;

/// Partial pass: one simdgroup, one (batch, head, kv-chunk).
///
/// `Tkv` lives on the device, so the host cannot size the grid to the live
/// chunk count and dispatches for the K buffer's capacity instead. Chunks past
/// `Tkv` return immediately; the scratch they would have written is zeroed by
/// the host so the reduce pass cannot read a stale partial from a previous
/// dispatch and treat it as this one's.
#define DECODE_PARTIAL_KERNEL(NAME, D)                                        \
kernel void NAME(                                                             \
    device const float *Q [[buffer(0)]],                                      \
    device const float *K [[buffer(1)]],                                      \
    device const float *V [[buffer(2)]],                                      \
    device float *partials [[buffer(3)]],                                     \
    constant uint &B [[buffer(4)]],                                           \
    device const uint *Tkv_ptr [[buffer(6)]],                                 \
    constant uint &H [[buffer(7)]],                                           \
    constant uint &Hkv [[buffer(8)]],                                         \
    constant uint &window [[buffer(9)]],                                      \
    constant float &scale [[buffer(10)]],                                     \
    device const uint *q_pos_offset_ptr [[buffer(11)]],                       \
    device const uint *kv_pos_offset_ptr [[buffer(12)]],                      \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint2 tpitg [[thread_position_in_threadgroup]])                           \
{                                                                             \
    constexpr uint DPL = D / SIMD_W;                                          \
    const uint lid = tpitg.x;                                                 \
    const uint Tkv = *Tkv_ptr;                                                \
    const uint chunk = tgpig.x;                                               \
    const uint bh = tgpig.y;                                                  \
    const uint h = bh % H;                                                    \
    const uint b = bh / H;                                                    \
    const uint t_k0 = chunk * KV_CHUNK;                                       \
    if (t_k0 >= Tkv) { return; }                                              \
    const uint n_k = min(KV_CHUNK, Tkv - t_k0);                               \
    const uint group = max(H / Hkv, 1u);                                      \
    const uint hkv = h / group;                                               \
                                                                              \
    const int q_abs = (int)(*q_pos_offset_ptr);                               \
    const uint kv_pos_offset = *kv_pos_offset_ptr;                            \
    const int k_lo = (window == 0u) ? 0 : max(0, q_abs - (int)window + 1);    \
                                                                              \
    /* Tq == 1, so the single query row is index 0. */                        \
    const uint q_off = (b * H + h) * D;                                       \
    float q_reg[DPL];                                                         \
    for (uint j = 0; j < DPL; ++j) { q_reg[j] = Q[q_off + lid + j * SIMD_W]; }\
                                                                              \
    float acc[DPL];                                                           \
    for (uint j = 0; j < DPL; ++j) { acc[j] = 0.0f; }                         \
    float m_i = -INFINITY;                                                    \
    float l_i = 0.0f;                                                         \
                                                                              \
    for (uint t = 0; t < n_k; ++t) {                                          \
        const int k_abs = (int)(kv_pos_offset + t_k0 + t);                    \
        if (k_abs > q_abs || k_abs < k_lo) { continue; }                      \
        const uint kv_off = ((b * Tkv + (t_k0 + t)) * Hkv + hkv) * D;         \
        float part = 0.0f;                                                    \
        for (uint j = 0; j < DPL; ++j) {                                      \
            part += q_reg[j] * K[kv_off + lid + j * SIMD_W];                  \
        }                                                                     \
        /* Every lane receives the full dot, so m/l stay uniform. */          \
        const float s = simd_sum(part) * scale;                               \
        const float m_new = max(m_i, s);                                      \
        /* m_i == -inf means nothing has been accumulated yet, so the         \
           rescale is exactly zero. Computing exp(-inf - -inf) here would be  \
           exp(NaN) and would poison the row. */                              \
        const float alpha = (m_i == -INFINITY) ? 0.0f : exp(m_i - m_new);     \
        const float p = exp(s - m_new);                                       \
        for (uint j = 0; j < DPL; ++j) {                                      \
            acc[j] = acc[j] * alpha + p * V[kv_off + lid + j * SIMD_W];       \
        }                                                                     \
        l_i = l_i * alpha + p;                                                \
        m_i = m_new;                                                          \
    }                                                                         \
                                                                              \
    const uint n_chunks_max = tgpig.x + 1u;                                   \
    (void)n_chunks_max;                                                       \
    /* Layout per (bh, chunk): [m, l, acc[D]]. */                             \
    const uint stride = D + 2u;                                              \
    const uint base = (bh * ((Tkv + KV_CHUNK - 1u) / KV_CHUNK) + chunk) * stride; \
    if (lid == 0u) { partials[base] = m_i; partials[base + 1u] = l_i; }        \
    for (uint j = 0; j < DPL; ++j) {                                          \
        partials[base + 2u + lid + j * SIMD_W] = acc[j];                      \
    }                                                                         \
    (void)B;                                                                  \
}

/// Reduce pass: combine every chunk's (m, l, acc) for one (batch, head).
#define DECODE_REDUCE_KERNEL(NAME, D)                                         \
kernel void NAME(                                                             \
    device const float *partials [[buffer(0)]],                               \
    device float *O [[buffer(1)]],                                            \
    constant uint &B [[buffer(2)]],                                           \
    device const uint *Tkv_ptr [[buffer(3)]],                                 \
    constant uint &H [[buffer(4)]],                                           \
    constant uint &out_bf16 [[buffer(5)]],                                    \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint2 tpitg [[thread_position_in_threadgroup]])                           \
{                                                                             \
    constexpr uint DPL = D / SIMD_W;                                          \
    const uint lid = tpitg.x;                                                 \
    const uint Tkv = *Tkv_ptr;                                                \
    const uint bh = tgpig.y;                                                  \
    const uint h = bh % H;                                                    \
    const uint b = bh / H;                                                    \
    const uint n_chunks = (Tkv + KV_CHUNK - 1u) / KV_CHUNK;                   \
    const uint stride = D + 2u;                                              \
                                                                              \
    float m_all = -INFINITY;                                                  \
    for (uint c = 0; c < n_chunks; ++c) {                                     \
        m_all = max(m_all, partials[(bh * n_chunks + c) * stride]);           \
    }                                                                         \
    float acc[DPL];                                                           \
    for (uint j = 0; j < DPL; ++j) { acc[j] = 0.0f; }                         \
    float l_all = 0.0f;                                                       \
    /* Every chunk masked: the general kernels emit zeros rather than NaN,    \
       and exp(-inf - -inf) below would be NaN, so this is both the matching  \
       behaviour and the safe one. */                                         \
    if (m_all != -INFINITY) {                                                 \
        for (uint c = 0; c < n_chunks; ++c) {                                 \
            const uint base = (bh * n_chunks + c) * stride;                   \
            const float m_c = partials[base];                                 \
            if (m_c == -INFINITY) { continue; }                               \
            const float w = exp(m_c - m_all);                                 \
            l_all += partials[base + 1u] * w;                                 \
            for (uint j = 0; j < DPL; ++j) {                                  \
                acc[j] += partials[base + 2u + lid + j * SIMD_W] * w;         \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    const float inv_l = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;               \
    const uint o_off = (b * H + h) * D;                                       \
    if (out_bf16 != 0u) {                                                     \
        device bfloat *Ob = (device bfloat *)O;                               \
        for (uint j = 0; j < DPL; ++j) {                                      \
            Ob[o_off + lid + j * SIMD_W] = bfloat(acc[j] * inv_l);            \
        }                                                                     \
    } else {                                                                  \
        for (uint j = 0; j < DPL; ++j) {                                      \
            O[o_off + lid + j * SIMD_W] = acc[j] * inv_l;                     \
        }                                                                     \
    }                                                                         \
    (void)B;                                                                  \
}

DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h128, 128)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h256, 256)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h512, 512)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h128, 128)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h256, 256)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h512, 512)
