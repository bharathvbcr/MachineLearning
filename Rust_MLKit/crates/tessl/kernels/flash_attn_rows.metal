// Prefill FlashAttention: one simdgroup per query row.
//
// The general FA-2 kernels tile BR query rows into a 32-thread threadgroup and
// guard the inner loops with `row_valid = lid < BR`, so **8 lanes in 32 do the
// arithmetic** and the other 24 only help stage tiles. Measured on an M5 Pro,
// `swa128_prefill_4096` reached 241 GFLOP/s -- 3.7% of the same machine's f32
// GEMM peak -- against MLX at 998 GFLOP/s. Occupancy, not bandwidth.
//
// This inverts the mapping: a simdgroup owns one query row, and lane L owns
// head dims L, L+32, L+64, ... The consequences are what make it fast:
//
//   * every lane is live, and the K/V reads are one coalesced 128-byte line
//     per simdgroup step;
//   * the P@V accumulate needs no cross-lane communication -- each lane owns
//     its own slice of the output row, in registers, so there is no `Oacc`
//     threadgroup array and no barrier around it;
//   * a score is one `simd_sum`, which every lane then holds, so the
//     online-softmax state is uniform and the kernel is divergence free;
//   * because a simdgroup owns *one* row rather than a BR tile, its key range
//     is that row's exact `[max(0, q_abs-window+1), q_abs]`. The tiled kernels
//     had to take the union window over their BR rows and then mask inside it,
//     so they iterated key blocks that were fully masked for most of the tile.
//     Here masked keys are never visited at all.
//
// Same masking rule as everywhere else: q_abs = q_pos_offset + t_q,
// k_abs = kv_pos_offset + t_k, keep k_abs <= q_abs, and additionally
// k_abs >= max(0, q_abs - window + 1) when `window` is nonzero. `window == 0`
// is the global rule. A row with nothing unmasked is zeros, not NaN.
#include <metal_stdlib>
using namespace metal;

constant uint SG_W = 32;
// Simdgroups per threadgroup, so 8 query rows per threadgroup at 256 threads.
// They share the K/V lines they walk, which is why the rows are grouped rather
// than dispatched one per threadgroup.
constant uint SG_PER_TG = 8;

#define ROWS_KERNEL(NAME, D)                                                  \
kernel void NAME(                                                             \
    device const float *Q [[buffer(0)]],                                      \
    device const float *K [[buffer(1)]],                                      \
    device const float *V [[buffer(2)]],                                      \
    device float *O [[buffer(3)]],                                            \
    constant uint &B [[buffer(4)]],                                           \
    constant uint &Tq [[buffer(5)]],                                          \
    device const uint *Tkv_ptr [[buffer(6)]],                                 \
    constant uint &H [[buffer(7)]],                                           \
    constant uint &Hkv [[buffer(8)]],                                         \
    constant uint &window [[buffer(9)]],                                      \
    constant float &scale [[buffer(10)]],                                     \
    device const uint *q_pos_offset_ptr [[buffer(11)]],                       \
    device const uint *kv_pos_offset_ptr [[buffer(12)]],                      \
    constant uint &out_bf16 [[buffer(13)]],                                   \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint2 tpitg [[thread_position_in_threadgroup]])                           \
{                                                                             \
    constexpr uint DPL = D / SG_W;                                            \
    const uint tid = tpitg.x;                                                 \
    const uint sg = tid / SG_W;                                               \
    const uint lane = tid % SG_W;                                             \
                                                                              \
    const uint Tkv = *Tkv_ptr;                                                \
    const uint t_q = tgpig.x * SG_PER_TG + sg;                                \
    const uint bh = tgpig.y;                                                  \
    const uint h = bh % H;                                                    \
    const uint b = bh / H;                                                    \
    /* Uniform across the simdgroup, so the `simd_sum` below is never          \
       executed by a partially exited simdgroup. */                           \
    if (t_q >= Tq) { return; }                                                \
                                                                              \
    const uint group = max(H / Hkv, 1u);                                      \
    const uint hkv = h / group;                                               \
    const int q_abs = (int)(*q_pos_offset_ptr + t_q);                         \
    const int kv_off = (int)(*kv_pos_offset_ptr);                             \
    const int k_lo = (window == 0u) ? 0 : max(0, q_abs - (int)window + 1);    \
                                                                              \
    /* Absolute positions to indices into this dispatch's K/V. Signed          \
       throughout: kv_pos_offset can exceed q_abs, which is an ordinary        \
       fully-masked decode state and must not wrap into a huge unsigned        \
       loop bound. */                                                         \
    const int t_start_i = max(0, k_lo - kv_off);                              \
    const int t_end_i = min((int)Tkv, q_abs - kv_off + 1);                    \
                                                                              \
    const uint o_off = ((b * Tq + t_q) * H + h) * D;                          \
    if (t_start_i >= t_end_i) {                                               \
        if (out_bf16 != 0u) {                                                 \
            device bfloat *Ob = (device bfloat *)O;                           \
            for (uint j = 0; j < DPL; ++j) {                                  \
                Ob[o_off + lane + j * SG_W] = bfloat(0.0f);                   \
            }                                                                 \
        } else {                                                              \
            for (uint j = 0; j < DPL; ++j) {                                  \
                O[o_off + lane + j * SG_W] = 0.0f;                            \
            }                                                                 \
        }                                                                     \
        return;                                                               \
    }                                                                         \
    const uint t_start = (uint)t_start_i;                                     \
    const uint t_end = (uint)t_end_i;                                         \
                                                                              \
    const uint q_off = ((b * Tq + t_q) * H + h) * D;                          \
    float q_reg[DPL];                                                         \
    for (uint j = 0; j < DPL; ++j) { q_reg[j] = Q[q_off + lane + j * SG_W]; } \
                                                                              \
    float acc[DPL];                                                           \
    for (uint j = 0; j < DPL; ++j) { acc[j] = 0.0f; }                         \
    float m_i = -INFINITY;                                                    \
    float l_i = 0.0f;                                                         \
                                                                              \
    for (uint t = t_start; t < t_end; ++t) {                                  \
        const uint kv_base = ((b * Tkv + t) * Hkv + hkv) * D;                 \
        float part = 0.0f;                                                    \
        for (uint j = 0; j < DPL; ++j) {                                      \
            part += q_reg[j] * K[kv_base + lane + j * SG_W];                  \
        }                                                                     \
        const float s = simd_sum(part) * scale;                               \
        const float m_new = max(m_i, s);                                      \
        /* m_i == -inf on the first key: exp(-inf - -inf) would be NaN, and    \
           the accumulator is zero anyway, so the rescale is exactly zero. */ \
        const float alpha = (m_i == -INFINITY) ? 0.0f : exp(m_i - m_new);     \
        const float p = exp(s - m_new);                                       \
        for (uint j = 0; j < DPL; ++j) {                                      \
            acc[j] = acc[j] * alpha + p * V[kv_base + lane + j * SG_W];       \
        }                                                                     \
        l_i = l_i * alpha + p;                                                \
        m_i = m_new;                                                          \
    }                                                                         \
                                                                              \
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;                   \
    if (out_bf16 != 0u) {                                                     \
        device bfloat *Ob = (device bfloat *)O;                               \
        for (uint j = 0; j < DPL; ++j) {                                      \
            Ob[o_off + lane + j * SG_W] = bfloat(acc[j] * inv_l);             \
        }                                                                     \
    } else {                                                                  \
        for (uint j = 0; j < DPL; ++j) {                                      \
            O[o_off + lane + j * SG_W] = acc[j] * inv_l;                      \
        }                                                                     \
    }                                                                         \
    (void)B;                                                                  \
}

ROWS_KERNEL(flash_attn_rows_h128, 128)
ROWS_KERNEL(flash_attn_rows_h256, 256)
ROWS_KERNEL(flash_attn_rows_h512, 512)
