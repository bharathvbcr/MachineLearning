// Tile-geometry A/B for the bf16 NN GEMM. Isolates two suspects behind the
// ~2× gap vs PyTorch MPS bf16:
//   (a) output-tile size -> arithmetic intensity  (SM*SN/(SM+SN) FLOP/byte)
//   (b) the host-side zero_f32(C) pre-pass, which only exists because the
//       production kernel runs multiply_accumulate on the FIRST K block too.
// Interior-only (exact divisibility) — this is a measurement rig, not a
// drop-in: the production kernel keeps the ragged-edge paths.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

inline uint2 tune_morton_decode_2d(uint c) {
    uint x = 0, y = 0;
#pragma unroll
    for (uint i = 0; i < 16; ++i) {
        x |= ((c >> (2 * i)) & 1u) << i;
        y |= ((c >> (2 * i + 1)) & 1u) << i;
    }
    return uint2(x, y);
}

/// Mirrors `tile_from_linear` in matmul_tensorops.metal. Production uses Morton
/// order on square power-of-two tile grids; the rig must match or every variant
/// is unfairly penalised on square shapes.
inline uint2 tune_tile_from_linear(uint linear, uint tiles_n, uint tiles_m) {
    if (tiles_n == tiles_m && tiles_n != 0u && (tiles_n & (tiles_n - 1u)) == 0u) {
        return tune_morton_decode_2d(linear);
    }
    return uint2(linear % tiles_n, linear / tiles_n);
}

/// ACCUM_FIRST=true reproduces production (all blocks accumulate; C must be
/// pre-zeroed). ACCUM_FIRST=false makes block 0 `multiply`, retiring the zero.
template <int SM, int SN, int BK, int NSG, bool ACCUM_FIRST>
inline void mm_bf16_tune(device bfloat *A, device bfloat *B, device float *C,
                         uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d_mul = matmul2d_descriptor(
        SM, SN, BK, false, false, false, matmul2d_descriptor::mode::multiply);
    constexpr auto d_acc = matmul2d_descriptor(
        SM, SN, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d_mul, execution_simdgroups<NSG>> op_mul;
    matmul2d<d_acc, execution_simdgroups<NSG>> op_acc;

    uint2 tile = tune_tile_from_linear(tgpig, tiles_n, tiles_m);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;

    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    for (int k = 0; k + BK <= (int)K; k += BK) {
        auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                         array<int, 2>{1, (int)N});
        if (!ACCUM_FIRST && k == 0) {
            op_mul.run(tA, tB, tC);
        } else {
            op_acc.run(tA, tB, tC);
        }
    }
}

#define TUNE_KERNEL(NAME, SM, SN, BK, NSG, ACCF)                              \
    kernel void NAME(device bfloat *A [[buffer(0)]],                          \
                     device bfloat *B [[buffer(1)]],                          \
                     device float *C [[buffer(2)]],                           \
                     constant uint &M [[buffer(3)]],                          \
                     constant uint &N [[buffer(4)]],                          \
                     constant uint &K [[buffer(5)]],                          \
                     constant uint &tiles_n [[buffer(6)]],                    \
                     constant uint &tiles_m [[buffer(7)]],                    \
                     uint tgpig [[threadgroup_position_in_grid]]) {           \
        mm_bf16_tune<SM, SN, BK, NSG, ACCF>(A, B, C, M, N, K, tiles_n, tiles_m, tgpig); \
    }

// Control: production geometry + production accumulate-first behaviour.
TUNE_KERNEL(mm_bf16_64x32_bk128_sg4_accf,  64,  32, 128, 4, true)
// Same geometry, zero pre-pass retired.
TUNE_KERNEL(mm_bf16_64x32_bk128_sg4,       64,  32, 128, 4, false)
// Arithmetic-intensity ladder.
TUNE_KERNEL(mm_bf16_64x64_bk64_sg4,        64,  64,  64, 4, false)
TUNE_KERNEL(mm_bf16_128x64_bk64_sg4,      128,  64,  64, 4, false)
TUNE_KERNEL(mm_bf16_128x64_bk64_sg8,      128,  64,  64, 8, false)
TUNE_KERNEL(mm_bf16_128x128_bk64_sg8,     128, 128,  64, 8, false)
TUNE_KERNEL(mm_bf16_128x128_bk32_sg8,     128, 128,  32, 8, false)

// BK ladder at a fixed 64x64 tile: isolates C read-modify-write traffic, which
// scales as K/BK passes over the output tile and is independent of SM/SN.
TUNE_KERNEL(mm_bf16_64x64_bk32_sg4,   64, 64,  32, 4, false)
TUNE_KERNEL(mm_bf16_64x64_bk128_sg4,  64, 64, 128, 4, false)
TUNE_KERNEL(mm_bf16_64x64_bk256_sg4,  64, 64, 256, 4, false)
TUNE_KERNEL(mm_bf16_64x64_bk512_sg4,  64, 64, 512, 4, false)

// Asymptote: at BK >= K the loop runs a single block, so C is touched exactly
// once — the floor for this tile with no accumulate round-trips at all.
TUNE_KERNEL(mm_bf16_64x64_bk1024_sg4, 64, 64, 1024, 4, false)
TUNE_KERNEL(mm_bf16_64x64_bk2048_sg4, 64, 64, 2048, 4, false)
TUNE_KERNEL(mm_bf16_64x64_bk4096_sg4, 64, 64, 4096, 4, false)

// Missing cell: large tile AND large BK together. Tile size raises arithmetic
// intensity; BK cuts C round-trips. Earlier runs varied them one at a time, so
// every big-tile variant was still paying full accumulate traffic.
TUNE_KERNEL(mm_bf16_128x64_bk256_sg4,   128,  64, 256, 4, false)
TUNE_KERNEL(mm_bf16_128x64_bk256_sg8,   128,  64, 256, 8, false)
TUNE_KERNEL(mm_bf16_128x128_bk256_sg8,  128, 128, 256, 8, false)
TUNE_KERNEL(mm_bf16_128x128_bk256_sg4,  128, 128, 256, 4, false)
TUNE_KERNEL(mm_bf16_256x64_bk256_sg8,   256,  64, 256, 8, false)

// ---------------------------------------------------------------------------
// f32 relaxed-precision (tf32-class) ladder. Same structure as the bf16 rig, but
// f32 operands are 2x the bytes, so tile arithmetic intensity is half that of
// bf16 at the same SM/SN — the bf16 optimum is not assumed to transfer.
// ---------------------------------------------------------------------------

template <int SM, int SN, int BK, int NSG, bool ACCUM_FIRST>
inline void mm_f32r_tune(device float *A, device float *B, device float *C,
                         uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d_mul = matmul2d_descriptor(
        SM, SN, BK, false, false, true, matmul2d_descriptor::mode::multiply);
    constexpr auto d_acc = matmul2d_descriptor(
        SM, SN, BK, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d_mul, execution_simdgroups<NSG>> op_mul;
    matmul2d<d_acc, execution_simdgroups<NSG>> op_acc;

    uint2 tile = tune_tile_from_linear(tgpig, tiles_n, tiles_m);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;

    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    for (int k = 0; k + BK <= (int)K; k += BK) {
        auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                         array<int, 2>{1, (int)N});
        if (!ACCUM_FIRST && k == 0) {
            op_mul.run(tA, tB, tC);
        } else {
            op_acc.run(tA, tB, tC);
        }
    }
}

#define TUNE_KERNEL_F32R(NAME, SM, SN, BK, NSG, ACCF)                         \
    kernel void NAME(device float *A [[buffer(0)]],                           \
                     device float *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                           \
                     constant uint &M [[buffer(3)]],                          \
                     constant uint &N [[buffer(4)]],                          \
                     constant uint &K [[buffer(5)]],                          \
                     constant uint &tiles_n [[buffer(6)]],                    \
                     constant uint &tiles_m [[buffer(7)]],                    \
                     uint tgpig [[threadgroup_position_in_grid]]) {           \
        mm_f32r_tune<SM, SN, BK, NSG, ACCF>(A, B, C, M, N, K, tiles_n, tiles_m, tgpig); \
    }

// Control: production geometry + production accumulate-first behaviour.
TUNE_KERNEL_F32R(mm_f32r_64x32_bk128_sg4_accf,  64,  32, 128, 4, true)
// Same geometry, zero pre-pass retired.
TUNE_KERNEL_F32R(mm_f32r_64x32_bk128_sg4,       64,  32, 128, 4, false)
TUNE_KERNEL_F32R(mm_f32r_64x32_bk256_sg4,       64,  32, 256, 4, false)
TUNE_KERNEL_F32R(mm_f32r_64x64_bk128_sg4,       64,  64, 128, 4, false)
TUNE_KERNEL_F32R(mm_f32r_64x64_bk256_sg4,       64,  64, 256, 4, false)
TUNE_KERNEL_F32R(mm_f32r_128x64_bk256_sg8,     128,  64, 256, 8, false)
TUNE_KERNEL_F32R(mm_f32r_128x128_bk256_sg4,    128, 128, 256, 4, false)

// mlp_down (N=768) regressed under 128x64/sg8; probe narrower/wider tiles and
// simdgroup counts for a single geometry that wins without regressing it.
TUNE_KERNEL_F32R(mm_f32r_128x64_bk256_sg4,     128,  64, 256, 4, false)
TUNE_KERNEL_F32R(mm_f32r_128x32_bk256_sg8,     128,  32, 256, 8, false)
TUNE_KERNEL_F32R(mm_f32r_128x32_bk256_sg4,     128,  32, 256, 4, false)
TUNE_KERNEL_F32R(mm_f32r_256x64_bk256_sg8,     256,  64, 256, 8, false)

// ---------------------------------------------------------------------------
// TN / NT bf16 tile ladder. These kernels are single-shot `mode::multiply` over
// the full K (no BK loop), so they carry no C round-trip traffic — tile geometry
// and occupancy are the only knobs. Real gradient shapes have small M/N and
// K=BT=4096, so tiles-in-flight matters more here than arithmetic intensity.
// ---------------------------------------------------------------------------

template <int SM, int SN, int NSG, bool TRANSPOSE_LEFT>
inline void mm_bf16_tn_nt_tune(device bfloat *A, device bfloat *B, device float *C,
                               uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, TRANSPOSE_LEFT, !TRANSPOSE_LEFT,
        false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    uint2 tile = tune_tile_from_linear(tgpig, tiles_n, tiles_m);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;

    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    if (TRANSPOSE_LEFT) {
        // TN: A stored [K,M], strides {1,M}; B stored [K,N], strides {1,N}.
        auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K}, array<int, 2>{1, (int)M});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K}, array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        // NT: A stored [M,K], strides {1,K}; B stored [N,K], strides {1,K}.
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM}, array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN}, array<int, 2>{1, (int)K});
        op.run(tA, tB, tC);
    }
}

#define TUNE_KERNEL_TNNT(NAME, SM, SN, NSG, TL)                               \
    kernel void NAME(device bfloat *A [[buffer(0)]],                          \
                     device bfloat *B [[buffer(1)]],                          \
                     device float *C [[buffer(2)]],                           \
                     constant uint &M [[buffer(3)]],                          \
                     constant uint &N [[buffer(4)]],                          \
                     constant uint &K [[buffer(5)]],                          \
                     constant uint &tiles_n [[buffer(6)]],                    \
                     constant uint &tiles_m [[buffer(7)]],                    \
                     uint tgpig [[threadgroup_position_in_grid]]) {           \
        mm_bf16_tn_nt_tune<SM, SN, NSG, TL>(A, B, C, M, N, K, tiles_n, tiles_m, tgpig); \
    }

// TN ladder (control first = production geometry).
TUNE_KERNEL_TNNT(mm_tn_64x32_sg4,   64,  32, 4, true)
TUNE_KERNEL_TNNT(mm_tn_64x64_sg4,   64,  64, 4, true)
TUNE_KERNEL_TNNT(mm_tn_128x64_sg4, 128,  64, 4, true)
TUNE_KERNEL_TNNT(mm_tn_128x64_sg8, 128,  64, 8, true)
TUNE_KERNEL_TNNT(mm_tn_64x64_sg8,   64,  64, 8, true)
TUNE_KERNEL_TNNT(mm_tn_32x32_sg4,   32,  32, 4, true)

// NT ladder.
TUNE_KERNEL_TNNT(mm_nt_64x32_sg4,   64,  32, 4, false)
TUNE_KERNEL_TNNT(mm_nt_64x64_sg4,   64,  64, 4, false)
TUNE_KERNEL_TNNT(mm_nt_128x64_sg4, 128,  64, 4, false)
TUNE_KERNEL_TNNT(mm_nt_128x64_sg8, 128,  64, 8, false)
TUNE_KERNEL_TNNT(mm_nt_64x64_sg8,   64,  64, 8, false)
TUNE_KERNEL_TNNT(mm_nt_32x32_sg4,   32,  32, 4, false)

// Narrow-N candidates: at N=128 a 64-wide tile leaves only 2 column tiles, so
// the wide geometry loses parallelism on the smallest preset.
TUNE_KERNEL(mm_bf16_64x32_bk256_sg4, 64, 32, 256, 4, false)
TUNE_KERNEL(mm_bf16_32x32_bk256_sg4, 32, 32, 256, 4, false)

// Accumulating TN/NT ladder. These are the dominant dW path (24 call sites in
// model_bwd). Structure matches the plain kernels except mode; C is a genuine
// += target so no zero-removal applies — only the tile is in question.
template <int SM, int SN, int NSG, bool TRANSPOSE_LEFT>
inline void mm_bf16_acc_tune(device bfloat *A, device bfloat *B, device float *C,
                             uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, TRANSPOSE_LEFT, !TRANSPOSE_LEFT,
        false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    uint2 tile = tune_tile_from_linear(tgpig, tiles_n, tiles_m);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    if (TRANSPOSE_LEFT) {
        auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K}, array<int, 2>{1, (int)M});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K}, array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM}, array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN}, array<int, 2>{1, (int)K});
        op.run(tA, tB, tC);
    }
}

#define TUNE_KERNEL_ACC(NAME, SM, SN, NSG, TL)                                \
    kernel void NAME(device bfloat *A [[buffer(0)]],                          \
                     device bfloat *B [[buffer(1)]],                          \
                     device float *C [[buffer(2)]],                           \
                     constant uint &M [[buffer(3)]],                          \
                     constant uint &N [[buffer(4)]],                          \
                     constant uint &K [[buffer(5)]],                          \
                     constant uint &tiles_n [[buffer(6)]],                    \
                     constant uint &tiles_m [[buffer(7)]],                    \
                     uint tgpig [[threadgroup_position_in_grid]]) {           \
        mm_bf16_acc_tune<SM, SN, NSG, TL>(A, B, C, M, N, K, tiles_n, tiles_m, tgpig);  \
    }

TUNE_KERNEL_ACC(mm_tnacc_64x32_sg4,   64,  32, 4, true)
TUNE_KERNEL_ACC(mm_tnacc_64x64_sg4,   64,  64, 4, true)
TUNE_KERNEL_ACC(mm_tnacc_64x64_sg8,   64,  64, 8, true)
TUNE_KERNEL_ACC(mm_tnacc_128x64_sg4, 128,  64, 4, true)
TUNE_KERNEL_ACC(mm_tnacc_128x64_sg8, 128,  64, 8, true)
TUNE_KERNEL_ACC(mm_tnacc_32x32_sg4,   32,  32, 4, true)
TUNE_KERNEL_ACC(mm_ntacc_64x32_sg4,   64,  32, 4, false)
TUNE_KERNEL_ACC(mm_ntacc_64x64_sg4,   64,  64, 4, false)
TUNE_KERNEL_ACC(mm_ntacc_64x64_sg8,   64,  64, 8, false)
TUNE_KERNEL_ACC(mm_ntacc_128x64_sg4, 128,  64, 4, false)
TUNE_KERNEL_ACC(mm_ntacc_128x64_sg8, 128,  64, 8, false)
TUNE_KERNEL_ACC(mm_ntacc_32x32_sg4,   32,  32, 4, false)

// ---------------------------------------------------------------------------
// Register-resident accumulator (cooperative_tensor). The blocked kernels
// accumulate into a *device-memory* C tile once per K block, so C round-trips
// scale with K/BK — which is exactly where the PyTorch MPS gap lives. Holding
// the accumulator in registers across the whole K loop reduces that to a single
// store, independent of K. Per MPPTensorOpsMatMul2d.h.
// ---------------------------------------------------------------------------

template <int SM, int SN, int BK, int NSG>
inline void mm_bf16_coop(device bfloat *A, device bfloat *B, device float *C,
                         uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d_acc = matmul2d_descriptor(
        SM, SN, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d_acc, execution_simdgroups<NSG>> op;

    uint2 tile = tune_tile_from_linear(tgpig, tiles_n, tiles_m);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;

    auto tA0 = tensor(A + ty * (int)K, dextents<int, 2>{BK, SM}, array<int, 2>{1, (int)K});
    auto tB0 = tensor(B + tx, dextents<int, 2>{SN, BK}, array<int, 2>{1, (int)N});
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) { cT[i] = 0.0f; }
    }

    for (int k = 0; k + BK <= (int)K; k += BK) {
        auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, cT);
    }

    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    cT.store(tC);
}

#define TUNE_KERNEL_COOP(NAME, SM, SN, BK, NSG)                               \
    kernel void NAME(device bfloat *A [[buffer(0)]],                          \
                     device bfloat *B [[buffer(1)]],                          \
                     device float *C [[buffer(2)]],                           \
                     constant uint &M [[buffer(3)]],                          \
                     constant uint &N [[buffer(4)]],                          \
                     constant uint &K [[buffer(5)]],                          \
                     constant uint &tiles_n [[buffer(6)]],                    \
                     constant uint &tiles_m [[buffer(7)]],                    \
                     uint tgpig [[threadgroup_position_in_grid]]) {           \
        mm_bf16_coop<SM, SN, BK, NSG>(A, B, C, M, N, K, tiles_n, tiles_m, tgpig);      \
    }

TUNE_KERNEL_COOP(mm_coop_64x64_bk64_sg4,    64,  64,  64, 4)
TUNE_KERNEL_COOP(mm_coop_64x64_bk128_sg4,   64,  64, 128, 4)
TUNE_KERNEL_COOP(mm_coop_64x64_bk256_sg4,   64,  64, 256, 4)
TUNE_KERNEL_COOP(mm_coop_128x64_bk128_sg8, 128,  64, 128, 8)
TUNE_KERNEL_COOP(mm_coop_128x128_bk64_sg8, 128, 128,  64, 8)

// ---------------------------------------------------------------------------
// Register-resident accumulator for the remaining production layouts.
//
// NN bf16 already ships `matmul2d_tensorops_bf16_f32_coop`. These variants ask
// the same question for f32-relaxed (which has the identical BK-blocked
// device-C structure) and for TN/NT (which today issue ONE full-K `matmul2d`,
// so they have no host-visible C round-trip at all — whether MPP spills the
// accumulator internally is exactly what the `_blk` controls below isolate).
//
// Three shapes per layout, so a win can be attributed:
//   *_blk   : explicit BK loop, device-memory C, block 0 = multiply  (control)
//   *_coop  : explicit BK loop, register accumulator, single store   (candidate)
//   (production = one full-K matmul2d into device C)
// ---------------------------------------------------------------------------

template <int SM, int SN, int BK, int NSG>
inline void mm_f32r_coop(device float *A, device float *B, device float *C,
                         uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d_acc = matmul2d_descriptor(
        SM, SN, BK, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d_acc, execution_simdgroups<NSG>> op;

    uint2 tile = tune_tile_from_linear(tgpig, tiles_n, tiles_m);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;

    auto tA0 = tensor(A + ty * (int)K, dextents<int, 2>{BK, SM}, array<int, 2>{1, (int)K});
    auto tB0 = tensor(B + tx, dextents<int, 2>{SN, BK}, array<int, 2>{1, (int)N});
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) { cT[i] = 0.0f; }
    }
    for (int k = 0; k + BK <= (int)K; k += BK) {
        auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM}, array<int, 2>{1, (int)K});
        auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK}, array<int, 2>{1, (int)N});
        op.run(tA, tB, cT);
    }
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM}, array<int, 2>{1, (int)N});
    cT.store(tC);
}

#define TUNE_KERNEL_F32R_COOP(NAME, SM, SN, BK, NSG)                          \
    kernel void NAME(device float *A [[buffer(0)]],                           \
                     device float *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                           \
                     constant uint &M [[buffer(3)]],                          \
                     constant uint &N [[buffer(4)]],                          \
                     constant uint &K [[buffer(5)]],                          \
                     constant uint &tiles_n [[buffer(6)]],                    \
                     constant uint &tiles_m [[buffer(7)]],                    \
                     uint tgpig [[threadgroup_position_in_grid]]) {           \
        mm_f32r_coop<SM, SN, BK, NSG>(A, B, C, M, N, K, tiles_n, tiles_m, tgpig);      \
    }

TUNE_KERNEL_F32R_COOP(mm_f32rcoop_128x64_bk64_sg4,   128,  64,  64, 4)
TUNE_KERNEL_F32R_COOP(mm_f32rcoop_128x64_bk128_sg4,  128,  64, 128, 4)
TUNE_KERNEL_F32R_COOP(mm_f32rcoop_128x64_bk256_sg4,  128,  64, 256, 4)
TUNE_KERNEL_F32R_COOP(mm_f32rcoop_64x64_bk128_sg4,    64,  64, 128, 4)
TUNE_KERNEL_F32R_COOP(mm_f32rcoop_128x128_bk128_sg4, 128, 128, 128, 4)
TUNE_KERNEL_F32R_COOP(mm_f32rcoop_128x64_bk128_sg8,  128,  64, 128, 8)

/// TN (transpose_left): A is physically [K,M] strides {1,M}; a K block starts
/// at A + k*M. NT (transpose_right): B is physically [N,K] strides {1,K}; a K
/// block of row `tx` starts at B + tx*K + k.
template <int SM, int SN, int BK, int NSG, bool TRANSPOSE_LEFT, bool COOP>
inline void mm_bf16_tnnt_kblk(device bfloat *A, device bfloat *B, device float *C,
                              uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d_mul = matmul2d_descriptor(
        SM, SN, BK, TRANSPOSE_LEFT, !TRANSPOSE_LEFT, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto d_acc = matmul2d_descriptor(
        SM, SN, BK, TRANSPOSE_LEFT, !TRANSPOSE_LEFT, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d_mul, execution_simdgroups<NSG>> op_mul;
    matmul2d<d_acc, execution_simdgroups<NSG>> op_acc;

    uint2 tile = tune_tile_from_linear(tgpig, tiles_n, tiles_m);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;

    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM}, array<int, 2>{1, (int)N});

    if (COOP) {
        // Seed the operand types from block 0 so the destination type resolves.
        auto a0 = TRANSPOSE_LEFT
            ? tensor(A + ty, dextents<int, 2>{SM, BK}, array<int, 2>{1, (int)M})
            : tensor(A + ty * (int)K, dextents<int, 2>{BK, SM}, array<int, 2>{1, (int)K});
        auto b0 = TRANSPOSE_LEFT
            ? tensor(B + tx, dextents<int, 2>{SN, BK}, array<int, 2>{1, (int)N})
            : tensor(B + tx * (int)K, dextents<int, 2>{BK, SN}, array<int, 2>{1, (int)K});
        auto cT = op_acc.template get_destination_cooperative_tensor<
            decltype(a0), decltype(b0), float>();
#pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) { cT[i] = 0.0f; }
        }
        for (int k = 0; k + BK <= (int)K; k += BK) {
            if (TRANSPOSE_LEFT) {
                auto tA = tensor(A + k * (int)M + ty, dextents<int, 2>{SM, BK},
                                 array<int, 2>{1, (int)M});
                auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                                 array<int, 2>{1, (int)N});
                op_acc.run(tA, tB, cT);
            } else {
                auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                                 array<int, 2>{1, (int)K});
                auto tB = tensor(B + tx * (int)K + k, dextents<int, 2>{BK, SN},
                                 array<int, 2>{1, (int)K});
                op_acc.run(tA, tB, cT);
            }
        }
        cT.store(tC);
    } else {
        for (int k = 0; k + BK <= (int)K; k += BK) {
            if (TRANSPOSE_LEFT) {
                auto tA = tensor(A + k * (int)M + ty, dextents<int, 2>{SM, BK},
                                 array<int, 2>{1, (int)M});
                auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                                 array<int, 2>{1, (int)N});
                if (k == 0) { op_mul.run(tA, tB, tC); } else { op_acc.run(tA, tB, tC); }
            } else {
                auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                                 array<int, 2>{1, (int)K});
                auto tB = tensor(B + tx * (int)K + k, dextents<int, 2>{BK, SN},
                                 array<int, 2>{1, (int)K});
                if (k == 0) { op_mul.run(tA, tB, tC); } else { op_acc.run(tA, tB, tC); }
            }
        }
    }
}

#define TUNE_KERNEL_TNNT_K(NAME, SM, SN, BK, NSG, TL, COOP)                   \
    kernel void NAME(device bfloat *A [[buffer(0)]],                          \
                     device bfloat *B [[buffer(1)]],                          \
                     device float *C [[buffer(2)]],                           \
                     constant uint &M [[buffer(3)]],                          \
                     constant uint &N [[buffer(4)]],                          \
                     constant uint &K [[buffer(5)]],                          \
                     constant uint &tiles_n [[buffer(6)]],                    \
                     constant uint &tiles_m [[buffer(7)]],                    \
                     uint tgpig [[threadgroup_position_in_grid]]) {           \
        mm_bf16_tnnt_kblk<SM, SN, BK, NSG, TL, COOP>(A, B, C, M, N, K,        \
                                                     tiles_n, tiles_m, tgpig); \
    }

// TN: control (device-C K-blocking) then coop.
TUNE_KERNEL_TNNT_K(mm_tnblk_128x64_bk128_sg4,  128,  64, 128, 4, true,  false)
TUNE_KERNEL_TNNT_K(mm_tncoop_128x64_bk64_sg4,  128,  64,  64, 4, true,  true)
TUNE_KERNEL_TNNT_K(mm_tncoop_128x64_bk128_sg4, 128,  64, 128, 4, true,  true)
TUNE_KERNEL_TNNT_K(mm_tncoop_128x64_bk256_sg4, 128,  64, 256, 4, true,  true)
TUNE_KERNEL_TNNT_K(mm_tncoop_64x64_bk128_sg4,   64,  64, 128, 4, true,  true)
TUNE_KERNEL_TNNT_K(mm_tncoop_128x128_bk128_sg4,128, 128, 128, 4, true,  true)
TUNE_KERNEL_TNNT_K(mm_tncoop_128x64_bk128_sg8, 128,  64, 128, 8, true,  true)

// NT: control then coop.
TUNE_KERNEL_TNNT_K(mm_ntblk_128x64_bk128_sg4,  128,  64, 128, 4, false, false)
TUNE_KERNEL_TNNT_K(mm_ntcoop_128x64_bk64_sg4,  128,  64,  64, 4, false, true)
TUNE_KERNEL_TNNT_K(mm_ntcoop_128x64_bk128_sg4, 128,  64, 128, 4, false, true)
TUNE_KERNEL_TNNT_K(mm_ntcoop_128x64_bk256_sg4, 128,  64, 256, 4, false, true)
TUNE_KERNEL_TNNT_K(mm_ntcoop_64x64_bk128_sg4,   64,  64, 128, 4, false, true)
TUNE_KERNEL_TNNT_K(mm_ntcoop_128x128_bk128_sg4,128, 128, 128, 4, false, true)
TUNE_KERNEL_TNNT_K(mm_ntcoop_128x64_bk128_sg8, 128,  64, 128, 8, false, true)

/// Accumulating TN/NT with a register accumulator: load C once, accumulate the
/// whole K loop in registers, store once. Same read-and-write traffic as the
/// single-shot kernel, but the K loop never touches C.
template <int SM, int SN, int BK, int NSG, bool TRANSPOSE_LEFT>
inline void mm_bf16_acc_coop(device bfloat *A, device bfloat *B, device float *C,
                             uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d_acc = matmul2d_descriptor(
        SM, SN, BK, TRANSPOSE_LEFT, !TRANSPOSE_LEFT, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d_acc, execution_simdgroups<NSG>> op;

    uint2 tile = tune_tile_from_linear(tgpig, tiles_n, tiles_m);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;

    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM}, array<int, 2>{1, (int)N});
    auto a0 = TRANSPOSE_LEFT
        ? tensor(A + ty, dextents<int, 2>{SM, BK}, array<int, 2>{1, (int)M})
        : tensor(A + ty * (int)K, dextents<int, 2>{BK, SM}, array<int, 2>{1, (int)K});
    auto b0 = TRANSPOSE_LEFT
        ? tensor(B + tx, dextents<int, 2>{SN, BK}, array<int, 2>{1, (int)N})
        : tensor(B + tx * (int)K, dextents<int, 2>{BK, SN}, array<int, 2>{1, (int)K});
    auto cT = op.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), float>();
    cT.load(tC);
    for (int k = 0; k + BK <= (int)K; k += BK) {
        if (TRANSPOSE_LEFT) {
            auto tA = tensor(A + k * (int)M + ty, dextents<int, 2>{SM, BK},
                             array<int, 2>{1, (int)M});
            auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                             array<int, 2>{1, (int)N});
            op.run(tA, tB, cT);
        } else {
            auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                             array<int, 2>{1, (int)K});
            auto tB = tensor(B + tx * (int)K + k, dextents<int, 2>{BK, SN},
                             array<int, 2>{1, (int)K});
            op.run(tA, tB, cT);
        }
    }
    cT.store(tC);
}

#define TUNE_KERNEL_ACC_COOP(NAME, SM, SN, BK, NSG, TL)                       \
    kernel void NAME(device bfloat *A [[buffer(0)]],                          \
                     device bfloat *B [[buffer(1)]],                          \
                     device float *C [[buffer(2)]],                           \
                     constant uint &M [[buffer(3)]],                          \
                     constant uint &N [[buffer(4)]],                          \
                     constant uint &K [[buffer(5)]],                          \
                     constant uint &tiles_n [[buffer(6)]],                    \
                     constant uint &tiles_m [[buffer(7)]],                    \
                     uint tgpig [[threadgroup_position_in_grid]]) {           \
        mm_bf16_acc_coop<SM, SN, BK, NSG, TL>(A, B, C, M, N, K,               \
                                              tiles_n, tiles_m, tgpig);       \
    }

TUNE_KERNEL_ACC_COOP(mm_tnacccoop_128x64_bk128_sg4, 128, 64, 128, 4, true)
TUNE_KERNEL_ACC_COOP(mm_tnacccoop_128x64_bk256_sg4, 128, 64, 256, 4, true)
TUNE_KERNEL_ACC_COOP(mm_tnacccoop_64x64_bk128_sg4,   64, 64, 128, 4, true)
TUNE_KERNEL_ACC_COOP(mm_ntacccoop_64x64_bk128_sg4,   64, 64, 128, 4, false)
TUNE_KERNEL_ACC_COOP(mm_ntacccoop_64x64_bk256_sg4,   64, 64, 256, 4, false)
TUNE_KERNEL_ACC_COOP(mm_ntacccoop_128x64_bk128_sg4, 128, 64, 128, 4, false)

// Wider output tiles for the bf16 coop kernel. The paired cross-runtime run
// leaves torch ahead on exactly the large-N shapes (mlp_up N=3072 at 0.95x,
// tall_k1024 N=4096 at 0.95x); a 64-wide tile there means many column tiles and
// the lowest arithmetic intensity of any candidate, so tile width is the first
// suspect. SM*SN/(SM+SN) FLOP/byte: 64x64 -> 32, 128x64 -> 42.7, 128x128 -> 64.
TUNE_KERNEL_COOP(mm_coop_128x64_bk128_sg4,   128,  64, 128, 4)
TUNE_KERNEL_COOP(mm_coop_128x128_bk128_sg8, 128, 128, 128, 8)
TUNE_KERNEL_COOP(mm_coop_128x128_bk128_sg4, 128, 128, 128, 4)
TUNE_KERNEL_COOP(mm_coop_64x128_bk128_sg4,   64, 128, 128, 4)
TUNE_KERNEL_COOP(mm_coop_256x64_bk128_sg8,  256,  64, 128, 8)
