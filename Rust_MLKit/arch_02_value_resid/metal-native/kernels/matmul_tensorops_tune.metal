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

inline uint2 tune_tile_from_linear(uint linear, uint tiles_n) {
    return uint2(linear % tiles_n, linear / tiles_n);
}

/// ACCUM_FIRST=true reproduces production (all blocks accumulate; C must be
/// pre-zeroed). ACCUM_FIRST=false makes block 0 `multiply`, retiring the zero.
template <int SM, int SN, int BK, int NSG, bool ACCUM_FIRST>
inline void mm_bf16_tune(device bfloat *A, device bfloat *B, device float *C,
                         uint M, uint N, uint K, uint tiles_n, uint tgpig) {
    constexpr auto d_mul = matmul2d_descriptor(
        SM, SN, BK, false, false, false, matmul2d_descriptor::mode::multiply);
    constexpr auto d_acc = matmul2d_descriptor(
        SM, SN, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d_mul, execution_simdgroups<NSG>> op_mul;
    matmul2d<d_acc, execution_simdgroups<NSG>> op_acc;

    uint2 tile = tune_tile_from_linear(tgpig, tiles_n);
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
                     uint tgpig [[threadgroup_position_in_grid]]) {           \
        mm_bf16_tune<SM, SN, BK, NSG, ACCF>(A, B, C, M, N, K, tiles_n, tgpig); \
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
