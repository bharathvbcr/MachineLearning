// TensorOps GEMM via Metal Performance Primitives (M5 neural accelerators).
// Primary path for Phase 0+; requires Metal 4 / macOS 26+.
//
// GEMM v2 (MPP §2.3):
//   - Morton 1D threadgroup walk (cache-friendly tile traversal)
//   - execution_simdgroups<4> on bf16 / relaxed hot paths (64×32 TG tiles)
//   - BK=128 cooperative K-accumulate for large K (interior tiles)
//   - Compile-time tile extents via offset+dextents{SN,SM} (pointer tensors
//     lack static_slice; this is the equivalent bounds-check elision)
//   - mode::multiply still needs C zeroed once (packed with matmul on host)
//
// Note: device pointers must be non-const — `const` poisons MPP type matching.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/// Decode Morton/Z-order code → (x, y) tile coordinates.
inline uint2 morton_decode_2d(uint c) {
    uint x = 0, y = 0;
#pragma unroll
    for (uint i = 0; i < 16; ++i) {
        x |= ((c >> (2 * i)) & 1u) << i;
        y |= ((c >> (2 * i + 1)) & 1u) << i;
    }
    return uint2(x, y);
}

/// Decode linear TG id → (x, y) tile. Uses Morton when the grid is square and
/// power-of-two (cache-friendly); otherwise compact row-major (avoids pad tax).
inline uint2 tile_from_linear(uint linear, uint tiles_n, uint tiles_m) {
    if (tiles_n == tiles_m && tiles_n != 0u && (tiles_n & (tiles_n - 1u)) == 0u) {
        return morton_decode_2d(linear);
    }
    return uint2(linear % tiles_n, linear / tiles_n);
}

// =============================================================================
// Bank-batched exact f32 TensorOps GEMMs for optimizer matrix contractions.
// One flat grid covers [batch, output tiles], eliminating per-layer launches.
// =============================================================================

kernel void matmul2d_tensorops_batched_f32(
    device float *A [[buffer(0)]], device float *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]], constant uint &tiles_m [[buffer(7)]],
    constant uint &batch [[buffer(8)]], uint flat_tg [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32, SN = 32;
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    uint tiles = tiles_n * tiles_m;
    uint mid = flat_tg / tiles;
    if (mid >= batch) return;
    uint2 tile = tile_from_linear(flat_tg - mid * tiles, tiles_n, tiles_m);
    int tx = (int)tile.x * SN, ty = (int)tile.y * SM;
    device float *Ab = A + mid * M * K;
    device float *Bb = B + mid * K * N;
    device float *Cb = C + mid * M * N;
    auto mA = tensor(Ab, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    auto mB = tensor(Bb, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
    auto mC = tensor(Cb, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
    auto tA = mA.slice(0, ty);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

kernel void matmul2d_tensorops_batched_tn_f32(
    device float *A [[buffer(0)]], device float *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]], constant uint &tiles_m [[buffer(7)]],
    constant uint &batch [[buffer(8)]], uint flat_tg [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32, SN = 32;
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, true, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    uint tiles = tiles_n * tiles_m;
    uint mid = flat_tg / tiles;
    if (mid >= batch) return;
    uint2 tile = tile_from_linear(flat_tg - mid * tiles, tiles_n, tiles_m);
    int tx = (int)tile.x * SN, ty = (int)tile.y * SM;
    device float *Ab = A + mid * K * M; // physical [K,M]
    device float *Bb = B + mid * K * N;
    device float *Cb = C + mid * M * N;
    auto mA = tensor(Ab, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
    auto mB = tensor(Bb, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
    auto mC = tensor(Cb, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
    auto tA = mA.slice(ty, 0);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

kernel void matmul2d_tensorops_batched_nt_f32(
    device float *A [[buffer(0)]], device float *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]], constant uint &tiles_m [[buffer(7)]],
    constant uint &batch [[buffer(8)]], uint flat_tg [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32, SN = 32;
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    uint tiles = tiles_n * tiles_m;
    uint mid = flat_tg / tiles;
    if (mid >= batch) return;
    uint2 tile = tile_from_linear(flat_tg - mid * tiles, tiles_n, tiles_m);
    int tx = (int)tile.x * SN, ty = (int)tile.y * SM;
    device float *Ab = A + mid * M * K;
    device float *Bb = B + mid * N * K; // physical [N,K]
    device float *Cb = C + mid * M * N;
    auto mA = tensor(Ab, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    auto mB = tensor(Bb, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
    auto mC = tensor(Cb, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
    auto tA = mA.slice(0, ty);
    auto tB = mB.slice(0, tx);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

// =============================================================================
// f32 exact — execution_simdgroup, SM=SN=32 (golden-safe)
// =============================================================================

kernel void matmul2d_tensorops_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &use_interior [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
    auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});

    // Interior offset tensors measured slower on M5 Pro f32 training shapes;
    // gated by host METAL_NATIVE_GEMM_INTERIOR=1.
    bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] = A_stored[K,M]^T @ B[K,N] (TN).
/// Physical A[K,M]: extents {M,K} strides {1,M}. transpose_left → [M,K].
kernel void matmul2d_tensorops_tn_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &use_interior [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        // A physical [K,M] row-major; MPP TN view extents {M,K} stride {1,M}.
        // Tile origin (ty, 0) in that view → pointer A + ty (col-major-ish dim0).
        auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K},
                         array<int, 2>{1, (int)M});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(ty, 0);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] = A[M,K] @ B_stored[N,K]^T (NT).
kernel void matmul2d_tensorops_nt_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &use_interior [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, true, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        // B physical [N,K]; MPP NT view extents {K,N} stride {1,K}.
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                         array<int, 2>{1, (int)K});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(0, tx);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] += A_stored[K,M]^T @ B[K,N] (TN accumulate; no C zero).
kernel void matmul2d_tensorops_tn_accum_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &use_interior [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K},
                         array<int, 2>{1, (int)M});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(ty, 0);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] += A[M,K] @ B_stored[N,K]^T (NT accumulate; no C zero).
kernel void matmul2d_tensorops_nt_accum_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &use_interior [[buffer(8)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = use_interior && (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                         array<int, 2>{1, (int)K});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(0, tx);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// Split-K TN accumulate for one K-partition.
kernel void matmul2d_tensorops_tn_splitk_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &k0 [[buffer(6)]],
    constant uint &k_tile [[buffer(7)]],
    constant uint &tiles_n [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto mmul_mode = matmul2d_descriptor::mode::multiply_accumulate;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false, mmul_mode);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    uint k_len = min(k_tile, K - k0);
    auto mA = tensor(A + k0 * M, dextents<int, 2>{(int)M, (int)k_len}, array<int, 2>{1, (int)M});
    auto mB = tensor(B + k0 * N, dextents<int, 2>{(int)N, (int)k_len}, array<int, 2>{1, (int)N});
    auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});

    auto tA = mA.slice(ty, 0);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

// =============================================================================
// bf16 → f32 accum TN/NT/split-K — execution_simdgroups<4>, 64×32 tiles.
// (The NN bf16 and relaxed-f32 kernels moved to the cooperative-destination
// section at the end of this file.)
// =============================================================================

kernel void matmul2d_tensorops_tn_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K},
                         array<int, 2>{1, (int)M});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(ty, 0);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

kernel void matmul2d_tensorops_nt_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, true, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                         array<int, 2>{1, (int)K});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(0, tx);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] += A_stored[K,M]^T @ B[K,N] (TN accumulate bf16→f32).
kernel void matmul2d_tensorops_tn_accum_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K},
                         array<int, 2>{1, (int)M});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(ty, 0);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// C[M,N] += A[M,K] @ B_stored[N,K]^T (NT accumulate bf16→f32).
kernel void matmul2d_tensorops_nt_accum_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                         array<int, 2>{1, (int)K});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(0, tx);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

kernel void matmul2d_tensorops_tn_splitk_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &k0 [[buffer(6)]],
    constant uint &k_tile [[buffer(7)]],
    constant uint &tiles_n [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 64;
    constexpr int SN = 32;
    constexpr auto mmul_mode = matmul2d_descriptor::mode::multiply_accumulate;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false, mmul_mode);
    matmul2d<desc, execution_simdgroups<4>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    uint k_len = min(k_tile, K - k0);
    auto mA = tensor(A + k0 * M, dextents<int, 2>{(int)M, (int)k_len}, array<int, 2>{1, (int)M});
    auto mB = tensor(B + k0 * N, dextents<int, 2>{(int)N, (int)k_len}, array<int, 2>{1, (int)N});
    auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});

    auto tA = mA.slice(ty, 0);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

// =============================================================================
// bf16 → f32 and relaxed-f32 NN — cooperative destination tensor
// (2026-08-30 tile tune, bench/results/bf16_tile_tune_m5pro_coop.txt).
// The f32 accumulator lives in registers for the whole K reduction and C is
// written exactly once by cT.store() — this retires both the per-BK device
// round-trips over the C tile (the root cause of the 2× gap vs MPS bf16) and
// the host zero_f32(C) pre-pass (every in-bounds element is overwritten).
// Edge tiles run the same coop path over origin-shifted full-extent slices,
// which bounds-check per the MPPTensorOpsMatMul2d.h cooperative example.
// Geometry is selected host-side (gemm.rs nn_coop_kernel): 128×64 sg4
// default, 64×64 sg4 for narrow N, 256×64 sg8 for ≥4096² squares with
// K ≥ 2048. Register cost of the accumulator is SM*SN*4/(32*NSG) B/thread.
// =============================================================================

template <typename ElemT, int SM, int SN, int NSG, bool RELAXED>
inline void mm_nn_coop_f32acc(device ElemT *A, device ElemT *B, device float *C,
                              uint M, uint N, uint K, uint tiles_n,
                              uint tiles_m, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, false, RELAXED,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>,
            metal::remove_addrspace_t<decltype(tB)>, float>();
        // set() wraps the is_valid_element mask check.
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i)
            cT.set(i, 0.0f);
        op.run(tA, tB, cT);
        cT.store(tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>,
            metal::remove_addrspace_t<decltype(tB)>, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i)
            cT.set(i, 0.0f);
        op.run(tA, tB, cT);
        cT.store(tC);
    }
}

#define NN_COOP_KERNEL(NAME, ELEM, SM, SN, NSG, RELAXED)                       \
    kernel void NAME(device ELEM *A [[buffer(0)]],                             \
                     device ELEM *B [[buffer(1)]],                             \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_nn_coop_f32acc<ELEM, SM, SN, NSG, RELAXED>(                         \
            A, B, C, M, N, K, tiles_n, tiles_m, tgpig);                        \
    }

NN_COOP_KERNEL(matmul2d_tensorops_bf16_f32,            bfloat, 128, 64, 4, false)
NN_COOP_KERNEL(matmul2d_tensorops_bf16_f32_64x64_sg4,  bfloat,  64, 64, 4, false)
NN_COOP_KERNEL(matmul2d_tensorops_bf16_f32_256x64_sg8, bfloat, 256, 64, 8, false)
NN_COOP_KERNEL(matmul2d_tensorops_f32_relaxed,            float, 128, 64, 4, true)
NN_COOP_KERNEL(matmul2d_tensorops_f32_relaxed_64x64_sg4,  float,  64, 64, 4, true)
NN_COOP_KERNEL(matmul2d_tensorops_f32_relaxed_256x64_sg8, float, 256, 64, 8, true)
