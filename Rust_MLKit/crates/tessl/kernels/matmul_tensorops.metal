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

/// Phase H bridge: f32 GEMM with `relaxed_precision` (tf32-class).
/// GEMM v2: execution_simdgroups<4>, 64×32 tiles, Morton, BK, static tile extents.
kernel void matmul2d_tensorops_f32_relaxed(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 128;
    constexpr int SN = 64;
    constexpr int BK = 256;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, true,
                            matmul2d_descriptor::mode::multiply);
    constexpr auto desc_bk =
        matmul2d_descriptor(SM, SN, BK, false, false, true,
                            matmul2d_descriptor::mode::multiply_accumulate);
    // Block 0 overwrites rather than accumulates, so C needs no pre-zero.
    constexpr auto desc_bk_first =
        matmul2d_descriptor(SM, SN, BK, false, false, true,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    matmul2d<desc_bk, execution_simdgroups<4>> op_bk;
    matmul2d<desc_bk_first, execution_simdgroups<4>> op_bk_first;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    bool use_bk = interior && ((int)K >= BK);

    if (use_bk) {
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        int k = 0;
        for (; k + BK <= (int)K; k += BK) {
            auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                             array<int, 2>{1, (int)K});
            auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                             array<int, 2>{1, (int)N});
            // use_bk implies K >= BK, so block 0 always runs and seeds C.
            if (k == 0) {
                op_bk_first.run(tA, tB, tC);
            } else {
                op_bk.run(tA, tB, tC);
            }
        }
        if (k < (int)K) {
            int k_rem = (int)K - k;
            auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{k_rem, SM},
                             array<int, 2>{1, (int)K});
            auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, k_rem},
                             array<int, 2>{1, (int)N});
            constexpr auto desc_tail =
                matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, true,
                                    matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<desc_tail, execution_simdgroups<4>> op_tail;
            op_tail.run(tA, tB, tC);
        }
    } else if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}

/// NN relaxed-f32 with a register-resident (cooperative_tensor) accumulator.
///
/// Same fix, same reason as `matmul2d_tensorops_bf16_f32_coop`: the blocked
/// kernel above re-reads and re-writes a device-memory C tile once per K block,
/// so C traffic grows with K/BK while the useful work grows with K. Holding the
/// accumulator in registers across the whole K loop makes C traffic one store.
///
/// Measured with a paired, interleaved A/B (`bench_gemm_coop_ab`) so GPU clock
/// drift cancels: at M=N=2048 the median speedup over the blocked kernel is
/// 1.08x at K=512 rising to 1.10-1.13x at K>=768. Below K=512 the blocked
/// kernel runs at most one full block plus a tail — already a single C store —
/// and this kernel loses (0.93x at K=256), which is what `COOP_MIN_K` encodes.
///
/// The host guarantees the preconditions (M%SM==0, N%SN==0, K%BKC==0,
/// K>=COOP_MIN_K), so there are no ragged or short-K branches here. Kept as a
/// separate kernel rather than a branch inside the blocked one so neither pays
/// for the other's `matmul2d` instantiations.
kernel void matmul2d_tensorops_f32_relaxed_coop(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 128;
    constexpr int SN = 64;
    constexpr int BKC = 128;
    constexpr auto desc_c = matmul2d_descriptor(
        SM, SN, BKC, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_c, execution_simdgroups<4>> op_c;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    auto tA0 = tensor(A + ty * (int)K, dextents<int, 2>{BKC, SM},
                      array<int, 2>{1, (int)K});
    auto tB0 = tensor(B + tx, dextents<int, 2>{SN, BKC},
                      array<int, 2>{1, (int)N});
    auto cT = op_c.template get_destination_cooperative_tensor<
        decltype(tA0), decltype(tB0), float>();
#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) { cT[i] = 0.0f; }
    }
    for (int k = 0; k + BKC <= (int)K; k += BKC) {
        auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BKC, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BKC},
                         array<int, 2>{1, (int)N});
        op_c.run(tA, tB, cT);
    }
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    cT.store(tC);
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
// bf16 → f32 accum — execution_simdgroups<4>, 64×32 tiles
// =============================================================================

kernel void matmul2d_tensorops_bf16_f32(
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
    constexpr int SN = 64;
    constexpr int BK = 256;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, false,
                            matmul2d_descriptor::mode::multiply);
    constexpr auto desc_bk =
        matmul2d_descriptor(SM, SN, BK, false, false, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    // Block 0 overwrites rather than accumulates, so C needs no pre-zero.
    constexpr auto desc_bk_first =
        matmul2d_descriptor(SM, SN, BK, false, false, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    matmul2d<desc_bk, execution_simdgroups<4>> op_bk;
    matmul2d<desc_bk_first, execution_simdgroups<4>> op_bk_first;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    bool use_bk = interior && ((int)K >= BK);

    if (use_bk) {
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        int k = 0;
        for (; k + BK <= (int)K; k += BK) {
            auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                             array<int, 2>{1, (int)K});
            auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                             array<int, 2>{1, (int)N});
            // use_bk implies K >= BK, so block 0 always runs and seeds C.
            if (k == 0) {
                op_bk_first.run(tA, tB, tC);
            } else {
                op_bk.run(tA, tB, tC);
            }
        }
        if (k < (int)K) {
            int k_rem = (int)K - k;
            auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{k_rem, SM},
                             array<int, 2>{1, (int)K});
            auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, k_rem},
                             array<int, 2>{1, (int)N});
            constexpr auto desc_tail =
                matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, false,
                                    matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<desc_tail, execution_simdgroups<4>> op_tail;
            op_tail.run(tA, tB, tC);
        }
    } else if (interior) {
        auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                         array<int, 2>{1, (int)N});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        op.run(tA, tB, tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        op.run(tA, tB, tC);
    }
}
/// NN bf16 with a register-resident (cooperative_tensor) accumulator.
///
/// The blocked kernel accumulates into a device-memory C tile once per K block,
/// so C traffic scales with K/BK — the entire remaining gap versus PyTorch MPS
/// lived there. Holding the accumulator in registers makes C traffic a single
/// store, independent of K.
///
/// Kept separate from `matmul2d_tensorops_bf16_f32` rather than added as a
/// branch: merging them would put four `matmul2d` instantiations in one kernel
/// for no gain, since the host can already decide between them from (M,N,K)
/// alone. (An earlier revision of this comment attributed a ~25% cost to that
/// register pressure; that number came from a stale benchmark binary and was
/// never actually measured. The split stands on the host-side gate, not on it.)
/// The host guarantees the preconditions (M%SM==0, N%SN==0, K%BKC==0,
/// K>=COOP_MIN_K), so this kernel has no ragged or short-K branches.
kernel void matmul2d_tensorops_bf16_f32_coop(
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
    constexpr int SN = 64;
    constexpr int BKC = 128;
    constexpr auto desc_c = matmul2d_descriptor(
        SM, SN, BKC, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_c, execution_simdgroups<4>> op_c;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    auto tA0 = tensor(A + ty * (int)K, dextents<int, 2>{BKC, SM},
                      array<int, 2>{1, (int)K});
    auto tB0 = tensor(B + tx, dextents<int, 2>{SN, BKC},
                      array<int, 2>{1, (int)N});
    auto cT = op_c.template get_destination_cooperative_tensor<
        decltype(tA0), decltype(tB0), float>();
#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) { cT[i] = 0.0f; }
    }
    for (int k = 0; k + BKC <= (int)K; k += BKC) {
        auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BKC, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BKC},
                         array<int, 2>{1, (int)N});
        op_c.run(tA, tB, cT);
    }
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    cT.store(tC);
}

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
    constexpr int SM = 128;
    constexpr int SN = 64;
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
    constexpr int SM = 128;
    constexpr int SN = 64;
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
    constexpr int SM = 128;
    constexpr int SN = 64;
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
    constexpr int SN = 64;
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
