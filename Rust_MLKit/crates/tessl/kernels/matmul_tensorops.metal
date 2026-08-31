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

/// Activation applied by a fused GEMM epilogue.
///
/// Values are ABI: they cross from `tessl::gemm::Activation` as a `uint`, so
/// reordering them silently changes what every caller computes.
enum GemmActivation : uint {
    GEMM_ACT_NONE = 0u,
    GEMM_ACT_RELU = 1u,
    GEMM_ACT_GELU_TANH = 2u,
    GEMM_ACT_SILU = 3u,
};

/// `gelu_pytorch_tanh`, in the *same* formulation as `mlp_gelu_tanh.metal`.
///
/// Deliberately a copy of that kernel's math rather than a fresh derivation:
/// the clamp before cubing and `precise::tanh` are both load-bearing. At `-O2`
/// MSL lowers plain `tanh` to `air.fast_tanh`, which returns NaN past roughly
/// |10|, and the inner term reaches ~301 at |x| = 20. A crate with two
/// different GELUs would be a worse defect than a slow one.
static inline float gemm_gelu_tanh(float x) {
    float xc = clamp(x, -20.0f, 20.0f);
    float x3 = xc * xc * xc;
    float inner = 0.7978845608028654f * (xc + 0.044715f * x3);
    float t = precise::tanh(clamp(inner, -10.0f, 10.0f));
    return 0.5f * xc * (1.0f + t);
}

static inline float gemm_apply_activation(float v, uint act) {
    switch (act) {
        case GEMM_ACT_RELU: return fmax(v, 0.0f);
        case GEMM_ACT_GELU_TANH: return gemm_gelu_tanh(v);
        // silu(x) = x * sigmoid(x), matching `mlp_silu.metal`.
        case GEMM_ACT_SILU: return v / (1.0f + exp(-v));
        default: return v;
    }
}

template <typename ElemT, int SM, int SN, int NSG, bool RELAXED, bool EPILOGUE>
inline void mm_nn_coop_f32acc(device ElemT *A, device ElemT *B, device float *C,
                              uint M, uint N, uint K, uint tiles_n,
                              uint tiles_m, uint tgpig,
                              device const float *bias, float alpha, float beta,
                              uint act) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, false, RELAXED,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;

    // Large grids take a column-panel swizzle (8 tile-rows per band): bounds
    // B-tile rereads to tiles_m/8 full passes — +11% at 4096^3, neutral on
    // tall_k1024/mlp_up, gated off where it measured -3% (square_2048).
    uint2 tile;
    if (tiles_n * tiles_m >= 2048u) {
        constexpr uint PH = 8;
        uint band = tgpig / (PH * tiles_n);
        uint rem = tgpig - band * PH * tiles_n;
        uint local_h = min(PH, tiles_m - band * PH);
        tile = uint2(rem / local_h, band * PH + rem % local_h);
    } else {
        tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    }
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
        if (EPILOGUE) {
            // `beta * C_prev` reuses the accumulate path's trick: a second
            // cooperative tensor loaded from C, combined in registers, so C is
            // read once and written once rather than round-tripped per term.
            if (beta != 0.0f) {
                auto prevT = op.template get_destination_cooperative_tensor<
                    metal::remove_addrspace_t<decltype(tA)>,
                    metal::remove_addrspace_t<decltype(tB)>, float>();
                prevT.load(tC);
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                    if (cT.is_valid_element(i)) cT[i] = cT[i] * alpha + beta * prevT[i];
            } else if (alpha != 1.0f) {
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                    if (cT.is_valid_element(i)) cT[i] *= alpha;
            }
            if (bias) {
                // Row stride 0: every one of the SM rows reads the same SN
                // bias values, so a per-column bias broadcasts through the
                // same `load` path as C with no separate indexing.
                auto tBias = tensor(const_cast<device float *>(bias) + tx,
                                    dextents<int, 2>{SN, SM}, array<int, 2>{1, 0});
                auto biasT = op.template get_destination_cooperative_tensor<
                    metal::remove_addrspace_t<decltype(tA)>,
                    metal::remove_addrspace_t<decltype(tB)>, float>();
                biasT.load(tBias);
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                    if (cT.is_valid_element(i)) cT[i] += biasT[i];
            }
            if (act != GEMM_ACT_NONE) {
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                    if (cT.is_valid_element(i)) cT[i] = gemm_apply_activation(cT[i], act);
            }
        }
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
        if (EPILOGUE) {
            if (beta != 0.0f) {
                auto prevT = op.template get_destination_cooperative_tensor<
                    metal::remove_addrspace_t<decltype(tA)>,
                    metal::remove_addrspace_t<decltype(tB)>, float>();
                prevT.load(tC);
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                    if (cT.is_valid_element(i)) cT[i] = cT[i] * alpha + beta * prevT[i];
            } else if (alpha != 1.0f) {
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                    if (cT.is_valid_element(i)) cT[i] *= alpha;
            }
            if (bias) {
                // Edge tiles take the bounds-checked slice path, so the bias
                // view is built the same way `mC` is and sliced identically.
                auto mBias = tensor(const_cast<device float *>(bias),
                                    dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, 0});
                auto tBias = mBias.slice(tx, ty);
                auto biasT = op.template get_destination_cooperative_tensor<
                    metal::remove_addrspace_t<decltype(tA)>,
                    metal::remove_addrspace_t<decltype(tB)>, float>();
                biasT.load(tBias);
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                    if (cT.is_valid_element(i)) cT[i] += biasT[i];
            }
            if (act != GEMM_ACT_NONE) {
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                    if (cT.is_valid_element(i)) cT[i] = gemm_apply_activation(cT[i], act);
            }
        }
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
        mm_nn_coop_f32acc<ELEM, SM, SN, NSG, RELAXED, false>(                   \
            A, B, C, M, N, K, tiles_n, tiles_m, tgpig, nullptr, 1.0f, 0.0f, 0u);\
    }

NN_COOP_KERNEL(matmul2d_tensorops_bf16_f32,            bfloat, 128, 64, 4, false)
NN_COOP_KERNEL(matmul2d_tensorops_bf16_f32_64x64_sg4,  bfloat,  64, 64, 4, false)
NN_COOP_KERNEL(matmul2d_tensorops_f32_relaxed,            float, 128, 64, 4, true)
NN_COOP_KERNEL(matmul2d_tensorops_f32_relaxed_64x64_sg4,  float,  64, 64, 4, true)

/// Epilogue variants: `C = act(alpha * A@B + beta * C_prev + bias)`.
///
/// A separate entry point rather than extra parameters on the kernels above.
/// Metal faults on a declared-but-unbound buffer, so widening the existing
/// signatures would force every current caller to bind four operands it does
/// not use — and the tuned geometries above are the crate's most measured
/// code. `EPILOGUE` is a template parameter, so these share one source with
/// them and the plain path compiles to exactly what it did before.
#define NN_COOP_EPI_KERNEL(NAME, ELEM, SM, SN, NSG, RELAXED)                   \
    kernel void NAME(device ELEM *A [[buffer(0)]],                             \
                     device ELEM *B [[buffer(1)]],                             \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     device const float *bias [[buffer(8)]],                   \
                     constant float &alpha [[buffer(9)]],                      \
                     constant float &beta [[buffer(10)]],                      \
                     constant uint &act [[buffer(11)]],                        \
                     constant uint &has_bias [[buffer(12)]],                   \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_nn_coop_f32acc<ELEM, SM, SN, NSG, RELAXED, true>(                   \
            A, B, C, M, N, K, tiles_n, tiles_m, tgpig,                         \
            has_bias ? bias : nullptr, alpha, beta, act);                      \
    }

NN_COOP_EPI_KERNEL(matmul2d_tensorops_bf16_f32_epi,    bfloat, 128, 64, 4, false)
NN_COOP_EPI_KERNEL(matmul2d_tensorops_f32_relaxed_epi,  float, 128, 64, 4, true)

/// TN / NT bf16 GEMMs — cooperative destination tensor (2026-08-30 round 2,
/// bench/results/bf16_tnnt_coop_m5pro.txt): register accumulator, C touched
/// once (plus one load for ACCUM), descriptor transposes per lane. Interior
/// tiles use offset pointer tensors; edges the bounds-checked slice path.
/// ACCUM adds the prior C via the header's cooperative bias pattern
/// (load-add-store): one C read + one C write total, measured 1.4-1.5x over
/// multiply_accumulate. Geometry: 128x64 sg4 (plain), 64x64 sg4 (accum).
template <int SM, int SN, int NSG, bool ACCUM>
inline void mm_tn_coop_bf16(device bfloat *A, device bfloat *B, device float *C,
                            uint M, uint N, uint K, uint tiles_n, uint tiles_m,
                            uint tgpig) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, true, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;

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
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>,
            metal::remove_addrspace_t<decltype(tB)>, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i)
            cT.set(i, 0.0f);
        op.run(tA, tB, cT);
        if (ACCUM) {
            auto prevT = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(tA)>,
                metal::remove_addrspace_t<decltype(tB)>, float>();
            prevT.load(tC);
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i))
                    cT[i] += prevT[i];
            }
        }
        cT.store(tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
        auto mB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(ty, 0);
        auto tB = mB.slice(tx, 0);
        auto tC = mC.slice(tx, ty);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>,
            metal::remove_addrspace_t<decltype(tB)>, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i)
            cT.set(i, 0.0f);
        op.run(tA, tB, cT);
        if (ACCUM) {
            auto prevT = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(tA)>,
                metal::remove_addrspace_t<decltype(tB)>, float>();
            prevT.load(tC);
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i))
                    cT[i] += prevT[i];
            }
        }
        cT.store(tC);
    }
}

template <int SM, int SN, int NSG, bool ACCUM>
inline void mm_nt_coop_bf16(device bfloat *A, device bfloat *B, device float *C,
                            uint M, uint N, uint K, uint tiles_n, uint tiles_m,
                            uint tgpig) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, true, false,
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
        auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                         array<int, 2>{1, (int)K});
        auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                         array<int, 2>{1, (int)N});
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>,
            metal::remove_addrspace_t<decltype(tB)>, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i)
            cT.set(i, 0.0f);
        op.run(tA, tB, cT);
        if (ACCUM) {
            auto prevT = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(tA)>,
                metal::remove_addrspace_t<decltype(tB)>, float>();
            prevT.load(tC);
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i))
                    cT[i] += prevT[i];
            }
        }
        cT.store(tC);
    } else {
        auto mA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
        auto mB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
        auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
        auto tA = mA.slice(0, ty);
        auto tB = mB.slice(0, tx);
        auto tC = mC.slice(tx, ty);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>,
            metal::remove_addrspace_t<decltype(tB)>, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i)
            cT.set(i, 0.0f);
        op.run(tA, tB, cT);
        if (ACCUM) {
            auto prevT = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(tA)>,
                metal::remove_addrspace_t<decltype(tB)>, float>();
            prevT.load(tC);
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i))
                    cT[i] += prevT[i];
            }
        }
        cT.store(tC);
    }
}

#define TN_NT_COOP_KERNEL(NAME, IMPL, SM, SN, NSG, ACCUM)                      \
    kernel void NAME(device bfloat *A [[buffer(0)]],                           \
                     device bfloat *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        IMPL<SM, SN, NSG, ACCUM>(A, B, C, M, N, K, tiles_n, tiles_m, tgpig);   \
    }

TN_NT_COOP_KERNEL(matmul2d_tensorops_tn_bf16_f32,       mm_tn_coop_bf16, 128, 64, 4, false)
TN_NT_COOP_KERNEL(matmul2d_tensorops_nt_bf16_f32,       mm_nt_coop_bf16, 128, 64, 4, false)
TN_NT_COOP_KERNEL(matmul2d_tensorops_tn_accum_bf16_f32, mm_tn_coop_bf16,  64, 64, 4, true)
TN_NT_COOP_KERNEL(matmul2d_tensorops_nt_accum_bf16_f32, mm_nt_coop_bf16,  64, 64, 4, true)
