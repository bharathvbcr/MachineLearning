// FA-2 causal GQA attention for CUDA.
//
// This is a port of this repo's Metal kernels --
// `Rust_MLKit/arch_02_value_resid/metal-native/kernels/flash_attn_{fwd,bwd}*.metal`
// -- and it keeps their contract exactly, so the two backends stay comparable
// line by line:
//
//   Q [B,T,H,D]   K,V [B,T,Hkv,D]   O [B,T,H,D]   L [B,H,T]
//   causal; GQA via `group = H/Hkv`, `hkv = h/group`; caller-supplied `scale`
//   L      = m + log(l)                     (the taped LSE the backward re-reads)
//   Delta  = sum_d dO*O                     [B,H,T]
//   p      = exp(score*scale - L)
//   dss    = p*(dp - Delta)*scale
//   dQ += dss*K,  dK += dss*Q,  dV += p*dO
//
// Two contract differences from the Metal side, both deliberate:
//
//   * O follows Q's dtype (SDPA's convention) instead of always being f32.
//     Every accumulator -- scores, softmax state, dQ/dK/dV -- is f32 regardless,
//     which is the part suite 19's "accumulation needs fp32" lesson was about.
//   * GQA is native. `nanolab.mixers.Attention` currently calls
//     `repeat_interleave` to widen K/V to H heads before SDPA; this kernel
//     indexes the KV head directly, so that materialisation disappears.
//
// CUDA-specific work, beyond a literal transcription:
//
//  1. `__restrict__` on every pointer. Metal audit 7 found the generic backward
//     re-reading K/V inside the t_k loop because `device const float *K` may
//     alias `device float *dK`, so the compiler could not hoist the load.
//     `__restrict__` states the non-aliasing directly -- it both hoists those
//     loads and lets them take the read-only cache path.
//  2. Compile-time HEAD_DIM (audit 8's ~11x finding). Every `d` loop is a
//     `#pragma unroll` over a template parameter, so the accumulators are real
//     registers rather than dynamically-indexed local memory.
//  3. Base-2 softmax. `exp2f` is one SFU instruction; `expf` is `exp2f` plus a
//     multiply by log2(e). Folding log2(e) into the QK scale takes that multiply
//     out of the inner loop. The taped L is still a *natural* log, so the tape
//     remains interchangeable with the Metal one.
//  4. Rescale once per 8 keys, not once per key. The Metal inner loop runs
//     `acc[d] = acc[d]*alpha + p*V[d]` for every key -- two operations per
//     element per key. Taking the running max over a sub-tile of KSUB keys first
//     turns that into one rescale plus KSUB plain FMAs. Any partition of the key
//     axis is a valid online-softmax step, so this is exact, not an approximation.
//  5. Shared-memory staging in the backward as well. The Metal `_row` backward
//     kernels stream K/V (dQ) or Q/dO (dKV) out of device memory for every
//     (query, key) pair; here every tile is staged once per block -- and on the
//     tensor-core path, staged one tile ahead of where it is needed.
//  6. Deterministic gradients, on both backward paths: two passes (dQ over key
//     tiles, dK/dV over query tiles), no atomics. A single fused pass with
//     `atomicAdd` on dQ would halve the score recomputation, but it makes dQ
//     run-to-run non-reproducible, which would break golden/exact-gate style
//     comparisons.
//
// There are two implementations of each direction:
//
//   * FMA -- scalar loops, one thread per row. Every dtype (f32/f16/bf16) and
//     every supported head dim. The reference implementation and the fallback.
//   * tensor core -- `mma.sync.aligned.m16n8k16`, one warp per 16 rows, f16/bf16
//     on sm_80+. Forward at head_dim 32/64/128; backward at 32/64, which is
//     where the register file allows it.
//
// Both produce the same tape and the same gradients up to accumulation order, so
// the two are interchangeable in either combination.
//
// The tensor-core kernels stage their tiles with `cp.async` and double buffer:
// the copy for tile n+1 is issued before the wait for tile n, so the next tile
// is in flight across global memory while the current one is being multiplied.
// That is also why nothing is staged in two orientations any more -- cp.async
// copies bytes and cannot transpose, so a transposed staging copy would be a
// hole in the pipeline. The products that want the other orientation read the
// same buffer down a column instead (`load_b_strided`).
//
// What is NOT here, stated so nobody has to guess: no shared-memory swizzling
// (the staging is plain row padding), no split-K, no `ldmatrix`, no more than
// two stages in the pipeline. torch's own SDPA on CUDA dispatches to a CUTLASS
// FlashAttention-2 that has all of those, and this kernel is not going to beat
// it on a shape SDPA supports. What it is for: the GQA path without
// `repeat_interleave`, an algebraic twin of the Metal kernel for cross-backend
// parity work, deterministic gradients, and a base for the masked variants (SWA
// + attention sinks) that SDPA can only serve by materialising a dense [T,T] mask.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_pipeline.h>   // __pipeline_memcpy_async / _commit / _wait_prior
#endif

namespace {

// exp2-space softmax constants (see note 3 above).
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

// Keys per online-softmax rescale (note 4). Must divide every BC below.
constexpr int KSUB = 8;

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }

/// Offset of element (b, t, head, 0) in a [B, T, n_head, D] contiguous tensor.
/// 64-bit on purpose: B*T*H*D passes 2^31 at long context well before it passes
/// any memory limit, and a silently wrapped index is the worst kind of bug.
__device__ __forceinline__ long long row_off(int b, int t, int T, int head,
                                             int n_head, int D) {
  return ((static_cast<long long>(b) * T + t) * n_head + head) *
         static_cast<long long>(D);
}

/// Offset of (b, h, t) in the [B, H, T] LSE / Delta tapes.
__device__ __forceinline__ long long lse_off(int b, int h, int H, int t, int T) {
  return (static_cast<long long>(b) * H + h) * T + t;
}

/// Tile geometry. Shared memory is the occupancy limiter here exactly as
/// threadgroup memory was on Metal (audit 8, note 2), so the key tile is sized
/// to a ~8 KB staging budget (K and V together) and clamped to a sensible range.
///
/// One template parameter, deliberately: `__launch_bounds__` takes a comma-
/// separated argument list, so `Tiles<scalar_t, HEAD_DIM>::BR` written there
/// would be parsed as two arguments. `kRowBytes` is HEAD_DIM * sizeof(scalar_t)
/// -- one key's worth of staging -- which is all the geometry depends on.
template <int kRowBytes>
struct Tiles {
  static constexpr int kRaw = 8192 / (2 * kRowBytes);
  static constexpr int BC = kRaw < 32 ? 32 : (kRaw > 64 ? 64 : (kRaw / KSUB) * KSUB);
  static constexpr int BR = 64;   // query rows per forward / dQ block
  static constexpr int BQ = BC;   // query rows staged per dK/dV block
  static_assert(BC % KSUB == 0, "key tile must be a whole number of sub-tiles");
  static_assert(BR % KSUB == 0, "query tile must be a whole number of sub-tiles");
};

template <typename scalar_t, int HEAD_DIM>
using TilesFor = Tiles<HEAD_DIM * static_cast<int>(sizeof(scalar_t))>;

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

/// One block per (q_block, b*h); one thread per query row.
template <typename scalar_t, int HEAD_DIM>
__global__ __launch_bounds__(Tiles<HEAD_DIM * static_cast<int>(sizeof(scalar_t))>::BR)
void flash_fwd_kernel(const scalar_t* __restrict__ Q,
                      const scalar_t* __restrict__ K,
                      const scalar_t* __restrict__ V,
                      scalar_t* __restrict__ O,
                      float* __restrict__ L,
                      const int T, const int H, const int Hkv,
                      const float scale) {
  using Tile = TilesFor<scalar_t, HEAD_DIM>;
  constexpr int BR = Tile::BR;
  constexpr int BC = Tile::BC;

  __shared__ scalar_t Ks[BC * HEAD_DIM];
  __shared__ scalar_t Vs[BC * HEAD_DIM];

  const int tid = threadIdx.x;
  const int bh = blockIdx.y;
  const int h = bh % H;
  const int b = bh / H;
  const int group = H / Hkv;
  const int hkv = h / group;

  const int t_q0 = blockIdx.x * BR;
  if (t_q0 >= T) return;
  const int t_q = t_q0 + tid;
  const bool row_valid = t_q < T;

  const float qk_scale = scale * kLog2e;   // score -> log2 space, one multiply

  float q_reg[HEAD_DIM];
  float acc[HEAD_DIM];
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    q_reg[d] = 0.0f;
    acc[d] = 0.0f;
  }
  if (row_valid) {
    const scalar_t* qp = Q + row_off(b, t_q, T, h, H, HEAD_DIM);
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) q_reg[d] = static_cast<float>(qp[d]);
  }

  float m_i = -INFINITY;   // running row max, in log2 space
  float l_i = 0.0f;        // running denominator

  // Causal: this block never needs a key past its own last query row.
  const int t_q_max = imin(t_q0 + BR, T) - 1;
  const int n_k_blocks = (t_q_max / BC) + 1;

  for (int kb = 0; kb < n_k_blocks; ++kb) {
    const int t_k0 = kb * BC;
    const int n_k = imin(BC, T - t_k0);

    // Stage the tile. Consecutive threads take consecutive `d`, so the global
    // reads are fully coalesced. Rows past the tail are zero-filled rather than
    // left stale: phase 2 below multiplies them by p == 0, and 0 * NaN is NaN.
    for (int i = tid; i < BC * HEAD_DIM; i += BR) {
      const int tk = i / HEAD_DIM;
      const int d = i - tk * HEAD_DIM;
      if (tk < n_k) {
        const long long off = row_off(b, t_k0 + tk, T, hkv, Hkv, HEAD_DIM) + d;
        Ks[i] = K[off];
        Vs[i] = V[off];
      } else {
        Ks[i] = static_cast<scalar_t>(0.0f);
        Vs[i] = static_cast<scalar_t>(0.0f);
      }
    }
    __syncthreads();

    // Keys this row may see in this tile: causal cut, clipped to the tail.
    const int k_hi = imin(n_k, t_q - t_k0 + 1);
    if (row_valid && k_hi > 0) {
      for (int sub = 0; sub < k_hi; sub += KSUB) {
        float s[KSUB];
        float m_sub = -INFINITY;
#pragma unroll
        for (int j = 0; j < KSUB; ++j) {
          const int tk = sub + j;
          if (tk < k_hi) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
              dot = fmaf(q_reg[d], static_cast<float>(Ks[tk * HEAD_DIM + d]), dot);
            }
            s[j] = dot * qk_scale;
            m_sub = fmaxf(m_sub, s[j]);
          } else {
            s[j] = -INFINITY;
          }
        }

        // One rescale for the whole sub-tile, then KSUB plain FMAs per element.
        // `m_sub` is always finite here: `sub < k_hi` means j == 0 is live, so
        // the m_i == m_sub == -inf case that would make `alpha` a NaN cannot
        // arise. (On the very first sub-tile m_i is -inf and m_sub is finite,
        // which gives alpha == 0 and zeroes the untouched accumulator -- right.)
        const float m_new = fmaxf(m_i, m_sub);
        const float alpha = exp2f(m_i - m_new);
        l_i *= alpha;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] *= alpha;
#pragma unroll
        for (int j = 0; j < KSUB; ++j) {
          const float p = exp2f(s[j] - m_new);   // exactly 0 for masked lanes
          l_i += p;
          // In bounds without a check: `sub` is a multiple of KSUB and less than
          // k_hi <= BC, and BC is a whole number of sub-tiles (static_assert), so
          // sub <= BC - KSUB and sub + j <= BC - 1. Masked and past-the-tail
          // entries are still *read* -- which is why the staging loop zero-fills
          // the tail rather than leaving whatever the last tile put there.
          const int base = (sub + j) * HEAD_DIM;
#pragma unroll
          for (int d = 0; d < HEAD_DIM; ++d) {
            acc[d] = fmaf(p, static_cast<float>(Vs[base + d]), acc[d]);
          }
        }
        m_i = m_new;
      }
    }
    __syncthreads();
  }

  if (row_valid) {
    const float inv_l = 1.0f / l_i;
    scalar_t* op = O + row_off(b, t_q, T, h, H, HEAD_DIM);
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      op[d] = static_cast<scalar_t>(acc[d] * inv_l);
    }
    // Back to a natural log so the tape matches the Metal kernel's L exactly.
    L[lse_off(b, h, H, t_q, T)] = m_i * kLn2 + logf(l_i);
  }
}

// ---------------------------------------------------------------------------
// forward, on tensor cores (mma.sync.aligned.m16n8k16)
// ---------------------------------------------------------------------------
//
// Same maths as `flash_fwd_kernel`, same tape, same causal GQA contract -- the
// QK^T and PV products move onto the tensor cores and the online softmax stays
// in registers between them.
//
// Warp-level shape: one warp owns 16 query rows and runs
//   S = Q K^T  as (BC/8) x (D/16) mma tiles,
//   O += P V   as (D/8)  x (BC/16) mma tiles,
// with a block of 4 warps covering BR = 64 query rows and sharing the staged
// K / V^T tiles. f16 and bf16 only: there is no m16n8k16 for f32, and the tf32
// path is a different instruction (m16n8k8) with a different fragment layout,
// so f32 keeps the FMA kernel rather than getting a half-checked third path.
//
// THE FRAGMENT LAYOUT BELOW IS THE ONE THING HERE THAT CANNOT BE CHECKED
// WITHOUT THE HARDWARE. It is the PTX ISA's mapping for
// `mma.sync.aligned.m16n8k16.row.col`, with
//
//     gid = laneid >> 2   (0..7)      tig = laneid & 3   (0..3)
//
//   A (16 m x 16 k, f16/bf16, 4 x b32 = 8 elements per lane)
//     reg0 -> (row gid,   col 2*tig    ), (row gid,   col 2*tig + 1)
//     reg1 -> (row gid+8, col 2*tig    ), (row gid+8, col 2*tig + 1)
//     reg2 -> (row gid,   col 2*tig + 8), (row gid,   col 2*tig + 9)
//     reg3 -> (row gid+8, col 2*tig + 8), (row gid+8, col 2*tig + 9)
//
//   B (16 k x 8 n, f16/bf16, 2 x b32 = 4 elements per lane)
//     reg0 -> (row 2*tig,     col gid), (row 2*tig + 1, col gid)
//     reg1 -> (row 2*tig + 8, col gid), (row 2*tig + 9, col gid)
//
//   C/D (16 m x 8 n, f32, 4 floats per lane)
//     c0 -> (row gid,   col 2*tig)      c1 -> (row gid,   col 2*tig + 1)
//     c2 -> (row gid+8, col 2*tig)      c3 -> (row gid+8, col 2*tig + 1)
//
// `mma_probe_kernel` at the bottom of this section multiplies known matrices
// through exactly these helpers so the mapping is verified on the device before
// anything trusts it -- `nanolab.flash_cuda` runs it once at load and refuses
// the tensor-core path if it disagrees with a plain matmul. An assumption that
// cannot be checked here gets checked there; it does not get assumed.
//
// Three consequences of that table are worth stating, because they are why this
// shape was picked and what makes the kernel cheap:
//
//  * C and A agree. Two adjacent 16x8 score tiles concatenate straight into one
//    16x16 A fragment -- (c0,c1,c2,c3) of the low tile become A's reg0/reg1, and
//    the high tile's become reg2/reg3. Turning S into P's operand is a pack, not
//    a shuffle through shared memory.
//  * Row reductions are cheap. A lane holds rows gid and gid+8 and nothing else,
//    so a row max/sum is a local reduce over its own floats followed by
//    __shfl_xor over the 4 lanes of the group (masks 1 and 2). No smem, no
//    barrier, no cross-row contamination.
//  * K wants its natural [key][d] layout and V wants to be transposed to
//    [d][key]: with .row.col, both then read as one *contiguous 32-bit pair* per
//    lane. That is the only reason V^T is staged.

/// Two f32 values -> one b32 of two f16/bf16, in A/B operand order (low half
/// first). alignas(4) so the reinterpret is aligned by construction.
template <typename scalar_t>
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  alignas(4) scalar_t v[2] = {static_cast<scalar_t>(lo), static_cast<scalar_t>(hi)};
  return *reinterpret_cast<const uint32_t*>(v);
}

/// A contiguous pair of scalar_t at `p` as one b32 operand register. Callers
/// only ever pass even element offsets, so this is 4-byte aligned.
__device__ __forceinline__ uint32_t ld2(const void* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

/// D = A*B + C for one 16x8x16 tile, accumulating in place.
template <typename scalar_t>
__device__ __forceinline__ void mma_m16n8k16(const uint32_t (&a)[4],
                                             const uint32_t (&b)[2],
                                             float (&d)[4]);

template <>
__device__ __forceinline__ void mma_m16n8k16<at::Half>(const uint32_t (&a)[4],
                                                       const uint32_t (&b)[2],
                                                       float (&d)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#elif defined(__CUDA_ARCH__)
  // sm_75 and below have no m16n8k16. Trap rather than return something: the
  // host gate should have kept us off this path, and a wrong answer here would
  // be a silent training corruption.
  __trap();
#else
  // Host translation unit only -- nvcc never runs this, and neither does any
  // device. It exists so the file can be parsed by a plain C++ front end on a
  // machine with no CUDA toolkit (see the syntax-check note in the header).
  (void)a; (void)b;
  d[0] = d[1] = d[2] = d[3] = 0.0f;
#endif
}

template <>
__device__ __forceinline__ void mma_m16n8k16<at::BFloat16>(const uint32_t (&a)[4],
                                                           const uint32_t (&b)[2],
                                                           float (&d)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#elif defined(__CUDA_ARCH__)
  __trap();
#else
  (void)a; (void)b;
  d[0] = d[1] = d[2] = d[3] = 0.0f;
#endif
}

/// Whether tile staging goes through cp.async. Compile time, not a runtime knob:
/// the mma kernels are sm_80+ only, which is exactly where cp.async exists, so
/// there is no configuration to expose -- but the synchronous path stays
/// compiled so that flipping one constant bisects a suspected pipeline problem.
[[maybe_unused]] constexpr bool kAsyncStaging = true;

/// One 16-byte chunk, global -> shared. Under cp.async the copy is *issued* and
/// the thread moves on, so the next tile is in flight while the current one is
/// being multiplied; the data is not there until a matching wait.
///
/// Both pointers must be 16-byte aligned. That is why every staging buffer is
/// `alignas(16)` and why the row padding is a multiple of 8 halves.
__device__ __forceinline__ void copy16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (kAsyncStaging) {
    __pipeline_memcpy_async(dst, src, 16);
  } else {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
  }
#elif defined(__CUDA_ARCH__)
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#else
  (void)dst;
  (void)src;
#endif
}

/// Close the current batch of issued copies.
__device__ __forceinline__ void copy_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (kAsyncStaging) __pipeline_commit();
#endif
}

/// Wait until at most `kKeep` committed batches are still in flight. kKeep=1
/// leaves the prefetch of the *next* tile outstanding while the current one is
/// ready; kKeep=0 drains. A compile-time constant because the underlying
/// `cp.async.wait_group` takes an immediate.
template <int kKeep>
__device__ __forceinline__ void copy_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (kAsyncStaging) __pipeline_wait_prior(kKeep);
#endif
}

/// Stage one [TILE, HEAD_DIM] tile of a [B,T,head,D] tensor into a padded
/// shared buffer, as whole 16-byte chunks.
///
/// Rows past the tail clamp to the last live row rather than zero-filling:
/// cp.async copies bytes and cannot synthesise zeros, and every consumer
/// already drops those columns (`col < T`), so all the buffer owes them is a
/// *finite* value. Nothing is ever read out of range -- `n_live` is at least 1
/// whenever the tile exists.
template <typename scalar_t, int HEAD_DIM, int TILE, int PAD, int THREADS>
__device__ __forceinline__ void stage_tile(scalar_t* dst, const scalar_t* src,
                                           int b, int t0, int n_live, int T,
                                           int head, int n_head, int tid) {
  constexpr int kChunk = 16 / static_cast<int>(sizeof(scalar_t));
  constexpr int kPerRow = HEAD_DIM / kChunk;
  static_assert(HEAD_DIM % kChunk == 0,
                "head_dim must be a whole number of 16-byte copies");
  static_assert((HEAD_DIM + PAD) % kChunk == 0,
                "padded row stride must keep every row 16-byte aligned");
  for (int i = tid; i < TILE * kPerRow; i += THREADS) {
    const int row = i / kPerRow;
    const int c = (i - row * kPerRow) * kChunk;
    const int live = imin(row, n_live - 1);
    copy16(dst + row * (HEAD_DIM + PAD) + c,
           src + row_off(b, t0 + live, T, head, n_head, HEAD_DIM) + c);
  }
}

/// Two scalar_t -> one b32, low half first. Bit copy, no rounding.
template <typename scalar_t>
__device__ __forceinline__ uint32_t pack_bits(scalar_t lo, scalar_t hi) {
  alignas(4) scalar_t v[2] = {lo, hi};
  return *reinterpret_cast<const uint32_t*>(v);
}

/// B operand for one 16x8 tile, read from a staged matrix laid out so that the
/// *n* index selects a row and the *k* index runs along it. Every B operand in
/// this file is loaded this way; what changes is which matrix is staged in which
/// orientation:
///
///   product          B is    staged        n index   k index   loader
///   S   = Q K^T      K^T     K  [key][d]   key       d         load_b
///   dP  = dO V^T     V^T     V  [key][d]   key       d         load_b
///   O  += P V        V       V  [key][d]   d         key       load_b_strided
///   dQ += dS K       K       K  [key][d]   d         key       load_b_strided
///   dK += dS^T Q     Q       Q  [query][d] d         query     load_b_strided
///   dV += P^T dO     dO      dO [query][d] d         query     load_b_strided
///
/// Every matrix is staged exactly once, in its natural [row][d] layout. The
/// products that want the other orientation get it from `load_b_strided`, which
/// walks the same buffer down a column instead of along a row.
template <typename scalar_t>
__device__ __forceinline__ void load_b(const scalar_t* base, int stride,
                                       int ntile, int gid, int k0, int tig,
                                       uint32_t (&b)[2]) {
  const scalar_t* p = base + (ntile * 8 + gid) * stride + k0 + tig * 2;
  b[0] = ld2(p);
  b[1] = ld2(p + 8);
}

/// B operand read from a matrix staged in the *other* orientation: the k index
/// walks rows and the n index picks a column, so a register's two elements are
/// a row apart rather than adjacent.
///
/// This is what lets every matrix be staged once. The alternative -- writing a
/// transposed copy into shared memory so these become contiguous pairs -- costs
/// a whole extra tile of scattered shared-memory writes per stage, against two
/// extra 16-bit loads per operand register here. It also leaves every staging
/// buffer a plain contiguous global->shared copy, which is the precondition for
/// cp.async carrying all of it rather than half.
template <typename scalar_t>
__device__ __forceinline__ void load_b_strided(const scalar_t* base, int stride,
                                               int ntile, int gid, int k0,
                                               int tig, uint32_t (&b)[2]) {
  const scalar_t* p = base + (k0 + tig * 2) * stride + ntile * 8 + gid;
  b[0] = pack_bits<scalar_t>(p[0], p[stride]);
  b[1] = pack_bits<scalar_t>(p[8 * stride], p[9 * stride]);
}

/// A operand for all k-chunks of two [B,T,H,D] rows, straight from global.
/// A lane owns rows gid and gid+8 of the 16-row tile; an out-of-range row reads
/// nothing and stays zero.
template <typename scalar_t, int KCHUNKS>
__device__ __forceinline__ void load_a_rows(const scalar_t* row_a, bool ok_a,
                                            const scalar_t* row_b, bool ok_b,
                                            int tig, uint32_t (&a)[KCHUNKS][4]) {
#pragma unroll
  for (int kc = 0; kc < KCHUNKS; ++kc) {
    const int c0 = kc * 16 + tig * 2;
    a[kc][0] = ok_a ? ld2(row_a + c0) : 0u;
    a[kc][1] = ok_b ? ld2(row_b + c0) : 0u;
    a[kc][2] = ok_a ? ld2(row_a + c0 + 8) : 0u;
    a[kc][3] = ok_b ? ld2(row_b + c0 + 8) : 0u;
  }
}

/// Two adjacent 16x8 accumulator tiles -> one 16x16 A operand.
///
/// This is the identity that makes the whole scheme cheap: the C layout's
/// (c0,c1,c2,c3) for columns 0..7 are exactly A's reg0/reg1, and the next tile's
/// are reg2/reg3. Whatever a warp just computed as an accumulator -- P in the
/// forward, dS or P in the backward -- becomes the next product's A operand with
/// a pack and no data movement.
template <typename scalar_t>
__device__ __forceinline__ void pack_a_from_acc(const float (&lo)[4],
                                                const float (&hi)[4],
                                                uint32_t (&a)[4]) {
  a[0] = pack2<scalar_t>(lo[0], lo[1]);
  a[1] = pack2<scalar_t>(lo[2], lo[3]);
  a[2] = pack2<scalar_t>(hi[0], hi[1]);
  a[3] = pack2<scalar_t>(hi[2], hi[3]);
}

/// Tile geometry for the tensor-core forward. Fixed rather than derived: the
/// mma shape already pins the interesting factors (16 rows per warp, k in
/// chunks of 16, n in chunks of 8), and BC=32 keeps the score accumulator --
/// (BC/8)*4 floats per lane -- inside the register budget at head_dim 128.
///
/// Expect head_dim 128 to be occupancy-limited even so: the output accumulator
/// alone is (HEAD_DIM/8)*4 = 64 floats per lane, plus 32 for the Q fragments and
/// 16 for the scores, so ~112 registers are spoken for before any temporaries.
/// That is a known cost of holding O in registers, not an oversight -- but it is
/// the first thing to measure with `ncu` if head_dim 128 underperforms.
struct MmaTiles {
  static constexpr int kWarps = 4;
  static constexpr int kThreads = kWarps * 32;
  // BR is the block's *m* dimension: query rows in the forward and the dQ pass,
  // key rows in the dK/dV pass, where keys are what the warp iterates over.
  static constexpr int BR = kWarps * 16;
  static constexpr int BC = 32;            // key tile, forward and dQ
  static constexpr int BQ = 32;            // query tile, dK/dV
  static constexpr int kPad = 8;           // halves of shared-memory row padding
  static_assert(BC % 16 == 0, "key tile must be a whole number of mma k-chunks");
  static_assert(BQ % 16 == 0, "query tile must be a whole number of mma k-chunks");
};

/// FA-2 forward on tensor cores. Block per (q_block, b*h), 4 warps.
template <typename scalar_t, int HEAD_DIM>
__global__ __launch_bounds__(MmaTiles::kThreads)
void flash_fwd_mma_kernel(const scalar_t* __restrict__ Q,
                          const scalar_t* __restrict__ K,
                          const scalar_t* __restrict__ V,
                          scalar_t* __restrict__ O,
                          float* __restrict__ L,
                          const int T, const int H, const int Hkv,
                          const float scale) {
  constexpr int BR = MmaTiles::BR;
  constexpr int BC = MmaTiles::BC;
  constexpr int PAD = MmaTiles::kPad;
  constexpr int KCHUNKS = HEAD_DIM / 16;   // k-chunks of the QK^T product
  constexpr int NS = BC / 8;               // n-tiles of the score matrix
  constexpr int NO = HEAD_DIM / 8;         // n-tiles of the output
  constexpr int PCHUNKS = BC / 16;         // k-chunks of the PV product
  static_assert(HEAD_DIM % 16 == 0, "head_dim must be a whole number of k-chunks");

  // Double buffered: the copy for tile kb+1 is in flight while tile kb is being
  // multiplied. Both are staged naturally; the P V product reads V down a
  // column (load_b_strided) rather than from a transposed copy.
  alignas(16) __shared__ scalar_t Ks[2][BC * (HEAD_DIM + PAD)];   // [key][d]
  alignas(16) __shared__ scalar_t Vs[2][BC * (HEAD_DIM + PAD)];   // [key][d]

  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  const int bh = blockIdx.y;
  const int h = bh % H;
  const int b = bh / H;
  const int group = H / Hkv;
  const int hkv = h / group;

  const int t_q0 = blockIdx.x * BR;
  if (t_q0 >= T) return;
  const int q_base = t_q0 + wid * 16;        // this warp's first query row
  const int row_a = q_base + gid;            // the two query rows this lane owns
  const int row_b = q_base + gid + 8;
  const bool ok_a = row_a < T;
  const bool ok_b = row_b < T;

  const float qk_scale = scale * kLog2e;

  // Q fragments, loaded once. Rows past T read nothing and stay zero; their
  // output is never stored.
  uint32_t q_frag[KCHUNKS][4];
  load_a_rows<scalar_t, KCHUNKS>(Q + row_off(b, ok_a ? row_a : 0, T, h, H, HEAD_DIM),
                                 ok_a,
                                 Q + row_off(b, ok_b ? row_b : 0, T, h, H, HEAD_DIM),
                                 ok_b, tig, q_frag);

  float acc_o[NO][4];
#pragma unroll
  for (int n = 0; n < NO; ++n) {
#pragma unroll
    for (int e = 0; e < 4; ++e) acc_o[n][e] = 0.0f;
  }
  float m_a = -INFINITY, m_b = -INFINITY;   // running max for the lane's 2 rows
  float l_a = 0.0f, l_b = 0.0f;

  const int t_q_max = imin(t_q0 + BR, T) - 1;
  const int n_k_blocks = (t_q_max / BC) + 1;

  // Prime the pipeline with tile 0.
  stage_tile<scalar_t, HEAD_DIM, BC, PAD, MmaTiles::kThreads>(
      Ks[0], K, b, 0, imin(BC, T), T, hkv, Hkv, tid);
  stage_tile<scalar_t, HEAD_DIM, BC, PAD, MmaTiles::kThreads>(
      Vs[0], V, b, 0, imin(BC, T), T, hkv, Hkv, tid);
  copy_commit();

  for (int kb = 0; kb < n_k_blocks; ++kb) {
    const int t_k0 = kb * BC;
    const int cur = kb & 1;
    const bool more = (kb + 1) < n_k_blocks;

    // Issue the next tile before waiting on this one: that overlap is the whole
    // point. The buffer being written was last read in iteration kb-1, and the
    // barrier at the bottom of that iteration is what makes it safe to reuse.
    if (more) {
      const int t_k1 = (kb + 1) * BC;
      stage_tile<scalar_t, HEAD_DIM, BC, PAD, MmaTiles::kThreads>(
          Ks[cur ^ 1], K, b, t_k1, imin(BC, T - t_k1), T, hkv, Hkv, tid);
      stage_tile<scalar_t, HEAD_DIM, BC, PAD, MmaTiles::kThreads>(
          Vs[cur ^ 1], V, b, t_k1, imin(BC, T - t_k1), T, hkv, Hkv, tid);
      copy_commit();
      copy_wait<1>();     // tile kb has landed; tile kb+1 still in flight
    } else {
      copy_wait<0>();
    }
    // cp.async completion is per-thread; this is what makes the whole tile
    // visible to the whole block.
    __syncthreads();

    // ---- S = Q K^T ----
    float s_acc[NS][4];
#pragma unroll
    for (int n = 0; n < NS; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) s_acc[n][e] = 0.0f;
    }
#pragma unroll
    for (int n = 0; n < NS; ++n) {
#pragma unroll
      for (int kc = 0; kc < KCHUNKS; ++kc) {
        uint32_t kfrag[2];
        load_b<scalar_t>(Ks[cur], HEAD_DIM + PAD, n, gid, kc * 16, tig, kfrag);
        mma_m16n8k16<scalar_t>(q_frag[kc], kfrag, s_acc[n]);
      }
    }

    // ---- mask, then the online softmax, all in registers ----
    float m_sub_a = -INFINITY, m_sub_b = -INFINITY;
#pragma unroll
    for (int n = 0; n < NS; ++n) {
      const int col0 = t_k0 + n * 8 + tig * 2;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = (e < 2) ? row_a : row_b;
        const int col = col0 + (e & 1);
        // Causal, plus the tail of the last tile.
        const bool keep = (col < T) && (col <= row);
        s_acc[n][e] = keep ? s_acc[n][e] * qk_scale : -INFINITY;
      }
      m_sub_a = fmaxf(m_sub_a, fmaxf(s_acc[n][0], s_acc[n][1]));
      m_sub_b = fmaxf(m_sub_b, fmaxf(s_acc[n][2], s_acc[n][3]));
    }
    // The 4 lanes of a group hold different columns of the same two rows.
#pragma unroll
    for (int mask = 1; mask <= 2; mask <<= 1) {
      m_sub_a = fmaxf(m_sub_a, __shfl_xor_sync(0xffffffffu, m_sub_a, mask));
      m_sub_b = fmaxf(m_sub_b, __shfl_xor_sync(0xffffffffu, m_sub_b, mask));
    }

    const float m_new_a = fmaxf(m_a, m_sub_a);
    const float m_new_b = fmaxf(m_b, m_sub_b);
    // -inf on both sides means this row has seen no key at all, which only
    // happens for a row past T. Take 0 instead of a NaN; nothing is stored for
    // those rows either way, but a NaN in flight is a bad thing to allow.
    const float alpha_a = (m_new_a == -INFINITY) ? 0.0f : exp2f(m_a - m_new_a);
    const float alpha_b = (m_new_b == -INFINITY) ? 0.0f : exp2f(m_b - m_new_b);

    float sum_a = 0.0f, sum_b = 0.0f;
#pragma unroll
    for (int n = 0; n < NS; ++n) {
      s_acc[n][0] = exp2f(s_acc[n][0] - m_new_a);
      s_acc[n][1] = exp2f(s_acc[n][1] - m_new_a);
      s_acc[n][2] = exp2f(s_acc[n][2] - m_new_b);
      s_acc[n][3] = exp2f(s_acc[n][3] - m_new_b);
      sum_a += s_acc[n][0] + s_acc[n][1];
      sum_b += s_acc[n][2] + s_acc[n][3];
    }
#pragma unroll
    for (int mask = 1; mask <= 2; mask <<= 1) {
      sum_a += __shfl_xor_sync(0xffffffffu, sum_a, mask);
      sum_b += __shfl_xor_sync(0xffffffffu, sum_b, mask);
    }
    l_a = l_a * alpha_a + sum_a;
    l_b = l_b * alpha_b + sum_b;
    m_a = m_new_a;
    m_b = m_new_b;

    // Rescale the output accumulator: rows a and b live in different halves of
    // every n-tile, so they take different alphas.
#pragma unroll
    for (int n = 0; n < NO; ++n) {
      acc_o[n][0] *= alpha_a;
      acc_o[n][1] *= alpha_a;
      acc_o[n][2] *= alpha_b;
      acc_o[n][3] *= alpha_b;
    }

    // ---- O += P V ----
    // Two adjacent score n-tiles are exactly one 16x16 A fragment (see the
    // layout note): no shuffle, no round trip through shared memory.
#pragma unroll
    for (int pc = 0; pc < PCHUNKS; ++pc) {
      uint32_t p_frag[4];
      pack_a_from_acc<scalar_t>(s_acc[pc * 2], s_acc[pc * 2 + 1], p_frag);
#pragma unroll
      for (int n = 0; n < NO; ++n) {
        uint32_t vfrag[2];
        load_b_strided<scalar_t>(Vs[cur], HEAD_DIM + PAD, n, gid, pc * 16, tig,
                                 vfrag);
        mma_m16n8k16<scalar_t>(p_frag, vfrag, acc_o[n]);
      }
    }
    __syncthreads();
  }

  // ---- epilogue ----
  const float inv_a = 1.0f / l_a;
  const float inv_b = 1.0f / l_b;
  scalar_t* op_a = O + row_off(b, ok_a ? row_a : 0, T, h, H, HEAD_DIM);
  scalar_t* op_b = O + row_off(b, ok_b ? row_b : 0, T, h, H, HEAD_DIM);
#pragma unroll
  for (int n = 0; n < NO; ++n) {
    const int d0 = n * 8 + tig * 2;
    if (ok_a) {
      op_a[d0] = static_cast<scalar_t>(acc_o[n][0] * inv_a);
      op_a[d0 + 1] = static_cast<scalar_t>(acc_o[n][1] * inv_a);
    }
    if (ok_b) {
      op_b[d0] = static_cast<scalar_t>(acc_o[n][2] * inv_b);
      op_b[d0 + 1] = static_cast<scalar_t>(acc_o[n][3] * inv_b);
    }
  }
  // All 4 lanes of a group hold the same reduced m/l, so one of them writes.
  if (tig == 0) {
    if (ok_a) L[lse_off(b, h, H, row_a, T)] = m_a * kLn2 + logf(l_a);
    if (ok_b) L[lse_off(b, h, H, row_b, T)] = m_b * kLn2 + logf(l_b);
  }
}

/// dQ on tensor cores. Block per (q_block, b*h); a warp owns 16 query rows.
///
/// Three products per key tile: S = Q K^T and dP = dO V^T (both exactly the
/// forward's shape, so K and V stage naturally), then dQ += dS K, which wants K
/// the other way round -- hence K is staged twice. dS never leaves registers
/// between the second and third: it is an accumulator, and an accumulator is
/// already an A operand.
template <typename scalar_t, int HEAD_DIM>
__global__ __launch_bounds__(MmaTiles::kThreads)
void flash_bwd_dq_mma_kernel(const scalar_t* __restrict__ Q,
                             const scalar_t* __restrict__ K,
                             const scalar_t* __restrict__ V,
                             const scalar_t* __restrict__ dO,
                             const float* __restrict__ L,
                             const float* __restrict__ Delta,
                             scalar_t* __restrict__ dQ,
                             const int T, const int H, const int Hkv,
                             const float scale) {
  constexpr int BR = MmaTiles::BR;
  constexpr int BC = MmaTiles::BC;
  constexpr int PAD = MmaTiles::kPad;
  constexpr int KCHUNKS = HEAD_DIM / 16;   // k-chunks of S and dP (k is d)
  constexpr int NS = BC / 8;               // n-tiles of S and dP (n is the key)
  constexpr int NO = HEAD_DIM / 8;         // n-tiles of dQ (n is d)
  constexpr int PCHUNKS = BC / 16;         // k-chunks of dS K (k is the key)
  static_assert(HEAD_DIM % 16 == 0, "head_dim must be a whole number of k-chunks");

  // K and V, each staged once and double buffered. dQ += dS K reads K down a
  // column (load_b_strided) instead of from a second, transposed copy.
  alignas(16) __shared__ scalar_t Ks[2][BC * (HEAD_DIM + PAD)];   // [key][d]
  alignas(16) __shared__ scalar_t Vs[2][BC * (HEAD_DIM + PAD)];   // [key][d]

  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  const int bh = blockIdx.y;
  const int h = bh % H;
  const int b = bh / H;
  const int group = H / Hkv;
  const int hkv = h / group;

  const int t_q0 = blockIdx.x * BR;
  if (t_q0 >= T) return;
  const int q_base = t_q0 + wid * 16;
  const int row_a = q_base + gid;
  const int row_b = q_base + gid + 8;
  const bool ok_a = row_a < T;
  const bool ok_b = row_b < T;

  const float qk_scale = scale * kLog2e;

  const long long qa_off = row_off(b, ok_a ? row_a : 0, T, h, H, HEAD_DIM);
  const long long qb_off = row_off(b, ok_b ? row_b : 0, T, h, H, HEAD_DIM);
  uint32_t q_frag[KCHUNKS][4];
  uint32_t do_frag[KCHUNKS][4];
  load_a_rows<scalar_t, KCHUNKS>(Q + qa_off, ok_a, Q + qb_off, ok_b, tig, q_frag);
  load_a_rows<scalar_t, KCHUNKS>(dO + qa_off, ok_a, dO + qb_off, ok_b, tig, do_frag);

  // Taped LSE and Delta for the lane's two rows, in log2 space.
  const float Li2_a = ok_a ? L[lse_off(b, h, H, row_a, T)] * kLog2e : 0.0f;
  const float Li2_b = ok_b ? L[lse_off(b, h, H, row_b, T)] * kLog2e : 0.0f;
  const float Di_a = ok_a ? Delta[lse_off(b, h, H, row_a, T)] : 0.0f;
  const float Di_b = ok_b ? Delta[lse_off(b, h, H, row_b, T)] : 0.0f;

  float acc_dq[NO][4];
#pragma unroll
  for (int n = 0; n < NO; ++n) {
#pragma unroll
    for (int e = 0; e < 4; ++e) acc_dq[n][e] = 0.0f;
  }

  const int t_q_max = imin(t_q0 + BR, T) - 1;
  const int n_k_blocks = (t_q_max / BC) + 1;

  stage_tile<scalar_t, HEAD_DIM, BC, PAD, MmaTiles::kThreads>(
      Ks[0], K, b, 0, imin(BC, T), T, hkv, Hkv, tid);
  stage_tile<scalar_t, HEAD_DIM, BC, PAD, MmaTiles::kThreads>(
      Vs[0], V, b, 0, imin(BC, T), T, hkv, Hkv, tid);
  copy_commit();

  for (int kb = 0; kb < n_k_blocks; ++kb) {
    const int t_k0 = kb * BC;
    const int cur = kb & 1;
    const bool more = (kb + 1) < n_k_blocks;

    if (more) {
      const int t_k1 = (kb + 1) * BC;
      stage_tile<scalar_t, HEAD_DIM, BC, PAD, MmaTiles::kThreads>(
          Ks[cur ^ 1], K, b, t_k1, imin(BC, T - t_k1), T, hkv, Hkv, tid);
      stage_tile<scalar_t, HEAD_DIM, BC, PAD, MmaTiles::kThreads>(
          Vs[cur ^ 1], V, b, t_k1, imin(BC, T - t_k1), T, hkv, Hkv, tid);
      copy_commit();
      copy_wait<1>();
    } else {
      copy_wait<0>();
    }
    __syncthreads();

    // ---- S = Q K^T and dP = dO V^T ----
    float s_acc[NS][4];
    float dp_acc[NS][4];
#pragma unroll
    for (int n = 0; n < NS; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        s_acc[n][e] = 0.0f;
        dp_acc[n][e] = 0.0f;
      }
    }
#pragma unroll
    for (int n = 0; n < NS; ++n) {
#pragma unroll
      for (int kc = 0; kc < KCHUNKS; ++kc) {
        uint32_t bf[2];
        load_b<scalar_t>(Ks[cur], HEAD_DIM + PAD, n, gid, kc * 16, tig, bf);
        mma_m16n8k16<scalar_t>(q_frag[kc], bf, s_acc[n]);
        load_b<scalar_t>(Vs[cur], HEAD_DIM + PAD, n, gid, kc * 16, tig, bf);
        mma_m16n8k16<scalar_t>(do_frag[kc], bf, dp_acc[n]);
      }
    }

    // ---- dS = P * (dP - Delta) * scale, elementwise, in registers ----
    // Masked entries become exactly 0, so they contribute nothing to the third
    // product and no mask has to survive into it.
#pragma unroll
    for (int n = 0; n < NS; ++n) {
      const int col0 = t_k0 + n * 8 + tig * 2;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const bool lo = e < 2;
        const int row = lo ? row_a : row_b;
        const int col = col0 + (e & 1);
        const bool keep = (lo ? ok_a : ok_b) && col < T && col <= row;
        const float p = keep ? exp2f(s_acc[n][e] * qk_scale - (lo ? Li2_a : Li2_b))
                             : 0.0f;
        s_acc[n][e] = p * (dp_acc[n][e] - (lo ? Di_a : Di_b)) * scale;
      }
    }

    // ---- dQ += dS K ----
#pragma unroll
    for (int pc = 0; pc < PCHUNKS; ++pc) {
      uint32_t ds_frag[4];
      pack_a_from_acc<scalar_t>(s_acc[pc * 2], s_acc[pc * 2 + 1], ds_frag);
#pragma unroll
      for (int n = 0; n < NO; ++n) {
        uint32_t bf[2];
        load_b_strided<scalar_t>(Ks[cur], HEAD_DIM + PAD, n, gid, pc * 16, tig,
                                 bf);
        mma_m16n8k16<scalar_t>(ds_frag, bf, acc_dq[n]);
      }
    }
    __syncthreads();
  }

  scalar_t* dq_a = dQ + qa_off;
  scalar_t* dq_b = dQ + qb_off;
#pragma unroll
  for (int n = 0; n < NO; ++n) {
    const int d0 = n * 8 + tig * 2;
    if (ok_a) {
      dq_a[d0] = static_cast<scalar_t>(acc_dq[n][0]);
      dq_a[d0 + 1] = static_cast<scalar_t>(acc_dq[n][1]);
    }
    if (ok_b) {
      dq_b[d0] = static_cast<scalar_t>(acc_dq[n][2]);
      dq_b[d0 + 1] = static_cast<scalar_t>(acc_dq[n][3]);
    }
  }
}

/// dK/dV on tensor cores. Block per (k_block, b*hkv); a warp owns 16 KEY rows.
///
/// Making keys the m dimension is what removes the register transpose that this
/// pass would otherwise need. Computing S^T = K Q^T (not S) puts keys on rows
/// and queries on columns, so dS^T lands in the accumulator already oriented for
/// dK += dS^T Q and dV += P^T dO -- the transposes that FA's backward is known
/// for become a choice of which matrix the warp iterates, not a data movement.
/// The GQA group is summed inside, so dK/dV land in the [B,T,Hkv,D] layout with
/// no reduction pass afterwards, and there are no atomics: this stays as
/// reproducible as the FMA backward.
template <typename scalar_t, int HEAD_DIM>
__global__ __launch_bounds__(MmaTiles::kThreads)
void flash_bwd_dkv_mma_kernel(const scalar_t* __restrict__ Q,
                              const scalar_t* __restrict__ K,
                              const scalar_t* __restrict__ V,
                              const scalar_t* __restrict__ dO,
                              const float* __restrict__ L,
                              const float* __restrict__ Delta,
                              scalar_t* __restrict__ dK,
                              scalar_t* __restrict__ dV,
                              const int T, const int H, const int Hkv,
                              const float scale) {
  constexpr int BK = MmaTiles::BR;         // key rows per block
  constexpr int BQ = MmaTiles::BQ;         // query tile
  constexpr int PAD = MmaTiles::kPad;
  constexpr int KCHUNKS = HEAD_DIM / 16;   // k-chunks of S^T and dP^T (k is d)
  constexpr int NQ = BQ / 8;               // n-tiles of S^T and dP^T (n is query)
  constexpr int NO = HEAD_DIM / 8;         // n-tiles of dK and dV (n is d)
  constexpr int PCHUNKS = BQ / 16;         // k-chunks of dS^T Q (k is the query)
  static_assert(HEAD_DIM % 16 == 0, "head_dim must be a whole number of k-chunks");

  // Q and dO, each staged once and double buffered; dK and dV read them down a
  // column (load_b_strided) rather than from transposed copies.
  alignas(16) __shared__ scalar_t Qs[2][BQ * (HEAD_DIM + PAD)];    // [query][d]
  alignas(16) __shared__ scalar_t dOs[2][BQ * (HEAD_DIM + PAD)];   // [query][d]
  __shared__ float Ls[BQ];                          // per-query tape, log2 space
  __shared__ float Ds[BQ];

  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  const int bhkv = blockIdx.y;
  const int hkv = bhkv % Hkv;
  const int b = bhkv / Hkv;
  const int group = H / Hkv;

  const int t_k0 = blockIdx.x * BK;
  if (t_k0 >= T) return;
  const int k_base = t_k0 + wid * 16;
  const int key_a = k_base + gid;
  const int key_b = k_base + gid + 8;
  const bool ok_a = key_a < T;
  const bool ok_b = key_b < T;

  const float qk_scale = scale * kLog2e;

  const long long ka_off = row_off(b, ok_a ? key_a : 0, T, hkv, Hkv, HEAD_DIM);
  const long long kb_off = row_off(b, ok_b ? key_b : 0, T, hkv, Hkv, HEAD_DIM);
  uint32_t k_frag[KCHUNKS][4];
  uint32_t v_frag[KCHUNKS][4];
  load_a_rows<scalar_t, KCHUNKS>(K + ka_off, ok_a, K + kb_off, ok_b, tig, k_frag);
  load_a_rows<scalar_t, KCHUNKS>(V + ka_off, ok_a, V + kb_off, ok_b, tig, v_frag);

  float acc_dk[NO][4];
  float acc_dv[NO][4];
#pragma unroll
  for (int n = 0; n < NO; ++n) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      acc_dk[n][e] = 0.0f;
      acc_dv[n][e] = 0.0f;
    }
  }

  // Causal: a key at t_k is only seen by queries at t_q >= t_k, so start at the
  // query tile holding this block's first key.
  const int qb_start = t_k0 / BQ;
  const int qb_end = (T + BQ - 1) / BQ;

  // The pipeline is indexed by a flat step so the prefetch rolls over into the
  // next GQA head instead of draining at the group boundary.
  int step = 0;
  stage_tile<scalar_t, HEAD_DIM, BQ, PAD, MmaTiles::kThreads>(
      Qs[0], Q, b, qb_start * BQ, imin(BQ, T - qb_start * BQ), T,
      hkv * group, H, tid);
  stage_tile<scalar_t, HEAD_DIM, BQ, PAD, MmaTiles::kThreads>(
      dOs[0], dO, b, qb_start * BQ, imin(BQ, T - qb_start * BQ), T,
      hkv * group, H, tid);
  copy_commit();

  for (int g = 0; g < group; ++g) {
    const int h = hkv * group + g;
    for (int qb = qb_start; qb < qb_end; ++qb) {
      const int t_q0 = qb * BQ;
      const int n_q = imin(BQ, T - t_q0);
      const int cur = step & 1;

      // Next (g, qb), rolling into the following head when this one runs out.
      int qb1 = qb + 1;
      int g1 = g;
      if (qb1 >= qb_end) {
        qb1 = qb_start;
        g1 = g + 1;
      }
      if (g1 < group) {
        const int t_q1 = qb1 * BQ;
        const int n_q1 = imin(BQ, T - t_q1);
        stage_tile<scalar_t, HEAD_DIM, BQ, PAD, MmaTiles::kThreads>(
            Qs[cur ^ 1], Q, b, t_q1, n_q1, T, hkv * group + g1, H, tid);
        stage_tile<scalar_t, HEAD_DIM, BQ, PAD, MmaTiles::kThreads>(
            dOs[cur ^ 1], dO, b, t_q1, n_q1, T, hkv * group + g1, H, tid);
        copy_commit();
        copy_wait<1>();
      } else {
        copy_wait<0>();
      }

      // L and Delta are tiny, and their rows are not 16-byte aligned (the tape
      // is [B,H,T], so a head's slice starts wherever T lands), so they stay a
      // synchronous copy of the *current* tile. Single-buffered: the barrier at
      // the bottom of the previous step is what makes overwriting them safe.
      for (int i = tid; i < BQ; i += MmaTiles::kThreads) {
        const long long lo = lse_off(b, h, H, t_q0 + imin(i, n_q - 1), T);
        Ls[i] = L[lo] * kLog2e;
        Ds[i] = Delta[lo];
      }
      __syncthreads();

      // ---- S^T = K Q^T and dP^T = V dO^T ----
      float st_acc[NQ][4];
      float dpt_acc[NQ][4];
#pragma unroll
      for (int n = 0; n < NQ; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          st_acc[n][e] = 0.0f;
          dpt_acc[n][e] = 0.0f;
        }
      }
#pragma unroll
      for (int n = 0; n < NQ; ++n) {
#pragma unroll
        for (int kc = 0; kc < KCHUNKS; ++kc) {
          uint32_t bf[2];
          load_b<scalar_t>(Qs[cur], HEAD_DIM + PAD, n, gid, kc * 16, tig, bf);
          mma_m16n8k16<scalar_t>(k_frag[kc], bf, st_acc[n]);
          load_b<scalar_t>(dOs[cur], HEAD_DIM + PAD, n, gid, kc * 16, tig, bf);
          mma_m16n8k16<scalar_t>(v_frag[kc], bf, dpt_acc[n]);
        }
      }

      // ---- P^T and dS^T, elementwise ----
      // Rows are keys and columns are queries here, so the tape is indexed by
      // the COLUMN -- the mirror image of the dQ pass, and the easiest thing in
      // this file to get backwards.
      float p_t[NQ][4];
#pragma unroll
      for (int n = 0; n < NQ; ++n) {
        const int qcol = n * 8 + tig * 2;          // index within the tile
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const bool lo = e < 2;
          const int row = lo ? key_a : key_b;      // the key
          const int cl = qcol + (e & 1);
          const int col = t_q0 + cl;               // the query
          const bool keep = (lo ? ok_a : ok_b) && col < T && col >= row;
          const float p = keep ? exp2f(st_acc[n][e] * qk_scale - Ls[cl]) : 0.0f;
          p_t[n][e] = p;
          st_acc[n][e] = p * (dpt_acc[n][e] - Ds[cl]) * scale;
        }
      }

      // ---- dK += dS^T Q and dV += P^T dO ----
#pragma unroll
      for (int pc = 0; pc < PCHUNKS; ++pc) {
        uint32_t ds_frag[4];
        uint32_t p_frag[4];
        pack_a_from_acc<scalar_t>(st_acc[pc * 2], st_acc[pc * 2 + 1], ds_frag);
        pack_a_from_acc<scalar_t>(p_t[pc * 2], p_t[pc * 2 + 1], p_frag);
#pragma unroll
        for (int n = 0; n < NO; ++n) {
          uint32_t bf[2];
          load_b_strided<scalar_t>(Qs[cur], HEAD_DIM + PAD, n, gid, pc * 16,
                                   tig, bf);
          mma_m16n8k16<scalar_t>(ds_frag, bf, acc_dk[n]);
          load_b_strided<scalar_t>(dOs[cur], HEAD_DIM + PAD, n, gid, pc * 16,
                                   tig, bf);
          mma_m16n8k16<scalar_t>(p_frag, bf, acc_dv[n]);
        }
      }
      __syncthreads();
      ++step;
    }
  }

  scalar_t* dk_a = dK + ka_off;
  scalar_t* dk_b = dK + kb_off;
  scalar_t* dv_a = dV + ka_off;
  scalar_t* dv_b = dV + kb_off;
#pragma unroll
  for (int n = 0; n < NO; ++n) {
    const int d0 = n * 8 + tig * 2;
    if (ok_a) {
      dk_a[d0] = static_cast<scalar_t>(acc_dk[n][0]);
      dk_a[d0 + 1] = static_cast<scalar_t>(acc_dk[n][1]);
      dv_a[d0] = static_cast<scalar_t>(acc_dv[n][0]);
      dv_a[d0 + 1] = static_cast<scalar_t>(acc_dv[n][1]);
    }
    if (ok_b) {
      dk_b[d0] = static_cast<scalar_t>(acc_dk[n][2]);
      dk_b[d0 + 1] = static_cast<scalar_t>(acc_dk[n][3]);
      dv_b[d0] = static_cast<scalar_t>(acc_dv[n][2]);
      dv_b[d0 + 1] = static_cast<scalar_t>(acc_dv[n][3]);
    }
  }
}

/// One mma tile on known matrices, through the same helpers the kernel uses.
/// A is [16,16] row-major, B is [16,8] row-major (k by n), D is [16,8]. If the
/// fragment layout documented above is wrong, this disagrees with an ordinary
/// matmul and the host refuses the tensor-core path.
template <typename scalar_t>
__global__ void mma_probe_kernel(const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ B,
                                 float* __restrict__ Dm) {
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  const uint32_t a[4] = {
      ld2(A + gid * 16 + tig * 2),
      ld2(A + (gid + 8) * 16 + tig * 2),
      ld2(A + gid * 16 + tig * 2 + 8),
      ld2(A + (gid + 8) * 16 + tig * 2 + 8),
  };
  // B is [k][n] row-major, so a lane's two k-adjacent elements are 8 apart.
  const uint32_t bf[2] = {
      pack2<scalar_t>(static_cast<float>(B[(tig * 2) * 8 + gid]),
                      static_cast<float>(B[(tig * 2 + 1) * 8 + gid])),
      pack2<scalar_t>(static_cast<float>(B[(tig * 2 + 8) * 8 + gid]),
                      static_cast<float>(B[(tig * 2 + 9) * 8 + gid])),
  };
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  mma_m16n8k16<scalar_t>(a, bf, d);

  Dm[gid * 8 + tig * 2] = d[0];
  Dm[gid * 8 + tig * 2 + 1] = d[1];
  Dm[(gid + 8) * 8 + tig * 2] = d[2];
  Dm[(gid + 8) * 8 + tig * 2 + 1] = d[3];
}

// ---------------------------------------------------------------------------
// backward
// ---------------------------------------------------------------------------

/// Delta[b,h,t] = sum_d dO[b,t,h,d] * O[b,t,h,d]. One warp per row, so the D
/// axis is read by consecutive lanes (coalesced) instead of one strided row per
/// thread.
template <typename scalar_t, int HEAD_DIM>
__global__ void flash_delta_kernel(const scalar_t* __restrict__ O,
                                   const scalar_t* __restrict__ dO,
                                   float* __restrict__ Delta,
                                   const int B, const int T, const int H) {
  constexpr int kWarp = 32;
  const int warps_per_block = blockDim.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const long long row = static_cast<long long>(blockIdx.x) * warps_per_block + warp;
  const long long total = static_cast<long long>(B) * H * T;
  if (row >= total) return;

  const int t = static_cast<int>(row % T);
  const long long tmp = row / T;
  const int h = static_cast<int>(tmp % H);
  const int b = static_cast<int>(tmp / H);

  const long long off = row_off(b, t, T, h, H, HEAD_DIM);
  float acc = 0.0f;
  for (int d = lane; d < HEAD_DIM; d += kWarp) {
    acc += static_cast<float>(dO[off + d]) * static_cast<float>(O[off + d]);
  }
#pragma unroll
  for (int shift = kWarp / 2; shift > 0; shift >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, shift);
  }
  if (lane == 0) Delta[lse_off(b, h, H, t, T)] = acc;
}

/// dQ. One block per (q_block, b*h); one thread per query row; K/V staged.
template <typename scalar_t, int HEAD_DIM>
__global__ __launch_bounds__(Tiles<HEAD_DIM * static_cast<int>(sizeof(scalar_t))>::BR)
void flash_bwd_dq_kernel(const scalar_t* __restrict__ Q,
                         const scalar_t* __restrict__ K,
                         const scalar_t* __restrict__ V,
                         const scalar_t* __restrict__ dO,
                         const float* __restrict__ L,
                         const float* __restrict__ Delta,
                         scalar_t* __restrict__ dQ,
                         const int T, const int H, const int Hkv,
                         const float scale) {
  using Tile = TilesFor<scalar_t, HEAD_DIM>;
  constexpr int BR = Tile::BR;
  constexpr int BC = Tile::BC;

  __shared__ scalar_t Ks[BC * HEAD_DIM];
  __shared__ scalar_t Vs[BC * HEAD_DIM];

  const int tid = threadIdx.x;
  const int bh = blockIdx.y;
  const int h = bh % H;
  const int b = bh / H;
  const int group = H / Hkv;
  const int hkv = h / group;

  const int t_q0 = blockIdx.x * BR;
  if (t_q0 >= T) return;
  const int t_q = t_q0 + tid;
  const bool row_valid = t_q < T;

  const float qk_scale = scale * kLog2e;

  // Hoisted once (Metal audit 7's finding #1); `__restrict__` is what makes the
  // hoist legal here without hand-carrying them.
  float q_reg[HEAD_DIM];
  float do_reg[HEAD_DIM];
  float dq[HEAD_DIM];
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    q_reg[d] = 0.0f;
    do_reg[d] = 0.0f;
    dq[d] = 0.0f;
  }
  float Li2 = 0.0f;   // taped LSE, converted to log2 space once
  float Di = 0.0f;
  if (row_valid) {
    const long long off = row_off(b, t_q, T, h, H, HEAD_DIM);
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      q_reg[d] = static_cast<float>(Q[off + d]);
      do_reg[d] = static_cast<float>(dO[off + d]);
    }
    const long long lo = lse_off(b, h, H, t_q, T);
    Li2 = L[lo] * kLog2e;
    Di = Delta[lo];
  }

  const int t_q_max = imin(t_q0 + BR, T) - 1;
  const int n_k_blocks = (t_q_max / BC) + 1;

  for (int kb = 0; kb < n_k_blocks; ++kb) {
    const int t_k0 = kb * BC;
    const int n_k = imin(BC, T - t_k0);

    for (int i = tid; i < n_k * HEAD_DIM; i += BR) {
      const int tk = i / HEAD_DIM;
      const int d = i - tk * HEAD_DIM;
      const long long off = row_off(b, t_k0 + tk, T, hkv, Hkv, HEAD_DIM) + d;
      Ks[i] = K[off];
      Vs[i] = V[off];
    }
    __syncthreads();

    const int k_hi = imin(n_k, t_q - t_k0 + 1);
    if (row_valid) {
      for (int tk = 0; tk < k_hi; ++tk) {
        const int base = tk * HEAD_DIM;
        float score = 0.0f;
        float dp = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
          const float kd = static_cast<float>(Ks[base + d]);
          score = fmaf(q_reg[d], kd, score);
          dp = fmaf(do_reg[d], static_cast<float>(Vs[base + d]), dp);
        }
        const float p = exp2f(score * qk_scale - Li2);
        const float dss = p * (dp - Di) * scale;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
          dq[d] = fmaf(dss, static_cast<float>(Ks[base + d]), dq[d]);
        }
      }
    }
    __syncthreads();
  }

  if (row_valid) {
    scalar_t* dqp = dQ + row_off(b, t_q, T, h, H, HEAD_DIM);
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) dqp[d] = static_cast<scalar_t>(dq[d]);
  }
}

/// dK/dV. One block per (k_block, b*hkv); one thread per key row; Q/dO/L/Delta
/// staged per query tile. The GQA group is summed inside, so dK/dV land in the
/// [B,T,Hkv,D] KV layout directly -- no reduction pass afterwards.
template <typename scalar_t, int HEAD_DIM>
__global__ __launch_bounds__(Tiles<HEAD_DIM * static_cast<int>(sizeof(scalar_t))>::BC)
void flash_bwd_dkv_kernel(const scalar_t* __restrict__ Q,
                          const scalar_t* __restrict__ K,
                          const scalar_t* __restrict__ V,
                          const scalar_t* __restrict__ dO,
                          const float* __restrict__ L,
                          const float* __restrict__ Delta,
                          scalar_t* __restrict__ dK,
                          scalar_t* __restrict__ dV,
                          const int T, const int H, const int Hkv,
                          const float scale) {
  using Tile = TilesFor<scalar_t, HEAD_DIM>;
  constexpr int BC = Tile::BC;
  constexpr int BQ = Tile::BQ;

  __shared__ scalar_t Qs[BQ * HEAD_DIM];
  __shared__ scalar_t dOs[BQ * HEAD_DIM];
  __shared__ float Ls[BQ];
  __shared__ float Ds[BQ];

  const int tid = threadIdx.x;
  const int bhkv = blockIdx.y;
  const int hkv = bhkv % Hkv;
  const int b = bhkv / Hkv;
  const int group = H / Hkv;

  const int t_k0 = blockIdx.x * BC;
  if (t_k0 >= T) return;
  const int t_k = t_k0 + tid;
  const bool key_valid = t_k < T;

  const float qk_scale = scale * kLog2e;

  float k_reg[HEAD_DIM];
  float v_reg[HEAD_DIM];
  float dk[HEAD_DIM];
  float dv[HEAD_DIM];
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    k_reg[d] = 0.0f;
    v_reg[d] = 0.0f;
    dk[d] = 0.0f;
    dv[d] = 0.0f;
  }
  if (key_valid) {
    const long long off = row_off(b, t_k, T, hkv, Hkv, HEAD_DIM);
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      k_reg[d] = static_cast<float>(K[off + d]);
      v_reg[d] = static_cast<float>(V[off + d]);
    }
  }

  // Causal: a key at t_k is only seen by queries at t_q >= t_k, so start at the
  // query tile that contains this block's first key.
  const int qb_start = t_k0 / BQ;
  const int qb_end = (T + BQ - 1) / BQ;

  for (int g = 0; g < group; ++g) {
    const int h = hkv * group + g;
    for (int qb = qb_start; qb < qb_end; ++qb) {
      const int t_q0 = qb * BQ;
      const int n_q = imin(BQ, T - t_q0);

      for (int i = tid; i < n_q * HEAD_DIM; i += BC) {
        const int tq = i / HEAD_DIM;
        const int d = i - tq * HEAD_DIM;
        const long long off = row_off(b, t_q0 + tq, T, h, H, HEAD_DIM) + d;
        Qs[i] = Q[off];
        dOs[i] = dO[off];
      }
      for (int i = tid; i < n_q; i += BC) {
        const long long lo = lse_off(b, h, H, t_q0 + i, T);
        Ls[i] = L[lo] * kLog2e;
        Ds[i] = Delta[lo];
      }
      __syncthreads();

      if (key_valid) {
        const int j_lo = imax(0, t_k - t_q0);   // causal: t_q >= t_k
        for (int j = j_lo; j < n_q; ++j) {
          const int base = j * HEAD_DIM;
          float score = 0.0f;
          float dp = 0.0f;
#pragma unroll
          for (int d = 0; d < HEAD_DIM; ++d) {
            score = fmaf(static_cast<float>(Qs[base + d]), k_reg[d], score);
            dp = fmaf(static_cast<float>(dOs[base + d]), v_reg[d], dp);
          }
          const float p = exp2f(score * qk_scale - Ls[j]);
          const float dss = p * (dp - Ds[j]) * scale;
#pragma unroll
          for (int d = 0; d < HEAD_DIM; ++d) {
            dk[d] = fmaf(dss, static_cast<float>(Qs[base + d]), dk[d]);
            dv[d] = fmaf(p, static_cast<float>(dOs[base + d]), dv[d]);
          }
        }
      }
      __syncthreads();
    }
  }

  if (key_valid) {
    const long long off = row_off(b, t_k, T, hkv, Hkv, HEAD_DIM);
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      dK[off + d] = static_cast<scalar_t>(dk[d]);
      dV[off + d] = static_cast<scalar_t>(dv[d]);
    }
  }
}

// ---------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------

struct Dims {
  int B, T, H, Hkv;
  float scale;
};

template <typename scalar_t, int HEAD_DIM>
void launch_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                at::Tensor& o, at::Tensor& l, const Dims& dm,
                cudaStream_t stream) {
  using Tile = TilesFor<scalar_t, HEAD_DIM>;
  const dim3 grid((dm.T + Tile::BR - 1) / Tile::BR, dm.B * dm.H);
  flash_fwd_kernel<scalar_t, HEAD_DIM><<<grid, Tile::BR, 0, stream>>>(
      q.const_data_ptr<scalar_t>(), k.const_data_ptr<scalar_t>(),
      v.const_data_ptr<scalar_t>(), o.data_ptr<scalar_t>(), l.data_ptr<float>(),
      dm.T, dm.H, dm.Hkv, dm.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int HEAD_DIM>
void launch_fwd_mma(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                    at::Tensor& o, at::Tensor& l, const Dims& dm,
                    cudaStream_t stream) {
  const dim3 grid((dm.T + MmaTiles::BR - 1) / MmaTiles::BR, dm.B * dm.H);
  flash_fwd_mma_kernel<scalar_t, HEAD_DIM><<<grid, MmaTiles::kThreads, 0, stream>>>(
      q.const_data_ptr<scalar_t>(), k.const_data_ptr<scalar_t>(),
      v.const_data_ptr<scalar_t>(), o.data_ptr<scalar_t>(), l.data_ptr<float>(),
      dm.T, dm.H, dm.Hkv, dm.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int HEAD_DIM>
void launch_bwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                const at::Tensor& o, const at::Tensor& dout,
                const at::Tensor& l, at::Tensor& delta, at::Tensor& dq,
                at::Tensor& dk, at::Tensor& dv, const Dims& dm,
                cudaStream_t stream) {
  using Tile = TilesFor<scalar_t, HEAD_DIM>;

  constexpr int kDeltaThreads = 128;
  constexpr int kWarpsPerBlock = kDeltaThreads / 32;
  const long long rows = static_cast<long long>(dm.B) * dm.H * dm.T;
  const int delta_blocks =
      static_cast<int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  flash_delta_kernel<scalar_t, HEAD_DIM><<<delta_blocks, kDeltaThreads, 0, stream>>>(
      o.const_data_ptr<scalar_t>(), dout.const_data_ptr<scalar_t>(),
      delta.data_ptr<float>(), dm.B, dm.T, dm.H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 dq_grid((dm.T + Tile::BR - 1) / Tile::BR, dm.B * dm.H);
  flash_bwd_dq_kernel<scalar_t, HEAD_DIM><<<dq_grid, Tile::BR, 0, stream>>>(
      q.const_data_ptr<scalar_t>(), k.const_data_ptr<scalar_t>(),
      v.const_data_ptr<scalar_t>(), dout.const_data_ptr<scalar_t>(),
      l.const_data_ptr<float>(), delta.const_data_ptr<float>(),
      dq.data_ptr<scalar_t>(), dm.T, dm.H, dm.Hkv, dm.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 dkv_grid((dm.T + Tile::BC - 1) / Tile::BC, dm.B * dm.Hkv);
  flash_bwd_dkv_kernel<scalar_t, HEAD_DIM><<<dkv_grid, Tile::BC, 0, stream>>>(
      q.const_data_ptr<scalar_t>(), k.const_data_ptr<scalar_t>(),
      v.const_data_ptr<scalar_t>(), dout.const_data_ptr<scalar_t>(),
      l.const_data_ptr<float>(), delta.const_data_ptr<float>(),
      dk.data_ptr<scalar_t>(), dv.data_ptr<scalar_t>(), dm.T, dm.H, dm.Hkv,
      dm.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int HEAD_DIM>
void launch_bwd_mma(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                    const at::Tensor& o, const at::Tensor& dout,
                    const at::Tensor& l, at::Tensor& delta, at::Tensor& dq,
                    at::Tensor& dk, at::Tensor& dv, const Dims& dm,
                    cudaStream_t stream) {
  // Delta is the same reduction either way; only the two big passes change.
  constexpr int kDeltaThreads = 128;
  constexpr int kWarpsPerBlock = kDeltaThreads / 32;
  const long long rows = static_cast<long long>(dm.B) * dm.H * dm.T;
  const int delta_blocks =
      static_cast<int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  flash_delta_kernel<scalar_t, HEAD_DIM><<<delta_blocks, kDeltaThreads, 0, stream>>>(
      o.const_data_ptr<scalar_t>(), dout.const_data_ptr<scalar_t>(),
      delta.data_ptr<float>(), dm.B, dm.T, dm.H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 dq_grid((dm.T + MmaTiles::BR - 1) / MmaTiles::BR, dm.B * dm.H);
  flash_bwd_dq_mma_kernel<scalar_t, HEAD_DIM><<<dq_grid, MmaTiles::kThreads, 0, stream>>>(
      q.const_data_ptr<scalar_t>(), k.const_data_ptr<scalar_t>(),
      v.const_data_ptr<scalar_t>(), dout.const_data_ptr<scalar_t>(),
      l.const_data_ptr<float>(), delta.const_data_ptr<float>(),
      dq.data_ptr<scalar_t>(), dm.T, dm.H, dm.Hkv, dm.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 dkv_grid((dm.T + MmaTiles::BR - 1) / MmaTiles::BR, dm.B * dm.Hkv);
  flash_bwd_dkv_mma_kernel<scalar_t, HEAD_DIM><<<dkv_grid, MmaTiles::kThreads, 0, stream>>>(
      q.const_data_ptr<scalar_t>(), k.const_data_ptr<scalar_t>(),
      v.const_data_ptr<scalar_t>(), dout.const_data_ptr<scalar_t>(),
      l.const_data_ptr<float>(), delta.const_data_ptr<float>(),
      dk.data_ptr<scalar_t>(), dv.data_ptr<scalar_t>(), dm.T, dm.H, dm.Hkv,
      dm.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Head dim is a template parameter (note 2), so it has to be switched on here.
// An unsupported one is a hard error, not a slow fallback: a kernel that
// silently did something else would be worse than no kernel.
#define NANOLAB_ON_HEAD_DIM(head_dim, FN, ...)                                \
  switch (head_dim) {                                                         \
    case 32:                                                                  \
      FN<scalar_t, 32>(__VA_ARGS__);                                          \
      break;                                                                  \
    case 64:                                                                  \
      FN<scalar_t, 64>(__VA_ARGS__);                                          \
      break;                                                                  \
    case 128:                                                                 \
      FN<scalar_t, 128>(__VA_ARGS__);                                         \
      break;                                                                  \
    default:                                                                  \
      TORCH_CHECK(false, "flash_attn_cuda: head_dim ", head_dim,              \
                  " is not compiled in (have 32, 64, 128)");                  \
  }

// The tensor-core backward stops at head_dim 64. At 128 a dK/dV warp would hold
// 64 registers of K/V fragments, 48 of scores, and 128 of dK/dV accumulators --
// past the 255-register file before temporaries, so it would spill to local
// memory and lose more than the tensor cores win. head_dim 128 uses the FMA
// backward, which is why this is a separate switch rather than a wider one.
#define NANOLAB_ON_HEAD_DIM_MMA_BWD(head_dim, FN, ...)                        \
  switch (head_dim) {                                                         \
    case 32:                                                                  \
      FN<scalar_t, 32>(__VA_ARGS__);                                          \
      break;                                                                  \
    case 64:                                                                  \
      FN<scalar_t, 64>(__VA_ARGS__);                                          \
      break;                                                                  \
    default:                                                                  \
      TORCH_CHECK(false,                                                      \
                  "flash_attn_cuda: the tensor-core backward supports "       \
                  "head_dim 32 and 64, not ", head_dim,                       \
                  " (register pressure); use the FMA backward there");        \
  }

void check_layout(const at::Tensor& q, const at::Tensor& k,
                  const at::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "flash_attn_cuda: all inputs must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "flash_attn_cuda: expected [B,T,H,D] / [B,T,Hkv,D], got ",
              q.dim(), "d");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                  q.scalar_type() == v.scalar_type(),
              "flash_attn_cuda: q/k/v dtypes must match");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "flash_attn_cuda: q/k/v must be contiguous");
  TORCH_CHECK(k.size(0) == q.size(0) && v.size(0) == q.size(0) &&
                  k.size(1) == q.size(1) && v.size(1) == q.size(1),
              "flash_attn_cuda: q/k/v disagree on batch or sequence length");
  TORCH_CHECK(k.size(2) == v.size(2),
              "flash_attn_cuda: k and v disagree on the KV head count");
  TORCH_CHECK(k.size(3) == q.size(3) && v.size(3) == q.size(3),
              "flash_attn_cuda: q/k/v disagree on head_dim");
  TORCH_CHECK(k.size(2) > 0 && q.size(2) % k.size(2) == 0,
              "flash_attn_cuda: n_head (", q.size(2),
              ") must be a positive multiple of n_kv_head (", k.size(2), ")");
}

}  // namespace

/// Returns {O [B,T,H,D] (q's dtype), L [B,H,T] (f32)}.
std::vector<at::Tensor> flash_attn_fwd(at::Tensor q, at::Tensor k, at::Tensor v,
                                       double scale, bool use_mma) {
  check_layout(q, k, v);
  const c10::cuda::CUDAGuard guard(q.device());

  Dims dm{static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
          static_cast<int>(q.size(2)), static_cast<int>(k.size(2)),
          static_cast<float>(scale)};
  const int head_dim = static_cast<int>(q.size(3));

  at::Tensor o = at::empty_like(q);
  at::Tensor l = at::empty({dm.B, dm.H, dm.T}, q.options().dtype(at::kFloat));
  if (dm.B == 0 || dm.T == 0 || dm.H == 0) return {o, l};

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  switch (q.scalar_type()) {
    case at::kFloat: {
      using scalar_t = float;
      TORCH_CHECK(!use_mma,
                  "flash_attn_cuda: the tensor-core path is f16/bf16 only -- "
                  "m16n8k16 has no f32 form, and the tf32 instruction (m16n8k8) "
                  "is a different fragment layout that this file does not "
                  "implement. Run float32 through the FMA kernel.");
      NANOLAB_ON_HEAD_DIM(head_dim, launch_fwd, q, k, v, o, l, dm, stream);
      break;
    }
    case at::kHalf: {
      using scalar_t = at::Half;
      if (use_mma) {
        NANOLAB_ON_HEAD_DIM(head_dim, launch_fwd_mma, q, k, v, o, l, dm, stream);
      } else {
        NANOLAB_ON_HEAD_DIM(head_dim, launch_fwd, q, k, v, o, l, dm, stream);
      }
      break;
    }
    case at::kBFloat16: {
      using scalar_t = at::BFloat16;
      if (use_mma) {
        NANOLAB_ON_HEAD_DIM(head_dim, launch_fwd_mma, q, k, v, o, l, dm, stream);
      } else {
        NANOLAB_ON_HEAD_DIM(head_dim, launch_fwd, q, k, v, o, l, dm, stream);
      }
      break;
    }
    default:
      TORCH_CHECK(false, "flash_attn_cuda: unsupported dtype ",
                  q.scalar_type(), " (have float32, float16, bfloat16)");
  }
  return {o, l};
}

/// Run one mma tile on caller-supplied matrices and hand back the result.
/// A [16,16] (row-major), B [16,8] (k by n, row-major) -> D [16,8] float32.
/// This exists so the fragment layout can be verified against the hardware
/// rather than trusted; see the layout note above `pack2`.
at::Tensor mma_probe(at::Tensor a, at::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(),
              "mma_probe: inputs must be CUDA tensors");
  TORCH_CHECK(a.dim() == 2 && a.size(0) == 16 && a.size(1) == 16,
              "mma_probe: A must be [16,16], got ", a.sizes());
  TORCH_CHECK(b.dim() == 2 && b.size(0) == 16 && b.size(1) == 8,
              "mma_probe: B must be [16,8], got ", b.sizes());
  TORCH_CHECK(a.scalar_type() == b.scalar_type(),
              "mma_probe: A and B dtypes must match");
  a = a.contiguous();
  b = b.contiguous();
  const c10::cuda::CUDAGuard guard(a.device());
  at::Tensor d = at::zeros({16, 8}, a.options().dtype(at::kFloat));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  switch (a.scalar_type()) {
    case at::kHalf:
      mma_probe_kernel<at::Half><<<1, 32, 0, stream>>>(
          a.const_data_ptr<at::Half>(), b.const_data_ptr<at::Half>(),
          d.data_ptr<float>());
      break;
    case at::kBFloat16:
      mma_probe_kernel<at::BFloat16><<<1, 32, 0, stream>>>(
          a.const_data_ptr<at::BFloat16>(), b.const_data_ptr<at::BFloat16>(),
          d.data_ptr<float>());
      break;
    default:
      TORCH_CHECK(false, "mma_probe: dtype must be float16 or bfloat16, got ",
                  a.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d;
}

/// Returns {dQ, dK, dV} in the input layouts.
std::vector<at::Tensor> flash_attn_bwd(at::Tensor dout, at::Tensor q,
                                       at::Tensor k, at::Tensor v,
                                       at::Tensor o, at::Tensor l,
                                       double scale, bool use_mma) {
  check_layout(q, k, v);
  TORCH_CHECK(dout.is_cuda() && o.is_cuda() && l.is_cuda(),
              "flash_attn_cuda: backward inputs must be CUDA tensors");
  TORCH_CHECK(dout.sizes() == q.sizes() && o.sizes() == q.sizes(),
              "flash_attn_cuda: dO/O must have Q's shape");
  TORCH_CHECK(dout.scalar_type() == q.scalar_type() &&
                  o.scalar_type() == q.scalar_type(),
              "flash_attn_cuda: dO/O dtypes must match Q's");
  TORCH_CHECK(l.scalar_type() == at::kFloat,
              "flash_attn_cuda: the LSE tape must be float32");
  TORCH_CHECK(l.dim() == 3 && l.size(0) == q.size(0) &&
                  l.size(1) == q.size(2) && l.size(2) == q.size(1),
              "flash_attn_cuda: expected an LSE tape of [B,H,T]");
  TORCH_CHECK(dout.is_contiguous() && o.is_contiguous() && l.is_contiguous(),
              "flash_attn_cuda: dO/O/L must be contiguous");
  const c10::cuda::CUDAGuard guard(q.device());

  Dims dm{static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
          static_cast<int>(q.size(2)), static_cast<int>(k.size(2)),
          static_cast<float>(scale)};
  const int head_dim = static_cast<int>(q.size(3));

  at::Tensor dq = at::empty_like(q);
  at::Tensor dk = at::empty_like(k);
  at::Tensor dv = at::empty_like(v);
  at::Tensor delta = at::empty({dm.B, dm.H, dm.T}, q.options().dtype(at::kFloat));
  if (dm.B == 0 || dm.T == 0 || dm.H == 0) return {dq, dk, dv};

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  switch (q.scalar_type()) {
    case at::kFloat: {
      using scalar_t = float;
      TORCH_CHECK(!use_mma,
                  "flash_attn_cuda: the tensor-core backward is f16/bf16 only");
      NANOLAB_ON_HEAD_DIM(head_dim, launch_bwd, q, k, v, o, dout, l, delta, dq,
                          dk, dv, dm, stream);
      break;
    }
    case at::kHalf: {
      using scalar_t = at::Half;
      if (use_mma) {
        NANOLAB_ON_HEAD_DIM_MMA_BWD(head_dim, launch_bwd_mma, q, k, v, o, dout,
                                    l, delta, dq, dk, dv, dm, stream);
      } else {
        NANOLAB_ON_HEAD_DIM(head_dim, launch_bwd, q, k, v, o, dout, l, delta,
                            dq, dk, dv, dm, stream);
      }
      break;
    }
    case at::kBFloat16: {
      using scalar_t = at::BFloat16;
      if (use_mma) {
        NANOLAB_ON_HEAD_DIM_MMA_BWD(head_dim, launch_bwd_mma, q, k, v, o, dout,
                                    l, delta, dq, dk, dv, dm, stream);
      } else {
        NANOLAB_ON_HEAD_DIM(head_dim, launch_bwd, q, k, v, o, dout, l, delta,
                            dq, dk, dv, dm, stream);
      }
      break;
    }
    default:
      TORCH_CHECK(false, "flash_attn_cuda: unsupported dtype ",
                  q.scalar_type(), " (have float32, float16, bfloat16)");
  }
  return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &flash_attn_fwd,
        "FA-2 causal GQA forward; use_mma selects the tensor-core path "
        "(returns {O, LSE})");
  m.def("mma_probe", &mma_probe,
        "One mma.m16n8k16 tile through this file's fragment layout, for "
        "verifying that layout against the hardware");
  m.def("bwd", &flash_attn_bwd,
        "FA-2 causal GQA backward; use_mma selects the tensor-core path "
        "(returns {dQ, dK, dV})");
}
