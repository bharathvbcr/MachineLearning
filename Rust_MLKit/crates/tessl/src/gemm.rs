//! GEMM dispatch: TensorOps `matmul2d` (preferred) or `simdgroup_matrix` fallback.
//!
//! GEMM v2: Morton 1D TG walk, packed zero+matmul (one binder), MLP/bf16 split-K,
//! `execution_simdgroups<4>` on bf16/relaxed kernels (see matmul_tensorops.metal).
//!
//! Phase H: `PrecisionMode::Bf16` uses bf16 TensorOps GEMMs (f32 accumulate).
//! Callers may keep persistent bf16 activation/weight buffers; `ensure_bf16`
//! is a no-op when the operand is already bf16. Residual/RMSNorm/CE stay f32.
//! Optional `relaxed_precision` (tf32-class) on f32 GEMMs as a bridge; off by
//! default for golden parity.

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::runtime::{mtl_size, GpuRuntime, PrecisionMode};
use crate::tensor::{DType, Tensor};

#[derive(Clone, Copy)]
enum Layout {
    NN,
    TN,
    NT,
}

/// All public GEMM paths validate before casting, allocating scratch, or encoding.
/// MPP uses signed 32-bit extents/offset arithmetic; reject larger matrices.
fn validate_gemm(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    layout: Layout,
    allow_bf16: bool,
) -> Result<(usize, usize, usize), String> {
    for t in [a, b, c] {
        t.validate()?;
        if t.shape.len() != 2 || t.shape.contains(&0) {
            return Err("GEMM requires nonempty rank-2 tensors".into());
        }
        if t.numel() > i32::MAX as usize {
            return Err("GEMM exceeds signed 32-bit kernel indexing".into());
        }
    }
    if !std::sync::Arc::ptr_eq(a.runtime(), b.runtime())
        || !std::sync::Arc::ptr_eq(a.runtime(), c.runtime())
    {
        return Err("GEMM tensors must belong to the same runtime".into());
    }
    if c.dtype != DType::F32 || (!allow_bf16 && (a.dtype != DType::F32 || b.dtype != DType::F32)) {
        return Err("GEMM operand dtype does not match the selected precision path".into());
    }
    let (m, k, k2, n) = match layout {
        Layout::NN => (a.shape[0], a.shape[1], b.shape[0], b.shape[1]),
        Layout::TN => (a.shape[1], a.shape[0], b.shape[0], b.shape[1]),
        Layout::NT => (a.shape[0], a.shape[1], b.shape[1], b.shape[0]),
    };
    if k != k2 || c.shape != [m, n] {
        return Err("GEMM inner dimensions or output shape do not match".into());
    }
    if a.overlaps(c) || b.overlaps(c) {
        return Err("GEMM output must not overlap either input".into());
    }
    Ok((m, n, k))
}

/// Tall-K / small-MN → split-K accumulate.
/// Attn dW: M=N=128, K=BT=4096. MLP dW: one side = mlp_dim=384.
fn prefer_tn_splitk(m: usize, n: usize, k: usize) -> bool {
    k >= 2048 && m <= 384 && n <= 384 && m.min(n) <= 128
}

/// Tile sizes for TensorOps kernels (must match matmul_tensorops.metal).
#[derive(Clone, Copy)]
struct TileGeom {
    sm: usize,
    sn: usize,
    /// Simdgroups per TG (`execution_simdgroups<N>`). Exact f32 uses 1.
    simdgroups: usize,
}

const TILE_F32: TileGeom = TileGeom {
    sm: 32,
    sn: 32,
    simdgroups: 1,
};
const TILE_V2: TileGeom = TileGeom {
    sm: 64,
    sn: 32,
    simdgroups: 4,
};
/// NN bf16 only. Must track `matmul2d_tensorops_bf16_f32`'s compile-time SM/SN;
/// the TN/NT/split-K bf16 kernels still use TILE_V2.
const TILE_BF16_NN: TileGeom = TileGeom {
    sm: 64,
    sn: 64,
    simdgroups: 4,
};

/// Preconditions shared by both `*_coop` NN kernels (bf16 and relaxed-f32).
/// They hold the C accumulator in registers for the whole K loop, so they have
/// no ragged or short-K branches: every tile must be interior, and K must
/// divide into `COOP_BKC` blocks.
///
/// `COOP_MIN_K` is where the win starts, and it is structural rather than
/// tuned. Both blocked kernels use BK=256, so below K=512 they run at most one
/// full block plus a tail — already a single C store — and the coop kernel only
/// adds the cost of zeroing the register accumulator. Paired interleaved
/// measurement (`bench_gemm_coop_ab`, which cancels clock drift) agrees: at
/// K=256 the coop kernels are 0.90x (bf16) and 0.93x (relaxed) of blocked,
/// crossing over to 1.02-1.13x from K=512 up.
const COOP_BKC: usize = 128;
const COOP_MIN_K: usize = 512;

/// True when the `*_coop` kernel compiled for `tile` may be dispatched. `tile`
/// must be the same TileGeom the caller then dispatches with, so that the
/// interior guarantee this returns is the one the kernel actually relies on.
fn use_coop_nn(tile: TileGeom, m: usize, n: usize, k: usize) -> bool {
    // `>= tile` as well as `% tile == 0`: divisibility alone also accepts M or
    // N of zero, and this predicate is the kernels' only guard.
    m >= tile.sm
        && m % tile.sm == 0
        && n >= tile.sn
        && n % tile.sn == 0
        && k >= COOP_MIN_K
        && k % COOP_BKC == 0
}

/// The TensorOps NN kernel `gemm` dispatches for these dimensions.
///
/// Factored out of `gemm` for two reasons: the coop gate then has exactly one
/// evaluation site, and tests can ask which path a shape *actually* takes
/// instead of assuming. A shape fuzz that silently never reaches the coop
/// kernels would pass just as loudly as one that covers them.
pub(crate) fn tensorops_nn_kernel(bf16: bool, relaxed: bool, m: usize, n: usize, k: usize) -> &'static str {
    if bf16 {
        if use_coop_nn(TILE_BF16_NN, m, n, k) {
            "matmul2d_tensorops_bf16_f32_coop"
        } else {
            "matmul2d_tensorops_bf16_f32"
        }
    } else if relaxed {
        if use_coop_nn(TILE_F32R_NN, m, n, k) {
            "matmul2d_tensorops_f32_relaxed_coop"
        } else {
            "matmul2d_tensorops_f32_relaxed"
        }
    } else {
        "matmul2d_tensorops_f32"
    }
}
/// NN relaxed-f32 only. Must track `matmul2d_tensorops_f32_relaxed`'s SM/SN.
/// Wider than the bf16 tile: f32 operands are 2x the bytes, so the same tile
/// carries half the arithmetic intensity. sg4 over sg8 — sg8 measured faster on
/// square shapes but regressed narrow-N (mlp_down, N=768) below baseline.
const TILE_F32R_NN: TileGeom = TileGeom {
    sm: 128,
    sn: 64,
    simdgroups: 4,
};
/// Plain (non-accumulating) TN/NT bf16. Must track the SM/SN compiled into
/// `matmul2d_tensorops_{tn,nt}_bf16_f32`. The split-K and `_accum_` bf16
/// kernels are separate and stay on TILE_V2.
const TILE_TNNT_BF16: TileGeom = TileGeom {
    sm: 128,
    sn: 64,
    simdgroups: 4,
};
/// Accumulating dW path (`matmul2d_tensorops_tn_accum_bf16_f32`). Measured
/// separately from the plain kernels: TN-accum and NT-accum peak at different
/// tiles, and each regresses below baseline on the other's choice.
const TILE_TNACC_BF16: TileGeom = TileGeom {
    sm: 128,
    sn: 64,
    simdgroups: 4,
};
/// Accumulating dX path (`matmul2d_tensorops_nt_accum_bf16_f32`).
const TILE_NTACC_BF16: TileGeom = TileGeom {
    sm: 64,
    sn: 64,
    simdgroups: 4,
};

/// Exact 1D TG count for a `tiles_n × tiles_m` rectangle (no power-of-two pad —
/// padding blew up tall NN shapes like BT×C and erased the binder win).
fn morton_tg_count(tiles_n: usize, tiles_m: usize) -> usize {
    tiles_n.saturating_mul(tiles_m).max(1)
}

/// Live TN/NT TensorOps descriptors (transpose_left/right). Fixed multi-tile
/// slice axes: TN slices A's M on dim0; NT slices B's N on dim1.
const USE_TN_NT_DESCRIPTORS: bool = true;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmBackend {
    /// MPP TensorOps `matmul2d` (Metal 4 / macOS 26+, M5 accelerators).
    TensorOps,
    /// Hand-tiled `simdgroup_matrix` portable path.
    Simdgroup,
}

impl GemmBackend {
    pub fn kernel_name_f32(self) -> &'static str {
        match self {
            GemmBackend::TensorOps => "matmul2d_tensorops_f32",
            GemmBackend::Simdgroup => "matmul_simdgroup_f32",
        }
    }

    pub fn kernel_name_f32_relaxed(self) -> &'static str {
        match self {
            GemmBackend::TensorOps => "matmul2d_tensorops_f32_relaxed",
            GemmBackend::Simdgroup => "matmul_simdgroup_f32",
        }
    }

    pub fn kernel_name_bf16(self) -> &'static str {
        match self {
            GemmBackend::TensorOps => "matmul2d_tensorops_bf16_f32",
            // No simdgroup bf16 kernel — callers cast to f32 first.
            GemmBackend::Simdgroup => "matmul_simdgroup_f32",
        }
    }
}

/// Pick TensorOps when the metallib contains it; else simdgroup.
pub fn select_backend(rt: &GpuRuntime) -> GemmBackend {
    if rt.has_tensorops() {
        GemmBackend::TensorOps
    } else {
        GemmBackend::Simdgroup
    }
}

fn validate_cast_input(src: &Tensor, dtype: DType) -> Result<(), String> {
    src.validate()?;
    if src.dtype != dtype || src.numel() == 0 || src.numel() > u32::MAX as usize {
        return Err("cast requires the declared dtype and 1..=u32::MAX elements".into());
    }
    Ok(())
}

/// Cast f32 tensor → bf16 (GPU). Used at GEMM boundaries under `PrecisionMode::Bf16`.
pub fn cast_f32_to_bf16(src: &Tensor) -> Result<Tensor, String> {
    validate_cast_input(src, DType::F32)?;
    let rt = src.runtime();
    let dst = rt.alloc_tensor_bf16(&src.shape)?;
    cast_f32_to_bf16_into(src, &dst)?;
    Ok(dst)
}

/// Cast into an existing bf16 buffer (persistent weight banks).
pub fn cast_f32_to_bf16_into(src: &Tensor, dst: &Tensor) -> Result<(), String> {
    validate_cast_input(src, DType::F32)?;
    dst.validate()?;
    if dst.dtype != DType::BF16
        || src.shape != dst.shape
        || !std::sync::Arc::ptr_eq(src.runtime(), dst.runtime())
        || src.overlaps(dst)
    {
        return Err(
            "cast destination must match shape/runtime, be bf16, and not overlap source".into(),
        );
    }
    let rt = src.runtime();
    let p = rt.pipeline("cast_f32_to_bf16")?;
    let n = src.numel();
    crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, dst, 1);
        crate::dispatch::set_u32(bnd, n as u32, 2);
    })?;
    Ok(())
}

/// Hot-resident bf16 clone of an f32 master (weights / EMA banks).
pub fn cast_f32_to_bf16_hot(src: &Tensor) -> Result<Tensor, String> {
    validate_cast_input(src, DType::F32)?;
    let rt = src.runtime();
    let dst = rt.alloc_tensor_bf16_hot(&src.shape)?;
    cast_f32_to_bf16_into(src, &dst)?;
    Ok(dst)
}

/// Cast bf16 tensor → f32 (GPU).
pub fn cast_bf16_to_f32(src: &Tensor) -> Result<Tensor, String> {
    validate_cast_input(src, DType::BF16)?;
    let rt = src.runtime();
    let dst = rt.alloc_tensor_f32(&src.shape)?;
    let p = rt.pipeline("cast_bf16_to_f32")?;
    let n = src.numel();
    crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, &dst, 1);
        crate::dispatch::set_u32(bnd, n as u32, 2);
    })?;
    Ok(dst)
}

fn ensure_bf16(t: &Tensor) -> Result<Tensor, String> {
    match t.dtype {
        DType::BF16 => Ok(t.clone()),
        DType::F32 => cast_f32_to_bf16(t),
    }
}

fn use_bf16_gemm(rt: &GpuRuntime, backend: GemmBackend) -> bool {
    rt.precision() == PrecisionMode::Bf16
        && backend == GemmBackend::TensorOps
        && rt.has_tensorops()
}

fn use_relaxed_f32(rt: &GpuRuntime, backend: GemmBackend) -> bool {
    rt.relaxed_precision()
        && rt.precision() == PrecisionMode::F32
        && backend == GemmBackend::TensorOps
        && rt.has_tensorops()
}

/// C[M,N] = A[M,K] @ B[K,N]. Overwrites C.
///
/// - f32×f32→f32 always supported (exact or relaxed via runtime flag)
/// - bf16×bf16→f32 accum (C must be f32) via TensorOps when available
pub fn gemm(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a, b, c, Layout::NN, true)?;

    let use_bf16 = a.dtype == DType::BF16 && b.dtype == DType::BF16;
    if a.dtype != b.dtype || (use_bf16 && backend != GemmBackend::TensorOps) {
        return Err("GEMM requires matching operand dtypes; bf16 requires TensorOps".into());
    }

    let rt = a.runtime();
    // bf16 already implies TensorOps: the check above rejects any other backend.
    let kernel = if use_bf16 {
        tensorops_nn_kernel(true, false, m, n, k)
    } else if use_relaxed_f32(rt, backend) {
        tensorops_nn_kernel(false, true, m, n, k)
    } else if backend == GemmBackend::Simdgroup && (m % 16 != 0 || n % 16 != 0 || k % 8 != 0) {
        "matmul_simdgroup_edges_f32"
    } else {
        backend.kernel_name_f32()
    };
    let pipeline = rt.pipeline(kernel)?;

    match backend {
        GemmBackend::TensorOps => {
            let tile = if use_bf16 {
                TILE_BF16_NN
            } else if use_relaxed_f32(rt, backend) {
                TILE_F32R_NN
            } else {
                TILE_F32
            };
            // Zero-tax: pack C-zero + matmul into one binder (~−1 binder/GEMM).
            let seeds_c = use_bf16 || use_relaxed_f32(rt, backend);
            dispatch_tensorops_nn(rt, &pipeline, a, b, c, m, n, k, tile, !seeds_c)?;
        }
        GemmBackend::Simdgroup => {
            // Both simdgroup kernels overwrite every logical output element.
            // No pre-zero dispatch or barrier is needed (including offset views).
            let m_u = m as u32;
            let n_u = n as u32;
            let k_u = k as u32;
            let (tg_w, tg_h, tpt) = threadgroup_geometry_simdgroup(&pipeline, m, n);
            rt.with_binder(|bnd| {
                bnd.set_pipeline(&pipeline);
                bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
                bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
                bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
                bnd.bind_u32(m_u, 3);
                bnd.bind_u32(n_u, 4);
                bnd.bind_u32(k_u, 5);
                bnd.dispatch(mtl_size(tg_w, tg_h, 1), mtl_size(tpt, 1, 1));
                Ok(())
            })?;
        }
    }

    Ok(())
}

/// Pack `zero_f32(C)` + TensorOps NN matmul into a single Metal 4 binder.
fn dispatch_tensorops_nn(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
    // False only for kernels whose first K block uses `mode::multiply` and so
    // seed C themselves; every other NN kernel accumulates from block 0.
    zero_c: bool,
) -> Result<(), String> {
    let zero_p = rt.pipeline("zero_f32")?;
    let numel = c.numel();
    let tiles_n = (n + tile.sn - 1) / tile.sn;
    let tiles_m = (m + tile.sm - 1) / tile.sm;
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(pipeline, tile);
    let z_width = zero_p.threadExecutionWidth() as usize;
    let z_tpt = z_width.min(numel).max(1);
    let z_groups = (numel + z_tpt - 1) / z_tpt;

    rt.with_binder(|bnd| {
        if zero_c {
            bnd.set_pipeline(&zero_p);
            bnd.bind_tensor(c, 0);
            bnd.bind_u32(numel as u32, 1);
            bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
            // Explicit barrier only when auto per-dispatch barriers are off.
            if crate::ab_flags::hazard_barriers() {
                bnd.barrier();
            }
        }

        bnd.set_pipeline(pipeline);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        // f32 exact NN/TN/NT read buffer(8); bf16/relaxed ignore extra bind.
        bnd.bind_u32(
            if crate::ab_flags::gemm_interior_offsets() {
                1
            } else {
                0
            },
            8,
        );
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

fn dispatch_tensorops_tn_nt(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
) -> Result<(), String> {
    // Same binder packing as NN. Plain TN/NT kernels are single-shot
    // `mode::multiply`, so they seed C themselves — no zero pre-pass.
    dispatch_tensorops_nn(rt, pipeline, a, b, c, m, n, k, tile, false)
}

/// TensorOps matmul with `mode::multiply_accumulate` — no C zero (1 binder).
fn dispatch_tensorops_accum(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
    bind_interior: bool,
) -> Result<(), String> {
    let tiles_n = (n + tile.sn - 1) / tile.sn;
    let tiles_m = (m + tile.sm - 1) / tile.sm;
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(pipeline, tile);

    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        if bind_interior {
            bnd.bind_u32(
                if crate::ab_flags::gemm_interior_offsets() {
                    1
                } else {
                    0
                },
                8,
            );
        }
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// Convenience: f32 GEMM (parity path). Honors `relaxed_precision` when set.
pub fn gemm_f32(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    gemm(a, b, c, backend)
}

/// Training GEMM: under `PrecisionMode::Bf16` uses bf16 TensorOps (f32 accum into
/// `c`). Already-bf16 operands skip cast (persistent bf16 activations/weights).
/// Falls back to f32 GEMM when TensorOps is absent.
pub fn gemm_train(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    validate_gemm(a, b, c, Layout::NN, true)?;
    let rt = a.runtime();
    if use_bf16_gemm(rt, backend) {
        let a_bf = ensure_bf16(a)?;
        let b_bf = ensure_bf16(b)?;
        assert_eq!(c.dtype, DType::F32);
        return gemm(&a_bf, &b_bf, c, backend);
    }
    gemm_f32(a, b, c, backend)
}

/// C[M,N] = A[K,M]^T @ B[K,N] (TN). A is stored [K,M], B [K,N].
pub fn gemm_tn_f32(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_km, b_kn, c, Layout::TN, false)?;

    if USE_TN_NT_DESCRIPTORS
        && backend == GemmBackend::TensorOps
        && a_km.runtime().has_tensorops()
    {
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_f32(a_km, b_kn, c, k);
        }
        let rt = a_km.runtime();
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, a_km, b_kn, c, m, n, k, TILE_F32);
    }

    // Default: explicit transpose + NN (golden-safe).
    let at = {
        let rt = a_km.runtime();
        let out = rt.alloc_temp_f32(&[m, k])?;
        let p = rt.pipeline("transpose2d_f32")?;
        crate::dispatch::dispatch_1d(rt, &p, m * k, |bnd| {
            crate::dispatch::set_tensor(bnd, a_km, 0);
            crate::dispatch::set_tensor(bnd, &out, 1);
            crate::dispatch::set_u32(bnd, k as u32, 2);
            crate::dispatch::set_u32(bnd, m as u32, 3);
        })?;
        out
    };
    gemm_f32(&at, b_kn, c, backend)
}

/// Training TN GEMM — bf16 TensorOps descriptor when `PrecisionMode::Bf16`.
pub fn gemm_tn_train(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    validate_gemm(a_km, b_kn, c, Layout::TN, use_bf16_gemm(a_km.runtime(), backend))?;
    let rt = a_km.runtime();
    if use_bf16_gemm(rt, backend) {
        assert_eq!(c.dtype, DType::F32);
        let a_bf = ensure_bf16(a_km)?;
        let b_bf = ensure_bf16(b_kn)?;
        let k = a_bf.shape[0];
        let m = a_bf.shape[1];
        let n = b_bf.shape[1];
        assert_eq!(c.shape, &[m, n]);
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_bf16(&a_bf, &b_bf, c, k);
        }
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_bf16_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, &a_bf, &b_bf, c, m, n, k, TILE_TNNT_BF16);
    }
    gemm_tn_f32(a_km, b_kn, c, backend)
}

fn gemm_tn_splitk_f32(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    k: usize,
) -> Result<(), String> {
    gemm_tn_splitk_f32_opts(a_km, b_kn, c, k, /*zero_first=*/ true)
}

fn gemm_tn_splitk_f32_opts(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    k: usize,
    zero_first: bool,
) -> Result<(), String> {
    let m = a_km.shape[1];
    let n = b_kn.shape[1];
    let rt = a_km.runtime();
    let pipeline = rt.pipeline("matmul2d_tensorops_tn_splitk_f32")?;
    let zero_p = rt.pipeline("zero_f32")?;
    let tile = TILE_F32;
    let tiles_n = (n + tile.sn - 1) / tile.sn;
    let tiles_m = (m + tile.sm - 1) / tile.sm;
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(&pipeline, tile);
    let numel = c.numel();
    let z_width = zero_p.threadExecutionWidth() as usize;
    let z_tpt = z_width.min(numel).max(1);
    let z_groups = (numel + z_tpt - 1) / z_tpt;
    let k_tile = 256u32;
    let partitions: Vec<u32> = (0..k as u32).step_by(k_tile as usize).collect();

    // Zero once (optional) + all K-partitions in one binder.
    let need_explicit = crate::ab_flags::hazard_barriers();
    rt.with_binder(|bnd| {
        if zero_first {
            bnd.set_pipeline(&zero_p);
            bnd.bind_tensor(c, 0);
            bnd.bind_u32(numel as u32, 1);
            bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
            if need_explicit {
                bnd.barrier();
            }
        }

        bnd.set_pipeline(&pipeline);
        for (pi, &k0) in partitions.iter().enumerate() {
            if pi > 0 && need_explicit {
                bnd.barrier();
            }
            bnd.bind_buf(a_km.buffer.metal(), a_km.byte_offset, 0);
            bnd.bind_buf(b_kn.buffer.metal(), b_kn.byte_offset, 1);
            bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
            bnd.bind_u32(m as u32, 3);
            bnd.bind_u32(n as u32, 4);
            bnd.bind_u32(k as u32, 5);
            bnd.bind_u32(k0, 6);
            bnd.bind_u32(k_tile, 7);
            bnd.bind_u32(tiles_n as u32, 8);
            bnd.bind_u32(tiles_m as u32, 9);
            bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })?;
    Ok(())
}

fn gemm_tn_splitk_bf16(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    k: usize,
) -> Result<(), String> {
    gemm_tn_splitk_bf16_opts(a_km, b_kn, c, k, /*zero_first=*/ true)
}

fn gemm_tn_splitk_bf16_opts(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    k: usize,
    zero_first: bool,
) -> Result<(), String> {
    let m = a_km.shape[1];
    let n = b_kn.shape[1];
    let rt = a_km.runtime();
    let pipeline = rt.pipeline("matmul2d_tensorops_tn_splitk_bf16_f32")?;
    let zero_p = rt.pipeline("zero_f32")?;
    let tile = TILE_V2;
    let tiles_n = (n + tile.sn - 1) / tile.sn;
    let tiles_m = (m + tile.sm - 1) / tile.sm;
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(&pipeline, tile);
    let numel = c.numel();
    let z_width = zero_p.threadExecutionWidth() as usize;
    let z_tpt = z_width.min(numel).max(1);
    let z_groups = (numel + z_tpt - 1) / z_tpt;
    let k_tile = 256u32;
    let partitions: Vec<u32> = (0..k as u32).step_by(k_tile as usize).collect();

    let need_explicit = crate::ab_flags::hazard_barriers();
    rt.with_binder(|bnd| {
        if zero_first {
            bnd.set_pipeline(&zero_p);
            bnd.bind_tensor(c, 0);
            bnd.bind_u32(numel as u32, 1);
            bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
            if need_explicit {
                bnd.barrier();
            }
        }

        bnd.set_pipeline(&pipeline);
        for (pi, &k0) in partitions.iter().enumerate() {
            if pi > 0 && need_explicit {
                bnd.barrier();
            }
            bnd.bind_buf(a_km.buffer.metal(), a_km.byte_offset, 0);
            bnd.bind_buf(b_kn.buffer.metal(), b_kn.byte_offset, 1);
            bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
            bnd.bind_u32(m as u32, 3);
            bnd.bind_u32(n as u32, 4);
            bnd.bind_u32(k as u32, 5);
            bnd.bind_u32(k0, 6);
            bnd.bind_u32(k_tile, 7);
            bnd.bind_u32(tiles_n as u32, 8);
            bnd.bind_u32(tiles_m as u32, 9);
            bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })?;
    Ok(())
}

/// C[M,N] = A[M,K] @ B[N,K]^T (NT). B is stored [N,K] (e.g. W[in,out]).
pub fn gemm_nt_f32(
    a_mk: &Tensor,
    b_nk: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_mk, b_nk, c, Layout::NT, false)?;

    if USE_TN_NT_DESCRIPTORS
        && backend == GemmBackend::TensorOps
        && a_mk.runtime().has_tensorops()
    {
        let rt = a_mk.runtime();
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, a_mk, b_nk, c, m, n, k, TILE_F32);
    }

    let bt = {
        let rt = b_nk.runtime();
        let out = rt.alloc_temp_f32(&[k, n])?;
        let p = rt.pipeline("transpose2d_f32")?;
        crate::dispatch::dispatch_1d(rt, &p, n * k, |bnd| {
            crate::dispatch::set_tensor(bnd, b_nk, 0);
            crate::dispatch::set_tensor(bnd, &out, 1);
            crate::dispatch::set_u32(bnd, n as u32, 2);
            crate::dispatch::set_u32(bnd, k as u32, 3);
        })?;
        out
    };
    gemm_f32(a_mk, &bt, c, backend)
}

/// Training NT GEMM — bf16 TensorOps descriptor when `PrecisionMode::Bf16`.
pub fn gemm_nt_train(
    a_mk: &Tensor,
    b_nk: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    validate_gemm(a_mk, b_nk, c, Layout::NT, use_bf16_gemm(a_mk.runtime(), backend))?;
    let rt = a_mk.runtime();
    if use_bf16_gemm(rt, backend) {
        assert_eq!(c.dtype, DType::F32);
        let a_bf = ensure_bf16(a_mk)?;
        let b_bf = ensure_bf16(b_nk)?;
        let m = a_bf.shape[0];
        let k = a_bf.shape[1];
        let n = b_bf.shape[0];
        assert_eq!(c.shape, &[m, n]);
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_bf16_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, &a_bf, &b_bf, c, m, n, k, TILE_TNNT_BF16);
    }
    gemm_nt_f32(a_mk, b_nk, c, backend)
}

/// C += A[K,M]^T @ B[K,N] (TN accumulate). No C zero — for dW into grad banks
/// and dx accumulate into a pre-zeroed buffer.
pub fn gemm_tn_accum_train(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_km, b_kn, c, Layout::TN, use_bf16_gemm(a_km.runtime(), backend))?;

    let rt = a_km.runtime();
    let use_accum = crate::ab_flags::gemm_accum();
    if use_accum && use_bf16_gemm(rt, backend) {
        let a_bf = ensure_bf16(a_km)?;
        let b_bf = ensure_bf16(b_kn)?;
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_bf16_opts(&a_bf, &b_bf, c, k, /*zero_first=*/ false);
        }
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_accum_bf16_f32")?;
        return dispatch_tensorops_accum(
            rt, &pipeline, &a_bf, &b_bf, c, m, n, k, TILE_TNACC_BF16, /*bind_interior=*/ false,
        );
    }

    if use_accum && USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && rt.has_tensorops()
    {
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_f32_opts(a_km, b_kn, c, k, /*zero_first=*/ false);
        }
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_accum_f32")?;
        return dispatch_tensorops_accum(
            rt, &pipeline, a_km, b_kn, c, m, n, k, TILE_F32, /*bind_interior=*/ true,
        );
    }

    // Fallback / Soft-bisect: temp + add (pre–Audit 6 P1a/P1a2 numerics).
    let tmp = rt.alloc_temp_f32(&[m, n])?;
    gemm_tn_train(a_km, b_kn, &tmp, backend)?;
    let p = rt.pipeline("add_inplace_f32")?;
    crate::dispatch::dispatch_1d(rt, &p, c.numel(), |bnd| {
        crate::dispatch::set_tensor(bnd, c, 0);
        crate::dispatch::set_tensor(bnd, &tmp, 1);
        crate::dispatch::set_u32(bnd, c.numel() as u32, 2);
    })?;
    Ok(())
}

/// C += A[M,K] @ B[N,K]^T (NT accumulate). No C zero.
///
/// All call sites are **dX-class** accumulations into fresh pre-zeroed
/// activation-grad buffers (never weight banks), so this path additionally
/// honors `TESSL_GEMM_ACCUM_DX` — accumulate-mode dX with dW kept on the
/// Soft-safe temp+add path (arch_02 Audit 7).
pub fn gemm_nt_accum_train(
    a_mk: &Tensor,
    b_nk: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_mk, b_nk, c, Layout::NT, use_bf16_gemm(a_mk.runtime(), backend))?;

    let rt = a_mk.runtime();
    let use_accum = crate::ab_flags::gemm_accum() || crate::ab_flags::gemm_accum_dx();
    if use_accum && use_bf16_gemm(rt, backend) {
        let a_bf = ensure_bf16(a_mk)?;
        let b_bf = ensure_bf16(b_nk)?;
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_accum_bf16_f32")?;
        return dispatch_tensorops_accum(
            rt, &pipeline, &a_bf, &b_bf, c, m, n, k, TILE_NTACC_BF16, /*bind_interior=*/ false,
        );
    }

    if use_accum && USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && rt.has_tensorops()
    {
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_accum_f32")?;
        return dispatch_tensorops_accum(
            rt, &pipeline, a_mk, b_nk, c, m, n, k, TILE_F32, /*bind_interior=*/ true,
        );
    }

    let tmp = rt.alloc_temp_f32(&[m, n])?;
    gemm_nt_train(a_mk, b_nk, &tmp, backend)?;
    let p = rt.pipeline("add_inplace_f32")?;
    crate::dispatch::dispatch_1d(rt, &p, c.numel(), |bnd| {
        crate::dispatch::set_tensor(bnd, c, 0);
        crate::dispatch::set_tensor(bnd, &tmp, 1);
        crate::dispatch::set_u32(bnd, c.numel() as u32, 2);
    })?;
    Ok(())
}

/// Prefer bf16 / relaxed GEMM per runtime precision policy.
pub fn gemm_auto(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    gemm_train(a, b, c, backend)
}

fn threads_per_tg(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    tile: TileGeom,
) -> usize {
    let width = pipeline.threadExecutionWidth() as usize;
    width * tile.simdgroups
}

fn threadgroup_geometry_simdgroup(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    m: usize,
    n: usize,
) -> (usize, usize, usize) {
    let width = pipeline.threadExecutionWidth() as usize;
    let tg_w = (n + 15) / 16;
    let tg_h = (m + 15) / 16;
    (tg_w, tg_h, width * 4)
}

/// CPU reference GEMM for tests.
pub fn gemm_f32_cpu(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[cfg(test)]
mod tests {

    /// The `*_coop` kernels have no ragged, short-K or tail branch: they trust
    /// `use_coop_nn` completely. Every rejection below is a case where the
    /// kernel would produce a wrong answer with no error reported, so this
    /// pins the predicate rather than the kernels' behaviour under it.
    #[test]
    fn coop_nn_gate_rejects_every_shape_the_kernels_cannot_handle() {
        let bf16 = TILE_BF16_NN;
        let f32r = TILE_F32R_NN;

        // Accepted: exactly on the boundary, and comfortably past it.
        assert!(use_coop_nn(bf16, 128, 128, COOP_MIN_K));
        assert!(use_coop_nn(bf16, 4096, 768, 2304));
        assert!(use_coop_nn(f32r, 256, 192, 768));
        assert!(use_coop_nn(f32r, 4096, 768, 4096));

        // K below the crossover: the blocked kernel does at most one full block
        // plus a tail there and measured faster.
        assert!(!use_coop_nn(bf16, 128, 128, COOP_MIN_K - 1));
        assert!(!use_coop_nn(bf16, 128, 128, 384));
        assert!(!use_coop_nn(f32r, 128, 128, 256));

        // K not a whole number of BKC blocks: the kernel's `k + BKC <= K` loop
        // would drop the tail and silently under-compute.
        assert!(!use_coop_nn(bf16, 128, 128, COOP_MIN_K + 8));
        assert!(!use_coop_nn(bf16, 128, 128, 520));
        assert!(!use_coop_nn(f32r, 256, 128, 1000));
        for k in [513usize, 575, 639, 767, 1151] {
            assert!(!use_coop_nn(bf16, 128, 128, k), "K={k} is not a BKC multiple");
        }

        // Ragged tiles: the kernel has no edge path, so a partial tile would
        // read and write outside the logical matrix.
        assert!(!use_coop_nn(bf16, 127, 128, 1024));
        assert!(!use_coop_nn(bf16, 128, 127, 1024));
        assert!(!use_coop_nn(f32r, 64, 128, 1024), "M=64 is ragged for a 128-row tile");
        assert!(!use_coop_nn(f32r, 128, 32, 1024), "N=32 is ragged for a 64-col tile");

        // The two NN tiles differ, and using the wrong one is a live hazard:
        // M=192 is interior for bf16 (64 rows) but ragged for relaxed (128).
        assert!(use_coop_nn(bf16, 192, 64, 1152));
        assert!(!use_coop_nn(f32r, 192, 64, 1152));

        // Degenerate shapes must never reach a kernel with no edge path.
        assert!(!use_coop_nn(bf16, 0, 128, 1024));
        assert!(!use_coop_nn(bf16, 128, 0, 1024));
        assert!(!use_coop_nn(bf16, 1, 1, 1));
        assert!(!use_coop_nn(f32r, 1, 1, 4096));
    }

    use super::*;
    use crate::GpuRuntime;

    fn max_abs_err(got: &[f32], exp: &[f32]) -> f32 {
        assert_eq!(got.len(), exp.len(), "parity length mismatch");
        assert!(got.iter().chain(exp).all(|x| x.is_finite()), "nonfinite parity input");
        got.iter()
            .zip(exp.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max)
    }

    /// Same invariant as the bf16 case, for the relaxed-precision (tf32-class)
    /// f32 NN GEMM: its first K block uses `mode::multiply`, so the host skips
    /// `zero_f32(C)`. Regressing that, or desyncing the kernel's SM/SN from
    /// TILE_F32R_NN, leaves stale C behind.
    #[test]
    fn relaxed_f32_gemm_nn_overwrites_c_without_prezero() {
        let rt = GpuRuntime::new().expect("GpuRuntime");
        if !rt.has_tensorops() {
            eprintln!("skip: metallib has no TensorOps");
            return;
        }
        rt.set_relaxed_precision(true);
        let (m, n, k) = (256usize, 128usize, 512usize);
        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let ah: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        let bh: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) / 32.0).collect();
        a.buffer.write_f32(&ah);
        b.buffer.write_f32(&bh);

        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        c.buffer.write_f32(&vec![1.0e30f32; m * n]); // garbage a zero-pass would hide
        gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32()[..m * n].to_vec();
        rt.set_relaxed_precision(false);

        let mut worst = 0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += ah[i * k + kk] as f64 * bh[kk * n + j] as f64;
                }
                assert!(got[i * n + j].is_finite(), "C[{i},{j}] not finite — stale value survived");
                worst = worst.max((got[i * n + j] as f64 - acc).abs());
            }
        }
        // tf32-class mantissa truncation, so looser than exact f32 but far
        // tighter than any surviving-garbage failure.
        assert!(worst < 1e-1, "relaxed NN GEMM max abs err {worst:.3e} — C was not fully overwritten");
    }

    /// Randomized shape fuzz across the NN dispatch paths.
    ///
    /// `gemm_adversarial_shape_sweep` pins a hand-picked list, which only ever
    /// proves the cases someone thought of. The `*_coop` gate has four
    /// independent conditions (M tile, N tile, K floor, K divisibility) over two
    /// different tiles, and the kernels behind it have no ragged or tail path —
    /// so the interesting failures live at combinations of boundaries, not at
    /// any single one. This samples those combinations directly.
    ///
    /// Deterministic: a failure prints the seed and shape, and re-running with
    /// `GEMM_FUZZ_SEED` reproduces it exactly. `GEMM_FUZZ_CASES` scales the run
    /// for a longer soak without editing the test.
    #[test]
    fn gemm_randomized_shape_fuzz() {
        use crate::gemm::{cast_f32_to_bf16, gemm, tensorops_nn_kernel, GemmBackend};
        use crate::runtime::PrecisionMode;
        use std::collections::BTreeMap;
        let rt = GpuRuntime::new().expect("GpuRuntime");
        if !rt.has_tensorops() {
            eprintln!("skip: metallib has no TensorOps");
            return;
        }
        // Parse strictly and panic on a malformed value. Falling back to the
        // default would make a soak over several seeds silently re-run one
        // seed and still report success — a check that could not run must not
        // look like a check that ran and passed.
        fn env_u64(name: &str, default: u64) -> u64 {
            match std::env::var(name) {
                Err(_) => default,
                Ok(v) => {
                    let t = v.trim();
                    let parsed = t
                        .strip_prefix("0x")
                        .or_else(|| t.strip_prefix("0X"))
                        .map(|h| u64::from_str_radix(h, 16))
                        .unwrap_or_else(|| t.parse::<u64>());
                    parsed.unwrap_or_else(|e| panic!("{name}={v:?} is not a valid integer: {e}"))
                }
            }
        }
        let seed = env_u64("GEMM_FUZZ_SEED", 0x5eed_1234_abcd_ef01);
        let cases = env_u64("GEMM_FUZZ_CASES", 120) as usize;

        let mut s = seed;
        let mut next = move || -> u64 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s >> 16
        };
        // Oversample tile and gate boundaries: a uniform draw over 1..320 almost
        // never lands on 64/128/512, which is exactly where the gate flips.
        //
        // Independent per-dimension sampling is not enough on its own. The coop
        // gate needs M, N and K to satisfy their conditions *simultaneously*,
        // so drawing each at ~1/5 leaves the coop kernels reached in well under
        // 1% of cases — a run that covers them only by luck. One case in three
        // is therefore built to satisfy the gate by construction, and the
        // coverage assertion at the end refuses to pass a run that missed.
        let dim = |r: u64| -> usize {
            let anchors = [32usize, 64, 96, 128, 192, 256, 320];
            match r % 4 {
                0 => 1 + (r / 4) as usize % 320,
                1 => anchors[(r / 4) as usize % anchors.len()],
                2 => anchors[(r / 4) as usize % anchors.len()].saturating_sub(1).max(1),
                _ => anchors[(r / 4) as usize % anchors.len()] + 1,
            }
        };
        // K additionally straddles COOP_MIN_K (512) and COOP_BKC (128).
        let kdim = |r: u64| -> usize {
            let anchors = [
                1usize, 127, 128, 129, 255, 256, 383, 384, 511, 512, 513, 640, 767, 768, 1024,
            ];
            match r % 3 {
                0 => 1 + (r / 3) as usize % 800,
                _ => anchors[(r / 3) as usize % anchors.len()],
            }
        };
        // Gate-satisfying by construction, for both NN tiles (128x64 is the
        // stricter of the two, so these also satisfy the 64x64 tile).
        let gated = |r: u64| -> (usize, usize, usize) {
            let m = 128 * (1 + (r % 3) as usize);
            let n = 64 * (1 + ((r / 3) % 4) as usize);
            let k = 128 * (4 + ((r / 12) % 5) as usize);
            (m, n, k)
        };

        let val = |i: usize, salt: usize| -> f32 {
            (((i * 2654435761 + salt * 40503) % 1021) as f32 - 510.0) / 512.0
        };
        const SENTINEL: f32 = 1.0e30;

        let mut checked = 0usize;
        let mut path_hits: BTreeMap<&'static str, usize> = BTreeMap::new();
        for case in 0..cases {
            let r = next();
            let (m, n, k) = if r % 3 == 0 {
                gated(next())
            } else {
                (dim(next()), dim(next()), kdim(next()))
            };

            let a_h: Vec<f32> = (0..m * k).map(|i| val(i, 1)).collect();
            let b_h: Vec<f32> = (0..k * n).map(|i| val(i, 2)).collect();
            let mut refv = vec![0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f64;
                    for kk in 0..k {
                        acc += a_h[i * k + kk] as f64 * b_h[kk * n + j] as f64;
                    }
                    refv[i * n + j] = acc;
                }
            }
            let refmax = refv.iter().fold(0f64, |a, x| a.max(x.abs())).max(1e-30);

            let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
            let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
            a.buffer.write_f32(&a_h);
            b.buffer.write_f32(&b_h);
            let a_bf = cast_f32_to_bf16(&a).unwrap();
            let b_bf = cast_f32_to_bf16(&b).unwrap();

            for &(label, bf16, relaxed, tol) in &[
                ("NN f32", false, false, 1e-4f64),
                ("NN relaxed", false, true, 2e-2),
                ("NN bf16", true, false, 2e-2),
            ] {
                rt.set_precision(if bf16 {
                    PrecisionMode::Bf16
                } else {
                    PrecisionMode::F32
                });
                rt.set_relaxed_precision(relaxed);
                *path_hits
                    .entry(tensorops_nn_kernel(bf16, relaxed, m, n, k))
                    .or_default() += 1;

                let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
                c.buffer.write_f32(&vec![SENTINEL; m * n]);
                let res = if bf16 {
                    gemm(&a_bf, &b_bf, &c, GemmBackend::TensorOps)
                } else {
                    gemm(&a, &b, &c, GemmBackend::TensorOps)
                };
                res.unwrap_or_else(|e| {
                    panic!("case {case} seed {seed:#x} {label} {m}x{n}x{k}: dispatch failed: {e}")
                });
                rt.synchronize().unwrap();
                let got = c.buffer.read_f32();
                for idx in 0..m * n {
                    let g = got[idx] as f64;
                    // A sentinel that survived means a tile was never written —
                    // the specific failure the coop kernels' missing ragged path
                    // would produce if the gate ever admitted a partial tile.
                    assert!(
                        g.abs() < 1e20,
                        "case {case} seed {seed:#x} {label} {m}x{n}x{k}: \
                         C[{idx}] still holds the sentinel — tile never written"
                    );
                    let rel = (g - refv[idx]).abs() / refmax;
                    assert!(
                        rel < tol,
                        "case {case} seed {seed:#x} {label} {m}x{n}x{k}: \
                         C[{idx}] rel err {rel:.3e} (got {g}, want {})",
                        refv[idx]
                    );
                }
                checked += 1;
            }
        }
        rt.set_precision(PrecisionMode::F32);
        rt.set_relaxed_precision(false);
        eprintln!(
            "gemm_randomized_shape_fuzz: {checked} (path, shape) combinations, seed {seed:#x}"
        );
        for (kern, hits) in &path_hits {
            eprintln!("  {kern:<44} {hits:>6}");
        }
        assert!(checked >= cases * 3);
        // Coverage is asserted, not assumed. Every kernel `gemm` can select for
        // an NN shape must have been exercised, or this run proved nothing
        // about the ones it skipped.
        for kern in [
            "matmul2d_tensorops_f32",
            "matmul2d_tensorops_f32_relaxed",
            "matmul2d_tensorops_f32_relaxed_coop",
            "matmul2d_tensorops_bf16_f32",
            "matmul2d_tensorops_bf16_f32_coop",
        ] {
            let hits = path_hits.get(kern).copied().unwrap_or(0);
            assert!(
                hits * 100 >= cases,
                "seed {seed:#x}: {kern} was selected for only {hits} of {cases} cases —                  the fuzz did not meaningfully exercise it"
            );
        }
    }

    /// Adversarial sweep over every TensorOps GEMM dispatch path.
    ///
    /// Shapes are chosen to break tiling assumptions: degenerate (1), primes,
    /// one-off-tile-boundary (63/65/127/129/257), exact tile multiples, extreme
    /// aspect ratios, and the split-K trigger region. Every output buffer is
    /// pre-seeded with a huge sentinel so any tile the kernel fails to write is
    /// caught rather than silently reading as a plausible number. Accumulating
    /// paths are seeded with a known addend and checked for C0 + A@B.
    #[test]
    fn gemm_adversarial_shape_sweep() {
        use crate::runtime::PrecisionMode;
        let rt = GpuRuntime::new().expect("GpuRuntime");
        if !rt.has_tensorops() {
            eprintln!("skip: metallib has no TensorOps");
            return;
        }
        const SENTINEL: f32 = 1.0e30;
        // (M, N, K)
        let shapes: &[(usize, usize, usize)] = &[
            (1, 1, 1),
            (1, 1, 512),
            (1, 384, 256),
            (384, 1, 256),
            (3, 5, 7),
            (31, 33, 65),
            (63, 65, 127),
            (127, 129, 33),
            (128, 128, 256),
            (129, 65, 257),
            (200, 100, 300),
            (255, 257, 129),
            (65, 512, 33),
            (128, 128, 2048),   // split-K trigger region
            (384, 128, 2304),   // split-K trigger region
            (128, 128, 520),    // K >= COOP_MIN_K but K % BKC != 0 -> blocked fallback
            (128, 128, 512),    // exactly COOP_MIN_K -> cooperative path
            (128, 128, 384),    // just under COOP_MIN_K -> blocked path
            (192, 64, 1152),    // M%64==0 but M%128!=0: bf16 coop fires, relaxed
                                // falls back — the two NN tiles differ and a
                                // gate that used the wrong one would corrupt here
            (256, 192, 768),    // both NN coop kernels fire; N not a tile multiple of 128
            (128, 192, 1152),   // relaxed coop at a real mlp K
            (256, 128, 640),    // K % 128 == 0 but K % 256 != 0. If either coop
                                // kernel's BKC were 256, its `k + BKC <= K` loop
                                // would silently drop the last 128 of K while the
                                // host gate still admitted the shape.
            (128, 64, 4096),    // long K, both coop paths, many blocks
        ];
        let val = |i: usize, salt: usize| -> f32 {
            (((i * 2654435761 + salt * 40503) % 1021) as f32 - 510.0) / 512.0
        };

        let mut checked = 0usize;
        for &(m, n, k) in shapes {
            let a_nn: Vec<f32> = (0..m * k).map(|i| val(i, 1)).collect();
            let b_nn: Vec<f32> = (0..k * n).map(|i| val(i, 2)).collect();
            // Reference in f64 from the NN orientation; TN/NT feed transposed
            // storage of the same logical matrices so the reference is shared.
            let mut refv = vec![0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f64;
                    for kk in 0..k {
                        acc += a_nn[i * k + kk] as f64 * b_nn[kk * n + j] as f64;
                    }
                    refv[i * n + j] = acc;
                }
            }
            let refmax = refv.iter().fold(0f64, |a, x| a.max(x.abs())).max(1e-30);

            // Storage variants of the same logical A/B.
            let a_km: Vec<f32> = (0..k * m).map(|idx| a_nn[(idx % m) * k + idx / m]).collect();
            let b_nk: Vec<f32> = (0..n * k).map(|idx| b_nn[(idx % k) * n + idx / k]).collect();

            let mk = |shape: &[usize], host: &[f32]| {
                let t = rt.alloc_tensor_f32(shape).unwrap();
                t.buffer.write_f32(host);
                t
            };

            // (label, needs_bf16, relaxed, run)
            for &(label, bf16, relaxed) in &[
                ("NN f32", false, false),
                ("NN relaxed", false, true),
                ("NN bf16", true, false),
                ("TN f32", false, false),
                ("TN bf16", true, false),
                ("NT f32", false, false),
                ("NT bf16", true, false),
                ("TNacc bf16", true, false),
                ("NTacc bf16", true, false),
            ] {
                rt.set_precision(if bf16 { PrecisionMode::Bf16 } else { PrecisionMode::F32 });
                rt.set_relaxed_precision(relaxed);

                let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
                let accum = label.contains("acc");
                let c0: Vec<f32> = if accum {
                    (0..m * n).map(|i| val(i, 9)).collect()
                } else {
                    vec![SENTINEL; m * n]
                };
                c.buffer.write_f32(&c0);

                let res = if label.starts_with("NN") {
                    let a = mk(&[m, k], &a_nn);
                    let b = mk(&[k, n], &b_nn);
                    if bf16 {
                        let (ab, bb) = (cast_f32_to_bf16(&a).unwrap(), cast_f32_to_bf16(&b).unwrap());
                        gemm(&ab, &bb, &c, GemmBackend::TensorOps)
                    } else {
                        gemm(&a, &b, &c, GemmBackend::TensorOps)
                    }
                } else if label.starts_with("TNacc") {
                    let (a, b) = (mk(&[k, m], &a_km), mk(&[k, n], &b_nn));
                    gemm_tn_accum_train(&a, &b, &c, GemmBackend::TensorOps)
                } else if label.starts_with("NTacc") {
                    let (a, b) = (mk(&[m, k], &a_nn), mk(&[n, k], &b_nk));
                    gemm_nt_accum_train(&a, &b, &c, GemmBackend::TensorOps)
                } else if label.starts_with("TN") {
                    let (a, b) = (mk(&[k, m], &a_km), mk(&[k, n], &b_nn));
                    if bf16 { gemm_tn_train(&a, &b, &c, GemmBackend::TensorOps) }
                    else { gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps) }
                } else {
                    let (a, b) = (mk(&[m, k], &a_nn), mk(&[n, k], &b_nk));
                    if bf16 { gemm_nt_train(&a, &b, &c, GemmBackend::TensorOps) }
                    else { gemm_nt_f32(&a, &b, &c, GemmBackend::TensorOps) }
                };
                if let Err(e) = res {
                    panic!("{label} {m}x{n}x{k} dispatch failed: {e}");
                }
                rt.synchronize().unwrap();
                let got = c.buffer.read_f32()[..m * n].to_vec();

                let tol = if bf16 { 6e-2 } else if relaxed { 1e-2 } else { 1e-4 };
                for i in 0..m * n {
                    let want = refv[i] + if accum { c0[i] as f64 } else { 0.0 };
                    assert!(
                        got[i].is_finite() && got[i] != SENTINEL,
                        "{label} {m}x{n}x{k}: C[{i}] = {} — tile never written",
                        got[i]
                    );
                    let rel = (got[i] as f64 - want).abs() / refmax;
                    assert!(
                        rel < tol,
                        "{label} {m}x{n}x{k}: C[{i}] rel err {rel:.3e} (got {}, want {want})",
                        got[i]
                    );
                }
                checked += 1;
            }
        }
        rt.set_precision(PrecisionMode::F32);
        rt.set_relaxed_precision(false);
        eprintln!("adversarial sweep: {checked} (path, shape) combinations verified");
    }

    /// Plain TN/NT GEMMs are single-shot `mode::multiply`, so the host skips
    /// `zero_f32(C)`. The tile ladder that justified this only ran exactly
    /// divisible shapes, so this deliberately uses ragged M/N to exercise the
    /// out-of-bounds slice path, with garbage in C. bf16 and exact-f32, TN and NT.
    #[test]
    fn tn_nt_gemm_overwrites_ragged_c_without_prezero() {
        use crate::runtime::PrecisionMode;
        let rt = GpuRuntime::new().expect("GpuRuntime");
        if !rt.has_tensorops() {
            eprintln!("skip: metallib has no TensorOps");
            return;
        }
        let (m, n, k) = (200usize, 100usize, 300usize);
        let ah: Vec<f32> = (0..k * m.max(n) * 2).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        let bh: Vec<f32> = (0..k * m.max(n) * 2).map(|i| ((i % 13) as f32 - 6.0) / 32.0).collect();

        for &bf16 in &[false, true] {
            rt.set_precision(if bf16 { PrecisionMode::Bf16 } else { PrecisionMode::F32 });
            for &tn in &[true, false] {
                let ash = if tn { vec![k, m] } else { vec![m, k] };
                let bsh = if tn { vec![k, n] } else { vec![n, k] };
                let a = rt.alloc_tensor_f32(&ash).unwrap();
                let b = rt.alloc_tensor_f32(&bsh).unwrap();
                a.buffer.write_f32(&ah[..ash[0] * ash[1]]);
                b.buffer.write_f32(&bh[..bsh[0] * bsh[1]]);

                let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
                c.buffer.write_f32(&vec![1.0e30f32; m * n]);
                match (tn, bf16) {
                    (true, true) => gemm_tn_train(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
                    (true, false) => gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
                    (false, true) => gemm_nt_train(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
                    (false, false) => gemm_nt_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
                }
                rt.synchronize().unwrap();
                let got = c.buffer.read_f32()[..m * n].to_vec();

                let mut worst = 0f64;
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0f64;
                        for kk in 0..k {
                            let av = if tn { ah[kk * m + i] } else { ah[i * k + kk] };
                            let bv = if tn { bh[kk * n + j] } else { bh[j * k + kk] };
                            acc += av as f64 * bv as f64;
                        }
                        assert!(got[i * n + j].is_finite(),
                            "{} {} C[{i},{j}] not finite — stale value survived a ragged tile",
                            if tn { "TN" } else { "NT" }, if bf16 { "bf16" } else { "f32" });
                        worst = worst.max((got[i * n + j] as f64 - acc).abs());
                    }
                }
                let tol = if bf16 { 1e-1 } else { 1e-2 };
                assert!(worst < tol, "{} {} ragged max abs err {worst:.3e} — C was not fully overwritten",
                    if tn { "TN" } else { "NT" }, if bf16 { "bf16" } else { "f32" });
            }
        }
        rt.set_precision(PrecisionMode::F32);
    }

    /// The NN bf16 GEMM skips the host `zero_f32(C)` pre-pass because its first
    /// K block uses `mode::multiply`. If that regresses to `multiply_accumulate`
    /// (or the kernel's SM/SN desyncs from TILE_BF16_NN, leaving tiles
    /// unwritten), stale C survives — so seed C with garbage and require an
    /// exact-overwrite result.
    #[test]
    fn bf16_gemm_nn_overwrites_c_without_prezero() {
        let rt = GpuRuntime::new().expect("GpuRuntime");
        if !rt.has_tensorops() {
            eprintln!("skip: metallib has no TensorOps");
            return;
        }
        // K > BK(256) so the blocked accumulate path runs; non-square so a
        // tiles_n/tiles_m mixup cannot alias into a passing result.
        let (m, n, k) = (256usize, 128usize, 512usize);
        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let ah: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        let bh: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) / 32.0).collect();
        a.buffer.write_f32(&ah);
        b.buffer.write_f32(&bh);
        let a_bf = cast_f32_to_bf16(&a).unwrap();
        let b_bf = cast_f32_to_bf16(&b).unwrap();

        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        c.buffer.write_f32(&vec![1.0e30f32; m * n]); // garbage a zero-pass would hide
        gemm(&a_bf, &b_bf, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32()[..m * n].to_vec();

        let round = |x: &f32| crate::tensor::bf16_bits_to_f32(crate::tensor::f32_to_bf16_bits(*x));
        let abf: Vec<f32> = ah.iter().map(round).collect();
        let bbf: Vec<f32> = bh.iter().map(round).collect();
        let mut worst = 0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += abf[i * k + kk] as f64 * bbf[kk * n + j] as f64;
                }
                assert!(got[i * n + j].is_finite(), "C[{i},{j}] not finite — stale value survived");
                worst = worst.max((got[i * n + j] as f64 - acc).abs());
            }
        }
        assert!(worst < 1e-2, "bf16 NN GEMM max abs err {worst:.3e} — C was not fully overwritten");
    }

    #[test]
    fn parity_metric_rejects_nonfinite_and_length_mismatch() {
        for (got, expected) in [
            (vec![f32::NAN], vec![0.0]),
            (vec![f32::INFINITY], vec![f32::INFINITY]),
            (vec![0.0], vec![0.0, 1.0]),
        ] {
            assert!(std::panic::catch_unwind(|| max_abs_err(&got, &expected)).is_err());
        }
    }

    fn run_case(m: usize, n: usize, k: usize, backend: GemmBackend) {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        eprintln!(
            "device={} encode=Metal4 tensorops={} backend={:?}",
            rt.device_name(),
            rt.has_tensorops(),
            backend
        );

        let mut a_host = vec![0.0f32; m * k];
        let mut b_host = vec![0.0f32; k * n];
        for i in 0..a_host.len() {
            a_host[i] = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for i in 0..b_host.len() {
            b_host[i] = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_host, &b_host, m, n, k);

        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_host);
        b.buffer.write_f32(&b_host);

        gemm_f32(&a, &b, &c, backend).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        let err = max_abs_err(&got, &expected);
        assert!(
            err < 1e-4,
            "GEMM {m}x{k}@{k}x{n} backend={backend:?} max_abs_err={err}"
        );
    }

    #[test]
    fn gemm_simdgroup_16() {
        run_case(16, 16, 16, GemmBackend::Simdgroup);
    }

    #[test]
    fn gemm_simdgroup_32() {
        run_case(32, 32, 32, GemmBackend::Simdgroup);
    }

    #[test]
    fn gemm_auto_small() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        let backend = select_backend(&rt);
        let dim = if backend == GemmBackend::TensorOps {
            32
        } else {
            16
        };
        run_case(dim, dim, dim, backend);
    }

    #[test]
    fn gemm_tensorops_32_if_available() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping TensorOps test: kernel not in metallib");
            return;
        }
        run_case(32, 32, 64, GemmBackend::TensorOps);
        run_case(64, 32, 32, GemmBackend::TensorOps);
    }

    #[test]
    fn gemm_bf16_tensorops_if_available() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping bf16 TensorOps test");
            return;
        }
        rt.set_precision(crate::runtime::PrecisionMode::Bf16);
        let m = 32usize;
        let n = 32usize;
        let k = 64usize;
        let mut a_f = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; k * n];
        for i in 0..a_f.len() {
            a_f[i] = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for i in 0..b_f.len() {
            b_f[i] = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);
        let a = rt.alloc_tensor_bf16(&[m, k]).unwrap();
        let b = rt.alloc_tensor_bf16(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer
            .write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&a_f));
        b.buffer
            .write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&b_f));
        gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        let err = max_abs_err(&got, &expected);
        // bf16 rounding — looser than f32
        assert!(err < 2e-2, "bf16 GEMM max_abs_err={err}");
    }

    /// Phase H: `gemm_train` under Bf16 casts f32 masters → bf16 TensorOps.
    #[test]
    fn gemm_train_bf16_casts_f32_operands() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping gemm_train bf16 test");
            return;
        }
        rt.set_precision(PrecisionMode::Bf16);
        let m = 32usize;
        let n = 32usize;
        let k = 64usize;
        let mut a_f = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; k * n];
        for i in 0..a_f.len() {
            a_f[i] = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for i in 0..b_f.len() {
            b_f[i] = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);
        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_f);
        b.buffer.write_f32(&b_f);
        gemm_train(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        let err = max_abs_err(&got, &expected);
        assert!(err < 2e-2, "gemm_train bf16 max_abs_err={err}");
    }

    /// Phase H bridge: `relaxed_precision` numerics vs exact f32 / CPU.
    /// Kept behind a flag for train; documents whether 1e-5 goldens survive.
    #[test]
    fn gemm_relaxed_precision_numerics() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping relaxed_precision test");
            return;
        }
        // Ensure pipeline exists (metallib rebuilt with Phase H kernel).
        if rt.pipeline("matmul2d_tensorops_f32_relaxed").is_err() {
            eprintln!("skipping: matmul2d_tensorops_f32_relaxed not in metallib");
            return;
        }
        let m = 64usize;
        let n = 64usize;
        let k = 128usize;
        let mut a_f = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; k * n];
        for i in 0..a_f.len() {
            a_f[i] = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for i in 0..b_f.len() {
            b_f[i] = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);

        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c_exact = rt.alloc_tensor_f32(&[m, n]).unwrap();
        let c_relax = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_f);
        b.buffer.write_f32(&b_f);

        rt.set_precision(PrecisionMode::F32);
        rt.set_relaxed_precision(false);
        gemm_f32(&a, &b, &c_exact, GemmBackend::TensorOps).unwrap();
        rt.set_relaxed_precision(true);
        gemm_f32(&a, &b, &c_relax, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();

        let got_exact = c_exact.buffer.read_f32();
        let got_relax = c_relax.buffer.read_f32();
        let err_exact = max_abs_err(&got_exact, &expected);
        let err_relax = max_abs_err(&got_relax, &expected);
        let err_vs_exact = max_abs_err(&got_relax, &got_exact);
        eprintln!(
            "relaxed_precision: err_vs_cpu_exact={err_exact:.3e} err_vs_cpu_relax={err_relax:.3e} \
             err_relax_vs_exact={err_vs_exact:.3e}"
        );
        assert!(err_exact < 1e-4, "exact f32 GEMM drifted: {err_exact}");
        // Smoke: relaxed must be finite and within a generous bound (tf32-class).
        assert!(
            err_relax < 5e-2,
            "relaxed GEMM too far from CPU: {err_relax}"
        );
        // Document 1e-5 golden gate: if this fails, keep --tf32 off for parity.
        if err_relax >= 1e-5 {
            eprintln!(
                "NOTE: relaxed_precision breaks 1e-5 golden atol (err={err_relax:.3e}); \
                 leave flag off for f32 parity / enable only for throughput experiments"
            );
        } else {
            eprintln!("relaxed_precision within 1e-5 of CPU on this shape");
        }
        rt.set_relaxed_precision(false);
    }

    #[test]
    fn gemm_train_bf16_awkward_k() {
        // sota shapes: bigram_dim=48, ve_dim=24 — must not NaN under bf16 TensorOps.
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping awkward-K bf16 test");
            return;
        }
        rt.set_precision(PrecisionMode::Bf16);
        for (m, n, k) in [(64usize, 128usize, 48usize), (64, 128, 24), (4096, 128, 48)] {
            let mut a_f = vec![0.0f32; m * k];
            let mut b_f = vec![0.0f32; k * n];
            for i in 0..a_f.len() {
                a_f[i] = ((i % 17) as f32) * 0.01 - 0.08;
            }
            for i in 0..b_f.len() {
                b_f[i] = ((i % 13) as f32) * 0.007 - 0.04;
            }
            let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);
            let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
            let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a.buffer.write_f32(&a_f);
            b.buffer.write_f32(&b_f);
            gemm_train(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            let n_bad = got.iter().filter(|x| !x.is_finite()).count();
            let err = max_abs_err(&got, &expected);
            eprintln!("bf16 awkward {m}x{k}@{k}x{n}: nonfinite={n_bad} err={err:.3e}");
            assert_eq!(n_bad, 0, "NaN/Inf in bf16 GEMM {m}x{k}@{k}x{n}");
            assert!(err < 5e-2, "bf16 awkward K err={err}");
        }
    }

    #[test]
    fn gemm_tn_nt_bf16_train_smoke() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping tn/nt bf16 smoke");
            return;
        }
        if rt.pipeline("matmul2d_tensorops_tn_bf16_f32").is_err() {
            eprintln!("skipping: tn/nt bf16 kernels missing");
            return;
        }
        rt.set_precision(PrecisionMode::Bf16);
        let m = 32usize;
        let n = 32usize;
        let k = 64usize;
        // TN: A[K,M], B[K,N] → C[M,N]
        let mut a_km = vec![0.0f32; k * m];
        let mut b_kn = vec![0.0f32; k * n];
        for i in 0..a_km.len() {
            a_km[i] = ((i % 11) as f32) * 0.05 - 0.2;
        }
        for i in 0..b_kn.len() {
            b_kn[i] = ((i % 7) as f32) * 0.04 - 0.1;
        }
        // CPU: C = A^T @ B
        let mut a_mk = vec![0.0f32; m * k];
        for i in 0..k {
            for j in 0..m {
                a_mk[j * k + i] = a_km[i * m + j];
            }
        }
        let exp_tn = gemm_f32_cpu(&a_mk, &b_kn, m, n, k);
        let a_t = rt.alloc_tensor_f32(&[k, m]).unwrap();
        let b_t = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c_tn = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a_t.buffer.write_f32(&a_km);
        b_t.buffer.write_f32(&b_kn);
        gemm_tn_train(&a_t, &b_t, &c_tn, GemmBackend::TensorOps).unwrap();

        // NT: A[M,K], B[N,K] → C[M,N]
        let mut b_nk = vec![0.0f32; n * k];
        for i in 0..n {
            for j in 0..k {
                b_nk[i * k + j] = b_kn[j * n + i];
            }
        }
        let mut b_kn_from_nk = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                b_kn_from_nk[j * n + i] = b_nk[i * k + j];
            }
        }
        let exp_nt = gemm_f32_cpu(&a_mk, &b_kn_from_nk, m, n, k);
        let a_n = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b_n = rt.alloc_tensor_f32(&[n, k]).unwrap();
        let c_nt = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a_n.buffer.write_f32(&a_mk);
        b_n.buffer.write_f32(&b_nk);
        gemm_nt_train(&a_n, &b_n, &c_nt, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();

        let err_tn = max_abs_err(&c_tn.buffer.read_f32(), &exp_tn);
        let err_nt = max_abs_err(&c_nt.buffer.read_f32(), &exp_nt);
        assert!(err_tn < 2e-2, "tn bf16 err={err_tn}");
        assert!(err_nt < 2e-2, "nt bf16 err={err_nt}");
    }

    fn gemm_tn_cpu(a_km: &[f32], b_kn: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut a_mk = vec![0.0f32; m * k];
        for i in 0..k {
            for j in 0..m {
                a_mk[j * k + i] = a_km[i * m + j];
            }
        }
        gemm_f32_cpu(&a_mk, b_kn, m, n, k)
    }

    fn gemm_nt_cpu(a_mk: &[f32], b_nk: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut b_kn = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                b_kn[j * n + i] = b_nk[i * k + j];
            }
        }
        gemm_f32_cpu(a_mk, &b_kn, m, n, k)
    }

    #[test]
    fn gemm_tn_nt_tensorops_descriptors() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping tn/nt descriptor test");
            return;
        }
        for (m, n, k) in [(32usize, 32, 64), (64, 128, 128), (128, 128, 256)] {
            let mut a_km = vec![0.0f32; k * m];
            let mut b_kn = vec![0.0f32; k * n];
            for i in 0..a_km.len() {
                a_km[i] = ((i % 11) as f32) * 0.05 - 0.2;
            }
            for i in 0..b_kn.len() {
                b_kn[i] = ((i % 7) as f32) * 0.04 - 0.1;
            }
            let exp = gemm_tn_cpu(&a_km, &b_kn, m, n, k);
            let a = rt.alloc_tensor_f32(&[k, m]).unwrap();
            let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a.buffer.write_f32(&a_km);
            b.buffer.write_f32(&b_kn);
            gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let err = max_abs_err(&c.buffer.read_f32(), &exp);
            assert!(err < 1e-4, "TN desc {m}x{k}^T@{k}x{n} err={err}");

            let mut a_mk = vec![0.0f32; m * k];
            let mut b_nk = vec![0.0f32; n * k];
            for i in 0..m {
                for j in 0..k {
                    a_mk[i * k + j] = ((i * k + j) % 13) as f32 * 0.03 - 0.15;
                }
            }
            for i in 0..n {
                for j in 0..k {
                    b_nk[i * k + j] = ((i * k + j) % 17) as f32 * 0.02 - 0.1;
                }
            }
            let exp_nt = gemm_nt_cpu(&a_mk, &b_nk, m, n, k);
            let a2 = rt.alloc_tensor_f32(&[m, k]).unwrap();
            let b2 = rt.alloc_tensor_f32(&[n, k]).unwrap();
            let c2 = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a2.buffer.write_f32(&a_mk);
            b2.buffer.write_f32(&b_nk);
            gemm_nt_f32(&a2, &b2, &c2, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let err_nt = max_abs_err(&c2.buffer.read_f32(), &exp_nt);
            assert!(err_nt < 1e-4, "NT desc {m}x{k}@{n}x{k}^T err={err_nt}");
        }
    }

    #[test]
    fn gemm_tn_splitk_tall_dw_shape() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping split-K test");
            return;
        }
        // dW-shaped: M=N=128, K=4096 (BT).
        let m = 128usize;
        let n = 128usize;
        let k = 4096usize;
        let mut a_km = vec![0.0f32; k * m];
        let mut b_kn = vec![0.0f32; k * n];
        for i in 0..a_km.len() {
            a_km[i] = ((i % 19) as f32) * 0.01 - 0.08;
        }
        for i in 0..b_kn.len() {
            b_kn[i] = ((i % 23) as f32) * 0.008 - 0.05;
        }
        let exp = gemm_tn_cpu(&a_km, &b_kn, m, n, k);
        let a = rt.alloc_tensor_f32(&[k, m]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_km);
        b.buffer.write_f32(&b_kn);
        assert!(prefer_tn_splitk(m, n, k));
        gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let err = max_abs_err(&c.buffer.read_f32(), &exp);
        assert!(err < 1e-3, "split-K TN dW shape err={err}");
    }

    #[test]
    fn gemm_tn_splitk_mlp_dw_shape() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        if !rt.has_tensorops() {
            eprintln!("skipping MLP split-K test");
            return;
        }
        // MLP-up dW: M=128, N=384, K=4096
        let m = 128usize;
        let n = 384usize;
        let k = 4096usize;
        assert!(prefer_tn_splitk(m, n, k));
        let mut a_km = vec![0.0f32; k * m];
        let mut b_kn = vec![0.0f32; k * n];
        for i in 0..a_km.len() {
            a_km[i] = ((i % 19) as f32) * 0.01 - 0.08;
        }
        for i in 0..b_kn.len() {
            b_kn[i] = ((i % 23) as f32) * 0.008 - 0.05;
        }
        let exp = gemm_tn_cpu(&a_km, &b_kn, m, n, k);
        let a = rt.alloc_tensor_f32(&[k, m]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_km);
        b.buffer.write_f32(&b_kn);
        gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let err = max_abs_err(&c.buffer.read_f32(), &exp);
        assert!(err < 1e-3, "MLP-up split-K TN err={err}");
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn stress_seeded_shapes_precision_transposes_and_guards() {
        let rt = GpuRuntime::new().unwrap();
        assert!(rt.has_tensorops(), "stress requires TensorOps, no capability skip");
        rt.set_async_encode(true).unwrap();
        let edges = [1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 65];
        let mut seed = 0x42ab_19d3u32;
        let mut random = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed % 17) as f32 - 8.0) / 32.0
        };
        let mut cases = 0;
        for case in 0..24 {
            let (m, n, k) = (edges[case % 12], edges[(case * 5 + 3) % 12],
                if case < 12 { edges[(case * 7 + 1) % 12] } else { 257 });
            for backend in [GemmBackend::Simdgroup, GemmBackend::TensorOps] {
                for precision in [PrecisionMode::F32, PrecisionMode::Bf16] {
                    rt.set_precision(precision);
                    for transpose in 0..3 {
                        let ashape = if transpose == 1 { [k, m] } else { [m, k] };
                        let bshape = if transpose == 2 { [n, k] } else { [k, n] };
                        let av: Vec<f32> = (0..m*k).map(|_| random()).collect();
                        let bv: Vec<f32> = (0..k*n).map(|_| random()).collect();
                        let a = rt.alloc_tensor_f32(&ashape).unwrap();
                        let b = rt.alloc_tensor_f32(&bshape).unwrap();
                        let bank = rt.alloc_tensor_f32(&[m*n+8]).unwrap();
                        a.buffer.write_f32(&av); b.buffer.write_f32(&bv);
                        let mut poison = vec![f32::NAN; m*n+8];
                        poison[..4].fill(-777.0); poison[m*n+4..].fill(777.0);
                        bank.buffer.write_f32(&poison);
                        let c = bank.view(&[m,n],4);
                        let launch = match transpose { 0 => gemm_train, 1 => gemm_tn_train,
                            _ => gemm_nt_train };
                        launch(&a,&b,&c,backend).unwrap();
                        let got = bank.buffer.read_f32();
                        assert_eq!(&got[..4], &[-777.0;4]);
                        assert_eq!(&got[m*n+4..], &[777.0;4]);
                        for row in 0..m { for col in 0..n {
                            let expected: f64 = (0..k).map(|p| {
                                f64::from(av[if transpose==1 {p*m+row} else {row*k+p}]) *
                                f64::from(bv[if transpose==2 {col*k+p} else {p*n+col}])
                            }).sum();
                            let actual = got[4+row*n+col];
                            assert!(actual.is_finite() && (f64::from(actual)-expected).abs()<1e-4,
                                "case={case} shape={m}x{n}x{k} transpose={transpose} {backend:?} {precision:?}: {actual} vs {expected}");
                        }}
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 288);
        eprintln!("STRESS_GEMM cases={cases} seed=0x42ab19d3");
    }

    type Launch = fn(&Tensor, &Tensor, &Tensor, GemmBackend) -> Result<(), String>;
    const LAUNCHES: &[Launch] = &[
        gemm,
        gemm_train,
        gemm_tn_f32,
        gemm_nt_f32,
        gemm_tn_train,
        gemm_nt_train,
        gemm_tn_accum_train,
        gemm_nt_accum_train,
    ];

    /// Isolate the removed clear with alternating paired measurements on one build.
    #[test]
    #[ignore = "manual wall-clock benchmark; run alone on an idle GPU"]
    fn benchmark_simdgroup_zero_cost() {
        use std::time::Instant;
        let rt = GpuRuntime::new().unwrap();
        for d in [16, 128, 512] {
            let a = rt.alloc_tensor_f32(&[d, d]).unwrap();
            let b = rt.alloc_tensor_f32(&[d, d]).unwrap();
            let c = rt.alloc_tensor_f32(&[d, d]).unwrap();
            a.buffer.write_f32(&vec![0.125; d * d]);
            b.buffer.write_f32(&vec![0.125; d * d]);
            let zero = rt.pipeline("zero_f32").unwrap();
            let run = |clear: bool| {
                if clear {
                    crate::dispatch::dispatch_1d(&rt, &zero, d * d, |bnd| {
                        crate::dispatch::set_tensor(bnd, &c, 0);
                        crate::dispatch::set_u32(bnd, (d * d) as u32, 1);
                    })
                    .unwrap();
                }
                gemm(&a, &b, &c, GemmBackend::Simdgroup).unwrap();
                rt.synchronize().unwrap();
            };
            for _ in 0..10 {
                run(true);
                run(false);
            }
            let mut before = Vec::new();
            let mut after = Vec::new();
            for i in 0..100 {
                for clear in if i % 2 == 0 {
                    [true, false]
                } else {
                    [false, true]
                } {
                    let start = Instant::now();
                    run(clear);
                    let us = start.elapsed().as_secs_f64() * 1e6;
                    if clear {
                        before.push(us)
                    } else {
                        after.push(us)
                    };
                }
            }
            assert!(c.buffer.read_f32().iter().all(|&x| x == d as f32 / 64.0));
            before.sort_by(f64::total_cmp);
            after.sort_by(f64::total_cmp);
            let old = (before[49] + before[50]) * 0.5;
            let new = (after[49] + after[50]) * 0.5;
            eprintln!("SIMDGROUP_AB d={d} samples=100 clear_us={old:.3} overwrite_us={new:.3} speedup={:.3}",old/new);
        }
    }

    #[test]
    fn rejects_invalid_metadata_and_mixed_runtimes() {
        let rt = GpuRuntime::new().unwrap();
        let other = GpuRuntime::new().unwrap();
        let a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let c = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let foreign = other.alloc_tensor_f32(&[16, 16]).unwrap();
        let mut cases = Vec::new();
        for (shape, offset, dtype) in [
            (vec![0, 16], 0, DType::F32),
            (vec![16, 16], 1, DType::F32),
            (vec![16, 16], 4, DType::F32),
            (vec![usize::MAX, usize::MAX], 0, DType::F32),
            (vec![16, 16], 0, DType::BF16),
            (vec![16, 15], 0, DType::F32),
        ] {
            let mut bad = c.clone();
            bad.shape = shape;
            bad.byte_offset = offset;
            bad.dtype = dtype;
            cases.push(bad);
        }
        for precision in [PrecisionMode::F32, PrecisionMode::Bf16] {
            rt.set_precision(precision);
            for launch in LAUNCHES {
                for bad in &cases {
                    assert!(launch(&a, &b, bad, GemmBackend::TensorOps).is_err());
                }
                assert!(launch(&a, &foreign, &c, GemmBackend::TensorOps).is_err());
                // A mismatched inner dimension must fail before any bf16 cast.
                let bad_b = b.view(&[16, 15], 0);
                assert!(launch(&a, &bad_b, &c, GemmBackend::TensorOps).is_err());
            }
        }
        assert_eq!(rt.take_dispatch_count(), 0);
    }

    #[test]
    fn disjoint_bank_views_work_and_overlap_is_rejected() {
        let rt = GpuRuntime::new().unwrap();
        let bank = rt.alloc_tensor_f32(&[3 * 256]).unwrap();
        bank.buffer.write_f32(&vec![1.0; 3 * 256]);
        let a = bank.view(&[16, 16], 0);
        let b = bank.view(&[16, 16], 256);
        let c = bank.view(&[16, 16], 512);
        for launch in LAUNCHES {
            let overlap = bank.view(&[16, 16], 128);
            assert!(launch(&a, &b, &overlap, GemmBackend::TensorOps).is_err());
        }
        rt.take_dispatch_count();
        gemm(&a, &b, &c, GemmBackend::Simdgroup).unwrap();
        rt.synchronize().unwrap();
        assert_eq!(rt.take_dispatch_count(), 1, "simdgroup must not pre-zero C");
        let got = bank.buffer.read_f32();
        assert!(got[..512].iter().all(|&x| x == 1.0));
        assert!(got[512..].iter().all(|&x| x == 16.0));
    }

    #[test]
    fn transpose_edges_precision_and_accumulation() {
        let rt = GpuRuntime::new().unwrap();
        assert!(
            rt.has_tensorops(),
            "TensorOps coverage requires the actual metallib"
        );
        for (m, n, k) in [(1, 3, 1), (17, 31, 9), (33, 65, 129), (17, 31, 2049)] {
            for backend in [GemmBackend::Simdgroup, GemmBackend::TensorOps] {
                for precision in [PrecisionMode::F32, PrecisionMode::Bf16] {
                    rt.set_precision(precision);
                    for (tn, accum) in [(true, false), (false, false), (true, true), (false, true)]
                    {
                        let ashape = if tn { [k, m] } else { [m, k] };
                        let bshape = if tn { [k, n] } else { [n, k] };
                        let av: Vec<f32> =
                            (0..m * k).map(|i| (i % 13) as f32 / 16.0 - 0.25).collect();
                        let bv: Vec<f32> =
                            (0..n * k).map(|i| (i % 7) as f32 / 16.0 - 0.125).collect();
                        let a = rt.alloc_tensor_f32(&ashape).unwrap();
                        let b = rt.alloc_tensor_f32(&bshape).unwrap();
                        let bank = rt.alloc_tensor_f32(&[m * n + 8]).unwrap();
                        a.buffer.write_f32(&av);
                        b.buffer.write_f32(&bv);
                        bank.buffer.write_f32(&vec![2.0; m * n + 8]);
                        let c = bank.view(&[m, n], 4);
                        let launch: Launch = match (tn, accum) {
                            (true, false) => gemm_tn_train,
                            (false, false) => gemm_nt_train,
                            (true, true) => gemm_tn_accum_train,
                            (false, true) => gemm_nt_accum_train,
                        };
                        launch(&a, &b, &c, backend).unwrap();
                        rt.synchronize().unwrap();
                        let got = bank.buffer.read_f32();
                        assert_eq!(&got[..4], &[2.0; 4]);
                        assert_eq!(&got[m * n + 4..], &[2.0; 4]);
                        for row in 0..m {
                            for col in 0..n {
                                let mut expected = if accum { 2.0 } else { 0.0 };
                                for p in 0..k {
                                    expected += av[if tn { p * m + row } else { row * k + p }]
                                        * bv[if tn { p * n + col } else { col * k + p }];
                                }
                                let x = got[4 + row * n + col];
                                assert!(x.is_finite() && (x-expected).abs()<1e-4,
                                "{m}x{n}x{k} {backend:?} {precision:?} TN={tn} accum={accum}: {x} vs {expected}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn casts_reject_dtype_and_shape_before_encoding() {
        let rt = GpuRuntime::new().unwrap();
        let a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = rt.alloc_tensor_bf16(&[256]).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cast_f32_to_bf16_into(&a, &b)
        }));
        assert!(result.is_ok(), "cast Result API panicked");
        assert!(result.unwrap().is_err());
        assert!(cast_bf16_to_f32(&a).is_err());
        assert!(cast_f32_to_bf16(&b).is_err());
        assert_eq!(rt.take_dispatch_count(), 0);
    }

    #[test]
    fn rejects_bad_rank_without_panicking_or_encoding() {
        let rt = GpuRuntime::new().unwrap();
        let a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = a.deep_copy().unwrap();
        let c = a.deep_copy().unwrap();
        rt.synchronize().unwrap();
        let bad = a.view(&[256], 0);
        for launch in LAUNCHES {
            rt.take_dispatch_count();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                launch(&bad, &b, &c, GemmBackend::TensorOps)
            }));
            assert!(result.is_ok(), "public Result API panicked");
            assert!(result.unwrap().is_err(), "invalid rank accepted");
            assert_eq!(rt.take_dispatch_count(), 0);
        }
    }

    #[test]
    fn rejects_output_alias_before_encoding() {
        let rt = GpuRuntime::new().unwrap();
        let a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        for launch in LAUNCHES {
            rt.take_dispatch_count();
            assert!(launch(&a, &b, &a, GemmBackend::TensorOps).is_err());
            assert_eq!(rt.take_dispatch_count(), 0);
        }
    }

    #[test]
    fn rejects_wrong_dtype_on_transpose_paths() {
        let rt = GpuRuntime::new().unwrap();
        let mut a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let c = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        a.dtype = DType::BF16; // backing allocation remains large enough for old buggy path
        assert!(gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).is_err());
        assert!(gemm_nt_f32(&a, &b, &c, GemmBackend::TensorOps).is_err());
    }

    #[test]
    fn simdgroup_edges_and_offset_guards() {
        let rt = GpuRuntime::new().unwrap();
        for (m, n, k) in [
            (1, 1, 1),
            (7, 9, 3),
            (16, 16, 16),
            (17, 31, 9),
            (33, 65, 129),
        ] {
            let av: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 / 16.0 - 0.25).collect();
            let bv: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 / 16.0 - 0.125).collect();
            let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
            let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
            let bank = rt.alloc_tensor_f32(&[m * n + 8]).unwrap();
            a.buffer.write_f32(&av);
            b.buffer.write_f32(&bv);
            let mut poisoned = vec![f32::NAN; m * n + 8];
            poisoned[..4].fill(123.0);
            poisoned[m * n + 4..].fill(123.0);
            bank.buffer.write_f32(&poisoned);
            let c = bank.view(&[m, n], 4);
            gemm(&a, &b, &c, GemmBackend::Simdgroup).unwrap();
            rt.synchronize().unwrap();
            let got = bank.buffer.read_f32();
            assert_eq!(&got[..4], &[123.0; 4]);
            assert_eq!(&got[m * n + 4..], &[123.0; 4]);
            let expected = gemm_f32_cpu(&av, &bv, m, n, k);
            for (x, y) in got[4..m * n + 4].iter().zip(expected) {
                assert!(
                    x.is_finite() && (x - y).abs() < 1e-4,
                    "{m}x{n}x{k}: {x} vs {y}"
                );
            }
        }
    }
}
