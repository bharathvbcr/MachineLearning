//! A/B the bf16 NN GEMM tile geometry against the production kernel.
//!
//! Tests two suspects behind the ~2× gap vs PyTorch MPS bf16: output-tile
//! arithmetic intensity, and the zero_f32(C) pre-pass that only exists because
//! the production kernel accumulates on the first K block. Every variant is
//! checked against the production kernel's output before its time is reported.

use tessl_arch02::gemm::{cast_f32_to_bf16, gemm, gemm_nt_accum_train, gemm_nt_train, gemm_tn_accum_train, gemm_tn_train, GemmBackend};
use tessl_arch02::runtime::{mtl_size, GpuRuntime, PrecisionMode};
use tessl_arch02::tensor::Tensor;
use objc2_metal::MTLComputePipelineState;
use std::time::Instant;

struct Variant {
    kernel: &'static str,
    sm: usize,
    sn: usize,
    bk: usize,
    nsg: usize,
    /// Mirrors production: first K block accumulates, so C must be pre-zeroed.
    needs_zero: bool,
}

const VARIANTS: &[Variant] = &[
    Variant { kernel: "mm_bf16_64x64_bk256_sg4",      sm: 64, sn: 64, bk: 256, nsg: 4, needs_zero: false },
    Variant { kernel: "mm_coop_64x64_bk64_sg4",       sm: 64, sn: 64, bk: 64,  nsg: 4, needs_zero: false },
    Variant { kernel: "mm_coop_64x64_bk128_sg4",      sm: 64, sn: 64, bk: 128, nsg: 4, needs_zero: false },
    // The *production* coop kernel, driven by the rig's dispatch. If this beats
    // production, the difference is the dispatch path, not the kernel.
    Variant { kernel: "matmul2d_tensorops_bf16_f32_coop", sm: 64, sn: 64, bk: 128, nsg: 4, needs_zero: false },
    Variant { kernel: "mm_coop_64x64_bk256_sg4",      sm: 64, sn: 64, bk: 256, nsg: 4, needs_zero: false },
    Variant { kernel: "mm_coop_128x64_bk128_sg8",     sm: 128, sn: 64, bk: 128, nsg: 8, needs_zero: false },
    Variant { kernel: "mm_coop_128x128_bk64_sg8",     sm: 128, sn: 128, bk: 64, nsg: 8, needs_zero: false },
]; 

const F32R_VARIANTS: &[Variant] = &[
    Variant { kernel: "mm_f32r_64x32_bk128_sg4_accf", sm: 64, sn: 32, bk: 128, nsg: 4, needs_zero: true },
    Variant { kernel: "mm_f32r_128x64_bk256_sg4",     sm: 128, sn: 64, bk: 256, nsg: 4, needs_zero: false },
    Variant { kernel: "mm_f32r_128x64_bk256_sg8",     sm: 128, sn: 64, bk: 256, nsg: 8, needs_zero: false },
    Variant { kernel: "mm_f32r_128x32_bk256_sg4",     sm: 128, sn: 32, bk: 256, nsg: 4, needs_zero: false },
    Variant { kernel: "mm_f32r_256x64_bk256_sg8",     sm: 256, sn: 64, bk: 256, nsg: 8, needs_zero: false },
    Variant { kernel: "mm_f32rcoop_128x64_bk64_sg4",   sm: 128, sn: 64, bk: 64,  nsg: 4, needs_zero: false },
    Variant { kernel: "mm_f32rcoop_128x64_bk128_sg4",  sm: 128, sn: 64, bk: 128, nsg: 4, needs_zero: false },
    Variant { kernel: "mm_f32rcoop_128x64_bk256_sg4",  sm: 128, sn: 64, bk: 256, nsg: 4, needs_zero: false },
    Variant { kernel: "mm_f32rcoop_64x64_bk128_sg4",   sm: 64,  sn: 64, bk: 128, nsg: 4, needs_zero: false },
    Variant { kernel: "mm_f32rcoop_128x128_bk128_sg4", sm: 128, sn: 128, bk: 128, nsg: 4, needs_zero: false },
    Variant { kernel: "mm_f32rcoop_128x64_bk128_sg8",  sm: 128, sn: 64, bk: 128, nsg: 8, needs_zero: false },
];

/// (SM, SN, NSG, kernel-prefix-suffix) ladder shared by the TN and NT lanes.
const TNNT_TILES: &[(usize, usize, usize, &str)] = &[
    (64, 32, 4, "64x32_sg4"),
    (64, 64, 4, "64x64_sg4"),
    (64, 64, 8, "64x64_sg8"),
    (128, 64, 4, "128x64_sg4"),
    (128, 64, 8, "128x64_sg8"),
    (32, 32, 4, "32x32_sg4"),
];

/// K-blocked TN/NT candidates, named explicitly because they vary in BK as well
/// as tile. `_blk` writes each K block into device-memory C (control); `_coop`
/// holds the accumulator in registers and stores once. Production issues a
/// single full-K `matmul2d`, so `_blk` vs `_coop` separates "K-blocking helps"
/// from "the register accumulator helps".
/// (kernel, SM, SN, BK, NSG)
const TN_KBLK: &[(&str, usize, usize, usize, usize)] = &[
    ("mm_tnblk_128x64_bk128_sg4",   128,  64, 128, 4),
    ("mm_tncoop_128x64_bk64_sg4",   128,  64,  64, 4),
    ("mm_tncoop_128x64_bk128_sg4",  128,  64, 128, 4),
    ("mm_tncoop_128x64_bk256_sg4",  128,  64, 256, 4),
    ("mm_tncoop_64x64_bk128_sg4",    64,  64, 128, 4),
    ("mm_tncoop_128x128_bk128_sg4", 128, 128, 128, 4),
    ("mm_tncoop_128x64_bk128_sg8",  128,  64, 128, 8),
];
const NT_KBLK: &[(&str, usize, usize, usize, usize)] = &[
    ("mm_ntblk_128x64_bk128_sg4",   128,  64, 128, 4),
    ("mm_ntcoop_128x64_bk64_sg4",   128,  64,  64, 4),
    ("mm_ntcoop_128x64_bk128_sg4",  128,  64, 128, 4),
    ("mm_ntcoop_128x64_bk256_sg4",  128,  64, 256, 4),
    ("mm_ntcoop_64x64_bk128_sg4",    64,  64, 128, 4),
    ("mm_ntcoop_128x128_bk128_sg4", 128, 128, 128, 4),
    ("mm_ntcoop_128x64_bk128_sg8",  128,  64, 128, 8),
];
/// Accumulating variants load C into the register accumulator, run the whole K
/// loop there, and store once — same C traffic as production, zero per-block.
const TNACC_KBLK: &[(&str, usize, usize, usize, usize)] = &[
    ("mm_tnacccoop_128x64_bk128_sg4", 128, 64, 128, 4),
    ("mm_tnacccoop_128x64_bk256_sg4", 128, 64, 256, 4),
    ("mm_tnacccoop_64x64_bk128_sg4",   64, 64, 128, 4),
];
const NTACC_KBLK: &[(&str, usize, usize, usize, usize)] = &[
    ("mm_ntacccoop_64x64_bk128_sg4",   64, 64, 128, 4),
    ("mm_ntacccoop_64x64_bk256_sg4",   64, 64, 256, 4),
    ("mm_ntacccoop_128x64_bk128_sg4", 128, 64, 128, 4),
];

/// Real backward-pass gradient shapes. BT = batch*seq_len = 16*256 = 4096 for
/// every arch_02 preset; d/mlp come from sota_toy(128/384), medium_16m(384/1152)
/// and arch02_128m(768/2304). TN is dW = X^T @ dY; NT is dX = dY @ W^T.
const TN_SHAPES: &[(usize, usize, usize, &str)] = &[
    (768, 768, 4096, "dW_attn_768"),
    (768, 2304, 4096, "dW_mlp_up_768"),
    (2304, 768, 4096, "dW_mlp_down_768"),
    (384, 1152, 4096, "dW_mlp_up_384"),
    (1024, 768, 4096, "dW_vocab_768"),
];
const NT_SHAPES: &[(usize, usize, usize, &str)] = &[
    (4096, 768, 768, "dX_attn_768"),
    (4096, 2304, 768, "dX_mlp_down_768"),
    (4096, 768, 2304, "dX_mlp_up_768"),
    (4096, 384, 384, "dX_attn_384"),
    (4096, 1152, 384, "dX_mlp_down_384"),
];

/// Real forward-pass NN shapes. BT = 16*256 = 4096 for every preset; the three
/// (d, mlp) pairs are sota_toy(128,384), medium_16m(384,1152), arch02_128m(768,2304).
/// The narrow ones (N=128) are the risk case for a 64-wide tile and were absent
/// from the original 8-shape sweep the constants were first chosen on.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (4096, 768, 768, "fwd_attn_768"),
    (4096, 2304, 768, "fwd_mlp_up_768"),
    (4096, 768, 2304, "fwd_mlp_down_768"),
    (4096, 1024, 768, "fwd_logits_768"),
    (4096, 384, 384, "fwd_attn_384"),
    (4096, 1152, 384, "fwd_mlp_up_384"),
    (4096, 384, 1152, "fwd_mlp_down_384"),
    (4096, 128, 128, "fwd_attn_128"),
    (4096, 384, 128, "fwd_mlp_up_128"),
    (4096, 128, 384, "fwd_mlp_down_128"),
];

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((((s >> 32) as u32) as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32
    }).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn run_variant(
    rt: &std::sync::Arc<GpuRuntime>,
    v: &Variant,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    let p = rt.pipeline(v.kernel)?;
    let zero_p = rt.pipeline("zero_f32")?;
    let tiles_n = n / v.sn;
    let tg = tiles_n * (m / v.sm);
    let tpt = p.threadExecutionWidth() as usize * v.nsg;
    let numel = c.numel();
    let z_tpt = (zero_p.threadExecutionWidth() as usize).min(numel).max(1);
    let z_groups = (numel + z_tpt - 1) / z_tpt;
    let needs_zero = v.needs_zero;
    rt.with_binder(|bnd| {
        if needs_zero {
            bnd.set_pipeline(&zero_p);
            bnd.bind_tensor(c, 0);
            bnd.bind_u32(numel as u32, 1);
            bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
            bnd.barrier();
        }
        bnd.set_pipeline(&p);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32((m / v.sm) as u32, 7);
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}


/// The variant kernels live in `matmul_tensorops_tune.metal`, which is only
/// linked when `METAL_NATIVE_GEMM_TUNE=1` was set at build time. Without it
/// every candidate would report `skip(pipe)` and the run would exit 0 having
/// measured nothing — so check once, up front, and say exactly what to do.
fn require_tune_kernels(rt: &std::sync::Arc<GpuRuntime>, probe: &str) {
    if rt.pipeline(probe).is_err() {
        eprintln!(
            "error: tuning kernel `{probe}` is not in the metallib.\n\
             The GEMM A/B rig is opt-in so it stays out of the shipped metallib.\n\
             Rebuild with it:\n\
             \n    METAL_NATIVE_GEMM_TUNE=1 cargo build --release --bins\n"
        );
        std::process::exit(2);
    }
}

fn main() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    require_tune_kernels(&rt, "mm_bf16_64x64_bk256_sg4");
    let warmup: usize = std::env::var("BENCH_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(30);

    println!("{:<32}{:>12}{:>14}{:>16}", "kernel", "maxTPTG", "tgMem(B)", "requested TPTG");
    for v in VARIANTS {
        match rt.pipeline(v.kernel) {
            Ok(p) => {
                let w = p.threadExecutionWidth() as usize;
                println!("{:<32}{:>12}{:>14}{:>16}", v.kernel,
                         p.maxTotalThreadsPerThreadgroup(),
                         p.staticThreadgroupMemoryLength(),
                         w * v.nsg);
            }
            Err(e) => println!("{:<32}  pipeline error: {e}", v.kernel),
        }
    }

    let env_shapes: Option<Vec<(usize, usize, usize, String)>> =
        std::env::var("BENCH_SHAPES").ok().map(|raw| {
            raw.split(',').filter(|s| !s.trim().is_empty()).map(|spec| {
                let d: Vec<usize> = spec.trim().split('x').map(|v| v.parse().unwrap()).collect();
                (d[0], d[1], d[2], format!("{}x{}x{}", d[0], d[1], d[2]))
            }).collect()
        });
    let nn_shapes: Vec<(usize, usize, usize, String)> = match env_shapes {
        Some(v) => v,
        None => SHAPES.iter().map(|&(m, n, k, l)| (m, n, k, l.to_string())).collect(),
    };

    for (m, n, k, label) in nn_shapes.iter().map(|(m, n, k, l)| (*m, *n, *k, l.as_str())) {
        let a = rt.alloc_tensor_f32(&[m, k])?;
        let b = rt.alloc_tensor_f32(&[k, n])?;
        a.buffer.write_f32(&fill(m * k, 1));
        b.buffer.write_f32(&fill(k * n, 2));
        let a_bf = cast_f32_to_bf16(&a)?;
        let b_bf = cast_f32_to_bf16(&b)?;

        // Production reference.
        let c_ref = rt.alloc_tensor_f32(&[m, n])?;
        gemm(&a_bf, &b_bf, &c_ref, GemmBackend::TensorOps)?;
        rt.synchronize()?;
        let refv = c_ref.buffer.read_f32()[..m * n].to_vec();
        let refmax = refv.iter().fold(0f32, |acc, x| acc.max(x.abs())) as f64;

        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let prod = {
            for _ in 0..warmup { gemm(&a_bf, &b_bf, &c_ref, GemmBackend::TensorOps)?; rt.synchronize()?; }
            let mut s = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                gemm(&a_bf, &b_bf, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                s.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            median(s)
        };
        println!("\n{label}  M={m} N={n} K={k}   production {prod:.3} ms  {:.0} GFLOP/s",
                 flop / (prod * 1e6));
        println!("  {:<32}{:>10}{:>12}{:>9}{:>12}", "variant", "ms", "GFLOP/s", "vs prod", "max_rel_err");

        for v in VARIANTS {
            if m % v.sm != 0 || n % v.sn != 0 || k % v.bk != 0 {
                println!("  {:<32}{:>10}", v.kernel, "skip(div)");
                continue;
            }
            let c = rt.alloc_tensor_f32(&[m, n])?;
            if run_variant(&rt, v, &a_bf, &b_bf, &c, m, n, k).is_err() {
                println!("  {:<32}{:>10}", v.kernel, "skip(pipe)");
                continue;
            }
            rt.synchronize()?;
            let got = c.buffer.read_f32()[..m * n].to_vec();
            let err = got.iter().zip(&refv)
                .map(|(x, y)| (*x as f64 - *y as f64).abs())
                .fold(0.0, f64::max) / refmax;

            for _ in 0..warmup { run_variant(&rt, v, &a_bf, &b_bf, &c, m, n, k)?; rt.synchronize()?; }
            let mut s = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                run_variant(&rt, v, &a_bf, &b_bf, &c, m, n, k)?;
                rt.synchronize()?;
                s.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let med = median(s);
            println!("  {:<32}{:>10.3}{:>12.0}{:>8.2}×{:>12.2e}",
                     v.kernel, med, flop / (med * 1e6), prod / med, err);
        }

        let prod_after = {
            let mut s = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                gemm(&a_bf, &b_bf, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                s.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            median(s)
        };
        println!("  production re-measured after: {prod_after:.3} ms ({:.0} GFLOP/s) \
— drift vs before: {:+.1}%",
                 flop / (prod_after * 1e6), (prod_after / prod - 1.0) * 100.0);
    }

    // ---- f32 relaxed-precision (tf32-class) lane ----
    rt.set_relaxed_precision(true);
    println!("\n================ f32 relaxed-precision ================");
    for &(m, n, k, label) in SHAPES {
        let a = rt.alloc_tensor_f32(&[m, k])?;
        let b = rt.alloc_tensor_f32(&[k, n])?;
        a.buffer.write_f32(&fill(m * k, 1));
        b.buffer.write_f32(&fill(k * n, 2));

        let c_ref = rt.alloc_tensor_f32(&[m, n])?;
        gemm(&a, &b, &c_ref, GemmBackend::TensorOps)?;
        rt.synchronize()?;
        let refv = c_ref.buffer.read_f32()[..m * n].to_vec();
        let refmax = refv.iter().fold(0f32, |acc, x| acc.max(x.abs())) as f64;

        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        for _ in 0..warmup { gemm(&a, &b, &c_ref, GemmBackend::TensorOps)?; rt.synchronize()?; }
        let mut s = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            gemm(&a, &b, &c_ref, GemmBackend::TensorOps)?;
            rt.synchronize()?;
            s.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let prod = median(s);
        println!("\n{label}  M={m} N={n} K={k}   production(relaxed) {prod:.3} ms  {:.0} GFLOP/s",
                 flop / (prod * 1e6));
        println!("  {:<32}{:>10}{:>12}{:>9}{:>12}", "variant", "ms", "GFLOP/s", "vs prod", "max_rel_err");

        for v in F32R_VARIANTS {
            if m % v.sm != 0 || n % v.sn != 0 || k % v.bk != 0 {
                println!("  {:<32}{:>10}", v.kernel, "skip(div)");
                continue;
            }
            let c = rt.alloc_tensor_f32(&[m, n])?;
            if run_variant(&rt, v, &a, &b, &c, m, n, k).is_err() {
                println!("  {:<32}{:>10}", v.kernel, "skip(pipe)");
                continue;
            }
            rt.synchronize()?;
            let got = c.buffer.read_f32()[..m * n].to_vec();
            let err = got.iter().zip(&refv)
                .map(|(x, y)| (*x as f64 - *y as f64).abs())
                .fold(0.0, f64::max) / refmax;

            for _ in 0..warmup { run_variant(&rt, v, &a, &b, &c, m, n, k)?; rt.synchronize()?; }
            let mut s = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                run_variant(&rt, v, &a, &b, &c, m, n, k)?;
                rt.synchronize()?;
                s.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let med = median(s);
            println!("  {:<32}{:>10.3}{:>12.0}{:>8.2}×{:>12.2e}",
                     v.kernel, med, flop / (med * 1e6), prod / med, err);
        }
    }
    // ---- TN / NT bf16 lanes ----
    // gemm_{tn,nt}_train only take the bf16 path when the runtime is in Bf16
    // precision mode; ensure_bf16 then passes already-bf16 operands through.
    rt.set_relaxed_precision(false);
    rt.set_precision(PrecisionMode::Bf16);
    for (lane, shapes) in [("TN", TN_SHAPES), ("NT", NT_SHAPES)] {
        println!("\n================ {lane} bf16 (gradient shapes) ================");
        for &(m, n, k, label) in shapes {
            // TN: A is [K,M], B is [K,N]. NT: A is [M,K], B is [N,K].
            let (ash, bsh) = if lane == "TN" {
                (vec![k, m], vec![k, n])
            } else {
                (vec![m, k], vec![n, k])
            };
            let a = rt.alloc_tensor_f32(&ash)?;
            let b = rt.alloc_tensor_f32(&bsh)?;
            a.buffer.write_f32(&fill(ash[0] * ash[1], 1));
            b.buffer.write_f32(&fill(bsh[0] * bsh[1], 2));
            let a_bf = cast_f32_to_bf16(&a)?;
            let b_bf = cast_f32_to_bf16(&b)?;

            let c_ref = rt.alloc_tensor_f32(&[m, n])?;
            let run_prod = |c: &Tensor| -> Result<(), String> {
                if lane == "TN" { gemm_tn_train(&a_bf, &b_bf, c, GemmBackend::TensorOps) }
                else { gemm_nt_train(&a_bf, &b_bf, c, GemmBackend::TensorOps) }
            };
            run_prod(&c_ref)?;
            rt.synchronize()?;
            let refv = c_ref.buffer.read_f32()[..m * n].to_vec();
            let refmax = refv.iter().fold(0f32, |acc, x| acc.max(x.abs())) as f64;

            let flop = 2.0 * m as f64 * n as f64 * k as f64;
            for _ in 0..warmup { run_prod(&c_ref)?; rt.synchronize()?; }
            let mut sm_ = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                run_prod(&c_ref)?;
                rt.synchronize()?;
                sm_.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let prod = median(sm_);
            println!("\n{lane} {label}  M={m} N={n} K={k}   production {prod:.3} ms  {:.0} GFLOP/s",
                     flop / (prod * 1e6));
            println!("  {:<24}{:>10}{:>12}{:>9}{:>12}", "tile", "ms", "GFLOP/s", "vs prod", "max_rel_err");

            for &(tsm, tsn, nsg, suffix) in TNNT_TILES {
                let kernel = format!("mm_{}_{}", lane.to_lowercase(), suffix);
                let v = Variant { kernel: Box::leak(kernel.into_boxed_str()),
                                  sm: tsm, sn: tsn, bk: 1, nsg, needs_zero: false };
                if m % tsm != 0 || n % tsn != 0 {
                    println!("  {suffix:<24}{:>10}", "skip(div)");
                    continue;
                }
                let c = rt.alloc_tensor_f32(&[m, n])?;
                if run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k).is_err() {
                    println!("  {suffix:<24}{:>10}", "skip(pipe)");
                    continue;
                }
                rt.synchronize()?;
                let got = c.buffer.read_f32()[..m * n].to_vec();
                let err = got.iter().zip(&refv)
                    .map(|(x, y)| (*x as f64 - *y as f64).abs())
                    .fold(0.0, f64::max) / refmax;

                for _ in 0..warmup { run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k)?; rt.synchronize()?; }
                let mut sv = Vec::new();
                for _ in 0..iters {
                    let t0 = Instant::now();
                    run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k)?;
                    rt.synchronize()?;
                    sv.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                let med = median(sv);
                println!("  {suffix:<24}{:>10.3}{:>12.0}{:>8.2}×{:>12.2e}",
                         med, flop / (med * 1e6), prod / med, err);
            }

            // K-blocked / register-accumulator candidates.
            for &(kernel, tsm, tsn, tbk, nsg) in if lane == "TN" { TN_KBLK } else { NT_KBLK } {
                let label = &kernel[3..];
                if m % tsm != 0 || n % tsn != 0 || k % tbk != 0 {
                    println!("  {label:<28}{:>10}", "skip(div)");
                    continue;
                }
                let v = Variant { kernel, sm: tsm, sn: tsn, bk: tbk, nsg, needs_zero: false };
                let c = rt.alloc_tensor_f32(&[m, n])?;
                if run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k).is_err() {
                    println!("  {label:<28}{:>10}", "skip(pipe)");
                    continue;
                }
                rt.synchronize()?;
                let got = c.buffer.read_f32()[..m * n].to_vec();
                let err = got.iter().zip(&refv)
                    .map(|(x, y)| (*x as f64 - *y as f64).abs())
                    .fold(0.0, f64::max) / refmax;

                for _ in 0..warmup {
                    run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k)?;
                    rt.synchronize()?;
                }
                let mut sv = Vec::new();
                for _ in 0..iters {
                    rt.synchronize()?;
                    let t0 = Instant::now();
                    run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k)?;
                    rt.synchronize()?;
                    sv.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                let med = median(sv);
                println!("  {label:<28}{:>10.3}{:>12.0}{:>8.2}×{:>12.2e}",
                         med, flop / (med * 1e6), prod / med, err);
            }
        }
    }
    // ---- accumulating TN / NT bf16 (dominant dW path: 24 call sites) ----
    for (lane, shapes) in [("tnacc", TN_SHAPES), ("ntacc", NT_SHAPES)] {
        let tn = lane == "tnacc";
        println!("\n================ {lane} bf16 (accumulate) ================");
        for &(m, n, k, label) in shapes {
            let (ash, bsh) = if tn { (vec![k, m], vec![k, n]) } else { (vec![m, k], vec![n, k]) };
            let a = rt.alloc_tensor_f32(&ash)?;
            let b = rt.alloc_tensor_f32(&bsh)?;
            a.buffer.write_f32(&fill(ash[0] * ash[1], 1));
            b.buffer.write_f32(&fill(bsh[0] * bsh[1], 2));
            let a_bf = cast_f32_to_bf16(&a)?;
            let b_bf = cast_f32_to_bf16(&b)?;

            let c_ref = rt.alloc_tensor_f32(&[m, n])?;
            let run_prod = |c: &Tensor| -> Result<(), String> {
                if tn { gemm_tn_accum_train(&a_bf, &b_bf, c, GemmBackend::TensorOps) }
                else { gemm_nt_accum_train(&a_bf, &b_bf, c, GemmBackend::TensorOps) }
            };
            // Accumulate onto a zeroed C once, so the reference is comparable.
            c_ref.buffer.write_f32(&vec![0f32; m * n]);
            run_prod(&c_ref)?;
            rt.synchronize()?;
            let refv = c_ref.buffer.read_f32()[..m * n].to_vec();
            let refmax = refv.iter().fold(0f32, |acc, x| acc.max(x.abs())) as f64;

            let flop = 2.0 * m as f64 * n as f64 * k as f64;
            for _ in 0..warmup { run_prod(&c_ref)?; rt.synchronize()?; }
            let mut sm_ = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                run_prod(&c_ref)?;
                rt.synchronize()?;
                sm_.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let prod = median(sm_);
            println!("\n{lane} {label}  M={m} N={n} K={k}   production {prod:.3} ms  {:.0} GFLOP/s",
                     flop / (prod * 1e6));
            println!("  {:<24}{:>10}{:>12}{:>9}{:>12}", "tile", "ms", "GFLOP/s", "vs prod", "max_rel_err");

            for &(tsm, tsn, nsg, suffix) in TNNT_TILES {
                let kernel = format!("mm_{lane}_{suffix}");
                let v = Variant { kernel: Box::leak(kernel.into_boxed_str()),
                                  sm: tsm, sn: tsn, bk: 1, nsg, needs_zero: false };
                if m % tsm != 0 || n % tsn != 0 {
                    println!("  {suffix:<24}{:>10}", "skip(div)");
                    continue;
                }
                let c = rt.alloc_tensor_f32(&[m, n])?;
                c.buffer.write_f32(&vec![0f32; m * n]);
                if run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k).is_err() {
                    println!("  {suffix:<24}{:>10}", "skip(pipe)");
                    continue;
                }
                rt.synchronize()?;
                let got = c.buffer.read_f32()[..m * n].to_vec();
                let err = got.iter().zip(&refv)
                    .map(|(x, y)| (*x as f64 - *y as f64).abs())
                    .fold(0.0, f64::max) / refmax;

                for _ in 0..warmup { run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k)?; rt.synchronize()?; }
                let mut sv = Vec::new();
                for _ in 0..iters {
                    let t0 = Instant::now();
                    run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k)?;
                    rt.synchronize()?;
                    sv.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                let med = median(sv);
                println!("  {suffix:<24}{:>10.3}{:>12.0}{:>8.2}×{:>12.2e}",
                         med, flop / (med * 1e6), prod / med, err);
            }

            // K-blocked / register-accumulator candidates.
            for &(kernel, tsm, tsn, tbk, nsg) in if tn { TNACC_KBLK } else { NTACC_KBLK } {
                let label = &kernel[3..];
                if m % tsm != 0 || n % tsn != 0 || k % tbk != 0 {
                    println!("  {label:<28}{:>10}", "skip(div)");
                    continue;
                }
                let v = Variant { kernel, sm: tsm, sn: tsn, bk: tbk, nsg, needs_zero: false };
                let c = rt.alloc_tensor_f32(&[m, n])?;
                c.buffer.write_f32(&vec![0f32; m * n]);
                if run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k).is_err() {
                    println!("  {label:<28}{:>10}", "skip(pipe)");
                    continue;
                }
                rt.synchronize()?;
                let got = c.buffer.read_f32()[..m * n].to_vec();
                let err = got.iter().zip(&refv)
                    .map(|(x, y)| (*x as f64 - *y as f64).abs())
                    .fold(0.0, f64::max) / refmax;

                for _ in 0..warmup {
                c.buffer.write_f32(&vec![0f32; m * n]);
                    run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k)?;
                    rt.synchronize()?;
                }
                let mut sv = Vec::new();
                for _ in 0..iters {
                c.buffer.write_f32(&vec![0f32; m * n]);
                    rt.synchronize()?;
                    let t0 = Instant::now();
                    run_variant(&rt, &v, &a_bf, &b_bf, &c, m, n, k)?;
                    rt.synchronize()?;
                    sv.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                let med = median(sv);
                println!("  {label:<28}{:>10.3}{:>12.0}{:>8.2}×{:>12.2e}",
                         med, flop / (med * 1e6), prod / med, err);
            }
        }
    }
    Ok(())
}