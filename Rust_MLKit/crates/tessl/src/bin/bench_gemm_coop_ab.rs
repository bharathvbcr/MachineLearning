//! Paired, interleaved A/B for the register-accumulator ("coop") GEMM kernels.
//!
//! The tile-tune rig measures a baseline block and a variant block minutes
//! apart, so GPU clock drift lands entirely in the ratio: running it four times
//! showed the *production coop kernel against itself* ranging 0.92x–1.46x. That
//! noise floor is wider than every effect being measured.
//!
//! Here each round interleaves baseline and candidate iteration-by-iteration
//! (A,B,A,B,...) and takes the ratio of the two per-round medians, so drift
//! affects both arms equally. Rounds are repeated and the spread is reported —
//! a claim is only kept when the *worst* round still wins.
//!
//! Both arms go through the same binder and dispatch code, so the only
//! difference is the kernel. Correctness is checked against the baseline's own
//! output before any timing is reported.

use tessl::gemm::{cast_f32_to_bf16, GemmBackend};
use tessl::runtime::{mtl_size, GpuRuntime, PrecisionMode};
use tessl::tensor::Tensor;
use objc2_metal::MTLComputePipelineState;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Arm {
    kernel: &'static str,
    sm: usize,
    sn: usize,
    nsg: usize,
    /// K must divide by this for the kernel to be legal (1 = no constraint).
    bk: usize,
}

/// (label, baseline, candidates, bf16?)
struct Lane {
    label: &'static str,
    base: Arm,
    cands: &'static [Arm],
    bf16: bool,
}

const LANES: &[Lane] = &[
    Lane {
        label: "NN bf16",
        base: Arm { kernel: "matmul2d_tensorops_bf16_f32", sm: 64, sn: 64, nsg: 4, bk: 1 },
        cands: &[
            Arm { kernel: "matmul2d_tensorops_bf16_f32_coop", sm: 64, sn: 64, nsg: 4, bk: 128 },
            Arm { kernel: "mm_coop_64x64_bk256_sg4", sm: 64, sn: 64, nsg: 4, bk: 256 },
            Arm { kernel: "mm_coop_64x64_bk64_sg4", sm: 64, sn: 64, nsg: 4, bk: 64 },
            // Wider tiles: candidates for the large-N shapes where torch leads.
            Arm { kernel: "mm_coop_128x64_bk128_sg4", sm: 128, sn: 64, nsg: 4, bk: 128 },
            Arm { kernel: "mm_coop_128x64_bk128_sg8", sm: 128, sn: 64, nsg: 8, bk: 128 },
            Arm { kernel: "mm_coop_128x128_bk128_sg8", sm: 128, sn: 128, nsg: 8, bk: 128 },
            Arm { kernel: "mm_coop_128x128_bk128_sg4", sm: 128, sn: 128, nsg: 4, bk: 128 },
            Arm { kernel: "mm_coop_64x128_bk128_sg4", sm: 64, sn: 128, nsg: 4, bk: 128 },
            Arm { kernel: "mm_coop_256x64_bk128_sg8", sm: 256, sn: 64, nsg: 8, bk: 128 },
        ],
        bf16: true,
    },
    Lane {
        label: "NN f32-relaxed",
        base: Arm { kernel: "matmul2d_tensorops_f32_relaxed", sm: 128, sn: 64, nsg: 4, bk: 1 },
        cands: &[
            Arm { kernel: "mm_f32rcoop_128x64_bk128_sg4", sm: 128, sn: 64, nsg: 4, bk: 128 },
            Arm { kernel: "mm_f32rcoop_128x64_bk256_sg4", sm: 128, sn: 64, nsg: 4, bk: 256 },
            Arm { kernel: "mm_f32rcoop_128x64_bk64_sg4", sm: 128, sn: 64, nsg: 4, bk: 64 },
        ],
        bf16: false,
    },
];

/// Forward NN shapes for every arch_02 preset, plus a K ladder at fixed M=N that
/// isolates the C-round-trip term (which scales with K/BK and nothing else).
const DEFAULT_SHAPES: &[(usize, usize, usize, &str)] = &[
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
    (2048, 2048, 256, "kladder_k256"),
    (2048, 2048, 512, "kladder_k512"),
    (2048, 2048, 1024, "kladder_k1024"),
    (2048, 2048, 2048, "kladder_k2048"),
    (2048, 2048, 4096, "kladder_k4096"),
    (2048, 2048, 8192, "kladder_k8192"),
];

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((((s >> 32) as u32) as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32
        })
        .collect()
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn run(
    rt: &std::sync::Arc<GpuRuntime>,
    arm: &Arm,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    let p = rt.pipeline(arm.kernel)?;
    let tiles_n = n / arm.sn;
    let tiles_m = m / arm.sm;
    let tpt = p.threadExecutionWidth() as usize * arm.nsg;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        bnd.dispatch(mtl_size(tiles_n * tiles_m, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

fn time_once(
    rt: &std::sync::Arc<GpuRuntime>,
    arm: &Arm,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<f64, String> {
    rt.synchronize()?;
    let t0 = Instant::now();
    run(rt, arm, a, b, c, m, n, k)?;
    rt.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1000.0)
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
             \n    TESSL_GEMM_TUNE=1 cargo build --release --bins\n"
        );
        std::process::exit(2);
    }
}

fn main() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    require_tune_kernels(&rt, "mm_coop_64x64_bk128_sg4");
    let rounds: usize = env_num("BENCH_ROUNDS", 7);
    let iters: usize = env_num("BENCH_ITERS", 21);
    let warmup: usize = env_num("BENCH_WARMUP", 8);

    let shapes: Vec<(usize, usize, usize, String)> = match std::env::var("BENCH_SHAPES") {
        Ok(s) => s
            .split(',')
            .filter(|t| !t.trim().is_empty())
            .map(|t| {
                let d: Vec<usize> = t.trim().split('x').map(|x| x.parse().unwrap()).collect();
                assert_eq!(d.len(), 3, "shape must be MxNxK");
                (d[0], d[1], d[2], format!("{}x{}x{}", d[0], d[1], d[2]))
            })
            .collect(),
        Err(_) => DEFAULT_SHAPES
            .iter()
            .map(|&(m, n, k, l)| (m, n, k, l.to_string()))
            .collect(),
    };

    println!(
        "paired interleaved A/B — {rounds} rounds x {iters} iters (warmup {warmup})\n\
         ratio = baseline_ms / candidate_ms per round; >1.00 means the candidate is faster.\n"
    );

    for lane in LANES {
        println!("================ {} ================", lane.label);
        rt.set_precision(if lane.bf16 {
            PrecisionMode::Bf16
        } else {
            PrecisionMode::F32
        });
        rt.set_relaxed_precision(!lane.bf16);
        let _ = GemmBackend::TensorOps;

        for (m, n, k, label) in &shapes {
            let (m, n, k) = (*m, *n, *k);
            if m % lane.base.sm != 0 || n % lane.base.sn != 0 {
                continue;
            }
            let a_f = rt.alloc_tensor_f32(&[m, k])?;
            let b_f = rt.alloc_tensor_f32(&[k, n])?;
            a_f.buffer.write_f32(&fill(m * k, 1));
            b_f.buffer.write_f32(&fill(k * n, 2));
            let (a, b) = if lane.bf16 {
                (cast_f32_to_bf16(&a_f)?, cast_f32_to_bf16(&b_f)?)
            } else {
                (a_f.clone(), b_f.clone())
            };

            let c_base = rt.alloc_tensor_f32(&[m, n])?;
            run(&rt, &lane.base, &a, &b, &c_base, m, n, k)?;
            rt.synchronize()?;
            let refv = c_base.buffer.read_f32()[..m * n].to_vec();
            let refmax = refv.iter().fold(0f32, |acc, x| acc.max(x.abs())) as f64;
            let flop = 2.0 * m as f64 * n as f64 * k as f64;

            println!("\n{label}  M={m} N={n} K={k}");
            println!(
                "  {:<36}{:>10}{:>11}{:>9}{:>9}{:>9}{:>11}",
                "candidate", "base_ms", "cand_ms", "median", "worst", "best", "max_rel_err"
            );

            // One candidate output buffer for the whole shape. Allocating a
            // fresh one per candidate let allocations pile up across the run:
            // with nine candidates the baseline itself drifted 0.268 -> 0.665 ms
            // *within a single shape block*, which would have been read as a
            // candidate win. Every row must see the same machine.
            let c_cand = rt.alloc_tensor_f32(&[m, n])?;
            // The baseline is the same kernel on every row, so its spread across
            // rows measures nothing but contamination. Reported, not assumed.
            let mut base_across_rows: Vec<f64> = Vec::new();

            for cand in lane.cands {
                if m % cand.sm != 0 || n % cand.sn != 0 || k % cand.bk != 0 {
                    println!("  {:<36}{:>10}", cand.kernel, "skip(div)");
                    continue;
                }
                if run(&rt, cand, &a, &b, &c_cand, m, n, k).is_err() {
                    println!("  {:<36}{:>10}", cand.kernel, "skip(pipe)");
                    continue;
                }
                rt.synchronize()?;
                let got = c_cand.buffer.read_f32()[..m * n].to_vec();
                let err = got
                    .iter()
                    .zip(&refv)
                    .map(|(x, y)| (*x as f64 - *y as f64).abs())
                    .fold(0.0, f64::max)
                    / refmax.max(f64::MIN_POSITIVE);

                let mut ratios = Vec::with_capacity(rounds);
                let mut base_ms_all = Vec::new();
                let mut cand_ms_all = Vec::new();
                for _ in 0..rounds {
                    for _ in 0..warmup {
                        run(&rt, &lane.base, &a, &b, &c_base, m, n, k)?;
                        run(&rt, cand, &a, &b, &c_cand, m, n, k)?;
                    }
                    rt.synchronize()?;
                    let mut bt = Vec::with_capacity(iters);
                    let mut ct = Vec::with_capacity(iters);
                    // Interleave so clock drift lands on both arms equally.
                    for i in 0..iters {
                        if i % 2 == 0 {
                            bt.push(time_once(&rt, &lane.base, &a, &b, &c_base, m, n, k)?);
                            ct.push(time_once(&rt, cand, &a, &b, &c_cand, m, n, k)?);
                        } else {
                            ct.push(time_once(&rt, cand, &a, &b, &c_cand, m, n, k)?);
                            bt.push(time_once(&rt, &lane.base, &a, &b, &c_base, m, n, k)?);
                        }
                    }
                    let bm = median(&mut bt);
                    let cm = median(&mut ct);
                    ratios.push(bm / cm);
                    base_ms_all.push(bm);
                    cand_ms_all.push(cm);
                }
                let med = median(&mut ratios.clone());
                let worst = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
                let best = ratios.iter().cloned().fold(0.0, f64::max);
                let bm = median(&mut base_ms_all);
                let cm = median(&mut cand_ms_all);
                base_across_rows.push(bm);
                println!(
                    "  {:<36}{:>10.3}{:>11.3}{:>8.2}×{:>8.2}×{:>8.2}×{:>11.2e}   \
                     ({:.0} vs {:.0} GFLOP/s)",
                    cand.kernel,
                    bm,
                    cm,
                    med,
                    worst,
                    best,
                    err,
                    flop / (bm * 1e6),
                    flop / (cm * 1e6)
                );
            }

            if base_across_rows.len() > 1 {
                let lo = base_across_rows.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = base_across_rows.iter().cloned().fold(0.0, f64::max);
                let spread = hi / lo - 1.0;
                let flag = if spread > 0.10 { "  <-- EXCEEDS 10%: ratios above are not comparable" } else { "" };
                println!(
                    "  {:<36}{:>10.3}{:>11.3}   baseline spread across rows {:.1}%{}",
                    "(same baseline kernel, every row)", lo, hi, spread * 100.0, flag
                );
            }
        }
    }
    Ok(())
}

fn env_num(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
