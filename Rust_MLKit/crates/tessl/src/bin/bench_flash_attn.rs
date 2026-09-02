//! Flash-attention shape sweep: the three attention kernels, JSON out for the
//! cross-runtime compare in `bench/flash_attn_torch_mlx.py`.
//!
//! These kernels had thorough correctness coverage (`tests/attention.rs` scores
//! them against an f64 transcription of their own masking rule) and **no timing
//! lane at all** — for the kernel that dominates inference cost.
//!
//! Protocol is pinned to `bench_gemm_sweep`: same env contract, synchronize
//! every iteration, median over iters. `--dump-parity DIR` writes Q/K/V and
//! every lane's O for the numeric check.
//!
//! FLOP accounting is *exact*, not nominal: the kernels skip masked work, so
//! the useful-work count is over unmasked (q, k) pairs only, computed from the
//! same rule the kernels apply. `dense_gflops` is reported alongside so a
//! reader can see how much of the square was skipped.

mod common;

use common::{env_usize, fill_dist, median, Dist};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tessl::nn::{self, AttnDims, AttnHeadDim};
use tessl::npy::write_npy_f32;
use tessl::runtime::GpuRuntime;
use tessl::tensor::GpuBuffer;

/// One attention configuration.
#[derive(Clone, Copy)]
struct Cfg {
    label: &'static str,
    b: usize,
    tq: usize,
    tkv: usize,
    h: usize,
    hkv: usize,
    d: usize,
    /// `None` selects the global (causal, D=512) kernel.
    window: Option<usize>,
    q_off: usize,
    kv_off: usize,
}

impl Cfg {
    fn scale(&self) -> f32 {
        1.0 / (self.d as f32).sqrt()
    }

    /// Unmasked (query, key) pairs, by the kernels' own rule:
    /// `q_abs = q_off + t_q`, `k_abs = kv_off + t_k`, sliding window keeps
    /// `max(0, q_abs - w + 1) <= k_abs <= q_abs`, global keeps `k_abs <= q_abs`.
    fn live_pairs(&self) -> u64 {
        let mut n = 0u64;
        for t_q in 0..self.tq {
            let q_abs = (self.q_off + t_q) as i64;
            let lo = match self.window {
                Some(w) => (q_abs - w as i64 + 1).max(0),
                None => 0,
            };
            for t_k in 0..self.tkv {
                let k_abs = (self.kv_off + t_k) as i64;
                if k_abs >= lo && k_abs <= q_abs {
                    n += 1;
                }
            }
        }
        n
    }

    /// 2*D for the QK dot and 2*D for the PV accumulate, per live pair, per
    /// batch and query head.
    fn live_flop(&self) -> f64 {
        4.0 * self.d as f64 * (self.b * self.h) as f64 * self.live_pairs() as f64
    }

    fn dense_flop(&self) -> f64 {
        4.0 * self.d as f64 * (self.b * self.h) as f64 * (self.tq * self.tkv) as f64
    }

    fn head_dim(&self) -> Result<AttnHeadDim, String> {
        match self.d {
            128 => Ok(AttnHeadDim::D128),
            256 => Ok(AttnHeadDim::D256),
            other => Err(format!(
                "{}: head dim {other} has no sliding-window kernel (128 or 256)",
                self.label
            )),
        }
    }

    fn dims(&self) -> AttnDims {
        AttnDims {
            batch: self.b as u32,
            tq: self.tq as u32,
            heads: self.h as u32,
            heads_kv: self.hkv as u32,
            window: self.window.unwrap_or(0) as u32,
            scale: self.scale(),
        }
    }

    fn q_elems(&self) -> usize {
        self.b * self.tq * self.h * self.d
    }

    fn kv_elems(&self) -> usize {
        self.b * self.tkv * self.hkv * self.d
    }
}

/// Prefill and decode at each head dimension. `window` of `None` routes to
/// `flash_attn_global_h512`, which is the only D=512 path.
///
/// Head counts follow the grouped-query ratios production models actually use
/// (H:Hkv of 4:1), because the kernel indexes K/V by `h / (H/Hkv)` and a 1:1
/// ratio would never exercise that divide.
const CFGS: &[Cfg] = &[
    // ---- D=128 sliding window, 4:1 GQA ----
    Cfg {
        label: "swa128_prefill_512",
        b: 1,
        tq: 512,
        tkv: 512,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 0,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_prefill_2048",
        b: 1,
        tq: 2048,
        tkv: 2048,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 0,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_prefill_4096",
        b: 1,
        tq: 4096,
        tkv: 4096,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 0,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_decode_1k",
        b: 1,
        tq: 1,
        tkv: 1024,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 1023,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_decode_4k",
        b: 1,
        tq: 1,
        tkv: 4096,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 4095,
        kv_off: 0,
    },
    Cfg {
        label: "swa128_decode_b8_4k",
        b: 8,
        tq: 1,
        tkv: 4096,
        h: 32,
        hkv: 8,
        d: 128,
        window: Some(1024),
        q_off: 4095,
        kv_off: 0,
    },
    // ---- D=256 sliding window ----
    Cfg {
        label: "swa256_prefill_2048",
        b: 1,
        tq: 2048,
        tkv: 2048,
        h: 16,
        hkv: 4,
        d: 256,
        window: Some(1024),
        q_off: 0,
        kv_off: 0,
    },
    Cfg {
        label: "swa256_decode_4k",
        b: 1,
        tq: 1,
        tkv: 4096,
        h: 16,
        hkv: 4,
        d: 256,
        window: Some(1024),
        q_off: 4095,
        kv_off: 0,
    },
    // ---- D=512 global causal ----
    Cfg {
        label: "global512_prefill_1024",
        b: 1,
        tq: 1024,
        tkv: 1024,
        h: 8,
        hkv: 2,
        d: 512,
        window: None,
        q_off: 0,
        kv_off: 0,
    },
    Cfg {
        label: "global512_decode_4k",
        b: 1,
        tq: 1,
        tkv: 4096,
        h: 8,
        hkv: 2,
        d: 512,
        window: None,
        q_off: 4095,
        kv_off: 0,
    },
];

fn u32_buf(rt: &Arc<GpuRuntime>, v: u32) -> Result<GpuBuffer, String> {
    let b = rt.alloc_buffer(4)?;
    b.write_u32(&[v]);
    Ok(b)
}

/// The seven buffers every attention dispatch needs. Bundled so the launch and
/// timing helpers take one operand set rather than a seven-argument tail whose
/// order is easy to transpose silently.
struct Bufs<'a> {
    q: &'a GpuBuffer,
    k: &'a GpuBuffer,
    v: &'a GpuBuffer,
    o: &'a GpuBuffer,
    tkv: &'a GpuBuffer,
    qo: &'a GpuBuffer,
    ko: &'a GpuBuffer,
}

/// Which implementation a lane measures.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Impl {
    /// The original BR-tiled kernels, kept as the A/B baseline.
    Tiled,
    /// What the public entry points actually dispatch, routing by Tq.
    Routed,
    /// FlashDecoding: split over KV, `Tq == 1` only.
    Decode,
    /// Row-parallel: one simdgroup per query row.
    Rows,
}

impl Impl {
    fn tag(self) -> &'static str {
        match self {
            Impl::Tiled => "tessl-tiled",
            Impl::Routed => "tessl",
            Impl::Decode => "tessl-decode",
            Impl::Rows => "tessl-rows",
        }
    }
}

/// One dispatch of whichever kernel this configuration selects.
fn launch_impl(rt: &Arc<GpuRuntime>, c: &Cfg, b: &Bufs, imp: Impl) -> Result<(), String> {
    if imp == Impl::Decode {
        return nn::flash_attn_decode(
            rt,
            b.q,
            b.k,
            b.v,
            b.o,
            b.tkv,
            b.qo,
            b.ko,
            c.dims(),
            c.d as u32,
            c.tkv,
            false,
        );
    }
    if imp == Impl::Routed {
        // The public entry points, exactly as a caller reaches them.
        return match c.window {
            Some(_) => nn::flash_attn_swa(
                rt,
                c.head_dim()?,
                b.q,
                b.k,
                b.v,
                b.o,
                b.tkv,
                b.qo,
                b.ko,
                c.dims(),
            ),
            None => nn::flash_attn_global_h512(
                rt,
                b.q,
                b.k,
                b.v,
                b.o,
                b.tkv,
                b.qo,
                b.ko,
                c.dims(),
                false,
            ),
        };
    }
    if imp == Impl::Rows {
        return nn::flash_attn_rows(
            rt,
            b.q,
            b.k,
            b.v,
            b.o,
            b.tkv,
            b.qo,
            b.ko,
            c.dims(),
            c.d as u32,
            false,
        );
    }
    launch(rt, c, b)
}

fn launch(rt: &Arc<GpuRuntime>, c: &Cfg, b: &Bufs) -> Result<(), String> {
    match c.window {
        Some(_) => nn::flash_attn_swa_tiled(
            rt,
            c.head_dim()?,
            b.q,
            b.k,
            b.v,
            b.o,
            b.tkv,
            b.qo,
            b.ko,
            c.dims(),
        ),
        None => {
            nn::flash_attn_global_h512(rt, b.q, b.k, b.v, b.o, b.tkv, b.qo, b.ko, c.dims(), false)
        }
    }
}

fn time_cfg(
    rt: &Arc<GpuRuntime>,
    c: &Cfg,
    b: &Bufs,
    imp: Impl,
    warmup: usize,
    iters: usize,
) -> Result<Vec<f64>, String> {
    for _ in 0..warmup {
        launch_impl(rt, c, b, imp)?;
        rt.synchronize()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        launch_impl(rt, c, b, imp)?;
        rt.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(samples)
}

/// Sentinel the output is seeded with, so a kernel that leaves rows untouched
/// is caught rather than inheriting whatever the allocator handed back.
const UNWRITTEN: f32 = -6.5e28;

/// Emit the kernel trace for `bench/kernel_coverage.py`. Prints nothing when
/// tracing is off, so normal runs are unchanged.
fn emit_kernel_trace() {
    if tessl::runtime::kernel_trace_enabled() {
        eprintln!(
            "KERNEL_TRACE {}",
            tessl::runtime::traced_kernels().join(",")
        );
    }
}

fn run() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    let args: Vec<String> = std::env::args().collect();
    let dump_dir = match args.iter().position(|a| a == "--dump-parity") {
        Some(i) => Some(
            args.get(i + 1)
                .filter(|v| !v.starts_with("--"))
                .ok_or("--dump-parity requires a directory argument")?
                .clone(),
        ),
        None => None,
    };

    let warmup = env_usize("BENCH_WARMUP", 10, 0)?;
    let iters = env_usize("BENCH_ITERS", 50, 1)?;
    let dist = match std::env::var("BENCH_ATTN_DIST") {
        Ok(v) => Dist::parse(v.trim())?,
        Err(std::env::VarError::NotPresent) => Dist::Uniform,
        Err(e) => return Err(format!("BENCH_ATTN_DIST: {e}")),
    };
    let only = match std::env::var("BENCH_ATTN_CFGS") {
        Ok(v) => Some(
            v.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect::<Vec<_>>(),
        ),
        Err(std::env::VarError::NotPresent) => None,
        Err(e) => return Err(format!("BENCH_ATTN_CFGS: {e}")),
    };
    let cfgs: Vec<&Cfg> = match &only {
        None => CFGS.iter().collect(),
        Some(want) => {
            let known: Vec<&str> = CFGS.iter().map(|c| c.label).collect();
            for w in want {
                if !known.contains(&w.as_str()) {
                    return Err(format!(
                        "BENCH_ATTN_CFGS: {w:?} is not a configuration; expected from {known:?}"
                    ));
                }
            }
            CFGS.iter()
                .filter(|c| want.contains(&c.label.to_string()))
                .collect()
        }
    };

    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {dir}: {e}"))?;
    }

    let mut rows: Vec<String> = Vec::new();
    let mut dumped: Vec<String> = Vec::new();
    for c in cfgs {
        let qh = fill_dist(c.q_elems(), 1, dist);
        let kh = fill_dist(c.kv_elems(), 2, dist);
        let vh = fill_dist(c.kv_elems(), 3, dist);
        let q = rt.alloc_buffer(qh.len() * 4)?;
        let k = rt.alloc_buffer(kh.len() * 4)?;
        let v = rt.alloc_buffer(vh.len() * 4)?;
        let o = rt.alloc_buffer(c.q_elems() * 4)?;
        q.write_f32(&qh);
        k.write_f32(&kh);
        v.write_f32(&vh);
        o.write_f32(&vec![UNWRITTEN; c.q_elems()]);

        let tkv = u32_buf(&rt, c.tkv as u32)?;
        let qo = u32_buf(&rt, c.q_off as u32)?;
        let ko = u32_buf(&rt, c.kv_off as u32)?;

        let bufs = Bufs {
            q: &q,
            k: &k,
            v: &v,
            o: &o,
            tkv: &tkv,
            qo: &qo,
            ko: &ko,
        };
        // Every Tq == 1 config is timed on both implementations in the same
        // run, so the comparison cannot pick up drift between two invocations.
        let impls: &[Impl] = if c.tq == 1 {
            &[Impl::Tiled, Impl::Routed, Impl::Decode, Impl::Rows]
        } else {
            &[Impl::Tiled, Impl::Routed, Impl::Rows]
        };
        for &imp in impls {
            let samples = time_cfg(&rt, c, &bufs, imp, warmup, iters)?;
            let med = median(samples.clone())?;
            if med <= 0.0 {
                return Err(format!("{}: median {med} ms is not positive", c.label));
            }
            let best = samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let live = c.live_flop();
            let gflops = live / (med * 1e6);
            let kernel = match (imp, c.window) {
                (Impl::Decode, _) => "flash_attn_decode",
                (Impl::Rows, _) => "flash_attn_rows",
                (Impl::Routed, _) => "routed",
                (Impl::Tiled, Some(_)) => c.head_dim()?.kernel(),
                (Impl::Tiled, None) => "flash_attn_global_h512",
            };
            eprintln!(
                "{:<22} {:<13} {kernel:<28} Tkv={} H={} D={}  {med:8.3} ms  {gflops:9.1} GFLOP/s",
                c.label,
                imp.tag(),
                c.tkv,
                c.h,
                c.d
            );
            rows.push(format!(
                r#"{{"cfg":"{}","kernel":"{kernel}","runtime":"{}","b":{},"tq":{},"tkv":{},"h":{},"hkv":{},"d":{},"window":{},"q_off":{},"kv_off":{},"median_ms":{med:.6},"best_ms":{best:.6},"live_pairs":{},"gflops":{gflops:.3},"dense_gflops":{:.3}}}"#,
                c.label, imp.tag(), c.b, c.tq, c.tkv, c.h, c.hkv, c.d,
                match c.window { Some(w) => w as i64, None => -1 },
                c.q_off, c.kv_off, c.live_pairs(),
                c.dense_flop() / (med * 1e6)
            ));
        }

        if let Some(dir) = &dump_dir {
            let out = o.read_f32()[..c.q_elems()].to_vec();
            if let Some(i) = out.iter().position(|x| *x == UNWRITTEN) {
                return Err(format!(
                    "{}: output element {i} was never written by the kernel",
                    c.label
                ));
            }
            if let Some(i) = out.iter().position(|x| !x.is_finite()) {
                return Err(format!(
                    "{}: non-finite output {} at element {i}",
                    c.label, out[i]
                ));
            }
            let d = Path::new(dir).join(c.label);
            std::fs::create_dir_all(&d).map_err(|e| format!("mkdir {}: {e}", d.display()))?;
            write_npy_f32(&d.join("q.npy"), &[c.b, c.tq, c.h, c.d], &qh)?;
            write_npy_f32(&d.join("k.npy"), &[c.b, c.tkv, c.hkv, c.d], &kh)?;
            write_npy_f32(&d.join("v.npy"), &[c.b, c.tkv, c.hkv, c.d], &vh)?;
            write_npy_f32(&d.join("o_tessl.npy"), &[c.b, c.tq, c.h, c.d], &out)?;
            // The decode path is a second implementation of the same rule, so
            // it is scored against the same f64 reference rather than only
            // against the kernel it replaces.
            if c.tq == 1 {
                o.write_f32(&vec![UNWRITTEN; c.q_elems()]);
                launch_impl(&rt, c, &bufs, Impl::Decode)?;
                rt.synchronize()?;
                let dec = o.read_f32()[..c.q_elems()].to_vec();
                if let Some(i) = dec.iter().position(|x| *x == UNWRITTEN) {
                    return Err(format!(
                        "{}: decode left output element {i} unwritten",
                        c.label
                    ));
                }
                if let Some(i) = dec.iter().position(|x| !x.is_finite()) {
                    return Err(format!(
                        "{}: decode produced non-finite {} at element {i}",
                        c.label, dec[i]
                    ));
                }
                write_npy_f32(&d.join("o_decode.npy"), &[c.b, c.tq, c.h, c.d], &dec)?;
            }
            // The parity dump always describes the general kernel's output,
            // which is what `o_tessl.npy` holds after the loop above.
            let kernel = match c.window {
                Some(_) => c.head_dim()?.kernel(),
                None => "flash_attn_global_h512",
            };
            dumped.push(format!(
                r#"{{"cfg":"{}","kernel":"{kernel}","b":{},"tq":{},"tkv":{},"h":{},"hkv":{},"d":{},"window":{},"q_off":{},"kv_off":{},"scale":{}}}"#,
                c.label, c.b, c.tq, c.tkv, c.h, c.hkv, c.d,
                match c.window { Some(w) => w as i64, None => -1 },
                c.q_off, c.kv_off, c.scale()
            ));
        }
    }

    // Manifest last and only on success, so an interrupted dump cannot be
    // scored as a complete one.
    if let Some(dir) = &dump_dir {
        std::fs::write(
            Path::new(dir).join("attn_manifest.json"),
            format!(
                r#"{{"dist":"{}","configs":[{}]}}"#,
                dist.name(),
                dumped.join(",")
            ),
        )
        .map_err(|e| format!("write manifest: {e}"))?;
        eprintln!("attention parity dump complete: {} configs", dumped.len());
    }
    println!("[{}]", rows.join(","));
    Ok(())
}

fn main() -> Result<(), String> {
    // Wrapped rather than called before each `return`: --dump-parity exits
    // early and an error exits earlier still, and a trace that silently skips
    // those paths would under-report coverage exactly where it matters.
    let outcome = run();
    emit_kernel_trace();
    outcome
}
