# tessl

Metal 4 GEMM and encode runtime for Apple silicon, built on Metal Performance
Primitives (MPP) TensorOps `matmul2d`.

Requires **macOS 26+**, Xcode 26 with the Metal Toolchain component
(`xcodebuild -downloadComponent MetalToolchain`), and an Apple GPU with neural
accelerators for the TensorOps path. There is a portable `simdgroup_matrix`
fallback, but it is roughly 2–3x slower and exists for A/B, not for shipping.

## What it does

| Module | Role |
|--------|------|
| `gemm` | TensorOps `matmul2d` GEMM — NN / TN / NT, plain and accumulating, f32 / tf32-relaxed / bf16→f32, split-K, register-resident accumulators |
| `runtime` | Device, MTL4 command buffer and compute encoder, residency sets, Hot/Cold/Bump pools, packed binder, constant arena, SharedEvent sync |
| `dispatch` | Argument-table binds and 1D/2D/3D dispatch helpers |
| `tensor` | `GpuBuffer` / `Tensor` views and dtypes, bounds-checked |
| `ops` | `softcap_f32` and small util launches |
| `npy` | numpy `.npy` writer, for checking GPU output against a host reference |
| `mtl_tensor` | Quantized `MTLTensor` prep (Int8 today; Int4/FP8 pending SDK) |
| `decode_icb`, `cb_replay`, `icb_smoke` | Indirect command buffer capture and replay for decode-shaped workloads |

Encode is **Metal 4 only**. The classic `MTLCommandQueue` path is deliberately
absent, not merely unused.

## Where it stands against PyTorch MPS

Measured on one M5 Pro with `bench/paired_cross_runtime.py`, which alternates
the two lanes round by round so GPU clock drift cancels. Geomean of per-shape
medians over 5 rounds, on an 8-shape ladder:

| | vs torch MPS | worst shape | best shape |
|---|---|---|---|
| bf16 → f32 accumulate | **1.00x** | 0.84x | 1.12x |
| f32 exact | 1.07x | 0.92x | 1.47x |
| tf32-relaxed vs torch f32 | 1.98x | 1.49x | 2.34x |

bf16 is at **parity** with Apple's own MPS, not ahead of it. The tf32 row is not
like-for-like — relaxed precision truncates the mantissa — and should be read as
what the opt-in buys, not as an f32 result.

Do not quote single-run cross-runtime numbers from this or any other harness:
two back-to-back runs of the same unpaired sweep disagreed by 16–21% on the
torch lane alone. Below roughly 2 GFLOP of work both runtimes sit on a ~0.25 ms
per-GEMM dispatch floor, so ratios there measure submit-and-wait latency rather
than the kernel.

## Using it

```rust
use tessl::{gemm, GemmBackend, GpuRuntime, PrecisionMode};

let rt = GpuRuntime::new()?;
let a = rt.alloc_tensor_f32(&[4096, 2304])?;
let b = rt.alloc_tensor_f32(&[2304, 768])?;
let c = rt.alloc_tensor_f32(&[4096, 768])?;
gemm(&a, &b, &c, GemmBackend::TensorOps)?;   // C = A @ B
rt.synchronize()?;
```

`build.rs` AOT-compiles `kernels/*.metal` into a `default.metallib` and bakes
its absolute path in, so no metallib shipping or lookup is needed.

### Consuming the kernels from another crate

A crate that builds its own metallib should compile **these** kernel sources
rather than copying them. `tessl` sets `links = "tessl"` and its build script
publishes the directory:

```rust
// your build.rs
let tessl_kernels = PathBuf::from(env::var("DEP_TESSL_KERNELS").unwrap());
// compile tessl_kernels.join("matmul_tensorops.metal") into your metallib
```

Then build your runtime from your own metallib —
`GpuRuntime::from_metallib_path(...)` — or overlay it onto tessl's with
`GpuRuntime::add_metallib(...)`. `tessl-arch02` takes the first route,
`gemma-metal` the second.

## Verifying

```bash
cargo test --release -- --test-threads=1        # GPU tests are not thread-safe
python3 scripts/audit_gemm_tiles.py             # runs from any directory
```

The audit cross-checks every Rust `TileGeom` against the `SM`/`SN` compiled into
the kernel it dispatches, and every cooperative kernel's `BKC` against
`COOP_BKC`. A drift in either silently corrupts results rather than failing, so
it is a build-time gate, not a nicety.

`gemm_randomized_shape_fuzz` is a seeded shape fuzz that **asserts its own
coverage** — it fails if any selectable NN kernel was chosen for under 1% of
cases. An earlier version passed three injected faults because it never
dispatched the kernels they were injected into.

```bash
GEMM_FUZZ_SEED=0xdeadbeef GEMM_FUZZ_CASES=1200 \
  cargo test --release --lib -- --test-threads=1 --nocapture gemm_randomized_shape_fuzz
```

`GEMM_FUZZ_SEED` accepts hex or decimal and **panics** on a malformed value
rather than falling back to the default — falling back made an eight-seed soak
silently re-run one seed eight times and pass every time.

## Benchmarking

The A/B rig is 92 measurement-only kernels and is **not** in the default
metallib (it takes the artifact from 0.20 MB to 1.07 MB). Opt in:

```bash
TESSL_GEMM_TUNE=1 cargo build --release --bins
```

| binary | purpose |
|---|---|
| `bench_gemm_coop_ab` | Paired, interleaved kernel A/B. Use this for kernel comparisons. |
| `bench_gemm_tile_tune` | Broad tile/BK ladder. |
| `bench_gemm_sweep` | Cross-runtime lane (f32 exact / tf32 / bf16), JSON out. |
| `probe_gemm_parity` | Bit-equality probe across backends. |
| `bench/paired_cross_runtime.py` | Alternates the tessl and torch/MLX lanes. |

Both tuning binaries `exit(2)` with the rebuild command when the rig is absent,
rather than printing a page of skips and exiting 0.

`bench/results/bf16_tile_tune_FINDINGS.md` is the working record: what was
measured, what was rejected, and the measurement errors made along the way.

## Environment variables

Canonically `TESSL_*`. The legacy `METAL_RUNTIME_*` and `METAL_NATIVE_*`
spellings are still read, in that order, so scripts written before the rename
keep working.

| variable | effect |
|---|---|
| `TESSL_GEMM_TUNE` | Build the A/B rig into the metallib (build-time) |
| `TESSL_GEMM_ACCUM` | TensorOps `multiply_accumulate` for TN/NT accumulate paths (default off) |
| `TESSL_GEMM_ACCUM_DX` | Accumulate for the dX NT path only (default off) |
| `TESSL_GEMM_INTERIOR` | f32 GEMM interior-offset tiles (default off) |
| `TESSL_HAZARD_BARRIERS` | Skip the always-on device barrier after each dispatch |
| `TESSL_COARSE_BARRIERS` | Phase-coarsened rather than per-RAW barriers |
| `METAL_RUNTIME_SKIP_AOT` | Skip metallib compilation; trusts a pre-existing `default.metallib` (hazard: may be stale) |

## Documentation

`../../docs/gemm_architecture.md` covers kernel selection, the cooperative
accumulator gate clause by clause, why the TN/NT paths deliberately have no
cooperative variant, and the benchmarking pitfalls above.

## Status

Not yet published. `Cargo.toml` carries `publish = false` and no `license`
field: a license has to be chosen before this can go to a registry.
