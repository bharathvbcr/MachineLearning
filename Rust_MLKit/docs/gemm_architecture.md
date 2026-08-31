# tessl GEMM: kernel selection, the coop gate, and how to check it

Covers the TensorOps (MPP `matmul2d`) GEMM path, which lives in `tessl`
(`crates/tessl`) and has exactly one copy.

`tessl-arch02` (`arch_02_value_resid/metal-native`) used to carry a byte-identical
fork of the kernels and of `gemm.rs` / `runtime.rs` / `tensor.rs` / `dispatch.rs`,
kept in step by a static audit. It now depends on `tessl` and compiles tessl's
kernel sources through `DEP_TESSL_KERNELS`, so there is nothing left to drift.

## Kernel selection

Every NN TensorOps shape resolves through one function, `tensorops_nn_kernel`
in `src/gemm.rs`. It is the single evaluation site for the cooperative gate, and
it is `pub(crate)` so tests can ask which path a shape takes rather than assume.

| dtype | gate | kernel |
|---|---|---|
| bf16 | `use_coop_nn(TILE_BF16_NN, …)` | `matmul2d_tensorops_bf16_f32_coop` |
| bf16 | otherwise | `matmul2d_tensorops_bf16_f32` |
| f32 relaxed (`--tf32`) | `use_coop_nn(TILE_F32R_NN, …)` | `matmul2d_tensorops_f32_relaxed_coop` |
| f32 relaxed | otherwise | `matmul2d_tensorops_f32_relaxed` |
| f32 exact | — | `matmul2d_tensorops_f32` |

TN, NT and the accumulating TN/NT paths are selected separately and have no
coop variant — see "Why TN/NT have no coop kernel" below.

## The two kernel shapes

**Blocked** (`matmul2d_tensorops_bf16_f32`, `…_f32_relaxed`): loops over K in
BK=256 blocks, accumulating into a **device-memory** C tile. Block 0 uses
`mode::multiply` so it seeds C and no host-side pre-zero is needed; later blocks
use `multiply_accumulate`. Handles ragged edges and any K.

**Cooperative** (`…_coop`): holds the C accumulator in registers
(`get_destination_cooperative_tensor`) across the whole K loop and stores once,
so C traffic is independent of K. Has **no ragged, short-K or tail branch** — it
trusts the host gate completely.

That difference is the whole point. In the blocked kernel C traffic scales with
K/BK while useful work scales with K, so throughput falls as K grows: measured
at M=N=4096 it peaked at K=2048 and then declined, reaching 0.79x of PyTorch MPS
at K=8192. The coop kernel removes that term.

## `use_coop_nn` — the only guard those kernels have

```rust
m >= tile.sm && m % tile.sm == 0
    && n >= tile.sn && n % tile.sn == 0
    && k >= COOP_MIN_K && k % COOP_BKC == 0
```

Each clause maps to something the kernel cannot do for itself:

- `m >= tile.sm`, `n >= tile.sn` — divisibility alone also admits `m == 0`.
- `% tile == 0` — every tile must be interior; there is no edge path.
- `k % COOP_BKC == 0` — the loop is `for k = 0; k + BKC <= K; k += BKC`. A K
  that is not a whole number of blocks silently drops the tail.
- `k >= COOP_MIN_K` (512) — **structural, not tuned.** The blocked kernels use
  BK=256, so below K=512 they run at most one full block plus a tail, which is
  already a single C store. Measurement agrees: at K=256 coop is 0.90x (bf16)
  and 0.93x (relaxed); it crosses over from K=512.

`COOP_BKC` is 128 for both kernels. Note it is deliberately *smaller* than the
blocked BK=256: 128 divides every K the presets produce (768, 1152, 2304, 3072,
4096) whereas 256 does not divide 1152. Paired measurement showed BK=256 and
BK=128 within noise of each other on the shapes both cover, so the one with
wider coverage wins.

The two NN tiles differ — `TILE_BF16_NN` is 64x64, `TILE_F32R_NN` is 128x64 —
so a shape can be gate-eligible for bf16 and not for relaxed. M=192 is the
canonical example and is covered by both the boundary test and the shape sweep.

## Why TN/NT have no coop kernel

They issue a **single full-K `matmul2d`**, so there is no host-visible C
round-trip to remove. Two controls establish this rather than assuming it:

- An explicit device-C K-blocked control (`mm_tnblk_*`, `mm_ntblk_*`) *regresses*
  to 0.34–0.86x of production.
- Every coop TN/NT variant is **bit-identical** to production (max_rel_err
  exactly `0.00e0`), while the device-C control differs by ~1.2e-6.

Identical bits mean MPP is already accumulating in registers inside its own K
loop; the device-C control differs precisely because it rounds to f32 once per
block. Coop variants measured 0.77–1.10x — no win. Same for the accumulating
TN/NT kernels: 0.88–1.12x against the real accumulate kernel.

(Measured against the *default* build the accumulate coop looked like 1.19–2.74x,
but `METAL_NATIVE_GEMM_ACCUM` defaults off, so that baseline was the
temp-buffer + `add_inplace` fallback. That fallback really is 1.19–2.74x slower
than the accumulate kernel, but enabling it is a training-quality decision
already made on numerics grounds.)

## Checking it

```bash
# Static: every Rust TileGeom vs the kernel's constexpr SM/SN, and every coop
# kernel's constexpr BKC vs Rust's COOP_BKC. Resolves paths from its own
# location, so it runs from any directory and survives `cargo package`. It also
# fails if tessl-arch02 grows a local copy of the kernels again.
python3 Rust_MLKit/crates/tessl/scripts/audit_gemm_tiles.py
```

```bash
# Runtime: hand-picked adversarial shapes + a seeded fuzz that asserts its own
# per-kernel coverage.
cd Rust_MLKit/crates/tessl
cargo test --release --lib -- --test-threads=1 gemm_adversarial_shape_sweep gemm_randomized_shape_fuzz
```

```bash
# Deep soak. GEMM_FUZZ_SEED accepts hex or decimal and PANICS on a malformed
# value rather than falling back to the default — an earlier version silently
# ignored the variable and re-ran one seed eight times, passing every time.
GEMM_FUZZ_SEED=0xdeadbeef GEMM_FUZZ_CASES=1200 \
  cargo test --release --lib -- --test-threads=1 --nocapture gemm_randomized_shape_fuzz
```

The fuzz fails if any selectable NN kernel is chosen for under 1% of cases. That
assertion is load-bearing: the first version of the fuzz passed three injected
coop-kernel faults because independent per-dimension sampling reached the gate
in well under 1% of cases and it never dispatched those kernels at all.

## Benchmarking

The A/B rig (`crates/tessl/kernels/tune/`, 92 measurement-only kernels) is
**not** in the default metallib — linking it takes tessl's artifact from 0.20 MB
to 1.07 MB. It sits in a subdirectory so neither build script's directory glob
can pick it up by accident. Opt in:

```bash
TESSL_GEMM_TUNE=1 cargo build --release --bins
```

Both tuning binaries `exit(2)` with that command when the variants are absent,
rather than printing a page of `skip(pipe)` and exiting 0.

| tool | what it is for |
|---|---|
| `bench_gemm_coop_ab` | Paired, **interleaved** kernel A/B. Use this for kernel comparisons. |
| `bench_gemm_tile_tune` | Broad tile/BK ladder. Blocked-style timing — see the warning below. |
| `bench_gemm_sweep` | Cross-runtime lane (f32 exact / tf32 / bf16), JSON out. |
| `bench/paired_cross_runtime.py` | Alternates the tessl and torch/MLX lanes round by round. |

**Timing protocol matters more than anything else here.** Measuring a baseline
block and a variant block minutes apart puts all GPU clock drift into the ratio:
run four times, `bench_gemm_tile_tune` showed the *production coop kernel against
itself* ranging 0.92x–1.46x. `bench_gemm_coop_ab` interleaves the two arms
iteration by iteration and reports the ratio of per-round medians across repeated
rounds, which brings the spread to about ±5%. It also allocates one output
buffer per shape (a fresh one per candidate let allocations pile up until the
*baseline* drifted 0.268 -> 0.665 ms inside a single shape block) and prints its
own baseline spread across rows, flagging `EXCEEDS 10%: ratios above are not
comparable` when the run cannot support a comparison.

The same applies across runtimes. Two back-to-back runs of the identical
cross-runtime sweep disagreed by 16–21% on the torch lane alone, which is why
`paired_cross_runtime.py` exists and why single-run cross-runtime numbers should
not be quoted.

Below roughly 2 GFLOP of work both runtimes sit on a ~0.25 ms per-GEMM floor
under this submit-and-wait protocol — tessl's wall time is flat from 4 MFLOP to
2416 MFLOP. Ratios there measure dispatch latency, not the kernel.

## Adding a coop kernel

1. Compile-time `SM`/`SN` must equal the `TileGeom` the host dispatches with,
   and `constexpr int BKC` must equal `COOP_BKC` (if using explicit K loop blocking).
2. Pin it in `NN_PAIRS` in `scripts/audit_gemm_tiles.py`. The audit verifies both
   explicit `kernel void` and macro-instantiated (`NN_COOP_KERNEL`, `TN_NT_COOP_KERNEL`)
   tiles against Rust constants.
3. Route it through `nn_coop_kernel` / `src/gemm.rs` so tests and benches exercise it.
4. Verify the net can fail: inject a fault (tile drift, drop store, offset column)
   and confirm the test suite catches it before trusting a pass.

## Round 2 Evolution (2026-08-30)

In Round 2, cooperative destination tensors (`get_destination_cooperative_tensor`)
were expanded across all primary GEMM paths:
- **TN & NT bf16 descriptors:** Landed cooperative destination kernels (`TILE_COOP_TN_NT` 128×64 sg4),
  yielding 1.5–2.0× speedups over single dynamic-K multiply kernels.
- **Accumulate kernels:** Replaced `multiply_accumulate` with cooperative zero→run→load-add-store
  bias pattern (`TILE_COOP_ACCUM` 64×64 sg4), cutting memory traffic to 1 read + 1 write total.
- **NN Grid Swizzle:** Added column-panel swizzling (8 tile-rows per band) for large grids
  (`tiles_n * tiles_m >= 2048`), delivering 29,022 GFLOP/s at $4096^3$ on M5 Pro.
- **Edge Slices:** Ragged boundary tiles execute origin-shifted slices with register accumulation,
  eliminating the need for separate pre-zeroing passes.
