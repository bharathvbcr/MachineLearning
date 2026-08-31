# tessl GEMM: kernel selection, the coop gate, and how to check it

Covers the TensorOps (MPP `matmul2d`) GEMM path shared by `tessl`
(`crates/tessl`) and `tessl-arch02` (`arch_02_value_resid/metal-native`).
Both crates carry a mirrored copy; they must stay identical, and
`scripts/audit_gemm_tiles.py` checks that mechanically.

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
# Static: every Rust TileGeom vs the kernel's constexpr SM/SN, every coop
# kernel's constexpr BKC vs Rust's COOP_BKC, and both crates in sync.
# Runs from any directory; fails loudly if a crate path is missing.
python3 Rust_MLKit/arch_02_value_resid/metal-native/scripts/audit_gemm_tiles.py
```

```bash
# Runtime: hand-picked adversarial shapes + a seeded fuzz that asserts its own
# per-kernel coverage.
cd Rust_MLKit/arch_02_value_resid/metal-native
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

The A/B rig (`kernels/matmul_tensorops_tune.metal`, 92 measurement-only kernels)
is **not** in the default metallib — linking it took the artifact from 0.87 MB to
1.74 MB. Opt in:

```bash
METAL_NATIVE_GEMM_TUNE=1 cargo build --release --bins
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
   and `constexpr int BKC` must equal `COOP_BKC`.
2. Pin it in `NN_PAIRS` in `audit_gemm_tiles.py`. The audit fails on any
   `*_coop` kernel that is not pinned, because coop kernels are dispatched
   through a variable and the pipeline-literal scanner cannot see them.
3. Route it through `tensorops_nn_kernel` so the fuzz's coverage assertion
   counts it.
4. Verify the net can fail: inject a fault (BKC drift, tile drift, drop the
   store, seed the accumulator non-zero, skip every other K block, offset a
   column) and confirm the sweep or fuzz catches it before trusting a pass.
