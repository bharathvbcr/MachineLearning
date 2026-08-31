# bf16 GEMM: why metal-native was ~2× off PyTorch MPS

M5 Pro / macOS 27.0. Measured from a **pristine HEAD worktree** — a concurrent
session was editing `src/gemm.rs` mid-run and its added per-call validation
inflated the production baseline by ~20%.

## Root cause

`matmul2d_tensorops_bf16_f32` accumulates into a **device-memory** C tile on
every K block:

    for (k += BK) { op_bk.run(tA, tB, tC); }   // tC is a device pointer

At 4096³/BK=128 that is 32 read-modify-write passes over a 67 MB C tile.
`staticThreadgroupMemoryLength` is 0 for every variant, so nothing is staged;
maxTPTG is 1024, so it is not register- or occupancy-limited.

Confirmed by a BK ladder at fixed 64×64 tile (square_4096) — time tracks the
number of K blocks, independent of tile size:

| BK | K-blocks | GFLOP/s |
|----|---------|---------|
| 32 | 128 | 9,376 |
| 64 | 64 | 13,979 |
| 128 | 32 | 15,596 |
| 256 | 16 | 16,360 |
| 512 | 8 | 16,810 |

Three additive factors: the wasted `zero_f32(C)` pre-pass (only needed because
block 0 uses multiply_accumulate rather than multiply), the 64×32 tile, and BK.
Tile and BK interact — every large-tile variant tested at small BK looked bad
because it was still paying full accumulate traffic.

## Result (bf16 GFLOP/s, median of 40, sync/iter)

| shape | production | tuned | torch | tuned/prod | tuned/torch |
|---|---|---|---|---|---|
| square_2048 | 10,086 | 19,610 | 17,446 | 1.94× | 1.12× |
| square_4096 | 10,226 | 20,271 | 21,648 | 1.98× | 0.94× |
| mlp_up | 10,388 | 22,435 | 24,426 | 2.16× | 0.92× |
| mlp_down | 10,503 | 19,990 | 20,215 | 1.90× | 0.99× |
| tall_k1024 | 11,153 | 22,613 | 22,876 | 2.03× | 0.99× |
| **geomean** | | | | **2.00×** | **0.99×** |

Best geometry is shape-dependent (64×64 and 128×128 each win two shapes), so
landing this needs a small selection table, not one constant.

## Not yet done

Variants are interior-only (exact divisibility) measurement kernels; the
production kernel's ragged-edge paths are untouched. `get_destination_cooperative_tensor`
(register-resident accumulator, per Apple's MPPTensorOpsMatMul2d.h) was not
tested and should remove the remaining C traffic entirely.

---

# f32 relaxed-precision (tf32-class) — same defect, different optimum

`matmul2d_tensorops_f32_relaxed` had the identical structure: BK loop accumulating
into device-memory C from block 0, requiring the host `zero_f32(C)` pre-pass.

The bf16 optimum did **not** transfer. f32 operands are 2× the bytes, so the same
tile carries half the arithmetic intensity — the f32 lane wants a wider tile
(128×64) where bf16 wanted 64×64.

Candidates over 5 shapes (two 60-iter runs, ratios vs pre-fix production):

| variant | geomean | worst shape |
|---|---|---|
| **128×64 BK256 sg4** | **1.67×** | **1.51×** |
| 256×64 BK256 sg8 | 1.65× | 1.46× |
| 128×64 BK256 sg8 | 1.66× | 0.82× |
| 128×32 BK256 sg4 | 1.68× | 0.85× |

Geomeans are within noise of each other, but the sg8 variants reproducibly
regress `mlp_down` (N=768) below baseline. Picked 128×64/BK256/**sg4** for the
worst case, not the mean.

Landed: SM 64→128, SN 32→64, BK 128→256, block 0 `mode::multiply`.

| shape | before | after | speedup |
|---|---|---|---|
| square_2048 | 7,650 | 11,992 | 1.57× |
| square_4096 | 5,496 | 8,275 | 1.51× |
| mlp_up | 7,334 | 13,932 | 1.90× |
| mlp_down | 7,377 | 12,818 | 1.74× |
| tall_k1024 | 7,228 | 12,613 | 1.74× |
| **geomean** | | | **1.69×** |

Control check: the `mm_f32r_64x32_bk128_sg4_accf` variant (old geometry, old
accumulate-first) still reproduces the pre-fix production numbers to within 1%,
confirming the A/B is measuring the change and not drift.

## Still not done

TN/NT/split-K bf16 kernels remain 64×32 with no BK blocking. Per-shape tile
selection was not encoded — the single constants above were chosen for
worst-case safety over a 5-shape sample.

---

# Closing the remaining gaps: TN/NT, accumulate, and the overfit check

## Correction to the earlier framing

The plain TN/NT kernels do **not** have the C round-trip defect — they are
single-shot `mode::multiply` over the full K, so there is no BK loop and no
accumulate-from-block-0. What they did have was a host `zero_f32(C)` pre-pass
that is pure dead work for a kernel that overwrites, plus an untuned 64×32 tile.

## Shapes

Re-derived from the real presets instead of invented squares. BT = 16*256 = 4096
for every preset; (d, mlp) pairs are sota_toy(128,384), medium_16m(384,1152),
arch02_128m(768,2304). TN dW is (d|mlp, d|mlp, 4096) — small M/N with huge K, a
regime the original 8-shape sweep never covered.

## Results (bf16, geomean over 5 real shapes each)

| path | tile | before → after | geomean | worst |
|---|---|---|---|---|
| plain TN | 64×32 → **128×64 sg4** | 7.3–11.6k → 11.3–19.6k GFLOP/s | **1.81×** | 1.55× |
| plain NT | 64×32 → **128×64 sg4** | 4.0–10.8k → 4.7–21.4k GFLOP/s | **1.73×** | 1.17× |

TN-accum and NT-accum were measured separately and peak at **different** tiles —
TN-accum 128×64 (geomean 1.37, worst 1.05), NT-accum 64×64 (geomean 2.00, worst
1.49). Each regresses below baseline on the other's choice, so they got separate
constants. These only take effect under `METAL_NATIVE_GEMM_ACCUM=1`, which is
**off by default for numerics** (optimization_map.md: restores late gnorm,
partial BPB). That default was not touched — it is a quality decision, not a
perf one.

## The overfit check

The single constants were re-validated against the narrow-N regime absent from
the original sweep (N=128, the sota_toy preset). `fwd_attn_128` measured 1.07,
0.81, 1.16, 0.85 across four runs — mean ~0.97 with ±18% spread at 0.19–0.23 ms,
where launch overhead dominates. **That is noise, not a regression**, so no
per-shape selection table was added; fitting one here would be fitting noise.

Separately, production's first-measured shape reads high (0.634 → 0.503 → 0.404
ms across runs while the rig variant sits at 0.372/0.368/0.370) — warmup in the
production lane, not a dispatch cost.

## Genuinely still untouched

- **split-K TN bf16** — a different algorithm (partitioned K), and
  `prefer_tn_splitk` only fires for sota_toy (needs m,n ≤ 384 and min ≤ 128);
  arch02_128m and medium_16m never reach it.
- **f32-exact NN/TN/NT tiles** (32×32) — untuned. The f32-exact path already
  measured at parity with MLX and PyTorch MPS, i.e. at the hardware roof.
- **f32 accumulate kernels** — untuned, and behind the same off-by-default flag.

---

# Audit, hardening, and cross-runtime comparison

## Audit result: no defects found in the GEMM tile wiring

`scripts/audit_gemm_tiles.py` cross-checks every Rust `TileGeom` against the
`constexpr SM/SN` and `execution_simdgroups<N>` compiled into the kernel it
dispatches. A desync makes the host launch the wrong threadgroup count and
silently leaves output tiles unwritten — the highest-severity failure mode the
retiling work introduced, and one nothing previously enforced.

**15 pairs across both crates: 0 mismatches.**

The script's first run reported 2 mismatches on `matmul2d_tensorops_tn_splitk_f32`.
Both were false positives: the regex matched `execution_simdgroups<4>` inside a
*comment* documenting the next kernel. The script now strips comments before
matching. The kernel genuinely uses `execution_simdgroup` (1), matching TILE_F32.

## Two independent guards, both proven to fire

1. **Static** — the audit script above (CI-able, no GPU needed).
2. **Runtime** — `gemm_adversarial_shape_sweep`: 135 (path, shape) combinations
   over 9 dispatch paths × 15 shapes, each against an f64 CPU reference, with
   every output buffer pre-seeded to `1e30` so an unwritten tile is caught rather
   than read as a plausible number. Accumulating paths are seeded with a known
   addend and checked for `C0 + A@B`.

Shapes deliberately break tiling assumptions: degenerate (1×1×1), primes (3×5×7),
one-off-tile-boundary (63/65/127/129/257), exact multiples, extreme aspect
(65×512×33), and the split-K trigger region.

Fault injection (TILE_BF16_NN sn 64→128, kernel left at 64) is caught by **both**:
the sweep fails with `C[192] = 1e30 — tile never written`, the audit with
`MISMATCH ... kernel=64x64 rust=64x128`.

Stability: 10 consecutive sweep runs, 0 failures. Full suites: metal-native 97,
metal-runtime 65, gemma-metal 134.

## Cross-runtime, same session (M5 Pro, GFLOP/s)

bf16:

| shape | metal-native | torch MPS | MLX | Burn (raw) |
|---|---|---|---|---|
| square_2048 | 20,574 | 18,184 | 6,689 | 3,185 |
| square_4096 | 20,741 | 22,146 | 6,996 | 5,970 |
| mlp_up | 22,219 | 23,882 | 6,940 | 4,869 |
| mlp_down | 20,116 | 21,258 | 7,056 | 4,869 |

f32 is a four-way tie at 6.2–6.8k for everyone — the hardware roof.

**Burn caveat:** measured single-op eager, which is not Burn's design point —
`fusion` amortizes across a graph, and one matmul per sync defeats it. Burn shows
a **fixed 2.85 ms per-op cost**: a 16³ matmul (8 KFLOP) takes 2.845 ms, the same
as 512³ at 2.884 ms, against a 0.024 ms bare sync. Verified not to be lazy-operand
recomputation (materialized operands: 2.822 / 2.852 ms). Subtracting that constant
puts Burn's *kernels* at ~6.6–7.5k f32 — at the roof with everyone else. Burn's
bf16 lane is identical to its f32 (23.021 vs 23.011 ms at 4096³), i.e. it is not
reaching the bf16 accelerator path.

**candle was not measured.** It appears only in burn-port's lockfile via the
unused `burn-candle` feature; benchmarking it needs a new dependency, which was
not added.

## candle lane (added)

`candle-core` 0.10.2 with the `metal` feature, added as a **dev-dependency** of
burn-port (already resolved in that lockfile via `burn-candle`, so nothing new
was pulled). Lane lives at `burn-port/examples/candle_matmul.rs` — an example
rather than a bin, because bins cannot use dev-dependencies.

Numerics verified before timing (256³ f32 vs an f64 reference: max abs err
0.000e0). Per-op floor measured the same way as Burn: **0.197 ms** at 16³ vs
0.363 ms at 512³ — low overhead, unlike Burn's 2.85 ms.

**candle's bf16 matches its own f32** (7,058 vs 6,712 GFLOP/s at 4096³), so like
MLX and Burn it is not reaching the M5 bf16 accelerator path.

## Final standing (bf16, GFLOP/s, same session)

| shape | metal-native | torch MPS | candle | MLX | Burn (raw) |
|---|---|---|---|---|---|
| square_2048 | 20,363 | 18,184 | 6,855 | 6,689 | 3,185 |
| square_4096 | 19,721 | 22,146 | 7,058 | 6,996 | 5,970 |
| mlp_up | 21,734 | 23,882 | 6,808 | 6,940 | 4,869 |
| mlp_down | 19,523 | 21,258 | 7,097 | 7,056 | 4,869 |

geomean vs metal-native: torch **0.96×** (torch ahead), candle **2.92×**,
MLX **2.94×**, Burn ~2.9× kernel-adjusted.

f32 is a five-way tie at 6.1–6.8k — the hardware roof.

The bf16 lead over candle/MLX/Burn is **not** a code-quality gap: those three run
bf16 through their f32 path. It is a real advantage today and a fragile one —
any of them closes it in a single release by wiring up the same primitive.

---

# Closing the PyTorch MPS gap: register-resident accumulator

## Diagnosis

Sweeping K alone at fixed M=N=4096 localised the gap precisely — it is a pure
function of K, monotonic, and it crosses over:

| K | before | torch | ratio |
|---|---|---|---|
| 256 | 14,876 | 11,322 | 1.31x |
| 512 | 21,497 | 16,132 | 1.33x |
| 1024 | 23,347 | 24,748 | 0.94x |
| 2048 | 24,398 | 27,566 | 0.89x |
| 4096 | 21,147 | 25,395 | 0.83x |
| 8192 | 18,666 | 23,746 | 0.79x |

metal-native *won* at small K and lost progressively as K grew. Own throughput
peaked at K=2048 then declined — which a compute-bound kernel should not do.
With BK=256 those K values are 1, 2, 4, 8, 16, 32 accumulator round-trips
through device memory.

## Fix

`matmul2d_tensorops_bf16_f32_coop`: a separate kernel holding the C accumulator
in registers via `get_destination_cooperative_tensor` across the whole K loop,
storing once. C traffic becomes independent of K.

Gated host-side (`use_coop_bf16_nn`) on M%64==0, N%64==0, K>=512, K%128==0.
BKC=128 divides every K the presets produce (768/1152/2304/3072/4096). Below
COOP_MIN_K=512 the blocked kernel measured *faster* (0.90x at K=256), and that is
also where metal-native already beat torch by 1.3x — so the gate is a measured
boundary, not a guess.

Note the header's usage example is stale: the real API needs the `template`
keyword on the dependent call, `is_valid_element(i)` rather than `get_mask(i)`,
and plain `#pragma unroll`.

## Result

| K | before | after | torch | ratio |
|---|---|---|---|---|
| 256 | 14,876 | 15,059 | 11,322 | **1.33x** |
| 512 | 21,497 | 21,394 | 16,132 | **1.33x** |
| 1024 | 23,347 | 24,821 | 24,748 | 1.00x |
| 2048 | 24,398 | 27,154 | 27,566 | 0.99x |
| 4096 | 21,147 | **26,679** | 25,395 | **1.05x** |
| 8192 | 18,666 | **23,182** | 23,746 | 0.98x |

Standard 8-shape ladder, fresh same-session torch: geomean **1.07x** (was 0.96x).

## Two measurement errors worth recording

1. **The tune rig used row-major tile traversal while production uses Morton
   order** on square power-of-two tile grids. Every variant was therefore
   penalised on square shapes — the control read 0.71–0.88x of production
   despite being identical geometry. The rig now mirrors `tile_from_linear`.
   Earlier square-shape ladder numbers in this document understate the variants.
2. **`cargo test --lib` rebuilds the library but not the bins.** Two rounds of
   "the fix isn't working" (20,012 and 20,496 GFLOP/s) were a stale
   `bench_gemm_sweep` binary. The conclusion drawn from them — that in-kernel
   register pressure was costing 25% — was never actually tested. The split into
   a separate kernel stands on design merit (host-side gate, no dead branches in
   the hot path), not on that evidence.

---

# Round 2: extending the accumulator fix, and a correction

Goal: apply the register-accumulator fix to the remaining layouts
(f32-relaxed, TN, NT, TN/NT-accumulate), then find and close whatever gap
versus PyTorch MPS remained.

## The headline correction

**The "geomean 1.07x vs torch" above does not survive a paired measurement.**

That number came from running the Rust sweep once and the Python sweep once, in
separate processes, minutes apart. Two back-to-back runs of that identical
benchmark disagreed by 16–21% *on the torch lane alone* — larger than the
effect being reported. `bench/paired_cross_runtime.py` now alternates the two
lanes round by round (and alternates which goes first) so drift cancels.
Under that protocol, over 5 rounds:

| comparison | geomean of per-shape medians | worst shape | best shape |
|---|---|---|---|
| bf16 vs torch MPS bf16 | **1.00x** | 0.84x | 1.12x |
| f32-exact vs torch MPS f32 | 1.07x | 0.92x | 1.47x |
| tf32-relaxed vs torch MPS f32 | 1.98x | 1.49x | 2.34x |
| bf16 vs MLX bf16 | 2.63x | 1.13x | 3.64x |

So bf16 is at **parity** with PyTorch MPS, not ahead of it. The K-scaling gap
found in round 1 was real and is closed; what remains is a few percent either
way that flips shape to shape. Both runtimes plateau around 25 TFLOP/s on the
large shapes, which is consistent with a shared hardware ceiling rather than a
tiling difference in either.

Two caveats on that table. The tf32 row is not like-for-like — relaxed
precision truncates the mantissa, so it should be read as "what the tf32 opt-in
buys", not as an f32 result. And MLX bf16 measures within noise of MLX f32 on
this machine (~6.5 vs ~6.7 TFLOP/s), which suggests MLX is not reaching the
neural accelerators for matmul here; the 2.63x is reported as measured, not as
a considered claim about MLX.

## Where the accumulator fix does and does not apply

The round-1 diagnosis generalises to a rule: **the fix helps exactly where a
kernel runs an explicit BK loop that reads and writes C in device memory.**

| path | structure before | coop result | shipped |
|---|---|---|---|
| NN bf16 | explicit BK=256 loop into device C | 1.02–1.26x | yes (round 1) |
| NN f32-relaxed | explicit BK=256 loop into device C | 1.05–1.13x | **yes (this round)** |
| TN bf16 | one full-K `matmul2d` | 0.84–1.06x | no |
| NT bf16 | one full-K `matmul2d` | 0.77–1.10x | no |
| TN/NT accumulate | one full-K `matmul2d` | 0.88–1.12x | no |

TN and NT were the interesting negative. They issue a single `matmul2d` over
the whole K, and the controls explain why coop cannot help: an explicit
device-C K loop (`mm_tnblk_*`) *regresses* to 0.34–0.86x, and every coop
variant is **bit-identical** to production (max_rel_err exactly 0.00e0) while
the device-C control differs by ~1.2e-6. Identical bits mean MPP is already
accumulating in registers internally across its own K loop; the device-C
control differs precisely because it rounds to f32 once per block. There is no
C round-trip to remove, so there is nothing to fix.

The accumulate lanes needed one correction of their own. Measured against the
*default* build, coop looked like a 1.19–2.74x win — but `METAL_NATIVE_GEMM_ACCUM`
defaults off, so that baseline was the temp-buffer + `add_inplace` fallback,
not the accumulate kernel. Re-measured with the flag on, coop is 0.88–1.12x:
noise. (The 1.19–2.74x gap between the fallback and the accumulate kernel is
real and reproducible, but turning that flag on is a training-quality decision
that was already made on numerics grounds, not a performance one. Left alone.)

## What shipped

`matmul2d_tensorops_f32_relaxed_coop` — 128x64 tile, BKC=128, register-resident
accumulator, gated by the same predicate as the bf16 kernel. The two gates are
now one function, `use_coop_nn(tile, m, n, k)`, and kernel selection for every
NN TensorOps shape is one function, `tensorops_nn_kernel`, so the gate has a
single evaluation site that tests can call directly.

`COOP_MIN_K = 512` is structural, not tuned: both blocked kernels use BK=256,
so below K=512 they run at most one full block plus a tail — already a single C
store. Measurement agrees, at K=256 coop is 0.90x (bf16) and 0.93x (relaxed),
crossing over from K=512.

Relaxed precision is opt-in (`--tf32`), so this improves that path, not the
default f32 path.

## Measurement errors found this round

Three more, in addition to the two recorded above.

3. **The tile-tune rig's noise floor was wider than every effect in it.**
   It times a baseline block and a variant block minutes apart, so drift lands
   entirely in the ratio. Running it four times showed the *production coop
   kernel measured against itself* ranging 0.92x–1.46x. `bench_gemm_coop_ab`
   now interleaves baseline and candidate iteration by iteration and reports
   the ratio of per-round medians across repeated rounds; that brought the
   spread to roughly ±5%. Round-1 tile conclusions drawn from single runs of
   the tune rig should be treated as provisional.

4. **The A/B harness contaminated its own baseline.** It allocated a fresh
   output tensor per candidate; those piled up across the run and later rows
   measured a degraded machine. With nine candidates the *baseline kernel*
   drifted 0.268 -> 0.665 ms within one shape block — which reads as a
   candidate win. Fixed by allocating once per shape. The harness now also
   reports the baseline's own spread across rows and prints
   `EXCEEDS 10%: ratios above are not comparable` when it is too wide, because
   this failure is invisible otherwise. One shape (2048x768x768) still trips
   that guard and its numbers are therefore not used.

5. **`GEMM_FUZZ_SEED` was silently ignored.** `"0xdeadbeef".parse::<u64>()`
   fails, and the code fell back to the default seed, so a soak across eight
   seeds re-ran one seed eight times and reported success each time. Env
   parsing now accepts hex and **panics** on a malformed value rather than
   falling back — a check that could not run must not report the same result as
   a check that ran and passed.

## Hardening

- **`use_coop_nn` now requires `m >= tile.sm` and `n >= tile.sn`**, not just
  divisibility. The boundary unit test found that `m % 64 == 0` also admits
  `m == 0`, and this predicate is the only guard those kernels have.
- **The static audit covered neither `*_coop` kernel** — they are dispatched
  through a variable, never `pipeline("literal")`, so the pair-scanner never
  saw them. Both are now pinned, the audit cross-checks each kernel's
  `constexpr int BKC` against Rust's `COOP_BKC` (a drift there would let the
  host admit K values whose tail the kernel's `k + BKC <= K` loop drops), and it
  fails on any future `*_coop` kernel that is not pinned. Verified by injecting
  all three faults. It also resolves paths from its own location, so it runs
  from any directory.
- **`gemm_randomized_shape_fuzz`**: deterministic, seeded shape fuzz over the
  NN paths, sentinel-seeded C, f64 reference. One case in three is built to
  satisfy the coop gate by construction — with independent per-dimension
  sampling the gate needs M, N and K to align simultaneously and was reached in
  under 1% of cases. It **asserts its own coverage**: every kernel `gemm` can
  select must be chosen for at least 1% of cases or the run fails.
  This mattered — the first version of the fuzz passed all three injected coop
  faults because it never dispatched those kernels.
- **The A/B rig is no longer in the shipped metallib.** 92 measurement-only
  kernels were being linked into the default artifact every process loads,
  taking it from 0.87 MB to 1.74 MB. Now behind `METAL_NATIVE_GEMM_TUNE=1`, and
  both tuning binaries exit(2) with the rebuild command when the variants are
  absent rather than printing a page of `skip(pipe)` and exiting 0.

## Verification

- Static audit: PASS, 0 mismatches, both crates; fires on 3 injected faults.
- Fuzz soak: 10 seeds x 1200 cases x 3 paths = 36,000 (path, shape)
  combinations, ~7,980 of them dispatching a coop kernel. All pass.
- Fault injection into the coop kernels — BKC drift, tile drift, store removed,
  accumulator seeded non-zero, every other K block skipped, column offset by
  one: 6 of 6 caught.
- Suites: metal-native 99, metal-runtime 67, gemma-metal 134. All pass.
- The two crates' coop logic (constants, gate, selection) is byte-identical.

## Still open

- Every constant here is tuned on one M5 Pro. Nothing has been checked on
  another machine.
- Wider output tiles (128x64, 64x128, 128x128, 256x64) were tested against the
  shipped 64x64 on the large-N shapes where torch leads. All within the
  baseline's own 3–4% spread; the hypothesis that tile width explains that
  shortfall is refuted, and no tile change is justified.
- Both runtimes hit a ~0.25 ms floor per GEMM under this submit-and-wait
  protocol: tessl's wall time is flat from 4 MFLOP to 2416 MFLOP. Below roughly
  2 GFLOP of work the sweep measures dispatch latency, not the kernel, and
  ratios there (including bf16 `square_512` at 0.84x) should not be read as
  throughput.
