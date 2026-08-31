# bf16 GEMM: why metal-native was ~2× off PyTorch MPS

> **Landed 2026-08-30.** `get_destination_cooperative_tensor` (the "Not yet
> done" item below) beat every BK/tile variant and shipped as the production
> NN kernel for bf16 and relaxed f32, with a three-entry shape selection
> (128×64 sg4 default; 64×64 sg4 for N ≤ 512; 256×64 sg8 for ≥4096² squares
> with K ≥ 2048) — see `bf16_tile_tune_m5pro_coop.txt` and the "Landed
> result" section at the end.

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

## Follow-up (2026-08-30): the "~20% validation cost" attribution was wrong

Re-measured on a HEAD that *includes* the per-call GEMM validation, on an
otherwise idle machine (40 iters, sync/iter): production bf16 medians land at
or above the pristine baseline table — square_2048 10,897 vs 10,086;
square_4096 10,385 vs 10,226; mlp_up 10,732 vs 10,388; mlp_down 11,658 vs
10,503; tall_k1024 10,892 vs 11,153 GFLOP/s. Validation cost does not resolve
above run-to-run variance at these shapes. The ~20% inflation in the original
run came from the concurrent session's compile load on the same machine, not
from the validation it was adding. "Measure on a quiet machine" stands;
"validation is measurably not free" does not.

## Landed result (2026-08-30): cooperative destination closes the gap

`get_destination_cooperative_tensor` keeps the f32 accumulator in registers
for the entire K reduction and writes C exactly once (`cT.store`), which also
retires the host `zero_f32(C)` pre-pass. It beat the best BK/tile variant on
every shape (bf16_tile_tune_m5pro_coop.txt run before landing), needs no BK
parameter at all, and one geometry — 128×64 sg4 — is within 5% of the
per-shape winner everywhere, so the selection table is three entries:

| condition | kernel geometry | evidence |
|---|---|---|
| N ≤ 512 | 64×64 sg4 | narrow_n128/n256: ~6% over 128×64 |
| M,N ≥ 4096 and K ≥ 2048 | 256×64 sg8 | square_4096: +5% over 128×64 |
| otherwise | 128×64 sg4 | best or ≤3% off best on all others |

Production bf16 through the full `gemm()` API (validation included), median
of 40, sync/iter, vs the same-session live torch-MPS lane:

| shape | before | after | torch bf16 | after/torch |
|---|---|---|---|---|
| square_2048 | 10,897 | 22,230 | 20,436 | 1.09× |
| square_4096 | 10,385 | 25,071 | 23,880 | 1.05× |
| mlp_up | 10,732 | 26,197 | 25,605 | 1.02× |
| mlp_down | 11,658 | 26,108 | 22,130 | 1.18× |
| tall_k1024 | 10,892 | 25,103 | 25,555 | 0.98× |

Geomean 1.06× of torch (from ~0.5×), still writing f32 C where torch writes
bf16 C. The relaxed-f32 (tf32) NN kernel took the same rework and moved from
~2.0× to 2.0–2.9× of torch f32 (12.8–18.0 TFLOP/s on the large shapes).
The TN/NT/split-K bf16 kernels are untouched: they were never on the BK
accumulate path (single dynamic-K `multiply` run) and were not implicated in
the measured gap; a coop round for them is possible follow-up, unmeasured.
