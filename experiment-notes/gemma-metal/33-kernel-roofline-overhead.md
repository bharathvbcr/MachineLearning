# 33: Kernel roofline vs per-token overhead

## Executive summary

- **Question:** Isolated Hot-resident Q4 GEMV microbenches would show whether the ~4× gap vs mlx-lm (~76 tok/s) lives in kernel bandwidth or in per-token overhead (~780 dispatches).
- **Result:** ~77% of each quiet decode token is overhead, not GEMV — rewrite kernels ≤~1.3×; fusion and block-verify are the real path to mlx-class tok/s.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `33-kernel-roofline-overhead` |
| Dates | `2026-07-13` – `2026-07-13` |
| Hardware | Apple M5 Pro · ~273 GB/s unified peak |
| Status | `done` |

## Hypothesis

Isolated Hot-resident Q4 GEMV microbenches would show whether the ~4× gap vs mlx-lm (~76 tok/s) lives in kernel bandwidth or in per-token overhead (~780 dispatches).

## Setup

- Trainer / preset: upload-once Hot-bank microbench, 200 iters, dispatch+sync only (not `gemv_quant_host`)
- Fixed knobs: old 1-thread/row vs simd MLX GEMV shapes for E4B and 31B
- Env flags: measured under concurrent GPU contention (absolute GB/s mildly depressed; ratios robust)

## Variants

| Variant | Change |
|---------|--------|
| Old GEMV | one-thread-per-row |
| Simd MLX | `gemv_q4_mlx_simd` |
| Control (false) | `gemv_quant_host` re-upload ~4.9 MB/call |

## Results

| Shape | Old GB/s | Simd GB/s | Ratio |
|-------|----------|-----------|-------|
| mlp_up 10240×2560 | 170–226 | 254–300 | 1.33–1.50× |
| mlp_down 2560×10240 | 213 | 283–287 | 1.33–1.35× |
| lm_head 262144×2560 | 208–211 | 226–229 | 1.09× |
| 31B attn 5376×5376 | 277 | 285–289 | 1.03–1.04× |
| 31B down 5376×10752 | 173–178 | 230–232 | 1.30–1.33× |

Finding: old kernels already **62–100%** of ~273 GB/s peak; simd gains only **~1.03–1.5×**, not 2–4×. Roofline at ~21.5 tok/s (46.5 ms/tok): ~2.86 GB/tok streaming @ ~270 GB/s ≈ **10.6 ms**; remaining **~36 ms (~77%)** is overhead (encode, barriers, casts, ~780 dispatches). mlx 76 tok/s ≈ 13.2 ms/tok = same ~10.6 ms stream + **~2.6 ms** overhead. Effective levers: fewer dispatches / megakernel; block speculative verify; encode/GPU overlap.

**Interpretation boundary.** Kernel bandwidth is measured; the ~77% overhead share is a calculation from measured bytes/tok and time/tok. Its exact percentage inherits those model assumptions.

## Failures

- Prior bottleneck doc body claimed ~20% peak / 4× GEMV headroom — **superseded** (measured upload BW, not kernel BW)
- Concurrent contention depresses absolute GB/s slightly

## Lesson

**~77% of each quiet decode token is overhead, not GEMV — rewrite kernels ≤~1.3×; fusion and block-verify are the real path to mlx-class tok/s.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: the Hot-bank microbench implementation in `Rust_MLKit/gemma-metal`.
- Required replay inputs: `bench/results/kernel_roofline_finding.json` for shapes, iterations, and measured inputs. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Direct microbench bandwidth and an explicit roofline decomposition support the result; concurrent contention and model assumptions limit precision.

## Artifacts

- `Rust_MLKit/gemma-metal/bench/results/kernel_roofline_finding.json`
- `Rust_MLKit/gemma-metal/docs/bottleneck.md` (CORRECTION header)
- `Rust_MLKit/gemma-metal/docs/speed_findings.md`

## Why this experiment happened

The native plateau admitted two incompatible explanations: weak GEMV kernels or excessive orchestration. Direct Hot-resident bandwidth measurements were needed because an earlier document had incorrectly inferred about 4× GEMV headroom from upload bandwidth. The preceding notebook context is [32-native-fusion-2026-07-14](32-native-fusion-2026-07-14.md).

## Experiment story

**Baseline.** The native plateau admitted two incompatible explanations: weak GEMV kernels or excessive orchestration. Direct Hot-resident bandwidth measurements were needed because an earlier document had incorrectly inferred about 4× GEMV headroom from upload bandwidth. The preceding notebook context is [32-native-fusion-2026-07-14](32-native-fusion-2026-07-14.md).

**Hypothesis.** Isolated Hot-resident Q4 GEMV microbenches would show whether the ~4× gap vs mlx-lm (~76 tok/s) lives in kernel bandwidth or in per-token overhead (~780 dispatches).

**Test contract.** Trainer / preset: upload-once Hot-bank microbench, 200 iters, dispatch+sync only (not `gemv_quant_host`) Fixed knobs: old 1-thread/row vs simd MLX GEMV shapes for E4B and 31B Env flags: measured under concurrent GPU contention (absolute GB/s mildly depressed; ratios robust)

**Variant sequence.** The preserved comparison matrix was: Old GEMV — one-thread-per-row; Simd MLX — `gemv_q4_mlx_simd`; Control (false) — `gemv_quant_host` re-upload ~4.9 MB/call.

**Measured turn.** The result board records mlp_up 10240×2560 — 170–226 — 254–300 — 1.33–1.50×; mlp_down 2560×10240 — 213 — 283–287 — 1.33–1.35×; lm_head 262144×2560 — 208–211 — 226–229 — 1.09×; 31B attn 5376×5376 — 277 — 285–289 — 1.03–1.04×; 31B down 5376×10752 — 173–178 — 230–232 — 1.30–1.33×.

**Turning point and readout.** Finding: old kernels already **62–100%** of ~273 GB/s peak; simd gains only **~1.03–1.5×**, not 2–4×. Roofline at ~21.5 tok/s (46.5 ms/tok): ~2.86 GB/tok streaming @ ~270 GB/s ≈ **10.6 ms**; remaining **~36 ms (~77%)** is overhead (encode, barriers, casts, ~780 dispatches). mlx 76 tok/s ≈ 13.2 ms/tok = same ~10.6 ms stream + **~2.6 ms** overhead. Effective levers: fewer dispatches / megakernel; block speculative verify; encode/GPU overlap. **Interpretation boundary.** Kernel bandwidth is measured; the ~77% overhead share is a calculation from measured bytes/tok and time/tok. Its exact percentage inherits those model assumptions.

**Failures and surprises.** Prior bottleneck doc body claimed ~20% peak / 4× GEMV headroom — **superseded** (measured upload BW, not kernel BW) Concurrent contention depresses absolute GB/s slightly

## Decision and aftermath

**Kept:** ~77% of each quiet decode token is overhead, not GEMV — rewrite kernels ≤~1.3×; fusion and block-verify are the real path to mlx-class tok/s. The notebook continues with [34-mlx-dflash-product](34-mlx-dflash-product.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: mlp_up 10240×2560 — 170–226 — 254–300 — 1.33–1.50×.
- The result artifact reports: mlp_down 2560×10240 — 213 — 283–287 — 1.33–1.35×.
- The result artifact reports: lm_head 262144×2560 — 208–211 — 226–229 — 1.09×.
- The result artifact reports: 31B attn 5376×5376 — 277 — 285–289 — 1.03–1.04×.
- Failure/operational record: Prior bottleneck doc body claimed ~20% peak / 4× GEMV headroom — **superseded** (measured upload BW, not kernel BW)
- Failure/operational record: Concurrent contention depresses absolute GB/s slightly

## What this does not prove

**Confidence: Medium.** Direct microbench bandwidth and an explicit roofline decomposition support the result; concurrent contention and model assumptions limit precision. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- Related suites: [`31-native-decode-speed-ladder`](31-native-decode-speed-ladder.md), [`34-mlx-dflash-product`](34-mlx-dflash-product.md), [`40-ddtree-frontier`](40-ddtree-frontier.md)

---

[Previous](32-native-fusion-2026-07-14.md) · [Index](../00-INDEX.md) · [Next](34-mlx-dflash-product.md)
