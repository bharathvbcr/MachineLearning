# 50: arch_02 Metal-native training

## Executive summary

- **Question:** A from-scratch Rust + Metal 4 training path can preserve the `arch_02` math while making M5-native TensorOps GEMM, packed command encoding, fused backward kernels, and explicit residency materially faster than the Burn reference. Quality must be judged separately from the step-time gate: the fast row-wise flash path and the slower tiled flash path can have different Soft-clipping dynamics and final BPB.
- **Result:** Keep speed and quality gates separate: row FA sustains the 56.6 ms default, while FA_TILED plus horizon-appropriate warmdown delivers the best measured Soft EMA (1.8969 at 20k; 1.8828 at 100k).
- **Implication:** Do not promote this beyond the completed stages; the missing or failed confirmation is decision-relevant.
- **Status:** `partial`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `50-arch02-metal-native-train` |
| Dates | `2026-07-11` – `2026-07-13` |
| Hardware | Apple M5 Pro, 20-core GPU, 64 GB unified memory, macOS 26.5.2 |
| Status | `partial` |

## Hypothesis

A from-scratch Rust + Metal 4 training path can preserve the `arch_02` math while making M5-native TensorOps GEMM, packed command encoding, fused backward kernels, and explicit residency materially faster than the Burn reference. Quality must be judged separately from the step-time gate: the fast row-wise flash path and the slower tiled flash path can have different Soft-clipping dynamics and final BPB.

## Setup

- Trainer / preset: `Rust_MLKit/arch_02_value_resid/metal-native`, `train --f32`, default `sota` preset, FineWeb, seed 1337, B=16 × T=256 (4096 tokens/step).
- Fixed knobs: Metal 4-only encode; MPP TensorOps primary GEMM; default Soft-split clipping (Muon × √clip coefficient, AdamW × clip coefficient); `METAL_NATIVE_GEMM_ACCUM` off after the Soft regression bisect.
- Env flags: default speed gate uses row-wise FA; quality runs set `METAL_NATIVE_FA_TILED=1`. The validated 20k arm uses `--warmdown 3500`; the 100k arm uses `--warmdown-start 16000 --warmdown 24000 --lr-floor 0.1 --final-warmdown 10000`.

## Variants

| Variant | Change |
|---------|--------|
| Soft-harden default | Row FA, GEMM accumulate off |
| Speed A/B | `METAL_NATIVE_GEMM_ACCUM=1` |
| FA_TILED quality | Tiled FA backward, GEMM accumulate off |
| TensorOps flash probe | `--flash-tensorops`, GEMM accumulate off |
| 20k Soft recipe | FA_TILED + Soft-split + 3500-step final warmdown |
| 100k Soft WSD | FA_TILED + staged warmdown/hold/final-warmdown schedule |
| 16M scale-up | `--preset 16m`, FA_TILED, B=16 × T=256 |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | 100k Soft WSD, seed 1337 | FINAL EMA BPB **1.8828** | Best live 1.8819 @96999; ~5.6 h, ~60k tok/s |
| 2 | 20k Soft + warmdown, seed 42 reseed | FINAL EMA BPB **1.8876** | gnorm 60 @19999 |
| 3 | 20k Soft + warmdown, seed 1337 golden | FINAL EMA BPB **1.8969** | gnorm 216 @19999; ~24–27 min |
| 4 | 3k Soft-split + FA_TILED | FINAL EMA BPB **2.0222** | Best documented 3k Soft-split quality result |
| 5 | 3k Soft-everywhere + FA_TILED | FINAL EMA BPB **2.0369** | ~69 ms/step; quality opt-in |
| 6 | 3k Soft-harden default | FINAL EMA BPB **2.0502** | ~56.6 ms/step, ~72k tok/s, 250 binders |

The live default gate is **56.6 ms/step** and **~72k tok/s** at B=16. Enabling GEMM accumulate reduces the binder count from 250 to 211 and measures about **55–56.5 ms/step**, but its 3k Soft EMA regresses to **2.0580**, so it is not the default. FA_TILED costs roughly **69 ms/step / ~60k tok/s** but improves the documented 3k quality results; at longer horizons, explicit scheduling produces FINAL EMA **1.8969** at 20k and **1.8828** at 100k.

The first `--preset 16m` verification is not yet a long-run result: FA_TILED B=16/T=256 benchmarks at **~954 ms/step / ~4292 tok/s**, and its 1k Soft check ends at FINAL EMA BPB **2.1462**. The planned 16M 20k run therefore remains open.

**Interpretation boundary.** Step time/throughput and BPB come from different flash paths, so the fastest row-FA result is not the best-quality result. The 100k BPB is measured at one seed; cross-hardware parity remains open.

## Failures

- Audit 6 `multiply_accumulate` improved speed/binders but regressed 3k Soft EMA to **2.0580** with late gnorm ~9; `METAL_NATIVE_GEMM_ACCUM` is off by default.
- `METAL_NATIVE_HAZARD_BARRIERS=1` measured ~53.5 ms but produced NaN loss by step 3; strict barriers remain the default.
- `--flash-tensorops` reached EMA **2.0462** but late gnorm ~13 @2999; it was rejected for the Soft ladder.
- Soft-everywhere FA_TILED diverged after roughly 3.5–4.5k and ended at EMA **2.2575** on the 20k attempt.
- A 100k last-10%-only warmdown never reached warmdown: best live BPB was **1.9137 @15999**, followed by rebound to ~1.96–1.97; it stopped near 53.7k.
- Full TensorOps flash remains blocked by the lack of a production multi-block backward+LSE path and by unresolved 3k quality parity versus the CUDA reference (**1.9944**).

## Lesson

**Keep speed and quality gates separate: row FA sustains the 56.6 ms default, while FA_TILED plus horizon-appropriate warmdown delivers the best measured Soft EMA (1.8969 at 20k; 1.8828 at 100k).**

## Reproduction

- Replay the headline 100k result from `Rust_MLKit/arch_02_value_resid/metal-native`:
  `METAL_NATIVE_FA_TILED=1 METAL_NATIVE_DATA_SEED=0 cargo run --release --bin train -- --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 --token-bytes ../burn-port/token_bytes.json --out out/sota_f32_clipsoft_seed1337_100k_fa_tiled_softsplit_wsd --iters 100000 --warmdown-start 16000 --warmdown 24000 --lr-floor 0.1 --final-warmdown 10000 --seed 1337 --golden-init --f32 --clip-soft --log-every 100 --eval-every 1000`
- Ensure `METAL_NATIVE_GEMM_ACCUM` is unset, as required by the preserved recipe.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: Medium.** Multiple long runs and two 20k seeds support the quality direction, but speed/quality use different flash paths and 16M plus CUDA parity remain incomplete.

## Artifacts

- [`Rust_MLKit/docs/optimization_map.md`](../../Rust_MLKit/docs/optimization_map.md) — live speed, binder, BPB, failure, and test gates.
- [`Rust_MLKit/arch_02_value_resid/metal-native/README.md`](../../Rust_MLKit/arch_02_value_resid/metal-native/README.md) — recipes, benchmark tables, long-run results, and 16M scale-up.
- [`Rust_MLKit/arch_02_value_resid/metal-native/DECISIONS.md`](../../Rust_MLKit/arch_02_value_resid/metal-native/DECISIONS.md) — accepted/rejected defaults and supporting evidence.
- `Rust_MLKit/arch_02_value_resid/metal-native/out/sota_f32_clipsoft_seed1337_20k_fa_tiled_softsplit_warmdown/`
- `Rust_MLKit/arch_02_value_resid/metal-native/out/sota_f32_clipsoft_seed1337_100k_fa_tiled_softsplit_wsd/`
- `Rust_MLKit/arch_02_value_resid/metal-native/out/sota_f32_clipsoft_16m_seed1337_1k_opt/`
- `Rust_MLKit/arch_02_value_resid/metal-native/out/opt16m_ab/`

## Why this experiment happened

The local architecture ladder had identified value residual as the supported quality lever, while the M5 work had built substantial Metal runtime expertise. This suite joined those threads by porting `arch_02` training itself and requiring separate speed and quality gates. The preceding notebook context is [41-audit-deep-2026-07-14](../gemma-metal/41-audit-deep-2026-07-14.md).

## Experiment story

**Baseline.** The local architecture ladder had identified value residual as the supported quality lever, while the M5 work had built substantial Metal runtime expertise. This suite joined those threads by porting `arch_02` training itself and requiring separate speed and quality gates. The preceding notebook context is [41-audit-deep-2026-07-14](../gemma-metal/41-audit-deep-2026-07-14.md).

**Hypothesis.** A from-scratch Rust + Metal 4 training path can preserve the `arch_02` math while making M5-native TensorOps GEMM, packed command encoding, fused backward kernels, and explicit residency materially faster than the Burn reference. Quality must be judged separately from the step-time gate: the fast row-wise flash path and the slower tiled flash path can have different Soft-clipping dynamics and final BPB.

**Test contract.** Trainer / preset: `Rust_MLKit/arch_02_value_resid/metal-native`, `train --f32`, default `sota` preset, FineWeb, seed 1337, B=16 × T=256 (4096 tokens/step). Fixed knobs: Metal 4-only encode; MPP TensorOps primary GEMM; default Soft-split clipping (Muon × √clip coefficient, AdamW × clip coefficient); `METAL_NATIVE_GEMM_ACCUM` off after the Soft regression bisect. Env flags: default speed gate uses row-wise FA; quality runs set `METAL_NATIVE_FA_TILED=1`. The validated 20k arm uses `--warmdown 3500`; the 100k arm uses `--warmdown-start 16000 --warmdown 24000 --lr-floor 0.1 --final-warmdown 10000`.

**Variant sequence.** The preserved comparison matrix was: Soft-harden default — Row FA, GEMM accumulate off; Speed A/B — `METAL_NATIVE_GEMM_ACCUM=1`; FA_TILED quality — Tiled FA backward, GEMM accumulate off; TensorOps flash probe — `--flash-tensorops`, GEMM accumulate off; 20k Soft recipe — FA_TILED + Soft-split + 3500-step final warmdown; 100k Soft WSD — FA_TILED + staged warmdown/hold/final-warmdown schedule.

**Measured turn.** The result board records 1 — 100k Soft WSD, seed 1337 — FINAL EMA BPB **1.8828** — Best live 1.8819 @96999; ~5.6 h, ~60k tok/s; 2 — 20k Soft + warmdown, seed 42 reseed — FINAL EMA BPB **1.8876** — gnorm 60 @19999; 3 — 20k Soft + warmdown, seed 1337 golden — FINAL EMA BPB **1.8969** — gnorm 216 @19999; ~24–27 min; 4 — 3k Soft-split + FA_TILED — FINAL EMA BPB **2.0222** — Best documented 3k Soft-split quality result; 5 — 3k Soft-everywhere + FA_TILED — FINAL EMA BPB **2.0369** — ~69 ms/step; quality opt-in.

**Turning point and readout.** The live default gate is **56.6 ms/step** and **~72k tok/s** at B=16. Enabling GEMM accumulate reduces the binder count from 250 to 211 and measures about **55–56.5 ms/step**, but its 3k Soft EMA regresses to **2.0580**, so it is not the default. FA_TILED costs roughly **69 ms/step / ~60k tok/s** but improves the documented 3k quality results; at longer horizons, explicit scheduling produces FINAL EMA **1.8969** at 20k and **1.8828** at 100k. The first `--preset 16m` verification is not yet a long-run result: FA_TILED B=16/T=256 benchmarks at **~954 ms/step / ~4292 tok/s**, and its 1k Soft check ends at FINAL EMA BPB **2.1462**. The planned 16M 20k run therefore remains open. **Interpretation boundary.** Step time/throughput and BPB come from different flash paths, so the fastest row-FA result is not the best-quality result. The 100k BPB is measured at one seed; cross-hardware parity remains open.

**Failures and surprises.** Audit 6 `multiply_accumulate` improved speed/binders but regressed 3k Soft EMA to **2.0580** with late gnorm ~9; `METAL_NATIVE_GEMM_ACCUM` is off by default. `METAL_NATIVE_HAZARD_BARRIERS=1` measured ~53.5 ms but produced NaN loss by step 3; strict barriers remain the default. `--flash-tensorops` reached EMA **2.0462** but late gnorm ~13 @2999; it was rejected for the Soft ladder. Soft-everywhere FA_TILED diverged after roughly 3.5–4.5k and ended at EMA **2.2575** on the 20k attempt.

## Decision and aftermath

**Kept:** Keep speed and quality gates separate: row FA sustains the 56.6 ms default, while FA_TILED plus horizon-appropriate warmdown delivers the best measured Soft EMA (1.8969 at 20k; 1.8828 at 100k). **Boundary:** Incomplete or failed confirmation prevents promotion beyond the explicitly successful stages.

## Detailed observations

- The result artifact reports: 1 — 100k Soft WSD, seed 1337 — FINAL EMA BPB **1.8828** — Best live 1.8819 @96999; ~5.6 h, ~60k tok/s.
- The result artifact reports: 2 — 20k Soft + warmdown, seed 42 reseed — FINAL EMA BPB **1.8876** — gnorm 60 @19999.
- The result artifact reports: 3 — 20k Soft + warmdown, seed 1337 golden — FINAL EMA BPB **1.8969** — gnorm 216 @19999; ~24–27 min.
- The result artifact reports: 4 — 3k Soft-split + FA_TILED — FINAL EMA BPB **2.0222** — Best documented 3k Soft-split quality result.
- Failure/operational record: Audit 6 `multiply_accumulate` improved speed/binders but regressed 3k Soft EMA to **2.0580** with late gnorm ~9; `METAL_NATIVE_GEMM_ACCUM` is off by default.
- Failure/operational record: `METAL_NATIVE_HAZARD_BARRIERS=1` measured ~53.5 ms but produced NaN loss by step 3; strict barriers remain the default.

## What this does not prove

**Confidence: Medium.** Multiple long runs and two 20k seeds support the quality direction, but speed/quality use different flash paths and 16M plus CUDA parity remain incomplete. Incomplete or failed confirmation prevents promotion beyond the explicitly successful stages. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## 128M M5 Pro engineering gate (2026-07-14)

This is a systems/correctness result, not an optimizer-quality result. The new
`arch02-128m` preset is exactly **128,367,988 parameters** at L=24, C=768,
Hq=24, Hkv=12, D=32, and MLP=2304. The target-width QKV post-backward path no
longer truncates C=768/KV=384 gradients, and the independent MLX oracle agrees
with the manual GQA causal-attention backward to maximum absolute errors
`dQ=1.19e-6`, `dK=9.10e-7`, and `dV=0` on an irregular sequence.

On the verified 20-core-GPU, 64GB M5 Pro running macOS 27.0, replacing serialized
per-matrix Muon with same-shape bank-batched Metal 4 TensorOps reduced the exact
128M one-step optimizer time from **4201.3 ms to 478.8 ms** (8.8x) and total step
time from **6813.3 ms to 2930.5 ms**. After three warmup steps, a six-step smoke
settled at **2816.2 ms/step, 1454 tokens/s, 1701 dispatches/step, and 16.12GB RSS**
at B=16, T=256. This is below the 52GiB footprint gate and showed no NaN,
dispatch rollover, or swap pressure during the smoke.

The version-2 checkpoint now covers every master weight, the actual persisted
bf16 shadow bits, every Adam tensor, Muon momentum tensor, EMA, schedule state,
seed, step, and data cursor. Save/reload replay matched the uninterrupted next step with zero
loss delta, `8.94e-8` gradient-norm delta, and `2.98e-8` maximum QO-master delta.
The complete release suite passes **50/50**, the Python optimizer/oracle suite
passes **22/22**, and the C++ and Swift interfaces typecheck.

**Gate decision:** do not start the 2,000-step champion. The native quality
funnel has not selected an optimizer yet; only Muon NS5/NS3 are currently
native-qualified, while the remaining candidates are independent Python
oracles/capability-gated research arms. These measurements establish that the
128M engine can enter the optimizer study, not that Muon is the champion.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — consolidated scoreboard and teaching context.
- Related suites: [`04-sota-arch-ladder`](../training/04-sota-arch-ladder.md), [`05-sota-arch-followup-value-resid`](../training/05-sota-arch-followup-value-resid.md)

---

Previous · [Index](../00-INDEX.md) · Next
