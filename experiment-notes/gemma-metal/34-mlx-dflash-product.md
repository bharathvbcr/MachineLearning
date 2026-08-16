# 34: MLX DFlash product path (~31 tok/s)

## Executive summary

- **Question:** DFlash block speculative decoding with a 4-bit draft on mlx 0.32 (M5 NAX-accelerated M=block verify) would clear 31B gates ≥15 / ≥25 while staying exact vs greedy (honest lane).
- **Result:** MLX 0.32 + DFlash block=5 is the shippable 31B path on this Mac (~31 tok/s median, exact vs greedy); custom Metal is not yet in the race.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **High**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `34-mlx-dflash-product` |
| Dates | `2026-07-13` – `2026-07-14` |
| Hardware | Apple M5 Pro · mlx **0.32.0** (`~/.venvs/dflash32`) |
| Status | `done` |

## Hypothesis

DFlash block speculative decoding with a 4-bit draft on mlx 0.32 (M5 NAX-accelerated M=block verify) would clear 31B gates ≥15 / ≥25 while staying exact vs greedy (honest lane).

## Setup

- Trainer / preset: `bench/dflash_fast_31b.py`, `bench/serve_dflash.py` (:8788); target `mlx-community/gemma-4-31b-it-4bit`; draft `z-lab/gemma-4-31B-it-DFlash`
- Fixed knobs: greedy exact verify; typical **block=5**; q4 draft `nn.quantize` g64
- Env flags: mlx 0.32.0 required for speed (dflash `[mlx]` extra pins 0.31.2 — install separately)

## Variants

| Variant | Change |
|---------|--------|
| Plain 31B | greedy mlx-lm decode |
| DFlash + bf16 draft | block verify |
| DFlash + q4 draft | −2.2 GB, +~6% |
| mlx 0.31.2 vs 0.32.0 | NAX on M=block GEMMs |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | Diversity sweep DFlash bs=5 | median **31.7** tok/s (range **18.94–36.12**) | vs plain median **12.49**; speedup median **2.49×** |
| 2 | Fine-sweep best (bs=5) | median **37.17** tok/s; accept **3.46** | Structured peak |
| 3 | mlx032 NAX A/B (bs=8) | **27.77** vs **18.64** median | **1.49×** from 0.31.2→0.32.0 |
| 4 | q4 vs bf16 draft (bs=8) | q4 **18.6** vs bf16 **17.56** | **1.059×**; exact verify |
| — | Plain decode | ~**12.5–12.7** | Phase-0 / diversity plain |

Across 8 prompt types every type clears ≥15; all but creative prose clear ≥25 (prose **18.94**, accept **2.16**). MLX golden parity: DFlash stream == greedy (`dflash_parity_mlx_golden.json`, mean_accept≈3.0 @ bs=5, ~37.5 tok/s on “Say hi”).

**Interpretation boundary.** Throughput, prompt spread, acceptance, and exact parity are separately measured. Faster decode does not imply changed output quality because exact greedy verification preserves the stream.

## Failures

- Wired memory (`set_wired_limit`): no effect
- Lower-bit / mxfp4 target: not pursued (honest doctrine / M=1 BW-bound)
- Native gemma-metal DFlash still far behind this product path

## Lesson

**MLX 0.32 + DFlash block=5 is the shippable 31B path on this Mac (~31 tok/s median, exact vs greedy); custom Metal is not yet in the race.**

## Reproduction

- Replay: From `Rust_MLKit/gemma-metal`, use the mlx 0.32 environment documented in `docs/gates.md`: `~/.venvs/dflash32/bin/python bench/dflash_fast_31b.py`; server path: `~/.venvs/dflash32/bin/python bench/serve_dflash.py`.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: High.** Eight prompt types, medians/ranges, version A/Bs, and exact greedy parity support the product conclusion; creative prose remains the weak case.

## Artifacts

- `Rust_MLKit/gemma-metal/docs/speed_findings.md`
- `Rust_MLKit/gemma-metal/bench/results/diversity_sweep_result.json`
- `Rust_MLKit/gemma-metal/bench/results/mlx032_nax_ab_31b.json`
- `Rust_MLKit/gemma-metal/bench/results/dflash_q4draft_interleaved_31b.json`
- `Rust_MLKit/gemma-metal/bench/results/dflash_parity_mlx_golden.json`
- `Rust_MLKit/gemma-metal/docs/gates.md` — Phase-0 DFlash row

## Why this experiment happened

The roofline result said native single-token decode would not reach mlx-class speed through another kernel rewrite alone. DFlash on mlx 0.32 tested the alternative: amortize target work across accepted draft blocks while preserving exact greedy output. The preceding notebook context is [33-kernel-roofline-overhead](33-kernel-roofline-overhead.md).

## Experiment story

**Baseline.** The roofline result said native single-token decode would not reach mlx-class speed through another kernel rewrite alone. DFlash on mlx 0.32 tested the alternative: amortize target work across accepted draft blocks while preserving exact greedy output. The preceding notebook context is [33-kernel-roofline-overhead](33-kernel-roofline-overhead.md).

**Hypothesis.** DFlash block speculative decoding with a 4-bit draft on mlx 0.32 (M5 NAX-accelerated M=block verify) would clear 31B gates ≥15 / ≥25 while staying exact vs greedy (honest lane).

**Test contract.** Trainer / preset: `bench/dflash_fast_31b.py`, `bench/serve_dflash.py` (:8788); target `mlx-community/gemma-4-31b-it-4bit`; draft `z-lab/gemma-4-31B-it-DFlash` Fixed knobs: greedy exact verify; typical **block=5**; q4 draft `nn.quantize` g64 Env flags: mlx 0.32.0 required for speed (dflash `[mlx]` extra pins 0.31.2 — install separately)

**Variant sequence.** The preserved comparison matrix was: Plain 31B — greedy mlx-lm decode; DFlash + bf16 draft — block verify; DFlash + q4 draft — −2.2 GB, +~6%; mlx 0.31.2 vs 0.32.0 — NAX on M=block GEMMs.

**Measured turn.** The result board records 1 — Diversity sweep DFlash bs=5 — median **31.7** tok/s (range **18.94–36.12**) — vs plain median **12.49**; speedup median **2.49×**; 2 — Fine-sweep best (bs=5) — median **37.17** tok/s; accept **3.46** — Structured peak; 3 — mlx032 NAX A/B (bs=8) — **27.77** vs **18.64** median — **1.49×** from 0.31.2→0.32.0; 4 — q4 vs bf16 draft (bs=8) — q4 **18.6** vs bf16 **17.56** — **1.059×**; exact verify; — — Plain decode — ~**12.5–12.7** — Phase-0 / diversity plain.

**Turning point and readout.** Across 8 prompt types every type clears ≥15; all but creative prose clear ≥25 (prose **18.94**, accept **2.16**). MLX golden parity: DFlash stream == greedy (`dflash_parity_mlx_golden.json`, mean_accept≈3.0 @ bs=5, ~37.5 tok/s on “Say hi”). **Interpretation boundary.** Throughput, prompt spread, acceptance, and exact parity are separately measured. Faster decode does not imply changed output quality because exact greedy verification preserves the stream.

**Failures and surprises.** Wired memory (`set_wired_limit`): no effect Lower-bit / mxfp4 target: not pursued (honest doctrine / M=1 BW-bound) Native gemma-metal DFlash still far behind this product path

## Decision and aftermath

**Kept:** MLX 0.32 + DFlash block=5 is the shippable 31B path on this Mac (~31 tok/s median, exact vs greedy); custom Metal is not yet in the race. The notebook continues with [35-mlx-dflash-block-tuning](35-mlx-dflash-block-tuning.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — Diversity sweep DFlash bs=5 — median **31.7** tok/s (range **18.94–36.12**) — vs plain median **12.49**; speedup median **2.49×**.
- The result artifact reports: 2 — Fine-sweep best (bs=5) — median **37.17** tok/s; accept **3.46** — Structured peak.
- The result artifact reports: 3 — mlx032 NAX A/B (bs=8) — **27.77** vs **18.64** median — **1.49×** from 0.31.2→0.32.0.
- The result artifact reports: 4 — q4 vs bf16 draft (bs=8) — q4 **18.6** vs bf16 **17.56** — **1.059×**; exact verify.
- Failure/operational record: Wired memory (`set_wired_limit`): no effect
- Failure/operational record: Lower-bit / mxfp4 target: not pursued (honest doctrine / M=1 BW-bound)

## What this does not prove

**Confidence: High.** Eight prompt types, medians/ranges, version A/Bs, and exact greedy parity support the product conclusion; creative prose remains the weak case. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- Related suites: [`35-mlx-dflash-block-tuning`](35-mlx-dflash-block-tuning.md), [`36-native-dflash-parity-accept`](36-native-dflash-parity-accept.md), [`39-mlx-serve-ttft`](39-mlx-serve-ttft.md)

---

[Previous](33-kernel-roofline-overhead.md) · [Index](../00-INDEX.md) · [Next](35-mlx-dflash-block-tuning.md)
