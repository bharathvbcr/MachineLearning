# 11: Phase 1 — instrumented 124M FineWeb base

## Executive summary

- **Question:** A short, fully instrumented 124M FineWeb-edu run should establish the logging contract (tok/s, MFU, val_loss) that all later mixer/optimizer/GPU suites reuse.
- **Result:** Instrument MFU/tok/s on the first FineWeb run — every later GPU win is measured against this contract.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `11-phase1-fineweb` |
| Dates | 2026-06-15 – 2026-06-15 |
| Hardware | RTX 3070 Ti Laptop 8 GB |
| Status | `done` |

## Hypothesis

A short, fully instrumented 124M FineWeb-edu run should establish the logging contract (tok/s, MFU, val_loss) that all later mixer/optimizer/GPU suites reuse.

## Setup

- Trainer / preset: `nanolab` — `phase1_fineweb`
- Fixed knobs: 12L / 768d / 12H, GPT-2 vocab, SwiGLU, attention mixer, Muon lr 6e-4, matrix_lr 0.025, block 1024, bs16, 150 steps, eval every 50, bf16, grad checkpoint, FineWeb-edu
- Env flags: device `auto`; seed 1337; fused CE + TF32 available in config

## Variants

| Variant | Change |
|---------|--------|
| `phase1_fineweb` | Single instrumented base (no mixer lever) |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | `phase1_fineweb` | best_val **5.713** @ 9.83M tok | Eval path: 6.218 (50) → 5.870 (100) → 5.713 (end) |
| — | throughput | peak **~11.7K tok/s**, MFU **~21.7%** | Confirms bf16+Muon path on 124M before gpu_max stack |

This is a short burn-in, not a quality bakeoff — the scoreboard purpose is “metrics present and healthy,” not competitor ranking. Params logged at start: **123.7M**.

**Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

## Failures

- None recorded (clean `done` with best_val).

## Lesson

**Instrument MFU/tok/s on the first FineWeb run — every later GPU win is measured against this contract.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.train`.
- Required replay inputs: `nanolab/out/phase1_fineweb/config.json`. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Single short seed completed with healthy instrumentation; it establishes a measurement contract, not a quality ranking.

## Artifacts

- `nanolab/out/phase1_fineweb/config.json`, `metrics.jsonl`, `best.pt`

## Why this experiment happened

The TinyStories smoke proved basic learning, but the later systems questions required a realistic 124M FineWeb shape and consistent telemetry. Phase 1 therefore established the tok/s, MFU, validation-loss, and parameter-count contract reused by the rest of nanolab. The preceding notebook context is [10-phase0-smoke](10-phase0-smoke.md).

## Experiment story

**Baseline.** The TinyStories smoke proved basic learning, but the later systems questions required a realistic 124M FineWeb shape and consistent telemetry. Phase 1 therefore established the tok/s, MFU, validation-loss, and parameter-count contract reused by the rest of nanolab. The preceding notebook context is [10-phase0-smoke](10-phase0-smoke.md).

**Hypothesis.** A short, fully instrumented 124M FineWeb-edu run should establish the logging contract (tok/s, MFU, val_loss) that all later mixer/optimizer/GPU suites reuse.

**Test contract.** Trainer / preset: `nanolab` — `phase1_fineweb` Fixed knobs: 12L / 768d / 12H, GPT-2 vocab, SwiGLU, attention mixer, Muon lr 6e-4, matrix_lr 0.025, block 1024, bs16, 150 steps, eval every 50, bf16, grad checkpoint, FineWeb-edu Env flags: device `auto`; seed 1337; fused CE + TF32 available in config

**Variant sequence.** The preserved comparison matrix was: `phase1_fineweb` — Single instrumented base (no mixer lever).

**Measured turn.** The result board records 1 — `phase1_fineweb` — best_val **5.713** @ 9.83M tok — Eval path: 6.218 (50) → 5.870 (100) → 5.713 (end); — — throughput — peak **~11.7K tok/s**, MFU **~21.7%** — Confirms bf16+Muon path on 124M before gpu_max stack.

**Turning point and readout.** This is a short burn-in, not a quality bakeoff — the scoreboard purpose is “metrics present and healthy,” not competitor ranking. Params logged at start: **123.7M**. **Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

**Failures and surprises.** None recorded (clean `done` with best_val).

## Decision and aftermath

**Kept:** Instrument MFU/tok/s on the first FineWeb run — every later GPU win is measured against this contract. The notebook continues with [12-mixer-ab-tinystories](12-mixer-ab-tinystories.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `phase1_fineweb` — best_val **5.713** @ 9.83M tok — Eval path: 6.218 (50) → 5.870 (100) → 5.713 (end).
- The result artifact reports: — — throughput — peak **~11.7K tok/s**, MFU **~21.7%** — Confirms bf16+Muon path on 124M before gpu_max stack.
- Failure/operational record: None recorded (clean `done` with best_val).

## What this does not prove

**Confidence: Medium.** Single short seed completed with healthy instrumentation; it establishes a measurement contract, not a quality ranking. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.6, §8.8
- Related suites: [`18-gpu-maximization`](18-gpu-maximization.md), [`13-mixer-bakeoff-2M`](13-mixer-bakeoff-2M.md), [`20-run128m-long`](20-run128m-long.md)

---

[Previous](10-phase0-smoke.md) · [Index](../00-INDEX.md) · [Next](12-mixer-ab-tinystories.md)
