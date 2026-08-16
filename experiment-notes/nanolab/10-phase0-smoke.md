# 10: Phase 0 smoke — cpu_smoke + TinyStories

## Executive summary

- **Question:** A minimal char-LM smoke and a small GPT-2 TinyStories run should both descend cleanly — proving the trainer, logging, and Muon/bf16 stack before any mixer or scale A/B.
- **Result:** If smoke and phase0 do not descend with clean logs, nothing upstream of A/Bs is trustworthy.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `10-phase0-smoke` |
| Dates | 2026-06-15 – 2026-06-15 |
| Hardware | RTX 3070 Ti Laptop 8 GB (phase0); CPU path also exercised |
| Status | `done` |

## Hypothesis

A minimal char-LM smoke and a small GPT-2 TinyStories run should both descend cleanly — proving the trainer, logging, and Muon/bf16 stack before any mixer or scale A/B.

## Setup

- Trainer / preset: `nanolab` — `cpu_smoke`, `phase0_tinystories`
- Fixed knobs:
  - `cpu_smoke`: 2L/128d, block 128, AdamW lr 3e-3, Shakespeare char, fp32, ~15–35 steps (log has 3 restarts)
  - `phase0_tinystories`: 6L/384d/6H, block 256, bs64, Muon lr 1e-3, GPT-2 tok, TinyStories, bf16, 1500 steps, grad checkpoint
- Env flags: device `auto`; seed 1337

## Variants

| Variant | Change |
|---------|--------|
| `cpu_smoke` | Tiny char LM on Shakespeare — harness smoke |
| `phase0_tinystories` | ~30M-class attention LM on TinyStories — first real descent |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | `phase0_tinystories` | best_val **1.527** @ 49.2M tok | Smooth eval curve 2.10→1.83→1.70→1.59→1.53; peak ~44K tok/s, MFU ~18% |
| 2 | `cpu_smoke` (longest restart) | best_val **2.601** @ 41K tok | Loss 4.35→~2.6; val_ppl ~16.2 at step 20 |

`cpu_smoke` was restarted three times in the same `metrics.jsonl` (best_val 2.601 / 2.855 / 2.956 as step budget shrank). Phase0 is the pedagogical “watch it learn” run: monotonic val improvement every 300 steps with stable grad norms.

**Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

## Failures

- No OOMs or NaNs.
- `cpu_smoke` log aggregates multiple short restarts — use the first `done` (best_val 2.601) as the smoke pass, not the later truncated ones.

## Lesson

**If smoke and phase0 do not descend with clean logs, nothing upstream of A/Bs is trustworthy.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.train`.
- Required replay inputs: `nanolab/out/cpu_smoke/config.json` and `nanolab/out/phase0_tinystories/config.json`. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Single-seed smoke evidence shows clean descent, but the CPU log combines restarts and this is a harness check rather than a comparative result.

## Artifacts

- `nanolab/out/cpu_smoke/config.json`, `metrics.jsonl`
- `nanolab/out/phase0_tinystories/config.json`, `metrics.jsonl`, `best.pt`

## Why this experiment happened

The nanolab track began by establishing that the teaching trainer, data paths, logging, precision, and optimizer stack could learn at all. This was a prerequisite rather than an architecture contest: a failed descent here would invalidate every later A/B. The preceding notebook context is [07-h100-conductor-planned](../training/07-h100-conductor-planned.md).

## Experiment story

**Baseline.** The nanolab track began by establishing that the teaching trainer, data paths, logging, precision, and optimizer stack could learn at all. This was a prerequisite rather than an architecture contest: a failed descent here would invalidate every later A/B. The preceding notebook context is [07-h100-conductor-planned](../training/07-h100-conductor-planned.md).

**Hypothesis.** A minimal char-LM smoke and a small GPT-2 TinyStories run should both descend cleanly — proving the trainer, logging, and Muon/bf16 stack before any mixer or scale A/B.

**Test contract.** Trainer / preset: `nanolab` — `cpu_smoke`, `phase0_tinystories` Fixed knobs: - `cpu_smoke`: 2L/128d, block 128, AdamW lr 3e-3, Shakespeare char, fp32, ~15–35 steps (log has 3 restarts) - `phase0_tinystories`: 6L/384d/6H, block 256, bs64, Muon lr 1e-3, GPT-2 tok, TinyStories, bf16, 1500 steps, grad checkpoint

**Variant sequence.** The preserved comparison matrix was: `cpu_smoke` — Tiny char LM on Shakespeare — harness smoke; `phase0_tinystories` — ~30M-class attention LM on TinyStories — first real descent.

**Measured turn.** The result board records 1 — `phase0_tinystories` — best_val **1.527** @ 49.2M tok — Smooth eval curve 2.10→1.83→1.70→1.59→1.53; peak ~44K tok/s, MFU ~18%; 2 — `cpu_smoke` (longest restart) — best_val **2.601** @ 41K tok — Loss 4.35→~2.6; val_ppl ~16.2 at step 20.

**Turning point and readout.** `cpu_smoke` was restarted three times in the same `metrics.jsonl` (best_val 2.601 / 2.855 / 2.956 as step budget shrank). Phase0 is the pedagogical “watch it learn” run: monotonic val improvement every 300 steps with stable grad norms. **Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

**Failures and surprises.** No OOMs or NaNs. `cpu_smoke` log aggregates multiple short restarts — use the first `done` (best_val 2.601) as the smoke pass, not the later truncated ones.

## Decision and aftermath

**Kept:** If smoke and phase0 do not descend with clean logs, nothing upstream of A/Bs is trustworthy. The notebook continues with [11-phase1-fineweb](11-phase1-fineweb.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `phase0_tinystories` — best_val **1.527** @ 49.2M tok — Smooth eval curve 2.10→1.83→1.70→1.59→1.53; peak ~44K tok/s, MFU ~18%.
- The result artifact reports: 2 — `cpu_smoke` (longest restart) — best_val **2.601** @ 41K tok — Loss 4.35→~2.6; val_ppl ~16.2 at step 20.
- Failure/operational record: No OOMs or NaNs.
- Failure/operational record: `cpu_smoke` log aggregates multiple short restarts — use the first `done` (best_val 2.601) as the smoke pass, not the later truncated ones.

## What this does not prove

**Confidence: Medium.** Single-seed smoke evidence shows clean descent, but the CPU log combines restarts and this is a harness check rather than a comparative result. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.8 (Phase 0 in flowchart)
- Related suites: [`11-phase1-fineweb`](11-phase1-fineweb.md), [`21-diffusion-lm`](21-diffusion-lm.md)

---

Previous · [Index](../00-INDEX.md) · [Next](11-phase1-fineweb.md)
