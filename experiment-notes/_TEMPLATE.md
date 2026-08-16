# <suite-id>: <short title>

## Executive summary

- **Question:** State the decision this suite tests.
- **Result:** State the strongest measured result, or explicitly say that no result exists.
- **Implication:** State what changes (or must not change) because of the evidence.
- **Status:** `done` / `partial` / `planned` / `blocked`; evidence confidence **High** / **Medium** / **Low**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `<suite-id>` |
| Dates | `YYYY-MM-DD` – `YYYY-MM-DD` |
| Hardware | `<device / GPU / relevant runtime versions>` |
| Status | `done` / `partial` / `planned` / `blocked` |

## Hypothesis

State the expected result and the decision it would affect.

## Setup

- Trainer / preset:
- Fixed knobs:
- Env flags:
- Seeds / stages:

## Variants

| Variant | Change |
|---------|--------|
| control | none |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | | | |

Separate measured facts from inference. Compare effect size with seed/run noise where available, and never use throughput as a proxy for quality.

## Failures

- Record crashes, OOMs, NaNs, parity mismatches, missing stages, and non-comparable cells.

## Lesson

**One bounded takeaway that follows from the evidence.**

## Reproduction

- Exact command and working directory, or: **No exact replay command was preserved.**
- If no exact command exists, name the closest runner and every required config/artifact.

## Evidence quality

**Confidence: High / Medium / Low.** Justify with seeds, stage completeness, hardware noise, failures, proxy/target match, and parity state.

## Artifacts

- Link concrete summaries, configs, logs, and champions.

## Why this experiment happened

Name the preceding result, failure, or product constraint that created this question. Link the prior suite when applicable. If chronology is reconstructed from artifacts rather than directly recorded, label it as reconstructed.

## Experiment story

Tell the source-grounded chronology in roughly 4–10 substantive paragraphs: baseline → hypothesis → variants/stages → turning point → failures or surprises → final result. Use timestamps, stage order, configs, gates, and compact tables when preserved. Separate “the log reports” from “this suggests”; do not invent rationale or emotion.

## Decision and aftermath

State exactly what was kept, rejected, deferred, or carried forward. Link the downstream suite and distinguish a recorded decision from an inferred connection.

## Detailed observations

- Add 3–8 concrete observations from primary artifacts: curve shape, stage reversals, effect sizes, speed/quality tradeoffs, failure signatures, or operational details.
- Avoid repeating only the headline result.

## What this does not prove

State limits from seeds, stages, prompts, hardware, parity, metric choice, confounding config changes, or missing artifacts. Give plausible alternative explanations without overstating them.

## See also

- [Experiment index](00-INDEX.md)
- [Learning notes scoreboard](../learning-notes/08-experiments-and-results.md)
- Related suite links:

---

Previous · [Index](00-INDEX.md) · Next
