# 35: MLX DFlash block-size tuning

## Executive summary

- **Question:** Fine-sweeping DFlash `block_size` (and optionally EMA adaptive block) would beat fixed block=5 on 31B, especially on creative prose where accept is low.
- **Result:** Keep fixed block=5; block tuning cannot fix the prose accept gap — need a better draft (or trees), not an adaptive size policy.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `35-mlx-dflash-block-tuning` |
| Dates | `2026-07-13` – `2026-07-13` |
| Hardware | Apple M5 Pro · mlx 0.32.0 |
| Status | `done` |

## Hypothesis

Fine-sweeping DFlash `block_size` (and optionally EMA adaptive block) would beat fixed block=5 on 31B, especially on creative prose where accept is low.

## Setup

- Trainer / preset: DFlash q4-draft on `gemma-4-31b-it-4bit` + `gemma-4-31B-it-DFlash`
- Fixed knobs: mlx 0.32; greedy exact verify
- Env flags: n/a

## Variants

| Variant | Change |
|---------|--------|
| Fine sweep | blocks 3–8 on structured prompt (finesweep32) |
| By-type probe | prose vs code × blocks 2–8 |
| Adaptive EMA | accept→block policy (A/B aborted) |
| Earlier coarse sweep | blocks 8–32 on mlx pre-fine path |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | Fine-sweep **block=5** | median **37.17** tok/s; accept **3.46** | Best overall |
| 2 | block=4 | 35.01 / accept 2.91 | |
| 3 | block=3 | 32.6 / accept 2.51 | |
| — | block=6 | 30.54 / accept 3.76 | Non-monotonic drop |
| — | block=8 | 27.84 / accept 4.13 | |
| Code probe | best **block=5** → **30.62** | bs=3 only 21.3 — cannot shrink |
| Prose probe | best **block=3** → **19.88** | vs fixed5 **19.17** (+**3.7%**) |

Adaptive verdict (`adaptive_block_finding.json`): **NOT WORTH IT** — fixed block=5 near-optimal; adaptive headroom **<~4%** and only on prose; a single accept→block policy cannot hit code(5) and prose(3) optima together. Prose ceiling is draft-acceptance-bound (accept ~2.16), not block-bound.

**Interpretation boundary.** The block=5 ranking is measured on the sampled prompts. The claim that a better draft is the next lever is an inference from low prose acceptance and small tuning headroom.

## Failures

- Adaptive A/B aborted by GPU Timeout under concurrent custom-engine load
- Coarse old sweep (blocks ≥8) misleading vs fine 0.32 curve (5 optimal; ≥12 cliffs in earlier work)

## Lesson

**Keep fixed block=5; block tuning cannot fix the prose accept gap — need a better draft (or trees), not an adaptive size policy.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: the MLX DFlash benchmark scripts.
- Required replay inputs: `block_finesweep32_result.json`, `block_by_type_result.json`, and `adaptive_block_finding.json` for prompt/block matrices. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Measured fine and by-type sweeps agree on block=5, but the adaptive A/B was aborted and the <4% headroom is workload-specific.

## Artifacts

- `Rust_MLKit/gemma-metal/bench/results/block_finesweep32_result.json`
- `Rust_MLKit/gemma-metal/bench/results/block_by_type_result.json`
- `Rust_MLKit/gemma-metal/bench/results/adaptive_block_finding.json`
- `Rust_MLKit/gemma-metal/bench/results/dflash_blocksize_sweep_31b.json` (earlier coarse)

## Why this experiment happened

Block=5 made the MLX DFlash path shippable overall, but prompt diversity exposed creative prose as the weak case. The tuning suite tested whether block policy, rather than draft quality, explained that acceptance and throughput gap. The preceding notebook context is [34-mlx-dflash-product](34-mlx-dflash-product.md).

## Experiment story

**Baseline.** Block=5 made the MLX DFlash path shippable overall, but prompt diversity exposed creative prose as the weak case. The tuning suite tested whether block policy, rather than draft quality, explained that acceptance and throughput gap. The preceding notebook context is [34-mlx-dflash-product](34-mlx-dflash-product.md).

**Hypothesis.** Fine-sweeping DFlash `block_size` (and optionally EMA adaptive block) would beat fixed block=5 on 31B, especially on creative prose where accept is low.

**Test contract.** Trainer / preset: DFlash q4-draft on `gemma-4-31b-it-4bit` + `gemma-4-31B-it-DFlash` Fixed knobs: mlx 0.32; greedy exact verify Env flags: n/a

**Variant sequence.** The preserved comparison matrix was: Fine sweep — blocks 3–8 on structured prompt (finesweep32); By-type probe — prose vs code × blocks 2–8; Adaptive EMA — accept→block policy (A/B aborted); Earlier coarse sweep — blocks 8–32 on mlx pre-fine path.

**Measured turn.** The result board records 1 — Fine-sweep **block=5** — median **37.17** tok/s; accept **3.46** — Best overall; 2 — block=4 — 35.01 / accept 2.91 — ; 3 — block=3 — 32.6 / accept 2.51 — ; — — block=6 — 30.54 / accept 3.76 — Non-monotonic drop; — — block=8 — 27.84 / accept 4.13 — .

**Turning point and readout.** Adaptive verdict (`adaptive_block_finding.json`): **NOT WORTH IT** — fixed block=5 near-optimal; adaptive headroom **<~4%** and only on prose; a single accept→block policy cannot hit code(5) and prose(3) optima together. Prose ceiling is draft-acceptance-bound (accept ~2.16), not block-bound. **Interpretation boundary.** The block=5 ranking is measured on the sampled prompts. The claim that a better draft is the next lever is an inference from low prose acceptance and small tuning headroom.

**Failures and surprises.** Adaptive A/B aborted by GPU Timeout under concurrent custom-engine load Coarse old sweep (blocks ≥8) misleading vs fine 0.32 curve (5 optimal; ≥12 cliffs in earlier work)

## Decision and aftermath

**Kept:** Keep fixed block=5; block tuning cannot fix the prose accept gap — need a better draft (or trees), not an adaptive size policy. The notebook continues with [36-native-dflash-parity-accept](36-native-dflash-parity-accept.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — Fine-sweep **block=5** — median **37.17** tok/s; accept **3.46** — Best overall.
- The result artifact reports: 2 — block=4 — 35.01 / accept 2.91 — .
- The result artifact reports: 3 — block=3 — 32.6 / accept 2.51 — .
- The result artifact reports: — — block=6 — 30.54 / accept 3.76 — Non-monotonic drop.
- Failure/operational record: Adaptive A/B aborted by GPU Timeout under concurrent custom-engine load
- Failure/operational record: Coarse old sweep (blocks ≥8) misleading vs fine 0.32 curve (5 optimal; ≥12 cliffs in earlier work)

## What this does not prove

**Confidence: Medium.** Measured fine and by-type sweeps agree on block=5, but the adaptive A/B was aborted and the <4% headroom is workload-specific. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- Related suites: [`34-mlx-dflash-product`](34-mlx-dflash-product.md), [`40-ddtree-frontier`](40-ddtree-frontier.md)

---

[Previous](34-mlx-dflash-product.md) · [Index](../00-INDEX.md) · [Next](36-native-dflash-parity-accept.md)
