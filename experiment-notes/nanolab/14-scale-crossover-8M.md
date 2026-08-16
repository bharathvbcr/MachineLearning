# 14: Token-budget crossover @ 8.2M

## Executive summary

- **Question:** If bakeoff ranking is budget-dependent, extending the same 124M configs past ~2M tokens should show attention closing the gap and eventually overtaking minGRU — bias early, capacity late.
- **Result:** Bias wins early, capacity wins late — attention overtakes minGRU between 6.6M and 7.4M tokens.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **High**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `14-scale-crossover-8M` |
| Dates | 2026-06-15 – 2026-06-15 |
| Hardware | RTX 3070 Ti Laptop 8 GB |
| Status | `done` |

## Hypothesis

If bakeoff ranking is budget-dependent, extending the same 124M configs past ~2M tokens should show attention closing the gap and eventually overtaking minGRU — bias early, capacity late.

## Setup

- Trainer / preset: `nanolab` — `scale_{mingru,attention,mamba2}`
- Fixed knobs: same as bakeoff family — 12L/768d, block 512, bs8, Muon lr 6e-4, FineWeb-edu, 2000 steps (= **8.192M tokens**), eval every 200, seed 1337, mixer_chunk 32
- Env flags: device `auto`

## Variants

| Variant | Change |
|---------|--------|
| `scale_mingru` | mixer=`mingru` |
| `scale_attention` | mixer=`attention` |
| `scale_mamba2` | mixer=`mamba2` |

## Results

Headline table matches [08.3](../../learning-notes/08-experiments-and-results.md) (steps×bs8×512 → token millions):

| Tokens | minGRU | Attention | Mamba-2 | gap (A−G) |
|--------|--------|-----------|---------|-----------|
| 0.8M (step 200) | **6.334** | 6.516 | 6.493 | +0.182 |
| 4.1M (step 1000) | **5.549** | 5.624 | 5.769 | +0.075 |
| 6.6M (step 1600) | **5.353** | 5.358 | 5.560 | +0.005 (tied) |
| 7.4M (step 1800) | 5.249 | **5.239** | 5.469 | −0.010 ← overtakes |
| 8.2M (done) | 5.155 | **5.136** | 5.383 | −0.019 |

Final best_val: attention **5.136**, minGRU **5.155**, mamba2 **5.383**. Gap shrinks monotonically then flips between 6.6M and 7.4M — the project’s cleanest scaling result.

**Interpretation boundary.** The sign flip in held-out loss is measured; the “bias-to-capacity” explanation is inference. No throughput claim follows from this quality crossover.

## Failures

- None. Mamba-2 stays last at every checkpoint (bias without enough capacity/throughput story at this scale).

## Lesson

**Bias wins early, capacity wins late — attention overtakes minGRU between 6.6M and 7.4M tokens.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.train`.
- Required replay inputs: the three `nanolab/out/scale_*/config.json` files. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: High.** Dense matched checkpoints show a monotonic gap reversal, and suite 15 independently reproduces the crossover pattern at two scales; each arm itself uses one seed.

## Artifacts

- `nanolab/out/scale_attention/config.json`, `metrics.jsonl`, `best.pt`
- `nanolab/out/scale_mingru/config.json`, `metrics.jsonl`
- `nanolab/out/scale_mamba2/config.json`, `metrics.jsonl`

## Why this experiment happened

The 2M board put all recurrent/SSM mixers ahead of attention, but that snapshot could not distinguish a permanent architecture advantage from an early-training advantage. Extending matched runs to 8.192M tokens tested whether the ranking itself changed with budget. The preceding notebook context is [13-mixer-bakeoff-2M](13-mixer-bakeoff-2M.md).

## Experiment story

**Baseline.** The 2M board put all recurrent/SSM mixers ahead of attention, but that snapshot could not distinguish a permanent architecture advantage from an early-training advantage. Extending matched runs to 8.192M tokens tested whether the ranking itself changed with budget. The preceding notebook context is [13-mixer-bakeoff-2M](13-mixer-bakeoff-2M.md).

**Hypothesis.** If bakeoff ranking is budget-dependent, extending the same 124M configs past ~2M tokens should show attention closing the gap and eventually overtaking minGRU — bias early, capacity late.

**Test contract.** Trainer / preset: `nanolab` — `scale_{mingru,attention,mamba2}` Fixed knobs: same as bakeoff family — 12L/768d, block 512, bs8, Muon lr 6e-4, FineWeb-edu, 2000 steps (= **8.192M tokens**), eval every 200, seed 1337, mixer_chunk 32 Env flags: device `auto`

**Variant sequence.** The preserved comparison matrix was: `scale_mingru` — mixer=`mingru`; `scale_attention` — mixer=`attention`; `scale_mamba2` — mixer=`mamba2`.

**Measured turn.** The result board records 0.8M (step 200) — **6.334** — 6.516 — 6.493 — +0.182; 4.1M (step 1000) — **5.549** — 5.624 — 5.769 — +0.075; 6.6M (step 1600) — **5.353** — 5.358 — 5.560 — +0.005 (tied); 7.4M (step 1800) — 5.249 — **5.239** — 5.469 — −0.010 ← overtakes; 8.2M (done) — 5.155 — **5.136** — 5.383 — −0.019.

**Turning point and readout.** Headline table matches [08.3](../../learning-notes/08-experiments-and-results.md) (steps×bs8×512 → token millions): Final best_val: attention **5.136**, minGRU **5.155**, mamba2 **5.383**. Gap shrinks monotonically then flips between 6.6M and 7.4M — the project’s cleanest scaling result. **Interpretation boundary.** The sign flip in held-out loss is measured; the “bias-to-capacity” explanation is inference. No throughput claim follows from this quality crossover.

**Failures and surprises.** None. Mamba-2 stays last at every checkpoint (bias without enough capacity/throughput story at this scale).

## Decision and aftermath

**Kept:** Bias wins early, capacity wins late — attention overtakes minGRU between 6.6M and 7.4M tokens. The notebook continues with [15-crossover-followups](15-crossover-followups.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 0.8M (step 200) — **6.334** — 6.516 — 6.493 — +0.182.
- The result artifact reports: 4.1M (step 1000) — **5.549** — 5.624 — 5.769 — +0.075.
- The result artifact reports: 6.6M (step 1600) — **5.353** — 5.358 — 5.560 — +0.005 (tied).
- The result artifact reports: 7.4M (step 1800) — 5.249 — **5.239** — 5.469 — −0.010 ← overtakes.
- Failure/operational record: None. Mamba-2 stays last at every checkpoint (bias without enough capacity/throughput story at this scale).

## What this does not prove

**Confidence: High.** Dense matched checkpoints show a monotonic gap reversal, and suite 15 independently reproduces the crossover pattern at two scales; each arm itself uses one seed. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.3
- Related suites: [`13-mixer-bakeoff-2M`](13-mixer-bakeoff-2M.md), [`15-crossover-followups`](15-crossover-followups.md)

---

[Previous](13-mixer-bakeoff-2M.md) · [Index](../00-INDEX.md) · [Next](15-crossover-followups.md)
