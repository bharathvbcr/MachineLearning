# 21: Diffusion LM — phase0 adapt + 128M block32

## Executive summary

- **Question:** An autoregressive checkpoint can be converted to a masked-diffusion LM (causal→bidirectional anneal, absorbing-[MASK], 1/t reweighting) and adapt on TinyStories with falling val perplexity — proving the stack transfers across decoding paradigms.
- **Result:** AR stacks transfer to masked diffusion — but the loss must target clean tokens, or it collapses to 0.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `21-diffusion-lm` |
| Dates | 2026-06-15 – 2026-06-16 |
| Hardware | RTX 3070 Ti Laptop 8 GB |
| Status | `done` |

## Hypothesis

An autoregressive checkpoint can be converted to a masked-diffusion LM (causal→bidirectional anneal, absorbing-[MASK], 1/t reweighting) and adapt on TinyStories with falling val perplexity — proving the stack transfers across decoding paradigms.

## Setup

- Trainer / preset: `nanolab/diffusion.py` → `diffusion_phase0`, `diffusion128_block32`
- Fixed knobs:
  - **diffusion_phase0:** 6L/384d from phase0-class AR, TinyStories, bs24, Muon lr 1e-3, block 256, 1500 steps, eval every 300
  - **diffusion128_block32:** 12L/768d, FineWeb-edu, AdamW lr 2e-4, bs4, 700 steps, `diffusion_block_len=32`, complementary masking, CUDA
- Env flags: MASK id reuse (GPT-2 pad slot); confidence-based parallel decode for gen benches
- Gen bench: `nanolab/out/bench_block_scale.txt` (block_len=32, cached vs uncached)

## Variants

| Variant | Change |
|---------|--------|
| `diffusion_phase0` | AR→diffusion adapt on TinyStories |
| `diffusion128_block32` | 124M FineWeb diffusion with block 32 |
| `bench_block_scale` | Decode throughput vs gen_len |

## Results

Matches [08.7](../../learning-notes/08-experiments-and-results.md) / file 15 curve; `best_val` 2.103 ⇒ ppl ≈ **8.19**:

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | `diffusion_phase0` | val_ppl **19.49 → 9.16** (steps 300→1200); best_val **2.103** (~**8.2** ppl) | ~7 min adapt; coherent iterative decode |
| 2 | `diffusion128_block32` | best_val **5.467** (ppl 254 @ step 525 → improving) | Short FineWeb diffusion; incomplete relative to phase0 story |
| — | block decode | 68–121 tok/s | gen_len 128–896 @ block32; cache helps at longer lens |

Phase0 eval ladder: 19.49 → 13.03 → 10.62 → 9.16, with final best_val implying ~8.2 ppl — the headline conversion result.

**Interpretation boundary.** Perplexity improvement and decode tok/s measure different axes. The phase0 quality result does not imply that diffusion decoding is faster than autoregressive decoding.

## Failures

- **Loss→0 bug (caught in logs):** targeting masked input instead of clean tokens makes CE trivial; fixed by `diffusion_loss(..., x_clean, ...)`. Classic “zero is a scream” debug lesson ([16](../../learning-notes/16-debugging-and-failure-modes.md)).
- `diffusion128_block32`: multiple `start` events in metrics (restarts); quality still high-ppl vs phase0 TinyStories adapt.

## Lesson

**AR stacks transfer to masked diffusion — but the loss must target clean tokens, or it collapses to 0.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.diffusion train`.
- Required replay inputs: `nanolab/out/diffusion_phase0/config.json` and `nanolab/out/diffusion128_block32/config.json`, plus the `--init` checkpoint for adaptation. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** The phase0 adaptation completed with a descending eval ladder; the 128M arm is short/restarted and decode speed is not a quality measure.

## Artifacts

- `nanolab/out/diffusion_phase0/config.json`, `metrics.jsonl`, `best.pt`
- `nanolab/out/diffusion128_block32/config.json`, `metrics.jsonl`, `best.pt`
- `nanolab/out/bench_block_scale.txt`
- `nanolab/diffusion.py`

## Why this experiment happened

The completed autoregressive stack provided a checkpoint, instrumentation, and debugging discipline that could be stress-tested under a different objective. The diffusion suite asked whether those assets transferred to masked, bidirectional denoising and parallel generation. The preceding notebook context is [20-run128m-long](20-run128m-long.md).

## Experiment story

**Baseline.** The completed autoregressive stack provided a checkpoint, instrumentation, and debugging discipline that could be stress-tested under a different objective. The diffusion suite asked whether those assets transferred to masked, bidirectional denoising and parallel generation. The preceding notebook context is [20-run128m-long](20-run128m-long.md).

**Hypothesis.** An autoregressive checkpoint can be converted to a masked-diffusion LM (causal→bidirectional anneal, absorbing-[MASK], 1/t reweighting) and adapt on TinyStories with falling val perplexity — proving the stack transfers across decoding paradigms.

**Test contract.** Trainer / preset: `nanolab/diffusion.py` → `diffusion_phase0`, `diffusion128_block32` Fixed knobs: - **diffusion_phase0:** 6L/384d from phase0-class AR, TinyStories, bs24, Muon lr 1e-3, block 256, 1500 steps, eval every 300 - **diffusion128_block32:** 12L/768d, FineWeb-edu, AdamW lr 2e-4, bs4, 700 steps, `diffusion_block_len=32`, complementary masking, CUDA

**Variant sequence.** The preserved comparison matrix was: `diffusion_phase0` — AR→diffusion adapt on TinyStories; `diffusion128_block32` — 124M FineWeb diffusion with block 32; `bench_block_scale` — Decode throughput vs gen_len.

**Measured turn.** The result board records 1 — `diffusion_phase0` — val_ppl **19.49 → 9.16** (steps 300→1200); best_val **2.103** (~**8.2** ppl) — ~7 min adapt; coherent iterative decode; 2 — `diffusion128_block32` — best_val **5.467** (ppl 254 @ step 525 → improving) — Short FineWeb diffusion; incomplete relative to phase0 story; — — block decode — 68–121 tok/s — gen_len 128–896 @ block32; cache helps at longer lens.

**Turning point and readout.** Matches [08.7](../../learning-notes/08-experiments-and-results.md) / file 15 curve; `best_val` 2.103 ⇒ ppl ≈ **8.19**: Phase0 eval ladder: 19.49 → 13.03 → 10.62 → 9.16, with final best_val implying ~8.2 ppl — the headline conversion result. **Interpretation boundary.** Perplexity improvement and decode tok/s measure different axes. The phase0 quality result does not imply that diffusion decoding is faster than autoregressive decoding.

**Failures and surprises.** **Loss→0 bug (caught in logs):** targeting masked input instead of clean tokens makes CE trivial; fixed by `diffusion_loss(..., x_clean, ...)`. Classic “zero is a scream” debug lesson ([16](../../learning-notes/16-debugging-and-failure-modes.md)). `diffusion128_block32`: multiple `start` events in metrics (restarts); quality still high-ppl vs phase0 TinyStories adapt.

## Decision and aftermath

**Kept:** AR stacks transfer to masked diffusion — but the loss must target clean tokens, or it collapses to 0. The notebook continues with [30-phase0-runtime-baselines](../gemma-metal/30-phase0-runtime-baselines.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `diffusion_phase0` — val_ppl **19.49 → 9.16** (steps 300→1200); best_val **2.103** (~**8.2** ppl) — ~7 min adapt; coherent iterative decode.
- The result artifact reports: 2 — `diffusion128_block32` — best_val **5.467** (ppl 254 @ step 525 → improving) — Short FineWeb diffusion; incomplete relative to phase0 story.
- The result artifact reports: — — block decode — 68–121 tok/s — gen_len 128–896 @ block32; cache helps at longer lens.
- Failure/operational record: **Loss→0 bug (caught in logs):** targeting masked input instead of clean tokens makes CE trivial; fixed by `diffusion_loss(..., x_clean, ...)`. Classic “zero is a scream” debug lesson ([16](../../learning-notes/16-debugging-and-failure-modes.md)).
- Failure/operational record: `diffusion128_block32`: multiple `start` events in metrics (restarts); quality still high-ppl vs phase0 TinyStories adapt.

## What this does not prove

**Confidence: Medium.** The phase0 adaptation completed with a descending eval ladder; the 128M arm is short/restarted and decode speed is not a quality measure. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.7
- [learning-notes/15-diffusion-language-models.md](../../learning-notes/15-diffusion-language-models.md)
- Related suites: [`10-phase0-smoke`](10-phase0-smoke.md)

---

[Previous](20-run128m-long.md) · [Index](../00-INDEX.md) · [Next](22-gh200-crossover-50m.md)
