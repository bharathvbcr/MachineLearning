# 20: Long 124M FineWeb runs — 2k / 10k / 20k

## Executive summary

- **Question:** With the gpu_max stack held fixed, extending a 124M attention LM on FineWeb-edu should produce a clean long loss curve and quantify how far laptop training can push best_val.
- **Result:** Longer isn’t automatically better — 10k @ 6e-4 beats the lower-LR 20k continuation on best_val.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `20-run128m-long` |
| Dates | 2026-06-15 – 2026-06-18 |
| Hardware | RTX 3070 Ti Laptop 8 GB |
| Status | `done` |

## Hypothesis

With the gpu_max stack held fixed, extending a 124M attention LM on FineWeb-edu should produce a clean long loss curve and quantify how far laptop training can push best_val.

## Setup

- Trainer / preset: `nanolab` — `run128m_fineweb_2k`, `run128m_10k`, `run128m_20k`
- Fixed knobs: 12L/768d attention, SwiGLU, block 1024, bs32, Muon, FineWeb-edu, bf16, grad checkpoint, `mem_fraction` 0.92, seed 1337
- Schedule differences:
  - **2k:** lr 6e-4, warmup 150, 2000 steps → **65.5M tokens**
  - **10k:** lr 6e-4, warmup 500, 10000 steps → **327.7M tokens**
  - **20k:** lr 1.2e-4, matrix_lr 0.005, warmup 100, 20000 steps (lower-LR continuation-style), `diffusion_mode=none`
- Env flags: device `auto`

## Variants

| Variant | Change |
|---------|--------|
| `run128m_fineweb_2k` | 2k-step short long-run |
| `run128m_10k` | 10k-step mid |
| `run128m_20k` | 20k-step, reduced LR |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | `run128m_10k` | best_val **3.621** @ step 9000 | Peak MFU ~25.9%, ~13.9K tok/s; 328M tokens seen |
| 2 | `run128m_20k` | best_val **3.607** @ step 4500 | Then **regressed** to ~3.79 by step 19500 under lower LR |
| 3 | `run128m_fineweb_2k` | best_val **3.877** | 65.5M tok; on-ramp to the longer runs |

2k→10k is the real quality climb (3.88→3.62). The 20k config starts from a stronger early point (val already ~3.67 @ 2k) but the reduced LR schedule does not improve the global best past ~3.61 and drifts upward — treat 10k’s 3.621 as the reliable laptop long-run scoreboard winner.

**Interpretation boundary.** The 10k and 20k values are measured, but their learning rates and schedules differ; attributing the regression to horizon alone would be a contradiction.

> **Correction (2026-08-22): `run128m_20k` is eight resumed segments, not one run, and the
> ranking above depends on which number you read.** Its `metrics.jsonl` holds **9 `start` and
> 8 `done` events**. Per-segment `best_val`:
>
> ```
> 3.6279  3.5869  3.5806  3.5854  3.5928  3.5998  3.6059  3.6109
> ```
>
> Two consequences the original table did not state:
>
> 1. **The global minimum across segments is 3.5806**, which is *better* than
>    `run128m_10k`'s 3.621. Ranked on global minimum, `run128m_20k` wins; ranked on final
>    value, `run128m_10k` wins. The note asserts the second ordering without saying a
>    different reading exists. Neither is a controlled comparison — LR (6e-4 vs 1.2e-4),
>    `matrix_lr` (0.025 vs 0.005) and warmup (500 vs 100) all move with the horizon.
> 2. **The monotone drift `3.5806 → 3.6109` across the last six segments is a
>    resume artifact, not established late-train degradation.** Each resume re-warms the LR
>    and re-seeds the EMA, so the "regressed to ~3.79 by step 19500" reading conflates the
>    schedule with the restart boundaries.
>
> **Token accounting for this run is also wrong.** Every `done` event records
> `tokens: 98,304,000` (= 3000 steps × bs32 × ctx1024) although the run reached step
> **19,990**; 20,000 steps at this shape is **655,360,000** tokens. The counter reports the
> current segment, not the cumulative total, so any tokens-seen figure for `run128m_20k` is
> unusable.
>
> **This suite cannot support a horizon claim in either direction as run.** Closing it needs
> one uninterrupted 20k run at the 10k learning rate.

## Failures

- `run128m_20k`: late-train degradation (best early, worse final) — LR/schedule mismatch, not a crash.
- No OOMs at bs32/ctx1024 with the max stack.

## Lesson

**Longer isn’t automatically better — 10k @ 6e-4 beats the lower-LR 20k continuation on best_val.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.train`.
- Required replay inputs: the three `nanolab/out/run128m_*/config.json` files; schedules differ and must not be merged. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Long single-seed curves completed, but schedules differ, so the 10k-versus-20k result is a recipe comparison rather than a pure horizon A/B.

## Artifacts

- `nanolab/out/run128m_fineweb_2k/config.json`, `metrics.jsonl`
- `nanolab/out/run128m_10k/config.json`, `metrics.jsonl`
- `nanolab/out/run128m_20k/config.json`, `metrics.jsonl`, `best.pt`

## Why this experiment happened

After suite 18 made bs32/ctx1024 resident and suite 19 removed the worst mixer-kernel blockers, the project could ask how far a laptop 124M attention run actually improved with duration. The 2k, 10k, and lower-LR 20k recipes record that long-horizon attempt. The preceding notebook context is [19-chunk-parallel-kernels](19-chunk-parallel-kernels.md).

## Experiment story

**Baseline.** After suite 18 made bs32/ctx1024 resident and suite 19 removed the worst mixer-kernel blockers, the project could ask how far a laptop 124M attention run actually improved with duration. The 2k, 10k, and lower-LR 20k recipes record that long-horizon attempt. The preceding notebook context is [19-chunk-parallel-kernels](19-chunk-parallel-kernels.md).

**Hypothesis.** With the gpu_max stack held fixed, extending a 124M attention LM on FineWeb-edu should produce a clean long loss curve and quantify how far laptop training can push best_val.

**Test contract.** Trainer / preset: `nanolab` — `run128m_fineweb_2k`, `run128m_10k`, `run128m_20k` Fixed knobs: 12L/768d attention, SwiGLU, block 1024, bs32, Muon, FineWeb-edu, bf16, grad checkpoint, `mem_fraction` 0.92, seed 1337 Schedule differences: - **2k:** lr 6e-4, warmup 150, 2000 steps → **65.5M tokens**

**Variant sequence.** The preserved comparison matrix was: `run128m_fineweb_2k` — 2k-step short long-run; `run128m_10k` — 10k-step mid; `run128m_20k` — 20k-step, reduced LR.

**Measured turn.** The result board records 1 — `run128m_10k` — best_val **3.621** @ step 9000 — Peak MFU ~25.9%, ~13.9K tok/s; 328M tokens seen; 2 — `run128m_20k` — best_val **3.607** @ step 4500 — Then **regressed** to ~3.79 by step 19500 under lower LR; 3 — `run128m_fineweb_2k` — best_val **3.877** — 65.5M tok; on-ramp to the longer runs.

**Turning point and readout.** 2k→10k is the real quality climb (3.88→3.62). The 20k config starts from a stronger early point (val already ~3.67 @ 2k) but the reduced LR schedule does not improve the global best past ~3.61 and drifts upward — treat 10k’s 3.621 as the reliable laptop long-run scoreboard winner. **Interpretation boundary.** The 10k and 20k values are measured, but their learning rates and schedules differ; attributing the regression to horizon alone would be a contradiction.

**Failures and surprises.** `run128m_20k`: late-train degradation (best early, worse final) — LR/schedule mismatch, not a crash. No OOMs at bs32/ctx1024 with the max stack.

## Decision and aftermath

**Kept:** Longer isn’t automatically better — 10k @ 6e-4 beats the lower-LR 20k continuation on best_val. The notebook continues with [21-diffusion-lm](21-diffusion-lm.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `run128m_10k` — best_val **3.621** @ step 9000 — Peak MFU ~25.9%, ~13.9K tok/s; 328M tokens seen.
- The result artifact reports: 2 — `run128m_20k` — best_val **3.607** @ step 4500 — Then **regressed** to ~3.79 by step 19500 under lower LR.
- The result artifact reports: 3 — `run128m_fineweb_2k` — best_val **3.877** — 65.5M tok; on-ramp to the longer runs.
- Failure/operational record: `run128m_20k`: late-train degradation (best early, worse final) — LR/schedule mismatch, not a crash.
- Failure/operational record: No OOMs at bs32/ctx1024 with the max stack.

## What this does not prove

**Confidence: Medium.** Long single-seed curves completed, but schedules differ, so the 10k-versus-20k result is a recipe comparison rather than a pure horizon A/B. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.8
- Related suites: [`18-gpu-maximization`](18-gpu-maximization.md), [`11-phase1-fineweb`](11-phase1-fineweb.md)

---

[Previous](19-chunk-parallel-kernels.md) · [Index](../00-INDEX.md) · [Next](21-diffusion-lm.md)
