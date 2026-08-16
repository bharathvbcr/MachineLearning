# 15: Crossover follow-ups — xs_* + ext_*

## Executive summary

- **Question:** The ~7M crossover should be robust to model scale: a smaller (xs) and a longer (ext) FineWeb A/B of attention vs minGRU should still show early minGRU lead and late attention overtake.
- **Result:** The attention↔minGRU crossover is robust across model size — only the token timing of the flip changes.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `15-crossover-followups` |
| Dates | 2026-06-15 – 2026-06-16 |
| Hardware | RTX 3070 Ti Laptop 8 GB |
| Status | `done` |

## Hypothesis

The ~7M crossover should be robust to model scale: a smaller (xs) and a longer (ext) FineWeb A/B of attention vs minGRU should still show early minGRU lead and late attention overtake.

## Setup

- Trainer / preset: `nanolab` — `xs_{attention,mingru}`, `ext_{attention,mingru}`
- Fixed knobs:
  - **xs_***: 6L/384d, block 512, bs16, Muon lr 6e-4, 2000 steps → **16.4M tokens**, FineWeb-edu, eval every 200
  - **ext_***: 12L/768d, block 512, bs16, Muon lr 6e-4, 3700 steps → **30.3M tokens**, eval every 300
- Env flags: device `auto`; seed 1337; bf16; grad checkpoint

## Variants

| Variant | Change |
|---------|--------|
| `xs_mingru` / `xs_attention` | Smaller width/depth, same lever = mixer |
| `ext_mingru` / `ext_attention` | Full 124M, longer schedule |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | `ext_attention` | best_val **4.260** @ 30.3M tok | Overtakes by ~step 900; gap ~−0.32 late |
| 2 | `ext_mingru` | best_val **4.575** | Led early (5.83 vs 6.02 @ 300) then fades |
| 3 | `xs_attention` | best_val **4.852** @ 16.4M tok | Overtakes ~step 1000 (A−G −0.013) |
| 4 | `xs_mingru` | best_val **5.021** | Early lead (+0.15 @ 200) → −0.16 gap by 1800 |

**xs gap (A−G):** +0.148 → +0.012 → **−0.013** (step 1000) → −0.155 (1800).  
**ext gap (A−G):** +0.196 → **−0.024** (step 900) → ~−0.32 plateau from ~2.1k–3.6k.

Same story as suite 14, confirmed at half width and at ~4× the original scale token budget: recurrent bias helps first; attention’s capacity wins once data is plentiful.

**Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

## Failures

- None. Both pairs completed full schedules with dense eval curves.

## Lesson

**The attention↔minGRU crossover is robust across model size — only the token timing of the flip changes.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.train`.
- Required replay inputs: the four `nanolab/out/{xs,ext}_*/config.json` files. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Two matched scale pairs completed with dense curves, but each arm is single-seed and crossover timing is setup-dependent.

## Artifacts

- `nanolab/out/xs_attention/`, `xs_mingru/`
- `nanolab/out/ext_attention/`, `ext_mingru/` — `config.json`, `metrics.jsonl` (+ `best.pt` where present)

## Why this experiment happened

Suite 14 produced the project’s sharpest reversal: attention crossed minGRU between 6.6M and 7.4M tokens. Two follow-up pairs were needed to determine whether that was a single-scale accident or a repeatable early-bias/late-capacity pattern. The preceding notebook context is [14-scale-crossover-8M](14-scale-crossover-8M.md).

## Experiment story

**Baseline.** Suite 14 produced the project’s sharpest reversal: attention crossed minGRU between 6.6M and 7.4M tokens. Two follow-up pairs were needed to determine whether that was a single-scale accident or a repeatable early-bias/late-capacity pattern. The preceding notebook context is [14-scale-crossover-8M](14-scale-crossover-8M.md).

**Hypothesis.** The ~7M crossover should be robust to model scale: a smaller (xs) and a longer (ext) FineWeb A/B of attention vs minGRU should still show early minGRU lead and late attention overtake.

**Test contract.** Trainer / preset: `nanolab` — `xs_{attention,mingru}`, `ext_{attention,mingru}` Fixed knobs: - **xs_***: 6L/384d, block 512, bs16, Muon lr 6e-4, 2000 steps → **16.4M tokens**, FineWeb-edu, eval every 200 - **ext_***: 12L/768d, block 512, bs16, Muon lr 6e-4, 3700 steps → **30.3M tokens**, eval every 300

**Variant sequence.** The preserved comparison matrix was: `xs_mingru` / `xs_attention` — Smaller width/depth, same lever = mixer; `ext_mingru` / `ext_attention` — Full 124M, longer schedule.

**Measured turn.** The result board records 1 — `ext_attention` — best_val **4.260** @ 30.3M tok — Overtakes by ~step 900; gap ~−0.32 late; 2 — `ext_mingru` — best_val **4.575** — Led early (5.83 vs 6.02 @ 300) then fades; 3 — `xs_attention` — best_val **4.852** @ 16.4M tok — Overtakes ~step 1000 (A−G −0.013); 4 — `xs_mingru` — best_val **5.021** — Early lead (+0.15 @ 200) → −0.16 gap by 1800.

**Turning point and readout.** **xs gap (A−G):** +0.148 → +0.012 → **−0.013** (step 1000) → −0.155 (1800). **ext gap (A−G):** +0.196 → **−0.024** (step 900) → ~−0.32 plateau from ~2.1k–3.6k. Same story as suite 14, confirmed at half width and at ~4× the original scale token budget: recurrent bias helps first; attention’s capacity wins once data is plentiful. **Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

**Failures and surprises.** None. Both pairs completed full schedules with dense eval curves.

## Decision and aftermath

**Kept:** The attention↔minGRU crossover is robust across model size — only the token timing of the flip changes. The notebook continues with [16-optimizer-quality-bakeoff](16-optimizer-quality-bakeoff.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `ext_attention` — best_val **4.260** @ 30.3M tok — Overtakes by ~step 900; gap ~−0.32 late.
- The result artifact reports: 2 — `ext_mingru` — best_val **4.575** — Led early (5.83 vs 6.02 @ 300) then fades.
- The result artifact reports: 3 — `xs_attention` — best_val **4.852** @ 16.4M tok — Overtakes ~step 1000 (A−G −0.013).
- The result artifact reports: 4 — `xs_mingru` — best_val **5.021** — Early lead (+0.15 @ 200) → −0.16 gap by 1800.
- Failure/operational record: None. Both pairs completed full schedules with dense eval curves.

## What this does not prove

**Confidence: Medium.** Two matched scale pairs completed with dense curves, but each arm is single-seed and crossover timing is setup-dependent. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.3
- Related suites: [`14-scale-crossover-8M`](14-scale-crossover-8M.md), [`13-mixer-bakeoff-2M`](13-mixer-bakeoff-2M.md)

---

[Previous](14-scale-crossover-8M.md) · [Index](../00-INDEX.md) · [Next](16-optimizer-quality-bakeoff.md)
