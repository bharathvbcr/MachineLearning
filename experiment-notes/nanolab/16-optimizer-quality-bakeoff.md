# 16: Optimizer quality bakeoff (not throughput)

## Executive summary

- **Question:** At matched FineWeb steps/architecture, quality (best_val) — not tok/s — should separate modern optimizers; throughput ranking in suite 17 is a different question.
- **Result:** Quality ≠ throughput: Lion/AdamW lead this short FineWeb board; Prodigy/Schedule-Free need retuning or are wrong defaults.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `16-optimizer-quality-bakeoff` |
| Dates | 2026-06-16 – 2026-06-16 |
| Hardware | RTX 3070 Ti Laptop 8 GB |
| Status | `done` |

## Hypothesis

At matched FineWeb steps/architecture, quality (best_val) — not tok/s — should separate modern optimizers; throughput ranking in suite 17 is a different question.

## Setup

- Trainer / preset: `nanolab` — `optbake_{lion,adamw,muon,sophia,sgd,schedulefree,prodigy}`
- Fixed knobs: 12L/768d attention, block 512, bs8, 1000 steps (= **4.096M tokens**), FineWeb-edu, bf16, seed 1337, eval every 200
- Env flags: device `auto`
- Per-opt LRs (from configs): lion 1.2e-4; adamw/muon/sophia/schedulefree 6e-4; sgd 0.1; prodigy 1.0

## Variants

| Variant | Change |
|---------|--------|
| `optbake_lion` | Lion |
| `optbake_adamw` | AdamW |
| `optbake_muon` | Muon (matrix_lr 0.025) |
| `optbake_sophia` | Sophia |
| `optbake_sgd` | SGD+momentum |
| `optbake_schedulefree` | Schedule-Free |
| `optbake_prodigy` | Prodigy |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | **Lion** | best_val **5.557** | Saved `best.pt`; edges AdamW |
| 2 | AdamW | best_val **5.565** | Nearly tied with Lion |
| 3 | Muon | best_val **5.643** | +0.086 vs Lion at this short budget |
| 4 | Sophia | best_val **5.846** | Slower early (7.21 @ 200) |
| 5 | SGD | best_val **6.045** | Usable but clearly behind |
| 6 | Schedule-Free | best_val **7.918** | Plateaued high (~8.2–7.9) |
| 7 | Prodigy | best_val **20.16** | Soft-diverged (376→270→792→20) |

At ~4M tokens with attention+SwiGLU, Lion/AdamW win on final val; Muon is close (and elsewhere wins wall-clock once Newton–Schulz is batched — see suites 17–18). Prodigy/Schedule-Free are not drop-in wins at these defaults.

**Interpretation boundary.** Best-val is measured quality at the recorded defaults, not optimizer potential after tuning. Per-step speed from suite 17 is a separate outcome.

## Failures

- **Prodigy:** pathological val (hundreds) for most of training — treat as failed hyperparams, not a fair bakeoff cell.
- Schedule-Free stalled ~2.3 loss above AdamW — effective under-training, not a crash.

## Lesson

**Quality ≠ throughput: Lion/AdamW lead this short FineWeb board; Prodigy/Schedule-Free need retuning or are wrong defaults.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.train`.
- Required replay inputs: the seven `nanolab/out/optbake_*/config.json` files. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Matched one-seed quality runs completed, but optimizer defaults were not equally tuned and Prodigy behaved pathologically.

## Artifacts

- `nanolab/out/optbake_*/config.json`, `metrics.jsonl`
- `nanolab/out/optbake_lion/best.pt`

## Why this experiment happened

The mixer work made token-budget matching central, while upcoming GPU probes risked conflating speed with learning. This suite isolated optimizer quality at fixed architecture, steps, and data before suite 17 measured execution cost separately. The preceding notebook context is [15-crossover-followups](15-crossover-followups.md).

## Experiment story

**Baseline.** The mixer work made token-budget matching central, while upcoming GPU probes risked conflating speed with learning. This suite isolated optimizer quality at fixed architecture, steps, and data before suite 17 measured execution cost separately. The preceding notebook context is [15-crossover-followups](15-crossover-followups.md).

**Hypothesis.** At matched FineWeb steps/architecture, quality (best_val) — not tok/s — should separate modern optimizers; throughput ranking in suite 17 is a different question.

**Test contract.** Trainer / preset: `nanolab` — `optbake_{lion,adamw,muon,sophia,sgd,schedulefree,prodigy}` Fixed knobs: 12L/768d attention, block 512, bs8, 1000 steps (= **4.096M tokens**), FineWeb-edu, bf16, seed 1337, eval every 200 Env flags: device `auto` Per-opt LRs (from configs): lion 1.2e-4; adamw/muon/sophia/schedulefree 6e-4; sgd 0.1; prodigy 1.0

**Variant sequence.** The preserved comparison matrix was: `optbake_lion` — Lion; `optbake_adamw` — AdamW; `optbake_muon` — Muon (matrix_lr 0.025); `optbake_sophia` — Sophia; `optbake_sgd` — SGD+momentum; `optbake_schedulefree` — Schedule-Free.

**Measured turn.** The result board records 1 — **Lion** — best_val **5.557** — Saved `best.pt`; edges AdamW; 2 — AdamW — best_val **5.565** — Nearly tied with Lion; 3 — Muon — best_val **5.643** — +0.086 vs Lion at this short budget; 4 — Sophia — best_val **5.846** — Slower early (7.21 @ 200); 5 — SGD — best_val **6.045** — Usable but clearly behind.

**Turning point and readout.** At ~4M tokens with attention+SwiGLU, Lion/AdamW win on final val; Muon is close (and elsewhere wins wall-clock once Newton–Schulz is batched — see suites 17–18). Prodigy/Schedule-Free are not drop-in wins at these defaults. **Interpretation boundary.** Best-val is measured quality at the recorded defaults, not optimizer potential after tuning. Per-step speed from suite 17 is a separate outcome.

**Failures and surprises.** **Prodigy:** pathological val (hundreds) for most of training — treat as failed hyperparams, not a fair bakeoff cell. Schedule-Free stalled ~2.3 loss above AdamW — effective under-training, not a crash.

## Decision and aftermath

**Kept:** Quality ≠ throughput: Lion/AdamW lead this short FineWeb board; Prodigy/Schedule-Free need retuning or are wrong defaults. The notebook continues with [17-gpu-throughput-sweeps](17-gpu-throughput-sweeps.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — **Lion** — best_val **5.557** — Saved `best.pt`; edges AdamW.
- The result artifact reports: 2 — AdamW — best_val **5.565** — Nearly tied with Lion.
- The result artifact reports: 3 — Muon — best_val **5.643** — +0.086 vs Lion at this short budget.
- The result artifact reports: 4 — Sophia — best_val **5.846** — Slower early (7.21 @ 200).
- Failure/operational record: **Prodigy:** pathological val (hundreds) for most of training — treat as failed hyperparams, not a fair bakeoff cell.
- Failure/operational record: Schedule-Free stalled ~2.3 loss above AdamW — effective under-training, not a crash.

## What this does not prove

**Confidence: Medium.** Matched one-seed quality runs completed, but optimizer defaults were not equally tuned and Prodigy behaved pathologically. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.4
- Related suites: [`17-gpu-throughput-sweeps`](17-gpu-throughput-sweeps.md), [`18-gpu-maximization`](18-gpu-maximization.md)

---

[Previous](15-crossover-followups.md) · [Index](../00-INDEX.md) · [Next](17-gpu-throughput-sweeps.md)
