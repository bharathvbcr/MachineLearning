# 18: GPU maximization — baseline → max → validate → bs32

## Executive summary

- **Question:** On 8 GB, invisible sysmem thrash is the real limiter; stacking residency/fusion/checkpointing should drive util≈100% and ~2× tok/s without changing the architecture recipe.
- **Result:** On 8 GB, memory residency is the bottleneck — ~25% MFU is the laptop ceiling for 124M.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `18-gpu-maximization` |
| Dates | 2026-06-15 – 2026-06-15 |
| Hardware | RTX 3070 Ti Laptop 8 GB |
| Status | `done` |

## Hypothesis

On 8 GB, invisible sysmem thrash is the real limiter; stacking residency/fusion/checkpointing should drive util≈100% and ~2× tok/s without changing the architecture recipe.

## Setup

- Trainer / preset: `nanolab` — `gpu_baseline`, `gpu_max`, `gpu_opt_validate`, `gpu_opt_bs32`
- Fixed knobs: 12L/768d attention, block 1024, Muon, Shakespeare/short probe data, bf16
- Stack that distinguishes max vs baseline ([08.6](../../learning-notes/08-experiments-and-results.md)): fused linear CE (chunks=16), batched Muon, grad checkpoint, TF32/flash-SDPA, GPU-resident loader, `mem_fraction` 0.92
- Env flags: device `auto`; seed 1337

## Variants

| Variant | Change |
|---------|--------|
| `gpu_baseline` | Pre-opt stack; no fused CE / soft residency |
| `gpu_max` | Full max preset, bs16, 120 steps |
| `gpu_opt_validate` | Validated stack, bs24, mem_fraction 0.92 |
| `gpu_opt_bs32` | Same stack, bs32 push |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | `gpu_opt_bs32` | peak **13.7K tok/s**, MFU **25.5%** | Matches §8.6 headline ceiling |
| 2 | `gpu_opt_validate` | peak **13.6K tok/s**, MFU **25.3%** | best_val 4.808 @ bs24 |
| 3 | `gpu_max` | peak **11.9K tok/s**, MFU **22.1%** | best_val 4.800; mid-stack |
| — | `gpu_baseline` | util/power story in §8.6 | metrics.jsonl only has `start` (run aborted/thrash) — **14% util, ~18 s/step, >8 GB spill** per chapter |

§8.6 session read: baseline thrash (14% util / 57 W / ~18 s/step) → gpu_max preset **96–100% util, 130 W, 13.7K tok/s, 25.5% MFU, 6.1 GB**. Loss curves look fine under thrash; power/util/reserved-mem tell the truth.

**Interpretation boundary.** The optimized-stack throughput/MFU is directly measured. The exact baseline-to-max multiplier is less secure because the baseline run has no train-step metrics in its own JSONL.

## Failures

- `gpu_baseline`: no train steps logged — practical failure mode of oversubscribe + host spill.
- Pushing past ~bs32/ctx1024 with the full stack is the 8 GB cliff (moe from suite 17 already OOMs).

## Lesson

**On 8 GB, memory residency is the bottleneck — ~25% MFU is the laptop ceiling for 124M.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.train`.
- Required replay inputs: the four `nanolab/out/gpu_{baseline,max,opt_validate,opt_bs32}/config.json` files. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Multiple stack variants measured the ceiling, but the baseline lacks train-step metrics and some baseline telemetry is preserved only in the learning-note snapshot.

## Artifacts

- `nanolab/out/gpu_baseline/`, `gpu_max/`, `gpu_opt_validate/`, `gpu_opt_bs32/` — configs + metrics
- Chapter snapshot: [08.6](../../learning-notes/08-experiments-and-results.md)

## Why this experiment happened

The throughput sweeps revealed that kernels and memory behavior, not just model math, governed usable experiments on 8 GB. The baseline’s low utilization and spill signature motivated a stacked residency, fusion, checkpointing, and batching pass. The preceding notebook context is [17-gpu-throughput-sweeps](17-gpu-throughput-sweeps.md).

## Experiment story

**Baseline.** The throughput sweeps revealed that kernels and memory behavior, not just model math, governed usable experiments on 8 GB. The baseline’s low utilization and spill signature motivated a stacked residency, fusion, checkpointing, and batching pass. The preceding notebook context is [17-gpu-throughput-sweeps](17-gpu-throughput-sweeps.md).

**Hypothesis.** On 8 GB, invisible sysmem thrash is the real limiter; stacking residency/fusion/checkpointing should drive util≈100% and ~2× tok/s without changing the architecture recipe.

**Test contract.** Trainer / preset: `nanolab` — `gpu_baseline`, `gpu_max`, `gpu_opt_validate`, `gpu_opt_bs32` Fixed knobs: 12L/768d attention, block 1024, Muon, Shakespeare/short probe data, bf16 Stack that distinguishes max vs baseline ([08.6](../../learning-notes/08-experiments-and-results.md)): fused linear CE (chunks=16), batched Muon, grad checkpoint, TF32/flash-SDPA, GPU-resident loader, `mem_fraction` 0.92 Env flags: device `auto`; seed 1337

**Variant sequence.** The preserved comparison matrix was: `gpu_baseline` — Pre-opt stack; no fused CE / soft residency; `gpu_max` — Full max preset, bs16, 120 steps; `gpu_opt_validate` — Validated stack, bs24, mem_fraction 0.92; `gpu_opt_bs32` — Same stack, bs32 push.

**Measured turn.** The result board records 1 — `gpu_opt_bs32` — peak **13.7K tok/s**, MFU **25.5%** — Matches §8.6 headline ceiling; 2 — `gpu_opt_validate` — peak **13.6K tok/s**, MFU **25.3%** — best_val 4.808 @ bs24; 3 — `gpu_max` — peak **11.9K tok/s**, MFU **22.1%** — best_val 4.800; mid-stack; — — `gpu_baseline` — util/power story in §8.6 — metrics.jsonl only has `start` (run aborted/thrash) — **14% util, ~18 s/step, >8 GB spill** per chapter.

**Turning point and readout.** §8.6 session read: baseline thrash (14% util / 57 W / ~18 s/step) → gpu_max preset **96–100% util, 130 W, 13.7K tok/s, 25.5% MFU, 6.1 GB**. Loss curves look fine under thrash; power/util/reserved-mem tell the truth. **Interpretation boundary.** The optimized-stack throughput/MFU is directly measured. The exact baseline-to-max multiplier is less secure because the baseline run has no train-step metrics in its own JSONL.

**Failures and surprises.** `gpu_baseline`: no train steps logged — practical failure mode of oversubscribe + host spill. Pushing past ~bs32/ctx1024 with the full stack is the 8 GB cliff (moe from suite 17 already OOMs).

## Decision and aftermath

**Kept:** On 8 GB, memory residency is the bottleneck — ~25% MFU is the laptop ceiling for 124M. The notebook continues with [19-chunk-parallel-kernels](19-chunk-parallel-kernels.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `gpu_opt_bs32` — peak **13.7K tok/s**, MFU **25.5%** — Matches §8.6 headline ceiling.
- The result artifact reports: 2 — `gpu_opt_validate` — peak **13.6K tok/s**, MFU **25.3%** — best_val 4.808 @ bs24.
- The result artifact reports: 3 — `gpu_max` — peak **11.9K tok/s**, MFU **22.1%** — best_val 4.800; mid-stack.
- The result artifact reports: — — `gpu_baseline` — util/power story in §8.6 — metrics.jsonl only has `start` (run aborted/thrash) — **14% util, ~18 s/step, >8 GB spill** per chapter.
- Failure/operational record: `gpu_baseline`: no train steps logged — practical failure mode of oversubscribe + host spill.
- Failure/operational record: Pushing past ~bs32/ctx1024 with the full stack is the 8 GB cliff (moe from suite 17 already OOMs).

## What this does not prove

**Confidence: Medium.** Multiple stack variants measured the ceiling, but the baseline lacks train-step metrics and some baseline telemetry is preserved only in the learning-note snapshot. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.6
- Related suites: [`11-phase1-fineweb`](11-phase1-fineweb.md), [`17-gpu-throughput-sweeps`](17-gpu-throughput-sweeps.md), [`20-run128m-long`](20-run128m-long.md)

---

[Previous](17-gpu-throughput-sweeps.md) · [Index](../00-INDEX.md) · [Next](19-chunk-parallel-kernels.md)
