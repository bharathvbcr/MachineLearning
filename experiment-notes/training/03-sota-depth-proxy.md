# sota_depth_proxy: Depth × width × MLP tradeoffs

## Executive summary

- **Question:** On the local toy `sota` proxy, increasing depth / tweaking width×MLP (with optional LR-up) would beat the lean 4L×128 control at calibrated BPB, justifying a deeper H100 shape.
- **Result:** On the 3070 Ti depth proxy, LR-up on the lean 4L×128 control (calibrated BPB 2.0715) beats an 8L×128 candidate that is nearly 2× larger and slower — short-horizon depth leads reverse at long.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **High**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `sota_depth_proxy` |
| Dates | `2026-03-31` – `2026-03-31` |
| Hardware | RTX 3070 Ti (local ablation via `run_ablation_3070ti.py`) |
| Status | `done` |

## Hypothesis

On the local toy `sota` proxy, increasing depth / tweaking width×MLP (with optional LR-up) would beat the lean 4L×128 control at calibrated BPB, justifying a deeper H100 shape.

## Setup

- Trainer / preset: `base_preset: sota` → `train_gpt_sprint.py` (base control = 4 layers, dim 128, MLP×3)
- Fixed knobs: toy batch/seq as in `SOTA_CONFIG`; ranking metric `calibrated_bpb` with tie-breakers `sliding_bpb`, `artifact_bytes`, `step_avg_ms`
- Stages: short 300 (promote 4) → mid 1000 (promote 2) → long 3000 seeds `[1337, 42]`
- Env flags: architecture overrides per candidate; `latest_summary.txt` reports sanitized inherited env: (none)

## Variants

| Variant | Change |
|---------|--------|
| `control` | (none) — 4L / 128 / MLP×3 |
| `control_lr_up` | `MATRIX_LR=0.027`, `SCALAR_LR=0.027` |
| `d6_128_m3` | `NUM_LAYERS=6`, `MODEL_DIM=128`, `MLP_MULT=3.0` |
| `d6_128_m3_lr_up` | above + LR 0.027 |
| `d8_112_m3` | `NUM_LAYERS=8`, `MODEL_DIM=112`, `MLP_MULT=3.0` |
| `d8_112_m3_lr_up` | above + LR 0.027 |
| `d8_128_m26` | `NUM_LAYERS=8`, `MODEL_DIM=128`, `MLP_MULT=2.6` |
| `d10_96_m3` | `NUM_LAYERS=10`, `MODEL_DIM=96`, `MLP_MULT=3.0` |

## Results

Short (seed 1337) — deeper shapes lead early:

| Rank | Run / config | calibrated_bpb | artifact_bytes |
|------|--------------|----------------|----------------|
| 1 | `d6_128_m3_lr_up` | 2.698550 | 1,849,309 |
| 2 | `d8_128_m26` | 2.701922 | 2,188,945 |
| 3 | `control_lr_up` | 2.711021 | 1,315,049 |
| 4 | `d6_128_m3` | 2.717973 | 1,848,677 |
| … | `d10_96_m3` | 2.776570 | 1,726,757 (worst) |

Mid (seed 1337) — `d8_128_m26` briefly on top:

| Rank | Run / config | calibrated_bpb | step_avg_ms |
|------|--------------|----------------|-------------|
| 1 | `d8_128_m26` | 2.221561 | 1024.96 |
| 2 | `control_lr_up` | 2.231641 | 590.31 |
| 3 | `d6_128_m3_lr_up` | 2.234434 | 846.14 |
| 4 | `d6_128_m3` | 2.235309 | 810.55 |

Long (seeds 1337+42) — champion board:

| Rank | Run / config | calibrated_bpb | Notes |
|------|--------------|----------------|-------|
| 1 | `control_lr_up` | **2.071515** | sliding 2.073998; step **621.86** ms; artifact **1,333,843** B; rcs `[0,0]` |
| 2 | `d8_128_m26` | 2.072760 | Δ+0.001245; sliding 2.076759; step **1098.43** ms; artifact **2,209,613** B |

Champion.json: `control_lr_up`, calibrated_bpb **2.071514733558084**, sliding_bpb **2.07399818**, step_avg_ms **621.86**, artifact_bytes **1333843**.

Readout: short/mid favored deeper nets, but at 3000 steps the compact `control_lr_up` edges `d8_128_m26` by ~0.001 BPB while being ~1.7× smaller and ~1.8× faster per step. Depth/width bumps did not justify their cost on this local proxy.

**Interpretation boundary.** The 0.001245 BPB edge is too small to call a robust quality win by itself; the measured ~1.8× step-time and ~1.7× artifact advantages make the compact control the clear efficiency choice on this proxy.

## Failures

- All champion/long returncodes `[0, 0]`.
- Ancillary files `overlapped_check.*.txt` / `torch_check.*.txt` are harness smoke checks (empty/near-empty), not run failures.
- A `rerun.stdout.txt` exists from a March 31 rerun; board above is from the written `*.summary.json` / `champion.json`.

## Lesson

**On the 3070 Ti depth proxy, LR-up on the lean 4L×128 control (calibrated BPB 2.0715) beats an 8L×128 candidate that is nearly 2× larger and slower — short-horizon depth leads reverse at long.**

## Reproduction

- Replay: `python3 parameter-golf/run_ablation_3070ti.py sota_depth_proxy --resume`.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: High.** Short, mid, and two-seed long stages completed with zero return codes. The conclusion is still limited to the local toy proxy.

## Artifacts

- `parameter-golf/logs/ablations/sota_depth_proxy/champion.json`
- `parameter-golf/logs/ablations/sota_depth_proxy/long.summary.json` / `.txt`
- `parameter-golf/logs/ablations/sota_depth_proxy/mid.summary.json`
- `parameter-golf/logs/ablations/sota_depth_proxy/short.summary.json`
- `parameter-golf/logs/ablations/sota_depth_proxy/latest_summary.txt`
- `parameter-golf/conductor/ablation_suites_3070ti.json` → key `sota_depth_proxy`

## Why this experiment happened

After the lean-head result and the incomplete LR follow-up, the next expensive choice was model shape. The project needed evidence about whether a deeper H100 candidate was worth carrying forward, but only a 3070 Ti proxy was locally available. The preceding notebook context is [02-sota-lean-followup](02-sota-lean-followup.md).

## Experiment story

**Baseline.** After the lean-head result and the incomplete LR follow-up, the next expensive choice was model shape. The project needed evidence about whether a deeper H100 candidate was worth carrying forward, but only a 3070 Ti proxy was locally available. The preceding notebook context is [02-sota-lean-followup](02-sota-lean-followup.md).

**Hypothesis.** On the local toy `sota` proxy, increasing depth / tweaking width×MLP (with optional LR-up) would beat the lean 4L×128 control at calibrated BPB, justifying a deeper H100 shape.

**Test contract.** Trainer / preset: `base_preset: sota` → `train_gpt_sprint.py` (base control = 4 layers, dim 128, MLP×3) Fixed knobs: toy batch/seq as in `SOTA_CONFIG`; ranking metric `calibrated_bpb` with tie-breakers `sliding_bpb`, `artifact_bytes`, `step_avg_ms` Stages: short 300 (promote 4) → mid 1000 (promote 2) → long 3000 seeds `[1337, 42]` Env flags: architecture overrides per candidate; `latest_summary.txt` reports sanitized inherited env: (none)

**Variant sequence.** The preserved comparison matrix was: `control` — (none) — 4L / 128 / MLP×3; `control_lr_up` — `MATRIX_LR=0.027`, `SCALAR_LR=0.027`; `d6_128_m3` — `NUM_LAYERS=6`, `MODEL_DIM=128`, `MLP_MULT=3.0`; `d6_128_m3_lr_up` — above + LR 0.027; `d8_112_m3` — `NUM_LAYERS=8`, `MODEL_DIM=112`, `MLP_MULT=3.0`; `d8_112_m3_lr_up` — above + LR 0.027.

**Measured turn.** The result board records 1 — `d6_128_m3_lr_up` — 2.698550 — 1,849,309; 2 — `d8_128_m26` — 2.701922 — 2,188,945; 3 — `control_lr_up` — 2.711021 — 1,315,049; 4 — `d6_128_m3` — 2.717973 — 1,848,677; … — `d10_96_m3` — 2.776570 — 1,726,757 (worst).

**Turning point and readout.** Champion.json: `control_lr_up`, calibrated_bpb **2.071514733558084**, sliding_bpb **2.07399818**, step_avg_ms **621.86**, artifact_bytes **1333843**. Readout: short/mid favored deeper nets, but at 3000 steps the compact `control_lr_up` edges `d8_128_m26` by ~0.001 BPB while being ~1.7× smaller and ~1.8× faster per step. Depth/width bumps did not justify their cost on this local proxy. **Interpretation boundary.** The 0.001245 BPB edge is too small to call a robust quality win by itself; the measured ~1.8× step-time and ~1.7× artifact advantages make the compact control the clear efficiency choice on this proxy.

**Failures and surprises.** All champion/long returncodes `[0, 0]`. Ancillary files `overlapped_check.*.txt` / `torch_check.*.txt` are harness smoke checks (empty/near-empty), not run failures. A `rerun.stdout.txt` exists from a March 31 rerun; board above is from the written `*.summary.json` / `champion.json`.

## Decision and aftermath

**Kept:** On the 3070 Ti depth proxy, LR-up on the lean 4L×128 control (calibrated BPB 2.0715) beats an 8L×128 candidate that is nearly 2× larger and slower — short-horizon depth leads reverse at long. The notebook continues with [04-sota-arch-ladder](04-sota-arch-ladder.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `d6_128_m3_lr_up` — 2.698550 — 1,849,309.
- The result artifact reports: 2 — `d8_128_m26` — 2.701922 — 2,188,945.
- The result artifact reports: 3 — `control_lr_up` — 2.711021 — 1,315,049.
- The result artifact reports: 4 — `d6_128_m3` — 2.717973 — 1,848,677.
- Failure/operational record: All champion/long returncodes `[0, 0]`.
- Failure/operational record: Ancillary files `overlapped_check.*.txt` / `torch_check.*.txt` are harness smoke checks (empty/near-empty), not run failures.

## What this does not prove

**Confidence: High.** Short, mid, and two-seed long stages completed with zero return codes. The conclusion is still limited to the local toy proxy. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.1 (local proxy context); H100 depth still planned in [`07-h100-conductor-planned`](07-h100-conductor-planned.md)
- Related suites: [`02-sota-lean-followup`](02-sota-lean-followup.md), [`07-h100-conductor-planned`](07-h100-conductor-planned.md) (`sota_h100_depth`)

---

[Previous](02-sota-lean-followup.md) · [Index](../00-INDEX.md) · [Next](04-sota-arch-ladder.md)
