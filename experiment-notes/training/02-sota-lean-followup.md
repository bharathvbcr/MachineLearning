# sota_lean_followup: Aux floor / LR / XSA follow-ups

## Executive summary

- **Question:** Given `sota_local` champion `lean_aux`, further cutting aux capacity (`leaner_aux` / `aux_floor`) or recombining lean floor with LR / XSA / clip / Muon would improve calibrated BPB without waiting for a full architecture rewrite.
- **Result:** Do not shrink aux further; under the lean floor, raise LR (`0.027`) — mid calibrated BPB 2.232 — rather than turn XSA off.
- **Implication:** Do not promote this beyond the completed stages; the missing or failed confirmation is decision-relevant.
- **Status:** `partial`; evidence confidence **Low**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `sota_lean_followup` |
| Dates | `2026-03-28` – `2026-03-28` |
| Hardware | RTX 3070 Ti (local ablation via `run_ablation_3070ti.py`) |
| Status | `partial` |

## Hypothesis

Given `sota_local` champion `lean_aux`, further cutting aux capacity (`leaner_aux` / `aux_floor`) or recombining lean floor with LR / XSA / clip / Muon would improve calibrated BPB without waiting for a full architecture rewrite.

## Setup

- Trainer / preset: same path as `sota_local` — `base_preset: sota` → `train_gpt_sprint.py`
- Fixed knobs: `SOTA_CONFIG` toy proxy (4L / 128-d / 4 heads / 2 KV, seq 256, batch 4096 tokens)
- Stages: short 300 (promote top 4) → mid 1000 (promote top 2) → long 3000 seeds `[1337, 42]` planned; rank `calibrated_bpb`
- Env flags: candidate overrides; mid/long summaries sanitize inherited `XSA_LAST_N`, `XSA_MODE`
- Executed on disk: **short + mid only** (no `long.*` artifacts; champion is mid-stage)

## Variants

Conductor candidates (`ablation_suites_3070ti.json`):

| Variant | Change |
|---------|--------|
| `control` | (none) — lean floor already in base after suite 01 |
| `leaner_aux` | `BIGRAM_DIM=40`, `VE_DIM=20` |
| `aux_floor` | `BIGRAM_DIM=32`, `VE_DIM=16` |
| `lr_up` | `MATRIX_LR=0.027`, `SCALAR_LR=0.027` |
| `muon_097` | `MUON_MOMENTUM=0.97` |
| `clip_hi` | `GRAD_CLIP_NORM=0.35` |
| `xsa_off` | `XSA_MODE=off` |
| `lr_up_xsa_off` | LR up + XSA off |

Short `*.summary.json` also records duplicate-named rows (`lean_lr_up`, `lean_xsa_off`, `lean_clip_hi`) with identical metrics to the non-`lean_` twins — treat as the same levers.

## Results

Short (300 steps, seed 1337):

| Rank | Run / config | calibrated_bpb | Notes |
|------|--------------|----------------|-------|
| 1 | `lr_up` / `lean_lr_up` | **2.711021** | sliding 2.754002; artifact 1,315,049 B |
| 3 | `xsa_off` / `lean_xsa_off` | 2.727284 | |
| 5 | `muon_097` | 2.728245 | |
| 6 | `control` | 2.734760 | same BPB as suite-01 short `lean_aux` |
| 7 | `clip_hi` / `lean_clip_hi` | 2.736027 | |
| 9 | `leaner_aux` | 2.754318 | smaller aux hurts |
| 10 | `aux_floor` | 2.770736 | worst short |

Mid (1000 steps, seed 1337) — champion board:

| Rank | Run / config | calibrated_bpb | Notes |
|------|--------------|----------------|-------|
| 1 | `lr_up` | **2.231641** | sliding 2.238984; step 283.28 ms; artifact 1,329,565 B; rc `[0]` |
| 2 | `xsa_off` | 2.243953 | Δ+0.012 |

Champion.json: `candidate=lr_up`, stage `mid`, calibrated_bpb **2.2316411509892693**, sliding_bpb **2.23898379**, step_avg_ms **283.28**, artifact_bytes **1329565**.

Readout: cutting aux below the suite-01 floor (`40/20`, `32/16`) lost BPB on short. Under the lean floor, `lr_up` was the only mid promotion that clearly beat `xsa_off`. Long stage was never completed, so this remains a mid-horizon pick.

**Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

## Failures

- Long stage missing entirely (no `long.summary.*` / multi-seed confirmation).
- All recorded short/mid returncodes are `0`; no OOMs in summaries.

## Lesson

**Do not shrink aux further; under the lean floor, raise LR (`0.027`) — mid calibrated BPB 2.232 — rather than turn XSA off.**

## Reproduction

- Replay: `python3 parameter-golf/run_ablation_3070ti.py sota_lean_followup --resume`.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: Low.** Only one-seed short and mid stages completed; the planned two-seed long confirmation is absent.

## Artifacts

- `parameter-golf/logs/ablations/sota_lean_followup/champion.json`
- `parameter-golf/logs/ablations/sota_lean_followup/mid.summary.json` / `.txt`
- `parameter-golf/logs/ablations/sota_lean_followup/short.summary.json`
- `parameter-golf/logs/ablations/sota_lean_followup/latest_summary.txt`
- `parameter-golf/conductor/ablation_suites_3070ti.json` → key `sota_lean_followup`

## Why this experiment happened

Suite 01 made `lean_aux` the first durable local winner: 2.066 calibrated BPB, down from 2.093, with a smaller export. This follow-up asked whether that result was the start of a monotonic compression trend or a floor, and whether the lean base changed the preferred LR/XSA settings. The preceding notebook context is [01-sota-local](01-sota-local.md).

## Experiment story

**Baseline.** Suite 01 made `lean_aux` the first durable local winner: 2.066 calibrated BPB, down from 2.093, with a smaller export. This follow-up asked whether that result was the start of a monotonic compression trend or a floor, and whether the lean base changed the preferred LR/XSA settings. The preceding notebook context is [01-sota-local](01-sota-local.md).

**Hypothesis.** Given `sota_local` champion `lean_aux`, further cutting aux capacity (`leaner_aux` / `aux_floor`) or recombining lean floor with LR / XSA / clip / Muon would improve calibrated BPB without waiting for a full architecture rewrite.

**Test contract.** Trainer / preset: same path as `sota_local` — `base_preset: sota` → `train_gpt_sprint.py` Fixed knobs: `SOTA_CONFIG` toy proxy (4L / 128-d / 4 heads / 2 KV, seq 256, batch 4096 tokens) Stages: short 300 (promote top 4) → mid 1000 (promote top 2) → long 3000 seeds `[1337, 42]` planned; rank `calibrated_bpb` Env flags: candidate overrides; mid/long summaries sanitize inherited `XSA_LAST_N`, `XSA_MODE`

**Variant sequence.** The preserved comparison matrix was: `control` — (none) — lean floor already in base after suite 01; `leaner_aux` — `BIGRAM_DIM=40`, `VE_DIM=20`; `aux_floor` — `BIGRAM_DIM=32`, `VE_DIM=16`; `lr_up` — `MATRIX_LR=0.027`, `SCALAR_LR=0.027`; `muon_097` — `MUON_MOMENTUM=0.97`; `clip_hi` — `GRAD_CLIP_NORM=0.35`.

**Measured turn.** The result board records 1 — `lr_up` / `lean_lr_up` — **2.711021** — sliding 2.754002; artifact 1,315,049 B; 3 — `xsa_off` / `lean_xsa_off` — 2.727284 — ; 5 — `muon_097` — 2.728245 — ; 6 — `control` — 2.734760 — same BPB as suite-01 short `lean_aux`; 7 — `clip_hi` / `lean_clip_hi` — 2.736027 — .

**Turning point and readout.** Champion.json: `candidate=lr_up`, stage `mid`, calibrated_bpb **2.2316411509892693**, sliding_bpb **2.23898379**, step_avg_ms **283.28**, artifact_bytes **1329565**. Readout: cutting aux below the suite-01 floor (`40/20`, `32/16`) lost BPB on short. Under the lean floor, `lr_up` was the only mid promotion that clearly beat `xsa_off`. Long stage was never completed, so this remains a mid-horizon pick. **Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

**Failures and surprises.** Long stage missing entirely (no `long.summary.*` / multi-seed confirmation). All recorded short/mid returncodes are `0`; no OOMs in summaries.

## Decision and aftermath

**Kept:** Do not shrink aux further; under the lean floor, raise LR (`0.027`) — mid calibrated BPB 2.232 — rather than turn XSA off. **Boundary:** Incomplete or failed confirmation prevents promotion beyond the explicitly successful stages. The notebook continues with [03-sota-depth-proxy](03-sota-depth-proxy.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `lr_up` / `lean_lr_up` — **2.711021** — sliding 2.754002; artifact 1,315,049 B.
- The result artifact reports: 3 — `xsa_off` / `lean_xsa_off` — 2.727284 — .
- The result artifact reports: 5 — `muon_097` — 2.728245 — .
- The result artifact reports: 6 — `control` — 2.734760 — same BPB as suite-01 short `lean_aux`.
- Failure/operational record: Long stage missing entirely (no `long.summary.*` / multi-seed confirmation).
- Failure/operational record: All recorded short/mid returncodes are `0`; no OOMs in summaries.

## What this does not prove

**Confidence: Low.** Only one-seed short and mid stages completed; the planned two-seed long confirmation is absent. Incomplete or failed confirmation prevents promotion beyond the explicitly successful stages. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.1 (aux-head 2.066 finding from suite 01)
- Related suites: [`01-sota-local`](01-sota-local.md), [`03-sota-depth-proxy`](03-sota-depth-proxy.md)

---

[Previous](01-sota-local.md) · [Index](../00-INDEX.md) · [Next](03-sota-depth-proxy.md)
