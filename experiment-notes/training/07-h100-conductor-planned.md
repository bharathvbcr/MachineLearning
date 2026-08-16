# h100-conductor-planned: Depth / TTT / QAT export suites (no local outs)

## Executive summary

- **Question:** At full sprint scale (dim 512, multi-layer, 20k steps), (1) depth×MLP choices, (2) test-time training optimizer cross-products, and (3) late-QAT / int6–int8 clip recipes will move calibrated BPB and artifact size in ways the local 4L×128 proxy cannot predict.
- **Result:** Conductor defines three H100 follow-ups (depth, TTT×opt, QAT×export) ranked on calibrated BPB; until sprint outs exist, treat them as planned — never fill results from local toy numbers.
- **Implication:** This is an experiment definition, not measured evidence; execute it before drawing a result.
- **Status:** `planned`; evidence confidence **Low**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `sota_h100_depth` + `sota_ttt_optimizer` + `sota_qat_export` |
| Dates | n/a (definition only; not executed in this workspace) |
| Hardware | Target: 8×H100 sprint (`runner: sprint`, `WINDOWS_SAFE_MODE=0`); local 3070 Ti outs absent |
| Status | `planned` |

## Hypothesis

At full sprint scale (dim 512, multi-layer, 20k steps), (1) depth×MLP choices, (2) test-time training optimizer cross-products, and (3) late-QAT / int6–int8 clip recipes will move calibrated BPB and artifact size in ways the local 4L×128 proxy cannot predict.

## Setup

Source of truth: `parameter-golf/conductor/ablation_suites_3070ti.json` (H100-oriented suite blocks). Shared pattern:

- Runner / script: `sprint` / `train_gpt_sprint.py`
- Ranking: `calibrated_bpb`, tie-breakers `sliding_bpb`, `artifact_bytes`, `step_avg_ms`
- Stages: `timing` (400 steps, no calib, 120 s wall) → `full` (20000 steps, calib 65536 tokens, 600 s wall)
- No `parameter-golf/logs/ablations/sota_h100_*` directories exist locally

Common base_env kernels:

| Suite | Distinct base knobs |
|-------|---------------------|
| `sota_h100_depth` | `MODEL_DIM=512`, `NUM_HEADS=8`, `NUM_KV_HEADS=4`, lean aux `BIGRAM_DIM=48` / `VE_DIM=24`, `XSA_LAST_N=4` / `XSA_MODE=paper` (layers set per candidate) |
| `sota_ttt_optimizer` | above + fixed `NUM_LAYERS=11`, `MLP_MULT=3.0` |
| `sota_qat_export` | same fixed 11L / MLP×3 base as TTT suite |

## Variants

### `sota_h100_depth`

| Variant | Change |
|---------|--------|
| `10L_m26` | `NUM_LAYERS=10`, `MLP_MULT=2.6` |
| `10L_m30` | `NUM_LAYERS=10`, `MLP_MULT=3.0` |
| `11L_m26` | `NUM_LAYERS=11`, `MLP_MULT=2.6` |
| `11L_m30` | `NUM_LAYERS=11`, `MLP_MULT=3.0` |

### `sota_ttt_optimizer` (full stage seeds `[1337, 42]`)

| Variant | Change |
|---------|--------|
| `control` | (none) |
| `lr_up` | `MATRIX_LR=SCALAR_LR=0.027` |
| `muon_097` | `MUON_MOMENTUM=0.97` |
| `clip_hi` | `GRAD_CLIP_NORM=0.35` |
| `lr_up_muon_097` / `lr_up_clip_hi` | LR crosses |
| `ttt_base` | `TTT_ENABLED=1`, epochs 2, freeze 2, LR 0.002, chunk 32768, batch 32 |
| `ttt_conservative` | epochs 1, freeze 4, LR 0.0015, smaller chunks |
| `ttt_base_lr_up` / `ttt_base_muon_097` | TTT × train opt |

### `sota_qat_export` (full stage seeds `[1337, 42]`)

| Variant | Change |
|---------|--------|
| `control` | (none) |
| `qat_010` / `qat_012` / `qat_015` | `QAT_ENABLED=1`, `LATE_QAT_THRESHOLD` ∈ {0.10, 0.12, 0.15} |
| `int6_clip_{a,b,c}` | `INT6_CLIP_PCTS` ladders |
| `int8_clip_999` / `int8_clip_9995` | `INT8_CLIP_PERCENTILE` 99.9 / 99.95 |
| `qat_012_int6_clip_b` / `qat_012_int8_clip_9995` | QAT × clip crosses |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| — | *(none)* | — | No `*.summary.json` / `champion.json` under `parameter-golf/logs/ablations/` for these suite ids |

Do not invent BPB, step times, or artifact sizes. Local depth-proxy lesson ([`03-sota-depth-proxy`](03-sota-depth-proxy.md): compact + LR-up beat fat 8L) is a **proxy** signal only; these H100 suites remain the real depth / TTT / QAT votes.

**Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

## Failures

- Not run — no OOMs / rc codes to report.
- Blocker is execution environment (sprint / H100), not a recorded crash.

## Lesson

**Conductor defines three H100 follow-ups (depth, TTT×opt, QAT×export) ranked on calibrated BPB; until sprint outs exist, treat them as planned — never fill results from local toy numbers.**

## Reproduction

- Replay in the intended sprint/H100 environment using the manifest entries in `parameter-golf/conductor/ablation_suites_3070ti.json`:
  - `python3 parameter-golf/run_ablation_3070ti.py sota_h100_depth --stage full --resume`
  - `python3 parameter-golf/run_ablation_3070ti.py sota_ttt_optimizer --stage full --resume`
  - `python3 parameter-golf/run_ablation_3070ti.py sota_qat_export --stage full --resume`
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: Low.** Definition-only suite: no run outputs, seeds, timings, BPB, or artifact measurements exist.

## Artifacts

- `parameter-golf/conductor/ablation_suites_3070ti.json` → keys `sota_h100_depth`, `sota_ttt_optimizer`, `sota_qat_export`
- Expected future outs (not present): `parameter-golf/logs/ablations/sota_h100_depth/`, `.../sota_ttt_optimizer/`, `.../sota_qat_export/`

## Why this experiment happened

Local suites resolved proxy-scale choices but could not settle full-sprint depth, test-time training, or quantized export. The conductor entries preserve the intended escalation to 8×H100 so local 4L×128 results are not silently substituted for the target regime. The preceding notebook context is [06-toy-aprdh](06-toy-aprdh.md).

## Experiment story

**Baseline.** Local suites resolved proxy-scale choices but could not settle full-sprint depth, test-time training, or quantized export. The conductor entries preserve the intended escalation to 8×H100 so local 4L×128 results are not silently substituted for the target regime. The preceding notebook context is [06-toy-aprdh](06-toy-aprdh.md).

**Hypothesis.** At full sprint scale (dim 512, multi-layer, 20k steps), (1) depth×MLP choices, (2) test-time training optimizer cross-products, and (3) late-QAT / int6–int8 clip recipes will move calibrated BPB and artifact size in ways the local 4L×128 proxy cannot predict.

**Test contract.** Runner / script: `sprint` / `train_gpt_sprint.py` Ranking: `calibrated_bpb`, tie-breakers `sliding_bpb`, `artifact_bytes`, `step_avg_ms` Stages: `timing` (400 steps, no calib, 120 s wall) → `full` (20000 steps, calib 65536 tokens, 600 s wall) No `parameter-golf/logs/ablations/sota_h100_*` directories exist locally

**Variant sequence.** The preserved comparison matrix was: `10L_m26` — `NUM_LAYERS=10`, `MLP_MULT=2.6`; `10L_m30` — `NUM_LAYERS=10`, `MLP_MULT=3.0`; `11L_m26` — `NUM_LAYERS=11`, `MLP_MULT=2.6`; `11L_m30` — `NUM_LAYERS=11`, `MLP_MULT=3.0`; `control` — (none); `lr_up` — `MATRIX_LR=SCALAR_LR=0.027`.

**Measured turn.** The result board records — — *(none)* — — — No `*.summary.json` / `champion.json` under `parameter-golf/logs/ablations/` for these suite ids.

**Turning point and readout.** Do not invent BPB, step times, or artifact sizes. Local depth-proxy lesson ([`03-sota-depth-proxy`](03-sota-depth-proxy.md): compact + LR-up beat fat 8L) is a **proxy** signal only; these H100 suites remain the real depth / TTT / QAT votes. **Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

**Failures and surprises.** Not run — no OOMs / rc codes to report. Blocker is execution environment (sprint / H100), not a recorded crash.

## Decision and aftermath

**Kept:** Conductor defines three H100 follow-ups (depth, TTT×opt, QAT×export) ranked on calibrated BPB; until sprint outs exist, treat them as planned — never fill results from local toy numbers. **Boundary:** No execution occurred, so the design does not establish any BPB, speed, scaling, TTT, QAT, or export result. The notebook continues with [10-phase0-smoke](../nanolab/10-phase0-smoke.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: — — *(none)* — — — No `*.summary.json` / `champion.json` under `parameter-golf/logs/ablations/` for these suite ids.
- Failure/operational record: Not run — no OOMs / rc codes to report.
- Failure/operational record: Blocker is execution environment (sprint / H100), not a recorded crash.

## What this does not prove

**Confidence: Low.** Definition-only suite: no run outputs, seeds, timings, BPB, or artifact measurements exist. No execution occurred, so the design does not establish any BPB, speed, scaling, TTT, QAT, or export result. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.1 local proxy vs end roadmap (QAT / TTT / H100); §8.7 submission path
- Related suites: [`03-sota-depth-proxy`](03-sota-depth-proxy.md) (local depth proxy only)

---

[Previous](06-toy-aprdh.md) · [Index](../00-INDEX.md) · Next
