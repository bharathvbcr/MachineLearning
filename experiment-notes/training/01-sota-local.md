# sota_local: Grad clip / Muon / LR / XSA / lean aux

## Executive summary

- **Question:** On the local `sota` toy preset, training-side knobs (grad clip, Muon momentum, LR, XSA last-N) and a leaner aux-head sizing would move calibrated BPB meaningfully; the suite expected at least one clear winner to promote into follow-up ablations.
- **Result:** Lean aux heads (`BIGRAM_DIM=48`, `VE_DIM=24`) beat training-hyperparameter tweaks: calibrated BPB 2.066 vs control 2.093, with a smaller export.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **High**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `sota_local` |
| Dates | `2026-03-27` – `2026-03-27` |
| Hardware | RTX 3070 Ti (local ablation via `run_ablation_3070ti.py`) |
| Status | `done` |

## Hypothesis

On the local `sota` toy preset, training-side knobs (grad clip, Muon momentum, LR, XSA last-N) and a leaner aux-head sizing would move calibrated BPB meaningfully; the suite expected at least one clear winner to promote into follow-up ablations.

## Setup

- Trainer / preset: `run_ablation_3070ti.py` → `run_toy_3070ti.py sota` → `train_gpt_sprint.py` (`base_preset: sota`)
- Fixed knobs (from `SOTA_CONFIG` in `parameter-golf/run_toy_3070ti.py`): `NUM_LAYERS=4`, `MODEL_DIM=128`, `NUM_HEADS=4`, `NUM_KV_HEADS=2`, `MLP_MULT=3`, `TRAIN_SEQ_LEN=256`, `TRAIN_BATCH_TOKENS=4096`, base `GRAD_CLIP_NORM=0.3`, `MUON_MOMENTUM=0.95`, `MATRIX_LR=SCALAR_LR=0.025`, `XSA_MODE=paper` / `XSA_LAST_N=2`, VE+bigram aux enabled
- Stages (conductor): short 300 steps (calib 16k tokens) → mid 1000 → long 3000 with seeds `[1337, 42]`; rank by `calibrated_bpb`, then `sliding_bpb`, `step_avg_ms`, `artifact_bytes`
- Env flags: per-candidate overrides only; long stage sanitized inherited `XSA_LAST_N` / `XSA_MODE`
- Note: artifacts on disk are `short` + `long` only (no `mid.summary.*` for this suite)

## Variants

| Variant | Change |
|---------|--------|
| `control` | (none) |
| `clip_lo` | `GRAD_CLIP_NORM=0.25` |
| `clip_hi` | `GRAD_CLIP_NORM=0.35` |
| `muon_097` | `MUON_MOMENTUM=0.97` |
| `lr_up` | `MATRIX_LR=0.027`, `SCALAR_LR=0.027` |
| `xsa_1` | `XSA_MODE=paper`, `XSA_LAST_N=1` |
| `xsa_3` | `XSA_MODE=paper`, `XSA_LAST_N=3` |
| `xsa_off` | `XSA_MODE=off` |
| `lean_aux` | `BIGRAM_DIM=48`, `VE_DIM=24` |

## Results

Long stage (3000 steps, mean of seeds 1337+42) — primary board:

| Rank | Run / config | calibrated_bpb | Notes |
|------|--------------|----------------|-------|
| 1 | `lean_aux` | **2.066382** | sliding 2.069032; step 375.76 ms; artifact **1,333,749** B; rcs `[0,0]` |
| 2 | `clip_hi` | 2.089436 | Δ+0.023; sliding 2.091946 |
| 3 | `clip_lo` | 2.091645 | Δ+0.025 |
| 4 | `control` | 2.093256 | Δ+0.027; artifact 1,415,155 B |
| 5 | `lr_up` | 2.100277 | Δ+0.034 |
| 6 | `xsa_1` | 2.102677 | Δ+0.036 |
| 7 | `muon_097` | 2.104508 | Δ+0.038 |
| 8 | `xsa_3` | 2.105036 | Δ+0.039 |
| 9 | `xsa_off` | 2.113424 | Δ+0.047; worst calibrated (sliding 2.103980 still middling) |

Short stage (300 steps, seed 1337) already ranked `lean_aux` first at calibrated_bpb **2.734760** (vs control 2.781817). Champion.json matches long #1: `lean_aux`, calibrated_bpb **2.066382**, sliding_bpb **2.069032**, step_avg_ms **375.755**, artifact_bytes **1333749**.

Readout: lean aux sizing was the only clear win — ~0.027 BPB over control and a smaller artifact. Grad-clip nudges slightly beat control; raising LR or Muon momentum did not. Turning XSA off was the worst calibrated long result (+0.020 vs control), consistent with keeping paper XSA on.

**Interpretation boundary.** The ~0.027 BPB lean-aux gain over control is measured across two long-stage seeds; the claim that aux sizing is generally superior to hyperparameter tuning is an inference limited to this proxy.

## Failures

- No OOMs / NaNs; all long returncodes `[0, 0]`.
- Mid-stage summaries absent from `logs/ablations/sota_local/` (short→long present only).

## Lesson

**Lean aux heads (`BIGRAM_DIM=48`, `VE_DIM=24`) beat training-hyperparameter tweaks: calibrated BPB 2.066 vs control 2.093, with a smaller export.**

## Reproduction

- Replay: `python3 parameter-golf/run_ablation_3070ti.py sota_local --resume`.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: High.** Two-seed long stage completed; all long return codes were zero. The missing mid summary does not affect the recorded long comparison.

## Artifacts

- `parameter-golf/logs/ablations/sota_local/champion.json`
- `parameter-golf/logs/ablations/sota_local/long.summary.json` / `.txt` / `.csv`
- `parameter-golf/logs/ablations/sota_local/short.summary.json`
- `parameter-golf/logs/ablations/sota_local/latest_summary.txt`
- `parameter-golf/conductor/ablation_suites_3070ti.json` → key `sota_local`

## Why this experiment happened

The local `sota` preset needed a first controlled vote before architecture work could be trusted. The immediate uncertainty was whether ordinary training controls—clipping, Muon momentum, learning rate, and XSA depth—or the auxiliary-head allocation was consuming the limited proxy budget most effectively.

## Experiment story

**Baseline.** The local `sota` preset needed a first controlled vote before architecture work could be trusted. The immediate uncertainty was whether ordinary training controls—clipping, Muon momentum, learning rate, and XSA depth—or the auxiliary-head allocation was consuming the limited proxy budget most effectively.

**Hypothesis.** On the local `sota` toy preset, training-side knobs (grad clip, Muon momentum, LR, XSA last-N) and a leaner aux-head sizing would move calibrated BPB meaningfully; the suite expected at least one clear winner to promote into follow-up ablations.

**Test contract.** Trainer / preset: `run_ablation_3070ti.py` → `run_toy_3070ti.py sota` → `train_gpt_sprint.py` (`base_preset: sota`) Fixed knobs (from `SOTA_CONFIG` in `parameter-golf/run_toy_3070ti.py`): `NUM_LAYERS=4`, `MODEL_DIM=128`, `NUM_HEADS=4`, `NUM_KV_HEADS=2`, `MLP_MULT=3`, `TRAIN_SEQ_LEN=256`, `TRAIN_BATCH_TOKENS=4096`, base `GRAD_CLIP_NORM=0.3`, `MUON_MOMENTUM=0.95`, `MATRIX_LR=SCALAR_LR=0.025`, `XSA_MODE=paper` / `XSA_LAST_N=2`, VE+bigram aux enabled Stages (conductor): short 300 steps (calib 16k tokens) → mid 1000 → long 3000 with seeds `[1337, 42]`; rank by `calibrated_bpb`, then `sliding_bpb`, `step_avg_ms`, `artifact_bytes` Env flags: per-candidate overrides only; long stage sanitized inherited `XSA_LAST_N` / `XSA_MODE`

**Variant sequence.** The preserved comparison matrix was: `control` — (none); `clip_lo` — `GRAD_CLIP_NORM=0.25`; `clip_hi` — `GRAD_CLIP_NORM=0.35`; `muon_097` — `MUON_MOMENTUM=0.97`; `lr_up` — `MATRIX_LR=0.027`, `SCALAR_LR=0.027`; `xsa_1` — `XSA_MODE=paper`, `XSA_LAST_N=1`.

**Measured turn.** The result board records 1 — `lean_aux` — **2.066382** — sliding 2.069032; step 375.76 ms; artifact **1,333,749** B; rcs `[0,0]`; 2 — `clip_hi` — 2.089436 — Δ+0.023; sliding 2.091946; 3 — `clip_lo` — 2.091645 — Δ+0.025; 4 — `control` — 2.093256 — Δ+0.027; artifact 1,415,155 B; 5 — `lr_up` — 2.100277 — Δ+0.034.

**Turning point and readout.** Short stage (300 steps, seed 1337) already ranked `lean_aux` first at calibrated_bpb **2.734760** (vs control 2.781817). Champion.json matches long #1: `lean_aux`, calibrated_bpb **2.066382**, sliding_bpb **2.069032**, step_avg_ms **375.755**, artifact_bytes **1333749**. Readout: lean aux sizing was the only clear win — ~0.027 BPB over control and a smaller artifact. Grad-clip nudges slightly beat control; raising LR or Muon momentum did not. Turning XSA off was the worst calibrated long result (+0.020 vs control), consistent with keeping paper XSA on. **Interpretation boundary.** The ~0.027 BPB lean-aux gain over control is measured across two long-stage seeds; the claim that aux sizing is generally superior to hyperparameter tuning is an inference limited to this proxy.

**Failures and surprises.** No OOMs / NaNs; all long returncodes `[0, 0]`. Mid-stage summaries absent from `logs/ablations/sota_local/` (short→long present only).

## Decision and aftermath

**Kept:** Lean aux heads (`BIGRAM_DIM=48`, `VE_DIM=24`) beat training-hyperparameter tweaks: calibrated BPB 2.066 vs control 2.093, with a smaller export. The notebook continues with [02-sota-lean-followup](02-sota-lean-followup.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `lean_aux` — **2.066382** — sliding 2.069032; step 375.76 ms; artifact **1,333,749** B; rcs `[0,0]`.
- The result artifact reports: 2 — `clip_hi` — 2.089436 — Δ+0.023; sliding 2.091946.
- The result artifact reports: 3 — `clip_lo` — 2.091645 — Δ+0.025.
- The result artifact reports: 4 — `control` — 2.093256 — Δ+0.027; artifact 1,415,155 B.
- Failure/operational record: No OOMs / NaNs; all long returncodes `[0, 0]`.
- Failure/operational record: Mid-stage summaries absent from `logs/ablations/sota_local/` (short→long present only).

## What this does not prove

**Confidence: High.** Two-seed long stage completed; all long return codes were zero. The missing mid summary does not affect the recorded long comparison. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.1 (aux-head / XSA sub-findings)
- Related suites: [`02-sota-lean-followup`](02-sota-lean-followup.md), [`03-sota-depth-proxy`](03-sota-depth-proxy.md)

---

Previous · [Index](../00-INDEX.md) · [Next](02-sota-lean-followup.md)
