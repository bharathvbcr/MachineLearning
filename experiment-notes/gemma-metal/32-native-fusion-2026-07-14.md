# 32: Native fusion session (2026-07-14)

## Executive summary

- **Question:** Fusing dual-norm residual, bf16 producer casts, K+V store, and one-pass softcap-argmax would cut per-token dispatch tax and raise quiet E4B/31B tok/s without breaking hazard lanes.
- **Result:** Step-3 bf16 producer fuse is the only clear win (~24.9 E4B); other fusions landed safely but did not move the needle under a noisy GPU.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Low**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `32-native-fusion-2026-07-14` |
| Dates | `2026-07-14` – `2026-07-14` |
| Hardware | Apple M5 Pro (noisy session: load ~4–5; Cursor helpers) |
| Status | `done` |

## Hypothesis

Fusing dual-norm residual, bf16 producer casts, K+V store, and one-pass softcap-argmax would cut per-token dispatch tax and raise quiet E4B/31B tok/s without breaking hazard lanes.

## Setup

- Trainer / preset: release `bench` / `diag_tok`; quiet historical refs E4B **23.92**, 31B **6.83**; session floor ~4.5 on 31B
- Fixed knobs: standing gates E4B ≥23, 31B greedy finite, E4B path untouched
- Env flags: `GEMMA_METAL_FUSE_DUAL_NORM`, `GEMMA_METAL_FUSE_BF16=0|rms|fa|mlp`, `METAL_RUNTIME_MID_COMMIT=0|128|256` (default 0), `GEMMA_METAL_ARGMAX_MULTIPASS=1`, `METAL_RUNTIME_HAZARD_BARRIERS=0` for golden always-on

## Variants

| Variant | Change |
|---------|--------|
| Preflight | Ownership quiet baseline |
| Step 1 | `rms_norm_residual_add_f32` dual-norm fuse |
| Step 2b | mid-commit sweep 0/128/256 |
| Step 3 | Producers emit bf16 (`FUSE_BF16`) |
| Step 4 | `kv_store_timestep_pair` + layer_scalar into residual |
| Step 5 | `softcap_argmax_one_pass` |

## Results

| Rank | Step | E4B | 31B decode | Notes |
|------|------|-----|------------|-------|
| Best session E4B | 3 bf16 cast fuse | **24.90** | **5.41** | A/B off: 24.45 / 5.57 (noise); HAZARD=0 bit-match |
| Preflight | baseline | 23.67 | 4.55 | mid-mmin settled |
| 1 | dual-norm fuse | 23.76 | ~4.56 | rollback `FUSE_DUAL_NORM=0` |
| 2b | mid-commit | 23.9–24.0 | 4.54→4.59 | leave default **0** |
| 4 | K+V + layer_scalar | 23.99 | **5.30** | |
| 5 | one-pass argmax | 23.99 | **5.26** | rollback `ARGMAX_MULTIPASS=1` |

After-step benches (`baselines_2026-07-14/*_after_step*.txt`) match the session table: e.g. `e4b_after_step3.txt` **24.90** / TTFT 135.1 ms; `31b_after_step3.txt` **5.41**; `31b_after_step5.txt` **5.26**. Net: modest E4B bump (~+1 tok/s at step 3); 31B session numbers stay below quiet historical **6.83** under contention. Standing gates held.

**Interpretation boundary.** The E4B +1 tok/s movement is the clearest measured effect. The 31B changes are smaller than session contention and should not be interpreted as quality or stable speed gains.

## Failures

- Mid-commit flat — default stays 0
- Golden `greet 16` with HAZARD=0: **match_prefix=0/16** (open port gap; not attributed to step 3)
- Session noise depresses absolute 31B vs quiet refs

## Lesson

**Step-3 bf16 producer fuse is the only clear win (~24.9 E4B); other fusions landed safely but did not move the needle under a noisy GPU.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: the release `bench` and `diag_tok` binaries.
- Required replay inputs: the exact environment A/B matrix and after-step logs in `bench/results/baselines_2026-07-14/SESSION_RESULTS.md`. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Low.** The session ran under substantial GPU contention; only the E4B bf16-fuse movement clearly exceeded observed session variation.

## Artifacts

- `Rust_MLKit/gemma-metal/bench/results/baselines_2026-07-14/SESSION_RESULTS.md`
- `…/e4b_pre.txt`, `e4b_after_step1.txt`, `e4b_after_step3.txt`
- `…/31b_pre.txt`, `31b_after_step{1,3,4,5}.txt`, `31b_mid{0,256}.txt`, `31b_after_step3_off.txt`
- `…/diag_tok_e4b_fuse_{on,off}.txt`

## Why this experiment happened

Suite 31 raised E4B from 4.78 to roughly 25 tok/s, then showed diminishing returns from qdot-level peels. This session tested whether eliminating whole dispatches through producer and residual fusion could move the plateau. The preceding notebook context is [31-native-decode-speed-ladder](31-native-decode-speed-ladder.md).

## Experiment story

**Baseline.** Suite 31 raised E4B from 4.78 to roughly 25 tok/s, then showed diminishing returns from qdot-level peels. This session tested whether eliminating whole dispatches through producer and residual fusion could move the plateau. The preceding notebook context is [31-native-decode-speed-ladder](31-native-decode-speed-ladder.md).

**Hypothesis.** Fusing dual-norm residual, bf16 producer casts, K+V store, and one-pass softcap-argmax would cut per-token dispatch tax and raise quiet E4B/31B tok/s without breaking hazard lanes.

**Test contract.** Trainer / preset: release `bench` / `diag_tok`; quiet historical refs E4B **23.92**, 31B **6.83**; session floor ~4.5 on 31B Fixed knobs: standing gates E4B ≥23, 31B greedy finite, E4B path untouched Env flags: `GEMMA_METAL_FUSE_DUAL_NORM`, `GEMMA_METAL_FUSE_BF16=0|rms|fa|mlp`, `METAL_RUNTIME_MID_COMMIT=0|128|256` (default 0), `GEMMA_METAL_ARGMAX_MULTIPASS=1`, `METAL_RUNTIME_HAZARD_BARRIERS=0` for golden always-on

**Variant sequence.** The preserved comparison matrix was: Preflight — Ownership quiet baseline; Step 1 — `rms_norm_residual_add_f32` dual-norm fuse; Step 2b — mid-commit sweep 0/128/256; Step 3 — Producers emit bf16 (`FUSE_BF16`); Step 4 — `kv_store_timestep_pair` + layer_scalar into residual; Step 5 — `softcap_argmax_one_pass`.

**Measured turn.** The result board records Best session E4B — 3 bf16 cast fuse — **24.90** — **5.41** — A/B off: 24.45 / 5.57 (noise); HAZARD=0 bit-match; Preflight — baseline — 23.67 — 4.55 — mid-mmin settled; 1 — dual-norm fuse — 23.76 — ~4.56 — rollback `FUSE_DUAL_NORM=0`; 2b — mid-commit — 23.9–24.0 — 4.54→4.59 — leave default **0**; 4 — K+V + layer_scalar — 23.99 — **5.30** — .

**Turning point and readout.** After-step benches (`baselines_2026-07-14/*_after_step*.txt`) match the session table: e.g. `e4b_after_step3.txt` **24.90** / TTFT 135.1 ms; `31b_after_step3.txt` **5.41**; `31b_after_step5.txt` **5.26**. Net: modest E4B bump (~+1 tok/s at step 3); 31B session numbers stay below quiet historical **6.83** under contention. Standing gates held. **Interpretation boundary.** The E4B +1 tok/s movement is the clearest measured effect. The 31B changes are smaller than session contention and should not be interpreted as quality or stable speed gains.

**Failures and surprises.** Mid-commit flat — default stays 0 Golden `greet 16` with HAZARD=0: **match_prefix=0/16** (open port gap; not attributed to step 3) Session noise depresses absolute 31B vs quiet refs

## Decision and aftermath

**Kept:** Step-3 bf16 producer fuse is the only clear win (~24.9 E4B); other fusions landed safely but did not move the needle under a noisy GPU. The notebook continues with [33-kernel-roofline-overhead](33-kernel-roofline-overhead.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: Best session E4B — 3 bf16 cast fuse — **24.90** — **5.41** — A/B off: 24.45 / 5.57 (noise); HAZARD=0 bit-match.
- The result artifact reports: Preflight — baseline — 23.67 — 4.55 — mid-mmin settled.
- The result artifact reports: 1 — dual-norm fuse — 23.76 — ~4.56 — rollback `FUSE_DUAL_NORM=0`.
- The result artifact reports: 2b — mid-commit — 23.9–24.0 — 4.54→4.59 — leave default **0**.
- Failure/operational record: Mid-commit flat — default stays 0
- Failure/operational record: Golden `greet 16` with HAZARD=0: **match_prefix=0/16** (open port gap; not attributed to step 3)

## What this does not prove

**Confidence: Low.** The session ran under substantial GPU contention; only the E4B bf16-fuse movement clearly exceeded observed session variation. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- Related suites: [`31-native-decode-speed-ladder`](31-native-decode-speed-ladder.md), [`33-kernel-roofline-overhead`](33-kernel-roofline-overhead.md), [`37-golden-token-parity`](37-golden-token-parity.md)

---

[Previous](31-native-decode-speed-ladder.md) · [Index](../00-INDEX.md) · [Next](33-kernel-roofline-overhead.md)
