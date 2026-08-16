# 37: Golden token / intermediate parity

## Executive summary

- **Question:** Native greedy (and later DFlash) should reproduce MLX golden token streams and first-block intermediates so speculative decode stays lossless vs the product path.
- **Result (historical, 2026-07-14):** MLX greedy/DFlash goldens were exact; native greet16 was 0/16 (`target_next` 929≠531) under capture-off / wrong RoPE.
- **Resolution (Lane A, 2026-07-19):** **Resolved under capture-on** — NeoX proportional RoPE + required hidden capture → `target_next=531`, **greet16 = 16/16**. Speculative residual is draft-accept, not Hot golden tokens.
- **Implication:** Golden-token Hot parity is closed for the capture-on path; do not re-chase 929≠531.
- **Status:** `resolved` (capture-on); evidence confidence **High** for greet16 / `target_next`.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `37-golden-token-parity` |
| Dates | `2026-07-14` – `2026-07-19` (resolved) |
| Hardware | Apple M5 Pro · mlx golden vs gemma-metal Hot 31B |
| Status | `resolved` under capture-on |

## Hypothesis

Native greedy (and later DFlash) should reproduce MLX golden token streams and first-block intermediates so speculative decode stays lossless vs the product path.

## Setup

- Trainer / preset: `cargo run --release --bin golden_parity -- greet 16`; MLX dumps via DFlash harness
- Fixed knobs: temp=0; `METAL_RUNTIME_HAZARD_BARRIERS=0` for always-on Device barriers on golden compare
- Env flags: as in SESSION_RESULTS A/B notes

## Variants

| Variant | Change |
|---------|--------|
| MLX golden tokens | `golden_tokens_31b.json` (greedy == DFlash) |
| MLX intermediates | `golden_intermediates_31b.json` block-1 tensors |
| Native greet16 | gemma-metal Hot vs gold ids |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| MLX DFlash vs greedy | exact match | `dflash_exact_vs_greedy: true`; greet gold starts `[100, 45518, 107, …]` | Product reference |
| MLX intermediates | block_size=5 | `target_next_argmax=531`; `proposed_block_tokens=[14359, 532, 107, 563]`; embed_scale **73.32121**; h_ctx absmean **0.069892** | Capture layers [1,12,23,35,46,57] |
| Native greet16 | **match_prefix=0/16** | first_mismatch at index 0 | got collapsed `[236773, 236799×…]` vs gold |

**Historical note (2026-07-14):** SESSION_RESULTS marked the 0/16 mismatch as an open port gap (not a fusion #3 regression). That Hot golden gap is **closed under capture-on** (Lane A, 2026-07-19). Remaining work is draft-accept fidelity (GPU Q4Mlx proposals vs host-dense/MLX), not Hot `target_next`.

**Interpretation boundary.** The original 0/16 was a correctness failure under capture-off / interleaved RoPE; it is not the current blocker.

## Failures (historical — superseded)

- ~~Native greedy collapsed vs MLX on greet (prefix 0)~~ → fixed: greet16 16/16 capture-on
- Intermediate dumps on native were incomplete at write-time; Lane A later landed native intermediates / host-dense ceiling for draft-accept

## Lesson

**Hot golden-token parity is resolved under capture-on (greet16 16/16, `target_next=531`). Speculative blocker is draft-accept, not Hot greet16.**

## Reproduction

- Replay: From `Rust_MLKit/gemma-metal`: `METAL_RUNTIME_HAZARD_BARRIERS=0 cargo run --release --bin golden_parity -- greet 16`.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: High** for capture-on Hot golden tokens (Lane A greet16 16/16 + `target_next=531`). Historical 0/16 artifacts remain as the pre-fix baseline.

## Artifacts

- `Rust_MLKit/gemma-metal/bench/results/baselines_2026-07-14/golden_parity_greet16.txt` (historical 0/16)
- `Rust_MLKit/gemma-metal/bench/results/golden_tokens_31b.json`
- `Rust_MLKit/gemma-metal/bench/results/golden_intermediates_31b.json`
- `Rust_MLKit/gemma-metal/docs/dflash_draft_contract.md`
- **Lane A resolution:** [`lane_a_todo_status_2026-07-19.md`](../../Rust_MLKit/gemma-metal/bench/results/lane_a_todo_status_2026-07-19.md), [`lane_a_status_2026-07-19.json`](../../Rust_MLKit/gemma-metal/bench/results/lane_a_status_2026-07-19.json), `greet16_cap.txt` / `greet16_final.txt`, `short_dump_final.txt`

## Why this experiment happened

Native DFlash’s low acceptance and historical false passes could not be debugged from end-to-end speed alone. MLX token and intermediate goldens created a concrete contract for locating the first divergence in greedy and draft execution. The preceding notebook context is [36-native-dflash-parity-accept](36-native-dflash-parity-accept.md).

## Experiment story

**Baseline.** Native DFlash’s low acceptance and historical false passes could not be debugged from end-to-end speed alone. MLX token and intermediate goldens created a concrete contract for locating the first divergence in greedy and draft execution. The preceding notebook context is [36-native-dflash-parity-accept](36-native-dflash-parity-accept.md).

**Hypothesis.** Native greedy (and later DFlash) should reproduce MLX golden token streams and first-block intermediates so speculative decode stays lossless vs the product path.

**Test contract.** Trainer / preset: `cargo run --release --bin golden_parity -- greet 16`; MLX dumps via DFlash harness Fixed knobs: temp=0; `METAL_RUNTIME_HAZARD_BARRIERS=0` for always-on Device barriers on golden compare Env flags: as in SESSION_RESULTS A/B notes

**Variant sequence.** The preserved comparison matrix was: MLX golden tokens — `golden_tokens_31b.json` (greedy == DFlash); MLX intermediates — `golden_intermediates_31b.json` block-1 tensors; Native greet16 — gemma-metal Hot vs gold ids.

**Measured turn.** The result board records MLX DFlash vs greedy — exact match — `dflash_exact_vs_greedy: true`; greet gold starts `[100, 45518, 107, …]` — Product reference; MLX intermediates — block_size=5 — `target_next_argmax=531`; `proposed_block_tokens=[14359, 532, 107, 563]`; embed_scale **73.32121**; h_ctx absmean **0.069892** — Capture layers [1,12,23,35,46,57]; Native greet16 — **match_prefix=0/16** — first_mismatch at index 0 — got collapsed `[236773, 236799×…]` vs gold.

**Turning point and readout.** SESSION_RESULTS marks the 0/16 mismatch as an open port gap (not a fusion #3 regression). Native must still reproduce target_hidden → fc_out → h_ctx → proposed_block_tokens per `dflash_draft_contract.md`. **Interpretation boundary.** A 0/16 prefix match is a measured correctness failure, not a speed result. The proposed localization path is diagnostic inference.

**Failures and surprises.** Native greedy collapsed vs MLX on greet (prefix 0) Intermediate dumps on native still incomplete vs gold for first-block proposals (audit notes)

## Decision and aftermath

**Resolved (Lane A, capture-on):** Hot greet16 / `target_next` parity closed; do not re-open 929≠531 as a live blocker. **Follow-on:** draft-accept parity (GPU Q4Mlx vs host-dense/MLX). The notebook continues with [38-clustered-mtp-e4b](38-clustered-mtp-e4b.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: MLX DFlash vs greedy — exact match — `dflash_exact_vs_greedy: true`; greet gold starts `[100, 45518, 107, …]` — Product reference.
- The result artifact reports: MLX intermediates — block_size=5 — `target_next_argmax=531`; `proposed_block_tokens=[14359, 532, 107, 563]`; embed_scale **73.32121**; h_ctx absmean **0.069892** — Capture layers [1,12,23,35,46,57].
- The result artifact reports: Native greet16 — **match_prefix=0/16** — first_mismatch at index 0 — got collapsed `[236773, 236799×…]` vs gold.
- Failure/operational record: Native greedy collapsed vs MLX on greet (prefix 0)
- Failure/operational record: Intermediate dumps on native still incomplete vs gold for first-block proposals (audit notes)

## What this does not prove

Capture-on Hot golden parity does **not** prove product draft-accept equals MLX (GPU Q4Mlx mean_accept still below host-dense ceiling). Capture-off free-decode still collapses — capture remains required for 31B. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets.

## See also

- Related suites: [`36-native-dflash-parity-accept`](36-native-dflash-parity-accept.md), [`32-native-fusion-2026-07-14`](32-native-fusion-2026-07-14.md)
- Lane A: [`lane_a_todo_status_2026-07-19.md`](../../Rust_MLKit/gemma-metal/bench/results/lane_a_todo_status_2026-07-19.md)

---

[Previous](36-native-dflash-parity-accept.md) · [Index](../00-INDEX.md) · [Next](38-clustered-mtp-e4b.md)
