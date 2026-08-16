# sota_arch_ladder: Gated attention + value residual champion

## Executive summary

- **Question:** Architectural levers (KV compression, VE placement, gated attention, value residual, recursive weight-sharing) would dominate hyperparameter tweaks; combinations must be re-tested because some levers hurt alone.
- **Result:** Champion = gated attention + value residual (calibrated BPB 1.985); re-test combos — gating alone is ~0.10 BPB worse, and recursive share (`recur_2x3` 2.851) destroys depth specialization.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **High**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `sota_arch_ladder` |
| Dates | `2026-04-01` – `2026-04-01` |
| Hardware | RTX 3070 Ti (local ablation via `run_ablation_3070ti.py`) |
| Status | `done` |

## Hypothesis

Architectural levers (KV compression, VE placement, gated attention, value residual, recursive weight-sharing) would dominate hyperparameter tweaks; combinations must be re-tested because some levers hurt alone.

## Setup

- Trainer / preset: `base_preset: sota` → `train_gpt_sprint.py` (4L / 128-d toy proxy)
- Fixed knobs: calib on; stages short 300 (promote 5) → mid 1000 (promote 3) → long 3000 seeds `[1337, 42]`
- Ranking: `calibrated_bpb`, then `sliding_bpb`, `step_avg_ms`, `artifact_bytes`
- Reference copies also archived under `Rust_MLKit/reference/ablation_results/`

## Variants

| Variant | Change |
|---------|--------|
| `control` | (none) |
| `kv1` | `NUM_KV_HEADS=1` |
| `kv1_ln` | `NUM_KV_HEADS=1`, `LN_SCALE=1` |
| `kv1_lean_ve` | `NUM_KV_HEADS=1`, `VE_DIM=16`, `VE_LAYERS=2,3` |
| `kv1_bigram64` | `NUM_KV_HEADS=1`, `BIGRAM_DIM=64` |
| `ve_mid` | `VE_DIM=16`, `VE_LAYERS=1,2` |
| `ve_late_wide` | `VE_DIM=32`, `VE_LAYERS=2,3` |
| `gated_attn` | `GATED_ATTENTION=1` |
| `value_resid` | `VALUE_RESIDUAL=1` |
| `gated_value_resid` | `GATED_ATTENTION=1`, `VALUE_RESIDUAL=1` |
| `recur_2x3` | `ARCH_VARIANT=recur_int6`, `RECUR_PASSES=3`, `RECUR_UNIQUE_LAYERS=2`, `NUM_LAYERS=6`, `XSA_LAST_N=2` |

## Results

Short (seed 1337) already preferred the combo; recursive sharing last:

| Rank | Run / config | calibrated_bpb | Notes |
|------|--------------|----------------|-------|
| 1 | `gated_value_resid` | 2.700980 | artifact 1,319,305 B |
| 2 | `control` | 2.703687 | fastest among leaders (432 ms) |
| 3 | `value_resid` | 2.730309 | |
| … | `recur_2x3` | **2.851402** | artifact 777,497 B but catastrophic BPB; sliding 3.139450 |

Mid (seed 1337) — gating alone briefly #1, combo within noise:

| Rank | Run / config | calibrated_bpb |
|------|--------------|----------------|
| 1 | `gated_attn` | 2.217801 |
| 2 | `gated_value_resid` | 2.218268 |
| 3 | `value_resid` | 2.219074 |
| 4 | `control` | 2.223079 |
| 5 | `ve_mid` | 2.229640 |

Long (seeds 1337+42) — primary champion board:

| Rank | Run / config | calibrated_bpb | Notes |
|------|--------------|----------------|-------|
| 1 | `gated_value_resid` | **1.984742** | sliding 1.987976; step 1441.18 ms; artifact **1,341,003** B; rcs `[0,0]` |
| 2 | `value_resid` | 1.987491 | Δ+0.002749 (seed-noise margin); step 837.09 ms |
| 3 | `gated_attn` | 2.088671 | Δ+0.103929 — gating alone collapses |

Champion.json (matches `Rust_MLKit/reference/ablation_results/champion_arch_ladder.json`): `gated_value_resid`, calibrated_bpb **1.9847424805646139**, sliding_bpb **1.98797577**, step_avg_ms **1441.175**, artifact_bytes **1341003**.

Readout: value residual carries almost all of the win; adding gated attention is a tiny long-edge improvement (~0.003 BPB) but gating **without** value residual is ~0.104 BPB worse than the champion. Recursive weight-sharing (`recur_2x3`) is a clear short-stage failure (2.851). KV1 / VE rearrangements did not crack the top of the long board (not promoted to long).

**Interpretation boundary.** The ~0.104 BPB gap to gating-alone is much larger than the ~0.0027 BPB edge over value-residual-only. Treat value residual as the supported quality lever and the combined champion ordering as near-noise.

## Failures

- All long returncodes `[0, 0]`; no OOMs in summaries.
- `recur_2x3` is a quality failure (completed with rc 0), not a crash.

## Lesson

**Champion = gated attention + value residual (calibrated BPB 1.985); re-test combos — gating alone is ~0.10 BPB worse, and recursive share (`recur_2x3` 2.851) destroys depth specialization.**

## Reproduction

- Replay: `python3 parameter-golf/run_ablation_3070ti.py sota_arch_ladder --resume`.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: High.** Short, mid, and two-seed long stages completed. The 0.0027 BPB champion edge over value-residual-only is within seed noise, while the larger gating-alone gap is clear.

## Artifacts

- `parameter-golf/logs/ablations/sota_arch_ladder/champion.json`
- `parameter-golf/logs/ablations/sota_arch_ladder/long.summary.json` / `.txt`
- `parameter-golf/logs/ablations/sota_arch_ladder/mid.summary.json`
- `parameter-golf/logs/ablations/sota_arch_ladder/short.summary.json`
- `parameter-golf/logs/ablations/sota_arch_ladder/latest_summary.txt`
- `Rust_MLKit/reference/ablation_results/champion_arch_ladder.json`
- `Rust_MLKit/reference/ablation_results/arch_ladder_summary.txt`
- `Rust_MLKit/reference/ablation_results/arch_ladder_long_results.json`
- `parameter-golf/conductor/ablation_suites_3070ti.json` → key `sota_arch_ladder`

## Why this experiment happened

The first three suites showed that cheap local knobs could matter, but also that short-stage and long-stage rankings could reverse. The next question was therefore architectural: could value flow, attention gating, KV compression, or weight reuse produce a gain large enough to survive the staged ladder? The preceding notebook context is [03-sota-depth-proxy](03-sota-depth-proxy.md).

## Experiment story

**Baseline.** The first three suites showed that cheap local knobs could matter, but also that short-stage and long-stage rankings could reverse. The next question was therefore architectural: could value flow, attention gating, KV compression, or weight reuse produce a gain large enough to survive the staged ladder? The preceding notebook context is [03-sota-depth-proxy](03-sota-depth-proxy.md).

**Hypothesis.** Architectural levers (KV compression, VE placement, gated attention, value residual, recursive weight-sharing) would dominate hyperparameter tweaks; combinations must be re-tested because some levers hurt alone.

**Test contract.** Trainer / preset: `base_preset: sota` → `train_gpt_sprint.py` (4L / 128-d toy proxy) Fixed knobs: calib on; stages short 300 (promote 5) → mid 1000 (promote 3) → long 3000 seeds `[1337, 42]` Ranking: `calibrated_bpb`, then `sliding_bpb`, `step_avg_ms`, `artifact_bytes` Reference copies also archived under `Rust_MLKit/reference/ablation_results/`

**Variant sequence.** The preserved comparison matrix was: `control` — (none); `kv1` — `NUM_KV_HEADS=1`; `kv1_ln` — `NUM_KV_HEADS=1`, `LN_SCALE=1`; `kv1_lean_ve` — `NUM_KV_HEADS=1`, `VE_DIM=16`, `VE_LAYERS=2,3`; `kv1_bigram64` — `NUM_KV_HEADS=1`, `BIGRAM_DIM=64`; `ve_mid` — `VE_DIM=16`, `VE_LAYERS=1,2`.

**Measured turn.** The result board records 1 — `gated_value_resid` — 2.700980 — artifact 1,319,305 B; 2 — `control` — 2.703687 — fastest among leaders (432 ms); 3 — `value_resid` — 2.730309 — ; … — `recur_2x3` — **2.851402** — artifact 777,497 B but catastrophic BPB; sliding 3.139450; 1 — `gated_attn` — 2.217801.

**Turning point and readout.** Champion.json (matches `Rust_MLKit/reference/ablation_results/champion_arch_ladder.json`): `gated_value_resid`, calibrated_bpb **1.9847424805646139**, sliding_bpb **1.98797577**, step_avg_ms **1441.175**, artifact_bytes **1341003**. Readout: value residual carries almost all of the win; adding gated attention is a tiny long-edge improvement (~0.003 BPB) but gating **without** value residual is ~0.104 BPB worse than the champion. Recursive weight-sharing (`recur_2x3`) is a clear short-stage failure (2.851). KV1 / VE rearrangements did not crack the top of the long board (not promoted to long). **Interpretation boundary.** The ~0.104 BPB gap to gating-alone is much larger than the ~0.0027 BPB edge over value-residual-only. Treat value residual as the supported quality lever and the combined champion ordering as near-noise.

**Failures and surprises.** All long returncodes `[0, 0]`; no OOMs in summaries. `recur_2x3` is a quality failure (completed with rc 0), not a crash.

## Decision and aftermath

**Kept:** Champion = gated attention + value residual (calibrated BPB 1.985); re-test combos — gating alone is ~0.10 BPB worse, and recursive share (`recur_2x3` 2.851) destroys depth specialization. The notebook continues with [05-sota-arch-followup-value-resid](05-sota-arch-followup-value-resid.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `gated_value_resid` — 2.700980 — artifact 1,319,305 B.
- The result artifact reports: 2 — `control` — 2.703687 — fastest among leaders (432 ms).
- The result artifact reports: 3 — `value_resid` — 2.730309 — .
- The result artifact reports: … — `recur_2x3` — **2.851402** — artifact 777,497 B but catastrophic BPB; sliding 3.139450.
- Failure/operational record: All long returncodes `[0, 0]`; no OOMs in summaries.
- Failure/operational record: `recur_2x3` is a quality failure (completed with rc 0), not a crash.

## What this does not prove

**Confidence: High.** Short, mid, and two-seed long stages completed. The 0.0027 BPB champion edge over value-residual-only is within seed noise, while the larger gating-alone gap is clear. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.1
- Related suites: [`05-sota-arch-followup-value-resid`](05-sota-arch-followup-value-resid.md)

---

[Previous](03-sota-depth-proxy.md) · [Index](../00-INDEX.md) · [Next](05-sota-arch-followup-value-resid.md)
