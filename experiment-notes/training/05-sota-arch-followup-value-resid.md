# sota_arch_followup_value_resid: Cross kv1 / LN / VE / XSA on champion

## Executive summary

- **Question:** Crossing the arch-ladder champion (`gated_value_resid`) and near-champion (`value_resid`) with KV1, LN scale, VE placement, and XSA last-N would find a cheap additive win — or at least confirm the champion is stable under those knobs.
- **Result:** Short hint (`gated_value_resid_xsa1` @ 2.6938) died on promotion — treat suite 04’s long champion (1.985) as held until mid/long rc=1 failures are fixed and re-run.
- **Implication:** Do not promote this beyond the completed stages; the missing or failed confirmation is decision-relevant.
- **Status:** `partial`; evidence confidence **Low**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `sota_arch_followup_value_resid` |
| Dates | `2026-04-02` – `2026-04-02` |
| Hardware | RTX 3070 Ti (local ablation via `run_ablation_3070ti.py`) |
| Status | `partial` |

## Hypothesis

Crossing the arch-ladder champion (`gated_value_resid`) and near-champion (`value_resid`) with KV1, LN scale, VE placement, and XSA last-N would find a cheap additive win — or at least confirm the champion is stable under those knobs.

## Setup

- Trainer / preset: `base_preset: sota` → `train_gpt_sprint.py`
- Fixed knobs: short 300 (promote 4) → mid 1000 (promote 2) → long 3000 seeds `[1337, 42]`; rank `calibrated_bpb`
- Env flags: each candidate keeps `VALUE_RESIDUAL` and/or `GATED_ATTENTION` plus one extra lever
- Reference copy: `Rust_MLKit/reference/ablation_results/champion_followup.json` / `followup_summary.txt`

## Variants

| Variant | Change |
|---------|--------|
| `value_resid` | `VALUE_RESIDUAL=1` |
| `value_resid_kv1` | + `NUM_KV_HEADS=1` |
| `value_resid_kv1_ln` | + `NUM_KV_HEADS=1`, `LN_SCALE=1` |
| `value_resid_ve_mid` | + `VE_DIM=16`, `VE_LAYERS=1,2` |
| `value_resid_ve_late` | + `VE_DIM=16`, `VE_LAYERS=2,3` |
| `gated_value_resid` | `GATED_ATTENTION=1`, `VALUE_RESIDUAL=1` |
| `gated_value_resid_kv1` | champion + `NUM_KV_HEADS=1` |
| `gated_value_resid_kv1_ln` | champion + KV1 + `LN_SCALE=1` |
| `gated_value_resid_xsa1` | champion + `XSA_LAST_N=1` |
| `gated_value_resid_xsa3` | champion + `XSA_LAST_N=3` |
| `gated_value_resid_ve_mid` | champion + mid VE |
| `gated_value_resid_ve_late` | champion + late VE |

## Results

Short (seed 1337) — only stage with completed metrics:

| Rank | Run / config | calibrated_bpb | Notes |
|------|--------------|----------------|-------|
| 1 | `gated_value_resid_xsa1` | **2.693802** | sliding 2.716102; step 620.63 ms; artifact 1,319,757 B; rc `[0]` |
| 2 | `gated_value_resid` | 2.700980 | Δ+0.007; matches arch-ladder short champion |
| 3 | `gated_value_resid_kv1_ln` | 2.724408 | artifact 1,259,297 B |
| 4 | `gated_value_resid_kv1` | 2.724408 | identical BPB to kv1_ln |
| 5 | `value_resid_kv1` | 2.728121 | |
| 6 | `value_resid_kv1_ln` | 2.728121 | |
| 7 | `value_resid` | 2.730309 | |
| 8 | `value_resid_ve_mid` | 2.732342 | |
| 9 | `value_resid_ve_late` | 2.744752 | |
| 10–12 | `gated_value_resid_xsa3` / `_ve_mid` / `_ve_late` | n/a | rc `[1]` — no metrics |

Mid: all four promoted candidates rc `[1]` with null BPB (`gated_value_resid`, `_kv1`, `_kv1_ln`, `_xsa1`).

Long: both remaining candidates rc `[1, 1]` with null metrics:

| Rank | Run / config | calibrated_bpb | returncodes |
|------|--------------|----------------|-------------|
| 1* | `gated_value_resid` | null | `[1, 1]` |
| 2* | `gated_value_resid_kv1` | null | `[1, 1]` |

\*Ordering from `long.summary.txt` is by candidate order among failed runs, not by BPB.

Champion.json (and reference `champion_followup.json`): still labeled `gated_value_resid`, stage `long`, but `calibrated_bpb` / `sliding_bpb` / `step_avg_ms` / `artifact_bytes` are all **null**, returncodes `[1, 1]`.

Readout: short suggests `XSA_LAST_N=1` on the gated+value-resid stack is a small edge (~0.007 BPB) over the ladder champion at 300 steps, while KV1 trades ~0.02 BPB for a smaller artifact. That signal was never confirmed — mid and long stages collapsed with rc=1 for every promoted gated combo.

**Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

## Failures

- Short rc=`1`: `gated_value_resid_xsa3`, `gated_value_resid_ve_mid`, `gated_value_resid_ve_late` (null metrics).
- Mid: all 4 candidates rc=`1`.
- Long: both seeds for `gated_value_resid` and `gated_value_resid_kv1` rc=`1` → champion record is non-evaluable.
- No successful mid/long calibrated_bpb to promote over suite 04’s long champion (1.984742).

## Lesson

**Short hint (`gated_value_resid_xsa1` @ 2.6938) died on promotion — treat suite 04’s long champion (1.985) as held until mid/long rc=1 failures are fixed and re-run.**

## Reproduction

- Replay: `python3 parameter-golf/run_ablation_3070ti.py sota_arch_followup_value_resid --resume`.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: Low.** Only one-seed short metrics are usable; all promoted mid and long variants failed with return code 1.

## Artifacts

- `parameter-golf/logs/ablations/sota_arch_followup_value_resid/champion.json`
- `parameter-golf/logs/ablations/sota_arch_followup_value_resid/short.summary.json` / `.txt`
- `parameter-golf/logs/ablations/sota_arch_followup_value_resid/mid.summary.json`
- `parameter-golf/logs/ablations/sota_arch_followup_value_resid/long.summary.json`
- `parameter-golf/logs/ablations/sota_arch_followup_value_resid/latest_summary.txt`
- `Rust_MLKit/reference/ablation_results/champion_followup.json`
- `Rust_MLKit/reference/ablation_results/followup_summary.txt`
- `parameter-golf/conductor/ablation_suites_3070ti.json` → key `sota_arch_followup_value_resid`

## Why this experiment happened

Suite 04 identified value residual as the robust lever and `gated_value_resid` as the nominal champion. This suite was designed to search around that narrow basin with KV1, LN scale, VE placement, and XSA depth rather than restart a broad architecture search. The preceding notebook context is [04-sota-arch-ladder](04-sota-arch-ladder.md).

## Experiment story

**Baseline.** Suite 04 identified value residual as the robust lever and `gated_value_resid` as the nominal champion. This suite was designed to search around that narrow basin with KV1, LN scale, VE placement, and XSA depth rather than restart a broad architecture search. The preceding notebook context is [04-sota-arch-ladder](04-sota-arch-ladder.md).

**Hypothesis.** Crossing the arch-ladder champion (`gated_value_resid`) and near-champion (`value_resid`) with KV1, LN scale, VE placement, and XSA last-N would find a cheap additive win — or at least confirm the champion is stable under those knobs.

**Test contract.** Trainer / preset: `base_preset: sota` → `train_gpt_sprint.py` Fixed knobs: short 300 (promote 4) → mid 1000 (promote 2) → long 3000 seeds `[1337, 42]`; rank `calibrated_bpb` Env flags: each candidate keeps `VALUE_RESIDUAL` and/or `GATED_ATTENTION` plus one extra lever Reference copy: `Rust_MLKit/reference/ablation_results/champion_followup.json` / `followup_summary.txt`

**Variant sequence.** The preserved comparison matrix was: `value_resid` — `VALUE_RESIDUAL=1`; `value_resid_kv1` — + `NUM_KV_HEADS=1`; `value_resid_kv1_ln` — + `NUM_KV_HEADS=1`, `LN_SCALE=1`; `value_resid_ve_mid` — + `VE_DIM=16`, `VE_LAYERS=1,2`; `value_resid_ve_late` — + `VE_DIM=16`, `VE_LAYERS=2,3`; `gated_value_resid` — `GATED_ATTENTION=1`, `VALUE_RESIDUAL=1`.

**Measured turn.** The result board records 1 — `gated_value_resid_xsa1` — **2.693802** — sliding 2.716102; step 620.63 ms; artifact 1,319,757 B; rc `[0]`; 2 — `gated_value_resid` — 2.700980 — Δ+0.007; matches arch-ladder short champion; 3 — `gated_value_resid_kv1_ln` — 2.724408 — artifact 1,259,297 B; 4 — `gated_value_resid_kv1` — 2.724408 — identical BPB to kv1_ln; 5 — `value_resid_kv1` — 2.728121 — .

**Turning point and readout.** Champion.json (and reference `champion_followup.json`): still labeled `gated_value_resid`, stage `long`, but `calibrated_bpb` / `sliding_bpb` / `step_avg_ms` / `artifact_bytes` are all **null**, returncodes `[1, 1]`. Readout: short suggests `XSA_LAST_N=1` on the gated+value-resid stack is a small edge (~0.007 BPB) over the ladder champion at 300 steps, while KV1 trades ~0.02 BPB for a smaller artifact. That signal was never confirmed — mid and long stages collapsed with rc=1 for every promoted gated combo. **Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

**Failures and surprises.** Short rc=`1`: `gated_value_resid_xsa3`, `gated_value_resid_ve_mid`, `gated_value_resid_ve_late` (null metrics). Mid: all 4 candidates rc=`1`. Long: both seeds for `gated_value_resid` and `gated_value_resid_kv1` rc=`1` → champion record is non-evaluable. No successful mid/long calibrated_bpb to promote over suite 04’s long champion (1.984742).

## Decision and aftermath

**Kept:** Short hint (`gated_value_resid_xsa1` @ 2.6938) died on promotion — treat suite 04’s long champion (1.985) as held until mid/long rc=1 failures are fixed and re-run. **Boundary:** Incomplete or failed confirmation prevents promotion beyond the explicitly successful stages. The notebook continues with [06-toy-aprdh](06-toy-aprdh.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `gated_value_resid_xsa1` — **2.693802** — sliding 2.716102; step 620.63 ms; artifact 1,319,757 B; rc `[0]`.
- The result artifact reports: 2 — `gated_value_resid` — 2.700980 — Δ+0.007; matches arch-ladder short champion.
- The result artifact reports: 3 — `gated_value_resid_kv1_ln` — 2.724408 — artifact 1,259,297 B.
- The result artifact reports: 4 — `gated_value_resid_kv1` — 2.724408 — identical BPB to kv1_ln.
- Failure/operational record: Short rc=`1`: `gated_value_resid_xsa3`, `gated_value_resid_ve_mid`, `gated_value_resid_ve_late` (null metrics).
- Failure/operational record: Mid: all 4 candidates rc=`1`.

## What this does not prove

**Confidence: Low.** Only one-seed short metrics are usable; all promoted mid and long variants failed with return code 1. Incomplete or failed confirmation prevents promotion beyond the explicitly successful stages. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.1 (combo re-test lesson)
- Related suites: [`04-sota-arch-ladder`](04-sota-arch-ladder.md)

---

[Previous](04-sota-arch-ladder.md) · [Index](../00-INDEX.md) · [Next](06-toy-aprdh.md)
