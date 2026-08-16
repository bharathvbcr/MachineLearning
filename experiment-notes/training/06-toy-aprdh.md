# toy_aprdh: Adaptive raw-byte vs RADA / DeltaHybrid (attempted)

## Executive summary

- **Question:** An adaptive raw-byte recurrent architecture (APRDH) — weight-shared block reused 2–5× with GDN + MLA routing + optional engram memory — could compete with toy token baselines (RADA / DeltaHybrid) on raw-byte BPB under a 3070 Ti budget.
- **Result:** APRDH did not finish a comparable benchmark here — v0/engram failed on a missing `apply_rotary_emb`; only an interrupted risky run reached val raw-byte BPB ~3.72.
- **Implication:** This records a blocker and partial learning signal, not a comparative architecture result.
- **Status:** `blocked`; evidence confidence **Low**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `toy_aprdh` (via `run_toy_benchmarks.py`) |
| Dates | `2026-03-26` – `2026-03-26` |
| Hardware | RTX 3070 Ti (`device:cuda`, single-GPU; logs spell out target) |
| Status | `blocked` |

## Hypothesis

An adaptive raw-byte recurrent architecture (APRDH) — weight-shared block reused 2–5× with GDN + MLA routing + optional engram memory — could compete with toy token baselines (RADA / DeltaHybrid) on raw-byte BPB under a 3070 Ti budget.

## Setup

- Trainer / harness: `parameter-golf/run_toy_benchmarks.py` → `run_toy_adaptive.py` / `run_rada.py` / `run_hypercascade.py`
- Default benchmark set (harness): `toy_aprdh_v0`, `toy_aprdh_risky`, `toy_aprdh_engram_risky`, `toy_aprdh_ttt`, `toy_rada`, `toy_deltahybrid`
- APRDH configs seen in logs: `arch_version=toy_aprdh_v1`, seed 1337, seq 256; v0 uses `model_dim=128 heads=4 batch_tokens=4096 recur:2->5`; risky / engram-risky use `model_dim=160 heads=5 batch_tokens=3072`
- Outputs expected: `logs/benchmarks/{run-name}.txt` + `toy_benchmark_summary.{json,csv}`

## Variants

| Variant | Change |
|---------|--------|
| `toy_aprdh_v0` | adaptive raw-byte baseline (`kind=adaptive_raw_byte`) |
| `toy_aprdh_engram_risky` | + n-gram engram memory (`kind=adaptive_raw_byte_engram`) |
| `toy_aprdh_risky` | riskier 160-d shape (log present; **not** in summary JSON) |
| `toy_rada` / `toy_deltahybrid` | token baselines in harness — **absent** from this summary dump |

## Results

From `toy_benchmark_summary.json` / `.csv` (only completed summary rows):

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| — | `toy_aprdh_v0` | all metrics null | `failed=true`, `returncode=1`, wall **83.51** s |
| — | `toy_aprdh_engram_risky` | all metrics null | `failed=true`, `returncode=1`, wall **131.43** s |

No calibrated/token BPB, int6/int8 sizes, or CUDA peak MB were written for either summary row. RADA and DeltaHybrid do not appear in this summary file.

Partial training curve from standalone log `toy_aprdh_risky.txt` (not in the summary JSON; interrupted):

| Step | val `raw_byte_bpb` |
|------|--------------------|
| 50 | 3.848453 |
| 100 | 3.725821 |
| 150 | **3.722959** (best logged) |

Training continued to step 160 (`train` bpb ~4.15) then hit `KeyboardInterrupt` during int8 quantize / packaging — no final roundtrip BPB.

Readout: the official benchmark summary is a failure board, not a quality ranking. The only numeric evidence of learning is the interrupted risky run (val raw-byte BPB down to ~3.72 by step 150). Pedagogy in learning-notes §8.7 still describes the architecture intent; it does not claim a win over RADA/DeltaHybrid from this dump.

**Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

## Failures

- `toy_aprdh_v0` / `toy_aprdh_engram_risky`: `NameError: name 'apply_rotary_emb' is not defined` in `train_toy_adaptive.py` (`_mla_sparse`) — hard crash before training metrics.
- `toy_aprdh_risky`: trained ~160/200 steps then `KeyboardInterrupt` during `quantize_state_dict_int8` / failure-bundle path; never entered the summary JSON.
- `toy_rada`, `toy_deltahybrid`, `toy_aprdh_ttt`: not present in `toy_benchmark_summary.*` for this date.

## Lesson

**APRDH did not finish a comparable benchmark here — v0/engram failed on a missing `apply_rotary_emb`; only an interrupted risky run reached val raw-byte BPB ~3.72.**

## Reproduction

- Replay: `python3 parameter-golf/run_toy_benchmarks.py` (currently reproduces the recorded blocker until `apply_rotary_emb` is restored).
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: Low.** Comparable benchmark rows failed before metrics, baselines are absent, and the sole learning curve was interrupted before packaging.

## Artifacts

- `parameter-golf/logs/benchmarks/toy_benchmark_summary.json`
- `parameter-golf/logs/benchmarks/toy_benchmark_summary.csv`
- `parameter-golf/logs/benchmarks/toy_aprdh_v0.txt`
- `parameter-golf/logs/benchmarks/toy_aprdh_engram_risky.txt`
- `parameter-golf/logs/benchmarks/toy_aprdh_risky.txt`
- `parameter-golf/run_toy_benchmarks.py`

## Why this experiment happened

The catastrophic `recur_2x3` quality result in suite 04 made naive weight sharing unattractive, but it did not answer whether routed, adaptive reuse could work. APRDH was the more structured test: recurrent application, GDN/MLA routing, optional engram memory, and a compute controller, compared against RADA and DeltaHybrid. The preceding notebook context is [05-sota-arch-followup-value-resid](05-sota-arch-followup-value-resid.md).

## Experiment story

**Baseline.** The catastrophic `recur_2x3` quality result in suite 04 made naive weight sharing unattractive, but it did not answer whether routed, adaptive reuse could work. APRDH was the more structured test: recurrent application, GDN/MLA routing, optional engram memory, and a compute controller, compared against RADA and DeltaHybrid. The preceding notebook context is [05-sota-arch-followup-value-resid](05-sota-arch-followup-value-resid.md).

**Hypothesis.** An adaptive raw-byte recurrent architecture (APRDH) — weight-shared block reused 2–5× with GDN + MLA routing + optional engram memory — could compete with toy token baselines (RADA / DeltaHybrid) on raw-byte BPB under a 3070 Ti budget.

**Test contract.** Trainer / harness: `parameter-golf/run_toy_benchmarks.py` → `run_toy_adaptive.py` / `run_rada.py` / `run_hypercascade.py` Default benchmark set (harness): `toy_aprdh_v0`, `toy_aprdh_risky`, `toy_aprdh_engram_risky`, `toy_aprdh_ttt`, `toy_rada`, `toy_deltahybrid` APRDH configs seen in logs: `arch_version=toy_aprdh_v1`, seed 1337, seq 256; v0 uses `model_dim=128 heads=4 batch_tokens=4096 recur:2->5`; risky / engram-risky use `model_dim=160 heads=5 batch_tokens=3072` Outputs expected: `logs/benchmarks/{run-name}.txt` + `toy_benchmark_summary.{json,csv}`

**Variant sequence.** The preserved comparison matrix was: `toy_aprdh_v0` — adaptive raw-byte baseline (`kind=adaptive_raw_byte`); `toy_aprdh_engram_risky` — + n-gram engram memory (`kind=adaptive_raw_byte_engram`); `toy_aprdh_risky` — riskier 160-d shape (log present; **not** in summary JSON); `toy_rada` / `toy_deltahybrid` — token baselines in harness — **absent** from this summary dump.

**Measured turn.** The result board records — — `toy_aprdh_v0` — all metrics null — `failed=true`, `returncode=1`, wall **83.51** s; — — `toy_aprdh_engram_risky` — all metrics null — `failed=true`, `returncode=1`, wall **131.43** s; Step — val `raw_byte_bpb`; 50 — 3.848453; 100 — 3.725821.

**Turning point and readout.** Training continued to step 160 (`train` bpb ~4.15) then hit `KeyboardInterrupt` during int8 quantize / packaging — no final roundtrip BPB. Readout: the official benchmark summary is a failure board, not a quality ranking. The only numeric evidence of learning is the interrupted risky run (val raw-byte BPB down to ~3.72 by step 150). Pedagogy in learning-notes §8.7 still describes the architecture intent; it does not claim a win over RADA/DeltaHybrid from this dump. **Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

**Failures and surprises.** `toy_aprdh_v0` / `toy_aprdh_engram_risky`: `NameError: name 'apply_rotary_emb' is not defined` in `train_toy_adaptive.py` (`_mla_sparse`) — hard crash before training metrics. `toy_aprdh_risky`: trained ~160/200 steps then `KeyboardInterrupt` during `quantize_state_dict_int8` / failure-bundle path; never entered the summary JSON. `toy_rada`, `toy_deltahybrid`, `toy_aprdh_ttt`: not present in `toy_benchmark_summary.*` for this date.

## Decision and aftermath

**Kept:** APRDH did not finish a comparable benchmark here — v0/engram failed on a missing `apply_rotary_emb`; only an interrupted risky run reached val raw-byte BPB ~3.72. **Boundary:** The preserved failures occur before a comparable board exists, so the suite does not rank APRDH against its named baselines. The notebook continues with [07-h100-conductor-planned](07-h100-conductor-planned.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: — — `toy_aprdh_v0` — all metrics null — `failed=true`, `returncode=1`, wall **83.51** s.
- The result artifact reports: — — `toy_aprdh_engram_risky` — all metrics null — `failed=true`, `returncode=1`, wall **131.43** s.
- The result artifact reports: Step — val `raw_byte_bpb`.
- The result artifact reports: 50 — 3.848453.
- Failure/operational record: `toy_aprdh_v0` / `toy_aprdh_engram_risky`: `NameError: name 'apply_rotary_emb' is not defined` in `train_toy_adaptive.py` (`_mla_sparse`) — hard crash before training metrics.
- Failure/operational record: `toy_aprdh_risky`: trained ~160/200 steps then `KeyboardInterrupt` during `quantize_state_dict_int8` / failure-bundle path; never entered the summary JSON.

## What this does not prove

**Confidence: Low.** Comparable benchmark rows failed before metrics, baselines are absent, and the sole learning curve was interrupted before packaging. The preserved failures occur before a comparable board exists, so the suite does not rank APRDH against its named baselines. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.7 (APRDH design; contrast with naive `recur_2x3` failure in §8.1 / suite 04)
- Related suites: [`04-sota-arch-ladder`](04-sota-arch-ladder.md) (`recur_2x3`)

---

[Previous](05-sota-arch-followup-value-resid.md) · [Index](../00-INDEX.md) · [Next](07-h100-conductor-planned.md)
