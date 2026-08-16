# 19: Chunk-parallel kernels — Mamba-2 / GDN

## Executive summary

- **Question:** Replacing O(T) sequential SSM/delta scans with verified chunk-parallel kernels should restore usable tok/s (and unlock longer ctx) without changing numerical behavior vs the sequential reference.
- **Result:** A custom kernel is worthless until verified against a brute-force reference; accumulation needs fp32.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **High**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `19-chunk-parallel-kernels` |
| Dates | 2026-06-15 – 2026-06-15 (ports + benches; ongoing in `mixers.py`) |
| Hardware | RTX 3070 Ti Laptop 8 GB |
| Status | `done` |

## Hypothesis

Replacing O(T) sequential SSM/delta scans with verified chunk-parallel kernels should restore usable tok/s (and unlock longer ctx) without changing numerical behavior vs the sequential reference.

## Setup

- Trainer / preset: ports into `nanolab/mixers.py` from verified parameter-golf refs
- Fixed knobs for headline benches: bs8/ctx512 (same shape as mixer sweep); `mixer_chunk` 32–64
- Verification: `verify_scan.py` (SSD, 1e-5), GDN sequential↔chunk tests in `nanolab/tests.py`; also `verify_gdn.py` / `verify_gdn_wy.py` in the golf tree
- Env flags: scan math in **fp32** (autocast disabled inside fwd/bwd)

## Variants

| Variant | Change |
|---------|--------|
| mamba2 sequential | O(T) reference loop |
| mamba2 `ssd_chunk_parallel` | Two-pass chunk SSD |
| gdn sequential | O(T) gated delta |
| gdn chunk / WY | Chunked (later fully vectorized WY per README) |

## Results

Primary scoreboard from [08.5](../../learning-notes/08-experiments-and-results.md) (pre→post at bs8/512):

| Kernel | Before → After | Unlocks |
|--------|----------------|---------|
| **Mamba-2 SSD** | 333 → **3,224 tok/s (9.7×)** | trainable at ctx1024 |
| **GDN** | 238 → **482 tok/s (2×)**, 4.7→2.6 GB | — |
| **GDN** bs16/ctx1024 | **OOM → 1,100 tok/s @ 4.0 GB** | was impossible |

`nanolab/README.md` also records a later fully-vectorized WY GDN path (~**1.6K** tok/s @ bs8/512, 6.7×) and mamba2 ctx1024 ~2.0K tok/s @ 6.0 GB with `mixer_chunk=32`. Both families are pinned by output **and** input-gradient checks, including non-divisible-T padding.

**Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

## Failures

- **fp32 accumulation bug:** CPU-only tests missed autocast drift — scan must run in fp32 (lesson from 08.5).
- Pre-kernel gdn/mamba2 in `gpu_sweep_mixer.json` remain the cautionary 238/333 tok/s baselines.

## Lesson

**A custom kernel is worthless until verified against a brute-force reference; accumulation needs fp32.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: the verification entry points in `nanolab/tests.py` and the parameter-golf `verify_scan.py` / `verify_gdn*.py` scripts.
- Required replay inputs: `nanolab/README.md` benchmark shapes and `nanolab/out/gpu_sweep_mixer.json` baselines. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: High.** Speedups are paired with output and input-gradient parity checks, fp32 accumulation checks, and non-divisible-length coverage.

## Artifacts

- `nanolab/mixers.py` — `ssd_chunk_parallel`, GDN chunk/WY
- `nanolab/tests.py` — sequential↔chunk parity
- `nanolab/out/gpu_sweep_mixer.json` — sequential baselines
- `nanolab/README.md` — kernel speedup table
- parameter-golf `verify_scan.py` / `verify_gdn*.py` (upstream refs)

## Why this experiment happened

Suite 17 found Mamba-2 and GDN at only 333 and 238 tok/s—24–33× behind attention-class paths. That gap made further wall-clock quality comparisons misleading until the sequential scans were replaced by verified parallel formulations. The preceding notebook context is [18-gpu-maximization](18-gpu-maximization.md).

## Experiment story

**Baseline.** Suite 17 found Mamba-2 and GDN at only 333 and 238 tok/s—24–33× behind attention-class paths. That gap made further wall-clock quality comparisons misleading until the sequential scans were replaced by verified parallel formulations. The preceding notebook context is [18-gpu-maximization](18-gpu-maximization.md).

**Hypothesis.** Replacing O(T) sequential SSM/delta scans with verified chunk-parallel kernels should restore usable tok/s (and unlock longer ctx) without changing numerical behavior vs the sequential reference.

**Test contract.** Trainer / preset: ports into `nanolab/mixers.py` from verified parameter-golf refs Fixed knobs for headline benches: bs8/ctx512 (same shape as mixer sweep); `mixer_chunk` 32–64 Verification: `verify_scan.py` (SSD, 1e-5), GDN sequential↔chunk tests in `nanolab/tests.py`; also `verify_gdn.py` / `verify_gdn_wy.py` in the golf tree Env flags: scan math in **fp32** (autocast disabled inside fwd/bwd)

**Variant sequence.** The preserved comparison matrix was: mamba2 sequential — O(T) reference loop; mamba2 `ssd_chunk_parallel` — Two-pass chunk SSD; gdn sequential — O(T) gated delta; gdn chunk / WY — Chunked (later fully vectorized WY per README).

**Measured turn.** The result board records **Mamba-2 SSD** — 333 → **3,224 tok/s (9.7×)** — trainable at ctx1024; **GDN** — 238 → **482 tok/s (2×)**, 4.7→2.6 GB — —; **GDN** bs16/ctx1024 — **OOM → 1,100 tok/s @ 4.0 GB** — was impossible.

**Turning point and readout.** Primary scoreboard from [08.5](../../learning-notes/08-experiments-and-results.md) (pre→post at bs8/512): `nanolab/README.md` also records a later fully-vectorized WY GDN path (~**1.6K** tok/s @ bs8/512, 6.7×) and mamba2 ctx1024 ~2.0K tok/s @ 6.0 GB with `mixer_chunk=32`. Both families are pinned by output **and** input-gradient checks, including non-divisible-T padding. **Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

**Failures and surprises.** **fp32 accumulation bug:** CPU-only tests missed autocast drift — scan must run in fp32 (lesson from 08.5). Pre-kernel gdn/mamba2 in `gpu_sweep_mixer.json` remain the cautionary 238/333 tok/s baselines.

## Decision and aftermath

**Kept:** A custom kernel is worthless until verified against a brute-force reference; accumulation needs fp32. The notebook continues with [20-run128m-long](20-run128m-long.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: **Mamba-2 SSD** — 333 → **3,224 tok/s (9.7×)** — trainable at ctx1024.
- The result artifact reports: **GDN** — 238 → **482 tok/s (2×)**, 4.7→2.6 GB — —.
- The result artifact reports: **GDN** bs16/ctx1024 — **OOM → 1,100 tok/s @ 4.0 GB** — was impossible.
- Failure/operational record: **fp32 accumulation bug:** CPU-only tests missed autocast drift — scan must run in fp32 (lesson from 08.5).
- Failure/operational record: Pre-kernel gdn/mamba2 in `gpu_sweep_mixer.json` remain the cautionary 238/333 tok/s baselines.

## What this does not prove

**Confidence: High.** Speedups are paired with output and input-gradient parity checks, fp32 accumulation checks, and non-divisible-length coverage. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.5
- [learning-notes/04-sequence-mixers.md](../../learning-notes/04-sequence-mixers.md) — §4.5
- Related suites: [`17-gpu-throughput-sweeps`](17-gpu-throughput-sweeps.md), [`12-mixer-ab-tinystories`](12-mixer-ab-tinystories.md)

---

[Previous](18-gpu-maximization.md) · [Index](../00-INDEX.md) · [Next](20-run128m-long.md)
