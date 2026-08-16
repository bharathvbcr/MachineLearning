# 31: Native decode speed ladder (Host-KV → Hot qmv)

## Executive summary

- **Question:** Moving from host KV densify + per-dispatch sync through GPU-resident Hot Q4, fused MLP/KV, and MLX-style qmv (bfloat2 sb + qdot) would climb gemma-metal E4B toward the ≥48–60 gate and 31B toward ≥15.
- **Result:** Native E4B peaked ~25 tok/s; further GEMV peel does not clear gates — the remaining gap is dispatch/overhead and speculative amortization, not another qdot tweak.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `31-native-decode-speed-ladder` |
| Dates | `2026-07-13` – `2026-07-14` |
| Hardware | Apple M5 Pro · 20 GPU · 64 GB |
| Status | `done` |

## Hypothesis

Moving from host KV densify + per-dispatch sync through GPU-resident Hot Q4, fused MLP/KV, and MLX-style qmv (bfloat2 sb + qdot) would climb gemma-metal E4B toward the ≥48–60 gate and 31B toward ≥15.

## Setup

- Trainer / preset: `cargo run --release --bin bench -- --e4b` / 31B Hot from `mlx-community/gemma-4-31b-it-4bit`
- Fixed knobs: TRACE/INFER off for quiet product; hazard barriers default ON; `GEMMA_METAL_FUSE_MLP` default ON; temp=0 greedy
- Env flags: various A/B (`GEMMA_METAL_FUSE_KV`, Interleaved4 off by default, etc.)

## Variants

| Variant | Change |
|---------|--------|
| Pre speed | Host KV densify + per-dispatch sync |
| Post speed → Hot/GPU embed | Packed async encode, tiled/vectorized GEMV |
| Fuse / simd / peels | fuse K∥V, simdgroup Q4, bfloat2+qdot, Interleaved4 A/B |

## Results

| Rank | Run / config | Decode tok/s | TTFT (T=4) / notes |
|------|--------------|--------------|--------------------|
| Peak quiet | bf16 scales + packs=2, SIMD_ROWS=4 | **25.10** | 133 ms |
| Near peak | fuse K∥V + coarse barriers | **25.05** | 132 ms |
| Near peak | bf16 activations on simd GEMV | **24.98** | 133 ms |
| Mid | MLX qdot peel chase | **24.45** (~24.5–25.4) | 136 ms |
| Latest quiet (bfloat2 + qmv) | Hot sb interleaved bfloat2 | **23.86** | 142 ms |
| gelu fix era | `precise::tanh` + fuse MLP | **23.61** | 141 ms |
| Earlier | uint peel + hazard + PLE Hot | **19.38** | 179 ms |
| Earlier | TG x-cache / vectorized | **15.17–15.91** | 230–243 ms |
| Post speed pass | GPU-resident KV | **13.90** | 265 ms |
| Pre speed | Host KV densify | **4.78** | 790 ms |
| 31B Hot quiet | bfloat2 + qmv | **6.83** | 548.8 ms; Hot 17.87 GiB |

Climb from **4.78 → ~25** tok/s on E4B (~5×) closed most of the early host-sync tax; last ALU peels (bfloat2/qdot/Interleaved4) were flat or regressive. Honest verdict: ~**23.9** this pass vs gate lower band **48** (~2.0×) and Phase-0 mlx **~76** (~3.2×). 31B custom **6.83** vs Ollama **12.27** and gate **≥15** — unmet.

**Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

## Failures

- Interleaved4 weight pack A/B ~**22.8** tok/s → default **OFF**
- Interim ≥30 (E4B) / ≥10 (31B) not cleared
- Prior `air.fast_tanh` NaNs on gelu → fixed with `precise::tanh`

## Lesson

**Native E4B peaked ~25 tok/s; further GEMV peel does not clear gates — the remaining gap is dispatch/overhead and speculative amortization, not another qdot tweak.**

## Reproduction

- Replay: From `Rust_MLKit/gemma-metal`: `GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0 cargo run --release --bin bench -- --e4b`; for native 31B use `cargo run --release --bin bench` with the cached model documented in `docs/gates.md`.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: Medium.** Many direct A/B measurements establish the speed climb, but most rows are session snapshots without repeated confidence intervals.

## Artifacts

- `Rust_MLKit/gemma-metal/docs/gates.md` — Phase 4 ladder table
- `Rust_MLKit/gemma-metal/bench/results/latest_31b.json` — 6.83 tok/s
- Quiet E4B summaries cited in gates (`bench_e4b_bfloat2_final.txt` era / Phase-4 rows)

## Why this experiment happened

With mlx-lm near 76 tok/s on E4B and Ollama near 12.3 tok/s on 31B, the native port had concrete gaps to close. The ladder attacked obvious host synchronization, KV residency, packing, fusion, and Q4 GEMV costs in sequence. The preceding notebook context is [30-phase0-runtime-baselines](30-phase0-runtime-baselines.md).

## Experiment story

**Baseline.** With mlx-lm near 76 tok/s on E4B and Ollama near 12.3 tok/s on 31B, the native port had concrete gaps to close. The ladder attacked obvious host synchronization, KV residency, packing, fusion, and Q4 GEMV costs in sequence. The preceding notebook context is [30-phase0-runtime-baselines](30-phase0-runtime-baselines.md).

**Hypothesis.** Moving from host KV densify + per-dispatch sync through GPU-resident Hot Q4, fused MLP/KV, and MLX-style qmv (bfloat2 sb + qdot) would climb gemma-metal E4B toward the ≥48–60 gate and 31B toward ≥15.

**Test contract.** Trainer / preset: `cargo run --release --bin bench -- --e4b` / 31B Hot from `mlx-community/gemma-4-31b-it-4bit` Fixed knobs: TRACE/INFER off for quiet product; hazard barriers default ON; `GEMMA_METAL_FUSE_MLP` default ON; temp=0 greedy Env flags: various A/B (`GEMMA_METAL_FUSE_KV`, Interleaved4 off by default, etc.)

**Variant sequence.** The preserved comparison matrix was: Pre speed — Host KV densify + per-dispatch sync; Post speed → Hot/GPU embed — Packed async encode, tiled/vectorized GEMV; Fuse / simd / peels — fuse K∥V, simdgroup Q4, bfloat2+qdot, Interleaved4 A/B.

**Measured turn.** The result board records Peak quiet — bf16 scales + packs=2, SIMD_ROWS=4 — **25.10** — 133 ms; Near peak — fuse K∥V + coarse barriers — **25.05** — 132 ms; Near peak — bf16 activations on simd GEMV — **24.98** — 133 ms; Mid — MLX qdot peel chase — **24.45** (~24.5–25.4) — 136 ms; Latest quiet (bfloat2 + qmv) — Hot sb interleaved bfloat2 — **23.86** — 142 ms.

**Turning point and readout.** Climb from **4.78 → ~25** tok/s on E4B (~5×) closed most of the early host-sync tax; last ALU peels (bfloat2/qdot/Interleaved4) were flat or regressive. Honest verdict: ~**23.9** this pass vs gate lower band **48** (~2.0×) and Phase-0 mlx **~76** (~3.2×). 31B custom **6.83** vs Ollama **12.27** and gate **≥15** — unmet. **Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

**Failures and surprises.** Interleaved4 weight pack A/B ~**22.8** tok/s → default **OFF** Interim ≥30 (E4B) / ≥10 (31B) not cleared Prior `air.fast_tanh` NaNs on gelu → fixed with `precise::tanh`

## Decision and aftermath

**Kept:** Native E4B peaked ~25 tok/s; further GEMV peel does not clear gates — the remaining gap is dispatch/overhead and speculative amortization, not another qdot tweak. The notebook continues with [32-native-fusion-2026-07-14](32-native-fusion-2026-07-14.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: Peak quiet — bf16 scales + packs=2, SIMD_ROWS=4 — **25.10** — 133 ms.
- The result artifact reports: Near peak — fuse K∥V + coarse barriers — **25.05** — 132 ms.
- The result artifact reports: Near peak — bf16 activations on simd GEMV — **24.98** — 133 ms.
- The result artifact reports: Mid — MLX qdot peel chase — **24.45** (~24.5–25.4) — 136 ms.
- Failure/operational record: Interleaved4 weight pack A/B ~**22.8** tok/s → default **OFF**
- Failure/operational record: Interim ≥30 (E4B) / ≥10 (31B) not cleared

## What this does not prove

**Confidence: Medium.** Many direct A/B measurements establish the speed climb, but most rows are session snapshots without repeated confidence intervals. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md)
- Related suites: [`30-phase0-runtime-baselines`](30-phase0-runtime-baselines.md), [`32-native-fusion-2026-07-14`](32-native-fusion-2026-07-14.md), [`33-kernel-roofline-overhead`](33-kernel-roofline-overhead.md)

---

[Previous](30-phase0-runtime-baselines.md) · [Index](../00-INDEX.md) · [Next](32-native-fusion-2026-07-14.md)
