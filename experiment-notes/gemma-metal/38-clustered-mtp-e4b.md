# 38: Clustered MTP E4B (legacy Phase 5)

## Executive summary

- **Question:** Clustered-assistant MTP (cross-KV into shared sliding/global + adaptive draft/verify) would multiply E4B tok/s toward the ~90–110 MTP gate once accept stayed high.
- **Result:** 75% accept with ~10–12 e2e tok/s proves wiring, not a throughput win — clustered per-token verify cannot beat backbone until batched GPU verify lands.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `38-clustered-mtp-e4b` |
| Dates | `2026-07-13` – `2026-07-14` |
| Hardware | Apple M5 Pro · E4B Hot + `google/gemma-4-E4B-it-assistant` |
| Status | `done` |

## Hypothesis

Clustered-assistant MTP (cross-KV into shared sliding/global + adaptive draft/verify) would multiply E4B tok/s toward the ~90–110 MTP gate once accept stayed high.

## Setup

- Trainer / preset: `generate_mtp_smoke` / `gpu_mtp_real_assistant_accept`; assistant ~160 MB (centroids/embeds/pre/post + 4 Q-consumer layers)
- Fixed knobs: real shared-KV densify via `sync_mtp_cross_kv`; early-reject on first mismatch
- Env flags: n/a beyond quiet GEMMA_METAL_LOG=0

## Variants

| Variant | Change |
|---------|--------|
| Synthetic smoke | Wired draft→verify loop |
| Real assistant weights | Loaded HF assistant |
| Early-reject e2e | Stop after first mismatch |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | Real asst e2e (early-reject) | **12.13** tok/s; accept **6/8 (75%)**; wall **659** ms | `latest_mtp.json` / `mtp_e2e_accept2.txt` |
| 2 | Earlier real asst measure | **10.01** tok/s; accept **6/8 (75%)**; wall **799** ms | `mtp_e2e_accept.txt` |
| — | Self-accept smoke | draft_len=4; self_accept **4/4** | `mtp_real_accept.txt` |

Accept rate is fine; e2e stays **below** quiet greedy (~23–25) because verify is full per-draft `step()` (no batched tree verify) plus host KV bridge + host draft. Gate **~90–110** needs base ≥48 **and** GPU draft / parallel verify. Superseded as the 31B product strategy by MLX DFlash block-verify (`34`).

**Interpretation boundary.** Acceptance is measured but does not guarantee speedup: the measured e2e path is slower than greedy because verify work is not batched.

## Failures

- Product MTP tok/s unmet
- Schedule not speculative throughput (work ≥ greedy for accepted prefix)

## Lesson

**75% accept with ~10–12 e2e tok/s proves wiring, not a throughput win — clustered per-token verify cannot beat backbone until batched GPU verify lands.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: the `generate_mtp_smoke` / `gpu_mtp_real_assistant_accept` binaries.
- Required replay inputs: `latest_mtp.json` and the two `mtp_e2e_accept*.txt` logs for assistant, draft length, and early-reject settings. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Real-assistant acceptance and two e2e timings were measured, but the per-token verify design is not equivalent to the intended batched product path.

## Artifacts

- `Rust_MLKit/gemma-metal/bench/results/latest_mtp.json`
- `Rust_MLKit/gemma-metal/bench/results/mtp_e2e_accept.txt`
- `Rust_MLKit/gemma-metal/bench/results/mtp_e2e_accept2.txt`
- `Rust_MLKit/gemma-metal/docs/gates.md` — Phase 5
- `Rust_MLKit/gemma-metal/docs/bottleneck.md` — Why MTP is slower

## Why this experiment happened

In parallel with 31B DFlash, an earlier E4B path explored clustered-assistant MTP. Its purpose was to learn whether high draft acceptance alone could multiply throughput before a fully batched GPU verifier existed. The preceding notebook context is [37-golden-token-parity](37-golden-token-parity.md).

## Experiment story

**Baseline.** In parallel with 31B DFlash, an earlier E4B path explored clustered-assistant MTP. Its purpose was to learn whether high draft acceptance alone could multiply throughput before a fully batched GPU verifier existed. The preceding notebook context is [37-golden-token-parity](37-golden-token-parity.md).

**Hypothesis.** Clustered-assistant MTP (cross-KV into shared sliding/global + adaptive draft/verify) would multiply E4B tok/s toward the ~90–110 MTP gate once accept stayed high.

**Test contract.** Trainer / preset: `generate_mtp_smoke` / `gpu_mtp_real_assistant_accept`; assistant ~160 MB (centroids/embeds/pre/post + 4 Q-consumer layers) Fixed knobs: real shared-KV densify via `sync_mtp_cross_kv`; early-reject on first mismatch Env flags: n/a beyond quiet GEMMA_METAL_LOG=0

**Variant sequence.** The preserved comparison matrix was: Synthetic smoke — Wired draft→verify loop; Real assistant weights — Loaded HF assistant; Early-reject e2e — Stop after first mismatch.

**Measured turn.** The result board records 1 — Real asst e2e (early-reject) — **12.13** tok/s; accept **6/8 (75%)**; wall **659** ms — `latest_mtp.json` / `mtp_e2e_accept2.txt`; 2 — Earlier real asst measure — **10.01** tok/s; accept **6/8 (75%)**; wall **799** ms — `mtp_e2e_accept.txt`; — — Self-accept smoke — draft_len=4; self_accept **4/4** — `mtp_real_accept.txt`.

**Turning point and readout.** Accept rate is fine; e2e stays **below** quiet greedy (~23–25) because verify is full per-draft `step()` (no batched tree verify) plus host KV bridge + host draft. Gate **~90–110** needs base ≥48 **and** GPU draft / parallel verify. Superseded as the 31B product strategy by MLX DFlash block-verify (`34`). **Interpretation boundary.** Acceptance is measured but does not guarantee speedup: the measured e2e path is slower than greedy because verify work is not batched.

**Failures and surprises.** Product MTP tok/s unmet Schedule not speculative throughput (work ≥ greedy for accepted prefix)

## Decision and aftermath

**Kept:** 75% accept with ~10–12 e2e tok/s proves wiring, not a throughput win — clustered per-token verify cannot beat backbone until batched GPU verify lands. The notebook continues with [39-mlx-serve-ttft](39-mlx-serve-ttft.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — Real asst e2e (early-reject) — **12.13** tok/s; accept **6/8 (75%)**; wall **659** ms — `latest_mtp.json` / `mtp_e2e_accept2.txt`.
- The result artifact reports: 2 — Earlier real asst measure — **10.01** tok/s; accept **6/8 (75%)**; wall **799** ms — `mtp_e2e_accept.txt`.
- The result artifact reports: — — Self-accept smoke — draft_len=4; self_accept **4/4** — `mtp_real_accept.txt`.
- Failure/operational record: Product MTP tok/s unmet
- Failure/operational record: Schedule not speculative throughput (work ≥ greedy for accepted prefix)

## What this does not prove

**Confidence: Medium.** Real-assistant acceptance and two e2e timings were measured, but the per-token verify design is not equivalent to the intended batched product path. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- Related suites: [`34-mlx-dflash-product`](34-mlx-dflash-product.md), [`36-native-dflash-parity-accept`](36-native-dflash-parity-accept.md)

---

[Previous](37-golden-token-parity.md) · [Index](../00-INDEX.md) · [Next](39-mlx-serve-ttft.md)
