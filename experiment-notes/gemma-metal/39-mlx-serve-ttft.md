# 39: MLX serve TTFT (prompt-cache + SSE)

## Executive summary

- **Question:** Sticky prompt-cache across multi-turn chat plus SSE writer overlap would cut server TTFT on follow-up turns while holding DFlash decode ~mid-30s tok/s.
- **Result:** Prompt-cache is sticky and decode is solid (~36 tok/s), but short-turn TTFT will not drop until follow-up prefill is smaller than the cached prefix tax.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `39-mlx-serve-ttft` |
| Dates | `2026-07-14` – `2026-07-14` |
| Hardware | Apple M5 Pro · `serve_dflash.py` MLX track |
| Status | `done` |

## Hypothesis

Sticky prompt-cache across multi-turn chat plus SSE writer overlap would cut server TTFT on follow-up turns while holding DFlash decode ~mid-30s tok/s.

## Setup

- Trainer / preset: MLX `bench/serve_dflash.py`; 3-turn client harness writing `ttft_3turn.json`
- Fixed knobs: DFlash product path (mlx 0.32 + draft); short chat turns
- Env flags: SESSION MLX track M1 prompt-cache, M2 SSE overlap **landed**

## Variants

| Variant | Change |
|---------|--------|
| M1 | Sticky prompt-cache (`cached_tokens` grows) |
| M2 | SSE writer thread overlap |
| 3-turn measure | Short prompts; 24 completion tokens each |

## Results

| Rank | Turn | Server TTFT | cached | prefill | decode tok/s |
|------|------|-------------|--------|---------|--------------|
| 1 | “Say hello…” | **370.1** ms | 0 | 23 | **36.2** |
| 2 | “What is 2+2?” | **365.7** ms | 23 | 22 | **35.9** |
| 3 | “Name one primary color.” | **363.3** ms | 45 | 15 | **36.9** |

Cache grows **0 → 23 → 45** as designed, but TTFT stays flat (~370→363 ms) because each short turn still prefills **~15–23** new tokens — cache hit does not dominate TTFT on these prompts. Decode holds ~**36** tok/s (matches `serve_dflash_ttft.log`). Client TTFT within ~30 ms of server on turn 1; nearly matched thereafter.

**Interpretation boundary.** Decode tok/s remains healthy while TTFT stays flat; these are distinct phases. The cached-prefix explanation is consistent with counters but needs longer-turn A/Bs.

## Failures

- Multi-turn TTFT win not realized on short-chat workload (prefill still large vs cached prompt)
- None blocking for decode path

## Lesson

**Prompt-cache is sticky and decode is solid (~36 tok/s), but short-turn TTFT will not drop until follow-up prefill is smaller than the cached prefix tax.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 bench/serve_dflash.py`.
- Required replay inputs: `ttft_3turn.json` for the three prompts and `SESSION_RESULTS.md` for M1/M2 server settings. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Three measured turns show stable decode and cache growth, but one short conversation is insufficient to generalize TTFT behavior.

## Artifacts

- `Rust_MLKit/gemma-metal/bench/results/baselines_2026-07-14/SESSION_RESULTS.md` — MLX track
- `Rust_MLKit/gemma-metal/bench/results/baselines_2026-07-14/ttft_3turn.json`
- `Rust_MLKit/gemma-metal/bench/results/baselines_2026-07-14/serve_dflash_ttft.log`

## Why this experiment happened

Suite 34 established fast exact decode in a benchmark, but product experience also depends on time to first token. The server suite added sticky prompt caching and SSE overlap, then measured three turns rather than assuming decode tok/s predicted interactivity. The preceding notebook context is [38-clustered-mtp-e4b](38-clustered-mtp-e4b.md).

## Experiment story

**Baseline.** Suite 34 established fast exact decode in a benchmark, but product experience also depends on time to first token. The server suite added sticky prompt caching and SSE overlap, then measured three turns rather than assuming decode tok/s predicted interactivity. The preceding notebook context is [38-clustered-mtp-e4b](38-clustered-mtp-e4b.md).

**Hypothesis.** Sticky prompt-cache across multi-turn chat plus SSE writer overlap would cut server TTFT on follow-up turns while holding DFlash decode ~mid-30s tok/s.

**Test contract.** Trainer / preset: MLX `bench/serve_dflash.py`; 3-turn client harness writing `ttft_3turn.json` Fixed knobs: DFlash product path (mlx 0.32 + draft); short chat turns Env flags: SESSION MLX track M1 prompt-cache, M2 SSE overlap **landed**

**Variant sequence.** The preserved comparison matrix was: M1 — Sticky prompt-cache (`cached_tokens` grows); M2 — SSE writer thread overlap; 3-turn measure — Short prompts; 24 completion tokens each.

**Measured turn.** The result board records 1 — “Say hello…” — **370.1** ms — 0 — 23 — **36.2**; 2 — “What is 2+2?” — **365.7** ms — 23 — 22 — **35.9**; 3 — “Name one primary color.” — **363.3** ms — 45 — 15 — **36.9**.

**Turning point and readout.** Cache grows **0 → 23 → 45** as designed, but TTFT stays flat (~370→363 ms) because each short turn still prefills **~15–23** new tokens — cache hit does not dominate TTFT on these prompts. Decode holds ~**36** tok/s (matches `serve_dflash_ttft.log`). Client TTFT within ~30 ms of server on turn 1; nearly matched thereafter. **Interpretation boundary.** Decode tok/s remains healthy while TTFT stays flat; these are distinct phases. The cached-prefix explanation is consistent with counters but needs longer-turn A/Bs.

**Failures and surprises.** Multi-turn TTFT win not realized on short-chat workload (prefill still large vs cached prompt) None blocking for decode path

## Decision and aftermath

**Kept:** Prompt-cache is sticky and decode is solid (~36 tok/s), but short-turn TTFT will not drop until follow-up prefill is smaller than the cached prefix tax. The notebook continues with [40-ddtree-frontier](40-ddtree-frontier.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — “Say hello…” — **370.1** ms — 0 — 23 — **36.2**.
- The result artifact reports: 2 — “What is 2+2?” — **365.7** ms — 23 — 22 — **35.9**.
- The result artifact reports: 3 — “Name one primary color.” — **363.3** ms — 45 — 15 — **36.9**.
- Failure/operational record: Multi-turn TTFT win not realized on short-chat workload (prefill still large vs cached prompt)
- Failure/operational record: None blocking for decode path

## What this does not prove

**Confidence: Medium.** Three measured turns show stable decode and cache growth, but one short conversation is insufficient to generalize TTFT behavior. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- Related suites: [`34-mlx-dflash-product`](34-mlx-dflash-product.md), [`35-mlx-dflash-block-tuning`](35-mlx-dflash-block-tuning.md)

---

[Previous](38-clustered-mtp-e4b.md) · [Index](../00-INDEX.md) · [Next](40-ddtree-frontier.md)
