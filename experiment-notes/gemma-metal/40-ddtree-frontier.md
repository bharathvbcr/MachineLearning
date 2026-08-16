# 40: DDTree frontier (modeled defer)

## Executive summary

- **Question:** Draft trees (DDTree / TAPS) over DFlash per-position marginals would raise acceptance (especially prose) and lift median ~31 → ~38–42 tok/s without extra draft cost.
- **Result:** DDTree is deferred on 31B MLX (modeled ≤~1.02×) — revive only after cheaper native M>1 verify or on E4B.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Low**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `40-ddtree-frontier` |
| Dates | `2026-07-13` – `2026-07-13` |
| Hardware | Apple M5 Pro · modeled from mlx-0.32 verify microbench |
| Status | `done` |

## Hypothesis

Draft trees (DDTree / TAPS) over DFlash per-position marginals would raise acceptance (especially prose) and lift median ~31 → ~38–42 tok/s without extra draft cost.

## Setup

- Trainer / preset: algorithmic core `bench/ddtree_core.py` + `bench/test_ddtree_core.py` (**19/19** CPU checks); throughput model from measured mlx-0.32 verify-cost curve
- Fixed knobs: paper accept saturation +35%; verify costs for mlp `[10752×5376]` q4g64: M=1→274µs, M=6→403, M=8→452, **M=10→588 (cliff)**, M=16→594
- Env flags: n/a (no full MLX tree-attn ship)

## Variants

| Variant | Change |
|---------|--------|
| Linear DFlash | M=6 reference accept **3.56** |
| Tree budget N=8 / 16 / 24 | Best-first nodes; one tree-attn verify |
| Unlock conditions | Native true M>1 Q4 GEMM or smaller E4B target |

## Results

| Rank | Tree budget N | Modeled accept | Rel. throughput vs linear(M=6) |
|------|---------------|----------------|--------------------------------|
| — | linear M=6 | 3.56 | **1.00** |
| Best tie | 8 | 3.84 | **0.98–1.01** |
| — | 24 | 4.67 | **0.94–1.02** |
| — | 16 | 4.45 | **0.90–0.98** |

**Break-even on 31B** — M=10 verify cliff + saturating accept means extra tree tokens cost ~what they earn. Paper +30–40% accept wins were on **small** Qwen3 models where verify is cheap vs draft. MLX path levers exhausted: DFlash ✓, q4 draft ✓, mlx 0.32 ✓, block=5 ✓, adaptive ✗, wired ✗, trees ✗@31B. Core kept for native engine once verify(M) flattens.

**Interpretation boundary.** Core correctness checks are measured; the ≤1.02× frontier is modeled, not an end-to-end tree-attention result.

## Failures

- No MLX tree-attention production path yet (custom mask + depth RoPE hard)
- Adaptive block already ruled out separately (<4%)

## Lesson

**DDTree is deferred on 31B MLX (modeled ≤~1.02×) — revive only after cheaper native M>1 verify or on E4B.**

## Reproduction

- Replay: From `Rust_MLKit/gemma-metal`: `python3 bench/test_ddtree_core.py` reproduces the 19/19 CPU core checks; the throughput conclusion must be recomputed from the measured verify-cost inputs in `docs/speed_research_frontier.md`.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: Low.** CPU core tests pass, but throughput is modeled from a verify-cost curve; no full tree-attention production run exists.

## Artifacts

- `Rust_MLKit/gemma-metal/docs/speed_research_frontier.md`
- `Rust_MLKit/gemma-metal/bench/ddtree_core.py`
- `Rust_MLKit/gemma-metal/bench/test_ddtree_core.py`
- `Rust_MLKit/gemma-metal/bench/results/adaptive_block_finding.json` (companion “not a lever”)

## Why this experiment happened

Fixed block tuning left less than about 4% workload-specific headroom, especially on prose. Draft trees were the remaining algorithmic proposal for increasing accepted work per target verification, so the project modeled their break-even point before building a production tree-attention path. The preceding notebook context is [39-mlx-serve-ttft](39-mlx-serve-ttft.md).

## Experiment story

**Baseline.** Fixed block tuning left less than about 4% workload-specific headroom, especially on prose. Draft trees were the remaining algorithmic proposal for increasing accepted work per target verification, so the project modeled their break-even point before building a production tree-attention path. The preceding notebook context is [39-mlx-serve-ttft](39-mlx-serve-ttft.md).

**Hypothesis.** Draft trees (DDTree / TAPS) over DFlash per-position marginals would raise acceptance (especially prose) and lift median ~31 → ~38–42 tok/s without extra draft cost.

**Test contract.** Trainer / preset: algorithmic core `bench/ddtree_core.py` + `bench/test_ddtree_core.py` (**19/19** CPU checks); throughput model from measured mlx-0.32 verify-cost curve Fixed knobs: paper accept saturation +35%; verify costs for mlp `[10752×5376]` q4g64: M=1→274µs, M=6→403, M=8→452, **M=10→588 (cliff)**, M=16→594 Env flags: n/a (no full MLX tree-attn ship)

**Variant sequence.** The preserved comparison matrix was: Linear DFlash — M=6 reference accept **3.56**; Tree budget N=8 / 16 / 24 — Best-first nodes; one tree-attn verify; Unlock conditions — Native true M>1 Q4 GEMM or smaller E4B target.

**Measured turn.** The result board records — — linear M=6 — 3.56 — **1.00**; Best tie — 8 — 3.84 — **0.98–1.01**; — — 24 — 4.67 — **0.94–1.02**; — — 16 — 4.45 — **0.90–0.98**.

**Turning point and readout.** **Break-even on 31B** — M=10 verify cliff + saturating accept means extra tree tokens cost ~what they earn. Paper +30–40% accept wins were on **small** Qwen3 models where verify is cheap vs draft. MLX path levers exhausted: DFlash ✓, q4 draft ✓, mlx 0.32 ✓, block=5 ✓, adaptive ✗, wired ✗, trees ✗@31B. Core kept for native engine once verify(M) flattens. **Interpretation boundary.** Core correctness checks are measured; the ≤1.02× frontier is modeled, not an end-to-end tree-attention result.

**Failures and surprises.** No MLX tree-attention production path yet (custom mask + depth RoPE hard) Adaptive block already ruled out separately (<4%)

## Decision and aftermath

**Kept:** DDTree is deferred on 31B MLX (modeled ≤~1.02×) — revive only after cheaper native M>1 verify or on E4B. The notebook continues with [41-audit-deep-2026-07-14](41-audit-deep-2026-07-14.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: — — linear M=6 — 3.56 — **1.00**.
- The result artifact reports: Best tie — 8 — 3.84 — **0.98–1.01**.
- The result artifact reports: — — 24 — 4.67 — **0.94–1.02**.
- The result artifact reports: — — 16 — 4.45 — **0.90–0.98**.
- Failure/operational record: No MLX tree-attention production path yet (custom mask + depth RoPE hard)
- Failure/operational record: Adaptive block already ruled out separately (<4%)

## What this does not prove

**Confidence: Low.** CPU core tests pass, but throughput is modeled from a verify-cost curve; no full tree-attention production run exists. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- Related suites: [`34-mlx-dflash-product`](34-mlx-dflash-product.md), [`35-mlx-dflash-block-tuning`](35-mlx-dflash-block-tuning.md), [`33-kernel-roofline-overhead`](33-kernel-roofline-overhead.md)

---

[Previous](39-mlx-serve-ttft.md) · [Index](../00-INDEX.md) · [Next](41-audit-deep-2026-07-14.md)
