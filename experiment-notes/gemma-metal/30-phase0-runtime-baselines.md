# 30: Phase-0 runtime baselines (Ollama / mlx-lm)

## Executive summary

- **Question:** Pinned honest-lane runs (ctx=4096, max_gen=128, temp=0, think=off) would establish live decode floors for E4B and 31B on Ollama and mlx-lm, so custom-stack and DFlash claims can be judged against real host bars rather than vendor marketing numbers.
- **Result:** On this host the honest E4B floor is ~56–76 tok/s and the 31B floor is ~12.3 tok/s — ship gates must beat those measured floors, not abstract targets alone.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `30-phase0-runtime-baselines` |
| Dates | `2026-07-13` – `2026-07-13` |
| Hardware | Apple M5 Pro · 20 GPU · 64 GB unified · macOS 26.x |
| Status | `done` |

## Hypothesis

Pinned honest-lane runs (ctx=4096, max_gen=128, temp=0, think=off) would establish live decode floors for E4B and 31B on Ollama and mlx-lm, so custom-stack and DFlash claims can be judged against real host bars rather than vendor marketing numbers.

## Setup

- Trainer / preset: Phase-0 bench harness (`bench/bench.py` ollama / mlx); no training
- Fixed knobs: `num_ctx` / `max_kv_size=4096`, max generation **128**, temperature **0**, think **off**, batch **1**
- Env flags: honest lane only; LiteRT-LM / BaseRT absent on PATH

## Variants

| Variant | Change |
|---------|--------|
| Ollama `gemma4:31b-mlx` | nvfp4 via Ollama |
| Ollama `gemma4:e4b-it-q4_K_M` | Q4_K_M |
| mlx-lm `mlx-community/gemma-4-e4b-it-4bit` | 4-bit mlx community |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | mlx-lm E4B 4bit | **75.72** tok/s bench avg; generate **76.09 / 76.085** | Phase-0 best E4B; peak mem ~4.46 GB |
| 2 | Ollama E4B Q4_K_M decode_pad | **55.79** tok/s; TTFT **331** ms | Math prompt **71.06** (n=4) |
| 3 | Ollama 31B mlx decode_pad | **12.27** tok/s; TTFT **426** ms warm | Math **9.84** (n=3); historical floor ~10.8–13 |
| — | LiteRT / BaseRT / MTP | not run | Runtimes / draft tags missing |

Pinned gates (product doctrine): E4B Q4 **~48–60** (practical bar ≥ Phase-0 best ~76); E4B+MTP **~90–110**; 31B Q4 **≥15**; 31B+MTP **≥25**. Headline: E4B Ollama already sits in the 48–60 band; mlx-lm is the practical E4B ceiling. 31B Ollama ~12.3 makes ≥15 a real stretch for any backend that only matches Ollama.

**Interpretation boundary.** The tok/s and TTFT values are measured host-specific floors. Product gates are decisions, not observed performance, and missing runtimes cannot be ranked.

## Failures

- LiteRT-LM and BaseRT not installed — MTP calibration skipped
- Ollama / mlx-lm MTP knobs not available for these tags at Phase 0 (later superseded by DFlash on mlx 0.32)

## Lesson

**On this host the honest E4B floor is ~56–76 tok/s and the 31B floor is ~12.3 tok/s — ship gates must beat those measured floors, not abstract targets alone.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 bench/bench.py` from `Rust_MLKit/gemma-metal`.
- Required replay inputs: the backend/model/ctx/max-gen settings in `docs/gates.md` and the three listed result JSON files. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Live pinned runtime measurements exist, but prompt counts are small and missing runtimes prevent a complete baseline matrix.

## Artifacts

- `Rust_MLKit/gemma-metal/docs/gates.md` — locked targets + Phase-0 table
- `Rust_MLKit/gemma-metal/bench/results/run_20260713_100106_ollama.json` — 31B
- `Rust_MLKit/gemma-metal/bench/results/run_20260713_100745_ollama.json` — E4B Ollama
- `Rust_MLKit/gemma-metal/bench/results/run_20260713_100411_mlx.json` — E4B mlx-lm

## Why this experiment happened

The inference track needed honest host-specific floors before custom Metal optimization or speculative decoding could claim progress. Pinned Ollama and mlx-lm lanes established what E4B and 31B already delivered on the same M5 Pro. The preceding notebook context is [21-diffusion-lm](../nanolab/21-diffusion-lm.md).

## Experiment story

**Baseline.** The inference track needed honest host-specific floors before custom Metal optimization or speculative decoding could claim progress. Pinned Ollama and mlx-lm lanes established what E4B and 31B already delivered on the same M5 Pro. The preceding notebook context is [21-diffusion-lm](../nanolab/21-diffusion-lm.md).

**Hypothesis.** Pinned honest-lane runs (ctx=4096, max_gen=128, temp=0, think=off) would establish live decode floors for E4B and 31B on Ollama and mlx-lm, so custom-stack and DFlash claims can be judged against real host bars rather than vendor marketing numbers.

**Test contract.** Trainer / preset: Phase-0 bench harness (`bench/bench.py` ollama / mlx); no training Fixed knobs: `num_ctx` / `max_kv_size=4096`, max generation **128**, temperature **0**, think **off**, batch **1** Env flags: honest lane only; LiteRT-LM / BaseRT absent on PATH

**Variant sequence.** The preserved comparison matrix was: Ollama `gemma4:31b-mlx` — nvfp4 via Ollama; Ollama `gemma4:e4b-it-q4_K_M` — Q4_K_M; mlx-lm `mlx-community/gemma-4-e4b-it-4bit` — 4-bit mlx community.

**Measured turn.** The result board records 1 — mlx-lm E4B 4bit — **75.72** tok/s bench avg; generate **76.09 / 76.085** — Phase-0 best E4B; peak mem ~4.46 GB; 2 — Ollama E4B Q4_K_M decode_pad — **55.79** tok/s; TTFT **331** ms — Math prompt **71.06** (n=4); 3 — Ollama 31B mlx decode_pad — **12.27** tok/s; TTFT **426** ms warm — Math **9.84** (n=3); historical floor ~10.8–13; — — LiteRT / BaseRT / MTP — not run — Runtimes / draft tags missing.

**Turning point and readout.** Pinned gates (product doctrine): E4B Q4 **~48–60** (practical bar ≥ Phase-0 best ~76); E4B+MTP **~90–110**; 31B Q4 **≥15**; 31B+MTP **≥25**. Headline: E4B Ollama already sits in the 48–60 band; mlx-lm is the practical E4B ceiling. 31B Ollama ~12.3 makes ≥15 a real stretch for any backend that only matches Ollama. **Interpretation boundary.** The tok/s and TTFT values are measured host-specific floors. Product gates are decisions, not observed performance, and missing runtimes cannot be ranked.

**Failures and surprises.** LiteRT-LM and BaseRT not installed — MTP calibration skipped Ollama / mlx-lm MTP knobs not available for these tags at Phase 0 (later superseded by DFlash on mlx 0.32)

## Decision and aftermath

**Kept:** On this host the honest E4B floor is ~56–76 tok/s and the 31B floor is ~12.3 tok/s — ship gates must beat those measured floors, not abstract targets alone. The notebook continues with [31-native-decode-speed-ladder](31-native-decode-speed-ladder.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — mlx-lm E4B 4bit — **75.72** tok/s bench avg; generate **76.09 / 76.085** — Phase-0 best E4B; peak mem ~4.46 GB.
- The result artifact reports: 2 — Ollama E4B Q4_K_M decode_pad — **55.79** tok/s; TTFT **331** ms — Math prompt **71.06** (n=4).
- The result artifact reports: 3 — Ollama 31B mlx decode_pad — **12.27** tok/s; TTFT **426** ms warm — Math **9.84** (n=3); historical floor ~10.8–13.
- The result artifact reports: — — LiteRT / BaseRT / MTP — not run — Runtimes / draft tags missing.
- Failure/operational record: LiteRT-LM and BaseRT not installed — MTP calibration skipped
- Failure/operational record: Ollama / mlx-lm MTP knobs not available for these tags at Phase 0 (later superseded by DFlash on mlx 0.32)

## What this does not prove

**Confidence: Medium.** Live pinned runtime measurements exist, but prompt counts are small and missing runtimes prevent a complete baseline matrix. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — inference / systems cross-links
- Related suites: [`31-native-decode-speed-ladder`](31-native-decode-speed-ladder.md), [`34-mlx-dflash-product`](34-mlx-dflash-product.md)

---

Previous · [Index](../00-INDEX.md) · [Next](31-native-decode-speed-ladder.md)
