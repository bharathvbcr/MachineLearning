# 17: GPU throughput sweeps — opt / mixer / FFN

## Executive summary

- **Question:** Microbenchmarks at fixed shape should expose which optimizer/mixer/FFN choices burn tok/s or VRAM — independent of final val quality (suite 16).
- **Result:** Pure sequential SSM kernels are 24–33× too slow — throughput must be fixed before quality ablations are fair.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `17-gpu-throughput-sweeps` |
| Dates | 2026-06-15 – 2026-06-15 |
| Hardware | RTX 3070 Ti Laptop 8 GB (124M probe) |
| Status | `done` |

## Hypothesis

Microbenchmarks at fixed shape should expose which optimizer/mixer/FFN choices burn tok/s or VRAM — independent of final val quality (suite 16).

## Setup

- Trainer / preset: `nanolab/probe_perf.py` → `gpu_sweep_{opt,mixer,ffn}.json`
- Fixed knobs (from [08.4](../../learning-notes/08-experiments-and-results.md) + JSON): ~124M params; **opt/FFN** at bs16/ctx1024; **mixer** at bs8/ctx512 (so slow sequential SSMs can finish)
- Env flags: CUDA bf16 path; Peak FLOPs assumed for MFU (~40 TFLOP/s laptop peak)

## Variants

| Variant | Change |
|---------|--------|
| `gpu_sweep_opt.json` | optimizer ∈ {adamw, muon, sgd, lion, sophia, schedulefree, prodigy} |
| `gpu_sweep_mixer.json` | mixer ∈ {mla, attention, mingru, mamba2, gdn} |
| `gpu_sweep_ffn.json` | ffn ∈ {swiglu, gelu, relu2, moe} |

## Results

Matches §8.4 (rounded); raw JSON in artifacts:

| Axis | Ranked tok/s | Notes |
|------|--------------|-------|
| Optimizer | adamw **11.4K** ≥ sgd **10.8K** ≈ muon **10.8K** > lion 10.2K > sophia 10.1K > schedulefree/prodigy ~9.8K | Muon opt_ms **117** vs AdamW **34** (Newton–Schulz tax) |
| FFN | swiglu **10.3K** > gelu **9.8K** > relu2 **9.2K** | **moe OOM** at bs16 |
| Mixer | mla **9.3K** > attention **7.9K** > mingru **6.7K** ≫ mamba2 **333** ≫ gdn **238** | Sequential SSM refs — 24–33× too slow |

Readout: AdamW is cheapest per step; Muon’s step tax is real but suite 16/18 still justify it on convergence/systems grounds once batched. Mixer sweep is the emergency: mamba2/gdn at 238–333 tok/s are unusable until chunk kernels (suite 19).

**Interpretation boundary.** Tok/s and optimizer milliseconds are measured execution efficiency. They do not rank final quality; suite 16 answers that separate question.

## Failures

- `ffn.moe`: `"ok": false, "err": "OOM"` at bs16 (8 dense experts + optimizer state).
- Sequential mamba2/gdn not OOMs but practically blocked (bwd alone ~9–12.5 s/step).

## Lesson

**Pure sequential SSM kernels are 24–33× too slow — throughput must be fixed before quality ablations are fair.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.sweep_gpu` / `python3 -m nanolab.probe_perf`.
- Required replay inputs: the per-axis shapes and raw `nanolab/out/gpu_sweep_*.json` artifacts. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Direct fixed-shape measurements support the throughput ranking; opt/FFN and mixer shapes differ, and throughput says nothing by itself about quality.

## Artifacts

- `nanolab/out/gpu_sweep_opt.json`
- `nanolab/out/gpu_sweep_mixer.json`
- `nanolab/out/gpu_sweep_ffn.json`

## Why this experiment happened

Suite 16 ranked quality but left a practical question unanswered: some choices might be too slow or memory-heavy to compare fairly in wall time. Fixed-shape optimizer, mixer, and FFN probes exposed those systems costs directly. The preceding notebook context is [16-optimizer-quality-bakeoff](16-optimizer-quality-bakeoff.md).

## Experiment story

**Baseline.** Suite 16 ranked quality but left a practical question unanswered: some choices might be too slow or memory-heavy to compare fairly in wall time. Fixed-shape optimizer, mixer, and FFN probes exposed those systems costs directly. The preceding notebook context is [16-optimizer-quality-bakeoff](16-optimizer-quality-bakeoff.md).

**Hypothesis.** Microbenchmarks at fixed shape should expose which optimizer/mixer/FFN choices burn tok/s or VRAM — independent of final val quality (suite 16).

**Test contract.** Trainer / preset: `nanolab/probe_perf.py` → `gpu_sweep_{opt,mixer,ffn}.json` Fixed knobs (from [08.4](../../learning-notes/08-experiments-and-results.md) + JSON): ~124M params; **opt/FFN** at bs16/ctx1024; **mixer** at bs8/ctx512 (so slow sequential SSMs can finish) Env flags: CUDA bf16 path; Peak FLOPs assumed for MFU (~40 TFLOP/s laptop peak)

**Variant sequence.** The preserved comparison matrix was: `gpu_sweep_opt.json` — optimizer ∈ {adamw, muon, sgd, lion, sophia, schedulefree, prodigy}; `gpu_sweep_mixer.json` — mixer ∈ {mla, attention, mingru, mamba2, gdn}; `gpu_sweep_ffn.json` — ffn ∈ {swiglu, gelu, relu2, moe}.

**Measured turn.** The result board records Optimizer — adamw **11.4K** ≥ sgd **10.8K** ≈ muon **10.8K** > lion 10.2K > sophia 10.1K > schedulefree/prodigy ~9.8K — Muon opt_ms **117** vs AdamW **34** (Newton–Schulz tax); FFN — swiglu **10.3K** > gelu **9.8K** > relu2 **9.2K** — **moe OOM** at bs16; Mixer — mla **9.3K** > attention **7.9K** > mingru **6.7K** ≫ mamba2 **333** ≫ gdn **238** — Sequential SSM refs — 24–33× too slow.

**Turning point and readout.** Matches §8.4 (rounded); raw JSON in artifacts: Readout: AdamW is cheapest per step; Muon’s step tax is real but suite 16/18 still justify it on convergence/systems grounds once batched. Mixer sweep is the emergency: mamba2/gdn at 238–333 tok/s are unusable until chunk kernels (suite 19). **Interpretation boundary.** Tok/s and optimizer milliseconds are measured execution efficiency. They do not rank final quality; suite 16 answers that separate question.

**Failures and surprises.** `ffn.moe`: `"ok": false, "err": "OOM"` at bs16 (8 dense experts + optimizer state). Sequential mamba2/gdn not OOMs but practically blocked (bwd alone ~9–12.5 s/step).

## Decision and aftermath

**Kept:** Pure sequential SSM kernels are 24–33× too slow — throughput must be fixed before quality ablations are fair. The notebook continues with [18-gpu-maximization](18-gpu-maximization.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: Optimizer — adamw **11.4K** ≥ sgd **10.8K** ≈ muon **10.8K** > lion 10.2K > sophia 10.1K > schedulefree/prodigy ~9.8K — Muon opt_ms **117** vs AdamW **34** (Newton–Schulz tax).
- The result artifact reports: FFN — swiglu **10.3K** > gelu **9.8K** > relu2 **9.2K** — **moe OOM** at bs16.
- The result artifact reports: Mixer — mla **9.3K** > attention **7.9K** > mingru **6.7K** ≫ mamba2 **333** ≫ gdn **238** — Sequential SSM refs — 24–33× too slow.
- Failure/operational record: `ffn.moe`: `"ok": false, "err": "OOM"` at bs16 (8 dense experts + optimizer state).
- Failure/operational record: Sequential mamba2/gdn not OOMs but practically blocked (bwd alone ~9–12.5 s/step).

## What this does not prove

**Confidence: Medium.** Direct fixed-shape measurements support the throughput ranking; opt/FFN and mixer shapes differ, and throughput says nothing by itself about quality. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.4
- Related suites: [`16-optimizer-quality-bakeoff`](16-optimizer-quality-bakeoff.md), [`19-chunk-parallel-kernels`](19-chunk-parallel-kernels.md), [`18-gpu-maximization`](18-gpu-maximization.md)

---

[Previous](16-optimizer-quality-bakeoff.md) · [Index](../00-INDEX.md) · [Next](18-gpu-maximization.md)
