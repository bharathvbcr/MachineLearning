# 12: Mixer A/B on TinyStories

## Executive summary

- **Question:** With architecture, opt, and token budget fixed, swapping only the sequence mixer on a short TinyStories run should show whether recurrent inductive bias helps before the larger FineWeb bakeoff.
- **Result:** One-lever mixer A/Bs on TinyStories reproduce the bakeoff ranking cheaply — bias helps when data is scarce.
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `12-mixer-ab-tinystories` |
| Dates | 2026-06-15 – 2026-06-15 |
| Hardware | RTX 3070 Ti Laptop 8 GB |
| Status | `done` |

## Hypothesis

With architecture, opt, and token budget fixed, swapping only the sequence mixer on a short TinyStories run should show whether recurrent inductive bias helps before the larger FineWeb bakeoff.

## Setup

- Trainer / preset: `nanolab` — `ab_{attention,mingru,mamba2,gdn}`
- Fixed knobs: 6L/384d/6H, block 128, bs12, Muon lr 1e-3, 300 steps (~461K tokens), TinyStories, GPT-2, bf16, seed 1337, mixer_chunk 64
- Env flags: device `auto`

## Variants

| Variant | Change |
|---------|--------|
| `ab_attention` | mixer=`attention` |
| `ab_mingru` | mixer=`mingru` |
| `ab_mamba2` | mixer=`mamba2` |
| `ab_gdn` | mixer=`gdn` |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | `ab_mingru` | best_val **3.194** | MFU ~10.5%; only run that saved `final.pt` |
| 2 | `ab_attention` | best_val **3.209** | MFU ~12.3% — within ~0.015 of minGRU |
| 3 | `ab_gdn` | best_val **3.276** | MFU ~0.5% (still-slow GDN path at this date) |
| 4 | `ab_mamba2` | best_val **3.402** | MFU ~0.7%; weakest quality here |

Short TinyStories ranking already tips toward minGRU over attention, with GDN/Mamba-2 trailing — same qualitative order as the later 2M FineWeb bakeoff, at much lower cost. Throughput tax on sequential-ish SSM paths shows up as MFU collapse even when loss is only modestly worse.

**Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

## Failures

- No crashes. Low MFU on `mamba2`/`gdn` foreshadows the chunk-kernel work in suite 19; quality signal is still valid at this token count.

## Lesson

**One-lever mixer A/Bs on TinyStories reproduce the bakeoff ranking cheaply — bias helps when data is scarce.**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.train`.
- Required replay inputs: the four `nanolab/out/ab_*/config.json` files. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** All four one-lever runs completed at one seed; the small minGRU–attention gap has no seed replication.

## Artifacts

- `nanolab/out/ab_attention/`, `ab_mingru/`, `ab_mamba2/`, `ab_gdn/` — each `config.json` + `metrics.jsonl`

## Why this experiment happened

Once the trainer and metrics were healthy, the cheapest architecture question was a one-lever mixer swap. TinyStories offered a low-cost screen for whether recurrent inductive bias was visible before committing to a 124M FineWeb bakeoff. The preceding notebook context is [11-phase1-fineweb](11-phase1-fineweb.md).

## Experiment story

**Baseline.** Once the trainer and metrics were healthy, the cheapest architecture question was a one-lever mixer swap. TinyStories offered a low-cost screen for whether recurrent inductive bias was visible before committing to a 124M FineWeb bakeoff. The preceding notebook context is [11-phase1-fineweb](11-phase1-fineweb.md).

**Hypothesis.** With architecture, opt, and token budget fixed, swapping only the sequence mixer on a short TinyStories run should show whether recurrent inductive bias helps before the larger FineWeb bakeoff.

**Test contract.** Trainer / preset: `nanolab` — `ab_{attention,mingru,mamba2,gdn}` Fixed knobs: 6L/384d/6H, block 128, bs12, Muon lr 1e-3, 300 steps (~461K tokens), TinyStories, GPT-2, bf16, seed 1337, mixer_chunk 64 Env flags: device `auto`

**Variant sequence.** The preserved comparison matrix was: `ab_attention` — mixer=`attention`; `ab_mingru` — mixer=`mingru`; `ab_mamba2` — mixer=`mamba2`; `ab_gdn` — mixer=`gdn`.

**Measured turn.** The result board records 1 — `ab_mingru` — best_val **3.194** — MFU ~10.5%; only run that saved `final.pt`; 2 — `ab_attention` — best_val **3.209** — MFU ~12.3% — within ~0.015 of minGRU; 3 — `ab_gdn` — best_val **3.276** — MFU ~0.5% (still-slow GDN path at this date); 4 — `ab_mamba2` — best_val **3.402** — MFU ~0.7%; weakest quality here.

**Turning point and readout.** Short TinyStories ranking already tips toward minGRU over attention, with GDN/Mamba-2 trailing — same qualitative order as the later 2M FineWeb bakeoff, at much lower cost. Throughput tax on sequential-ish SSM paths shows up as MFU collapse even when loss is only modestly worse. **Interpretation boundary.** The table values are measured from the listed artifacts. The lesson is an inference bounded to this hardware, seed, horizon, and configuration; unreported seed variance should not be assumed negligible. Throughput and quality are separate outcomes unless both were directly measured.

**Failures and surprises.** No crashes. Low MFU on `mamba2`/`gdn` foreshadows the chunk-kernel work in suite 19; quality signal is still valid at this token count.

## Decision and aftermath

**Kept:** One-lever mixer A/Bs on TinyStories reproduce the bakeoff ranking cheaply — bias helps when data is scarce. The notebook continues with [13-mixer-bakeoff-2M](13-mixer-bakeoff-2M.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — `ab_mingru` — best_val **3.194** — MFU ~10.5%; only run that saved `final.pt`.
- The result artifact reports: 2 — `ab_attention` — best_val **3.209** — MFU ~12.3% — within ~0.015 of minGRU.
- The result artifact reports: 3 — `ab_gdn` — best_val **3.276** — MFU ~0.5% (still-slow GDN path at this date).
- The result artifact reports: 4 — `ab_mamba2` — best_val **3.402** — MFU ~0.7%; weakest quality here.
- Failure/operational record: No crashes. Low MFU on `mamba2`/`gdn` foreshadows the chunk-kernel work in suite 19; quality signal is still valid at this token count.

## What this does not prove

**Confidence: Medium.** All four one-lever runs completed at one seed; the small minGRU–attention gap has no seed replication. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.2
- Related suites: [`13-mixer-bakeoff-2M`](13-mixer-bakeoff-2M.md), [`19-chunk-parallel-kernels`](19-chunk-parallel-kernels.md)

---

[Previous](11-phase1-fineweb.md) · [Index](../00-INDEX.md) · [Next](13-mixer-bakeoff-2M.md)
