# 13: Mixer bakeoff @ 2M FineWeb tokens

## Executive summary

- **Question:** Under a scarce ~2M-token FineWeb budget, recurrent/SSM mixers should beat standard attention — inductive bias matters most when data is short.
- **Result:** Recurrent/SSM inductive bias beats attention when data is scarce (~2M tokens).
- **Implication:** Use this as evidence for the recorded decision within the tested setup; broader transfer remains an inference.
- **Status:** `done`; evidence confidence **Medium**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `13-mixer-bakeoff-2M` |
| Dates | 2026-06-15 – 2026-06-15 |
| Hardware | RTX 3070 Ti Laptop 8 GB |
| Status | `done` |

## Hypothesis

Under a scarce ~2M-token FineWeb budget, recurrent/SSM mixers should beat standard attention — inductive bias matters most when data is short.

## Setup

- Trainer / preset: `nanolab` — `bakeoff_{mingru,gdn,mamba2,attention,mla}`
- Fixed knobs: 12L/768d/12H (~124M), block 512, bs8, Muon lr 6e-4, cosine, 500 steps (= **2.048M tokens**), FineWeb-edu, bf16, seed 1337, mixer_chunk 32, eval every 100
- Env flags: device `auto`; identical opt/sched across mixers

## Variants

| Variant | Change |
|---------|--------|
| `bakeoff_mingru` | mixer=`mingru` |
| `bakeoff_gdn` | mixer=`gdn` |
| `bakeoff_mamba2` | mixer=`mamba2` |
| `bakeoff_attention` | mixer=`attention` |
| `bakeoff_mla` | mixer=`mla` |

## Results

Matches [08.2](../../learning-notes/08-experiments-and-results.md) (artifact best_val rounded to 3 decimals there):

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| 1 | **minGRU** | best_val **5.837** | Saved `best.pt`; led every eval after step 100 |
| 2 | Gated DeltaNet | best_val **5.994** | +0.157 vs winner |
| 3 | Mamba-2 | best_val **6.040** | +0.203 |
| 4 | Attention | best_val **6.073** | The “default” loses here |
| 5 | MLA | best_val **6.156** | Worst quality; win is inference KV memory |

Curve snapshot (val @ 100 → final): minGRU 6.62→5.84; attention 6.74→6.07. All five finish within ~0.32 loss — **ranking** is the signal. Recurrent mixers take the podium at low tokens.

**Interpretation boundary.** The loss ranking is measured at equal token budget; the mechanism (“inductive bias”) is an inference. Slow kernels affect wall-clock cost, not the fixed-token quality values.

## Failures

- None. Sequential SSM paths were slow (low MFU on gdn/mamba2) but completed 500 steps.

## Lesson

**Recurrent/SSM inductive bias beats attention when data is scarce (~2M tokens).**

## Reproduction

- **No exact replay command was preserved.**
- Closest runner: `python3 -m nanolab.train`.
- Required replay inputs: the five `nanolab/out/bakeoff_*/config.json` files. Reconstruct every recorded field before comparing outputs.

## Evidence quality

**Confidence: Medium.** Five matched one-seed runs completed. Ranking is measured, but seed variance and unequal kernel throughput limit causal strength.

## Artifacts

- `nanolab/out/bakeoff_{mingru,gdn,mamba2,attention,mla}/config.json`, `metrics.jsonl`
- `nanolab/out/bakeoff_mingru/best.pt`

## Why this experiment happened

Suite 12 showed a small minGRU lead on TinyStories but only one seed and a 461K-token budget. The 2M FineWeb bakeoff moved the same question to a larger, matched 124M setting and added MLA to the comparison. The preceding notebook context is [12-mixer-ab-tinystories](12-mixer-ab-tinystories.md).

## Experiment story

**Baseline.** Suite 12 showed a small minGRU lead on TinyStories but only one seed and a 461K-token budget. The 2M FineWeb bakeoff moved the same question to a larger, matched 124M setting and added MLA to the comparison. The preceding notebook context is [12-mixer-ab-tinystories](12-mixer-ab-tinystories.md).

**Hypothesis.** Under a scarce ~2M-token FineWeb budget, recurrent/SSM mixers should beat standard attention — inductive bias matters most when data is short.

**Test contract.** Trainer / preset: `nanolab` — `bakeoff_{mingru,gdn,mamba2,attention,mla}` Fixed knobs: 12L/768d/12H (~124M), block 512, bs8, Muon lr 6e-4, cosine, 500 steps (= **2.048M tokens**), FineWeb-edu, bf16, seed 1337, mixer_chunk 32, eval every 100 Env flags: device `auto`; identical opt/sched across mixers

**Variant sequence.** The preserved comparison matrix was: `bakeoff_mingru` — mixer=`mingru`; `bakeoff_gdn` — mixer=`gdn`; `bakeoff_mamba2` — mixer=`mamba2`; `bakeoff_attention` — mixer=`attention`; `bakeoff_mla` — mixer=`mla`.

**Measured turn.** The result board records 1 — **minGRU** — best_val **5.837** — Saved `best.pt`; led every eval after step 100; 2 — Gated DeltaNet — best_val **5.994** — +0.157 vs winner; 3 — Mamba-2 — best_val **6.040** — +0.203; 4 — Attention — best_val **6.073** — The “default” loses here; 5 — MLA — best_val **6.156** — Worst quality; win is inference KV memory.

**Turning point and readout.** Matches [08.2](../../learning-notes/08-experiments-and-results.md) (artifact best_val rounded to 3 decimals there): Curve snapshot (val @ 100 → final): minGRU 6.62→5.84; attention 6.74→6.07. All five finish within ~0.32 loss — **ranking** is the signal. Recurrent mixers take the podium at low tokens. **Interpretation boundary.** The loss ranking is measured at equal token budget; the mechanism (“inductive bias”) is an inference. Slow kernels affect wall-clock cost, not the fixed-token quality values.

**Failures and surprises.** None. Sequential SSM paths were slow (low MFU on gdn/mamba2) but completed 500 steps.

## Decision and aftermath

**Kept:** Recurrent/SSM inductive bias beats attention when data is scarce (~2M tokens). The notebook continues with [14-scale-crossover-8M](14-scale-crossover-8M.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: 1 — **minGRU** — best_val **5.837** — Saved `best.pt`; led every eval after step 100.
- The result artifact reports: 2 — Gated DeltaNet — best_val **5.994** — +0.157 vs winner.
- The result artifact reports: 3 — Mamba-2 — best_val **6.040** — +0.203.
- The result artifact reports: 4 — Attention — best_val **6.073** — The “default” loses here.
- Failure/operational record: None. Sequential SSM paths were slow (low MFU on gdn/mamba2) but completed 500 steps.

## What this does not prove

**Confidence: Medium.** Five matched one-seed runs completed. Ranking is measured, but seed variance and unequal kernel throughput limit causal strength. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- [learning-notes/08-experiments-and-results.md](../../learning-notes/08-experiments-and-results.md) — §8.2
- Related suites: [`12-mixer-ab-tinystories`](12-mixer-ab-tinystories.md), [`14-scale-crossover-8M`](14-scale-crossover-8M.md)

---

[Previous](12-mixer-ab-tinystories.md) · [Index](../00-INDEX.md) · [Next](14-scale-crossover-8M.md)
