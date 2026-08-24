# Open issues and gaps — 2026-08-22

Companion to [`PAPER_2026-08_Recipe_Dependent_Rankings.md`](../PAPER_2026-08_Recipe_Dependent_Rankings.md).
Part 1 records defects found in the ScholarLM / WisDev tooling used to build the paper's
literature section, because the paper's provenance note claims that tooling and the claim needs
to be auditable. Part 2 is the experimental backlog. Part 3 is the compute plan.

Everything below is either a verified observation with a command attached, or is labelled as an
estimate.

---

## Part 1 — ScholarLM / WisDev issues encountered

### 1.1 Environment (fixed)

**Symptom.** `wisdevGenerateManuscript` over MCP returned a section scaffold with the body text
"Themes will be synthesized once an LLM backend is available."

**Cause.** The MCP server is registered as a stdio process with an empty environment:

```
wisdev-local -> /Users/bharath/go/bin/wisdev mcp --provider openalex,arxiv   env: {}
```

`wisdev check` in that environment reports:

```
ERROR google genai configuration missing  fallback_secrets=["GOOGLE_API_KEY","GEMINI_API_KEY"]
!  LLM providers: mode=cloud but Vertex/Gemini unavailable
```

The Go orchestrator on `127.0.0.1:8081` was healthy the whole time
(`llmChain: vertex_ai`, `credentialSource: vertex_ai:env:GOOGLE_CLOUD_PROJECT`); only the
MCP child process could not see the project.

**Fix applied.** Re-registered the server with the project id in its environment:

```bash
claude mcp add wisdev-local -s user -e GOOGLE_CLOUD_PROJECT=scholarlm-vbc -- \
  /Users/bharath/go/bin/wisdev mcp --provider openalex,arxiv
```

`wisdev check` then reports `OK` / `chain=vertex_ai`. Takes effect on client restart.

**Suggested upstream change.** `wisdev mcp` should fail loudly at startup when no LLM backend
resolves, rather than starting successfully and degrading silently at manuscript time. A
manuscript tool that cannot draft should not advertise itself in `tools/list`.

### 1.2 `paper_lookup` unavailable on the stdio server

`wisdevPaperLookup` against the stdio server returns:

```
error: paper lookup failed: no provider found for paper_lookup
```

for every identifier, while the same call against the HTTP plugin path resolves correctly. The
stdio server is launched with `--provider openalex,arxiv`; the lookup capability appears to
require providers that this flag excludes, but the tool is still exposed and returns a runtime
error rather than being withheld or falling back.

**Impact on the paper.** All 38 references were resolved through the HTTP path instead. Had the
stdio path been the only one available, the literature section could not have been built.

### 1.3 Expired ScholarDoc credential is reported as an entitlement problem

`scholarlmGetAuthStatus` reports:

```
userID: local-dev-anonymous   tier: free   tierAuthoritative: false
tierSource: entitlement_store_unavailable
degraded: "Your tier could not be authoritatively resolved ... retry before concluding
           the account lacks access."
scholarDocAccess: false
```

The actual cause is simpler: `~/.scholarlm/credentials.json` holds an `idToken` whose
`expiresAt` was **2026-08-21T22:56**, i.e. expired. The message points the reader at a transient
entitlement-store fault and at their subscription tier, neither of which is the problem.

**Suggested upstream change.** Check token expiry before consulting the entitlement store, and
say "your session expired, sign in again" when that is what happened.

### 1.4 Retrieval drifts to the wrong sense of polysemous terms

Three separate cases, all reproducible:

| Query | Expected domain | What came back |
|---|---|---|
| "Rankings of neural architectures or optimizers ... do not reliably transfer" (`wisdevEvidenceSearch`) | ML methodology | AI-and-the-labor-market, Alzheimer's detection, retinal vessel segmentation |
| "scaling law crossover rank reversal small scale proxy ... compute optimal" (`wisdevSearchPapers`) | ML scaling | C4 plant photosynthesis evolution, Jupiter's temperate belts, land-use planning |
| "rank reversal of architecture and optimizer rankings with training token budget" (`wisdev docgen`) | ML pretraining | IR neural **ranking** model distillation |

The third is the damaging one, because it silently propagated: see 1.5.

**Workaround used for the paper.** Targeted `wisdevPaperLookup` by arXiv identifier, and short
searches with unambiguous vocabulary. The `methodology` research mode plus short queries behaved
well; long natural-language questions behaved badly.

### 1.5 DocGen drafts a full manuscript on the wrong topic without flagging it

`wisdev docgen` run 2, query "rank reversal of architecture and optimizer rankings with training
token budget in language model pretraining":

- **7 sections, 2,043 words** drafted about information-retrieval ranking-model distillation
  (query latency, Margin-MSE, BERT teacher/student passage ranking).
- **1** grounded source: Hofstätter et al., *Improving Efficient Neural Ranking Models with
  Cross-Architecture Knowledge Distillation*.
- Requested `--min-citations 20`. Delivered 1 reference.
- Reported a peer-review score of **0.53** rather than failing.

At no point did the pipeline observe that the drafted topic and the requested topic were
unrelated.

### 1.6 `canonical_sources` collapses to 1 regardless of retrieval

| Run | Query | `paperCount` retrieved | `canonical_sources` | refs emitted |
|---|---|---|---|---|
| 1 | long-form question | 1 | 1 | — (killed) |
| 2 | "rank reversal ... token budget" | **5** | **1** | 1 |
| 3 | "training token budget determines whether attention or recurrent ..." | — | **0** | **0** |

Five retrieved papers reducing to one canonical source looks like a dedup or quality gate that is
too aggressive, or a cap. Run 3 grounded on **zero** sources and still emitted a seven-section
manuscript with a score of 0.21 and 7 open revision tasks.

**Suggested upstream change.** `--min-citations N` should be a gate, not a hint: a run that
grounds on fewer than N distinct sources should fail with that reason rather than return a
document. A manuscript with **zero** sources should never be written to disk.

### 1.7 Adversarial review degrades silently to lineage-only

Run 2, from the log:

```
WARN manuscript adversarial review failed — degrading to lineage critique
     reviewed_sections=7  error="context deadline exceeded"
```

The emitted manuscript still carries **"Overall score: 0.53"** and a Strengths/Weaknesses/Risks
block. The degradation is disclosed only in one italic line
(`Verification: citation-lineage only`) and in stderr.

This is the same defect class the paper argues against in §8: *a check that could not run must not
report the same shape of result as a check that ran and passed.* A timed-out reviewer should
produce a null score, not a number.

### 1.8 Summary of DocGen usability for this paper

Three runs, ~20 minutes of Vertex compute, **zero usable output**. None of it was merged. The
38 hand-verified references obtained through targeted `wisdevPaperLookup` remain strictly better
grounding. DocGen's *structure* (section flow, revision tasks, lineage critique) is sound; its
*retrieval* is the failing component for a niche systems-ML topic.

---

## Part 2 — Experimental gaps

### 2.1 Blocking for arXiv

| # | Gap | Cost |
|---|---|---|
| B1 | No LaTeX package; figures exist as PDF/SVG but are unembedded | ~1 day |
| B2 | No author/affiliation/license metadata | minutes |
| B3 | `nanolab/out/` is gitignored, so the per-job `metrics.jsonl` that §9 lists as required artifacts are not published. Either publish a reduced curve export or restate §9 | ~2 h |

### 2.2 Experimental

| # | Gap | Cost |
|---|---|---|
| E1 | **The µP arm (§7)** — specified with a pre-registered outcome table, not run | ~40 jobs @ 20M + LR sweep |
| E2 | **Suite 26 never reran attention/minGRU at 50M** — its top-2 rows are suite 22's sample, capping the combined ranking at Medium-High | 10 jobs @ 50M |
| E3 | **Optimizer axis is n=2** at `advance_1000` and `exact_128m_1000` | Metal/M5 only, not CUDA |
| E4 | **Suites 14/15 are single-seed with no preserved replay command**; suite 15's two follow-up scales are n=1 | needs 3070 Ti |
| E5 | **Hardware is never isolated** — suite 25 isolated batch on GH200 only | needs 3070 Ti |
| E6 | **No wall-clock/throughput artifact for suites 22–26.** | **FIXED 2026-08-22** (see below); suites 22–26 remain estimate-only |

**E6 detail (fixed).** `Logger.done` in `nanolab/utils.py` computed the elapsed time, printed it
to the console, and emitted a `done` record containing only `best_val` and `tokens` — so every
suite's wall clock was discarded at the moment it was known. That is why Part 3 below is an
estimate.

- `Logger.done` now persists `elapsed_s` and `mean_tok_s`; `train.py` passes them.
- `nanolab/crossover_replicate.py` gained `load_run_timing()`, `timing_summary()` and a
  `timing` subcommand: `python -m nanolab.crossover_replicate timing --out <run-dir> --json`.
- Runs finished **before** this change carry no `elapsed_s`. They are back-estimated from median
  `tok_s` and labelled `estimated_from_median_tok_s`; a run with no throughput records at all is
  labelled `missing` and contributes **nothing** to the totals rather than reading as zero.
  `gpu_hours_measured` and `gpu_hours_estimated` are reported separately so an estimate can never
  be mistaken for a measurement.
- Two new tests cover the measured, estimated and untimed paths. 50/50 pass.

Suites 22–26 are therefore still estimate-only; the first suite to carry a true wall clock will be
the µP arm.

### 2.3 Data integrity

| # | Gap | Status |
|---|---|---|
| D1 | `ci95` computed with the normal quantile, and `0.0` for n=1 | **FIXED 2026-08-22** — see below |
| D2 | `winner_exact_gate: "pending"` in `research/native-optimizer-funnel.json` while `research/exact-128m-gate-polar.json` records `passed: true` | **CLOSED 2026-08-22** — set to `"passed"` with the gate evidence inlined (`loss_delta` 1.43e-6, `grad_norm_delta` 0.0, 128,367,988 params, 1707 dispatches). Systems gate only; says nothing about the selection. |
| D3 | `research/champion-run.json` has `"locked": false` | **open, and must stay open.** D7 closed against the recorded candidate: at matched LR the runner-up wins 2-of-2. `lock_reason` now records this. Do **not** lock on `muon_polar_adamw`, and do not re-lock on `normuon_adamw` either — locking either on n=2 evidence would repeat the original error with the sign flipped. A lock needs n>=3 and a bracketed optimum for both. |
| D4 | Champion final EMA BPB recorded as **2.010659** in `champion-run.json` vs **2.015756** in `MASTER_ARCHITECTURAL_KB.md`, `Rust_MLKit/docs/optimization_map.md` and `metal-native/DECISIONS.md` | **RESOLVED 2026-08-22 — it was a typo.** Artifacts read **2.015576** (audit7) and **2.010659** (audit8); `2.015756` matches neither and is a `57`↔`75` transposition of audit7. Citable value **2.010659**. KB corrected and grade restored to A. **Diagnosis corrected 2026-08-23:** `2.015756` is *not* a typo — it is the pre-Audit-7 champion's own value (`out/champion_128m_seed1337`, ~2895 ms/step), corroborated by DECISIONS §M15's independent "2.0158", which `2.015576` does not round to. That artifact was superseded and deleted, so the figure is unverifiable rather than mistyped. `optimization_map.md`, `DECISIONS.md`, `blog_results.sh` and `ab_flags.rs` describe that earlier run correctly and were left unchanged. |
| D5 | Same-shape CUDA 128M reference still `null` — the matched control behind withdrawal #2 in §6.3 | open |
| D6 | Suite 25's 8.192M final eval is not logged as `event=eval`, so the last paired marker is 7.377M | open |
| **D7** | **CLOSED 2026-08-23 — and it closed against the recorded champion.** Both finalists' LRs were tuned at `arch02-16m` and never re-tuned at 128M. A 24-job matched-grid re-tune at the exact `exact_128m_1000` protocol (`research/d7-lr-retune.json`, 15.6 GPU-h, argv byte-identical to the recorded stage) finds **`normuon_adamw` ahead of `muon_polar_adamw` at all five matched LRs on both seeds** (mean gaps 0.0126–0.0163) and at each candidate's best tested LR by **0.016317 BPB**, sign-consistent 2-of-2. Inherited-LR penalties: Polar 0.032507 (1.04× the 0.031226 selection margin), NorMuon 0.079048 (2.53×). The funnel's margin measured which inherited LR was *less wrong*, not which optimizer was better. The 500-step figures that opened D7 (optimum ≈0.035, penalty 1.95×) were artifacts of the wrong horizon and a truncated grid and are withdrawn. **The recorded selection is retired; `normuon_adamw` is not crowned in its place** — n=2 supports a sign, not a magnitude. | **CLOSED** |
| **D8** | **`out/funnel/polar_exact_lr_spot/ledger.json` had lost three completed jobs.** It recorded lr 0.025 as `"running"` and omitted 0.035, 0.07 and 0.1, though all four finished 2026-07-16. Reconstructed from the run directories. | **CLOSED 2026-08-22** |
| **D9** | **Suite 18's results table omitted the one cell that showed a rank inversion.** `gpu_opt_bs32` `best_val` = **4.9065** (fastest arm, worst model) was never reported; `gpu_max` at 11.9K tok/s has the best loss at 4.8001. Note corrected; now paper §6.3. | **CLOSED 2026-08-22** |
| **D10** | **`run128m_20k` is eight resumed segments, undocumented.** 9 `start` / 8 `done`; per-segment `best_val` 3.6279…3.6109 with global min **3.5806**, which beats `run128m_10k`'s 3.621 — so suite 20's ranking inverts depending on which number is read. Token accounting is also wrong: every `done` records 98,304,000 tokens (3000 steps' worth) for a run that reached step 19,990 (expected 655,360,000). Note corrected. | **open — suite 20 supports no horizon claim as run** |
| **D11** | **Paper §7.3 claimed a source could not be located; it is in the repository.** The "29/29 oracle tests" figure is at `experiment-notes/arch-metal/51-m5-128m-optimizer-funnel-preflight.md:291`. Corrected. | **CLOSED 2026-08-22** |
| **D12** | **`MASTER_ARCHITECTURAL_KB.md` changelog stopped at 2026-07-20.** GH200 suites 22–26 (120 runs, n=5), the CI correction and the LR spot-check were all absent. Changelog updated; folding suites 22–26 into the KB body is still outstanding. | **partially closed 2026-08-22** |
| **D13** | **`experiment-notes/00-INDEX.md` carried suite 37 as `partial / Low / greet16 0/16`** while the note itself records `resolved / High / 16/16` since 2026-07-19. Index and open-questions list corrected. | **CLOSED 2026-08-22** |

**D1 detail (fixed).** `nanolab/native_funnel.py` computed
`ci95 = 1.96 * stdev / sqrt(n)` for `n > 1` and `0.0` for `n == 1`. Two defects:

1. At the funnel's two-seed stages the correct multiplier is `t_1 = 12.706`, not `z = 1.960` —
   the interval was **~6.5× too narrow**.
2. A single-seed arm was recorded with a **zero-width** interval, i.e. infinite precision for an
   arm that was measured once.

Both fed `confidence_interval_overlaps_best`, which gates the declared systems tie-breakers, so
the defect could route *selection*, not just reporting. On the real `exact_128m_1000` data:

| | `muon_polar_adamw` | `normuon_adamw` | verdict |
|---|---|---|---|
| pre-fix (z) | 2.169919 ± 0.005168 | 2.201145 ± 0.001884 | `overlaps_best = False` → **claimed separated** |
| post-fix (t₁) | 2.169919 ± 0.033500 | 2.201145 ± 0.012211 | `overlaps_best = None` → **not established** |

The recorded champion does not change: `muon_polar_adamw` leads on mean at every stage and on
both seeds at `exact_128m_1000`. What changes is that the ordering now rests on a
sign-consistent 2-of-2 result rather than on separated intervals — which is what §5.3 of the
paper states.

Changes: Student-t table (df 1–30, normal quantile beyond, documented); `n == 1` yields an
**infinite** half-width rather than zero; new `ci95_method`, `ci95_df`, `ci95_informative`,
`n_seeds` fields; systems tie-breakers are **skipped** when any arm has fewer than 3 seeds, with
`confidence_interval_overlaps_best = None` (not `False`) and a `tiebreaker_skip_reason`. Stored
history rankings were recomputed with the originals preserved under `ci95_legacy_z` and a
`corrections` block. Three new tests, all failing against the pre-fix code; 48/48 pass.

---

## Part 3 — Compute plan

### 3.1 What actually needs a GPU

Only the CUDA `nanolab` work. E3 is Rust/Metal on the M5 Pro; E4/E5 need the RTX 3070 Ti.

| Item | Jobs | Tokens/job |
|---|---|---|
| E1 µP 2×2 (all four cells, see 3.3) | 20 | 20M |
| E1 proxy-width LR sweep | ~10 short | ≪20M |
| E1 per-layer-SP arm (§7.3) | 10 | 20M |
| E1 embedding-LR ablation (§7.3) | 10 | 20M |
| E2 suite 26 completion | 10 | 50M |

**Estimated** ~10–14 H100-equivalent GPU-hours of compute, plus 1–2 h for data prep. This is an
estimate, not a measurement — see E6. Once the first jobs land, replace it with the real number:

```bash
python -m nanolab.crossover_replicate timing --out nanolab/out/<suite> --json
```

### 3.2 Instance recommendation

The workload is 50–60 **independent** jobs; `crossover_replicate launch --workers N` already
parallelises across them. So the figure that matters is **$ per GPU-hour**, and the multi-GPU
boxes cost the same per unit work while finishing sooner.

| Option | $/GPU/hr | Verdict |
|---|---|---|
| **8× A100 40 GB SXM4** — $22.32/hr | **$1.99** | **Recommended.** Cheapest GPU-hour on the list, 40 GB is ample for a 124M model at bs32×512, 124 vCPU / 1800 GiB feeds 8 workers comfortably. ~3–4 h wall-clock, **≈ $70–90 total**. |
| 1× A100 40 GB SXM4 — $1.99/hr | $1.99 | Same unit cost, ~28 h wall-clock. Only if you can babysit it; idle-billing between sessions is the real risk. **≈ $55–70.** |
| 1× H100 80 GB PCIe — $3.29/hr | $3.29 | Cheapest H100, but 65% more per unit work than A100 and a 124M model does not exploit H100's bandwidth. |
| 2× H100 SXM5 — $8.38/hr | $4.19 | Worst value here. |
| 1× H100 SXM5 — $4.29/hr | $4.29 | Worst value here. |
| 1× A10 24 GB — $1.29/hr | $1.29 | **Avoid.** Cheapest per hour but slowest, and bs32×512 activations on a 12L/768d model are likely to exceed 24 GB. |

### 3.3 The consideration that dominates price

**Changing hardware changes the recipe — which is this paper's own finding.**

The §7.3 design has two of its four cells (standard-parametrization attention and minGRU) already
run on the GH200. If the µP cells run on an A100 or H100, the 2×2 is confounded by hardware and
proves nothing — the exact error §4.2 documents, where the same pair differs by ~0.3 nats at
matched token markers across two GPUs.

So on any non-GH200 box, **re-run all four cells there**: +10 jobs at 20M, roughly +1 GPU-hour.
That is cheap, and it is not optional.

Also keep `compile=False`. It was forced on GH200 aarch64 by an Inductor stall; on x86 it would
work, and enabling it would be one more recipe change.

### 3.4 Wait for the GH200 if you can

The out-of-capacity **ARM64 + H100 (GH200) at $2.29/GPU/hr** is the right instance on both axes:
it is the second-cheapest GPU-hour on the list, *and* it is the box suites 22–26 ran on, so the
already-completed SP cells stay valid and E2 becomes a true same-hardware completion of suite 26.
Using it saves the 10 re-run jobs from 3.3 and removes a threat to validity that no amount of
compute buys back.

If the µP arm is not urgent, poll for GH200 capacity before booking an A100 box.
