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
| B3 | ~~`nanolab/out/` is gitignored, so the per-job `metrics.jsonl` that §9 lists as required artifacts are not published~~ | **CLOSED 2026-08-24.** The ignore rule carries explicit exceptions (`metrics.jsonl`, `config.json`, `queue.json`, `recipe.json`, `ledger.json`, `summary.json` at any depth), published in commit `2d540b7`. 128 run directories behind §4 are tracked — the 120 jobs of suites 22–26 plus the eight `wave0_bs8/` drift runs — along with 165 files under `out/funnel/` covering the whole D7 grid. Every §4 table and the entire §8.3 grid now recompute from the repository alone. Paper §10 said the opposite until 2026-08-24 and has been corrected. |

### 2.2 Experimental

| # | Gap | Cost |
|---|---|---|
| E1 | **RUN 2026-08-27 (250 jobs, 33.21 GH200-h) — but §8.4 rows 1-4 remain UNANSWERED.** Every µP cell failed the bundle's own comparator check; the cause was an attention-temperature defect, not µP (backlog E13). The corrected arm `e1_mup_spattn` crosses cleanly (10.12M on the 50M cosine, 5/5 seeds) but is µP-with-SP-attention, so it cannot be pointed at rows written for µP. Open item is a framing decision, not compute. **The µP arm (PAPER §8.4)** — specified with a pre-registered outcome table. **Design corrected 2026-08-25/26 and enlarged from 52 to 240 E1 jobs**, because as designed it could not answer its own decision rule: every cell ran at the s24 recipe while readout rows 2–3 are about s23 (20M cosine) and row 4 about s25 (batch 8), so the run would have answered **one readout of four**. Three further defects fixed in the same pass — the transferred LR was decided by **one seed** (§8.3's failure one axis over), each µP arm was measured at **one learning rate** (§8.3's offset-basin mechanism, rebuilt), and µP transfer from width 256 to 768 was **asserted, never measured** (the divisor in `optim.py` is the *Adam* µP rule applied to a Muon group). Runner is `scripts/gpu_bundle.py`; see `docs/GPU_BUNDLE.md`. | 240 jobs: `e1_proxy` 54 → `e1_sp_rerun` 10 → `e1_mup` 10 → `e1_mup_basin` 42 → `e1_mup_tuned` 10 → `e1_sp_basin` 54 → `e1_sp_sched20` 10 → `e1_mup_sched20` 10 → `e1_sp_bs8` 10 → `e1_mup_bs8` 10 → `e1_perlayer_sp` 10 → `e1_embed_lr` 10 
| E3 | **Optimizer axis is n=2** at `advance_1000` and `exact_128m_1000` | Metal/M5 only, not CUDA |
| E4 | **Suites 14/15 are single-seed with no preserved replay command**; suite 15's two follow-up scales are n=1 | needs 3070 Ti |
| E5 | **Hardware is never isolated** — suite 25 isolated batch on GH200 only | needs 3070 Ti |
| E6 | **No wall-clock/throughput artifact for suites 22–26.** | **FIXED 2026-08-22** (see below); suites 22–26 remain estimate-only |
| E7 | **The seeded-attention 20k Metal run (PAPER §6.6)** — **CLOSED 2026-08-26, no compute.** The run already existed: `metal-native/out/sota_f32_clipsoft_seed42_20k_fa_tiled_softfix_warmdown_reseed`, FINAL EMA BPB **1.887607**, logged 2026-07-12 and already quoted in the §6.6 table as "seed 42 / 1.8876" without being recognised as the seeded arm. Seeded attention beats seeded minGRU 1.887607 vs 1.993295 (gap 0.1057) against an init effect of 0.0049 and backend nondeterminism of 0.0044, so the crossing is not an init artifact; §6.6 updated. Evidence and residual caveats (cross-seed pair; arms not parameter-matched, 0.780M vs 0.977M): `docs/EXPERIMENT_BACKLOG_2026-08-26.md` Tier 0. Note: no Metal run of any arm is reproducible on this machine — `fineweb10B_sp1024` and `fineweb_1024_bpe.model` left with the deleted `parameter-golf` self-clone (verified absent by whole-disk Spotlight sweep, `~/Code` to depth 7, and both backups). | none — closed from artifacts |
| E8 | **RUN 2026-08-29/30 — 405 runs across a 3x2 difficulty x budget grid. Outcome was NEITHER pre-registered row; see PAPER §6.8 and §7.3 items 6-7.** Recall yields whichever ordering you select via difficulty and budget, and none once both arms are adequately trained. One durable result: `mingru` alone is immovable by budget (rate AND median), while pure `gdn` reaches 13/15 — so the split is a minGRU property, not a recurrence property. **Recall probe (MQAR-style)** — the §4.5 board is measured on held-out CE at 512 only, which cannot see in-context recall, the axis hybrids exist for; the attention/hybrid tie may be metric-dependent (§6.2, third metric). Needs new synthetic-task code in `nanolab`; Mac-runnable. Spec: backlog E8. | new code + small runs |
| E9 | **RUN — `crossover50m_ctx2048`, 25 jobs, backs PAPER §6.7.** At 2048 context the §4.5 tie holds: attention 4.177 vs `hybrid_mingru10_attn2` 4.176. **Sequence-length axis** — every quality result is `block_size=512`; the hybrid ordering at 2048/4096 is unmeasured. Needs a `CROSSOVER_BLOCK` override in `crossover_replicate.py`; cost unknown until a smoke run at the new shape (512-context medians do not transfer — D14's lesson). Spec: backlog E9. | runner extension + GH200 |
| E10 | **RUN, then RE-RUN WITHIN ONE SUITE — `crossover50m_ratio32` (20 jobs) superseded by `crossover50m_ratioplace32` (25 jobs, all five arms, `final_val`).** The first pass could only compare the four new ratios to 10+2 across suites *and* across statistics, so its placement claim was withdrawn; `lock_recipe` refuses a fifth arm in a suite whose recipe records four, so the fix was a fresh five-arm suite rather than a 5-run patch. Paired per seed, every comparison is 5/5 with an interval disjoint from zero: 9+3 beats 10+2 by 0.0375, beats 8+4 by 0.0077, beats 11+1 by 0.0591; and **10+2 beats bookend by 0.0283 at equal attention-layer count**, so placement is real and independent of count. Count is not monotone — three attention layers beat four. Free reproducibility check: the four re-run arms reproduce their original suite to within 0.0006 nats. **minGRU hybrid ratio/placement sweep** — the board's best hybrid family exists at one ratio (10+2 last-2) while GDN got three variants; the field's 3:1-periodic ratio was never run on the family that ties attention. Config-only: four new `Arm` entries, 20 jobs at the suite-26 recipe. Spec: backlog E10. | 20 GH200 jobs |
| E11 | **RUN, both phases; backs PAPER §6.9.** The first phase-2 attempt is retained as a documented failure (1.70x wall-clock spread against target). The retry held every arm within +2.2%/-1.0% of 691s: attention 4.1045 beats the best hybrid 4.1640 with disjoint intervals. Also surfaced the `best_val` eval-count bias (§7.4). **Wall-clock-matched board** — §4.5 is token-matched; practitioners adopt hybrids at matched wall-clock. Phase 1 re-indexes the committed suite-26 curves by median `tok_s` (free, analysis only, within-suite/same-concurrency caveat); phase 2 runs it for real only if phase 1 moves a rank. Spec: backlog E11. | phase 1 free |
| E12 | **NOT STARTED — but its gate is now satisfied** (E8 exists, so a windowed result is interpretable). Still the lowest-priority item. **Sliding-window attention arm** — no windowed-attention path exists in `mixers.py`; asks whether the hybrid's attention layers must be global (the QSA question at small scale). Only interpretable after E8. Spec: backlog E12. | new code; after E8 |

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
the µP arm — and it should be recorded at the concurrency the reference suites used, not
single-tenant. `ISOLATE_STAGES` records `"workers": 2` for suites 23, 25 and 26 and suite 22's
documented launch is `--workers 4`, so the median `tok_s` this repository prices against is
already a contended rate; a single-tenant run would produce throughput numbers that are not
comparable to the suites the µP arm is measured against. Concurrency cannot affect a crossing
token — separate processes, separate CUDA contexts, per-job seeds — so it is a reporting choice,
and `gpu_bundle.py` records the worker count in the ledger `meta` so a contended rate can never
be read as an uncontended one.

**D14 (new, 2026-08-25): the corpus is part of the recipe and was treated as a capacity
requirement.** `docs/GPU_BUNDLE.md` said E1/E2 "need ~50M tokens' worth of headroom and the
existing 497.5M-token corpus is ample." The `Batcher` samples windows uniformly **with
replacement**, so a 20M-token job over a 50M-token corpus and the same job over the 497.5M-token
corpus are two different training distributions — 0.4 epochs against 0.04. Every cell in this
bundle is compared against a published suite, so the corpus must be *the same corpus*. Preflight
now fails closed when `train.bin` is not 497,500,000 tokens, and the documented procedure is to
copy the reference `train.bin`/`val.bin` rather than re-tokenize, since a fresh tokenization of
the same nominal size is not guaranteed byte-identical.

### 2.3 Data integrity

| # | Gap | Status |
|---|---|---|
| D1 | `ci95` computed with the normal quantile, and `0.0` for n=1 | **FIXED 2026-08-22** — see below |
| D2 | `winner_exact_gate: "pending"` in `research/native-optimizer-funnel.json` while `research/exact-128m-gate-polar.json` records `passed: true` | **CLOSED 2026-08-22** — set to `"passed"` with the gate evidence inlined (`loss_delta` 1.43e-6, `grad_norm_delta` 0.0, 128,367,988 params, 1707 dispatches). Systems gate only; says nothing about the selection. |
| D3 | `research/champion-run.json` has `"locked": false` | **open, and must stay open.** The 52-job / three-seed D7 grid finds the ordering **crosses over** in learning rate rather than settling: `muon_polar_adamw` leads at lr 0.0035 and 0.005, `normuon_adamw` at the six higher points, and at each candidate's own best cell the paired gap is +0.004097 ± 0.005085 — spanning zero. Do **not** lock on `muon_polar_adamw`, and do **not** lock on `normuon_adamw` either. A lock now needs a **bracketed** optimum for both at a shared protocol; neither is bracketed today (both minima sit in flat basins unresolvable at n=3). `lock_reason` and `lr_transfer_finding` are **generated** by `scripts/d7_analyze.py --write` from the run ledger — an earlier hand-written version of both survived the round that superseded it and asserted the opposite conclusion; `--check` now fails on that drift. |
| D4 | Champion final EMA BPB recorded as **2.010659** in `champion-run.json` vs **2.015756** in `MASTER_ARCHITECTURAL_KB.md`, `Rust_MLKit/docs/optimization_map.md` and `metal-native/DECISIONS.md` | **RESOLVED 2026-08-22 — it was a typo.** Artifacts read **2.015576** (audit7) and **2.010659** (audit8); `2.015756` matches neither and is a `57`↔`75` transposition of audit7. Citable value **2.010659**. KB corrected and grade restored to A. **Diagnosis corrected 2026-08-23:** `2.015756` is *not* a typo — it is the pre-Audit-7 champion's own value (`out/champion_128m_seed1337`, ~2895 ms/step), corroborated by DECISIONS §M15's independent "2.0158", which `2.015576` does not round to. That artifact was superseded and deleted, so the figure is unverifiable rather than mistyped. `optimization_map.md`, `DECISIONS.md`, `blog_results.sh` and `ab_flags.rs` describe that earlier run correctly and were left unchanged. |
| D5 | Same-shape CUDA 128M reference still `null` — the matched control behind withdrawal #2 in §6.3 | open |
| D6 | Suite 25's 8.192M final eval is not logged as `event=eval`, so the last paired marker is 7.377M | open |
| **D7** | **CLOSED 2026-08-23 — the ordering CROSSES OVER in learning rate.** Both funnel finalists' LRs were tuned at `arch02-16m` and never re-tuned at 128M. A **52-job, three-seed** matched eight-point grid at the exact `exact_128m_1000` protocol (`research/d7-lr-retune.json`, argv byte-identical apart from `--out`) finds `muon_polar_adamw` ahead at lr 0.0035 and 0.005 and `normuon_adamw` ahead at 0.008, 0.0125, 0.018, 0.025, 0.035 and 0.05 — **every row sign-consistent across all three seeds**, 24 paired comparisons with no dissent. The mechanism is offset flat basins (Polar 0.005–0.008, NorMuon 0.008–0.0125), so each wins inside its own. Inherited-LR penalties: Polar 0.046020 ± 0.003175 (**1.47×** the 0.031226 selection margin), NorMuon 0.080400 (n=2, 2.57×). The funnel compared 0.05 against 0.1 — both on the high-LR wall, one side of a crossing it could not see. **The selection is retired, not reversed**: `normuon_adamw` is not crowned, because at each candidate's own best cell the gap is +0.004097 ± 0.005085, which spans zero. **Superseded readings of this row:** an earlier 24-job / two-seed round recorded here read "normuon ahead at all five matched LRs" and a best-cell gap of 0.016317, and a 500-step round before that put the optimum at ≈0.035 with a 1.95× penalty. Both were measured on truncated grids that did not reach below the crossing; both are withdrawn. Paper §8.3 and `research/d7-lr-retune.json` are the current record and this row is derived from them. | **CLOSED** |
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

**Everything in this part is derived, not typed.** Regenerate it with:

```bash
python3 scripts/gpu_bundle.py --cost
```

That reads the committed per-job `metrics.jsonl` of suites 22–26, takes the median
`tok_s` per (mixer, batch, context), and prices the 394-job matrix against it. E6 notes
that the trainer discarded elapsed time; it never discarded per-step throughput.

### 3.1 What actually needs a GPU

Only the CUDA `nanolab` work. E3 is Rust/Metal on the M5 Pro; E4/E5 need the RTX 3070 Ti.

| suite | jobs | tokens/job | GH200-hours |
|---|---|---|---|
| `e1_proxy` (width-256 matrix-LR sweep, per arm, 3 seeds, 9-point grid) | 54 | 19.99M | 1.91 |
| `e1_sp_rerun` (SP cells of the s24 2×2, hardware + drift control) | 10 | 19.99M | 1.06 |
| `e1_mup` (µP at the **transferred** LR -- the pre-registered cell) | 10 | 19.99M | 1.06 |
| `e1_mup_tuned` (µP at its **measured** target-width optimum, n=5) | 10 | 19.99M | 1.06 |
| `e1_mup_basin` (target-width µP LR curve, 0.25×…32×, 3 seeds) | 42 | 19.99M | 4.46 |
| `e1_sp_basin` (target-width **SP** LR curve, 512× span, 3 seeds) | 54 | 19.99M | 5.73 |
| `e1_sp_sched20` / `e1_mup_sched20` (s23, 20M cosine) | 20 | 19.99M | 2.12 |
| `e1_sp_bs8` / `e1_mup_bs8` (s25, batch 8) | 20 | 8.19M | 1.68 |
| `e1_perlayer_sp` | 10 | 19.99M | 1.06 |
| `e1_embed_lr` | 10 | 19.99M | 1.06 |
| `e2_matched32_50m` | 10 | 49.99M | 2.65 |
| **total (default matrix)** | **394** | | **≈ 43.72** (mixed-tenancy basis; see `GPU_BUNDLE.md`) |
| `d10_horizon` — **opt-in**, `--with-d10`, 3 seeds | 6 | 327.7M / 655.4M | ≈ 20 |

1.91 of those hours are **extrapolated, not measured**: no committed run covers width
256, so `--cost` applies a labelled factor to a measured rate. Every other row is
priced against a rate recorded on the GH200 the suites ran on.

### 3.2 Instance recommendation: 1× GH200, and it is not close

Two earlier versions of this section were wrong in opposite directions. The first
optimised $/GPU-hour and recommended the 8× A100 40 GB. The second recommended
running "E1+E2 on 8× A100" while `d10_horizon` was decided separately. Both are
wrong for **E2**, and the reason is the one this document had already established for
E1: `e2_matched32_50m`'s cells are **merged into suite 26's published board**, whose
other eight rows were measured on a GH200. Measured on an A100 they carry the
cross-hardware confound §7.1 refuses, and the board ends up worse than it started.
The runner now enforces this in preflight rather than documenting it.

| instance | $/GPU-hr | full 394-job bundle |
|---|---|---|
| **1× GH200 96 GB — $2.29/hr** | **2.29** | **$57 / 24.9 h** |
| 4× H100 SXM5 — $16.36/hr | 4.09 | $73–79 / 4.5–4.8 h |
| 8× A100 40 GB — $15.92/hr | 1.99 | $64–81 / 4.1–5.1 h |
| 2× H100 SXM5 — $8.38/hr | 4.19 | $66–72 / 7.9–8.6 h |
| 1× H100 SXM5 — $4.29/hr | 4.29 | $64–70 / 14.8–16.3 h |
| 1× H100 PCIe — $3.29/hr | 3.29 | $59–72 / 18.1–21.7 h |
| 1× A100 40 GB — $1.99/hr | 1.99 | $50–66 / 25.2–33.3 h |
| 1× A10 24 GB — $1.29/hr | 1.29 | $64–95 / 49.4–73.6 h |

The GH200 is the **cheapest row and the only correct one**. It is also the only row
whose range is not an assumption: every measured rate in the cost model was recorded
on that hardware, so its ratio is 1.0 by definition while every other row's is a
guess about relative throughput. Dropping `d10_horizon` to opt-in is what makes the
single-GPU box viable — the old matrix's critical path was one serial 4.4-hour job.

### 3.3 The SP re-run is not optional, and stays on even on a GH200

The 2×2 needs its SP and µP cells measured on the same box. Closed 2026-08-24:
`e1_sp_rerun` is a first-class suite, on by default. `--sp-cells suite24` remains the
opt-out but now **requires a GH200 by device name**, and is recommended against even
there: the re-run is 30 cheap jobs that convert an assumption about the environment
into a measured drift number against suites 23/24/25, and it is the only way to
detect a PyTorch or driver change.

Also keep `compile=False`. It was forced on GH200 aarch64 by an Inductor stall; on x86
it would work, and enabling it would be one more recipe change.

### 3.4 The corpus is part of the recipe — copy it, do not re-tokenize

`nanolab/data/` is gitignored, so a fresh box has no tokenized corpus. See D14 above:
because the `Batcher` samples **with replacement**, a corpus of a different size is a
different training distribution, and every cell here is compared against a published
suite. Preflight fails closed unless `train.bin` is 497,500,000 tokens.

```bash
rsync -az nanolab/data/HuggingFaceFW_fineweb-edu/ BOX:~/MLSystemsLab/nanolab/data/HuggingFaceFW_fineweb-edu/
```

Verify with `md5sum` on both ends; a fresh `prep_fineweb` run of the same nominal size
is not guaranteed byte-identical (shard selection, dataset revision).

**`d10_horizon` still does not fit**: its 20k arm requests 655.4M tokens, so it would
revisit training data at 1.32 epochs while its 10k partner sits at 0.66 — a second
variable moving alongside the one the pair exists to isolate. `--preflight` fails
closed on this; `--allow-data-repeat` accepts it instead and records `data_epochs` per
job so the repeat cannot go unreported.

### 3.5 GH200 capacity IS the deciding factor

An earlier version of this section said it was not, on the grounds that using the
original box saves only the 10-job SP re-run (~$2–5), which is not worth waiting for.
That reasoning covered E1 and silently omitted E2, whose ten cells join a GH200 board
and are not portable at any price. It is also now moot in the other direction: on the
current price list the GH200 at $2.29/GPU-hour is the **cheapest** instance for this
bundle as well as the only valid one, so there is no longer a trade-off to make.
