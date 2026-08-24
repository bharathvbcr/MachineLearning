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
| E1 | **The µP arm (PAPER §8.4)** — specified with a pre-registered outcome table, not run. Runner is `scripts/gpu_bundle.py`; see `docs/GPU_BUNDLE.md`. | 52 jobs (`e1_proxy` 12 → `e1_sp_rerun` 10 → `e1_mup` 10 → `e1_perlayer_sp` 10 → `e1_embed_lr` 10) |
| E2 | **Suite 26 never reran attention/minGRU at 50M** — its top and eighth rows are suite 22's sample (`26-matched32_lock.json` marks both `"source": "suite22"`), capping the combined ranking at Medium-High | 10 jobs @ 50M (`e2_matched32_50m`) |
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
`tok_s` per (mixer, batch, context), and prices the 64-job matrix against it. E6 notes
that the trainer discarded elapsed time; it never discarded per-step throughput.

### 3.1 What actually needs a GPU

Only the CUDA `nanolab` work. E3 is Rust/Metal on the M5 Pro; E4/E5 need the RTX 3070 Ti.

| suite | jobs | tokens/job | GH200-hours |
|---|---|---|---|
| `e1_proxy` (width-256 matrix-LR sweep, per arm) | 12 | 19.99M | 0.42 |
| `e1_sp_rerun` (SP cells of the 2×2, hardware control) | 10 | 19.99M | 1.06 |
| `e1_mup` (µP cells) | 10 | 19.99M | 1.06 |
| `e1_perlayer_sp` | 10 | 19.99M | 1.06 |
| `e1_embed_lr` | 10 | 19.99M | 1.06 |
| `e2_matched32_50m` | 10 | 49.99M | 2.65 |
| `d10_horizon` | 2 | 327.7M / 655.4M | 6.56 |
| **total** | **64** | | **≈ 13.9** |

6.98 of those hours are **extrapolated, not measured**: no committed run covers
context 1024 or width 256, so `--cost` applies labelled factors (×0.7 for ctx1024,
×3 for the narrow proxy) to a measured rate.

### 3.2 Instance recommendation

The old version of this section optimised **$/GPU-hour** and recommended the 8× A100
40 GB. That is wrong for this matrix, and the reason is worth stating because it is a
scheduling fact rather than a price fact.

`d10_horizon`'s 20k job is a single serial run of ≈ 4.4 GH200-hours — **47% of the
bundle's compute in 2 of its 64 jobs**, and no number of GPUs shortens it. On an
8-GPU box you rent eight GPUs while one job monopolises one of them, so the cheapest
box per GPU-hour becomes one of the most expensive per bundle.

| instance | $/GPU-hr | full bundle | E1+E2 only |
|---|---|---|---|
| 8× A100 40 GB — $15.92/hr | 1.99 | $132–171 / 8.3–10.7 h | **$41–49 / 2.6–3.1 h** |
| 1× A100 40 GB — $1.99/hr | 1.99 | **$48–63** / 24–32 h | $26–34 / 13–17 h |
| 8× A100 80 GB — $22.32/hr | 2.79 | $166–217 / 7.4–9.7 h | $53–64 / 2.4–2.9 h |
| 1× H100 PCIe — $3.29/hr | 3.29 | $57–69 / 17–21 h | $32–38 / 9.6–11.5 h |
| 4× H100 SXM5 — $16.36/hr | 4.09 | $84–92 / **5.2–5.6 h** | $45–48 / 2.8–2.9 h |
| 2× H100 SXM5 — $8.38/hr | 4.19 | $64–70 / 7.6–8.3 h | $38–41 / 4.5–4.9 h |
| 1× H100 SXM5 — $4.29/hr | 4.29 | $61–67 / 14–16 h | $34–37 / 8.0–8.7 h |
| 1× A10 24 GB — $1.29/hr | 1.29 | $61–91 / 47–70 h | $33–49 / 25–38 h |

Ranges come from an **assumed** throughput ratio against the GH200 (H100 SXM5
0.95–1.05×, since GH200 carries an H100 die; A100 40 GB 0.45–0.60×). The
assumption-free form of the same question is the break-even multiple: an H100 SXM5 at
$4.29/hr must beat **2.16×** an A100 40 GB, per GPU, to cost less for the same work;
4× H100 needs **2.06×**; 8× A100 80 GB needs **1.40×** over its own 40 GB sibling,
which it will not deliver on the same silicon generation.

**Run it as two decisions, not one:**

1. **E1 + E2 on 8× A100 40 GB — $41–49, under 3 hours.** 62 jobs, none longer than
   18 minutes, fully independent: the case the 8-GPU box exists for. This is the
   work PAPER §8.4's pre-registered readouts depend on, and it is the whole of E1
   and E2.
2. **`d10_horizon` separately, or not at all.** On **2× H100 SXM5** the whole bundle
   is $64–70 in 7.6–8.3 h; alone on **1× A100 40 GB** the pair is ~$25 over a day.
   Suite 20's horizon claim is already withdrawn in *both* directions (D10), so this
   pair adds a new measurement rather than settling a live question — and it is the
   only part that needs an enlarged corpus.

40 GB is ample: weights + gradients + optimizer state is **1.07 GiB** (attention) and
**1.33 GiB** (minGRU) at the 768-dim target, computed exactly from the parameter
split. Activations are the rest and are not measurable without a GPU; `--smoke` on
the rented box is the check. `d10_horizon` runs at context 1024, double every other
job, and is the memory high-water mark.

### 3.3 The hardware control is now in the matrix

**Changing hardware changes the recipe — which is this paper's own finding.**

§8.4's 2×2 puts new µP cells against suite 24's SP cells, and suite 24 ran on a
GH200. On any other box that comparison is confounded by hardware, which PAPER §7.1
refuses explicitly: the same architecture pair differs by ~0.18–0.3 nats at matched
token markers across two GPUs.

This section said so, and said the +10 re-run jobs were "not optional." `docs/GPU_BUNDLE.md`
and `scripts/gpu_bundle.py` said the SP cells "are **not** rerun." **The two documents
disagreed and the code followed the wrong one.** Closed 2026-08-24: `e1_sp_rerun` is a
first-class suite in the matrix, on by default, and `--sp-cells suite24` is the
explicit opt-out for a GH200 launch.

Also keep `compile=False`. It was forced on GH200 aarch64 by an Inductor stall; on x86
it would work, and enabling it would be one more recipe change.

### 3.4 Data preparation is a required, billed step

`nanolab/data/` is gitignored, so a fresh box has no tokenized corpus:

```bash
python -m nanolab.prep_fineweb --config sample-10BT --max_tokens 50000000
```

E1 and E2 fit in the existing 497.5M-token corpus with room to spare. **`d10_horizon`
does not**: its 20k arm requests 655.4M tokens, so it would revisit training data at
1.32 epochs while its 10k partner sits at 0.66 — a second variable moving alongside
the one the pair exists to isolate. `--preflight` fails closed on this and prints the
prep command for a larger corpus; `--allow-data-repeat` accepts it instead and records
`data_epochs` per job so the repeat cannot go unreported.

### 3.5 GH200 capacity is no longer the deciding factor

An earlier version of this section advised waiting for an out-of-capacity GH200,
because that was the box suites 22–26 ran on and using it would save the SP re-run.
That saving is 10 jobs ≈ 1.1 GPU-hours ≈ $2–5. It is not worth waiting for. Run the
control.
