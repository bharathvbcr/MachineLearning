# Experiment backlog — 2026-08-26

Prioritized follow-ups to `PAPER_2026-08_Recipe_Dependent_Rankings.md`, recorded after a
gap review of the hybrid-mixer evidence. Two buckets, deliberately separate:

- **Paper-1 closers** (Tier 0–1): finish the claims the paper already makes. Named in the
  paper itself; nothing new here, only the run commands.
- **Paper-2 candidates** (Tier 2): experiments the *hybrid-architecture* literature would
  ask for, which the current evidence cannot answer. Each is an instance of the paper's
  own thesis — a recipe axis (metric, sequence length, ratio, cost basis) along which the
  §4.5 board has been measured at exactly one point.

Numbering continues `docs/ISSUES_AND_GAPS_2026-08-22.md` §2.2 (E1–E6 taken; E2 lives in
`docs/GPU_BUNDLE.md`). Costs are estimates unless a committed run record is cited.

---

## Tier 0 — E7: the seeded-attention 20k Metal run — **CLOSED, no compute needed**

**Resolved 2026-08-26 from the run archive. Do not run this.** Paper §6.6 called a seeded
attention run at 20k "the cheapest experiment named anywhere in this paper". It was cheaper
than that: it had already been run on 2026-07-12 as
`metal-native/out/sota_f32_clipsoft_seed42_20k_fa_tiled_softfix_warmdown_reseed`, and its
result was already quoted in the §6.6 table (1.8876) without being recognised as the seeded
arm. Note 53's "optional later: Soft attn **seeded** sota 20k for fair mingru compare"
(written 07-27) was listing an item closed 15 days earlier.

**Evidence that `_reseed` is seeded weights, not a reseeded data order.** The golden banks
are seed-agnostic, so golden arms are identifiable from step-0 logs alone:

| run | init | step-0 loss | step-0 `bank_qo` | FINAL EMA BPB |
|---|---|---|---|---|
| `…_seed1337_20k_…_warmdown` | golden | 6.932130336761475 | 22.612141133495648 | 1.896880 |
| `…_seed42_20k_…_warmdown` | golden | 6.932130336761475 | 22.612141133495648 | 1.892465 |
| `…_seed42_20k_…_warmdown_reseed` | seeded | 6.926790714263916 | 22.611394802120305 | **1.887607** |
| `smoke_seed42_reseed` | seeded | 6.926790714263916 | 22.611394798859113 | — |
| `smoke_seed1337_reseed` | seeded | 6.929978370666504 | 22.612562183281874 | — |

The two golden arms agree in *every* logged step-0 field across different seeds — the
signature of seed-agnostic banks under `METAL_NATIVE_DATA_SEED=0`, the standing convention
(`metal-native/DECISIONS.md:180`). The reseed arms carry seed-dependent weight statistics,
and the seed-42 smoke and 20k reseed runs reproduce each other's fingerprint. `bank_qo` is a
weight norm, which is what separates initialization from data order here.

**Readout — the crossing is not an init artifact.** Both arms seeded, same 20k recipe
(FA_TILED, Soft-split, `--warmdown 3500`; warmdown onset verified in both logs at step
16500–16600):

| quantity | BPB |
|---|---|
| minGRU seeded 20k (T12, seed 1337) | 1.993295 |
| attention seeded 20k (seed 42) | **1.887607** |
| **gap** | **0.1057** |
| initialization effect (golden 1.892465 → seeded 1.887607, both seed 42) | 0.0049 |
| backend nondeterminism (two golden arms, bit-identical at step 0) | 0.0044 |

The gap is ~22× either nuisance, and initialization points the *wrong way* — seeding helps
attention, so the confound was suppressing the crossing rather than creating it. §6.6 has
been updated from "suggestive" to a replication carrying two named caveats: the 20k seeded
pair is cross-seed (attention 42, minGRU 1337), and the arms are not parameter-matched
(attention 0.780M vs minGRU 0.977M, note 53). The parameter gap works against minGRU at 20k,
so it weakens minGRU's 3k lead rather than attention's 20k one.

**Separately: the Metal track cannot be re-run on this machine at all.**
`fineweb10B_sp1024` and its tokenizer `fineweb_1024_bpe.model` left with the
`parameter-golf` self-clone that commit `3e5aed5` deleted. Verified absent 2026-08-26 by a
whole-disk Spotlight sweep (the only `*.model` on the machine is an unrelated Step-Audio
tokenizer), by `~/Code` to depth 7, and in `~/Backups/parameter_golf-paper-20260823.zip`
(no `.bin`, `.model` or `datasets` entries); the `.bundle` and `archive/parameter-golf-clone`
branch carry git history only — the data was never tracked. `burn-port/token_bytes.json`
survives, but it is a **derived** LUT (`base_bytes` / `has_leading_space` /
`is_boundary_token`, indexed by token id), valid only for the tokenizer that produced it.

Restoring the original bins from an external copy is the only option that keeps new Metal
numbers comparable to the recorded board. Rebuilding is mechanically possible — 
`burn-port/scripts/export_token_bytes.py` pins the tokenizer contract (SentencePiece BPE,
vocab 1024, `▁` marker, byte-fallback, 4 control ids, confirmed against the surviving LUT)
and `metal-native/src/data.rs` pins the shard format (1024-byte header, magic 20240520,
version 1, then u16 LE tokens) — but a retrained BPE is a different tokenizer, so a rebuilt
corpus is an island of numbers comparable to nothing already recorded, and it needs a
regenerated `token_bytes.json` or every BPB is silently wrong. Rebuild only to unblock
*future* Metal work, and re-baseline the arms you care about on the new corpus if you do.

---

## Tier 1 — E1 / E2 / D10: the GPU bundle (GH200, already specified)

Fully specified elsewhere; listed here only for ordering. Runner:
`scripts/gpu_bundle.py`; design and defect history: `docs/GPU_BUNDLE.md`; cost is derived,
not typed:

```bash
python3 scripts/gpu_bundle.py --cost
```

This is the arm the paper says gates every crossing-token claim (§8.4) and the §4.5
board's confidence cap (E2). It outranks everything in Tier 2: **paper-2 experiments
built on the current board inherit its E2 cap.**

**Status 2026-08-27: the bundle is RUN.** 250 jobs, 33.21 GH200-hours, completed in an
earlier session; `--analyse` and `--report` both read clean. Two things came out of it that
are recorded below rather than here: §8.4's rows 1-3 are unanswered because every µP arm is
NOT A VALID COMPARATOR (**E13**), and the inherited-LR price needed a pricing rule the
runner did not have (**E14**).

**E2 is answered.** `e2_matched32_50m` supplies suite 26's two missing cells at 50M / batch
32, measured on the GH200 the board lives on rather than imported across hardware, which is
what §7.1 requires:

| arm | final val (95% CI) | n |
|---|---|---|
| `attention` | **4.2213** [4.2022, 4.2403] | 5 |
| `mingru` | 4.4491 [4.4228, 4.4755] | 5 |

Paired, minGRU is +0.2279 [+0.2119, +0.2438] behind on 5/5 seeds. These are the board's
endpoints, not the hybrid claim: §4.5's "the best hybrid is statistically indistinguishable
from pure attention" concerns `hybrid_mingru10_attn2`, which this suite does not run.

---

## Tier 2 — paper-2 candidates (hybrid-architecture claims)

The §4.5 conclusion — "the best hybrid is statistically indistinguishable from pure
attention" — was measured at one metric (held-out CE), one sequence length (512), one
recurrent:attention ratio per family, and one cost basis (token-matched). Each item below
varies exactly one of those. Do not chase the tie itself with more seeds at the current
recipe: by the paper's own thesis, a separation there would be a property of that recipe.

### E8 — recall probe: the metric axis, at the axis hybrids exist for

**Status 2026-08-29: RUN, 356 jobs across a 3 x 2 difficulty x budget grid. The headline it
was built to produce did not survive its own extension** — see the Result block below. The
probe works; what it measures is budget- and difficulty-dependent, and the p=4/3000-step
separation first reported from it has been withdrawn as a metric claim. Sections below marked
"blocking the run" are build history, retained because two of the three design changes they
forced are what made the grid interpretable.

Held-out CE at 512 tokens barely exercises in-context recall — the documented failure
mode of recurrent mixers (paper ref [5]; the MQAR line of work: Arora et al.,
"Zoology: Measuring and Improving Recall in Efficient Language Models",
arXiv:2312.04927) and the stated reason hybrids keep attention layers at all. The §4.5
tie may be a property of the metric. This is §6.2 (quality vs throughput reorders the
board) on a third metric.

- **Build:** a multi-query associative-recall (MQAR-style) generator + eval in `nanolab`
  (synthetic key-value sequences; report exact-match recall accuracy). Small models
  suffice in that literature — this tier runs on the Mac (MPS) or the 3070 Ti.
- **Arms:** the §4.5 board's five distinct families: `attention`, `mingru`, `gdn`,
  `hybrid_mingru10_attn2`, `hybrid_gdn_periodic`. Five seeds, matched recipe fingerprint.
- **Pre-registered readout:** if recall accuracy separates arms that CE tied (in either
  direction), the §4.5 tie is metric-dependent and must be reported per-metric. If recall
  reproduces the CE ordering, hybrids' recall story is not visible at this scale — also a
  result.

**Build status 2026-08-27: machinery done, probe NOT yet cleared to run.** `nanolab/mqar.py`
generates the task and scores exact-match recall; `train()` gained a `batchers=` seam so the
existing loop consumes it unchanged (corpus path untouched when None); answer positions are
masked with `ignore_index=-1`, which both loss paths in `model.py` already honour, so the
training loss *is* recall loss. Five tests in `nanolab.tests`, 93/93.

Two things had to be established before spending GPU, and both changed the design:

1. *The first generator was gameable.* The model scored 0.622 while "answer with the most
   recent value" scored 0.628 against that heuristic's own 0.633 ceiling — it had learned
   position, not recall. Querying every pair (Q = P) fixes it: the shortcut can only ever be
   right for one query. A probe left in the first state would have reported every arm tied
   at the shortcut ceiling and called it a metric result.
2. *The board's own default makes the probe blind.* With `tie_embeddings=True` attention
   caps at **0.555** and training does not move it (0.575 at 3k steps, 0.566 at 8k).
   Untied, the identical config reaches **0.990**. `qk_norm`, `zero_init_proj` and
   full-width RoPE were ablated in the same sweep and moved nothing (0.504–0.557).

   Under tying the readout matrix IS the input embedding, so the residual at a query
   position — carrying `embed(k)` — projects onto `k` itself; emitting `v` means cancelling
   that first. **E8 must run untied, and that is itself a §6 entry:** with the recipe every
   suite in §4 uses, the recall metric cannot separate the arms at all, because the
   reference arm sits at a ceiling that has nothing to do with recall. That is the paper's
   thesis applied to its own instrumentation.

**Still blocking the run, and the pre-registered readout has to change.** Two CPU findings:

*The difficulty window is narrow.* At P=8/Q=8/K=32/V=32 untied no arm solves the task —
attention 0.358, minGRU 0.318, hybrid 0.375, all within 0.06 of each other against a 0.149
shortcut ceiling. Reported as-is that reads as "recall reproduces the CE ordering", which is
one of the two pre-registered outcomes, and it would be **wrong**: it is a ceiling, not a
tie. The pre-registered table has no row for "too hard for every arm", and that state is
indistinguishable from a real null unless the reference arm is known to saturate. At
P=4/Q=4/K=16/V=16 attention does have headroom.

*Outcomes are bimodal, and initialization decides which.* At a fixed config, identical in
every respect but init, recall lands at 0.553 / 0.542 / **0.957**. With `set_seed` pinned the
same config reproduces exactly (0.903 three times), so the harness is deterministic and the
0.415 spread is genuine initialization sensitivity: the induction head either forms within
the budget or it does not, and training longer does not rescue a run that missed it (a
low-init seed sat at 0.566 after 8k steps).

This invalidates a single-run comparison outright. An early cross-arm probe read attention
0.801 / minGRU 0.559 / hybrid 0.553, which looks like exactly the predicted separation and
is **not evidence** -- each arm got one unpinned init, and the init spread is larger than the
gap.

**Result (2026-08-28/29, 360 runs, batch 256, untied, 15 seeds per cell).** The sweep was
extended along two axes -- task difficulty (`pairs` = 4/6/8) and training budget (3000 vs
9000 steps) -- and the extension **overturns the reading of the first cell.**

*The first board, which stands as a measurement.* p=4, 3000 steps:

| arm | seeds forming the head | rate [95% Wilson] | median recall |
|---|---|---|---|
| `attention` | **15/15** | 1.00 [0.80, 1.00] | 1.000 |
| `hybrid_gdn_periodic` | 9/15 | 0.60 [0.36, 0.80] | 0.999 |
| `gdn` | 7/15 | 0.47 [0.25, 0.70] | 0.720 |
| `hybrid_mingru10_attn2` | 5/15 | 0.33 [0.15, 0.58] | 0.562 |
| `mingru` | 1/15 | 0.07 [0.01, 0.30] | 0.560 |

**The claim first drawn from it is WITHDRAWN.** That board was read as "§4.5's tie is
metric-dependent: on recall, attention beats the hybrid with disjoint intervals." The
separation is real at that cell and reproduces; what is false is that it is a property of
the metric. It is a property of the *budget*, and the difficulty axis inverts it outright.

*The grid.* `attention` vs `hybrid_mingru10_attn2`, the §4.5 tied pair:

| pairs | 3000 steps | 9000 steps |
|---|---|---|
| 4 | 15/15 vs 5/15 -- **disjoint**, attention ahead | 15/15 vs 12/15 -- overlapping |
| 6 | 9/15 vs 8/15 -- overlapping | 15/15 vs 12/15 -- overlapping |
| 8 | 2/15 vs 12/15 -- **disjoint, REVERSED** | 12/15 vs 13/15 -- overlapping |

At the matched 3000-step budget, difficulty alone carries the pair from a disjoint
attention win, through a tie, to a disjoint *hybrid* win. Both endpoints would have been
publishable in isolation and they contradict each other. At 9000 steps every cell is a tie.

**The readout is an outcome the pre-registered table has no row for.** E8 pre-registered two
possibilities -- recall separates the arms (tie is metric-dependent) or recall reproduces the
CE ordering (recall story invisible at this scale). What happened is neither: recall yields
whichever ordering you select via difficulty and budget, and yields none once both arms are
adequately trained. Per the runner's own rule, that is reported as such rather than mapped
onto the nearest row.

**What survives the budget control -- corrected 2026-08-30, once the GDN arms completed.**
The fine-grained ordering among attention-containing arms does not survive. An earlier
version of this paragraph said the *recurrent/attention* split does. **That was wrong**, and
the completed `gdn` row is what refutes it: GDN is strongly budget-responsive and nearly
saturates at p=4. What does not respond to budget is `mingru` specifically, not recurrence.

p=4 (all five arms, both budgets):

| arm | 3000 steps | 9000 steps |
|---|---|---|
| `attention` | 15/15 | **15/15** [0.80, 1.00] |
| `hybrid_gdn_periodic` | 9/15 | **15/15** [0.80, 1.00] |
| `gdn` | 7/15 | **13/15** [0.62, 0.96] |
| `hybrid_mingru10_attn2` | 5/15 | 12/15 [0.55, 0.93] |
| `mingru` | 1/15 | **1/15** [0.01, 0.30] |

p=8 (all five arms, both budgets):

| arm | 3000 steps | 9000 steps |
|---|---|---|
| `hybrid_gdn_periodic` | 6/15 | **13/15** [0.62, 0.96] |
| `hybrid_mingru10_attn2` | 12/15 | **13/15** [0.62, 0.96] |
| `attention` | 2/15 | **12/15** [0.55, 0.93] |
| `gdn` | 2/15 | 6/15 [0.20, 0.64] |
| `mingru` | 0/15 | **0/15** [0.00, 0.20] |

*The one arm budget cannot rescue is minGRU.* It forms the head on **2 of 75 runs** across
every difficulty and budget in the grid -- 1/15 at p=4 at *both* budgets, 0/15 everywhere
harder. Tripling compute moves it by nothing. Every other arm improves substantially.

*Pure GDN is not in that category.* 7/15 -> 13/15 at p=4 and 2/15 -> 6/15 at p=8. A pure
recurrent stack with the delta rule does form induction heads given budget; a pure minGRU
stack does not, at any budget tried. **The distinction is the mixer's recall mechanism, not
recurrence versus attention**, and the earlier framing obscured exactly the thing worth
saying.

*The Qwen-shaped arm is the strongest arm on this metric.* `hybrid_gdn_periodic` -- 3 GDN
layers per attention layer, the 3:1 ratio the field converged on -- is joint-first at p=4
(15/15, level with pure attention) and first at p=8 (13/15, above attention's 12/15). It is
the only arm at or near the top at *both* difficulties under adequate budget.

*That is a CE/recall disagreement worth stating carefully.* On held-out CE
`hybrid_gdn_periodic` (4.290) loses to `attention` (4.221) by 0.069 nats with disjoint
intervals, and loses to `hybrid_mingru10_attn2` (4.232). On recall at 9000 steps it matches
attention exactly at p=4 and edges it at p=8. The honest claim is **not** that the ordering
reverses -- the recall intervals overlap in both cells -- but that a CE separation which *is*
disjoint does not reproduce on recall at all. A board that ranked these arms on CE would
under-rate the arm that best does the thing hybrids are built for.

*Where the GDN-vs-minGRU family comparison stands:* not established, and direction-unstable.
`hybrid_gdn_periodic` vs `hybrid_mingru10_attn2` runs 9/15 vs 5/15 (p4/3k), 15/15 vs 12/15
(p4/9k), 6/15 vs 12/15 (p8/3k, reversed), 13/15 vs 13/15 (p8/9k, identical). No cell is
disjoint and the sign flips with difficulty. Reported as unresolved rather than as the
inversion an earlier note called "suggestive".

*Attention is the arm most sensitive to budget, not the least.* 2/15 -> 12/15 at p=8 is the
largest single jump in the grid. Reading a short-budget recall board as an architecture
result reads attention as the *worst* arm at p=8, which the 9000-step cell shows is a
statement about the budget.

*Failure remains binary across seeds.* `hybrid_gdn_periodic`'s median recall is 0.999 at a
0.60 rate -- when the head forms, the task is solved outright. That is what justifies a rate
over a mean, and it is the shape the design was changed to expect.

**Consequence for the paper.** E8 does not supply a second metric that breaks §4.5's tie. It
supplies a *third and fourth recipe axis* -- task difficulty and training budget -- on which
the same fixed architectures change places, with disjoint intervals at both ends of the
difficulty axis. That is a stronger §6 entry than the one it replaces, and it is the paper's
own thesis reproducing on the instrument built to escape it.

**Design consequence.** "5 seeds, report recall accuracy" averages a bimodal distribution and
yields a number describing no model that exists. E8's readout must instead be the *fraction
of seeds that form the head* (recall above a threshold near the upper mode), which is a rate
and needs materially more than five seeds: at sd ~= 0.2, n=5 gives a CI half-width of ~0.25,
wider than the effect. Settle the seed count against the observed mode separation before
spending GPU.

**A recall board must report its budget and difficulty or it is not interpretable.** Neither
is a nuisance parameter here; each one alone reverses the board.

### E9 — sequence length: the recipe axis the hybrid literature cares about most

**Status: RUN — result below. The planning notes in this section predate the run and are
kept for the design rationale.** Originally: needs a small runner extension — `scale_to_token_budget()` already takes
`block_size`, but the suite stages pin 512; add a `CROSSOVER_BLOCK` env override in
`nanolab/crossover_replicate.py` alongside `CROSSOVER_BATCH` / `CROSSOVER_TOKEN_BUDGET`,
and thread it through job configs. `Config.block_size` and `grad_checkpoint` exist.

- **Arms:** `attention`, `mingru`, `gdn`, `hybrid_mingru10_attn2`, `hybrid_gdn_periodic`
  at `block_size` 2048 (optionally a 4096 wave with `grad_checkpoint=True`), batch scaled
  to hold tokens/step near the suite-26 cadence, 5 seeds, 50M tokens, 50M cosine.
- **Cost: unknown until a smoke run.** The committed medians that price the GPU bundle
  cover (mixer, batch, context=512) cells only; attention's per-token cost grows with
  context and the GDN/Mamba chunked kernels change regime. Run
  `python -m nanolab.crossover_replicate smoke` at the new shape first and price from
  measured `tok_s` — do not reuse the 512-context medians (the D14 lesson: the recipe's
  cells are not interchangeable).
- **Pre-registered readout:** does the §4.5 ordering survive a 4× context change? Any
  rank move is a new §6 entry (sequence length as a recipe axis); a stable board is the
  first evidence the tie generalizes.

**Status: RUN. Result recorded 2026-08-29** (`nanolab/out/crossover50m_ctx2048`, 5 seeds,
`block_size` 2048, 50M tokens). The status line above describes the state before the run and
is kept for the design notes; the runner extension it asks for was made.

Reported on `final_val` -- the end-of-schedule eval, recovered from `final.pt` -- not
`best_val`. See E11's third defect for why that distinction is not cosmetic. All arms here
fire 61 evals, so the `best_val` bias is common-mode and the deltas would stand either way.

| arm | final val (95% CI) | n |
|---|---|---|
| `attention` | **4.2004** [4.1628, 4.2379] | 5 |
| `hybrid_mingru10_attn2` | **4.2022** [4.1681, 4.2362] | 5 |
| `hybrid_gdn_periodic` | 4.2677 [4.2338, 4.3015] | 5 |
| `gdn` | 4.4772 [4.4395, 4.5149] | 5 |
| `mingru` | 4.4915 [4.4563, 4.5268] | 5 |

**Pre-registered readout: the ordering survives.** §4.5's tie between `attention` and
`hybrid_mingru10_attn2` holds at 4x the context -- 0.0018 nats apart, intervals almost
entirely coincident. This is the first evidence in the record that the tie is a property of
the architectures rather than of the 512-token recipe it was measured at, and it is a
*negative* result worth reporting as such.

*A tie, not an ordering.* The two arms swap places between `best_val` (4.1668 vs 4.1659) and
`final_val` (4.2004 vs 4.2022). Both margins are far inside seed noise, so E9 supports a tie
between them and licenses no ranking. The recurrent/attention separation is the part that is
unambiguous: both pure recurrent arms sit ~0.28 nats behind, intervals disjoint from every
attention-containing arm.

*Contrast with E8, deliberately.* On held-out CE at 2048 the tie generalizes; on the recall
probe the same pair separates or reverses depending on difficulty and budget. Both are
measurements of the same two architectures. That is the paper's thesis, not a contradiction
to resolve.


### E10 — minGRU hybrid ratio and placement sweep

**Status: RUN — result below. The planning notes in this section predate the run and are
kept for the design rationale.** Originally: runnable now — pure config; add four `Arm` entries to `ARMS` in
`nanolab/crossover_replicate.py` (syntax identical to the existing hybrids):

```python
Arm("hybrid_mingru11_attn1", "mingru", "mingru*11,attention",
    "how little attention is enough: 11+1"),
Arm("hybrid_mingru_periodic", "mingru",
    "mingru*3,attention,mingru*3,attention,mingru*3,attention",
    "Qwen-style every-4th-layer attention (9 minGRU + 3 attn)"),
Arm("hybrid_mingru_bookend", "mingru", "attention,mingru*10,attention",
    "attention at first and last layer"),
Arm("hybrid_mingru8_attn4", "mingru", "mingru*8,attention*4",
    "1:2 ratio upper arm"),
```

The board's best hybrid family (minGRU) exists at exactly one ratio/placement (10+2,
last-2), while GDN got three variants. The field's converged ratio (3:1 periodic) has
never been run on the family that actually ties attention. 4 arms × 5 seeds = 20 jobs at
the suite-26 recipe (bs32, `eval_iters=20`, 50M budget, 50M cosine), priced by the same
median-`tok_s` model as the bundle (minGRU-hybrid cells are covered; expect same order as
20 suite-26 jobs). Launch with `CROSSOVER_ARMS` set to the four new arms and
`CROSSOVER_JOB_PREFIX=cx32r`.

**Pre-registered readout:** if 9+3-periodic or 11+1 matches 10+2, the tie is
ratio-robust; if ratio reorders the family, "the best hybrid" was itself a
recipe-dependent claim and §4.5's row 2 needs a ratio qualifier.

**Status: RUN. Result recorded 2026-08-29** (`nanolab/out/crossover50m_ratio32`, 4 arms x 5
seeds, suite-26 recipe, `final_val`). The four `Arm` entries the status line above specifies
were added and run.

| arm | layout | final val (95% CI) | n |
|---|---|---|---|
| `hybrid_mingru_periodic` | 9 minGRU + 3 attn, every 4th | **4.1942** [4.1661, 4.2223] | 5 |
| `hybrid_mingru8_attn4` | 8 + 4 | 4.2020 [4.1747, 4.2294] | 5 |
| `hybrid_mingru11_attn1` | 11 + 1 | 4.2529 [4.2261, 4.2798] | 5 |
| `hybrid_mingru_bookend` | attn first and last | 4.2590 [4.2316, 4.2864] | 5 |

**Pre-registered readout: the tie is ratio-robust upward, and ratio does reorder the
family.** Within this suite the board splits cleanly in two: the 3-and-4-attention-layer arms
at ~4.19-4.20, and the 1-and-2-layer arms at ~4.25-4.26. `hybrid_mingru_periodic` beats
`hybrid_mingru11_attn1` with **disjoint intervals** -- three attention layers buy something
one does not. Inside each pair the intervals overlap almost completely, so the board supports
two tiers, not four ranks.

*The field's converged 3:1 periodic ratio is the best arm measured here*, and it ties pure
attention rather than beating it: the E2 reference at the same recipe and the same statistic
is 4.2186 [4.1917, 4.2454] (n=5), whose interval overlaps `hybrid_mingru_periodic`'s
throughout. Cross-suite caveat: that attention cell comes from `gpu_bundle/e2_matched32_50m`,
so it is same-recipe, same-hardware and same-statistic, but not side-by-side.

*More attention is not monotonically better.* `hybrid_mingru8_attn4` (4 layers) does not beat
`hybrid_mingru_periodic` (3 layers); the two are indistinguishable. Whatever the attention
layers are doing here saturates by three of twelve.

*A placement claim was drawn and is WITHDRAWN.* An earlier reading set
`hybrid_mingru_bookend` against `hybrid_mingru10_attn2` -- both 2 attention layers, different
positions -- and called the 0.029 gap a placement effect. That comparison is invalid: the
10+2 cell lives in `crossover50m_matched32`, whose checkpoints were pruned, so its
`final_val` is unrecoverable and only its `best_val` exists. The two numbers were a
cross-suite *and* cross-statistic pair.

**RESOLVED 2026-08-29 (`crossover50m_ratioplace32`, 5 arms x 5 seeds, within-suite,
`final_val`).** `lock_recipe` refuses a fifth arm in `ratio32` -- its `recipe.json` records a
four-arm list and the guard exists to stop two arm sets being blended -- so all five arms were
re-run in a fresh suite at the identical recipe. Seed variance is common-mode, so the powered
test is paired; every row is 5/5 seeds, interval disjoint from zero:

| comparison | paired delta |
|---|---|
| 9+3 vs 10+2 | -0.0375 [-0.0426, -0.0323] |
| 9+3 vs 8+4 | -0.0077 [-0.0118, -0.0036] |
| 9+3 vs 11+1 | -0.0591 [-0.0641, -0.0541] |
| 10+2 vs bookend (both 2 attn layers) | **-0.0283** [-0.0307, -0.0259] |

Placement is real and independent of count: two arms spending exactly two attention layers
differ by 0.0283 nats on every seed, purely in where the layers sit. Count is not monotone:
3 attention layers beat 4. The pre-registered readout fired in its second branch -- ratio
reorders the family, so §4.5's row 2 needs a ratio qualifier.

*The withdrawn estimate was 0.029; the valid one is 0.0283.* It was withdrawn because the
comparison was invalid, not because the number was wrong. A right answer by an invalid route
is not a result, and the near-match is luck rather than vindication.

*Free reproducibility check.* The four previously-run arms reproduce their `ratio32` values to
within 0.0006 nats (periodic 4.1942 -> 4.1939, 11+1 4.2529 -> 4.2530).


### E11 — wall-clock-matched board: the cost basis practitioners actually use

**Status: phase 1 is free (analysis only); phase 2 needs runs.**

The §4.5 board is token-matched. §6.4 shows hybrid *throughput* reorders with batch size;
practitioners adopt hybrids because at matched wall-clock they see more tokens. That
recipe axis is unmeasured.

- **Phase 1 (no GPU):** from the committed per-job `metrics.jsonl` of suite 26, compute
  each arm's median `tok_s`, then re-read every arm's loss curve at the *token count the
  slowest arm reaches in the same wall-clock*. Report the re-indexed board next to the
  token-matched one. Caveat to carry: suite 22/26 throughputs are contended
  (2 jobs/GPU, documented in E6 detail) — the re-indexing is only valid within-suite,
  same-concurrency.
- **Phase 2 (GH200):** true matched-wall-clock runs for the top four arms (each arm runs
  the same minutes, not the same tokens), 5 seeds, single-tenant, with `elapsed_s` now
  persisted (E6 fix). Only worth pricing after phase 1 shows a rank move.

**Phase 2 first attempt failed and is retained as `crossover_wallclock32_unmatched`.**
It ran all 20 jobs, produced a board that looked publishable, and did not measure what
it claimed: against a 691s target the arms landed at 387.8s (attention) to 661.1s
(`hybrid_gdn_bookend`), a 1.70x spread in the single quantity the suite exists to hold
constant. Two compounding causes, both now closed in code:

1. *Tenancy was not a recorded quantity.* Budgets were sized from throughput scraped
   from suites that had run 3-to-a-GPU, then executed single-tenant. Arms do not recover
   from contention uniformly — attention and `hybrid_mingru10_attn2` gained 1.78x and
   1.72x, the two GDN arms only 1.04x and 1.05x — so a rate measured at one tenancy does
   not transfer. `workers` is now a recipe field, `measured_rate_by_arm(tenancy=…)`
   filters on it, and suites whose tenancy cannot be established are skipped rather than
   assumed.
2. *Step rate is not wall-clock rate.* Sizing from per-step `tok_s` ignores eval,
   checkpoint and startup time, and that overhead is arm-specific: single-tenant,
   attention realises 81.0% of its step rate over a whole run against 89.4% for
   `hybrid_gdn_bookend`. `effective_rate_by_arm` (tokens / `elapsed_s`) is what sizes a
   budget now.

The retry is sized from the failed run's own single-tenant `elapsed_s`, which is the only
tenancy-1 measurement in the repository — the reason that suite is archived rather than
deleted. `verify_wallclock` now checks the realised clock against the target and
`wcboard` refuses to emit a board outside 5%, so this failure cannot recur silently.

**Result (2026-08-27, 20 jobs, single-tenant, 3.84 GPU-h).** The retry held its clock:
every arm landed within +2.2%/-1.0% of the 691s target, against the 5% tolerance.

| # | arm | budget | elapsed | final val (95% CI) |
|---|---|---|---|---|
| 1 | `attention` | 73.2M tok | 684.1s (-1.0%) | **4.1045** [4.0727, 4.1363] |
| 2 | `hybrid_mingru10_attn2` | 60.2M | 686.5s (-0.6%) | 4.1640 [4.1489, 4.1791] |
| 3 | `hybrid_gdn_periodic` | 22.0M | 704.3s (+1.9%) | 4.8413 [4.8206, 4.8620] |
| 4 | `hybrid_gdn_bookend` | 20.0M | 706.2s (+2.2%) | 4.9312 [4.9146, 4.9479] |

Attention wins every pairing 5/5 on paired seeds; it beats the best hybrid by 0.0595 with
disjoint intervals. This is the opposite of the usual case for hybrids, which is precisely
that they win once the budget is time rather than tokens. At this scale, on this recipe,
single-tenant, the minGRU hybrid's 1.19x throughput does not cover its worse loss per
token, and the GDN arms are not close — they buy 22.0M and 20.0M tokens against
attention's 73.2M in the same 691 seconds. Scale- and recipe-bound, as the paper's own
thesis requires.

**A third defect, found in the board rather than the run.** `wcboard` read `best_val` from
the terminal record while its own docstring claimed it read the end of the schedule.
`best_val` is a running minimum over however many evals fired, and a minimum over more
draws sits lower. Eval fires on a fixed step interval, so the fast arm takes more steps,
draws more evals, and gets a lower minimum — a bias that grows with exactly the throughput
advantage the suite exists to measure. Measured on this suite:

| arm | evals | `best_val` - `final_val` |
|---|---|---|
| `attention` | 89 | -0.0133 |
| `hybrid_mingru10_attn2` | 73 | -0.0087 |
| `hybrid_gdn_periodic` | 26 | -0.0039 |
| `hybrid_gdn_bookend` | 24 | -0.0032 |

Monotone in eval count, and pointing the same way as the result. It cost 0.0046 of the
0.0640 attention-vs-hybrid gap; the corrected gap is 0.0595 and the ranking is unchanged,
but the statistic was not defensible. `train.py` computes a true end-of-schedule eval and
had always written it to `final.pt` without logging it; it is now in the `done` record as
`final_val`, `_final_by_seed` reads that (falling back to `final.pt` for older runs), and
it raises rather than substituting `best_val` — a silent substitution is the confounded
board. The 20 runs here predate the logging fix, so their board is recovered from
`final.pt`; no GPU time was rerun.

*Blast radius across published suites.* `crossover50m_matched32` and
`crossover50m_ratio32` run every arm for 61 evals, so the bias is common-mode and their
deltas stand. `crossover50m_ctx2048` (E9) is likewise uniform at 61, bias -0.029 to
-0.036; its one casualty is that `attention` and `hybrid_mingru10_attn2` swap order
between the two statistics (best: 4.1668 vs 4.1659; final: 4.2004 vs 4.2022) — both
margins are far inside noise, so E9 supports a tie between those two arms and no ordering.
`crossover50m` is the open item: its arms split 61/63 evals, and its checkpoints have been
pruned, so `final_val` is unrecoverable there and the residual differential can only be
bounded (~0.001, against ~0.004 of arm-to-arm bias variation at equal eval count) rather
than measured. Any published `crossover50m` gap under ~0.005 between a 61-eval and a
63-eval arm should not be leaned on.

**Complete `final_val` recoverability audit, 2026-08-29.** Prompted by pulling results off
the rented GH200 before teardown. 471 `done` records across eleven suites predate the
`final_val` logging fix, so their end-of-schedule loss existed only inside `final.pt` --
869 GB of checkpoints on a rented box, which is not a place results live. All 479 `final.pt`
on that machine were read and their `val_loss` extracted (479/479, zero failures) into
per-run `final_val.json` sidecars plus a consolidated
`nanolab/out/final_vals_recovered.json`. Validated against the E11 board above: the five
recovered `crossover_wallclock32` attention seeds mean **4.1045**, matching the published
figure exactly.

| suite | runs without logged `final_val` | status |
|---|---|---|
| `gpu_bundle` | 250 | recovered |
| `crossover50m_ctx2048` | 25 | recovered |
| `crossover50m_ratio32` | 20 | recovered |
| `crossover_wallclock32` | 20 | recovered |
| `crossover_wallclock32_unmatched` | 20 | recovered |
| `crossover50m` | 50 | **unrecoverable** -- no `final.pt` |
| `crossover50m_matched32` | 40 | **unrecoverable** -- no `final.pt` |
| `funnel` | 16 | **unrecoverable** -- no `final.pt` |
| `crossover20m_locked` | 10 | **unrecoverable** -- no `final.pt` |
| `crossover20m_matched_lr` | 10 | **unrecoverable** -- no `final.pt` |
| `crossover8m_bs8` | 10 | **unrecoverable** -- no `final.pt` |

The consequence worth carrying: **§4.5's own board (`crossover50m_matched32`) is in the
unrecoverable set.** Its arms all fire 61 evals, so the `best_val` bias is common-mode and
its published *deltas* stand -- but its absolute end-of-schedule losses cannot be produced,
and it cannot be set against any suite reported on `final_val` without re-running. That is
what blocks the E10 placement comparison (see E10's withdrawn claim), and it is the reason
E9 and E10 are reported on `final_val` while §4.5 is not.

Results were synced off the instance at 24 MB (metrics, recipes, logs, recovered values);
the 869 GB of checkpoints were deliberately not copied. For models this size, re-running a
suite costs less than storing its weights.

### E12 — sliding-window attention arm: must the hybrid's attention be global?

**Status: CODE COMPLETE 2026-08-31, unrun.** Superseded in detail by
`docs/SWA_BOARD_2026-08-31.md`, which specifies E12 together with two new items it
motivated — **E15** (the same arms at context 2048) and **E16** (MQAR across sequence
lengths that straddle the window). Both arms this entry asked for exist: `swa` (as
`swa_w64/w128/w256`) and `hybrid_mingru10_swa2`. Run all three with
`python -m nanolab.crossover_replicate swaboard`; ~285 runs, 33–90 GPU-h, $152–414.

The gating condition below is now met: E8 exists (360 runs), and E16 extends it to the
sequence lengths at which a window can actually bind — E8's own grid runs at `block_size`
15, which a 64-wide window spans entirely, so `swa` there IS `attention`.

The prompt for this was arXiv:2608.28444 (*Sliding-window beats linear attention*). Note
that paper is training-free masking of a **pretrained** model against a **post-trained**
linear retrofit; E12/E15/E16 pretrain from scratch and answer only the from-scratch
analogue. `docs/SWA_BOARD_2026-08-31.md` scopes what closing the rest would take, and
recommends against starting it as part of this paper.

**Original entry (2026-08-26), for provenance:** `nanolab/mixers.py` has no
windowed-attention path (verified: only the semi-AR block-window decode helper). Smallest honest version: a `window_size`
option on the existing attention mixer implemented as an attention mask, then two arms —
`swa` (all-12 windowed) and `hybrid_mingru10_swa2` — against their global-attention
twins. At 512 context use window 128 so the window binds.

This is the small-scale echo of the sparse-attention-inside-hybrids question
(Qwen3.8-Flash-Next's QSA design): if windowed layers match global ones inside the
hybrid, the hybrid's attention layers are doing local work; if they don't, the global
retrieval story earns its cost. Lowest priority: the result is only interpretable after
E8 exists, because CE at 512 cannot see the retrieval difference windowing removes.

### E13 — µP's attention temperature: why every µP cell is not a valid comparator

**Status: diagnosed 2026-08-27; ablation queued as `e1_mup_spattn` (10 jobs, ~1.2 GPU-h).**

The Tier-1 bundle completed (250 jobs, 33.21 GPU-h) and `--analyse` returned §8.4 rows 1-3
*unanswered*, because the µP arms **never crossed on any seed** — an outcome the
pre-registered table has no row for. The reason is in its own competitiveness check:

| suite | control | d attention | d minGRU | spread | verdict |
|---|---|---|---|---|---|
| `e1_mup_tuned` | `e1_sp_rerun` | +0.3537 | +0.0524 | 0.3014 | NOT A VALID COMPARATOR |
| `e1_mup` | `e1_sp_rerun` | +0.5446 | +0.0810 | 0.4636 | NOT A VALID COMPARATOR |
| `e1_mup_sched20` | `e1_sp_sched20` | +0.5724 | +0.3101 | 0.2623 | NOT A VALID COMPARATOR |
| `e1_mup_bs8` | `e1_sp_bs8` | +0.2721 | +0.0991 | 0.1730 | NOT A VALID COMPARATOR |
| `e1_perlayer_sp` | `e1_sp_rerun` | -0.0754 | -0.0688 | 0.0066 | ok |
| `e1_embed_lr` | `e1_sp_rerun` | -0.0634 | -0.0375 | 0.0258 | ok |

Every µP suite hurts attention 4-7x more than minGRU. The two non-µP parametrizations are
even-handed to within 0.03, so the asymmetry is µP's, not the harness's.

**It is not the learning rate.** `e1_mup_tuned` re-tunes at the target width off
`e1_mup_basin`'s five-point curve and attention is still +0.354. An LR explanation is ruled
out by the bundle's own design.

**It is the attention temperature.** µP prescribes `1/d` attention logits where SP uses
`1/sqrt(d)`, which is correct only when `q.k` grows as Θ(d). Every `nn.Linear` here inits at
a fixed `std=0.02`, so `q.k` grows as Θ(sqrt(d)) and the `1/d` rule over-cools by exactly
`sqrt(head_dim)`. Measured through a real forward pass at `d_model=768`, `head_dim=64`:

| | logit scale | logit std | attention entropy |
|---|---|---|---|
| SP | 0.12500 | 1.000 | 89.0% of uniform |
| µP | 0.01562 | 0.125 | **99.8% of uniform** |

µP attention begins training as very nearly an averaging layer. minGRU has no attention
logits, so the term is arm-asymmetric by construction — the exact shape of the table above.
The effect is present at `width_mult == 1` too (0.18x at `head_dim=32`): `_mup_init` is
skipped at base width but the `1/d` scale is not, so `e1_proxy` tuned attention inside the
over-cooled regime as well.

Asserted by `nanolab.tests.mup_attention_temperature_is_the_arm_asymmetric_term`, which
measures through a full forward pass — the effect does not reproduce on synthetic input,
because the attention input is a normed hidden state rather than a raw Gaussian.

**Pre-registered readout for `e1_mup_spattn`** (µP at the transferred LR, SP's `1/sqrt(d)`
temperature, one term changed): if attention's deficit against `e1_sp_rerun` collapses from
+0.354 toward minGRU's +0.05, then §8.4's µP cells measured a broken attention temperature
rather than µP, and every µP row in the paper needs that caveat. If it does not collapse,
the handicap is elsewhere in the parametrization and `1/d` is exonerated.

**Result (2026-08-27, 10 jobs, ~0.6 GPU-h). The readout fired.** `e1_mup_spattn` differs
from `e1_mup` in exactly one term — the attention logit scale — at the same transferred
`matrix_lr` of 0.0016:

| suite | d attention | d minGRU | spread | verdict |
|---|---|---|---|---|
| `e1_mup` | +0.5446 | +0.0810 | 0.4636 | NOT A VALID COMPARATOR |
| `e1_mup_tuned` | +0.3537 | +0.0524 | 0.3014 | NOT A VALID COMPARATOR |
| **`e1_mup_spattn`** | **+0.1299** | **+0.0810** | **0.0489** | **ok** |

Attention recovers 0.4147 of its 0.5446 deficit (76%) and the arm asymmetry falls 9.5x.
This is the first µP variant to pass the bundle's own comparator test.

**minGRU is the control that makes it an attribution rather than a correlation.** Its delta
is unchanged to four decimals, +0.0810 -> +0.0810. The ablation touches only the attention
logit scale; minGRU has no attention logits; it does not move. Nothing else about the
parametrization changed, so the 0.4147 nats belong to that one term.

**Consequences for the paper.** §8.4's µP cells were measuring an attention temperature
error, not µP. The three unanswered rows are unanswered for that reason: the µP arms never
crossed because attention under µP began training as very nearly an averaging layer. Every
µP row needs the caveat, and the residual +0.1299 / +0.0670 is the honest remaining µP
question. `e1_mup_basin`'s "TRANSFER MISSED: the interior minimum is at 4x (attention) /
2x (minGRU), not 1.0x" should be re-read in this light: a transfer measured through a
broken temperature is not evidence about µP transfer.

**The crossing returns.** §8.4 rows 1-3 were unanswered because the µP arms never crossed
on any seed. With the temperature corrected they do, at both schedules:

| cell | n | seeds crossing | first (M) | last (M) |
|---|---|---|---|---|
| `e1_mup` (s24) | 5 | **0** | - | - |
| `e1_mup_tuned` (s24) | 5 | **0** | - | - |
| `e1_mup_spattn` (s24) | 5 | 5 | 10.12 [9.93, 10.30] | 10.12 [9.93, 10.30] |
| `e1_mup_sched20` (s23) | 5 | **0** | - | - |
| `e1_mup_sched20_spattn` (s23) | 5 | 5 | 2.77 [1.50, 4.05] | 3.58 [3.47, 3.70] |

At s24 first and last coincide: one clean crossing per seed, where SP crosses at 1.05M and
again at 12.37M. So "µP never crosses" was a statement about the temperature, not about µP.

**The pre-registered rows are deliberately NOT repointed at these cells.** Rows 1-4 say
*under µP*, and `*_spattn` is µP with SP's attention temperature -- a different
parametrization. Mapping a modified arm onto a row written for µP is exactly the error the
runner warns about when it says an outcome "is not in the pre-registered table. Report it
as such rather than mapping it onto the nearest row." The rows stay unanswered; what
follows is a separate result.

**What the corrected arms say, reported separately.** The 20M cosine crosses *earlier* than
the 50M cosine (3.58M vs 10.12M), and the two do not collapse onto a common token. Under SP
on the same box the ordering runs the other way -- `e1_sp_sched20` last-crosses at 14.74M
against `e1_sp_rerun`'s 12.37M. Whether that inversion is a property of µP or of the
temperature correction is not established by these ten jobs, and should not be claimed from
them.

**The transfer miss survives the correction — it is a real µP result.** `e1_mup_basin_spattn`
(42 jobs) re-measures the target-width LR curve with SP's attention temperature. The interior
minimum sits at **4x for attention and 2x for minGRU — the same multipliers as the uncorrected
basin**. What the correction changed was quality, not transfer: attention's whole curve drops
~0.42 nats (4.685 vs 5.103 at the minimum) while the argmin does not move.

| rule | error vs measured optimum | verdict |
|---|---|---|
| `optim.py` `matrix_lr / width_mult` (Adam-derived) | off by [4x, 2x] | **same direction on every arm — systematically wrong** |
| Muon, no divisor | off by [1.333x, 0.667x] | straddles 1.0 — within a factor-2 grid, unbiased |

This measures the extrapolation `optim.py` already flags in prose: the µP hidden-LR exponent
is derived for pure Adam, while 2-D hidden matrices here go to Muon, "whose update is
normalized by construction". Applying the Adam divisor to the Muon group is wrong in a
consistent direction, and the no-divisor rule predicts the target-width optimum.

*Built-in control.* minGRU's corrected curve is identical to the uncorrected one in every
digit (5.145101 / 5.086472 / 5.016853 / 4.993277). `mup_sqrt_attn_scale` touches only the
attention mixer and pure minGRU has no attention layers, so the model is bit-identical at the
same seed. The analysis reads only `*_spattn` records, so those numbers come from genuinely
re-run jobs — the strongest available form of "the ablation moved only what it should".

The corrected basin is reported **beside** `e1_mup_basin`, never in place of it: the
uncorrected verdict is a real measurement of that parametrization, and `*_spattn` is
µP-with-SP-attention rather than µP, so its minimum answers a different question than rows
1-4 asked.

---

### E14 — pricing an inherited LR when the argmin will not resolve

**Status: implemented and tested 2026-08-27.**

`e1_sp_basin` at five seeds located attention's optimum (5/5 both neighbours) and priced the
inherited 0.025 at **+0.205450 [+0.198964, +0.211935]** nats. It could not locate minGRU's:
0.003125 and 0.00625 sit **0.0024 apart on a 4/5 sign test**, and no seed count fixes a
basin that flat. `--analyse` therefore refused to price minGRU at all.

That refusal conflated two questions. *Which LR is best* is genuinely unresolvable here.
*What does the inherited LR cost* is not: both tied candidates beat 0.025 on every seed and
price it to within 2.0%. `analyse_sp_basin` now prices across the tied set and reports the
**smallest** of the candidate prices, so the ambiguity can only cost confidence, never buy a
larger number than the data supports:

```
mingru: the inherited 0.025 is 4-8x its own optimum and costs AT LEAST
        +0.123201 [+0.120739, +0.125664] nats (paired, n=5)
    argmin unresolved -- 0.003125 / 0.00625 are tied inside the seed spread;
    each beats the inherited value on every seed and they price it to within
    0.002414 nats (2.0%)
```

It fails closed on both ways this can go wrong, and those guards are the point (tests
S16-S19): if any tied candidate fails to beat the inherited value on **every** seed, or if
the candidates disagree about the price by more than `PENALTY_ROBUST_FRAC` (10%), the
unresolved argmin *is* the answer and nothing is reported. Both arms of §8.4's inherited-LR
question are now priced by the harness rather than by hand.

---

## Explicitly not planned, and why

- **A Titans / test-time-memory mixer.** Referenced in reading notes only; implementing
  it is a large lift orthogonal to both papers. The stale README claim that a
  `train_gpt_sprint_core.py` carried "optional MicroTitans memory" is corrected in the
  README itself (the file is not in this repository).
- **More seeds to separate attention from `hybrid_mingru10_attn2` at the suite-26
  recipe.** A separation found that way would be a property of that recipe — the paper's
  own thesis. E8/E9/E10 vary the recipe instead.
- **Further optimizer-axis work.** D7 closed with "retired, not reversed"; that is the
  stopping point.

## Suggested order

**Superseded as of 2026-08-29 — recorded for provenance.** The original plan read: E7 is
closed; Tier 1 bundle → E10 (config-only, cheapest GPU item) → E8 (Mac-side build while GPU
runs) → E9 → E11 phase 1 (free, anytime) → E11 phase 2 / E12 as results warrant.

**Executed:** E7 closed from the archive; Tier 1 bundle (250 jobs); E2; E9; E10; E11 phases 1
and 2; E13; E14; E8 (360 runs across difficulty x budget). Any further Metal-track work is
still gated on restoring `fineweb10B_sp1024` from an external copy, not on compute.

**Remaining, in the order they earn their cost:**

0. **E12 / E15 / E16 — the sliding-window board.** Code complete and tested 2026-08-31,
   unrun. One command (`swaboard`), 285 runs, 33–90 GPU-h, $152–414, spec and cost basis
   in `docs/SWA_BOARD_2026-08-31.md`. Run the probe phase first: on CUDA an explicit
   `attn_mask` loses the flash kernel, and a math-path fallback at ctx 2048 would OOM.

1. **§8.4's µP rows 1-4 — a decision, not an experiment.** Still unanswered. E13 established
   the µP cells measured a broken attention temperature; `*_spattn` is µP-with-SP-attention
   and deliberately is not mapped onto rows written for µP. Either fix µP properly (pair the
   `1/d` rule with the q/k init that makes it correct) or restate the rows in terms of the
   arm actually run. Costs nothing to decide; blocks the paper's headline caveat.
2. ~~**E8 option C**~~ **DONE 2026-08-30** (45 runs, arm-parallel, 3.7 h wall-clock instead of
   the 8.6 h serial estimate; the arms are independent and E8's readout is a rate over steps,
   not wall-clock, so contention cannot affect it). **minGRU is the only arm budget does not
   move:** 1/15 -> 1/15 at p=4 and 0/15 -> 0/15 at p=8, while attention goes 2/15 -> 12/15,
   gdn 7/15 -> 13/15 and hybrid_mingru10_attn2 5/15 -> 12/15.

   *This refuted a claim we had already written into the paper.* Section 6.8 read the surviving
   effect as attention-containing vs not. `gdn` has no attention layers and reaches 13/15 at
   p=4/9000 -- above the minGRU hybrid's 12/15 -- so recurrence is not the problem and the
   limit is specific to minGRU's gating. Both the section and its withdrawn predecessor are
   now recorded (paper section 7.3 items 7-8).
3. **A scale ladder.** The loudest reviewer objection is that everything here is one small
   scale. Two or three model sizes would convert it into a measured quantity. Unpriced.
4. **E10 placement, properly.** Re-run `hybrid_mingru10_attn2` inside `crossover50m_ratio32`
   so bookend-vs-10+2 is a within-suite, `final_val` comparison. 5 runs.
5. **E12** (sliding-window attention) — unchanged, still needs code, still lowest priority.

**Not worth running:** further confirmatory cells of the E8 grid (p=6 @ 9000 for the three
non-headline arms, p=4 @ 9000 for the headline pair). Every 9000-step cell measured so far is
a tie; these would confirm, not decide.
