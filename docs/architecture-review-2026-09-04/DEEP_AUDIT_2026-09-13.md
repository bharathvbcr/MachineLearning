# Deeper architecture and evidence audit — September 13, 2026

**Verdict: the record supports useful, recipe-dependent architectural results, but
does not establish a new attention-scale primitive. The most promising next
investigation is the division between recurrent computation and reliable global
retrieval. The immediate prerequisite is a trustworthy experiment reader and
resumable training contract.**

This audit follows the September 4 discussion and September 8 mechanism audit.
It incorporates newer experiments, challenges the audit tools themselves, and
narrows several conclusions in the current manuscript. Its qualifications govern
current interpretation of the older review and protocol. Historical records and
registered verdicts remain intact.

The most consequential new observation is a reversal across objectives: periodic
9+3 has the better language-model endpoint, but solves **0/10** seeds in the
sequence-511 recall assay; late 8+4 and full attention each solve **8/10**. That
supports investigating *where, how much, and under what load* global retrieval is
needed. It does not isolate attention placement or demonstrate recall loss in the
language-model checkpoints themselves: the recall models were trained separately.

## Scope, evidence grades, and reproducibility

**Verified** means recomputed from accessible records, reproduced on a stated CPU
fixture, or derived for an explicit mathematical model. It does not mean training
was independently repeated. **Inferred** marks interpretation; **proposed** marks
future work; **unverified** marks claims outside the checks performed here.

The MLSystemsLab checkout was clean at the start, at
`2ee4c135d969345eb035a24f7a5755e3cdc0cd86`. The primary repository map contained
3,148 files and matched that HEAD. The separately inspected BINN checkout was at
`e3299f84430ca26dc39099a569132f6cbffcb289`; it was read without mutation.

| Current coverage | Verified count or boundary |
|---|---|
| NanoLab result inventory | All 1,512 JSONL files currently under `nanolab/out`; 1,501 are `metrics.jsonl`. |
| Terminal records | 1,502 `done` records; 974 metric files have exactly one finite `done.final_val`. These are different quantities. |
| Repeated lifecycle records | 29 metric files have multiple starts; eight have multiple terminal records. They are not automatically independent runs or necessarily corrupt. |
| Parsing | No malformed JSONL or nonfinite top-level numeric fields found in this inventory. Nested arbitrary payloads were not exhaustively validated. |
| Recall inventory | 1,281 rows, 1,122 distinct run-name strings; 30 names have conflicting recall values across ledgers. Names alone are not an experiment identity. |
| Recomputed comparisons | 20 explicitly selected paired LM contrasts; three scoped MQAR ledgers; all 48 BINN W29 measurement cells. This is not every possible contrast in the inventory. |
| Provenance | SHA-256 for 3,013 NanoLab metric/config files, 17 audit/source files, and 48 W29 cells. |
| Source review | Training/resume/evaluation, batching, MQAR construction and scoring, optimizer scaling, arm identity, comparison readers, and relevant campaign scripts. Not a fresh line-by-line audit of all 3,148 mapped files. |
| Literature | Targeted primary-source review of seven particularly close papers. Not an exhaustive novelty search or proof of priority. |

The [saved machine-readable results](evidence/deep-audit-results-2026-09-13.json)
contain seed deltas, configuration differences, paths, counts, and hashes. The
[read-only probe](evidence/deep_audit_probe.py) rejects duplicate selected arm/seed
records, missing planned comparison seeds, nonfinite terminal losses, and folded
token curves. It uses exact shared evaluation markers and t critical values for
the actual sample size. Its crossing estimate is explicitly the late recovery
after an initial learning advantage: an earlier warmup reversal is recorded and
allowed, but a reversal after recovery is rejected.

The [validation log](evidence/DEEP_AUDIT_VALIDATION_2026-09-13.md) records successful
checks, failures, and omitted checks. GitPulse could inspect the main checkout but
could not inspect two linked worktrees because they were not trusted; collision
coverage is incomplete. No GPU campaign, cloud job, historical checkpoint replay,
or model-source repair was performed. The resume diagnostic uses six-step CPU
training fixtures in temporary directories.

## 1. A capability result changes which hybrid deserves priority

### 1.1 The longer recall assay separates layouts

**Verified:** at 3,000 steps, the current scoped ledgers contain:

| Architecture | Sequence 255: p64, batch 64 | Sequence 511: p128, batch 128 |
|---|---:|---:|
| Attention | 11/15 solved | 8/10 solved |
| minGRU 8 + attention 4, late | 15/15 | 8/10 |
| minGRU 9 + attention 3, periodic | 14/15 | **0/10** |
| minGRU 10 + attention 2, late | 13/15 | **0/8 completed; 10 planned** |
| Pure minGRU | 0/15 | 0/10 |
| Repository GDN | 0/15 | 0/10 |

Source: [sequence-255 ledger](../../nanolab/out/mqar_e16/runs.jsonl),
[sequence-511 ledger](../../nanolab/out/mqar_e16_seq511/runs.jsonl), and
[G4 stage specification](../../scripts/g4_hybrid_seq511.sh).
The solve threshold is recall ≥ 0.8. Periodic seeds 6 and 8 reach approximately
0.6500 and 0.6885; “0/10 solved” does not mean zero accuracy on every seed.
The unfinished 10+2 cells are seeds 3 and 7, not two measured failures.

For sequence 511, periodic versus attention has zero favorable discordant seeds,
eight unfavorable, and two ties: exact two-sided paired binary p = **0.0078125**.
The late 8+4 comparison has two favorable, two unfavorable, and six ties: p = 1.0.
Equal aggregate 8/10 rates therefore hide different failed seeds and do not prove
equivalence. These exploratory p-values are not multiplicity-adjusted across all
architectures and task cells.

**Inferred:** late 8+4 is the better starting reference for a capability-focused
follow-up. Periodic 9+3 should remain the CE reference and an informative failure
case. Neither deserves an unconditional “best architecture” label.

**Unresolved:** this is not a clean positional intervention. The layouts differ
in both attention count and position. The task change jointly increases pairs,
sequence length, batch size, and vocabulary-related demands. Distance, load,
optimization, and attention allocation have not been separated. The expansion-1
LM variants in §2 have not been tested in this long-recall ledger.

The [current generator](../../nanolab/mqar.py#L124) also feeds previous query
answers back as later input tokens. It is a teacher-forced recall assay, not a
query-only memory-read stream. Use it for continuity with the existing results;
use a separately identified, trained query-only task to test whether reads
damage memory. The current suite's `run_one` returns metrics without saving model
weights, so intervention/replay studies need checkpoint capture before training.
The [September 8 plan](SECOND_AUDIT_PLAN_2026-09-08.md) specifies those controls;
its [September 14 amendment](SECOND_AUDIT_PLAN_2026-09-08.md#7-amendment--september-14-2026)
reconciles that plan's stage order and prerequisites with this audit.

### 1.2 The paper's noninferiority conclusion is unsupported

The [current manuscript, §6](../../PAPER_2026-09_The_Crossing_Scales.md#L695)
correctly reports the sequence-255 point estimates and paired signs, but then
uses overlapping Wilson intervals and nonsignificant superiority tests to claim
the hybrids are at least as good and retain capability without cost.

**Verified statistical correction:** overlap between two marginal intervals is
not a paired noninferiority test. Failure to establish superiority does not
establish noninferiority. A suitable test needs a justified, prespecified maximum
acceptable loss and uncertainty for the paired difference. Preregistering the
Wilson-overlap rule preserves the record of the decision; it does not make that
rule establish the scientific claim.

At sequence 255, the paired two-sided p-values are 0.125, 0.375, and 0.6875 for
8+4, periodic, and 10+2 respectively. The appropriate statement is that these
hybrids have higher observed solve rates in this cell, with limited paired
evidence. The new sequence-511 periodic result is a concrete counterexample to
extending the stronger capability claim across the tested task family.

## 2. The parameter-reduced CE gains survive, but trajectory dominance does not

**Verified:** independently recomputed terminal `done.final_val`, arm minus
within-suite attention, five paired seeds per row:

| Hybrid | Expansion | Parameters above attention | Mean ΔCE | Exploratory paired 95% t interval |
|---|---:|---:|---:|---|
| Late 8+4 | 2 | 19.01% | −0.016543 | [−0.021026, −0.012060] |
| Periodic 9+3 | 2 | 21.39% | −0.024238 | [−0.029391, −0.019085] |
| Late 8+4 | 1 | 3.75% | −0.005309 | [−0.008694, −0.001923] |
| Periodic 9+3 | 1 | 4.22% | −0.010983 | [−0.015944, −0.006021] |

Attention has 123,699,612 parameters; expansion-1 late and periodic have
128,343,364 and 128,923,833. Source suites:
[`crossover50m_ratioplace32`](../../nanolab/out/crossover50m_ratioplace32) and
[`crossover50m_parity32`](../../nanolab/out/crossover50m_parity32).
All four endpoint comparisons favor the hybrid on all five seeds. These terminal
draws are distinct from the saved periodic evaluations in the older review.

The expansion-1 endpoints therefore support a smaller real *observed* CE margin
under this recipe. They are approximate parameter parity, not equal parameter
count, and not yet independent confirmation after architecture selection.

**Verified limitation:** the expansion-1 mean curves are not uniformly better.
Late 8+4 is worse by **+0.000709** at a saved 36.88M-token marker; periodic is
worse by **+0.029371** at 18.86M. Across all 61 shared markers, their mean deltas
are negative at 58 and 33, while every seed is negative at only 35 and 27.
The tiny late-8+4 reversal is not evidence of a meaningful population regression;
it is enough to reject literal dominance of the recorded means. The original
expansion-2 late 8+4 still wins on every seed at all 60 markers after the first.

**Verified recipe sensitivity:** in the same ratio/placement suite, increasing
the joint LR multiplier reverses the endpoint ranking:

| Joint LR multiplier | Late 8+4 minus attention | Periodic minus attention |
|---|---:|---:|
| 1× | −0.016543 | −0.024238 |
| 4× | +0.027250 | +0.028879 |
| 8× | +0.022575 | +0.028059 |

All four positive comparisons have paired 95% t intervals above zero. Both
`lr` and `matrix_lr` change; this is not an isolated optimizer-group test.
Do not infer an intrinsic hybrid advantage from a single shared recipe, or
transfer an expansion-2 capability result to expansion 1 without testing it.

## 3. Two training-state contracts are incomplete

### 3.1 Resume restores counters, but not the sampled training trajectory

**Verified defect:** [training initialization](../../nanolab/train.py#L136)
reseeds execution; [resume](../../nanolab/train.py#L235) restores model,
optimizers, steps, and tokens; the
[checkpoint payload](../../nanolab/train.py#L454) does not save sampler RNG state.
The actual [Batcher](../../nanolab/data.py#L366) owns a private generator.
A fresh resumed batcher therefore starts a fresh sequence from its initial seed.

The new probe compares uninterrupted six-step training with interruption after
checkpointed step 2, whose `next_step` is 3, using the actual trainer and actual
batchers. It uses a one-layer CPU FP32 AdamW model, no dropout, and temporary
synthetic token files.

| Resume condition | Maximum final parameter difference from uninterrupted |
|---|---:|
| Current fresh-batcher resume | **0.0043052742** |
| Same checkpoint, externally restore both batcher RNG states | **0.0** |

This positive control isolates a causal sampler-state omission on the fixture.
The existing [resume tests](../../nanolab/tests.py#L2688) pass, but use constant
batches and test counters; they cannot detect this sample-stream change.

**Implication:** correcting the token axis does not retroactively restore a
historical optimization trajectory or validation bank. Claims that resumed
losses are unaffected require evidence beyond counter continuity. This result
does not invalidate every run: it does not replay the historical large models
or quantify the effect on their conclusions.

**Proposed repair contract:** checkpoint each sampler's state, global RNG state
where used, optimizer/schedule state, and relevant best-checkpoint state. Test
interrupted versus uninterrupted execution on nonconstant batches, including
evaluation boundaries. CPU exactness here is not CUDA determinism proof.

### 3.2 Train-loss evaluation consumes future training samples

**Verified:** [the training loop](../../nanolab/train.py#L362) gives the same
training batcher to `evaluate`; [evaluation](../../nanolab/train.py#L44) calls
that batcher repeatedly. The actual-generator fixture confirms its state changes.

Changing `eval_train`, `eval_iters`, or evaluation frequency can therefore change
the training sample sequence. These fields are not merely logging or throughput
settings. The paired-reader recipe guard also omits `eval_train`.

**Proposed repair:** use a separately seeded train-evaluation sampler or a fixed
evaluation bank. Preserve its identity and all sampler states across resume.
This is a training-data schedule issue, not a claim of validation-data leakage.

## 4. Analysis tools can silently change the meaning of evidence

The new probe runs adversarial fixtures against the unmodified repository tools.
Each result below is **verified behavior**, not a hypothetical code smell.

| Tool/contract | Reproduced behavior | Scientific consequence |
|---|---|---|
| [`paired_board.interval`](../../scripts/paired_board.py#L121) | Always uses the df=4 constants. For n=3 values [0,1,2], its 95% interval is [−0.6030, 2.6030], versus correct [−1.4841, 3.4841]. | Non-five-seed intervals are miscalibrated. Existing n=5 intervals are not wrong for this reason. |
| [`paired_board.load_arm`](../../scripts/paired_board.py#L68) | Duplicate arm/seed directories silently select the last sorted value; the fixture changes final loss from 1 to 9. A curve without `done` is also loaded. | Duplicate or partial records can masquerade as a valid comparison. `crossing_token` separately rejects missing final values, so this particular omission does not bypass that check. |
| [`paired_board.at`](../../scripts/paired_board.py#L117) | Independently selects nearest markers; a value at token 100 can be compared with one at 200 when 200 is requested. | CLI marker tables can compare unequal budgets unless actual selected markers are checked. The exact-intersection crossing path avoids this particular issue. |
| [`RECIPE_KEYS`](../../scripts/paired_board.py#L45) | Omits compile, fused CE, accumulation, dataset, dtype, train evaluation, and weight decay. | The guard does not establish a complete shared recipe. |
| [`lr_argmin.cells`](../../scripts/lr_argmin.py#L98) | Groups by width, mixer/layout, LR, seed; compile variants in the fixture collide and the last final loss wins. | Additional recurrence rules, expansion, and execution recipes need explicit identity, not just layout strings. |
| [`crossing_token`](../../scripts/crossing_token.py#L49) | For difference [−1,+1,−1], returns one crossing at 1.5; the CLI accepts different batch sizes and a terminal result favoring the original arm, yet prints that it was overtaken. | It bypasses the paired CLI recipe guard, ignores subsequent reverse crossings, and checks absolute rather than directed terminal separation. |

The crossing counterexample uses valid nonnegative synthetic losses
`[1,3,1]` versus `[2,2,2]`. Early warmup reversals before a late recovery are a
different case and are common in the real ladder. The defect is accepting a
reversal *after* the claimed recovery without qualification.

The original review's own [`recompute.py`](evidence/recompute.py) also fails on
the current corpus:

```text
RuntimeError: duplicate MQAR cell seed .../mqar_e28_p8/('gdn*3,attention,gdn*3,attention,gdn*3,attention', 8, 3000, 256, 31)
```

It collapses repository-rule and published-rule hybrid layouts into the same
group identity. The archived snapshot remains historical evidence; that command
is no longer a successful fresh-corpus regeneration. The new scoped reader uses
explicit arm identity and refuses duplicate arm/seed rows. Neither reader is a
substitute for a canonical experiment ID containing dataset, code, objective,
architecture, optimizer, execution recipe, sampler provenance, and lineage.

**Priority:** repair these contracts before using automatically generated tables
as publication gates. The audit fixtures characterize failures; they do not
deliver source fixes or claim the failing behavior is acceptable.

## 5. Width and budget conclusions need narrower language

### 5.1 The retuned crossing ladder is compatible with a plateau

**Verified**, mean late-recovery token in millions, with exploratory paired
across-seed t intervals:

| Width | Joint LR multiplier | Seeds | Crossing M | 95% interval M |
|---|---:|---:|---:|---|
| 384 | 2× | 5 | 10.707831 | [10.340255, 11.075408] |
| 768 | 1× | 5 | 12.373726 | [12.088426, 12.659025] |
| 1152, existing five-seed row | 1× | 5 | 13.457953 | [13.174182, 13.741725] |
| 1536 | 0.5× | 5 | 13.442107 | [13.272626, 13.611588] |
| 1152, later proposed optimum | 0.6667× | **3** | 13.413996 | [12.888691, 13.939301] |

Sources: [`crossover_ladder50m`](../../nanolab/out/crossover_ladder50m) and
[`crossover_ladder1536m_argmin`](../../nanolab/out/crossover_ladder1536m_argmin).
The late crossings are linear interpolations between saved markers, not observed
continuous event times. Several seeds have an earlier warmup reversal; the saved
JSON includes all sign transitions.

The five-seed width-1152 row still uses 1× even though later documents designate
0.6667× as the optimum. The newer row has seeds 42, 100, and 1337, with 777 and
2026 absent. Consequently the four-width five-seed table is not a fully completed
ladder at every currently claimed optimum. Its final step is −0.015846M rather
than a continued monotone increase. The updated three-seed row is also compatible
with a plateau, but does not complete that replication.

### 5.2 The inverse-width rate is an empirical candidate, not an established law

The [rate note](../MUON_MUP_RULE_2026-09-05.md#L100) and
[gap plan](../GAP_PLAN_2026-09-07.md#L729) treat an attention improvement over a
0.0031-nat “floor” as establishing the inverse-width law. Recomputing the crucial
width-1152, 0.6667× minus 1× contrast gives:

| Model | Paired mean ΔCE | Correct n=3 95% t interval |
|---|---:|---|
| Attention | −0.005295 | **[−0.011056, +0.000466]** |
| minGRU | −0.002359 | [−0.007340, +0.002622] |

All three attention signs favor 0.6667×; the two-sided sign-test p-value is 0.25.
The magnitude is useful exploratory evidence, but the uncertainty and selected
grid do not establish a universal law or a precise optimum. Most widths were
searched on coarse multiplicative grids, at one budget and dataset. Both Adam
and Muon learning-rate groups were scaled together.

Primary theory already studies hyperparameter transfer for matrix-preconditioned
optimizers and gives parameter-group-dependent scaling prescriptions. A single
inverse-width multiplier applied to this mixed optimizer is not, by itself, a
derivation of Muon maximal-update parameterization. The repository can report
that its tested minima are compatible with a useful transfer heuristic, and test
that heuristic on held-out widths. [Qiu et al., 2025](https://arxiv.org/html/2512.05620v2).

### 5.3 A small CE difference cannot identify a historical sampler

**Verified:** the cited 0.0031 maximum comes from ten reruns at **5M tokens**;
the [tuning document](../GPU_TUNING_2026-09-05.md#L391) explicitly leaves the 50M
floor unmeasured. The [G12 interpretation](../GAP_PLAN_2026-09-07.md#L852)
uses three 50M rerun differences, approximately 0.0021, 0.0048, and 0.0024, to
classify whether an unrecorded original sampler selected the same tokens.

**Verified logical limitation:** an observed finite-sample maximum is not an
upper bound on future differences, a significance threshold, or a classifier of
hidden execution state. Similar final losses do not identify identical samples;
different final losses do not identify memmap fallback. The original sampler is
**unverified** where it was not recorded. The checkpoint and evaluation findings
above add concrete alternative causes of trajectory changes.

Repeatability measurements are useful for planning replication, but effects
smaller than a single-rerun difference can be estimated with replication, and
effects larger than it are not automatically established. If historical rows are
replaced, use an explicit provenance-based replacement rule and retain lineage;
do not choose replacements because their endpoint fits the expected curve.

### 5.4 Budget independence has an execution confound and an identifiability limit

The [manuscript's 50M→200M comparison](../../PAPER_2026-09_The_Crossing_Scales.md#L456)
says only the budget and cosine horizon change. **Verified contradiction:** the
old width-384 `attention_lr80` and `mingru_lr40` configs record `compile: false`;
their 200M counterparts record `compile: true`. The new JSON saves every per-seed
config difference. Missing old fields versus newly explicit defaults are not
automatically changes, but the explicit compile values are different.

Similar observed crossings remain a result. They do not isolate the effect of
budget or establish equivalence. These old learning rates also differ from the
later 50M optima, and the 200M optimum has not been established by this comparison.
With constant tokens per update, tokens and optimizer updates are collinear;
the evidence cannot say which sets the crossing. One width and two horizons do
not establish schedule-independent screening for other architectures or budgets.

## 6. Comparator experiments are no longer merely pending

**Verified:** the small-pair published-GDN comparison now has results. In the
[p8, batch-256, 3,000-step ledger](../../nanolab/out/mqar_e28_p8/runs.jsonl),
repository GDN solves **2/15**, published-rule GDN **1/15**, repository periodic
GDN hybrid **6/15**, and published-rule periodic hybrid **9/15**. This does not
show that correcting the recurrence alone rescues pure GDN, nor that the rule
never matters. It completes these four cells, not every planned GDN task cell.

The [raw-weight top-1 MoE board](../../nanolab/out/crossover50m_moe32d) also exists:

| Raw-weight top-1 arm minus attention | Five-seed mean ΔCE | Paired 95% t interval |
|---|---:|---|
| One expert | −0.001169 | [−0.005040, +0.002703] |
| Four experts | +0.038951 | [+0.031830, +0.046072] |
| Eight experts | +0.031094 | [+0.026424, +0.035764] |

These comparisons narrow the earlier router-gradient ambiguity for the recorded
regime; they are not a general negative result about MoE capacity or specialization.

The project's LR reader on `crossover_probe50m` reports **MLA at 0.5×**, final
CE 4.5296, versus 4.6435 at 1×, on seed 1337. Mamba2 and GDN select 1× in that
probe. This is a one-seed grid result, not a validated cross-family transfer law.
Descriptions of MLA as still pending, or all these families sharing the same
optimum, require updating when used as current status.

## 7. BINN W29 confirms robustness to thinning, not absence of count information

**Verified:** independently reading all 48 W29 cells gives the following local
macOS means across the twelve complete seed quadruples:

| Condition | Rate arm | Attention arm | Attention advantage |
|---|---:|---:|---:|
| Intact | 0.706787 | 0.826082 | 0.119295 |
| Delete 90% of spikes | 0.595554 | 0.761521 | 0.165967 |
| Accuracy loss | 0.111234 | 0.064561 | — |

The paired difference in losses, attention minus rate, is **−0.046673**, 95% t
interval **[−0.056721, −0.036624]**. It is negative in 12/12 seeds and below
−0.03 in 11/12. This supports the registered W29 directional reading on its own
platform. W28's registered NOT MET outcome remains unchanged.
Source: [W29 result](</Users/bharath/Code/research/BINN/results/RESULT_2026-09-09_W29_THE_ASYMMETRY_HAS_ITS_OWN_BAR.md:55>);
the saved JSON bundles cell hashes and recomputed values, not the external cells.

**Qualification:** the attention arm loses about 6.46 percentage points of absolute
accuracy. Its advantage survives and grows; an unqualified claim that it is not
hurt is misleading. More fundamentally, deleting spikes does not remove all
count information. The actual [dropout implementation](</Users/bharath/Code/research/BINN/binn-learn/src/shd_temporal.rs:792>)
independently retains roughly one in ten integer spikes. Thus
`E[N_after | N_before] = 0.1 N_before`: relative count structure can remain useful.

The new analytical counterexample is deliberately simple: equally likely classes
have 100 versus 1,000 spikes, followed by 90% thinning. A count-only threshold at
40 retains approximately **0.9999999999995** expected accuracy. This is not an
SHD result. It disproves the general inference that successful classification
after thinning must rely on order instead of counts.

**Inferred research direction:** attention may be a robust readout of a noisy
temporal representation. Identifying the reason still needs the September 8
frozen-encoder × head intervention and controls that actually match count
information; that count-matched requirement is now folded into that plan's B2
by its [§7.5](SECOND_AUDIT_PLAN_2026-09-08.md#75-narrowings-to-the-retained-stages).
W29 provides within-block evidence; it does not by itself establish an
order-only mechanism or platform equivalence.

## 8. The memory-update oracle is useful, but finite protected rank is decisive

The [existing protocol](MEMORY_UPDATE_PROTOCOL.md) remains mathematically valid
for its specified linear memory and squared loss. The old numerical probes still
pass 10,000 scalar and 1,000 projected-update fixtures. The supplied 124/200
trajectory remains report-backed; it was not reconstructed in this audit.

Let `M` have shape `d_v × d_k`, let protected keys be columns of `Q`, and let `U`
span those keys orthonormally. For incoming key `k`, residual `e = v − Mk`, and
`p = (I − UUᵀ)k`, the minimum-norm exact-preservation fit, when `p ≠ 0`, is

```text
D = e pᵀ / ||p||²
||D||_F = ||e|| / ||p||
```

Under `DQ = 0` and `||D||_F ≤ B`, the minimum incoming squared error is
`max(||e|| − B||p||, 0)²`. These are diagnostic statements about fixed features,
not categorical-recall guarantees or a measured efficient implementation.

Four limits follow directly:

1. **Protecting a spanning set freezes every linear prediction.** If
   `rank(Q) = d_k`, `DQ = 0` forces `D = 0`. The new numerical rank sweep reaches
   projected-key norm below 1e−12 at rank 12. A large residual then cannot be
   fixed without relaxing protection, changing representation, or adding state.
2. **Witness selection determines what “retained” means.** Eight witnesses in
   twelve dimensions leave at least four unprotected directions by construction.
   Good behavior on a fixed witness set is not sustainable retention of every
   newly admitted independent fact. Exact protection also preserves wrong old
   predictions. Tolerating or repairing error is a distinct objective.
3. **A per-step norm bound does not bound resident state.** Repeated bounded
   writes can accumulate, and near-dependent keys can require expensive updates.
   A deployment claim needs a total-state/conditioning policy in addition to `B`.
4. **Protection is not free memory.** In the illustrative six-value, twelve-key,
   eight-witness fixture, `M` has 72 scalars. Storing eight keys and targets adds
   144, bringing the total to 216, or 3× the matrix alone, before any separately
   retained basis/workspace. This is an illustrative implementation accounting,
   not a lower bound; a basis may replace some stored keys and must not be counted
   twice. All methods must include comparable metadata, cache, and scratch cost.

**Inferred opportunity:** the ratio `||e|| / ||p||` is a diagnostic cost of fitting
an association while exactly preserving specified predictions. Testing whether
a cheap approximation predicts when to redirect a write or retain an exact
exception is more precise than calling a large residual “memory fullness.” It
remains a candidate signal, not a validated architecture or established novelty.

## 9. Primary literature substantially narrows novelty

These are claims reported by the cited papers, not experiments reproduced here.
The overlap column describes why each affects this proposal; different memory
contracts and information timing still require explicit comparison.

| Primary source | Relevant overlap | Requirement for a distinguishable contribution |
|---|---|---|
| [Preconditioned DeltaNet, April 2026](https://arxiv.org/html/2604.21100v1) | Curvature-aware recurrence, online least-squares/inverse-Gram interpretation, and practical preconditioning. | Show benefit beyond residual correction and diagonal/full covariance preconditioning at matched state and compute. A projected update alone is not enough. |
| [Memory by Design, August 2026 revision](https://arxiv.org/abs/2605.31163) | Probabilistic state with mean/covariance and writes steered by uncertainty while preserving confident information. | Compare retention/error control with uncertainty-directed writes; count covariance state. |
| [Hybrid Associative Memories, March 2026](https://arxiv.org/html/2603.22325v2) | Recurrent compression plus selectively retained key–value memory; learned selection versus surprise criteria. | Demonstrate that preservation cost selects exceptions better than surprise, residual, or learned selection under the same budget. |
| [HOLA, July 2026](https://arxiv.org/abs/2607.02303) | Bounded exact cache for linear attention, routed by a committed-residual signal. | Distinguish preservation-constrained cost from residual magnitude and include this close cache baseline. |
| [Memoir, July 2026](https://arxiv.org/html/2607.20792v1) | Read-only pondering versus writing intermediate reasoning states into memory. Its small-model short-budget gap disappears at longer training. | Treat query/read/write separation as an existing hypothesis with optimization controls. Query tokens and latent pondering are related, not identical interventions. |
| [Revisiting associative recall, 2025](https://arxiv.org/abs/2508.19029) | Recall performance depends on optimization, width, and depth; attention is not universally easy to train on every recall setup. | Distinguish storage limits from trainability using tuned baselines and binding/representation oracles. |
| [Qiu et al., 2025](https://arxiv.org/html/2512.05620v2) | Hyperparameter transfer for matrix-preconditioned optimizers and parameter-group scaling. | Derive and isolate the applicable optimizer parameterization rather than rename a jointly tuned inverse-width recipe. |

ScholarLM cloud was used for discovery. One search encountered an arXiv-provider
HTTP 429 and returned limited useful material; a later precise optimizer search
returned useful results. Primary arXiv/publisher pages were then inspected
directly. The obsolete WisDev connection was not used. Search coverage does not
justify “nobody has tried this.”

## 10. Revised research program and stopping rules

These are **proposed next actions**, not queued or executed experiments. The
order is intended to resolve the most consequential uncertainty before more
architecture combinations are trained.

| Priority | Concrete question and smallest informative experiment | Gate before expansion |
|---|---|---|
| 0 — Evidence contracts | Correct resume/sampler state, independent evaluation sampling, full experiment identity, completion/duplicate checks, sample-size-aware statistics, and directed crossing semantics. Regressions must fail against the old behavior, including real random batches. | Do not promote new tables or “unaffected resume” claims until fixtures and selected historical recomputations pass. Preserve excluded rows and lineage. |
| 1 — Global retrieval allocation | First confirm attention, late 8+4, and periodic in the same long-recall cell with recorded full recipe. Then compare fixed attention counts in late versus periodic positions. Independently vary pairs and distance while holding vocabulary, queries, steps/tokens, and batch policy explicit. Include expansion-1 variants before carrying over their cost advantage. | If tuning removes the gap, report an optimization finding. If placement survives fixed-count, fixed-load controls, proceed to mechanism interventions and a held-out task. Prespecify a meaningful paired noninferiority margin rather than use Wilson overlap. |
| 2 — Preservation-cost signal | Frozen-feature streams with independently varied key dependence, noisy/compressible values, corrections, and overwrite rates. Compare ordinary delta, preconditioned/covariance updates, projection oracle, and bounded exception memory. Compare residual-only and preservation-cost routing using identical stored bytes and information timing. | If benefit requires an unbounded witness/covariance store, oracle labels unavailable online, or favorable fixed witnesses, retain it as a diagnostic. Advance only if a cheap causal signal predicts failures beyond the strongest baselines on held-out streams. |
| 3 — BINN mechanism | Replay the same frozen encoder with alternative heads; separate count-matched order interventions from count thinning. Record platform and seed roles. | If encoder changes explain the benefit, retire the pure readout attribution. If count-matched controls remove it, narrow the order claim. Preserve W28/W29 registered verdicts. |
| 4 — Scaling paper, if retained as a goal | Complete independent confirmation at the newly selected width-1152 rate; separate Adam and Muon group scaling and use a held-out width. For a budget claim, use the same compile/sampler recipe and independently vary tokens per update. | Report only a transfer heuristic unless the predictions survive outside the fitted grid. A plateau or wide interval is a result, not a failed opportunity to fit a law. |

Do not begin with a composite of protected writes, cache routing, hybrid layout,
new optimizer scaling, and altered decoding. Each currently has a different
unresolved explanation. Combining them would make an improvement harder to
attribute and its novelty harder to defend.

The strongest current research contribution is a reproducible account of
recipe-dependent rankings and the failure of aggregate CE to determine recall
behavior, strengthened by a reliable measurement contract. A larger architectural
contribution remains possible if a small, affordable operation explains these
failures and improves held-out behavior beyond close existing methods. The
current evidence does not yet select that operation.
