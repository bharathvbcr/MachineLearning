# BINN and NanoLab: second audit and revised experiment plan

**Date:** September 8, 2026. **Status:** source-audited proposal, with executed
sampler/operator checks. No training campaign or mechanism result is reported.

**Amended September 14, 2026.** The [September 13 deeper audit](DEEP_AUDIT_2026-09-13.md)
governs current interpretation and priority. This plan remains the owner of the
mechanism-study controls, reordered and narrowed by
[§7](#7-amendment--september-14-2026). Four dated claims below are superseded
there and marked in place; the controls themselves stand.

This document replaces the experiment ordering and controls proposed in the
preceding conversation. It remains the execution plan for the two mechanism
studies, as amended in §7. [MEMORY_UPDATE_PROTOCOL.md](MEMORY_UPDATE_PROTOCOL.md)
remains the owner of the mathematical decomposition and oracle definitions. Existing registered
BINN waves retain their own hypotheses, thresholds, and amendments.

The second audit changes the recommendation substantially: establish a valid
assay first; distinguish trained representations from readout capability in BINN;
and distinguish query content, decay, and delta updates in NanoLab. A restoration
control or a shuffle-induced accuracy drop alone cannot identify a mechanism.

## 1. Findings that change the previous plan

“Verified” below means checked against current source or the stated local probe.
“Recorded” means a completed-result document was inspected, without retraining.
“Proposed” means work remains. A source check does not validate a trained-model claim.

| Finding | Evidence and status | Revision |
|---|---|---|
| BINN already tested hidden-sequence shuffling, reversal, and a window ladder. | **Recorded:** [W27](../../../BINN/results/RESULT_2026-09-05_W27_THE_TIMESCALE_IS_261_MS.md), especially sections 1–2; [W24](../../../BINN/results/shd_attention_campaign_v2/VERDICTS_W24.md). | Do not present another broad shuffle ladder as a new mechanism experiment. The 260.7 ms value is an operator-response scale at one anchor, not an intrinsic memory constant. |
| BINN's `fixed` means non-adaptive threshold, not frozen learned weights. | **Verified:** `MatchedArm` in [shd_matched_arms.rs](../../../BINN/binn-learn/src/shd_matched_arms.rs), lines 58–91; `ArmAdam::update`, lines 1087–1097, calls the base optimizer; [shd_matched.rs](../../../BINN/binn-learn/src/shd_matched.rs), lines 464–480, updates input weights. | The original headline compares jointly trained systems. A common frozen encoder is a new diagnostic protocol, not a description of that comparison. |
| The existing BINN temporal condition transforms training, validation, and test data before training. | **Verified:** [shd_instrument.rs](../../../BINN/binn-lab/experiments/shd_instrument.rs), lines 754–799. Distinct streams are used across splits. | Existing results supply the matched-distribution diagonal; they do not supply all four train/test combinations. Preserve fixed-per-sample versus resampled transformations as distinct protocols. |
| Full random shuffling can erase the label information itself. | **Verified analytical fixture:** the two balanced labels A-before-B and B-before-A produce identical input distributions after an unobserved uniform permutation; exhaustive enumeration gives Bayes accuracy 0.5. [Probe results](evidence/second-audit-results.json). | Chance performance on that fixture is an instrument control. It cannot establish a faulty architecture or recoverable memory loss. |
| BINN attention is unmasked across the utterance, and hidden shuffling reassigns positional codes. | **Verified:** [shd_attention.rs](../../../BINN/binn-learn/src/shd_attention.rs), lines 718–735 and 782–835. | Distinguish changing content–position associations from merely rearranging already-positioned rows. Do not transfer causal-decoder arguments directly to this readout. |
| NanoLab MQAR feeds earlier correct answers back as inputs and selects queries without replacement. | **Verified:** [mqar.py](../../nanolab/mqar.py), lines 114–135; the live sampler probe checks four earlier-answer tokens across two episodes. | A query-only task needs a new task identity and matched training. Answer refresh and elimination of already-queried facts can otherwise confound “destructive questions.” This is a task mismatch, not a claim that teacher-forced MQAR is invalid. |
| Disabling the GDN delta write does not freeze its state. | **Verified live operator:** with beta zero after an initial write, both rules produce `1, 0.5, 0.25` under decay; disabling decay too produces `1, 1, 1`. [Probe results](evidence/second-audit-results.json). | Separate delta updates from decay; define persistent-state changes explicitly. |
| Full-state restoration followed by an identical deterministic read is a recovery identity. | **Analytical:** if the decoder, query, position, and every causal state are restored, the same function receives the same inputs. | Treat full restoration as an upper-bound/sanity control. Selective interventions and content-matched controls carry the mechanism claim. |
| The proposed NanoLab state intervention is not exposed by the current GDN API. | **Verified:** [mixers.py](../../nanolab/mixers.py), lines 748–760 initializes and consumes state internally; lines 809–837 return outputs, not a reusable state. [model.py](../../nanolab/model.py), lines 329–359 has cached-window support for attention. | A trace/replay instrument and parity checks are prerequisites; “snapshot and compare” is not currently a ready-to-run GDN command. |
| Historical MQAR records are not automatically available as trained checkpoints. | **Verified source:** [mqar_suite.py](../../nanolab/mqar_suite.py), lines 166–199 trains and returns metrics without saving weights. Local filename inventory: 33 `.pt`/`.pth`/`.safetensors` files under `nanolab/out`, zero with MQAR/recall/E28 in their paths. | Inventory and verify checkpoint provenance before promising evaluation-only experiments. The filename check is not proof that no relevant checkpoint exists elsewhere. New pilot runs must save weights and full configurations. |
| E28 supplies a recurrence ablation, not a fully faithful published block or an automatic causal verdict. | **Verified:** both rules exist in [mixers.py](../../nanolab/mixers.py), lines 700–705; [e28_gdn_rule.sh](../../scripts/e28_gdn_rule.sh) reuses historical rows. ~~No published-arm rows were found in the local MQAR ledgers inspected.~~ **Superseded 2026-09-14:** `mqar_e28_p8` now holds four complete 15-seed arms, including both published-rule arms; see §7.1. | Run within-environment controls, distinguish rule-only from full-block fidelity, and do not read one solved seed as proof that the old rule caused all failures. Remote completion was not checked. |

## 2. BINN: separate representation learning from temporal readout

### B0 — validate meaning and replay before fitting anything

Use the existing headline anchor first: feed-forward, fixed threshold, h128,
published 2 ms bins, adjacent-sum-5, and the documented d32/L4 attention head.
Do not introduce recurrent instability, a width sweep, and a new task simultaneously.

1. Locate trained final weights, source/binary hashes, sample identities, and the
   exact geometry. Initialization files do not qualify. The existing
   `--save-final-weights` path is in `shd_instrument.rs`, lines 1083–1089.
   If matched checkpoints cannot be obtained, budget a newly identified paired
   training run; do not silently use a different model or the old accuracy row.
2. Reproduce the checkpoint's ordinary evaluation on its platform. Record logits,
   accuracy, CE, and the precise set of held-out utterances. Cache hidden sequences
   with encoder and sample hashes; check cached versus ordinary readout evaluation.
3. Use small, exhaustively checked count/order/coincidence fixtures as manipulation
   tests. Verify actual input and output statistics; these are not a second research
   benchmark yet. Reversal changes the meaning of an A-before-B label unless the
   label is transformed, despite being information-preserving and invertible.
   A padded shift must neither truncate events nor change the evaluation window.
4. Add a diagnostic permutation control at the attention input: permute content
   **and its already-assigned positional vector together**. For this unmasked,
   row-equivariant stack followed by mean pooling, its logits should agree within
   a declared floating-point tolerance. This differs from `hidden-shuffled`, which
   attaches the destination position. The expected agreement is an instrument
   identity, not a new biological finding. Stop if it fails.

### B1 — the first research experiment

**Question:** does attention extract useful temporal information from the same
representation, and how much does the answer depend on how that representation
was learned?

For each paired seed, obtain two trained encoder origins: one from the rate system
and one from the attention system. Freeze each encoder, extract its hidden
sequences, and fit fresh rate and attention heads on **each** origin. This is a
2 encoder-origins × 2 head-types comparison. Use the same train/validation/test
speakers, examples, exposure budgets, and head-tuning budgets within each origin.
Keep original jointly trained heads as separate reference rows.

The two encoder origins prevent a head being judged only on features learned to
favor its competitor. Hash frozen input weights before and after head fitting.
No gradient or optimizer weight decay may change them. The local-learning
[frozen-attention protocol](../../../BINN/results/PREREG_2026-08-19_SHD_FROZEN_ATTENTION_LOCAL.md)
freezes a different component and answers a different question; do not reuse its
protocol identity or gate outcomes.

Primary estimates are attention-minus-rate held-out accuracy within each encoder
origin, with CE alongside it. An advantage reproduced on both origins supports
readout utility conditional on these learned representations. A strong
origin-by-head interaction supports co-adaptation as a candidate explanation.
This is not an additive decomposition of the original 12.75-point gain.

Use a parameter-matched nonlinear head on pooled features only as a second-stage
capacity control if B1 finds a useful head effect. It distinguishes temporal access
from spending more parameters on a bag-of-frames representation. It still cannot
make every optimization and execution cost equal; measure those costs separately.

### B2 — temporal intervention after B1 is interpretable

At one frozen encoder, use intact and hidden-shuffled representations as the two
head-training conditions. Evaluate each fitted head on both intact and shuffled
held-out sequences. Derive all four evaluations from the corresponding two fits;
do not train four independent models for the four cells.

The permutation accompanies a sample identity, is independent across splits, and
does not enter the labels or head. Reuse training permutations throughout training
for direct comparability with the historical fixed operator; use a locked set of
independent evaluation permutations to estimate transformation variability. A
resampled-per-epoch augmentation study gets a separate identity.

| Observation | Supported conclusion |
|---|---|
| Intact-trained head fails on shuffled inputs but a shuffled-trained head recovers | Evidence for train/test mismatch or an avoidable learned dependence under that protocol. |
| Both fail on a fixture whose transformed labels are provably uninformative | Expected information destruction; no model defect is established. |
| Both fail on SHD despite successful manipulation controls | A limitation under the tested representation, head family, and training budget. It does not prove the remaining signal is information-theoretically absent. |
| A supplied binding oracle helps under identical causal information timing | Evidence that association construction limits the specified downstream system; not proof of a unique internal operation. |

Only after B1/B2, construct a controlled cue–value task if a concrete unresolved
binding hypothesis remains. Keep marginal counts and coincidence statistics
matched when isolating order, and keep encoder dynamics distinct from raw-input
statistics. Freeze a prediction on unseen delay/entity settings before testing a
second dataset. Do not assume synthetic order labels describe SHD speech labels.

### Existing BINN branches

- W29 remains a separately registered confirmation of the dropout asymmetry.
  Its [local-platform amendment](../../../BINN/results/AMENDMENT_2026-09-07_WAVE_29_RUNS_ON_THE_LOCAL_PLATFORM.md)
  governs source/binary/platform accounting. ~~No W29 cell path was found in the
  local v3 directory scan; this audit does not establish live job status or launch
  it.~~ **Superseded 2026-09-14:** W29 completed on 2026-09-09 and all 48 cells
  were independently recomputed by the deeper audit; see §7.1 and §7.5.
- The [September 7 decomposition measurement](../../../BINN/results/MEASUREMENT_2026-09-07_THE_DECOMPOSITION_HAS_NO_TASK.md)
  leaves the 160-cell transfer study not evaluable on its task precondition. Do not
  repurpose a frozen-head diagnostic as a fix for that separate local-learning gate.
- The [counterfactual eligibility arm](../../../BINN/results/DESIGN_2026-09-05_THE_COUNTERFACTUAL_ARM.md)
  is another hypothesis about learning, not evidence for a temporal readout claim.

## 3. NanoLab: distinguish query interference, decay, and delta updates

### N0 — establish a valid task and a reusable instrument

Keep the existing teacher-forced MQAR task as a reference. Introduce a distinct
query-only diagnostic task: encode all facts first, then present a sequence of
query keys and fixed delimiters. Score values at declared query positions without
feeding gold or generated values into later queries. Sample queries with
replacement independently of future queries, so all facts can remain useful.
Repeated and never-previously-queried facts are reported separately.

Train on this grammar before making in-distribution performance claims. Removing
answer tokens only at evaluation is an explicitly labeled distribution-shift test.
Include real fact-update episodes in a later policy evaluation so a method cannot
appear successful merely by refusing every persistent change.

The first comparison uses `gdn` and `gdn_pub`, plus attention as a task-solvability
reference. A published-rule setting is still NanoLab's block with that recurrence;
it does not establish reproduction of the complete published Gated DeltaNet.
Check a faithful-block reference before a broad claim about that architecture.
Keep minGRU and hybrids for follow-up after the single-operator question is clear.

Implement trace/replay at the existing GDN owner rather than a second training
stack. Return declared per-layer/head state boundaries and preserve absolute token
positions. Compare unmodified replay against current chunked and sequential
outputs, including non-multiple chunk lengths and multi-layer propagation, before
using an intervention. The primary analysis starts at a selected layer/head with
fixed projected queries, keys, values, and gates; whole-model interventions must
then allow downstream computation to respond. Fixed-feature replay alone cannot
establish a trained-model mechanism.

Save final checkpoints, exact task/configuration hashes, data RNG state, learning
curves, and evaluation-bank identities. The current MQAR `run_one` metrics-only
return is insufficient for this study. New task and instrument code are prerequisites,
not implemented by this documentation change.

### N1 — measure harm without confusing content with elapsed processing

For each fact episode and frozen checkpoint, branch from the same encoded state.
Place the final diagnostic query at the same absolute slot in every branch.
Compare intervening query blocks with equal-length delimiter/distractor blocks
that occur in training. Counterbalance the order of a fixed multiset of queries.
Keep pair count, value vocabulary, total sequence length, and memory dimensions
fixed for the initial contrast. Vary load and distance separately only afterward.

Run the final diagnostic on a disposable branch; diagnostic reads must not mutate
the trajectory being measured. Do not supply future queries or answers to the
model or its controller. More query tokens increase elapsed processing and, for
attention, available context; record those quantities rather than attributing any
order effect automatically to overwritten memory.

Measure exact-match accuracy and target CE for the current query and for later
probes, with paired per-episode differences. Report raw results for all episodes,
and a secondary stratum selected by success on a separate pre-intervention probe.
Do not select cases by which intervention subsequently succeeds.

The primary behavioral quantity is later-probe accuracy after matched filler
minus accuracy after query processing, at the same terminal position. The
no-intervening-update baseline is a diagnostic ceiling, not the matched content
control. If the query/filler difference is negligible but both lose recall, the
leading interpretation is generic progression/decay rather than query-specific harm.

### N2 — a factorial intervention, not just restoring all memory

For the selected GDN state, compare the following persistent transitions during
intervening queries. Compute current-query output using a private ordinary working
branch first, then apply the specified persistent commit. This preserves the
immediate calculation at a shared starting state. Also report ordinary-forward
ablations separately because they may change current-answer computation.

| Persistent transition | Delta update | Decay | Role |
|---|---|---|---|
| Ordinary | enabled | enabled | Reference |
| Decay-only | disabled | enabled | Identifies the contribution of delta updates |
| Delta-only | enabled | disabled | Identifies the contribution of decay |
| Identity | disabled | disabled | No-commit policy / upper control |

Specify the intervention algebra in the same value-by-key convention as the
protocol. For the ordinary gates, the repository rule is
`alpha*M + beta*(v - M*k)*k^T`; the published rule is
`alpha*M + beta*(v - alpha*M*k)*k^T`. Decay-only is `alpha*M`, delta-only is
`M + beta*(v - M*k)*k^T`, and identity is `M`. For the published rule, disabling
decay changes both its carry and its residual prediction; keep this disclosed.
At a shared state the delta-only branches coincide; subsequent trajectories need
not. Do not merely set beta to zero and call the result “read-only.”

Full restoration of all encoded facts before the terminal read remains a sanity
control. To identify a mechanism, use selective layer/head replacement with sham
replacement, a same-shaped state from another matched episode, and changes to
non-target components. Keep all non-target states, positions, and probe tokens
fixed at the comparison boundary. Record interventions that fail; partial patches
can themselves be off-distribution. Select locations on discovery episodes, then
lock them before confirmation.

Advance to a learned write policy only if a specific transition component explains
reproducible harm and a selective intervention recovers it on fresh episodes and
seeds. A useful policy must also retain immediate-answer quality and learn genuine
new facts; its oracle query/fact labels must be disclosed or replaced by a causal
learned decision. Charge any snapshot or auxiliary storage and temporary copies.
No universal ban on query-time writes follows from an immutable-fact toy task.

## 4. Decision rules and bounded execution

These are **proposed defaults**, not a completed preregistration or a power claim.
Keep existing registered studies' rules unchanged.

| Stage | Scope and stopping rule |
|---|---|
| Assay validation | No training. All identity, label-information, state-parity, and no-leakage checks must pass before interpreting a mechanism contrast. This audit executed the three probe groups in section 6; the new replay instrument remains to be built. |
| Discovery | Three fresh paired training seeds per selected task/arm; at most three learning-rate candidates per arm with the same exposure budget. If the best LR lies at the edge, permit one predeclared grid expansion of equal size for all affected comparisons; otherwise report unresolved tuning. Reuse checkpoints for all diagnostic conditions. |
| Task calibration | On discovery data, ordinary baseline recall must be clearly above chance and show measurable room to lose accuracy. Check statistical precision using repeated fixed checkpoints and independently generated episode banks. A weak baseline or a broken manipulation is an assay failure, not a scientific null. |
| Confirmation | Freeze task, sites, algorithms, effect direction, metrics, tuning result, evaluation count, and sample size before evaluating fresh seeds. Simulate the actual joint decision, including its confidence bounds and multiplicity rule, for 80% power under a planning alternative of 6-point benefits against the 3-point superiority margins and zero immediate-answer harm against the 1-point noninferiority margin. Use conservative discovery-variance sensitivity checks. Choose a fixed seed count in 12–32; if the estimated requirement exceeds 32, label the proposal underpowered within this budget and resize the design before starting confirmation. |
| Main effects | BINN: attention-minus-rate within each frozen encoder origin. Claim robustness across origins only if the lower 95% bound exceeds 3 points for both. NanoLab: query-specific later-recall harm exceeding 3 points, followed by a preselected selective rescue exceeding 3 points; report decay-only outcomes separately if query specificity fails. |
| Repair quality | Current-query accuracy must have a lower 95% bound for repaired-minus-ordinary greater than −1 point. Full policy claims also require a separately registered bound for learning new facts. Positive average recall alone is insufficient. |
| Null/inconclusive | Evidence for an effect below the 3-point practical threshold requires the appropriate upper confidence bound below that threshold. An interval crossing the threshold is inconclusive. Never add seeds until a preferred verdict appears. |
| Expansion | One held-out task setting first; then a second task/dataset, followed by byte/cost matching. A broader architecture sweep is conditional on these gates. |

Average evaluation examples and permutations within each training seed before
forming cross-seed conclusions. Use hierarchical uncertainty when also claiming
generalization over sampled speakers/episodes. Neither repeated queries nor hundreds
of checkpoint evaluations become independent training replicates. Use simultaneous
intervals or a declared multiplicity adjustment for extra confirmatory sites and
conditions; discovery heatmaps remain exploratory.

The 6-point planning alternative is a proposed design choice, not a prediction.
Power to reject zero at a true 3-point effect is not power to put a confidence bound
above a 3-point margin; do not substitute that easier calculation.

Recompute from final checkpoints; do not substitute `best_val` or select an epoch
by test accuracy. Failed/unstable runs remain in the denominator with their reasons.
Freeze the handling of missing pairs before training. Keep task RNG, model RNG,
and transformation RNG separate so equal seed labels actually yield matched data.

Do not price either new experiment from old cell medians without measuring its
actual shape, replay overhead, hardware, and concurrency. B1 entails four head fits
per seed plus up to two encoder-training runs if matched checkpoints are missing.
B2 adds only the transformed-condition head fits at the selected encoder. N0–N2
reuse each trained checkpoint across interventions; factorial evaluations are not
four training campaigns. No cloud jobs, subscriptions, or long training runs were
started by this audit.

## 5. Prior work and the claim boundary

Associative-recall benchmarking and hybrid comparisons already have substantial
precedent in [Zoology](https://arxiv.org/abs/2312.04927). A further relevant study
finds recurrent-model AR results sensitive to learning rate and architectural
scaling: [Revisiting associative recall in modern recurrent models](https://arxiv.org/abs/2508.19029).
This supports keeping tuning controls in the plan; it does not establish the cause
of any NanoLab result.

The interpretation of a patch depends on the intervention and metric. The audits
in [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042)
and [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
motivate explicit control interventions and limited causal claims. These primary
sources were checked for this revision; the search was targeted, not an exhaustive
novelty review. A useful contribution would be a measured, specific failure and
selective remedy that survives the controls above. A shuffle deficit, a successful
MQAR model, or a complete-state reset is insufficient on its own.

## 6. Audit artifacts, checks, and remaining limits

- [second_audit_probe.py](evidence/second_audit_probe.py) calls the existing MQAR
  sampler and both live GDN rules, and exhaustively enumerates 24 permutations per
  label in the synthetic order fixture. All three groups passed locally.
- [second-audit-results.json](evidence/second-audit-results.json) records exact
  outputs, Python/PyTorch versions, Git HEAD, and four source fingerprints. The
  GDN fixtures are scalar operator checks, not evidence from trained models.
- MLSystemsLab HEAD at audit: `fae1526b8b9fd198e0d91fe57dac4ee099cf449b`, with
  pre-existing uncommitted work. BINN HEAD: `a27ac7f7b1caa5698a4c5671388e1c843466a0f6`.
  Live source, not Git HEAD alone, defines the code inspected.
- ~~Local MQAR ledgers had zero published-arm rows in the inspected files.~~
  **Superseded 2026-09-14:** see §7.1. No remote queue, provider machine, or
  remote checkpoint store was inspected on September 8.
- No new task generator, trained-model state intervention, head-fitting experiment,
  model repair, or scientific training replication was implemented or run. Those
  are the explicitly gated next steps above.

Reproduce the local audit checks from the repository root:

```bash
python3 docs/architecture-review-2026-09-04/evidence/second_audit_probe.py
```

Navigation, source links, JSON validity, and scoped diff checks are recorded in
[the validation log](evidence/SECOND_AUDIT_VALIDATION.md).

## 7. Amendment — September 14, 2026

**Status:** reconciliation with the [September 13 deeper audit](DEEP_AUDIT_2026-09-13.md).
No experiment was run and no source was repaired for this amendment. Every claim
below is either labelled from one of the two audits or was re-verified against
live source and the current ledgers on 2026-09-14; those re-checks are marked
**re-verified**.

The deeper audit governs current interpretation and priority. This plan keeps
ownership of the mechanism-study controls: the deeper audit cites it for the
noninferiority controls (§1.1) and for the BINN frozen-encoder intervention (§7),
and does not restate either. What changes here is the order the stages run in, a
blocking prerequisite that did not exist on September 8, and four dated claims
that newer records have overtaken.

### 7.1 Superseded dated claims

These were accurate when written. They are not accurate now, and they are marked
in place at §1, §2, and §6.

| September 8 claim | Current status, re-verified 2026-09-14 | Effect |
|---|---|---|
| "No published-arm rows were found in the local MQAR ledgers inspected" (§1, E28 row; repeated in §6) | **False now.** [`mqar_e28_p8/runs.jsonl`](../../nanolab/out/mqar_e28_p8/runs.jsonl) holds four complete 15-seed arms: `gdn` 2/15 solved, `gdn_pub` 1/15, `hybrid_gdn_periodic` 6/15, `hybrid_gdn_periodic_pub` 9/15. | N0's "first comparison uses `gdn` and `gdn_pub`" now has a within-environment baseline instead of a gap. The rule-versus-full-block fidelity caution in the same row is unaffected. |
| "No W29 cell path was found in the local v3 directory scan" (§2, Existing BINN branches) | **Superseded.** BINN carries both `PREREG_2026-09-07_W29_…` and [`RESULT_2026-09-09_W29_THE_ASYMMETRY_HAS_ITS_OWN_BAR.md`](../../../BINN/results/RESULT_2026-09-09_W29_THE_ASYMMETRY_HAS_ITS_OWN_BAR.md). The deeper audit independently recomputed all 48 cells. | W29 is a completed registered result, not an unlocated branch. Its directional reading holds on its own platform; see §7.5 for what it does *not* establish. |
| "33 `.pt`/`.pth`/`.safetensors` files under `nanolab/out`, zero with MQAR/recall/E28 in their paths" (§1, checkpoint row) | **Still true.** Re-counted: 33 files, zero matching. [`run_one`](../../nanolab/mqar_suite.py#L167) still returns metrics and saves no weights. | Unchanged. N0's checkpoint-capture prerequisite still blocks every evaluation-only intervention study. |
| §5's prior-art set (Zoology, Revisiting associative recall, two activation-patching audits) | **Incomplete.** The deeper audit's §9 adds five close primary sources. | See §7.6. One of them overlaps the N-track premise directly. |

### 7.2 A blocking prerequisite that did not exist on September 8

The deeper audit reproduced two training-contract defects with the actual trainer
and actual batchers, not with fixtures standing in for them. Both are upstream of
every number this plan would produce, and neither is mentioned in §2–§4.

**Re-verified against live source:**

- The checkpoint payload at [`train.py:469`](../../nanolab/train.py#L469) stores
  `model`, `step`, `cfg`, `val_loss`, `tokens_seen`, `next_step`, and optimizer
  state. It stores no sampler state, and `train.py` contains no `get_state` /
  `set_state` / `rng_state` call anywhere. The [`Batcher`](../../nanolab/data.py#L364)
  owns a private generator seeded from `cfg.seed`, so a resumed batcher restarts
  its draw sequence. The audit's fixture measured a maximum final-parameter
  divergence of 0.0043052742 from uninterrupted training, falling to exactly 0.0
  when both sampler states are restored externally.
- [`train.py:357`](../../nanolab/train.py#L357) hands the *training* batcher to
  `evaluate`, which draws from it in a loop at [`train.py:61`](../../nanolab/train.py#L61).
  `eval_train`, `eval_iters`, and evaluation frequency therefore change the
  training sample sequence; they are not logging settings.

§4's stage table begins at "Assay validation … no training." That row must now
also gate the trainer, not only the task: **a discovery seed is not a matched
paired replicate until sampler state survives resume and train-evaluation draws
are separately sourced.** §4 already requires "Keep task RNG, model RNG, and
transformation RNG separate so equal seed labels actually yield matched data" —
that requirement is currently unmet by the code, which September 8 did not know.

This does not invalidate the existing ledgers. It means a resumed run is not
interchangeable with an uninterrupted one, and that any pairing argument resting
on seed labels across a resume boundary needs the repair first.

### 7.3 §4's statistics cannot be computed with the current reader

**Re-verified:** [`scripts/paired_board.py:45`](../../scripts/paired_board.py#L45)
defines `T4` as the two-sided t constants for **4 degrees of freedom**, and
`interval()` at [line 121](../../scripts/paired_board.py#L121) applies `T4` to
`gaps` of any length. The error is exact at five seeds and wrong in both
directions away from it:

| Seeds | Correct two-sided 95% t | Applied ÷ correct | Consequence |
|---:|---:|---:|---|
| 3 | 4.302653 | 0.645 | Interval **too narrow**; overstates confidence |
| 5 | 2.776445 | 1.000 | Exact — every existing five-seed interval is unaffected |
| 12 | 2.200985 | 1.261 | Interval **too wide**; understates precision |
| 32 | 2.039513 | 1.361 | Interval **too wide**; understates precision |

Both of §4's sample sizes are hit, in opposite directions. The **discovery**
stage specifies "three fresh paired training seeds per selected task/arm": at
n=3 this reader would report an interval 35.5% narrower than correct — the
deeper audit's §4 counterexample exactly. The **confirmation** stage specifies
"a fixed seed count in 12–32": there the interval is 26–36% too wide, so a design
powered to put a lower bound above a 3-point margin would systematically fail to
clear it even when the effect is real.

§4's power simulation is therefore not executable against the current tooling as
written. Its margins (3-point superiority, 1-point noninferiority) and its
6-point planning alternative are design choices and stand; what does not stand is
the assumption that the repository's reader will compute them. Related, from the
same audit section: `RECIPE_KEYS` at [line 46](../../scripts/paired_board.py#L46)
is `batch_size, block_size, eval_iters, max_steps, lr_max_steps, lr, matrix_lr,
warmup_steps, schedule, optimizer` — **re-verified**, and it omits compile,
dataset, dtype, gradient accumulation, `eval_train`, and weight decay, so the
guard does not establish a shared recipe. §4's "Freeze the handling of missing
pairs before training" now also requires freezing the identity fields.

### 7.4 Reordered execution

September 8 presented BINN (§2) before NanoLab (§3). The deeper audit's §10
places the BINN mechanism study third of four, and introduces a question this
plan does not contain at all. The merged order:

| Order | Work | Source | Status here |
|---:|---|---|---|
| 0 | Evidence contracts: sampler-state resume, independent train-evaluation sampling, full experiment identity, sample-size-aware intervals, directed crossing semantics. Regressions must fail against current behavior on real random batches. | Deeper audit priority 0 | **New and blocking.** See §7.2 and §7.3. |
| 1 | Global-retrieval allocation: confirm attention, late 8+4, and periodic 9+3 in the same long-recall cell with full recorded recipe, then vary attention count against position, and pairs against distance, separately. | Deeper audit priority 1 | **New.** Not in this plan. Reuses existing ledgers and needs no new instrument. |
| 2 | N0 → N1 → N2, broadened: the query-only task, the trace/replay instrument, the matched-filler contrast, and the persistent-transition factorial. | This plan §3, deeper audit priority 2 | **Retained, narrowed.** See §7.5. |
| 3 | B0 → B1 → B2: assay validation, frozen-encoder × head, temporal intervention. | This plan §2, deeper audit priority 3 | **Retained, narrowed.** See §7.5. |
| 4 | Scaling-paper completion: width-1152 confirmation at the selected rate, separated Adam/Muon group scaling, held-out width. | Deeper audit priority 4 | Outside this plan's scope. |

The reason order 1 precedes this plan's own stages: it is the cheapest
consequential question in the current record and it needs none of the new code
§3 gates on. The deeper audit's §1.1 records — and the ledger **re-verifies** at
[`mqar_e16_seq511/runs.jsonl`](../../nanolab/out/mqar_e16_seq511/runs.jsonl) —
that at sequence 511 periodic 9+3 solves 0/10 while late 8+4 and attention each
solve 8/10, with `mingru`, `gdn`, `swa_w64` and `swa_w64_nosink` all at 0/10 and
`hybrid_mingru10_attn2` at 0/8 of 10 planned. A layout with the better
language-model endpoint fails the capability assay outright. That is a larger
and better-evidenced gap than the query-interference effect §3 was built to
detect, and it is not identified: layout count and position are confounded, and
the task change moves pairs, length, batch size, and vocabulary demands together.

Order 1 does not replace §3. It is prior to it, and its outcome changes what §3
should be run on: N1's "frozen checkpoint" should be a layout the allocation
stage has shown to be informative, not one selected by CE endpoint.

### 7.5 Narrowings to the retained stages

**B2 needs count-matched controls, not only order controls.** §2's B2 table
distinguishes information destruction from model limitation, and §2 defers
count-matching to the later cue–value task. The deeper audit's §7 shows that is
too late. **Re-verified** at
[`shd_temporal.rs:792`](../../../BINN/binn-learn/src/shd_temporal.rs#L792): spike
dropout is a per-spike Bernoulli keep (`rng.next_u64() % 100 >= deleted_below`)
accumulating survivors, so at 90% deletion `E[kept | count] = 0.1 · count` and
relative count structure survives thinning in expectation. The audit's analytical
counterexample — equally likely classes at 100 versus 1,000 spikes, 90% thinning,
a count-only threshold at 40 — retains approximately 0.9999999999995 expected
accuracy. Consequence: a head that survives a thinning or shuffling manipulation
has not thereby been shown to use order. B2 must carry a count-matched arm from
the start, or its conclusions are confounded with count.

The same section qualifies W29: the attention arm loses about 6.46 percentage
points of absolute accuracy under thinning while its advantage over rate grows to
0.165967. "Attention is not hurt" is not a supportable reading of that result.

**N2's factorial is a component, not the frontier.** §3's four persistent
transitions (ordinary / decay-only / delta-only / identity) remain correctly
specified, and the algebra in §3 is unchanged. The deeper audit's §8 adds a
constraint the factorial alone cannot answer: protecting a spanning key set
forces `D = 0`, with the numerical sweep reaching projected-key norm below 1e−12
at rank 12, and the preservation cost `‖e‖ / ‖p‖` is the diagnostic quantity.
Deciding between "queries damage memory" and "this write rule cannot fit the
association under its protection constraint" needs the preconditioned,
covariance, and bounded-exception baselines from the audit's §9 — not the four
in-repository transitions compared only against each other. §3's advancement
gate ("advance to a learned write policy only if…") is unchanged but now also
requires clearing those baselines at matched stored bytes.

### 7.6 Claim boundary after the September 13 literature review

§5 said a useful contribution is "a measured, specific failure and selective
remedy that survives the controls above." That stands. What narrows is how much
of the N-track premise is already claimed elsewhere. From the deeper audit's §9,
reported as those papers' claims and not reproduced here:

- [Memoir, July 2026](https://arxiv.org/html/2607.20792v1) studies read-only
  pondering versus writing intermediate states into memory — the same
  read-versus-write separation §3 was designed to test — and reports that its
  small-model short-budget gap disappears at longer training. Query tokens and
  latent pondering are related interventions, not identical ones, but §3's
  "destructive questions" hypothesis is no longer an open framing.
- [HOLA, July 2026](https://arxiv.org/abs/2607.02303) routes a bounded exact
  cache by a committed-residual signal; [Hybrid Associative Memories, March 2026](https://arxiv.org/html/2603.22325v2)
  pairs recurrent compression with selectively retained key–value memory. Both
  are close baselines for anything N2 would propose as a write policy.
- [Preconditioned DeltaNet, April 2026](https://arxiv.org/html/2604.21100v1) and
  [Memory by Design, August 2026](https://arxiv.org/abs/2605.31163) cover
  curvature-aware recurrence and uncertainty-steered writes respectively.

The deeper audit states its search was targeted at seven close papers and does
not certify novelty; neither does this amendment. The practical effect is that
§3's result is publishable as a mechanism finding only with these comparisons
included at matched state and compute, which §4's expansion row should now name
explicitly.

### 7.7 What this amendment does not change

§2's B0/B1 design, §3's N0/N1 protocol and intervention algebra, §4's margins and
its rule that an interval crossing the threshold is inconclusive, and §6's
statement of what the September 8 probe did and did not execute. The September 8
probe artifacts remain valid for what they checked:
[second_audit_probe.py](evidence/second_audit_probe.py) was **re-run 2026-09-14**,
exits 0, and reports `"status": "PASS"` with all three groups (`mqar`, `gdn`,
`synthetic_order`) intact. The registered BINN waves keep their own hypotheses
and thresholds, and W28's NOT MET verdict is unchanged.
