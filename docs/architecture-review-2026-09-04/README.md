# Architecture research review — September 4, 2026

This is the consolidated assessment from the architecture-review discussion. It
records the experiment evidence, the supplied proposals, their corrections, and
the resulting research protocol. **No attention-scale architectural discovery is
established. A focused investigation of binding, memory updates, and retrieval is
justified.** The current recommendation is to run diagnostic comparisons before
selecting a new memory layer.

## Reading order and authority

**September 8 revision:** the [second audit and revised experiment plan](SECOND_AUDIT_PLAN_2026-09-08.md)
now governs the ordering and controls for the BINN and NanoLab mechanism studies.
It corrects the frozen-substrate assumption, teacher-forced query task, state-reset
interpretation, and missing replay/checkpoint prerequisites. The dated findings
below remain historical evidence; the mathematical protocol remains applicable.

1. This page: current verdict, corrections, and research priorities.
2. [Evidence review](EVIDENCE_REVIEW.md): the original broad review of MLSystemsLab
   and related BINN findings, with dated snapshot boundaries and supersession notes.
3. [Memory-update protocol](MEMORY_UPDATE_PROTOCOL.md): association construction,
   update direction, finite-error retention checks, oracles, and experiments.
4. [Evidence and reproduction](evidence/README.md): saved values, source hashes,
   portable CPU probes, and commands for fresh local checks.

The four supplied memos are preserved as historical inputs, including proposals
and estimates that were later rejected or narrowed:

- [Hybrid curves and overhaul](inputs/01-hybrid-memo.md).
- [Retention-budgeted scalar commits](inputs/02-retention-memo.md).
- [Direction-selection revision](inputs/03-direction-revision.md).
- [Efficiency, serving cost, and reconciliation](inputs/04-efficiency-reconciliation-memo.md)
  (version 3.1 of the first memo; supersedes 01 where they differ).

These inputs are not independent experiments. Their internal “verified” labels
do not override this review's evidence grades. The final protocol supersedes the
earlier recommendation to make scalar gating the leading architecture. Source
hashes and import transformations are in [input provenance](evidence/input-provenance.json).

## Evidence grades and time boundaries

| Label | Meaning here |
|---|---|
| Verified | Recomputed from accessible run records, checked against source, or established by a stated analytical/numerical check. This does not imply independent reproduction of training. |
| Inferred | An explanation or recommendation consistent with the evidence but not causally established. |
| Proposed | A future experiment, operator, decision margin, or resource allocation. |
| Report-backed / unverified | Stated in a supplied memo or external report; the underlying records were unavailable for independent checking. |
| Superseded | An earlier interpretation explicitly withdrawn or narrowed below. |

The broad inventory is frozen at **2026-09-04 21:48 UTC**, Git HEAD
`9ccc0530532ed04af07141a7a52eee498790271b`, in a dirty shared checkout. It covers
3,463 result artifacts, 3,140 distinct hashes, 1,252 metrics files, 448 crossover
metrics files in 18 directories, and 591 MQAR records. Those are file/record counts,
not independent experiment counts. Its final-CE tables contain 241 runs with an
explicit finite `final_val`. Queue counts in that snapshot are historical local
labels, not a live remote-job status.

The later discussion checked periodic learning-curve evaluations separately.
Neither importing the snapshot nor adding these docs launches training, repairs
the identified model issues, publishes results, or validates every systems
benchmark. External BINN cell artifacts are not bundled here.

## Status on September 5, 2026

Dated additions; nothing above this section was rewritten.

- **The comparators exist as code.** `1b6c5d0` (on `main`) adds `Config.gdn_rule`
  (`repo` | `published`), `Config.moe_router_weight` (`renorm` | `raw`),
  `Config.mingru_expand` and `Config.copy_probe`, each defaulting to the recorded
  behaviour; sixteen arms (`gdn_pub`, `hybrid_gdn_periodic_pub`, `moe_e{1,4,8}k1_raw`,
  `mingru_x1`, `hybrid_mingru8_attn4_x1`, `hybrid_mingru_periodic_x1`, `attention_untied`,
  `attention_novr`, `attention_untied_novr`, `attn6_w512`, `attn6_w576`,
  `w384_hybrid_mingru8_attn4_lr40/lr80`); stage scripts `scripts/e28_gdn_rule.sh` …
  `scripts/e35_w384_ladder.sh`; `scripts/paired_board.py`; five new tests (172/172 pass).
  The [Lambda handoff](LAMBDA_HANDOFF_2026-09-05.md) is the operating document.
- **Costs were measured, not divided.** A tuning sprint on the GH200
  (`docs/GPU_TUNING_2026-09-05.md`, on branch `claude/lambda-gh200-sept-2026-f54de2` at
  the time of writing) timed every arm at the board shape. Input 04 §5.1 priced arms by
  dividing recorded elapsed time by tenancy; for a saturating arm at three workers that
  overstates per-job cost by about 2.7x, and the recall cells were under-priced several
  times over. The handoff's §5 carries the re-priced program.
- **The 8+4 hybrid's throughput deficit is an expansion cost.** At expansion 1 the
  parity hybrid trains at attention's rate (131.2K vs 130.4K tok/s, sprint table A) with
  +3.8% parameters, where the recorded hybrid runs at 0.85x with +19%. E30 decides
  whether the CE margin survives; if it does, the parameter and throughput objections
  fall together.
- **A determinism floor now exists.** Two runs of one configuration and seed land
  0.0014 nats apart in `final_val` (max 0.0031, ten runs, 5M tokens). The paired 8+4
  margin of −0.0176 is about twelve times that floor; any future per-seed claim below
  ~0.003 nats is indistinguishable from a rerun.
- **E27 (width 1536) is half done.** Its minGRU learning-rate probe is complete (lr40,
  interior; lr20 is tied within the floor); the minGRU arm is running; four of five
  attention jobs OOMed when the repair collided with a tuning stage and need one relaunch
  (handoff §2.2).
- **BINN moved after this review's snapshot.** Wave 27 landed on 2026-09-05: the
  read-out's order-dependence has a half-life of 261 ms at the headline anchor; the
  QK-norm arm this review left open is NOT MET (a different read-out, −0.0692 accuracy);
  the p30 dropout instrument is insensitive and wave 28 is registered to find the rate
  at which it bites. The wave 26 and 27 cells this review could not see are tracked in
  BINN's repository (`results/shd_attention_campaign_v3/`, 530 cells). BINN's own
  number and verdict sweeps did not yet cover waves 26–27 on that date; the audit record
  in BINN (`results/AUDIT_2026-09-05_…`) states this against the record.

## What the hybrid curves establish

The paired learning-curve result is a useful addition to the endpoint review.
For seeds 42, 100, 777, 1337, and 2026, `hybrid_mingru8_attn4` from
`crossover50m_ratioplace32` beats both pure arms from `crossover50m` at **all 60
saved evaluations after the first**, from 1,654,784 to 49,987,584 tokens. These
are 60 repeated measurements of five trajectories, not 300 independent runs.

| Comparison at the last shared periodic evaluation | Mean CE difference | Pointwise 95% paired t interval | Seeds favoring hybrid |
|---|---:|---|---:|
| 8+4 minus attention | −0.017572 | [−0.022152, −0.012993] | 5/5 |
| 8+4 minus minGRU | −0.244821 | [−0.259369, −0.230273] | 5/5 |
| Periodic 9+3 minus attention | −0.026163 | [−0.029216, −0.023110] | 5/5 |

These values use **`eval.val_loss` at shared token markers**, not the separate
terminal `done.final_val` evaluation and not `best_val`. This distinction explains
why the means differ slightly from the [final-CE tables](evidence/evidence-tables.md).
The [full curve comparisons](evidence/memo-curve-check.json) preserve every marker
and seed delta, including attention rerun comparisons.

The 8+4 hybrid has **147,217,732 parameters** versus attention's **123,699,612**,
about **19.01% more**. Pure minGRU has 158,976,792 and periodic 9+3 has 150,157,497.
The same shared recipe does not establish equal optimization quality across arms.
The comparison is cross-suite; a within-suite replication with fresh confirmation
seeds remains useful. Attention reruns provide a consistency check, not proof that
all cross-suite differences are absent.

The pointwise intervals are exploratory. At the 99.9% pointwise level, a rough
allowance for the roughly 56 arm-by-marker comparisons scanned before this arm was
chosen, the 8+4 gap still excludes zero at every marker from 4.1M tokens: at 50M it is
[−0.032, −0.003], at 32.8M [−0.020, −0.000] (input 04, §2). Five negative signs remain
a sign statement, not a test.

Use “dominates the observed periodic evaluations under this recipe,” not a general
“no-regret architecture.” Equal-parameter, equal-time, independently tuned,
longer-horizon, and capability-panel comparisons remain proposed. Keep both 8+4
and periodic 9+3 as references: the former has the observed trajectory advantage,
while the latter has the better endpoint. Periodic 9+3 is not three late attention
layers; count, placement, and parameter count must be distinguished.

## Corrections that govern future claims

| Earlier claim or suggestion | Current interpretation |
|---|---|
| GDN fails because heads form slowly, not because of capacity | More training helps the small-pair cell. Binding, conditioning, update dynamics, storage, and retrieval remain alternatives at larger loads. MQAR success is not circuit identification. |
| The CE crossing is induction-head formation | A repeated-span copy-loss probe can show temporal association. Causal attribution requires targeted interventions and prediction on held-out recipes, separating tokens, optimizer steps, and LR phase. |
| Three or four late recall layers are selected by the data | Keep the tested hybrid layouts as baselines. The periodic and late placements differ, and a ratio sweep also changes parameters. |
| Successful ingredients select the first memo's §3 composite (that memo itself retired APRDH as a unit) | Value residual, reuse, routing, byte patching, and untied readout need interaction tests. Their separate successes do not establish an optimal combination. |
| Corrected MoE results establish that more expert capacity does not help | The one-expert loss correction is exact for that control. Unknown multi-expert auxiliary loss and the top-1 task-gradient issue prevent a general capacity/specialization conclusion. The `moe32c` worker log synced at 17:05 on 2026-09-04 shows `moe_e4k1` in progress at step 2000; the two running and six pending `e4k1`/`e8k1` jobs train a router that receives only the balancing gradient. Whether to stop them is an open decision. |
| Weight tying structurally caps all model recall | A probe-specific calibration failure motivates a factorial experiment, not a universal ceiling. Untying at width 768 adds 38,633,472 parameters. |
| Block-causal attention emits a block per decoding step | The mask alone supplies no generation algorithm; the algorithms the first memo relied on are nanolab's block-diffusion and lossless self-speculative decoders (`nanolab/diffusion.py`; `nanolab/out/bench_block_scale.txt` records 68–121 tok/s at 124M with 8 steps per 32-token block on CUDA). They exist for the attention mixer only. Specify the objective, inference iterations, quality, and execution cost for a hybrid before claiming a block per step. |
| Five wins or overlapping Wilson intervals establish equivalence | Five paired signs give a two-sided sign-test p=0.0625. Magnitude-based paired analysis can be informative; equivalence/noninferiority needs a specified margin and an interval for the difference. |
| The wall-clock loop board contains an under-budget six-block seed | Withdrawn later on 2026-09-04: `attn6` seed 777 reached its full pre-registered 35,602,432-token budget. It finished in 221 s instead of ~672 s because it started after every other job had ended and ran alone (tenancy 1 instead of 3). Tokens, not seconds, were the controlled variable, and every arm's budget was calibrated once at tenancy 3, so the loss board is matched by construction. The missing field is tenancy per run. |
| Prediction drift measures harmful interference | Old squared-error change includes a residual cross-term. Moving a prediction can repair it. Drift remains a comparator. |
| A blocked scalar write means memory is full | The chosen direction may be unsuitable. Compare it with redirected writes and an oracle under the same constraints. |
| A projected write solves retention | It preserves monitored predictions, including their errors. It may harm unmonitored queries, require large norms, or consume substantial computation. |
| A small aggregate CE regression protects retrieval quality | A four-nat loss increase on 0.5% of tokens contributes only 0.02 aggregate nats. Measure capability-specific loss too. |
| Per-job GPU cost is a suite's elapsed time divided by its tenancy (input 04 §5.1, §7) | Added 2026-09-05. Without MPS, co-resident processes time-slice: saturating arms (attention, minGRU, their hybrids, ~13% MFU) lose ~9% at any tenancy ≥ 2, dispatch-bound arms gain (GDN 1.53x). Elapsed ÷ tenancy overstates a saturating arm's cost by ~2.7x at three workers. Use the sprint's per-arm table (`docs/GPU_TUNING_2026-09-05.md`, table A). |
| `compile = False` is forced by an Inductor stall on GH200 aarch64 | Added 2026-09-05. torch 2.7.0 on the same box compiles attention in 31 s at 1.94x. `compile` is now a recorded recipe field, default off; turning it on is a numerics decision (0.0023 nats mean shift), not a hardware constraint. |
| BINN's wave-26 cells are unavailable | Added 2026-09-05. They were unavailable to this review's snapshot. They are tracked in BINN's repository under `results/shd_attention_campaign_v3/` together with wave 27's. |

The [GDN probe](evidence/gdn_rule_probe.py) reproduces the difference between the
implemented recurrence and the published decayed-read rule. This does not prove
that the difference caused recall failure. A unit-key sign-changing transition can
still be nonexpansive. The [MoE probe](evidence/router_gradient_probe.py) reproduces
the near-zero top-1 task-router gradient while expert and auxiliary gradients remain
nonzero. These are comparator findings, not source fixes delivered by this review.

## Research order

### 1. Repair and diagnose

Establish an explicit implemented-GDN versus published-GDN comparison; resolve the
router's task-gradient path before interpreting expert specialization (the `moe32c`
queue was still running `e4k1` at 17:05 on 2026-09-04); use task-only CE. Preserve
historical experiment identities. Record tenancy per run on wall-clock boards: the
loop board's token budgets were calibrated once at tenancy 3 and every seed reached
its budget, so it is matched by construction (see the corrections table).
*2026-09-05:* both comparators exist behind default-preserving flags (`gdn_rule`,
`moe_router_weight`); E28 and E29 in the handoff are the boards that run them.

Then use oracle binding, controlled compressibility, fixed-key least squares, and
the constrained-update diagnostics in the [protocol](MEMORY_UPDATE_PROTOCOL.md).
Hold vocabulary and query count fixed when isolating load or distance. Preserve
causal information timing and test unseen associations and relevant held-out symbols.

### 2. Compare the smallest memory operations

At identical states compare ordinary delta, scalar-limited delta, output-preserving
projection with a norm cap, preconditioning with/without the same error check, and
the constrained oracle. Follow with separate rollouts, repairable witnesses,
legitimate overwrites, and protected versus unprotected retrieval. Compare identical
witness storage exposed as a direct cache. No learned controller is required initially.

### 3. Extend horizons and tuning together

Retain attention, minGRU, 8+4, and periodic 9+3 as relevant references. Extend token
budgets with equal tuning budgets, multiple state-byte budgets, and fresh seeds.
The 50M/123.7M ratio is about 0.40 tokens per parameter; it is not itself a universal
compute-optimality prescription. The recorded width-384 attention/minGRU shapes are
about 40.6M/49.4M parameters, not a common 60M. Calculate each run's actual
tokens/parameter ratio instead of reusing the first memo's approximate labels.

The tied/untied × value-residual factorial is useful with its added parameters
accounted for. Aggregate CE, recall at several difficulties, repeated-span copy loss,
and a long-range task should be reported together. Estimate power from pilots and
use independent confirmation after architecture selection.

### 4. Keep secondary branches diagnostic

- **Anchored recurrent depth:** the current loop already preserves its first value
  anchor. Compare fixed/refreshed/absent anchors, fixed/random/learned pass budgets,
  and actual skipped execution. Computing then masking is not saved compute.
- **Query-time writes:** branch from identical stored states and compare ordinary
  queries, disabled persistent writes, and read-only refinement. Keep temporary
  computation available and measure query-order effects on later answers.
- **BINN:** separate temporal binding from storage with variable cue–value delays,
  overlapping entities, and distractors. Match count/order/coincidence tasks and
  test bin-width and timescale transfer. W25's recurrent mechanism was unevaluable;
  W26 is report-backed here and its saturation effect was not collapse-specific.
  *2026-09-05:* W27 measured the timescale (τ½ = 261 ms, one anchor) and rejected the
  QK-norm arm as a drop-in; the delay/entity/distractor factorial asked for here is not
  registered in BINN. Its record's audit of this review is
  `BINN/results/AUDIT_2026-09-05_THE_CROSS_REPOSITORY_REVIEW_READ_AGAINST_THE_RECORD.md`.
- **Metal/decode:** select a modeling hypothesis before kernel co-design. Measure
  training, prefill, decode, state bytes, routing, and synchronization separately.
  A restored/rebuilt corpus needs explicit tokenizer provenance and new baselines
  if its tokenization differs. Historical throughput is not a new hybrid benchmark.
  Input 04 (§5) collects the record's own throughput, decode, KV/state-byte, and
  quantization numbers in one place and proposes a third matching rule beside tokens
  and wall clock: arms matched on decode latency or resident bytes after a fixed
  training spend. The rule is proposed; the hybrid arms' serving numbers are estimates
  (nanolab has cached decode for the attention mixer only), and it should be applied
  only after the diagnostic stages above select an operator.
- **Other work:** retain the measurement paper, optimizer results, and an optional
  reusable experiment-auditing tool as separate outputs. D7 remains retired, not
  reversed. Additional recipe axes should close a specific uncertainty.

The first memo's dollar/hour estimates, run ordering, and “under $60” total are
historical proposals, not a verified budget or permission to launch paid jobs.
Reprice from actual shape, evaluation overhead, concurrency/tenancy, provider rate,
and expected tuning/retry work before execution.

## What would justify an architecture claim

A useful candidate must explain a reproducible failure, improve its predicted
mechanism through intervention, and retain an advantage over close prior work and
equivalent storage after actual costs are charged. A mathematically valid retention
check, a perfect fit to selected witnesses, or a favorable short CE curve is not
enough on its own. The [protocol's advance conditions](MEMORY_UPDATE_PROTOCOL.md#advance-conditions)
make those decisions explicit.
