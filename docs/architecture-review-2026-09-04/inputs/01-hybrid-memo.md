# Historical input: 01-hybrid-memo.md

Supplied in the September 4, 2026 architecture discussion. Preserved verbatim below; this is not the current verdict. **Hybrid curves and research overhaul; several mechanism, capacity, equivalence and cost claims are superseded.** See the [current assessment](../README.md) and [memory protocol](../MEMORY_UPDATE_PROTOCOL.md). Embedded labels such as “verified” belong to the supplied memo. Missing attachment glyphs are not working artifact links.

---

# What the record supports, what it does not, and where an architecture could come from

Written 2026-09-04 from a full read of the workspace: the paper, the master KB, the
experiment index, the 2026-08-26 backlog and its four September boards, the nanolab
model and mixer code, the APRDH trainer, the Metal training and inference notes, and
the committed per-seed run records under `nanolab/out/`. Every number below is either
copied from a committed artifact, recomputed today from `metrics.jsonl`, or labelled
as an estimate. Labels: **verified** (recomputed or read from a locked artifact),
**inferred** (a reading of verified numbers), **speculative** (a hypothesis).

---

## 0. Verdict in five lines

1. No architecture in this record, and none inferable from it, is an attention-scale
   discovery. The record's own thesis says why: one corpus, one 124M shape at 0.4
   tokens per parameter, 512 context, held-out cross-entropy plus one synthetic recall
   probe. That regime cannot certify a winner, which the paper proves eleven ways.
2. What the record *can* do is name the axis that an attention-class successor would
   have to win on, and it does so more sharply than most published work: the dividing
   line between mixers is the **recall mechanism** (key-addressed write and read), not
   recurrence versus attention. The cost axis that decides wall-clock winners is
   **kernel maturity and dispatch overhead**, not FLOPs.
3. One result the docs do not state is in the committed curves: a minGRU hybrid with
   four attention layers beats **both** pure arms at every evaluation from 1.65M to
   50M tokens on all five seeds, paired. It removes the attention/minGRU crossover the
   paper is about. This is the first "no-regret" architecture in the record.
4. The design the data would choose is: a cheap local mixer for most layers, three or
   four late recall layers, value residual, and depth by reuse with a per-token halt.
   That is roughly where the field has converged (Qwen3-Next, Jamba, Mixture of
   Recursions), which is evidence the inference is sound and evidence it is not new.
5. The highest-ceiling line that the data actually motivates is a **fast-weight recall
   mechanism that forms induction heads as fast as attention**. The record's own
   numbers (GDN 7/15 to 13/15 with budget, attention 15/15, minGRU 1/15 flat) say the
   gap between the delta rule and attention is head-formation speed, not capacity. The
   MQAR seed-rate instrument is the right tool to search that space, and it is built.

---

## 1. What the evidence base can and cannot say

| property | value | consequence |
|---|---|---|
| model shape | 12L / 768d, 123.7M params (E21 adds 384 and 1152 width) | one point in depth; width tested once |
| token budget | 50M on every board | **0.40 tokens/param, ~50x below compute-optimal** (verified: 50M / 123.7M) |
| context | 512 (E9 adds one 2048 board) | long-context claims are out of reach |
| data | FineWeb-Edu, sampled with replacement | one distribution |
| metric | held-out CE; MQAR recall (E8/E16) | no downstream tasks, no long-range benchmark |
| seeds | 5 (15 for recall rates) | signs, not magnitudes, below ~0.02 nats |

The 0.4 tokens/param figure is the single most important unstated fact about the
catalogue. In this regime the model is compute-per-token bound, not parameter bound,
and the record already shows it: adding 4.2x parameters at fixed compute buys at most
0.016 nats (E19, corrected control), removing half the parameters at fixed compute
costs 0.014 nats (E18, `looped_attn6x2`). Every "which mixer learns faster" result is
a statement about per-token learning speed in a data-starved regime. Whether the same
orderings hold at 20 tokens/param is the most valuable unrun question in the
workspace, and it is a **token** ladder, not the width ladder E21 ran.

---

## 2. New today: the hybrid that removes the crossover

**Method.** Per-seed evaluation curves from `nanolab/out/crossover50m` (attention,
minGRU; suite 22), `crossover50m_matched32` (suite 26), `crossover50m_ratioplace32`
(E10), `crossover50m_loop32` (E18), `crossover50m_moe32b` (E19b) and
`crossover_ladder50m` (E21). Same recipe throughout (batch 32, ctx 512, `eval_iters`
20, 50M cosine, GH200). Gaps are paired by seed at each shared token marker; intervals
are 95% Student-t, four degrees of freedom.

**Cross-suite pairing was checked, not assumed.** The attention arms rerun inside E18
and E19b reproduce suite 22's attention curve per seed at every marker checked (gap
−0.001 to +0.001, intervals ±0.002 to ±0.005), so pairing a hybrid from one suite
against attention from another at this recipe is sound. A within-suite attention arm
in `ratioplace32` (5 jobs; `lock_recipe` now lets arms grow) would remove the caveat
entirely.

**Result (verified).** Paired gap, arm minus attention, negative means the arm is
better; every cell below is 5 of 5 seeds on the sign unless marked:

| tokens | `mingru` | `hybrid_mingru10_attn2` | `hybrid_mingru_periodic` (9+3) | `hybrid_mingru8_attn4` |
|---|---|---|---|---|
| 0.84M | +0.056 | +0.039 | +0.037 | +0.032 |
| 4.11M | −0.228 | −0.256 | −0.274 | **−0.287** |
| 8.21M | −0.134 | −0.168 | −0.186 | **−0.209** |
| 12.30M | −0.002 (tie) | −0.058 | −0.084 | **−0.134** |
| 19.68M | +0.193 | +0.053 | +0.018 | **−0.024** |
| 32.78M | ~+0.21 | +0.028 | −0.008 | **−0.010** |
| 49.99M | +0.227 | +0.010 | −0.026 | **−0.018** [−0.022, −0.013] |

Mean-curve crossings against attention:

| arm | crossings (M tokens) |
|---|---|
| `mingru` | 1.05 (minGRU ahead), 12.35 (attention ahead) |
| `hybrid_mingru11_attn1` | 1.01, 12.93 |
| `hybrid_mingru_bookend` | 0.90, 12.72 |
| `hybrid_mingru10_attn2` | 0.97, 13.87 |
| `hybrid_mingru_periodic` | 0.96, 15.11, **27.65 (hybrid ahead again)** |
| `hybrid_mingru8_attn4` | **0.94 only — never re-crosses** |
| `hybrid_gdn_periodic` | 1.06, 4.22 |
| `gdn` | 1.26, 3.47 |

Three statements follow.

- **The attention-layer count sets the late crossing, monotonically.** One layer
  re-crosses at 12.9M, two at 13.9M, three at 15.1M and then re-wins at 27.7M, four
  never re-crosses. This converts E10's "count saturates at three" into a curve-level
  mechanism: each added attention layer buys the hybrid more of attention's late
  advantage without giving up minGRU's early one.
- **`hybrid_mingru8_attn4` is better than both pure arms at every marker after the
  first evaluation**, paired 5/5 with disjoint intervals at every marker against
  minGRU and at every marker except a 17–25M band against attention, where the paired
  gap is still 5/5 negative but the unpaired seed ranges overlap by up to 0.048 nats.
  State it as a paired claim.
- **The paper's 50M "tie" between attention and the best hybrid was an unpaired
  reading.** Paired by seed, 9+3 beats attention at 50M by 0.026 and 8+4 by 0.018,
  both 5/5, intervals excluding zero. Seed variance is common-mode across arms at this
  recipe (E10 already used this), so the paired test is the powered one.

**Two smaller findings from the same pass (verified).**

- At each arm's *own* learning rate (E21 phase 2) there is **one** crossing, not two:
  minGRU leads the first evaluation at every width, and attention overtakes at 10.7M,
  11.5M and 12.6M for widths 384, 768 and 1152. The paper's early crossing at ~1.05M
  is a property of the shared 6e-4 rate. The early minGRU lead grows with width
  (+0.37, +0.39, +0.46 nats at 4.1M).
- At context 2048 (E9) the late minGRU crossing moves from 12.35M to **16.0M**, and
  `hybrid_mingru10_attn2` leads attention until 18.2M. Sequence length does not
  reorder the final board, as the paper says, but it does move the crossing token.

**What this is and is not.** It is a token-matched claim under the E11 caveat: at
matched wall clock on a GH200 in PyTorch, attention beat the two-attention hybrid by
0.06, and 8+4 runs no faster. On Apple silicon, where minGRU measured 2.17x faster
than attention at T=256, the wall-clock reading would plausibly reverse; that is the
paper's own thesis and it is unmeasured for this arm. It is also unmeasured on recall.
Both are priced in §6.

---

## 3. What the data says an architecture must have

Each row is an inference from verified numbers; the pointer is where the number lives.

| requirement | evidence | grade |
|---|---|---|
| A cheap local mixer for most layers | minGRU leads attention by 0.23–0.46 nats through 4–8M tokens on every recipe and width; hybrids inherit the lead (§2) | verified → inferred |
| A key-addressed recall mechanism somewhere | minGRU forms induction heads on 2 of 75 runs at any budget; GDN 13/15 and attention 15/15 at p=4/9k (E8). At seq 255/511 only attention solves (E16) | verified |
| Only 3–4 recall layers out of 12, placed late | 9+3 and 8+4 saturate (E10); bookend loses to last-two by 0.028 on 5/5; §2 crossing table | verified |
| Value residual | the largest single architectural win in the record: 2.223 → 1.988 calibrated BPB; gating alone hurts (arch ladder) | verified (sprint proxy) |
| Depth by reuse is free per token, expensive per second | looped 3x4 beats attn3 by 0.136 at equal tokens and loses by 0.271 at equal wall clock (E18/E20) | verified |
| Parameters are not the bottleneck here | MoE 8 experts ≤ +0.016 nats over dense (E19, corrected) | verified, regime-bound |
| Attention must start sharp | µP's 1/d temperature left attention at 99.8% of uniform entropy and cost 0.41 nats; one term (E13) | verified |
| Weight tying caps recall | tied embeddings cap MQAR at 0.555 regardless of training; untied reaches 0.990 (E8 build notes). **Every CE board is tied.** | verified, consequence unexplored |

The composite: `mingru × 8–9, recall × 3–4 (late)` with value residual, untied readout,
a sharp attention temperature, and looping with a learned halt. Two things are worth
saying plainly about it. First, it is what the data selects, so it is the right base
for anything built here. Second, it is not new: the recall-layer ratio and placement
are where Qwen3-Next and the Gated DeltaNet hybrids landed, and looped depth with
per-token routing is Mixture of Recursions (2025). The record's contribution is that
it shows *why* those choices win on a token axis, with the crossing structure as the
mechanism, and that is publishable on its own.

---

## 4. Candidate directions, ranked

Ranked by ceiling × uniqueness of your assets, with the honest risk.

### 4.1 A fast-weight recall mechanism that forms heads as fast as attention

Attention's impact came from three things together: parallel training on the
hardware of the day, content-addressable memory, and scaling. Your record isolates
content-addressable memory as the decisive quality axis and hardware fit as the
decisive cost axis. The delta rule is already a content-addressable memory: the state
update `S ← αS + β(v − Sk)kᵀ` is one gradient step on ‖Sk − v‖², i.e. a linear model
trained online at test time. GDN's failure is not capacity, it is **formation speed**:
7/15 at 3k steps against attention's 15/15, and 0/15 at sequence 255 against 11/15.
minGRU's failure is structural: no key-addressed write at all.

The open design question the data poses is therefore: what is the cheapest state
update that forms an induction head as reliably and as early as softmax attention?
Candidates in the literature are the nonlinear fast-weight line (TTT, Titans, Atlas)
and the sharpened-kernel line (RWKV-7's vector-valued gates). The backlog declined
Titans as "a large lift orthogonal to both papers". The E8 grid argues the opposite:
it is the direct continuation of the one durable architectural finding in the record.

What you uniquely have: a seed-rate recall instrument with a calibration phase that
refuses to report an untrainable cell, a difficulty × budget grid that already
exposed budget artifacts, and Metal kernels for the delta rule. The experiment is
small models on MQAR first (the whole E8 grid was 405 runs on a rented box in a day),
then the 50M CE board only for arms that clear the recall gate. Grade: speculative
on outcome; the field is crowded; the ceiling is the only attention-class one here.

### 4.2 No-regret architecture selection, with a predictive theory of the crossing

§2 gives the first instance: an arm that dominates the pure arms over the whole token
axis. The paper's thesis is that rankings are recipe properties; the natural next
paper is that **some architectures have no crossing with their baselines**, and that
the crossing token is predictable. The mechanism is almost certainly induction-head
formation (Olsson et al., 2022): the late attention overtake at 12–16M is where the
attention stack acquires in-context copying, which E8 shows is bistable per seed and
strongly budget- and schedule-sensitive, exactly the properties of the crossing.

The test is cheap and uses existing curves plus one new probe: log an in-context copy
loss (loss on the second occurrence of a repeated random-token span) at every
evaluation of a 50M attention run and a minGRU run, and check whether the copy-loss
drop coincides with the crossing token on each seed, and whether the 20M-cosine and
batch-8 recipes move the copy-loss drop by the same amount they move the crossing.
If it does, you have a predictor of the quantity the paper says is unstable, which is
a much stronger result than a catalogue. Grade: inferred mechanism, cheap to test.

### 4.3 A decode-native hybrid for dispatch-bound, unified-memory hardware

The Metal track established that on an M5 Pro roughly 77% of a decode token is
dispatch and barrier overhead, quantized GEMV already sits at 62–100% of bandwidth,
and M>1 GEMMs reach the neural accelerators. Every path that won there widened the
decode step: DFlash 1.6x, NAX 1.5x. nanolab already trains block-causal attention and
has a lossless self-speculative decoder verified against greedy, and `tessl` has ICB
capture for decode graphs. The architecture this points to is the §3 hybrid whose
three or four attention layers are trained block-causal from the start, so the model
emits a block per step natively rather than through a separate drafter. Its
per-token cost is dominated by the minGRU trunk, which is 2.17x faster than attention
on Metal at T=256. Nobody else has this exact stack. Ceiling: a real product-class
result on Apple silicon, not a field-changing one. Cost is engineering, not GPU.

### 4.4 Adaptive-depth looping with a wall-clock-honest halt

E18/E20 is the cleanest demonstration in the record of "quality per token versus
quality per second". A looped stack with a per-token halt (the APRDH controller idea,
or Mixture of Recursions) is the obvious reconciliation: pay 12-layer compute only on
the tokens that need it. The unique contribution would be to evaluate it under E20's
wall-clock protocol, which the MoR line did not do with this rigor. APRDH as written
is too many mechanisms at once (patching, engram, dual-scan GDN, MLA routing, TTT
adapters, the controller); salvage only the halt controller into the looped
transformer that already works, with `attn3` and `attn6` as the controls E18 built.
Grade: crowded, medium ceiling, but every piece exists.

### 4.5 The screen auditor as a product

The most transferable artifact in the workspace is not a model. It is the harness:
recipe fingerprints, `lock_recipe`, wall-clock verification with tenancy as a field,
`final_val` instead of `best_val`, paired-seed boards, Student-t at n=2, and the
"a check that could not run must not report the same word as one that passed" rule.
Packaged as a standalone tool that takes any trainer's `metrics.jsonl` and emits a
recipe-dependence report, it would change more practice than another mixer. Not an
architecture; highest certainty of impact.

---

## 5. The overhaul: stop, change, start

**Stop.**

- Adding recipe-axis entries to §6. Eleven axes established the thesis; the twelfth
  does not change what a reader believes. The exception is the µP closers already
  queued (E22b, 20 jobs), which are cheap and close a caveat in the headline.
- Optimizer-axis work. D7 is correctly closed at "retired, not reversed".
- APRDH as a unit. Keep the controller idea (4.4), retire the rest.
- E12 at context 512. E16 already answered the window mechanism (recall returns
  monotonically as the window covers the distance, and the 512-window arm reproduces
  attention exactly). Only the context-2048 arm (E15) is still informative.

**Change.**

- **Report every board paired by seed.** It is already the powered test in E10; the
  §4.5 tie and the E10 "not resolved" placement call both change under it.
- **Untie the embeddings on the CE boards, or at least run the 2×2** of tied/untied
  × value-residual on/off. Every quality result was measured through a readout that
  structurally caps recall. The hypothesis that value residual's win is partly a
  workaround for tying (it gives every layer a non-embedding token representation)
  is testable in 20 jobs.
- **Ladder tokens, not width.** E21 showed width does not move the ranking. The
  regime is 0.4 tokens/param. Run the attention/minGRU/8+4 triple at 5 and 20
  tokens/param on the 60M (`w384`) shape, where it is affordable (§6).
- **Adopt a capability panel** beside CE: MQAR at three difficulties, a repeated-span
  copy loss logged every eval, and one long-range synthetic. The CE/recall
  disagreement (`hybrid_gdn_periodic` loses on CE with disjoint intervals and ties or
  leads on recall) shows CE alone under-rates the thing hybrids exist for.
- **Restore or rebuild the Metal corpus** (`fineweb10B_sp1024`). Until then the Metal
  track cannot produce a comparable number, and 4.3 depends on it.

**Start.** The experiments in §6, in that order.

---

## 6. Experiments, priced

Rates from the record: attention 33.5k tok/s and minGRU hybrids ~35k tok/s at
tenancy 3 on the GH200 (E20); `w384` attention roughly 2× that (estimate); GH200
$2.29/h. Hours are estimates unless a suite of the same shape has run.

| # | experiment | jobs | GH200-h (est.) | question | pre-registered readout |
|---|---|---|---|---|---|
| 1 | attention arm inside `ratioplace32` | 5 | ~1.5 | make §2 within-suite | 8+4 beats attention at 50M paired 5/5 → claim stands; else §2 is cross-suite only |
| 2 | copy-loss probe on the attention and minGRU 50M curves, 3 recipes | 30 | ~10 | is the crossing the induction bump? | copy-loss drop within ±1M of the crossing on ≥4/5 seeds per recipe → predictive; else mechanism withdrawn |
| 3 | 2×2 tied/untied × value residual, attention arm, 50M | 20 | ~7 | does tying carry part of the VR win? | VR gain shrinks by >50% untied → interaction; unchanged → independent |
| 4 | 8+4 hybrid on the recall grid (p=4,8 × 3k,9k, 15 seeds) | 60 | ~3 | does the no-regret arm also recall? | rate within Wilson interval of attention at every cell → no-regret on both metrics |
| 5 | wall-clock board with 8+4, on GH200 and on the M5 Pro | 10 + Metal | ~4 + local | does the cost basis reverse by hardware? | hybrid wins on Metal and loses on GH200 → hardware-lottery result, publishable |
| 6 | token ladder: attention, minGRU, 8+4 at `w384`, 300M and 1.2B tokens, 5 seeds | 30 | ~60–90 | do the orderings survive leaving the data-starved regime? | crossing token scales with budget → regime artifact; ordering at 20 tok/param is the durable one |
| 7 | fast-weight recall screen: GDN, GDN + previous-token shift, nonlinear fast-weight (TTT-style), on MQAR p=4..8, 15 seeds | ~90 | ~5 | which state update forms heads fastest? | any arm matching attention's 15/15 at 3k earns a 50M CE seat |
| 8 | looped 6x2 + learned halt, E20 protocol, with `attn6` control | 15 | ~5 | does a halt recover the wall-clock loss? | halted arm within 0.02 of `attn6` at equal clock and ahead of it at equal tokens → depth is bought back |

Items 1–4 are under $60 total and settle whether §2 is a paper. Item 6 is the one
that answers the reviewer objection E21 could not.

---

## 7. Evidence grades for the claims in this memo

| claim | grade |
|---|---|
| 8+4 hybrid beats attention paired 5/5 at every marker ≥1.65M; beats minGRU at every marker | **verified** (recomputed today; cross-suite pairing validated by two fresh attention arms) |
| attention-layer count sets the late crossing monotonically | **verified** (mean-curve interpolation; 5 arms) |
| one crossing, not two, at tuned learning rates; crossing at 10.7/11.5/12.6M by width | **verified** (E21 phase-2 curves) |
| ctx 2048 moves the minGRU crossing to 16.0M | **verified** (E9 curves) |
| 0.4 tokens/param; parameters not the bottleneck at this budget | **verified** (E18/E19) |
| the crossing is induction-head formation | **inferred**, testable (item 2) |
| value residual interacts with weight tying | **speculative**, testable (item 3) |
| a fast-weight mechanism can match attention's head-formation speed | **speculative**; the only attention-class ceiling here |
| the hybrid wins on wall clock on Apple silicon | **inferred** from 2.17× minGRU throughput; unmeasured |
