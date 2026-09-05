# Historical input: 04-efficiency-reconciliation-memo.md

Supplied in the September 4, 2026 architecture discussion, after the evidence review. Preserved verbatim below; this is not the current verdict. **Version 3.1 of the first memo: adds the training/inference efficiency analysis and the iso-serving-cost matching rule (§5), the reconciliation with the evidence review (§9), experiments 9–17, and corrections to the first memo's value-residual, parameter-count and width-384 statements. Its serving-cost numbers for the hybrid arms are estimates; no decode benchmark of those arms exists.** See the [current assessment](../README.md) and [memory protocol](../MEMORY_UPDATE_PROTOCOL.md). Embedded labels such as “verified” belong to the supplied memo.

---

# What the record supports, what it does not, and where an architecture could come from

Written 2026-09-04 from a full read of the workspace: the paper, the master KB, the
experiment index, the 2026-08-26 backlog and its four September boards, the nanolab
model and mixer code, the APRDH trainer, the Metal training and inference notes, and
the committed per-seed run records under `nanolab/out/`, and (§5) the throughput,
decode, export and kernel records of the Metal track; §9 reconciles a second,
independent review received the same day. Every number below is either
copied from a committed artifact, recomputed today from `metrics.jsonl`, or labelled
as an estimate. Labels: **verified** (recomputed or read from a locked artifact),
**inferred** (a reading of verified numbers), **speculative** (a hypothesis).

---

## 0. Verdict in seven lines

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
6. Under a deployment cost basis the winners change again. At batch 1 on Apple
   silicon a 124M-class model is dispatch-bound by roughly 20x before bandwidth
   matters, so launches per token decide latency, not bytes or FLOPs; the three-layer
   arm E20 crowned on wall clock is also the serving winner; looping is near-free at
   server batch sizes and paid twice at batch 1; and the no-regret hybrid is 19%
   heavier than attention, which no board states. The deployable form of the §3
   design is §5.4, and the missing instrument is an iso-serving-cost board (item 12).
7. A second review received today (§9) found two code-level facts I had missed, and
   I have reproduced both on the real functions. The repository's Gated DeltaNet
   applies its delta correction to the undecayed state, a different operator from
   the published rule that can flip the sign of a stored association whenever β
   exceeds α; and the top-1 MoE router receives no task gradient, only the balancing
   one. Every GDN result in the record is a result about that variant, and the
   `moe_e4k1`/`moe_e8k1` arms still queued in `moe32c` are training a router that
   cannot learn to route. Both precede any further GDN or MoE spend.

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
0.016 nats (E19, corrected control; the router in those arms could not learn to
route, §9.1, so this is a hash-routed MoE number), removing half the parameters at fixed compute
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
  State it as a paired claim. Five negative signs is p = 0.0625 under a two-sided
  sign test; the interval is the test. Because about eight arms at seven markers were
  scanned before this one was chosen, the 99.9% interval is the honest one: at 50M
  [−0.032, −0.003], at 32.8M [−0.020, −0.000], and excluding zero at every marker
  from 4.1M; 9+3 at 50M [−0.036, −0.017].
- **The paper's 50M "tie" between attention and the best hybrid was an unpaired
  reading.** Paired by seed, 9+3 beats attention at 50M by 0.026 and 8+4 by 0.018,
  both 5/5, intervals excluding zero. Seed variance is common-mode across arms at this
  recipe (E10 already used this), so the paired test is the powered one.

**Two smaller findings from the same pass (verified).**

- At each arm's *own* learning rate (E21 phase 2; the rates were chosen at 10M tokens
  and spent at 50M, the ladder's stated limitation) there is **one** crossing, not two:
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
paper's own thesis and it is unmeasured for this arm. The 8+4 arm is unmeasured on
recall; the 10+2 arm reached 13/15 at p=8/9k against attention's 12/15 on the E8
board (overlapping intervals), so the family has recall evidence already.
It is also not parameter-matched: 8+4 carries 19% more parameters and 13% more FLOPs
per token than attention (§5.2). All three are priced in §7.

---

## 3. What the data says an architecture must have

Each row is an inference from verified numbers; the pointer is where the number lives.

| requirement | evidence | grade |
|---|---|---|
| A cheap local mixer for most layers | minGRU leads attention by 0.23–0.46 nats through 4–8M tokens on every recipe and width; hybrids inherit the lead (§2) | verified → inferred |
| A key-addressed recall mechanism somewhere | minGRU forms induction heads on 2 of 75 runs at any budget; the repository's GDN variant (§9.1) 13/15 and attention 15/15 at p=4/9k (E8). At seq 255/511 only attention solves (E16) | verified |
| Only 3–4 recall layers out of 12, placed late | 9+3 and 8+4 saturate (E10); bookend loses to last-two by 0.028 on 5/5; §2 crossing table | verified |
| Value residual | supported relative to gating-alone at the long stage: `value_resid` 1.9875 vs `gated_attn` 2.0887 calibrated BPB (2 seeds, 4L×128 proxy). The control was not promoted to the long stage, and at the short stage the control beat `value_resid` (2.7037 vs 2.7303). "Largest win in the record" and "gating alone hurts" are both stronger than the ladder supports | verified (sprint proxy); vs control **unestablished** (item 17) |
| Depth by reuse is free per token, expensive per second | looped 3x4 beats attn3 by 0.136 at equal tokens and loses by 0.271 at equal wall clock (E18/E20) | verified |
| Parameters are not the bottleneck here | MoE 8 experts ≤ +0.016 nats over dense (E19, corrected) | verified, regime-bound |
| Attention must start sharp | µP's 1/d temperature left attention at 99.8% of uniform entropy and cost 0.41 nats; one term (E13) | verified |
| Launch count, not FLOPs, sets batch-1 latency on unified memory | Gemma roofline: ~77% of a decode token is dispatch and barrier overhead over ~780 launches; quantized GEMV already at 62–100% of bandwidth (§5.2) | verified on Gemma; transfer to these arms inferred |
| Weight tying caps recall | tied embeddings cap MQAR at 0.555 regardless of training; untied reaches 0.990 (E8 build notes). **Every CE board is tied.** | verified, consequence unexplored |

The composite: `mingru × 8–9, recall × 3–4 (late)` with value residual, untied readout,
a sharp attention temperature, and looping with a learned halt; §5.4 gives its
deployable form. Two things are worth
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

One caveat now governs this paragraph. The GDN in this repository is not the published
operator (§9.1): its correction term reads the undecayed state, so the transition of a
stored association along its own key is α−β rather than α(1−β), negative whenever β
exceeds α. Whether that is why the variant fails the hard cells is unknown; it is the
first experiment (item 14), not the fast-weight screen. The second review's proposal
(§9.3), a write policy that scores the damage a write does to a protected set of
associations and diverts damaging writes to a small budgeted explicit store, is the
most concrete instantiation of this line I have seen; its nearest neighbours are Hybrid
Associative Memories (2603.22325, prediction-error routing) and Gated DeltaNet-2
(2605.22791, separate erase and write gates), both confirmed today. It is item 16.

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

## 5. Training and inference efficiency: what the record measures, and the deployment-first design

Everything above ranks arms per training token. Deployment ranks them on three other
axes: seconds of training on the hardware you own, cost per decoded token on the
hardware users own, and bytes resident in memory. The record has measured pieces of
all three and never put them in one place. This section does, states the one design
that is best or near-best on every axis using parts that already exist here, and names
the place where the token-matched winner and the deployment winner disagree.

### 5.1 Training throughput per arm, from the run records (verified)

Median per-step tok/s from the committed `metrics.jsonl`. Tenancy (jobs per GPU)
moves absolute rates by up to 3x, so ratios are within suite only; the E10 rows are
compared with the E18 attention arm, same tenancy, same box.

| suite (jobs/GPU) | arm | tok/s | ratio to attention |
|---|---|---|---|
| E11 wall-clock (1) | attention | 130.9k | 1.00 |
| | `hybrid_mingru10_attn2` | 106.6k | 0.81 |
| | `hybrid_gdn_periodic` | 34.9k | 0.27 |
| | `hybrid_gdn_bookend` | 31.7k | 0.24 |
| E18/E20 (3) | `attn3` | 93.4k | 2.36 |
| | `attn6` | 64.3k | 1.62 |
| | `looped_attn3x4` | 41.9k | 1.06 |
| | `looped_attn6x2` | 41.2k | 1.04 |
| | attention | 39.6k | 1.00 |
| E10 (3) | `hybrid_mingru8_attn4` | 33.6k | 0.85 |
| | `hybrid_mingru_periodic` (9+3) | 33.0k | 0.83 |
| | `hybrid_mingru11_attn1` | 31.8k | 0.80 |
| E9 ctx 2048 (2) | attention | 56.7k | 1.00 |
| | `hybrid_mingru10_attn2` | 52.7k | 0.93 |
| | `mingru` | 37.4k | 0.66 |
| | `hybrid_gdn_periodic` | 8.6k | 0.15 |
| | `gdn` | 6.4k | 0.11 |
| E21 ladder (2), attention, LRs from a 10M probe | `w384` / `w768` / `w1152` | 107.6k / 59.6k / 39.1k | 1.80 / 1.00 / 0.66 |
| suite 26 (no attention arm) | `mla` 72.3k; `hybrid_mingru10_attn2` 50.6k; GDN hybrids 24–26k; `gdn` 21.2k; Mamba hybrids 17.8k; `mamba2` 15.6k | | |

Four readings.

- **On the GH200 in PyTorch, every linear-time mixer trains slower than attention at
  context 512, and more attention layers means faster.** 8+4 is the fastest minGRU
  hybrid and still 0.85x attention: the scan is an unfused log-space cumsum and
  attention is one fused SDPA kernel. At context 2048 minGRU closes (0.66x pure,
  0.93x hybrid) and GDN collapses (0.11x). The pure-PyTorch chunked delta rule is not
  a training mixer; the fused kernel is a new dependency that changes numerics, and I
  have not added it.
- **On the M5 Pro the order flips:** minGRU 144k, attention 66k, Mamba-2 33k tok/s at
  the ~1M-parameter sota shape and T=256 (note 53). That is a mixer-kernel ratio at a
  size where the mixer dominates; at 124M the GEMMs dominate and the ratio shrinks
  (inferred). There is no GDN kernel on Metal at all (`mixers.rs` carries minGRU,
  minGRU+VR blend, Mamba-2 and its conv1d).
- **Depth by reuse costs the full 12-layer step.** `looped_attn3x4` trains at 1.06x
  attention with 48% of its parameters; `attn3` at 2.36x. That is E20 in throughput
  form.
- **Absolute scale.** The 128M Metal trainer runs ~1.6 s per 4096-token step (≈2.6k
  tok/s); the single-tenant GH200 attention arm runs 107k tok/s effective, ~40x
  (inferred from both records). The M5 Pro is a training box at ≤16M (4.3k tok/s at
  B16×T256; 60–72k tok/s at the 1M sota toy) and an inference box above that. In this
  memo every training-speed statement is GH200 unless it says Metal, and every serving
  statement is the M5 Pro.

Where the Metal training step goes, and why the hybrid helps before any kernel work:
70% of the 128M backward is the scalar row flash-attention backward (audit 7b), and
the TensorOps backward is still off by default (DECISIONS M8). A hybrid with four
attention layers of twelve removes two-thirds of that share. With the minGRU backward
at c times an attention layer's, step ≈ 0.30 + 0.70 × (4/12 + 8/12 × c); c ≈ 0.46
from the sota bench ratio gives ≈0.75 of the attention step, **~1.3x faster on Metal**,
bounded above by ~1.9x if the minGRU backward were free (speculative, bounded). On the
GH200 the same arm is 0.85x. Item 5 measures exactly this reversal.

### 5.2 Serving cost per arm (counts verified from nanolab's model classes; ceilings inferred)

| arm | params | non-emb | train FLOPs/tok | int8 bytes streamed/tok | KV bytes/tok (bf16) | recurrent state | KV at 4k ctx |
|---|---|---|---|---|---|---|---|
| attention (12L) | 123.7M | 85.1M | 799M | 123.7 MB | 36,864 | 0 | 151 MB |
| `hybrid_mingru8_attn4` | 147.2M | 108.6M | 902M | 147.2 MB | 12,288 | 49 KB | 50 MB |
| `hybrid_mingru_periodic` (9+3) | 150.2M | 111.5M | 915M | 150.2 MB | 9,216 | 55 KB | 38 MB |
| `hybrid_gdn_periodic` | 123.8M | 85.2M | 757M | 123.8 MB | 9,216 | 1.77 MB | 38 MB |
| `mingru` | 159.0M | 120.3M | 954M | 159.0 MB | 0 | 74 KB | 0 |
| `gdn` | 123.8M | 85.2M | 743M | 123.8 MB | 0 | 2.36 MB | 0 |
| `looped_attn6x2` | 81.2M | 42.5M | 799M | 123.7 MB (81 resident) | 18,432 | 0 | 75 MB |
| `looped_attn3x4` | 59.9M | 21.3M | 799M | 123.7 MB (60 resident) | 9,216 | 0 | 38 MB |
| `attn6` | 81.2M | 42.5M | 515M | 81.2 MB | 18,432 | 0 | 75 MB |
| `attn3` | 59.9M | 21.3M | 374M | 59.9 MB | 9,216 | 0 | 38 MB |

**The minGRU hybrids are not parameter-matched.** nanolab's minGRU uses expansion 2
(`to_z`, `to_h`: d→2d, `out`: 2d→d, 6d² per layer against attention's 4d²), so 8+4
carries 19% more parameters, 28% more non-embedding parameters and 13% more training
FLOPs per token than attention. Every board in the record is token-matched, and §2's
claim is a token-matched claim; under a FLOP-matched or bytes-matched basis the
0.018-nat margin at 50M is **not established** (inferred; item 9 is the fair test).
The GDN arms are parameter-matched and FLOP-cheaper; their cost is kernels. KV bytes
use these boards' 12 heads × 64 with no GQA; the sprint architecture's 2:1 GQA halves
them.

What the columns mean on the two targets:

- **Unified memory, batch 1 (M5 Pro, ~273 GB/s).** The bytes column bounds decode at
  ~2,200 tok/s for the 12-layer attention arm at int8 and ~4,400 at int4; the hybrid
  at ~1,850 / ~3,700. Nothing reaches these. The Gemma roofline puts the native E4B
  path at ~20% of peak and MLX at 60–80%, with ~77% of the token in dispatch and
  barrier overhead across ~780 launches for 42 layers, about 40 µs per launch
  (verified numbers; per-launch cost inferred). At ~19 launches per layer a 12-layer
  model spends ~9–10 ms per token in overhead before a byte moves: **a ~100 tok/s
  dispatch ceiling under a ~2,000 tok/s bandwidth ceiling.** At this size the decode
  lever is launches per token, not bytes: fewer layers (`attn3` has a quarter of 12L's),
  per-layer fusion (the arch02 decoder already has stem and residual megakernels, ~403
  dispatches per training step), ICB replay (tessl decode is 4.5x torch MPS and within
  3–9% of MLX), and M>1 verification so several tokens share each launch (DFlash 1.6x
  and NAX 1.5x on Gemma; Qwen MTP 2.26x). KV and state are irrelevant below ~8k
  context here (151 MB at 4k for the worst arm).
- **Server GPU, batch ≥ 32 (GH200).** Launches amortize, decode is compute- and
  KV-bound, and the columns that matter are FLOPs/token and KV bytes × batch × context.
  Here the hybrid's 3x smaller KV and O(1) state are the win and its 13% extra FLOPs
  the price; GDN needs the fused kernel again.

### 5.3 The cost-basis conflict the record already contains

| basis | winner | evidence |
|---|---|---|
| tokens | attention 12L (4.218); `looped_attn6x2` within 0.014 | E18, verified |
| training seconds (GH200, tenancy 3) | `attn3` (4.411), by 0.24 over 12L | E20, verified |
| launches per decoded token | `attn3` (3 layers) | §5.2, inferred |
| resident weight bytes | `looped_attn3x4`: 4.274 at 60M, against `attn3`'s 4.410 at 60M | E18, verified; column inferred |
| bytes streamed per decoded token | `attn3`, unless the looped stack's layers stay in the system cache across passes | speculative |
| KV bytes at long context | 9+3 / 8+4 hybrids, 3–4x below 12L | §5.2, verified counts |
| ~21M non-embedding parameters | `attn3` (3L×768, 4.410) over `w384` attention (12L×384, 4.543 at its tuned LR) | E18 vs E21: **cross-suite, different LR, narrower tied readout**; inferred |

One anomaly in the E20 elapsed column, flagged by the second review: `attn6` seed 777
finished in 221 s instead of ~672 s. It started after every other job had finished and
ran alone (tenancy 1 instead of 3) to the same pre-registered 35.6M-token budget as the
other four seeds. Tokens were the controlled variable, so the loss rows above stand;
the board should carry tenancy per run, which is the paper's own rule (verified).

Three consequences.

(a) **Looping is a deployment technology, not a training one.** It buys 0.136 nats
over `attn3` at equal parameters (E18), costs no RAM, and its training penalty (E20)
is paid once. Whether it costs decode time depends on the regime: bandwidth-bound
decode re-streams the weights each pass (2x bytes), dispatch-bound decode pays 2x
launches, and both hold at batch 1 on the M5 Pro; at batch 32 on a server the second
pass rides on resident weights and is close to free. E18/E20 measured neither; item
10 does.

(b) **The wall-clock winner and the serving winner coincide.** `attn3` trains 2.4x
faster, decodes with a quarter of the launches, and holds half the RAM and KV; the
price is 0.19 nats at equal tokens. The last row of the table says a shallow-wide
model beats a deep-narrow one at equal non-embedding parameters in this regime, the
opposite of the sub-1B depth folklore, but that comparison crosses suites and learning
rates and the narrow model also has a narrower tied readout. Depth-for-width at fixed
parameters and fixed readout is the one shape axis no board has run (item 11), and it
is the axis a latency budget cares about.

(c) **The token-matched winner is the wrong arm for a batch-1 latency target and the
right arm for two other targets.** 8+4 trains at 0.85x attention on the GH200, at an
estimated 1.3x on Metal, is 19% heavier per decoded token, and holds a third of the KV.
It is the arm for long-context serving and for Apple-side training; at 512 context and
batch 1, `attn3`-class shapes win every cost column.

### 5.4 The deployment-first design

Inferred from 5.1–5.3; each component has a verified anchor and exists in the workspace.

1. **Trunk: minGRU at expansion 1.** Returns the hybrid to parameter parity (4d² per
   layer) and removes the 19% byte penalty; whether the early-token lead survives the
   halved hidden size is untested (item 9). It is the only linear mixer here with a
   forward and backward on both platforms.
2. **Recall: three late attention layers, GQA 2:1, at least one global.** The other
   two may be sliding-window, without sinks, where the window covers the retrieval
   distance the deployment needs. At w=64 the four sink positions cost 0.008 and 0.011
   nats at the two contexts, 5/5 paired (verified today); sinks are a streaming-eviction
   device, not a quality one here. E16 shows recall returns monotonically with window and the
   w=64 arm trains 2x faster at context 2048. Anchor: E10 saturation at three, §2
   crossing table.
3. **Value residual and untied readout.** VR costs one d-vector per token at decode
   (the layer-0 value, already cached if layer 0 is an attention layer) and one d×d
   projection. The untied readout adds 38.6M parameters at d=768, which is why the CE
   boards tie; at deployment the readout is streamed once per token either way, and
   the tied recall cap of 0.555 is the costlier defect.
4. **Attention layers trained block-causal from the start,** so the model emits and
   verifies a block per launch through the lossless self-speculative decoder nanolab
   already checks against greedy. Block width is the decode-side knob trading launches
   for acceptance. Anchor: 68–121 tok/s block decode at 124M on CUDA, ~7x fewer
   positions with self-speculation.
5. **Depth: 6 distinct layers for the latency variant; the same 6 looped twice with a
   learned per-token halt for the RAM variant.** Same shapes, same kernels; the halt is
   the APRDH controller salvaged into the looped transformer (§4.4).
6. **Precision and export: int8 per-row with GPTQ-lite clipping** (the sprint int6
   path reaches 1.34 MB per 16M at no calibrated-BPB loss), bf16 attention math, Core
   ML on the ANE for prefill (0.75 ms per 256-token forward at 16M), tessl ICB for
   decode. The stateful-KV Core ML package is still blocked at StateType lowering; the
   PyTorch decode-step reference passes.

What is new here is not a component but the **matching rule**. Every board in the
record matches arms on tokens or on training seconds. Deployment matches on a serving
budget: decode latency at batch 1 on the target device, or bytes per token at a given
context, after a fixed training spend. Under that rule the questions read "best loss
at 100 tok/s on an M5 Pro" and "best loss at 60 MB resident", and the arms that
qualify are not the arms on the token boards. I know of no published iso-serving-cost
crossing structure with E20's rigor, and the harness already has every field it needs
(tenancy, recipe fingerprint, paired seeds) plus a Metal decode path. It is a paper
and a product at once, and it is the natural second half of §4.3.

### 5.5 What is missing before any of it can be measured (verified gaps)

- nanolab's cached decode exists for the attention mixer only; minGRU, GDN, Mamba-2
  and MLA fall back to O(T²) recompute (`model.py`, `reason.py`). No hybrid can be
  served from nanolab today. The fix is small: the minGRU state is 2d per layer.
- metal-native has no decode path for its own models: prefill via Core ML, training
  via Metal, no cached generation. tessl's decode kernels serve Gemma and Qwen weights.
  Wiring the arch02 decoder through tessl's ICB path is the engineering behind §4.3.
- No GDN kernel on Metal and no fused GDN on CUDA without a new dependency. Until one
  exists GDN is a recall-probe arm, not a deployable one.
- The Metal corpus is missing (§6), so the Metal side of item 5 cannot run yet.
- The Metal 128M trainer's backward is 70% scalar flash attention; the TensorOps
  backward (DECISIONS M8) is the single largest systems lever if 128M Metal training
  continues. The hybrid sidesteps two-thirds of it.

---

## 6. The overhaul: stop, change, start

**Fix first, before any further spend (the comparison contract; each item reproduced
today, §9.1).**

- Decide what the repository's GDN is. Either change `gdn_chunked` and `_sequential`
  to the published rule (correction from the decayed state, eq. 8 of 2412.06464) and
  rerun the GDN cells that matter, or keep the variant and name it; either way the
  class docstring, the KB and learning note 21 must state the rule actually run. Item
  14 measures the difference.
- Stop or re-scope the eight `moe_e4k1`/`moe_e8k1` jobs still queued in `moe32c`
  unless a hash-routed control is what you want: the renormalised top-1 weight is
  identically 1, so the router trains on the balancing loss alone. Switch-style raw
  probability weighting restores the gradient; a changed router is a new experiment
  identity, not a continuation of E19.
- Record tenancy per run in E20-style boards (seed 777 of `attn6` ran alone at the
  full token budget); add a long-stage control to the value-residual ladder (item
  17); state the 10M→50M LR transfer beside every E21 number.

**Stop.**

- Adding recipe-axis entries to the paper's §6. Eleven axes established the thesis; the twelfth
  does not change what a reader believes. The exception is the µP closers already
  queued (E22b, 20 jobs), which are cheap and close a caveat in the headline.
- Optimizer-axis work. D7 is correctly closed at "retired, not reversed".
- APRDH as a unit. Keep the controller idea (4.4), retire the rest.
- E12 at context 512. E16 already answered the window mechanism (recall returns
  monotonically as the window covers the distance, and the 512-window arm reproduces
  attention exactly). Only the context-2048 arm (E15) is still informative.

**Change.**

- **Report every board paired by seed.** It is already the powered test in E10; the
  paper's §4.5 tie and the E10 "not resolved" placement call both change under it.
- **Untie the embeddings on the CE boards, or at least run the 2×2** of tied/untied
  × value-residual on/off. Every quality result was measured through a readout that
  structurally caps recall. The hypothesis that value residual's win is partly a
  workaround for tying (it gives every layer a non-embedding token representation)
  is testable in 20 jobs.
- **Ladder tokens, not width.** E21 showed width does not move the ranking. The
  regime is 0.4 tokens/param. Run the attention/minGRU/8+4 triple at 5 and 20
  tokens/param on the `w384` shape (40.6M parameters for attention, 49.4M for minGRU;
  not 60M, which is `attn3`), where it is affordable (§7).
- **Adopt a capability panel** beside CE: MQAR at three difficulties, a repeated-span
  copy loss logged every eval, and one long-range synthetic. The CE/recall
  disagreement (`hybrid_gdn_periodic` loses on CE with disjoint intervals and ties or
  leads on recall) shows CE alone under-rates the thing hybrids exist for.
- **Restore or rebuild the Metal corpus** (`fineweb10B_sp1024`). Until then the Metal
  track cannot produce a comparable number, and 4.3 depends on it.
- **Carry three cost columns on every board:** tokens; training seconds with the
  hardware and tenancy named; and a serving column (launches per token, resident MB,
  KV bytes per token). Print the parameter count beside every hybrid. Today 8+4 is
  19% heavier than attention and no board says so (§5.2).

**Start.** The experiments in §7, in that order.

---

## 7. Experiments, priced

Rates from the record: attention 33.5k tok/s and minGRU hybrids ~35k tok/s at
tenancy 3 on the GH200 (E20); `w384` attention 1.8x `w768` at tenancy 2 (E21,
verified); GH200 $2.29/h. Hours are estimates unless a suite of the same shape has run.

| # | experiment | jobs | GH200-h (est.) | question | pre-registered readout |
|---|---|---|---|---|---|
| 1 | attention arm inside `ratioplace32` | 5 | ~1.5 | make §2 within-suite | 8+4 beats attention at 50M paired 5/5 → claim stands; else §2 is cross-suite only |
| 2 | copy-loss probe on the attention and minGRU 50M curves, 3 recipes | 30 | ~10 | is the crossing the induction bump? | copy-loss drop within ±1M of the crossing on ≥4/5 seeds per recipe → predictive; else mechanism withdrawn |
| 3 | 2×2 tied/untied × value residual, attention arm, 50M | 20 | ~7 | does tying carry part of the VR win? | VR gain shrinks by >50% untied → interaction; unchanged → independent |
| 4 | 8+4 hybrid on the recall grid (p=4,8 × 3k,9k, 15 seeds) | 60 | ~3 | does the no-regret arm also recall? | rate within Wilson interval of attention at every cell → no-regret on both metrics |
| 5 | wall-clock board with 8+4, on GH200 and on the M5 Pro | 10 + Metal | ~4 + local | does the cost basis reverse by hardware? | hybrid wins on Metal and loses on GH200 → hardware-lottery result, publishable |
| 6 | token ladder: attention, minGRU, 8+4 at `w384` (40.6M attention), 200M and 800M tokens = 5 and 20 tokens/param, 5 seeds | 30 | ~40–60 | do the orderings survive leaving the data-starved regime? | crossing token scales with budget → regime artifact; ordering at 20 tok/param is the durable one |
| 7 | fast-weight recall screen: GDN, GDN + previous-token shift, nonlinear fast-weight (TTT-style), on MQAR p=4..8, 15 seeds | ~90 | ~5 | which state update forms heads fastest? | any arm matching attention's 15/15 at 3k earns a 50M CE seat |
| 8 | looped 6x2 + learned halt, E20 protocol, with `attn6` control | 15 | ~5 | does a halt recover the wall-clock loss? | halted arm within 0.02 of `attn6` at equal clock and ahead of it at equal tokens → depth is bought back |
| 9 | parameter-matched hybrid: minGRU expansion 1, 8+4 and 9+3 vs attention, 50M | 10 | ~3.5 | does no-regret survive parameter parity? | paired 5/5 at 50M and no re-crossing → §2 stands at parity; else the margin was parameters |
| 10 | decode benchmark on the M5 Pro: attention, `attn3`, `looped_attn3x4`, 8+4; int8; batch 1 and 8; ctx 512 and 4k; tok/s, launches/token, resident MB | 0 GPU; ~2 days engineering (minGRU state carry; arch02 decoder on tessl ICB) | local | is 124M decode dispatch-bound, and is looping free? | launches/token predicts tok/s within 20% → dispatch-bound; block decode ≥1.5x; looped 3x4 within 1.2x of `attn3` at batch 8 → free at batch |
| 11 | depth-for-width at fixed ~21M non-embedding and a fixed 768-wide untied readout: 3L×768, 6L×544, 12L×384; 50M | 10 | ~2.5 | does width buy back depth's loss at equal launches? | 6L within 0.05 of 12L → 6L is the latency shape; 3L within 0.05 → 3L |
| 12 | iso-serving-cost board: fix 100 tok/s at batch 1 int8 on the M5 Pro and 64 MB resident; every arm that qualifies trains at 50M and at equal wall clock; paired | ~20 | ~8 + item 10 | which arm wins at a serving budget? | pre-registered: the token-board winner is not the serving winner |
| 13 | Metal training-speed board, 8+4 vs attention at 16M (needs the corpus) | local | ~1 day | does 0.85x on the GH200 become >1x on Metal? | measured step ratio against the 1.3x estimate in §5.1 |
| 14 | GDN rule ablation: implemented vs published update, same block, on the E8 grid (p=4,8 × 3k,9k, 15 seeds) and the E16 seq-255 cell; a 50M CE seat only if recall moves | ~150 recall + 5 | ~8 + ~2 | is the hard-recall failure the operator variant? | published rule solves ≥4/15 more seeds at p=8 or any seq-255 seed → the variant was the defect; else the failure is capacity or optimisation and §4.1 proceeds on the published rule |
| 15 | MoE with a task-trained router (raw top-1 probability, Switch style), fresh experiment identity, 4 and 8 experts, `e1k1` control, 5 seeds | 15 | ~6 | does learned routing buy anything at 0.4 tokens/param? | ≤ +0.016 over dense → parameters still not the bottleneck; else E19's conclusion was a router artifact |
| 16 | damage-aware write policy on the recall diagnostic (§9.3): delta memory + protected-set interference score + budgeted explicit store, against HAM-style surprise routing and FIFO at equal state bytes | ~60 | ~4 | does measured interference predict and prevent forgetting better than prediction error? | the score predicts lost facts better than prediction error **and** intervening on it recovers them → mechanism; either fails → withdrawn |
| 17 | value-residual ladder with a long-stage control: control, VR, gate, gate+VR; 2 seeds; sprint proxy | 8 | local (3070 Ti) | is VR's win real against the control at the long stage? | VR beats control by >0.05 BPB → §3 row stands; else the VR win is a gating artifact |

Items 14 and 15 come first: they decide what every GDN and MoE number in the record
means, and 15 is cheaper than letting the queued `moe32c` arms finish. Items 1–4 are
under $60 total and settle whether §2 is a paper. Item 6 is the one
that answers the reviewer objection E21 could not. Items 9–12 are the deployment
program: 9 and 11 are cheap GPU boards, 10 is the engineering that unblocks 12, and
12 is the result nobody else can publish from this stack.

---

## 8. Evidence grades for the claims in this memo

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
| 8+4 is 19% heavier than attention; the no-regret margin at parameter parity | **verified** counts; margin at parity **unverified** (item 9) |
| linear mixers train slower than attention on the GH200 at ctx 512; GDN at 0.11x at ctx 2048 | **verified** (within-suite medians, §5.1) |
| 124M decode on the M5 Pro is dispatch-bound by ~20x | **inferred** from the Gemma roofline; unmeasured for these arms (item 10) |
| the hybrid trains ~1.3x faster than attention on Metal | **speculative**, bounded estimate from audit 7b and note 53 (item 13) |
| shallow-wide beats deep-narrow at equal non-embedding parameters | **inferred**, cross-suite and cross-LR (item 11) |
| the repository's GDN is a different operator from the published gated delta rule | **verified** (probe on `gdn_chunked` and `_sequential`; paper eq. 8 fetched) |
| the top-1 MoE router receives no task gradient | **verified** (probe on `MoE`; task-gradient norm 3e-18 against 0.12 from the balancing loss) |
| the GDN variant caused the hard-recall failures | **speculative** (item 14) |
| four sinks cost CE at w=64 | **verified**, paired 5/5 at both contexts |
| E20 `attn6` seed 777 ran alone at the full token budget | **verified** (finish order and budget) |
| value residual beats the no-VR control at the long stage | **unverified** (no long-stage control; item 17) |

---

## 9. The second review (Codex / "GPT-6 Astra", 2026-09-04), reconciled

Received after §0–§8 were written. It reviewed the same workspace plus BINN, ran a
read-only recomputation over 1,252 metrics files, and checked prior art live. I did
not review BINN and take its BINN statements as report-backed only. Below: what I
reproduced, where the two reviews agree, where they differ, and what changes.

### 9.1 Claims of theirs I reproduced today, on the real code and records

| claim | check | result |
|---|---|---|
| the repo's GDN applies the delta correction to the **undecayed** state | ran `gdn_chunked` on two unit tokens at α=β=0.5; read `_sequential` (`mixers.py:813–815`); fetched eq. 8 of arXiv 2412.06464 | kernel [0.5, 0.5]; published rule [0.5, 0.625]. At α=0.3, β=0.9 the kernel returns 0.36: the transition of a stored association along its own key is α−β = −0.60, against the paper's α(1−β) = +0.03. Keys are L2-normalised (`mixers.py:784`), so the scalar analysis is exact. The regression test compares the kernel with a reference that implements the same variant and cannot see this; the class docstring, the KB and learning note 21 all state the variant. **Verified.** |
| the top-1 MoE router receives no task gradient | ran the `MoE` module, 4 experts, top-1 vs top-2, float64 | task-loss gradient norm on the gate 2.8e-18 (top-1) vs 3.0e-2 (top-2); balancing-loss gradient 0.12 in both. Cause: `weights / weights.sum()` is identically 1 at k=1 (`model.py:93`). Switch multiplies by the raw probability precisely to keep this gradient. **Verified.** The two `moe_e4k1` jobs marked running and the six `moe_e4k1`/`moe_e8k1` jobs pending in `moe32c` (queue synced 16:53 today) train a router that can only learn to balance. |
| E20 `attn6` fails the 5% time tolerance | finish order from file times; token counts | seed 777 ran 221 s instead of ~672 s, but reached the same pre-registered 35.6M-token budget: it started after every other job had finished and ran at tenancy 1. Tokens were the controlled variable, so the loss board stands; the elapsed column does not, and tenancy per run is the missing field. Their reading overstates the damage. |
| four sink positions worsen CE at w=64 | paired by seed, both contexts | +0.0083 [0.0063, 0.0102] at ctx 512; +0.0112 [0.0100, 0.0125] at ctx 2048; 5/5. **Verified.** |
| the arch ladder has no long-stage control | read note 04 | correct. The 2.223 I quoted in §3 was the mid-stage control, a cross-stage comparison; at the short stage the control beat `value_resid`. Fixed in §3. |
| E21's LRs were chosen at 10M tokens and spent at 50M | read the ladder board | correct (its own §"Phase 1"); now noted beside every E21 number. |
| the looped forward keeps the first pass's v0 as a fixed anchor | read `model.py:371–380` | correct, by design comment. |
| APRDH's halt computes the full update and then masks it | read `train_toy_adaptive.py:1431` | correct: `h = mask·updated + (1−mask)·h`. No executed-compute saving. |
| GDN-2 and Hybrid Associative Memories exist with the described claims | fetched both arXiv abstracts | GDN-2 (2605.22791, May 2026): channel-wise erase and write gates, chunkwise WY. HAM (2603.22325, March 2026): attention stores what the RNN cannot predict, one continuous threshold; the abstract does not mention learned routers. |

### 9.2 Where the two reviews agree

- No attention-scale architecture is in or inferable from the record; the regime
  (0.4 tokens per parameter, one corpus, one shape) cannot certify one.
- The dividing line is the recall mechanism, and the next program should explain which
  memory operation fails under which demand rather than rank named mixers.
- Depth by reuse is a parameter-efficiency tool with a small measured token-matched
  cost and a large wall-clock cost; APRDH should be reduced to its halt controller.
- Prior art occupies value residual, looped depth, MoD-style routing, delta-rule
  memories, surprise caching and test-time memory; a contribution must be narrower.
- The harness is the most transferable artifact in the workspace.

### 9.3 Where they differ, and my position

- **Their primary proposal is damage-aware memory writes; mine was the
  head-formation-speed search (§4.1).** Same axis; theirs is the more concrete
  instantiation and I adopt it as item 16, with their own caveats: HAM and GDN-2 are
  the nearest neighbours, and a covariance-preconditioned or recursive-least-squares
  update is a baseline, not a novelty. Their hardest-issue note is right: a policy
  that depends on the evolving state breaks the WY chunk structure, so the parallel
  form must be specified up front or the result is a mechanism paper.
- **They put execution last (their stage 6); §5 makes it a co-equal axis with its own
  matching rule.** I keep my order. The iso-serving-cost board changes which operators
  are worth diagnosing, and their own stop condition ("serial state dependence or
  dispatch erases the advantage") is what §5.2 measures first.
- **They did not look at the curves.** Their tables are final-CE only, so the
  no-regret result (§2) and the crossing structure are absent from their reading.
  Their final-CE numbers for the same arms match mine to the fourth decimal.
- **On the sign test they are right, and it changes nothing.** Five of five is a sign
  statement (p = 0.0625 two-sided); the paired t-interval is the test, and at the
  99.9% level (§2) the result still excludes zero at every marker from 4.1M.
- **They did the comparison-contract audit I skipped.** I took the mixer
  implementations as faithful to their papers; two were not. That is the largest gap
  in my review and the reason a "fix first" block now heads §6.
- **BINN and the temporal-synchrony branch:** outside what I reviewed; no position.

### 9.4 What this changes in the ranking

§4.1 stays the highest-ceiling line, now with a mandatory first step (item 14) and a
concrete mechanism (item 16). §4.2 is unchanged. §4.3 and §5 are unchanged. §4.4 is
unchanged except that "APRDH's halt saves compute" is now verified false as
implemented. The §1 statement that parameters are not the bottleneck rests on E19's
corrected `e1k1` control and on `e4k1`/`e8k1` arms whose router could not learn to
route; it holds for a hash-routed MoE until item 15 runs. Every GDN number in this
memo, including the recall rates that motivate §4.1, is a number about the
repository's variant until item 14 runs.
