# The Crossing Scales: The Token Budget at Which Attention Overtakes a Recurrent Mixer Is a Function of Width, Context and Tuning Protocol

**Bharath Chandra Vaddaram** <bharath.vbcr@gmail.com> · independent researcher

`nanolab` · Draft 2026-09-07

**Code and artifacts:** <https://github.com/bharathvbcr/MachineLearning>
**License:** text CC BY 4.0; code and data artifacts MIT.
**Competing interests:** none. **Funding:** none; compute self-funded on one rented
NVIDIA GH200 instance.

---

> ## ⚠ STATUS 2026-09-09 — SUPERSEDED IN PART. DO NOT CIRCULATE THIS DRAFT.
>
> **§4.4 called horizon-invariance of the learning-rate argmin "the load-bearing
> assumption of both §3 and §4". It has since been measured, and it is false.**
>
> Boards G7 (`crossover50m_ratioplace32`, 2026-09-08) and G3 (`crossover_ladder50m`,
> 2026-09-09) find that at a 50M-token budget **every arm at every width tested prefers a
> lower learning rate than this draft gave it** — four architectures, three widths, no
> exception, every grid running off its bottom edge:
>
> The 50M argmin has since been **located** (boards G8a/G8b/G8d 2026-09-09, d1152
> re-probed at 0.667× by G11b 2026-09-10), every cell interior, both mixers agreeing at
> every width, and every argmin read **paired by seed**:
>
> | width | attention | minGRU | best `lr` | width × lr |
> |---|---|---|---|---|
> | 384 | **2×** | **2×** | 0.0012 | 0.4608 |
> | 768 | **1×** | **1×** | 0.0006 | 0.4608 |
> | 1152 | **0.667×** | **0.667×** † | 0.0004 | **0.4608** |
> | 1536 | **0.5×** | **0.5×** | 0.0003 | 0.4608 |
>
>
> **Every argmin now carries n=3, and none of them moved (G11b, 2026-09-10).** That was
> the pre-registered test: *any argmin that shifts once its own point carries three seeds
> was never located*. Each width's argmin and its tight-side neighbour were run at seeds
> 1337/42/100, and every cell returned the same multiplier it had at n=1. Margins are
> paired on the three seeds each pair shares, so seed-to-seed spread — which on these
> boards reaches 0.0574, eighteen times the rerun floor — is differenced out rather than
> averaged in:
>
> | width | mixer | argmin | runner-up | paired margin (n=3) | verdict |
> |---|---|---|---|---|---|
> | 384 | attention | 2× | 1× | +0.0144, 3/3 | resolved |
> | 384 | minGRU | 2× | 1× | +0.0155, 3/3 | resolved |
> | 768 | attention | 1× | 0.5× | +0.0391, 3/3 | resolved |
> | 768 | minGRU | 1× | 0.5× | +0.0529, 3/3 | resolved |
> | 1152 | attention | 0.667× | 1× | +0.0053, 3/3 | resolved |
> | 1152 | minGRU | 0.667× | 1× | +0.0024, 3/3 | **flat — under the 0.0031 floor** |
> | 1536 | attention | 0.5× | 1× | +0.0125, 3/3 | resolved |
> | 1536 | minGRU | 0.5× | 1× | +0.0067, 3/3 | resolved |
>
> **Seven of the eight cells are resolved; `width × lr = 0.4608` holds at all eight.** The
> exception is d1152 minGRU, where 0.667× leads 1× by 0.0024 — the right sign on 3 of 3
> seeds, but under the floor, so it is reported as consistent with the law rather than
> evidence for it.
>
> † At d1152, attention's 0.667× beats **both** neighbours above the 0.0031 rerun floor —
> 0.0100 over 0.5× (3/3 seeds) and 0.0053 over 1× (3/3) — so that argmin is located.
> minGRU's 0.667× beats 0.5× by 0.0167 (3/3) but leads 1× by only +0.0024, *inside* the
> floor. Its row is consistent with 0.667× and is not evidence for it.
>
> The mechanism is visible in the curves: at w1152 attention's lr40−lr80 gap is only
> −0.0417 at 4.11M tokens and widens monotonically to −0.1254 by 50M. The high LR really
> is better early — which is why the 610-step probe picked it — and loses over the
> remaining 2400 steps. **The argmin is a function of horizon**, so this is a sharper
> instance of the paper's own thesis rather than a calibration error.
>
> **A second result, not yet written into any section.** **All four widths** put the
> optimum at exactly `lr ∝ 1/width`. d1152 was the lone exception while the grid was
> 2×-spaced — 0.5× and 1× differ by 0.0064 there, straddling the 0.667× prediction — and
> G11b settled it by running the predicted point itself: the argmin was never at 1×, it is
> at 0.667×, and `width × lr = 0.4608` holds at every rung. At 10M the same probe found the
> argmin *width-invariant*. So **LR transfer across width fails at a short horizon and
> holds at a long one** — and these runs have `mup: False`, so hand-tuning each width
> recovered µP's own 1/width prescription with the µP divisor switched off. Still labelled
> **inferred**: d384/d768/d1536 were located on 2×-spaced grids and only d1152 has been
> probed at the law's own point, so the constant is measured at one width and consistent
> at three.
>
> **Do not read as current:** every phrase of the form "each arm at its own measured
> learning-rate argmin" (abstract, §3, §4.1, the §4.2 table row labelled *per-arm
> argmin*, §8.1); "attention's argmin is 8× base and minGRU's 4× **at every width**" is a
> **10M** result and does not transfer to 50M. The crossing values themselves
> (10.70M → 13.55M) are not withdrawn — they are a valid comparison **across widths at a
> fixed LR protocol** — but the protocol is no longer describable as per-arm-optimal, and
> the −0.81M per-arm-LR shift in §4.2 must be restated as a shift between two
> *10M-derived* protocols.
>
> **RESOLVED 2026-09-09 — the ladder has been re-run at the 50M argmins (G9/G9b), and
> the headline changes.** All four rungs, n=5, each arm at the multiplier measured on
> its own 50M grid (d384 2×, d768 1×, d1152 1×, d1536 0.5×). §3.2's table below is
> retained as the 10M-derived protocol and is joined by the re-tuned one. **The
> monotonicity claim does not survive at the top rung.** The crossing rises with width
> through d1152 and then stops: d1152 → d1536 is **−0.02M [−0.33, +0.30]**, 2 of 5
> seeds up, where the two steps below it are +1.67M and +1.08M, both 5/5 and both
> disjoint from zero. Anywhere §3.2, §9 or the abstract says "monotone across four
> widths, every consecutive 95% interval disjoint", that is now true of **three** rungs
> and false of the fourth.
>
> **Unaffected:** §5's placement contrasts (hybrid-vs-hybrid at one shared LR,
> common-mode); §6's recall results (E36); §7's wall-clock work. §5's
> hybrid-vs-attention rows survive re-measurement at the argmin — E35's inferred
> confound dissolves, because all three arms prefer the *same* multiplier — but acquire a
> new caveat: the hybrid lead exists only at 1×, and at 4× and 8× attention wins
> (+0.0272/+0.0289 and +0.0226/+0.0281, 95% intervals excluding zero).
>
> **In flight:** `crossover_probe50m` (G8a) is locating the actual 50M argmin on a grid
> extended down to 0.25×, because *an argmin at the edge is not an argmin*. Numbers here
> are restated only after it lands. See `docs/GAP_PLAN_2026-09-07.md` §2.2b.

## Abstract

Short-horizon screens decide which architectures get scaled. Their validity rests on the
assumption that the ordering they produce is a property of the architectures. A companion
paper showed that this ordering moves. Here we measure what it moves *with*, and find
that the crossing itself — the token budget at which softmax attention overtakes a
recurrent mixer — is a well-behaved quantity that can be estimated to within a few
percent and that scales.

On a 12-layer stack trained to 50M tokens with each arm at the learning-rate
optimum measured *at that budget*, attention overtakes minGRU at **10.71M** tokens
[10.34, 11.08] at `d_model` 384, **12.37M** at 768 and **13.46M** [13.17, 13.74] at 1152 —
rising, 5 of 5 seeds per step, consecutive 95% intervals disjoint. Then it stops: at
`d_model` 1536 the crossing is **13.44M** [13.27, 13.61], a paired step of −0.02M
[−0.33, +0.30] on 2 of 5 seeds. The crossing scales with width over a 3x range and
flattens at the top of the one we can afford; whether that is saturation or a peculiarity
of the last rung, four points cannot say. The *endpoint* ranking over the same four widths is flat: attention wins by
0.143–0.159 nats at every width. These are not in tension. They are answers to two
different questions, and a screen reports only the first.

Two further recipe axes move the crossing by comparable amounts. Giving each arm its own
learning rate instead of a shared one moves it **−0.81M** tokens [−1.07, −0.55]
(t = −8.6, 5/5 seeds); we measure the mechanism directly, as attention's optimum sits at
8x the base rate and minGRU's at 4x, at every width. [SUPERSEDED 2026-09-09: that is the
optimum at the 10M probe horizon; at 50M both arms want less — see the status block.] Raising context from 512 to 2048
while trading batch to hold tokens-per-step fixed moves it **+3.63M** [+2.93, +4.34]
(t = +14.3, 5/5). At a 12M-token budget, width alone decides the winner.

One axis does **not** move it, and that is the useful half. Quadrupling the training
budget from 50M to 200M tokens — a four-times-longer cosine, placing the crossing 21% of
the way through one schedule and 5.3% through the other — leaves it at **10.61M**
[10.21, 11.01] against 10.70M [10.24, 11.16]. The crossing is set by tokens consumed, not
by schedule position or by how long one intended to train. A screen therefore does not
have to run the full budget to locate it.

In hybrid stacks, *where* attention sits dominates *how much* of it there is. Moving two
attention layers from the end of the stack to its two ends costs **0.0283** nats
[+0.0259, +0.0307] at identical count — nine times the 0.0031-nat rerun floor — and a
second attention layer placed at position 0 is worse than omitting it entirely. Three
interleaved attention layers beat four stacked ones. These are hybrid-against-hybrid
contrasts at one shared learning rate, so the tuning is common-mode.

The hybrid-against-attention comparison is not, and we say so: the best hybrid beats dense
attention by 0.0165 nats, and by 0.0053 [−0.0087, −0.0019] at parameter parity, but on
boards run at a learning rate that is nobody's optimum — and a hybrid's optimum turns out
to follow its *majority* mixer's, at 4x base rather than attention's 8x (−0.1241,
t = −54.5). Re-measured where both arms sit nearer their own optima, the hybrid's margin
**widens** to 0.0777 [−0.0891, −0.0663] rather than vanishing. That is a mitigation at one
width and budget, not a repair of the boards.

That last result is where held-out loss stops being informative. On multi-query
associative recall at sequence 255, attention solves 11 of 15 seeds while minGRU and
gated DeltaNet solve **0 of 15 each** — 80 recurrent runs at sequence 255 and above
without a single solve, from architectures that trail attention by only 0.23 nats of
aggregate cross-entropy. A metric that separates these arms within hundredths of a nat is
silent about a capability difference that is total.

The hybrids do not inherit that floor. Measured at the same cell, an 8-recurrent /
4-attention stack solves **15 of 15**, and the two other layouts 14 and 13, against pure
attention's 11 — at least as good as attention on every arm, with no seed on which
attention solved and the 8+4 hybrid did not. We stop there rather than claim they beat it:
paired on the same seeds the sign test does not separate them (p = 0.125). Against the
pure recurrent arms it does, decisively — 0 solves against attention's 11, p = 0.001.

So the capability tracks the presence of global attention layers, not the bulk
composition of the stack: four of twelve layers restore in full what zero leave absent,
and eight recurrent layers cost nothing on this axis. Cross-entropy ordered these three
configurations within hundredths of a nat, and ordered them wrongly with respect to a
capability that is all-or-nothing.

We also report five measurement defects found while preparing this paper — a token
counter that restarted on resume, 135 double-counted records, a summary column that
mislabelled 165 runs, a learning-rate argmin read off unequal seed sets, and a data
loader that silently changes which tokens a run trains on when the GPU is crowded —
two of which were caught only by cross-checking a derived number
against an independently published one, and one of which had already put a wrong claim
into a draft of this paper.

Every result here sits between 0.03 and 1.2 tokens per parameter. Nothing in it speaks to
the compute-optimal regime, and by this paper's own argument it must not be read as
though it did.

---

## 1. Introduction

Short-horizon screens are the default currency of architecture selection. A run at one percent of
the target budget is cheap, and its ordering is spent as though it were the full run's. The
assumption doing the work is that the ordering is a property of the architectures.

Our earlier paper showed that it moves: batch size, learning-rate horizon and cost basis each
reordered otherwise identical boards, and a crossing we had published ourselves at 6.6–7.4M tokens
from one seed did not survive a five-seed replication. It catalogued the movement without saying
what the movement is a *function of*. This paper makes that quantitative.

Softmax attention and minGRU [2410.01201], a minimal parallelizable gated RNN, cross under a fixed
recipe: the recurrent arm leads early, attention overtakes later. That crossing token is what a
short screen implicitly reports — stop before it and one arm wins, stop after it and the other does.
We measure it at four widths, five seeds per rung, each arm at its own measured learning-rate
argmin — re-measured at the 50M budget (G9/G9b), which is the horizon these runs are actually
spent over: **10.71M** tokens at width 384, **12.37M** at 768, **13.46M** at 1152, **13.44M** at
1536. The rise is monotone and the intervals disjoint through width 1152; between 1152 and 1536 the
crossing stops moving (−0.02M [−0.33, +0.30], 2 of 5 seeds). At 12M tokens width 384 has crossed and
attention wins; width 1536 has not and minGRU wins. Same architectures, same recipe, same budget, opposite
answer, decided by width alone.

One axis does not move it, and that is the result a practitioner can spend. Quadrupling the schedule
to 200M tokens leaves the crossing at **10.61M** against 10.70M at 50M, 5 of 5 seeds both times. It
is fixed by tokens consumed, not by how long you intended to train. **A screen need not run the full
budget to locate the crossing.**

**Contributions.**

1. **The crossing scales with width; the endpoint ranking does not.** It moves +27% across 4× width
   and 10.3× parameters, 5 of 5 seeds crossing at every rung. The 50M endpoint gap over the same
   span is flat — minGRU − attention is +0.1471, +0.1585, +0.1517 and +0.1426 nats, minGRU behind on
   every seed. A board reporting only the endpoint sees a width-stable ranking and stops there.
   Width moves the crossing, not the ranking at 50M.

2. **The crossing is also a function of the tuning protocol, and the mechanism is measured.** Same
   shape, seeds and schedule; only the LR protocol changes. Shared LR puts the crossing at 12.33M,
   per-arm argmins at 11.53M — a paired shift of **−0.81M** [−1.07, −0.55], t = −8.6, 5 of 5 seeds.
   The cause is an asymmetry: attention's argmin is 8× the base LR at every width, minGRU's 4×, so a
   shared-LR board reads the arms at different distances from their own optima. Every 50M board in
   our repository except the width ladder does that.

3. **A second recipe axis moves it further than width does.** Raising context 512 → 2048 while
   trading batch 32 → 8 to hold tokens-per-step fixed moves the crossing **+3.63M** [+2.93, +4.34],
   t = +14.3, 5 of 5 seeds — **+29%**, against the +27% the whole width ladder buys across 10.3× the
   parameters. Context and batch move together and we do not separate them: the joint effect is
   verified, attribution to context alone is not measured.

4. **In hybrid stacks, placement dominates ratio.** At a fixed count of two attention layers,
   splitting them to the two ends rather than stacking them at the end costs **+0.0283** nats
   [+0.0259, +0.0307], t = +32.4, 0 of 5 seeds better — 9× this pipeline's 0.0031-nat determinism
   floor, from moving one layer. Three interleaved attention layers beat four stacked ones (−0.0077,
   5/5), and a second attention layer at position 0 leaves the model worse than omitting it (+0.0067
   [+0.0028, +0.0106], t = +4.8). Count helps monotonically only once the layout is fixed.

5. **Held-out loss and associative recall dissociate, and the cell that would close the argument has
   not been run.** At 50M the 8+4 hybrid beats pure attention by 0.0165 nats of held-out CE — on a
   board run at 1x base LR, which is neither arm's optimum, so §5 prices that confound — and pure
   minGRU trails it by 0.23 nats. On multi-query associative recall at sequence ≥ 255 the pure
   recurrent arms return **zero solves in 80 runs**; at sequence 255 attention solves 11 of 15 and
   both recurrent arms 0 of 15. A metric that separates these architectures within hundredths of a
   nat is silent about a capability difference that is total. The hybrids do not inherit that
   floor: at the same cell an 8-recurrent / 4-attention stack solves **15 of 15**, the other two
   layouts 14 and 13, against attention's 11 — at least as good on every arm, with no seed on
   which attention solved and the 8+4 hybrid did not. Paired, the sign test does not separate
   them from attention (p = 0.125); against the pure recurrent arms it does (p = 0.001). The
   capability tracks the presence of global attention layers, not the bulk of the stack.

6. **The token budget does not move the crossing, and that negative is the useful result.** Only the
   budget and its cosine horizon change. At 50M the crossing is 10.70M [10.24, 11.16]; at 200M it is
   **10.61M** [10.21, 11.01], 5 of 5 seeds each time, intervals almost coincident — and it sits 21%
   of the way through the shorter schedule against 5.3% through the longer. Width moves the
   crossing, the LR protocol moves it, context-with-batch moves it; the budget does not. It is set
   by tokens consumed, not by schedule position or intended run length. This does not contradict the
   earlier paper, where *truncating* the cosine to 20M moved a late crossing 12.34M → 14.58M while
   restoring the 50M cosine and stopping at 20M recovered 12.34M. Shortening a schedule through the
   crossing region moves it; lengthening it beyond does not, and that asymmetry is itself the
   finding (**inferred**).

**Scope, stated here rather than buried.** The width ladder is deeply under-trained: 50M tokens on
40.6M–558.6M parameters is 0.09–1.23 tokens per parameter, and the crossings sit at **0.032–0.264**,
falling across the ladder. E35 is the one rung outside that regime at **4.93** tokens per parameter,
still roughly 4× below compute-optimal (~20). Nothing here speaks to the compute-optimal regime, and
this paper's own thesis forbids extrapolating there. We report the exponent fitting the four ladder
points — crossing ∝ width^0.168, R² = 0.959 — and decline to call it a law; four points do not
determine a functional form.

---

## 2. Related work

**Comparisons that did not survive controlled re-evaluation.** Melis et al. re-evaluated LSTM
successors under large-scale black-box tuning and found properly regularized standard LSTMs beating
the newer architectures reported to beat them [1707.05589]. Musgrave et al. audited four years of
deep metric learning and found the accumulated gains marginal once methodology was controlled
[2003.08505]. Henderson et al. showed non-determinism and variance make published reinforcement-
learning comparisons hard to read without seed discipline [1709.06560]. Dodge et al. made the
dependence explicit: several published comparisons invert when the hyperparameter-search budget
changes [1909.03004]. Hooker's hardware lottery names the systems version: an idea can win because
it suits the available hardware [2009.06489]. That literature makes the measurement budget an axis
of the reported result. We do the same for the training horizon, and add which recipe factors move
the point where the answer changes and which leave it alone.

**Recall as the axis where recurrent models fail.** Zoology pretrains 17 attention and gated-
convolution models and decomposes perplexity into associative-recall and non-recall tokens,
attributing much of the remaining gap to the former [2312.04927]. Based maps the recall–throughput
tradeoff [2402.18668]. Just read twice reports that recurrent LMs cannot recall from bounded memory
[2407.05483]. Ren et al. explore Mamba's limitations on copying and chain-of-thought reasoning
[2410.03810], and Akyürek et al. add in-context language learning as a further probe [2401.12973].

Our claim is compatible with Zoology's and is not the same claim. Zoology *decomposes* the loss and
shows where the gap lives; split the tokens and it is visible. We report the **undecomposed**
number, the aggregate held-out loss a screen actually selects on, and it does not surface the
failure at all: at 50M a hybrid wins on aggregate CE while the pure recurrent architecture it is
built from returns zero solves at sequence ≥ 255. Zoology did not miss this — it measured the
decomposition that makes the gap visible. Our point is what survives when nobody decomposes, the
condition every short screen runs under.

**Hybrid attention/recurrent stacks.** Waleffe et al. train Mamba-based language models at 8B
parameters: a hybrid exceeds a matched Transformer where pure SSMs lag on copying and in-context
learning — hybrids need some attention [2406.07887]. Gated Linear Attention [2312.06635] and xLSTM
[2405.04517] are recurrent-side architectures such stacks build on. The closest comparator is
Cerruti et al., comparing DeltaNet, Gated DeltaNet and KDA at 350M parameters and 15B tokens,
finding hybrid stacks improve loss at a throughput cost and Muon beating AdamW [2607.07953]. It is
also the scale this paper does not reach: roughly 43 tokens per parameter against our 0.09–4.93,
between 9× and 480× more.

**Horizon, batch and learning rate as ranking confounds.** Optimal learning rate depends on the
token horizon, longer runs requiring smaller rates, with the optimum following its own scaling law
[2409.19913]. On the batch axis: Shallue et al. attributed much of the literature's disagreement
about batch-size effects to metaparameter tuning across 35 workloads [1811.03600]; McCandlish et al.
introduced the gradient noise scale as a predictor of the largest useful batch size, rising within a
run as loss falls [1812.06162]; Zhang et al. found critical batch size scales with data size, not
model size [2410.21676]; Marek et al. proposed fixing Adam's second-moment half-life in tokens
rather than its decay rate at small batch [2507.07101]. Compute-optimal scaling established that
conclusions drawn at fixed data mislead [2001.08361, 2203.15556], and Choshen et al. found seed
variability in scaling-law estimation large enough that several small models can beat one large
[2410.11840]. These pin the learning rate as a function of horizon and the batch size as a function
of loss; none pins the crossing token as a function of width.

**The gap.** A literature search across arXiv, OpenAlex, Semantic Scholar, Crossref and DBLP, run
2026-09-07, returned no work measuring the crossing token itself as a function of model width, and
no RAG evidence retrieval surfaced a source making that claim. This is **inferred, and weak**:
absence of retrieval is not proof of absence. The claim is that the search found no prior
measurement, not that none exists.

---

## 3. What moves the crossing, and what does not

The width ladder was built to test whether the 50M-token ranking survives a change of scale. It does: attention wins at every width and the margin does not trend. That is the result in `docs/LADDER_BOARD_2026-09-04.md`, which compares only the 50M endpoints at three widths and never locates the crossing; this section supersedes it, keeps its endpoint finding, and adds the quantity it did not measure.

### 3.1 How the crossing is measured

Each rung is a matched attention/minGRU pair at one `d_model`, five seeds each, on the common recipe: 12 layers, context 512, batch 32, SwiGLU FFN, `qk_norm` on, `muon_ns5_adamw`, a 50M-token cosine schedule, FineWeb-derived corpus. Each arm runs at **its own** measured learning-rate argmin — attention at 8× the repo's base LR, minGRU at 4×. §4 is about why that choice is not cosmetic.

Two further measurement decisions are worth stating, because each could have produced a different number.

**Curves are keyed on step, not on the logged token count.** The `tokens` field in `metrics.jsonl` does not survive a resume: `tokens_seen` resets to zero while `start_step` is restored from the checkpoint, so the token axis restarts mid-run. The blast radius is exactly 2 of 1,099 CE training runs, and both are on this ladder — `cx32lad1536_w1536_attention_lr80_s1337` and `_s42`, each recording `done.tokens = 25.41M` for a complete 50M-token run (`max_steps` 3051, so the loop runs steps 0-3050). `final_val` is unaffected, so every loss table stands; a token-axis analysis would have silently read half the axis for those two. This is one of three defects reported in §8; the other two are recall-side and do not touch this section. Step is exact, so tokens are recovered as step × 32 × 512.

**Differencing is per seed, before interpolation.** The attention and minGRU curves are differenced within a seed pair, and the last sign change on that per-seed difference is located by linear interpolation. A crossing read off two *mean* curves cannot tell you whether five seeds agree; this one can, and they do — 5 of 5 seeds resolve a crossing at every width.

### 3.2 The ladder

**Protocol A — each arm at the argmin of a 10M probe** (the ladder as first run):

| width | attention params | minGRU params | crossing token | 95% CI | per-seed range |
|---|---|---|---|---|---|
| 384 | 40.6M | 49.4M | **10.70M** | [10.24, 11.16] | 10.29–11.28 |
| 768 | 123.7M | 159.0M | **11.53M** | [11.41, 11.64] | 11.39–11.65 |
| 1152 | 249.3M | 328.7M | **12.60M** | [12.35, 12.85] | 12.36–12.90 |
| 1536 | 417.5M | 558.6M | **13.55M** | [12.96, 14.13] | 12.91–14.23 |

**Protocol B — each arm at the argmin of a 50M grid, the budget actually spent** (G9/G9b,
2026-09-09; attention 2×/1×/1×/0.5× of base by width, minGRU the same):

| width | crossing token | 95% CI | per-seed range | step from the rung below |
|---|---|---|---|---|
| 384 | **10.71M** | [10.34, 11.08] | 10.28–11.10 | — |
| 768 | **12.37M** | [12.09, 12.66] | 12.00–12.55 | +1.67M [+1.16, +2.17], 5/5 |
| 1152 | **13.46M** | [13.17, 13.74] | 13.13–13.71 | +1.08M [+0.80, +1.37], 5/5 |
| 1536 | **13.44M** | [13.27, 13.61] | 13.31–13.63 | **−0.02M [−0.33, +0.30], 2/5** |

Protocol A's rungs are reproduced here from the same boards by `scripts/crossing_token.py`
at 10.72M / 11.54M / 12.62M, within 0.02M of the published values — the estimator is the
same one, written down.

Two things move between the protocols. The crossing itself moves **+0.83M at d768 and
+0.84M at d1152 and not at all at d384** (10.72 → 10.71), so re-tuning is not a uniform
shift. And attention's endpoint margin over minGRU **widens** from ≈0.15 nats to ≈0.23 at
every width: tuning both arms to the horizon they are spent over helps attention more.

> **Footnote — the w1536 rung pairs two suites, and was audited for it.** The fourth rung draws attention from `crossover_ladder1536` and minGRU from `crossover_ladder1536_mingru`; every other rung is within-suite. It was therefore checked field by field against the per-run `config.json`, which is authoritative; the `recipe.json` is not, because it predates `LEGACY_RECIPE_DEFAULTS` and omits fields it did not record. Both w1536 arms and both w1152 arms carry `fused_ce=True`, `compile=False`, `tf32=True`, `dtype=bf16`, `grad_accum=1`, `batch_size=32`, `max_steps=3051`, `schedule=cosine`. **There is no `fused_ce` confound.** The minGRU suite's `recipe.json` reads `fused_ce: null`, which looks like one and is not — every run in it set `fused_ce=True`. That null is the unrecorded-field hazard the repo has since fixed, surfacing in an artifact rather than in a decision.
>
> One asymmetry is real: **`workers` is 2 for the w1536 attention suite and 1 for the w1536 minGRU suite.** Tenancy changes throughput, not the computation, so `final_val` and the crossing are unaffected — but **no wall-clock or tokens/second comparison may cross that rung.** Every other rung runs both arms at tenancy 2 inside one suite.

**(a) The sequence rises and then flattens; three of four steps separate.** Under protocol A every consecutive 95% interval is disjoint from its neighbour: 11.16 < 11.41, 11.64 < 12.35, 12.85 < 12.96, a total move of **+27%** across 4× width and 10.3× parameters. **Under protocol B the top step disappears.** Paired by seed, d384 → d768 is +1.67M and d768 → d1152 is +1.08M, both 5 of 5 and both disjoint from zero; d1152 → d1536 is **−0.02M [−0.33, +0.30]**, 2 of 5, not separable from no change. The total move to the highest rung is +25.7%, and it is all spent below d1152. Whether that is saturation or a peculiarity of d1536 cannot be settled from four points: a fifth rung at d_model 1920 is queued (G6) and is the measurement that decides it. The two rungs either side of the flat step also sit at different tenancy (2 and 1), forced by VRAM; cross-board replication at d384/d768 puts tenancy's effect on `final_val` at 0.0000–0.0128 nats, far too small to produce a 0.02M crossing shift in either direction, but it is not zero and is recorded here rather than assumed away. Fitted to these points the crossing goes as width^0.168 (R² = 0.959) and as params^0.100 (R² = 0.968). Those exponents are **verified** as a description of four points. Reading them as a law is **inferred**, and four points do not determine a functional form. We report the number and decline the extrapolation.

**(b) The endpoint gap is flat while the crossing is not.** Held-out CE at 50M, minGRU − attention, paired:

| width | endpoint gap (minGRU − attention) | 95% CI | seeds favouring minGRU |
|---|---|---|---|
| 384 | +0.1471 | [+0.1385, +0.1557] | 0 of 5 |
| 768 | +0.1585 | [+0.1485, +0.1685] | 0 of 5 |
| 1152 | +0.1517 | [+0.1353, +0.1681] | 0 of 5 |
| 1536 | +0.1426 | [+0.1212, +0.1640] | 0 of 5 |

These two tables are not in tension. They answer different questions. "Who wins at the end of this schedule" is width-stable across a 4× span. "At what token does the winner change" moves monotonically over the same span, with disjoint intervals. A board reporting only the endpoint — which is what the ladder board reported — cannot see the second quantity, and concludes that width does not move this comparison. Width does not move the *ranking at 50M*; it moves the *crossing*. **Verified**, both halves.

**(c) At a fixed budget, width alone flips the winner.** At 12M tokens, w384 has crossed and attention wins; w1536 has not crossed and minGRU wins. Same architectures, same recipe, same budget, opposite answer, decided by width. **Verified from the table.**

In tokens-per-parameter the crossing moves the other way, and sharply. Every denominator below is the attention parameter count.

| width | tok/param at 50M (attn) | tok/param at 50M (minGRU) | crossing, tok/param (attn) |
|---|---|---|---|
| 384 | 1.232 | 1.012 | **0.264** |
| 768 | 0.404 | 0.314 | **0.093** |
| 1152 | 0.201 | 0.152 | **0.051** |
| 1536 | 0.120 | 0.090 | **0.032** |

The crossing arrives later in tokens and much earlier in tokens-per-parameter. Which axis you plot decides which word applies; the headline here is the token axis, because that is what a screening budget is denominated in. Neither axis flatters this ladder: Chinchilla-optimal is ~20 tokens/parameter, the best-fed rung of the 50M ladder is 1.232, and the crossings sit between 0.032 and 0.264. §3.5 adds the one board that escapes this range.

One thing this ladder is *not* is a parameter-count effect in minGRU's favour. minGRU carries **22–34% more** parameters than attention at every width and still loses at 50M on every seed at every width.

### 3.3 Validating the method against a known answer

The pipeline above is new, so it was run on a suite whose crossings are already published. On `crossover50m` it recovers both:

| crossing | this pipeline | published |
|---|---|---|
| early (minGRU overtakes attention) | **1.03M** | 1.05M |
| late (attention overtakes minGRU) | **12.33M** | 12.35M |

Both to within 0.02M. The per-seed differencing and the step keying do not shift the answer on a suite where the answer was already known by a different route. **Verified.**

### 3.4 A second axis moves it further, and its confound is not optional

This is not a width result. It is here because it is the same measurement on a second axis, and it calibrates the width effect: trading context length against batch size moves the crossing further than the entire width ladder does.

The comparison is otherwise tight. Same model — `d_model` 768, 12 layers, 12 heads. Same **16,384 tokens per step**: batch 32 × ctx 512 against batch 8 × ctx 2048. Same 50M budget, same 3051-step cosine (steps 0-3050), same shared LR (6e-4 / 0.025), same five seeds. Suites `crossover50m` and `crossover50m_ctx2048`. Neither contains a resumed run, so the §3.1 defect cannot reach this row. The ctx 512 row is the same `crossover50m` measurement as §3.3.

| context | batch | crossing token | per-seed |
|---|---|---|---|
| 512 | 32 | **12.33M** | 12.46, 12.56, 12.20, 12.01, 12.44 |
| 2048 | 8 | **15.97M** | 15.61, 15.79, 16.43, 15.29, 16.72 |

Paired delta **+3.63M** [+2.93, +4.34], t = +14.3, 5 of 5 seeds. That is **+29%** — the same order as the +27% the whole width ladder buys across 10.3× the parameters, bought here by one change to the input pipeline.

**The confound is not optional, and we are not burying it.** Batch was lowered 4× to hold tokens-per-step fixed, so context and batch moved together. Batch is a known ranking confound in this repository's own record: the batch-8 board produced no crossing at all through 7.38M tokens on any seed. So the claim is the joint one — a 4× context increase at constant tokens-per-step, with batch traded to pay for it, moves the crossing **+3.63M**. That is **verified**. Attributing any part of it to context alone is **not measured**, cannot be disentangled from committed data, and is not queued.

One limit of the estimator belongs here rather than in a reviewer's report. At ctx 2048, attention against `hybrid_mingru10_attn2` crosses on 5 of 5 seeds — but at **31.94M with a 95% interval of [14.19, 49.70]**, per-seed 17.27, 17.61, 34.22, 41.11, 49.50. We do not report that as a crossing token. The two arms *tie* on final loss at ctx 2048 (+0.0018 [−0.0072, +0.0108], 1 of 5), and interpolating a sign change between curves that tie divides by a difference near zero. The estimator is only as sharp as the separation between the arms. Every crossing in §3.2 separates its arms by at least +0.1426 nats at 50M, which is why those intervals are narrow and this one is not.

### 3.5 The one axis that does not move it: the token budget

E35 ran the w384 rung a second time at a 200M-token budget — `crossover200m_w384`, 20 runs, n = 5 per arm, batch 32, ctx 512, compiled. Same arms, same width, same learning rates, same seeds. Only the budget, and therefore the cosine horizon, changes.

| budget | crossing | 95% CI | per-seed |
|---|---|---|---|
| 50M | **10.70M** | [10.24, 11.16] | 10.29–11.28 |
| 200M | **10.61M** | [10.21, 11.01] | 10.21, 10.40, 10.63, 10.78, 11.04 |

5 of 5 seeds cross in both, and the intervals almost coincide. A 4× longer cosine leaves the crossing where it was — and it does so while putting that crossing at completely different points in the schedule: **21%** of the way through the 50M run and **5.3%** through the 200M run.

This is the first axis tested in either section that does **not** move the crossing. Width moves it (+27%, §3.2), context-with-batch moves it (+3.63M, §3.4), the LR protocol moves it (−0.81M, §4.1). The training budget does not. The crossing is set by tokens consumed — not by where it falls in the schedule, and not by how long you intended to train. **Verified.**

The practical consequence is the only good news for short screens in this paper, and it should be said in exactly those terms: **a screen does not have to run the full budget to locate the crossing.** Every other result here says a screening answer depends on the recipe that produced it. This one names the axis where it does not, at the one width where that is measured.

**This does not overturn the truncation result, and the asymmetry is itself the finding.** The earlier paper measured that truncating the cosine to 20M moved the late crossing from 12.34M to 14.58M, and that restoring the 50M cosine while stopping at 20M recovered 12.34M. Here a 4× *longer* schedule moves nothing. Shortening the schedule through the crossing region moves the crossing; lengthening it beyond does not. Both are measured and they are not in tension. Reading the asymmetry that way is **inferred**.

E35 is also the first board outside this paper's under-training regime: 200M tokens on the 40.6M-parameter attention shape is **4.93** tokens/parameter, against the 0.09–1.23 of every other board. It remains roughly 4× below Chinchilla-optimal.

---

## 4. The crossing is a function of the tuning protocol

### 4.1 One factor, changed alone

The w768 rung of §3 and the `crossover50m` late crossing of §3.3 are the same architectures at the same shape, and they disagree. That is not a replication failure. They differ by exactly one factor, deliberately: whether both arms are handed the same learning rate.

Same shape (d768, L12, h12), same batch 32, same context 512, same 50M cosine schedule, same five seeds. Only the LR protocol changes.

| protocol | LR | crossing | per-seed |
|---|---|---|---|
| shared LR, both arms | `lr` 6e-4 / `matrix_lr` 0.025 | **12.33M** | 12.46, 12.56, 12.20, 12.01, 12.44 |
| per-arm argmin | attn 4.8e-3 / 0.2; minGRU 2.4e-3 / 0.1 | **11.53M** | 11.57, 11.52, 11.65, 11.39, 11.50 |

Paired shift **−0.81M** [−1.07, −0.55], t = −8.6, 5 of 5 seeds. The per-seed bands do not overlap: the shared-LR draw spans 12.01–12.56 and the per-arm draw spans 11.39–11.65.

### 4.2 The mechanism, measured

The shift is not a general appeal to tuning. It has a measured cause. In the 10M-token LR probe — n = 1 per cell, 42 probe runs across four widths — **attention's argmin is 8× base at every width and minGRU's is 4× base at every width**. Every argmin is interior to its grid; the sweep was extended twice to make that true, because an optimum at the edge of a grid is not an optimum. The 2× separation is stable across the whole width span.

That asymmetry is what makes a shared LR a confound rather than a simplification: a board that gives both arms one learning rate reads them at different distances from their own optima. The cost of that distance is measurable: near the argmin at 10M, one grid step in either direction costs **0.03–0.05 nats**, the largest observed being 0.053 for minGRU at w1152 going 4× → 8×.

The rule also extends past the pure arms, and the extension is not a blend of the two. An 8-minGRU / 4-attention stack wants **4× base** — the recurrent arm's argmin, not the 8× that attention wants: `hybrid_mingru8_attn4_lr40` − `hybrid_mingru8_attn4_lr80` = **−0.1241** [−0.1304, −0.1177], t = −54.5, 5 of 5. The majority mixer sets the optimum. **Verified** at width 384 and 200M tokens; that it holds at other widths or budgets is **not measured**.

That reaches further than the pure-arm boards. A hybrid-versus-attention board run at one shared learning rate sits under exactly the confound described above, with its two arms wanting different multipliers — which is why §5 now carries a caveat about its own headline.

### 4.3 What this invalidates, including in this repository

Every 50M board in this repository except the ladder hands both arms one learning rate. That includes `crossover50m`, whose ten arms all share `lr` 6e-4 / `matrix_lr` 0.025 — verified from `config.json`, not assumed from the suite name.

The consequence is specific and it lands on our own numbers. In `crossover50m`, Mamba-2 (+0.4348) and MLA (+0.3843) against attention are **not** clean architecture comparisons. Each is read at whatever distance from its own optimum a shared LR happens to place it, and no argmin is reported for Mamba-2 or MLA at any width or horizon — those are **not measured**. That the two rows are therefore unreliable as architecture claims is **inferred**, not verified; what is verified is that the two arms whose argmins *were* measured sit 2× apart. The paper does not lean on those rows, and no conclusion here depends on them.

The same reasoning applies outward. A shared-LR screen is not a cheap approximation to a per-arm-tuned one. It is a different measurement, and §4.1 prices the difference for the one pair where both exist.

### 4.4 The caveat we cannot close

> **RESOLVED AGAINST, 2026-09-09.** This section named the risk correctly and the risk
> materialised. G7 and G3 measured the 50M argmins directly: attention's is below 8× at
> every width tested and minGRU's is below 4×, by 0.086 to 0.255 nats. The section is kept
> verbatim below as the record of what was assumed and why; every number downstream of it
> is being re-measured. See the status block at the head of this file.

The argmins were measured at **10M** tokens and spent at **50M**. Two invariances are needed for that to be sound, and only one of them is measured.

That the argmin does not move with **width** is **verified** — 8× for attention and 4× for minGRU at every width in the probe. That it does not move with **horizon** is **not measured**. The probe was never re-run at 50M. Nothing here rules out the argmins separating, converging, or crossing between 10M and 50M, and the 0.03–0.05 nats per grid step of §4.2 is itself a 10M measurement.

E35 does not close it. That board re-located one argmin at 200M — the hybrid's, §4.2 — but ran attention only at 8× and minGRU only at 4×, so neither pure arm's optimum was re-measured at the longer horizon.

This is the load-bearing assumption of both §3 and §4, it is stated as an assumption, and no run in this paper repairs it.

---

## 5. Placement, not ratio

The hybrid recipe is normally specified as a ratio: how many attention layers survive in an
otherwise recurrent stack. On this board the count does not decide the outcome; placement does. The
placement result is also the one this section can defend, because it is measured between hybrids
inside one suite.

`crossover50m_ratioplace32` runs five minGRU/attention interleavings and pure `attention` in one
suite at 50M tokens, five seeds. One recipe for every arm, so seed variance is common-mode and the
test is paired per seed. Figures are `final_val`, differenced within seed; count and placement come
from `layer_mixers` in each `config.json` (**verified**).

| arm | attn layers | placement | mean Δ vs `attention` | 95% CI | seeds better |
|---|---:|---|---:|---|---|
| `hybrid_mingru_periodic` | 3 | interleaved, every 4th | **−0.0242** | [−0.0294, −0.0191] | 5/5 |
| `hybrid_mingru8_attn4` | 4 | last four | **−0.0165** | [−0.0210, −0.0121] | 5/5 |
| `hybrid_mingru10_attn2` | 2 | last two | +0.0132 | [+0.0075, +0.0189] | 0/5 |
| `hybrid_mingru11_attn1` | 1 | last | +0.0348 | [+0.0284, +0.0412] | 0/5 |
| `hybrid_mingru_bookend` | 2 | first and last | +0.0415 | [+0.0361, +0.0469] | 0/5 |

Read the count column. The board is **not ordered by attention count**: three interleaved layers
beat four stacked ones, and two layers split to the ends are worse than one at the end. Absolute
means: `periodic` (3) 4.1939 · `8+4` (4) 4.2016 · `attention` (12) 4.2181 · `10+2` (2) 4.2313 ·
`11+1` (1) 4.2530 · `bookend` (2) 4.2596.

Five paired contrasts, every seed agreeing in sign:

| comparison | attn layers | paired Δ | seeds |
|---|---|---|---|
| `bookend` − `10+2` | 2 vs 2 | **+0.0283** [+0.0259, +0.0307] | 0/5 |
| `periodic` − `8+4` | 3 vs 4 | **−0.0077** [−0.0118, −0.0036] | 5/5 |
| `bookend` − `11+1` | 2 vs 1 | **+0.0067** [+0.0028, +0.0106] | 0/5 |
| `10+2` − `11+1` | 2 vs 1 | −0.0216 [−0.0259, −0.0174] | 5/5 |
| `8+4` − `10+2` | 4 vs 2 | −0.0298 [−0.0334, −0.0261] | 5/5 |

Row 1 is the controlled test — same count, different position. Splitting the two layers to the ends
costs **0.0283 nats**, t = +32.4, every seed: **9x the 0.0031 max determinism floor, from moving one
layer.**

Row 3 is the sharpest line. `bookend` − `11+1` = **+0.0067**, t = +4.8, 0 of 5 seeds better: a
second attention layer, placed at position 0, leaves the model **worse than not adding it at all.**

Rows 4 and 5 stop this becoming "count does not matter": with placement fixed — all attention at the
end — count helps monotonically, −0.0216 then −0.0298. The ratio is real, and well defined only once
the layout is stated.

### The hybrid-versus-attention rows carry a tuning confound

The sign column of the first table looks settled: two hybrids ahead of pure attention on all five
seeds, one behind on all five. It is not, and this paper's own thesis is why.
`crossover50m_ratioplace32`, `crossover50m_parity32` and `crossover50m_copy32` all run at lr 6e-4 /
matrix_lr 0.025 — **1x base**, verified from `config.json`, which is nobody's argmin. Attention's
argmin is 8x base. The hybrid's is **4x base**, the recurrent arm's: `hybrid_mingru8_attn4_lr40` −
`hybrid_mingru8_attn4_lr80` = **−0.1241** [−0.1304, −0.1177], t = −54.5, 5/5 — the majority mixer
sets the optimum (**verified** at width 384 and 200M tokens; other widths and budgets **not
measured**). These boards read attention 8x below its optimum and the hybrids 4x below theirs: **the
arm that loses is the one read from further away.**

The mitigation is real and partial. At width 384 and 200M tokens, each arm nearer its own argmin,
the hybrid's advantage **widens rather than vanishes** — −0.0777 [−0.0891, −0.0663], t = −18.9, 5/5,
against −0.0165 on the shared-LR board. Correct tuning did not erase the effect at the one place
both arms are measured properly. That this carries to width 768 and 50M is **inferred, not
measured**, and the clean test — this board re-run at each arm's own argmin — is **not queued**.

The placement contrasts are untouched: every row of the second table is hybrid-against-hybrid within
one suite at one learning rate, where a shared LR is common-mode and differences out. **+0.0283
stands.** The confound is on the hybrid-versus-attention rows and nowhere else — a caveat, not a
retraction.

### Parameter parity does not remove it, and does not make it large

`crossover50m_parity32` forces minGRU expansion to x1 so a hybrid cannot buy its win with extra
parameters. Both winning layouts keep their sign:

- `hybrid_mingru_periodic_x1` − `attention` = **−0.0110** [−0.0159, −0.0060], 5/5
- `hybrid_mingru8_attn4_x1` − `attention` = **−0.0053** [−0.0087, −0.0019], 5/5

Both shrink. **−0.0053 is 1.7x the 0.0031 determinism floor** — real, consistently signed, and
small. Nothing here supports the reading that hybrids are a free win on cross-entropy, and these are
vs-attention rows, so they carry the confound above as well.

### Pure minGRU is not competitive on CE at either context length

In `crossover50m_copy32`, `mingru` − `attention` = **+0.2299** [+0.2188, +0.2409], 0/5, and
`hybrid_mingru8_attn4` − `mingru` = **−0.2452**, 5/5 (no interval recorded). At ctx 2048
(`crossover50m_ctx2048`) the pure arm is further back, **+0.2911**, 0/5, while
`hybrid_mingru10_attn2` − `attention` = **+0.0018** [−0.0072, +0.0108], 1/5 — a **tie**, not a win.
This section's claims are ctx 512 claims.

---

## 6. Cross-entropy does not see what the recurrent arms cannot do

### Why solve rate, and not mean recall

The recall probe is MQAR — multi-query associative recall over synthetic key–value sequences, scored
by exact match at the query positions. Its outcome is bimodal: a run lands at a floor or at exactly
1.000, so a mean averages two disjoint outcomes and describes neither. The measurable quantity is
the **fraction of seeds that solve the cell**, with a 95% Wilson interval on the binomial, not a
mean recall with a t-interval.

Every cell below is **n = 15 independent seeds**, except seq 511 / batch 128 where n = 10. An
earlier aggregation reported n = 30 by pooling reruns that share a seed and an effective learning
rate; **0 of those 106 rerun pairs disagree on `solved`**, so the bimodal outcome is reproducible at
fixed seed and a solve rate at n = 15 is a stable quantity rather than a coin flip.

### Zero solves for the recurrent arms at seq ≥ 255

Batch 64 unless stated:

| cell | `attention` | `mingru` | `gdn` |
|---|---|---|---|
| seq 63, 3,000 steps | 0/15 [0.00,0.20] | 0/15 [0.00,0.20] | 0/15 [0.00,0.20] |
| seq 255, 3,000 steps | **11/15** [0.48,0.89] | **0/15** [0.00,0.20] | **0/15** [0.00,0.20] |
| seq 511, 3,000 steps | 4/15 [0.11,0.52] | 0/15 [0.00,0.20] | 0/15 [0.00,0.20] |
| seq 511, 3,000 steps, batch 128 | **8/10** [0.49,0.94] | **0/10** [0.00,0.28] | **0/10** [0.00,0.28] |

The seq 63 row discriminates nothing — every arm is 0/15, attention included. We first read it as a
recurrent failure.

At seq ≥ 255 the two recurrent arms record **zero solves in 80 runs**: `mingru` 40, `gdn` 40. Two of
those cells separate by interval — at seq 255 and seq 511 / batch 128, attention's Wilson interval
is disjoint from both. **The middle cell does not** — [0.11, 0.52] overlaps [0.00, 0.20] between
0.11 and 0.20, so 4/15 against 0/15 is a consistent sign, not a separation. Calling the row three
disjoint cells would overclaim by one.

### The dissociation

Set this against §5. On held-out CE at 50M tokens `hybrid_mingru8_attn4` **beats** attention by
0.0165 nats on 5 of 5 seeds; pure `mingru` trails by 0.23 nats. At seq ≥ 255 the pure recurrent arms
return nothing — 0 solves in 80 runs — where attention reaches 11/15. A metric separating these arms
by hundredths of a nat is silent about a capability difference that is total.

Say precisely what that is. It is a statement about **the metric**: a CE-only screen puts a
recurrent stack 0.23 nats behind attention and its best hybrid ahead of it, and neither ordering
survives contact with recall. Whether the hybrid also inherits the recurrent floor at long sequence
was the open question when this section was first drafted. It has since been measured, and the
answer closes the section.

### The same task and metric, opposite orderings

Same task, metric and architectures, varying only difficulty and step budget. Arms come from run
names, reruns are collapsed by effective learning rate, and `—` marks a cell not run:

| cell | `attention` | `hybrid_mingru10_attn2` | `mingru` | `gdn` | `hybrid_gdn_periodic` |
|---|---|---|---|---|---|
| seq 15, 3,000 steps | **15/15** [0.80,1.00] | 5/15 [0.15,0.58] | 1/15 [0.01,0.30] | 7/15 [0.25,0.70] | 9/15 [0.36,0.80] |
| seq 15, 9,000 steps | **15/15** [0.80,1.00] | 12/15 [0.55,0.93] | 1/15 [0.01,0.30] | 13/15 [0.62,0.96] | 15/15 [0.80,1.00] |
| seq 23, 3,000 steps | **9/15** [0.36,0.80] | 8/15 [0.30,0.75] | 0/15 [0.00,0.20] | 1/15 [0.01,0.30] | 7/15 [0.25,0.70] |
| seq 23, 9,000 steps | **15/15** [0.80,1.00] | 12/15 [0.55,0.93] | — | — | — |
| seq 31, 3,000 steps | **2/15** [0.04,0.38] | **12/15** [0.55,0.93] | 0/15 [0.00,0.20] | 2/15 [0.04,0.38] | 6/15 [0.20,0.64] |
| seq 31, 9,000 steps | 12/15 [0.55,0.93] | **13/15** [0.62,0.96] | 0/15 [0.00,0.20] | 6/15 [0.20,0.64] | 13/15 [0.62,0.96] |

At a fixed 3,000-step budget, difficulty alone carries attention from **15/15 at seq 15 to 2/15 at
seq 31** while `hybrid_mingru10_attn2` goes the other way, **5/15 to 12/15**: the two arms swap ends
of the ranking on one task, one metric. Now hold seq 31 and add budget — attention recovers 2/15 →
12/15 while the hybrid barely moves, 12/15 → 13/15. The seq 31 ordering is a statement about **where
training stopped**, not about the architectures.

**The reversal replicates.** `hybrid_mingru8_attn4` traces the same crossing at a different
attention budget and a different placement: **7/15** [0.25, 0.70] at seq 15, where attention's 15/15
[0.80, 1.00] is disjoint and ahead, then **8/15** [0.30, 0.75] at seq 31 against attention's 2/15
[0.04, 0.38]. Pure `mingru` meanwhile stays at **0–1/15 in every cell it appears in**. Two hybrids,
two layouts, one crossing against attention as difficulty rises at fixed budget; the reversal
belongs to the hybrids, not to recurrence. `mingru` is not slow to form the induction head, it does
not form it. That arm's 9,000-step cells were still running when this was written.

### seq 255: the hybrids keep attention's recall

This subsection was a limitation. Two trends pointed at one unrun cell from opposite directions:
over seq 15 → 31 the hybrids gained on attention, over seq 63 → 255 the pure recurrent arms
collapsed to zero. E36 ran it. **The hybrid trend continued; the collapse did not reach the
hybrids.** Its runs are recipe-identical to the existing seq-255 rows and pool with them, bringing
MQAR to 967 distinct runs.

seq 255, p = 64, batch 64, 3,000 steps, n = 15. The paired columns count seeds the arm solved and
attention did not, seeds attention solved and the arm did not, and ties; p is an exact two-sided
sign test on those fifteen seeds:

| arm | attn layers | solved | 95% Wilson | arm / attn / tied | p |
|---|---:|---|---|---|---|
| `hybrid_mingru8_attn4` | 4 | **15/15** | [0.80, 1.00] | 4 / **0** / 11 | 0.125 |
| `hybrid_mingru_periodic` | 3 | **14/15** | [0.70, 0.99] | 4 / 1 / 10 | 0.375 |
| `hybrid_mingru10_attn2` | 2 | **13/15** | [0.62, 0.96] | 4 / 2 / 9 | 0.688 |
| `attention` | 12 | 11/15 | [0.48, 0.89] | — | — |
| `mingru` | 0 | **0/15** | [0.00, 0.20] | 0 / **11** / 4 | **0.001** |
| `gdn` | 0 | **0/15** | [0.00, 0.20] | 0 / **11** / 4 | **0.001** |

**No hybrid separates from attention.** Nine to eleven of the fifteen seeds tie in each comparison,
and p runs 0.125, 0.375, 0.688. The point estimates are higher; that is not a win. What the rows
support is *at least as good*: all three land inside or above attention's interval, and for
`hybrid_mingru8_attn4` there is **no seed on which attention solved and the hybrid did not** — four
the other way, none against. Reading four discordant seeds as a win is the error this paper is
about.

The significant contrast is the other one: `mingru` and `gdn` solve nothing where attention solves
11 of 15, paired, **p = 0.001** each. So the pre-registered reading is met — every hybrid's rate
lands inside or above attention's Wilson interval, the hybrid **inherits attention's recall**, and
§5's cross-entropy win is **not bought with a capability.**

That completes the dissociation. Held-out CE puts pure `mingru` 0.23 nats behind attention — a gap a
board renders as modest — while at seq 255 it solves nothing in fifteen seeds and attention solves
eleven. Add four attention layers to that same recurrent stack and the capability returns in full.
**The capability tracks the presence of global attention layers, not the bulk composition of the
stack: 4 of 12 layers is enough, and 8 recurrent layers cost nothing on this axis.** Cross-entropy
separated these three configurations by hundredths of a nat and ordered them wrongly against a
capability that is all-or-nothing.

One cell stays open. Each hybrid now has exactly one long-sequence cell, at seq 255; **seq 511 is
unmeasured for every hybrid.**

---

## 7. Equal seconds and equal tokens rank depth in opposite directions

Every quality ranking so far is token-matched. Section 6.9 showed that a wall-clock cost basis
reorders the hybrid board. This is the stronger version, on a board where the architecture never
changes and only the depth does. The two cost bases do not reorder the arms. They reverse the
sign of the effect.

`crossover50m_loop32` and `crossover_wcloop32` run the same four attention-only variants against
the same 12-layer `attention` baseline. No minGRU anywhere. The first gives every arm the same
tokens. The second gives every arm the same seconds. Δ is arm − baseline, so negative is better,
and the fraction counts how many of the five paired seeds beat the baseline.

| arm | token-matched (`crossover50m_loop32`) | wall-clock matched (`crossover_wcloop32`) |
|---|---|---|
| `attn3` | **+0.1916** [+0.1876, +0.1956] 0/5 | **−0.2400** [−0.2652, −0.2148] 5/5 |
| `attn6` | +0.0776 [+0.0742, +0.0809] 0/5 | −0.2208 [−0.2429, −0.1987] 5/5 |
| `looped_attn6x2` | +0.0137 [+0.0090, +0.0185] 0/5 | −0.0136 [−0.0468, +0.0196] 4/5 (ns) |
| `looped_attn3x4` | +0.0558 [+0.0491, +0.0625] 0/5 | +0.0314 [+0.0097, +0.0532] 0/5 |

`attn3` and `attn6` reverse completely. Both reversals carry |t| > 26 and 5 of 5 seeds agreeing
on the sign in *each* direction (**verified**). This is not a wide interval wobbling across
zero. It is two tight intervals pointing opposite ways, differing only in what the experimenter
chose to hold equal. Ask whether a 3-layer stack is worse than a 12-layer stack and the answer
is +0.19 nats or −0.24 nats, decided before any model was trained.

The looped arms do not follow the shallow ones, and one of them we cannot resolve.
`looped_attn3x4` loses on both bases — +0.0558 token-matched, +0.0314 wall-clock-matched, 0 of 5
seeds either way — so weight sharing does not buy back what depth-3 gives up. `looped_attn6x2`
is the honest hole in the table. Token-matched it is a small, clean loss (+0.0137, 0/5).
Wall-clock-matched its interval crosses zero (−0.0136 [−0.0468, +0.0196], 4 of 5 seeds), and we
report it as not significant: the point estimate favours the loop, the interval does not exclude
the baseline (**inferred**). We do not claim looping is free at equal seconds. We claim that at
n = 5 we cannot tell.

**The hybrid is not rescued by its speed.** The obvious reading of the reversal above is that
anything faster gains under wall-clock matching, so the recurrent hybrid should be the
beneficiary. It is not. In `crossover_wallclock32`, `hybrid_mingru10_attn2` − `attention` =
**+0.0595** [+0.0224, +0.0965], 0 of 5 seeds, at equal seconds; the same pair is +0.0530 at
equal tokens. Changing the cost basis moves the gap slightly *against* the hybrid. The mechanism
is in the token counts, not in anything subtle: at equal wall clock, attention spent **73.20M
tokens in 684s** while the hybrid spent **60.19M in 687s** (the existing draft reports the same
two runs to one decimal, 684.1s and 686.5s). The GDN hybrids are far worse — **+0.7368** and
**+0.8267** — for the same reason, in cruder form: they reached only **22.0M** and **20.0M**
tokens in the same time. Speed converts to quality only through tokens, and these arms did not
buy enough of them.

**No alternative shape wins either.** `crossover50m_shape32` trades depth for width at fixed
non-embedding parameters — the other way to spend the same budget. Paired against that suite's
`attention` (4.2186), `attn6_w576` is **+0.1424** [+0.1339, +0.1510], `attn6_w512` **+0.1697**
[+0.1631, +0.1764] and `w384_attention_lr10` **+0.1703** [+0.1612, +0.1794], each losing on 0 of
5 seeds. The penalty is flat at **0.14–0.17 nats**, and no shape here recovers what the depth
reduction costs.

The conclusion, stated plainly, because it governs how the rest of this paper should be read:
**wall-clock matching is not a robustness check on token matching.** A robustness check asks the
same question under jitter and expects the same answer. These are two different questions —
"which model is better per token" and "which model is better per second" — and on `attn3` and
`attn6` they have opposite answers with tight intervals and unanimous seeds. Neither is the true
one. A paper that reports one basis and not the other has not simplified its presentation. It
has chosen its result, and nothing in the numbers tells a reader that a choice was made.

---

## 8. Threats to validity, defects, and claims this paper withdraws

### 8.1 Scale

Every rung of the ladder is deeply under-trained. 50M tokens on 40.6M–558.6M parameters is
**0.09–1.23 tokens per parameter**, and the crossings themselves sit at **0.032–0.264 tokens per
parameter** — 0.264 at width 384, 0.093 at 768, 0.051 at 1152, 0.032 at 1536. **Nothing here
speaks to the compute-optimal regime.** Whether the crossings survive there is unmeasured, in
either direction.

The rest of the frame is equally narrow: one corpus, one depth (L12), one optimizer family, and
one context length for the main ladder (512). The learning-rate probe that fixes each arm's
argmin is **n = 1 per cell**. Every argmin is interior to its grid, and the argmin does not move
with width (**verified**); that it does not move with *horizon* is **not measured**, and it was
measured at 10M tokens and then spent at 50M. [SUPERSEDED 2026-09-09: now measured, at n = 5,
and it **does** move — every arm wants a lower LR at 50M than the 10M probe assigned it.]

This paper's own thesis forbids extrapolating from the above. If a ranking moves under batch
size, learning-rate horizon, cost basis and now width, then a ranking measured at 0.09–1.23
tokens per parameter is a statement about that regime and no other.

E35 has since run, and it puts one measured point inside that caveat. 200M tokens at width 384,
20 runs, n = 5 per arm, is **4.93 tokens per parameter** on the 40.6M attention shape — 4× the
budget of every other board here and the first rung outside the 0.09–1.23 range. What it found
is a negative result, reported in §3: the crossing does **not** move with the budget. That
narrows the regime objection at exactly one rung. It does not answer it. 4.93 tokens per
parameter is still roughly 4× below Chinchilla-optimal (~20), and the ladder's other three
widths have no second budget at all.

### 8.2 Baseline choice

minGRU is the simplest recurrent baseline available, and that is why it is here. It is not the
strongest. `crossover50m` carries Mamba-2 and MLA rows — **+0.4348** and **+0.3843** against
attention — and we decline to use them, because every arm in that suite shares one learning rate
(lr 6e-4 / matrix_lr 0.025, verified from `config.json`). Attention's measured argmin is 8× base
and minGRU's is 4× base at every width; a shared-LR board therefore reads its arms at different
distances from their own optima, and one grid step off the argmin costs **0.03–0.05 nats**
(largest observed 0.053). Those two rows are not clean architecture comparisons, and no argument
in this paper leans on them.

The problem is not confined to exotic arms. At width 384 and 200M tokens, `hybrid_mingru8_attn4`
run at 4× base beats the same arm at 8× base by **−0.1241** [−0.1304, −0.1177], t = **−54.5**, 5
of 5 seeds. An 8-minGRU/4-attention stack wants the recurrent arm's argmin, not attention's: the
majority mixer sets the optimum (**verified** at that width and budget; that it holds at other
widths and budgets is **not measured**). So the general statement is: **any board comparing
architectures with different majority mixers at one learning rate is reading them at different
distances from their own optima.** That reaches the hybrid rows §5 is built on, which run at 1×
base — nobody's argmin. §5 carries the consequence for its own numbers.

This is a limitation we impose on ourselves, not a defence of the comparison we do make. The
recurrent side of this paper is one architecture, chosen for simplicity and tuned to its own
argmin. Whether a stronger state-space model crosses attention elsewhere, or at all, is untested
here.

### 8.3 Five defects found while preparing this paper

Five defects surfaced while these sections were being written. Two are in the training harness.
The other three are in the analysis that produced this paper's own ledger, and both were caught
the same way: a section writer cross-checked a derived number against an independently published
one, and the two did not agree. We state that provenance rather than reporting three tidy fixes:
it is the strongest evidence this paper has for the practice it recommends, and neither analysis
defect was visible from inside the analysis.

**1. The token counter resets on resume.** `nanolab/train.py` resets `tokens_seen = 0` while
restoring `start_step` from the checkpoint, so the `tokens` field written to `metrics.jsonl`
restarts mid-run while the step counter does not. The blast radius is exactly **2 of the 1,099
CE training runs**: `cx32lad1536_w1536_attention_lr80_s1337` and
`cx32lad1536_w1536_attention_lr80_s42`, both complete 50M-token runs (`max_steps` 3051, steps 0-3050) recording
`done.tokens = 25.41M`. The token axis of those two runs is wrong. `final_val` is read at end of
schedule and is indifferent to the token counter, so it is unaffected and **every loss table in
this paper stands** (**verified**). The consequence for method is larger than for the numbers,
which is why a two-run defect is worth reporting: a token-axis analysis over a resumed run reads
half the axis, produces a plausible curve, and gives no signal that anything is wrong. **This is
why every crossing in this paper is keyed on step**, with tokens derived as step × 32 × 512
rather than read from the logged field. Fixed on this branch, with a test that fails against the
pre-fix code.

**2. 135 recall runs were counted twice.** `mqar_e16` and `mqar_e16_board` share 75 run names;
`mqar_e28_p8` and `mqar_e8` share 60. Concatenating those ledger files counted each shared run
twice and inflated every affected denominator by 2x. Deduplicating on run name left **922 distinct MQAR runs**, the corpus this
paper's recall tables were computed on; E36 has since added 45, for 967. Two things matter more here than the count. Dedup had to test for
divergence rather than simply collapse: 30 of the duplicated pairs — SWA arms, outside this
paper's scope — carry *different* recall values under the same run name, so there the name does
not identify the recipe. And MQAR runs are ledger lines, not run directories; they are a
population **disjoint** from the 1,099 CE training runs above, which the ledger header had
conflated.

**3. The `arm` column mislabels 165 recall runs.** For runs written before the launcher took an
explicit arm, `arm` records `cfg.mixer` — which for a hybrid is its recurrent half. Every
`hybrid_mingru10_attn2` is filed as `mingru` (**90 runs**) and every `hybrid_gdn_periodic` as
`gdn` (**75 runs**). The run name is authoritative, and the repository already carries a test
asserting that the run name encodes the recipe; nothing was asserting the same of the `arm`
column. This one reached a headline. An early draft of §6 reported "minGRU beats attention at
p=8" — an artifact of pooling hybrid runs into the pure arm. **It is withdrawn.** Re-parsed from
the run name, pure minGRU solves 0–1 of 15 in every cell of that grid; whatever recall the
hybrid has comes from its attention layers, not from the recurrence.

**4. The learning-rate argmin was read off unequal seed sets.** `scripts/lr_argmin.py` took the
plain mean of whatever seeds each grid point happened to carry. On these boards seed 100 is the
easiest seed at every cell and 777 the hardest — 4.1229 against 4.1803 at `d_model` 1152, a
**0.0574 spread, eighteen times the 0.0031 rerun floor**, against argmin margins of 0.005 to
0.05. So enriching one point with seed 100 and not its neighbour moves that point down by more
than any margin on the grid. A board that added two seeds to one point per curve duly reported
that two argmins had **moved**; neither had. The fix is structural rather than a threshold: the
reader now keys by seed, reads the grid on the seeds common to every point and the runner-up
margin on every seed those two share, and returns a located argmin only when it is interior, off
at least three points, and ahead by more than the rerun floor. This defect reached a stage
readout and no section — but the argmin table it would have corrupted is the one §4.4 rests on.

**5. The data loader changes which tokens a run trains on when the GPU is crowded.** Above 150M
tokens `should_gpu_resident` decides by free VRAM at construction time, and this paper's split is
497.5M tokens, so the resident path requires **more than 9.27 GiB free**. Below that the loader
falls back to a memmap that samples with a CPU generator where the resident path uses a CUDA one:
the same seed, a different token stream. The fallback needs *less* memory, so it converts an
out-of-memory failure into a run that finishes and looks entirely ordinary — and nothing on disk
recorded which path a run had taken. A scheduling fault that put four boards on one device at
once made this concrete: nine completed runs whose provenance cannot now be established from
their records. Throughput does not settle it, since a degraded rate is equally consistent with
merely sharing the device. Every run now writes its path into its `start` record —
`gpu_resident`, `memmap`, or `unknown` — six of the nine are quarantined rather than deleted,
and the remaining three were re-run at the same seeds against the rerun floor. **That check
found a real one.** Two of the three agree within 0.0031 (|d| = 0.0021 and 0.0024); the third,
`w1536_attention` at 0.5x base and seed 100, differs by **0.0048** — so the fallback did fire
and that run did train on a different token stream. Every rerun records `gpu_resident`; every
original records `unrecorded`, which is exactly the blind spot being closed. Substituting the
rerun moves that width's argmin margin from +0.0125 to ~+0.011, so the argmin stands — but it
now stands on a measurement rather than on the assumption that a crowded device costs only
wall-clock. **No headline in this paper depends on the condemned run**: the six runs on which
§4.4's `d_model` 1152 result rests are the fastest in their cohort, which is only possible on
the resident path running alone.

Defects 2 and 3 are §8.4's principle one level up. **A summary field that merely looks like the
thing it summarises is the same hazard as a check that could not run.** An `arm` column is not a
recipe, and a concatenation of ledger files is not a set of distinct runs, but both render
without complaint into a table a reader — and an author — reads as though they were. The house
practice is to name such defects, gate the fix behind a flag that preserves the recorded
default, and then measure. The MoE router is the worked example: repaired so that it receives a
task gradient, `crossover50m_moe32d`'s dense control `moe_e1k1_raw` lands at 4.2185 against that
suite's `attention` at 4.2196 — within 0.001, which is what that control exists to check — while
the multi-expert arms still lose (+0.031, +0.039). Fixing a broken component did not change the
conclusion. It made the board readable.

### 8.4 Claims this paper does not make

`crossover50m_tie32` varies weight tying and value residuals. Two of its arms behave:
`attention_novr` **+0.0309** [+0.0268, +0.0350] and `attention_untied` **+0.0799** [+0.0741,
+0.0857] against `attention`, both losing on 0 of 5 seeds. The third does not.
`attention_untied_novr` reaches **11.1491 on all five seeds** — above uniform, which for this
vocabulary is ln 50257 = **10.82** — while its train loss descends normally to **4.156**. Five
seeds agreeing to four decimals on a loss worse than predicting nothing is not a finding about
weight tying. It is a symptom of something we have not identified. The cell is **withheld as an
artifact, not reported as a result**, the diagnostic is queued, and no conclusion here rests on
that arm.

The second withheld claim concerns the architecture this paper likes best. **No hybrid arm has
ever been run at sequence length ≥ 255.** That was true when §6 was drafted and it is no longer:
E36 ran the cell, and §6 reports it. We leave the record of it here because the sequence matters
— the claim was withheld while it was unmeasured, and it was withheld in the direction that
turned out to be wrong. The hybrids kept attention's recall; nothing in the short cells obliged
them to. Had we predicted, we would have had even odds at best.

Two items on that axis remain unmeasured, and we name them rather than let their absence read as
checks that passed. **Sequence 511 is unmeasured for every hybrid** — attention itself is only
4/15 there at batch 64, so the cell discriminates and is the natural follow-on. And a second
experiment is not queued at all: the E10 board (`crossover50m_ratioplace32`) re-run with each arm
at its own argmin, which is now the cleanest outstanding test of §5's hybrid-versus-attention
rows.

Both withheld claims follow the rule this program tries to hold to: **a check that could not run
must never report the same result as a check that ran and passed.** Dropping the broken cell
would give a clean two-arm board; interpolating hybrid recall from sequence 31 would give a
trend line. Both would produce a table in which unexamined and examined look identical — not a
wrong number, but a silence that reads as a measurement.

---

## 9. Conclusion

We measured the token at which attention overtakes minGRU: four widths, five seeds per rung, each
arm at the argmin measured at the budget it is spent over. It is 10.71M at width 384, 12.37M at 768
and 13.46M at 1152 — rising, 5 of 5 seeds per step, consecutive intervals disjoint — and then 13.44M
at 1536, which is not a step at all (−0.02M [−0.33, +0.30], 2 of 5). One shared learning rate moves it −0.81M [−1.07, −0.55]; context
512 → 2048 with batch traded to pay for it moves it +3.63M. Quadrupling the budget does not move it
at all — 10.61M at 200M against 10.70M at 50M, 5 of 5 seeds each time. The crossing is a property of
the architectures, the width, the tuning protocol and the input pipeline, not of how long you meant
to train. The 50M endpoint ranking, the number a board usually reports, is flat across the same 4×
width span and shows none of it.

Three things follow for practice. **A screen must report the width and the recipe it ran at**,
because those decide the answer: at 12M tokens the same comparison names attention at width 384 and
minGRU at 1536, and neither result tells its reader a choice was made. **A screen reporting only
held-out loss cannot see a capability collapse**: pure minGRU sits 0.23 nats behind attention on
cross-entropy, a modest loss on any board, while the pure recurrent arms return zero solves in 80
runs of associative recall at sequence ≥ 255. And the one economy this paper supports: **a screen
need not run the full budget to locate the crossing**, because the crossing does not know what the
budget was.

One question this paper opened has closed since it was drafted. E36 measured the hybrids at
sequence 255: they retain attention's recall in full — 15 of 15, 14 of 15 and 13 of 15 against
attention's 11, at least as good on every arm — so §5's cross-entropy result is not bought with a
capability. Sequence 511 remains unmeasured for every hybrid.

Two things are still open. The hybrid boards ran at 1×
base, which is nobody's argmin: a hybrid wants 4× base, the recurrent arm's optimum rather than
attention's 8×, so §5's hybrid-versus-attention rows carry the confound §4 identifies. That board
re-run with each arm at its own argmin is the clean experiment; it is outstanding and not queued.
And every argmin here was measured at 10M tokens and spent at 50M and 200M: invariance to width is
verified, invariance to horizon is not measured, and it is load-bearing.

The width result is falsified by a rung whose crossing falls outside the monotone sequence by more
than its interval — a fifth width, or these same four re-run with each arm's argmin re-measured at
the horizon it is spent at.

---

## References

- [1707.05589] G. Melis, C. Dyer, P. Blunsom, "On the State of the Art of Evaluation in Neural Language Models," 2017.
- [1709.06560] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger, "Deep Reinforcement Learning that Matters," 2017.
- [1811.03600] C. J. Shallue, J. Lee, J. Antognini, J. Sohl-Dickstein, R. Frostig, G. E. Dahl, "Measuring the Effects of Data Parallelism on Neural Network Training," 2018.
- [1812.06162] S. McCandlish, J. Kaplan, D. Amodei, OpenAI Dota Team, "An Empirical Model of Large-Batch Training," 2018.
- [1909.03004] J. Dodge, S. Gururangan, D. Card, R. Schwartz, N. A. Smith, "Show Your Work: Improved Reporting of Experimental Results," 2019.
- [2001.08361] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, D. Amodei, "Scaling Laws for Neural Language Models," 2020.
- [2003.08505] K. Musgrave, S. Belongie, S.-N. Lim, "A Metric Learning Reality Check," 2020.
- [2009.06489] S. Hooker, "The Hardware Lottery," 2020.
- [2203.15556] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, J. W. Rae, O. Vinyals, L. Sifre, "Training Compute-Optimal Large Language Models," 2022.
- [2312.04927] Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré, "Zoology," 2023.
- [2312.06635] Yang, Wang, Shen, Panda, Kim, "Gated Linear Attention," 2023.
- [2401.12973] Akyürek, Wang, Kim, Andreas, "In-Context Language Learning," 2024.
- [2402.18668] Arora et al., "Based," 2024.
- [2405.04517] Beck et al., "xLSTM," 2024.
- [2406.07887] R. Waleffe, W. Byeon, D. Riach, B. Norick, V. Korthikanti, T. Dao, A. Gu, A. Hatamizadeh, S. Singh, D. Narayanan, G. Kulshreshtha, V. Singh, J. Casper, J. Kautz, M. Shoeybi, B. Catanzaro, "An Empirical Study of Mamba-based Language Models," 2024.
- [2407.05483] Arora et al., "Just read twice," 2024.
- [2409.19913] J. Bjorck, A. Benhaim, V. Chaudhary, F. Wei, X. Song, "Scaling Optimal LR Across Token Horizons," 2024.
- [2410.01201] L. Feng, F. Tung, M. O. Ahmed, Y. Bengio, H. Hajimirsadegh, "Were RNNs All We Needed?," 2024.
- [2410.03810] Ren, Li, Liu, "Exploring the Limitations of Mamba in COPY and CoT Reasoning," 2024.
- [2410.11840] L. Choshen, Y. Zhang, J. Andreas, "A Hitchhiker's Guide to Scaling Law Estimation," 2024.
- [2410.21676] H. Zhang, D. Morwani, N. Vyas, J. Wu, D. Zou, U. Ghai, D. Foster, S. Kakade, "How Does Critical Batch Size Scale in Pre-training?," 2024.
- [2507.07101] M. Marek, S. Lotfi, A. Somasundaram, A. G. Wilson, M. Goldblum, "Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful," 2025.
- [2607.07953] Cerruti, Rieder, Rowlands, Jin, Schlag, "Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing," 2026.
