# 23 — Evaluation Methodology: Making Comparisons Mean Something

Every result in file 08 is a *comparison* — mixer A vs B, gated-attn vs not, chunked vs sequential.
A comparison is only worth as much as its methodology. This file is how you measured: held-out eval,
seeds, the ablation ladder, calibration, and the traps that make a number lie. This is what turns
"I ran some training" into "I have evidence."

---

## 23.1 What you actually measure, and how (your `train.py::evaluate`)

```python
  @torch.no_grad()
  def evaluate(model, batcher, cfg, ...):
      model.eval()                                  # disable dropout etc.
      losses = zeros(cfg.eval_iters)                # average over MANY batches, not one
      for i in range(cfg.eval_iters):
          x, y = batcher.batch()
          loss = model(x, y)
          losses[i] = loss
      return losses.mean()                          # the reported val number
```

Two non-negotiables baked in:
- **`model.eval()` + `no_grad()`** — eval must not train (no gradient, no dropout). Forgetting this
  contaminates the measurement and wastes memory.
- **Average over `eval_iters` batches** — a single batch is noisy; one lucky/unlucky batch can swing
  the loss. You average ~20–40 batches so the number is stable. (For Schedule-Free, you even switch
  to the averaged iterate first, then back — file 05.)

You evaluate **both** train and val every interval (`evaluate` on `train_batcher` *and*
`val_batcher`, train.py:178–179) so you can watch the **gap** — the overfitting signal (file 01).

---

## 23.2 Loss → the metric you report (BPC, BPB)

From your `train.py:184`: `extra["bpc"] = val / math.log(2)` — bits-per-character, logged when the
tokenizer is char-level (text8/enwik8). The conversions (files 01, 11):

```
  bits_per_token = loss_nats / ln(2)
  BPC  = bits_per_token              (when 1 token = 1 char)
  BPB  = bits_per_token × tokens/bytes   (tokenizer-agnostic — the competition metric)
  perplexity = exp(loss)
```

**Report the metric that matches the question.** Comparing two char models? BPC. Comparing across
tokenizers or scoring the competition? BPB. Watching a single run converge? Raw loss or perplexity
is fine. Your diffusion run reported **perplexity** (19.5→8.2) because that's the natural readout
for a generative-quality check; your ablations reported **calibrated BPB** because that's the
leaderboard metric.

---

## 23.3 The cardinal rule: change ONE variable

Every clean result in the project comes from holding everything fixed except the thing under test:

```
  mixer bake-off:  SAME seed, optimizer, schedule, batch, ctx, tokens — ONLY cfg.mixer changes
  arch ladder:     SAME everything — ONLY the architecture flag (gated_attn / value_resid) changes
  kernel verify:   SAME inputs — chunked vs sequential must match to 1e-5
```

If you change two things and the number moves, you don't know which one did it. nanolab's entire
design ("one guide lever = one flag, every run changes one variable") exists to *enforce* this.
It's the difference between an experiment and an anecdote.

---

## 23.4 Seeds — separating signal from noise

Random init + data order means two runs of the *same* config give slightly different numbers. So a
small gap between two configs might be **noise, not signal**. Your defense (file 16):

- **Short/mid stages: 1 seed** (cheap, for ranking many candidates roughly).
- **Long stage: ≥2 seeds** (1337 + 42) — if config A beats B on *both* seeds, you believe it.

This is why the champion result is quoted as "long stage, 3000 steps, **seeds 1337+42**." The gap
between champion (1.985) and runner-up (1.987) is *tiny* — 0.002 BPB — and only trustworthy because
it held across seeds. Without the second seed you couldn't claim it. (Honestly, 0.002 is within
noise for many setups; the value-residual-alone vs combined call is close, which the notes
acknowledge — the bigger, trustworthy gap is vs gated-attn-alone at 2.089.)

---

## 23.5 The staged ladder — spend compute where it discriminates

You don't run every candidate to convergence — that's wasteful. The **short → mid → long funnel**:

```
  SHORT (cheap, many candidates):  run 20+ variants briefly, keep the top handful
  MID   (fewer, longer):           the survivors run longer, narrow to 2–3
  LONG  (few, multi-seed):         finalists run to 3000 steps × 2 seeds → champion
```

The funnel, drawn (width ∝ number of candidates, height ∝ compute per candidate):

```
  SHORT   ████████████████████  ~20 variants × cheap     → keep top ~6
  MID        ████████████        ~6 variants × longer     → keep top ~3
  LONG          ██████           ~3 finalists × 3000 steps × 2 seeds → champion.json
                 ▼
            gated attn + value residual, BPB 1.985
  └ most compute is SPENT on the few survivors, not wasted on obvious losers (successive halving)
```

This concentrates expensive long runs on candidates that already look good, and discards obvious
losers cheaply. It's the same logic as a hyperparameter search with successive halving. Your
`run_ablation_3070ti.py` conductor automates the funnel and emits `champion.json`.

### Why ≥2 seeds — a picture of "is this gap real?"

```
  champion vs value-resid-alone:   1.985 vs 1.987   gap 0.002
     seed scatter (±~0.01):  ●━━━━━━●          ← the two error bars OVERLAP → within noise
                              └ can't call a winner on one seed

  champion vs gating-alone:        1.985 vs 2.089   gap 0.104
     seed scatter (±~0.01):  ●━━●        ......        ●━━●   ← bars DON'T overlap → real
```

A gap smaller than the seed-to-seed scatter isn't a result — it's noise. This is exactly why the
0.002 BPB champion margin is reported honestly as "within noise," while the 0.104 margin over
gating-alone is the trustworthy claim (file 08).

---

## 23.6 Calibration — a fair final number

"Calibrated BPB" (file 07) means the logits were temperature/softcap-adjusted before scoring. Why
it's part of *evaluation*, not cheating: every candidate gets the same calibration sweep, so it's a
fair, consistent readout of each model's true predictive quality — and it matches what the
submission pipeline actually ships. Calibrating one model and not another would be the unfair move;
calibrating all of them identically is just measuring them at their best.

---

## 23.7 The traps that make a number lie

A checklist drawn from your actual debugging (file 16):

| Trap | Why it lies | Guard |
|---|---|---|
| **Val leakage** | model memorized the test → fake-low loss | strict split; `.bin` check before data import |
| **Loss = 0** | trivial target / identity shortcut | suspect too-good numbers (diffusion bug) |
| **Single batch eval** | noise read as signal | average `eval_iters` batches |
| **Single seed** | random gap read as real | ≥2 seeds at the decision stage |
| **Changed 2 variables** | can't attribute the change | one lever per run |
| **CPU-only test** | misses precision/device bugs | test on real device + dtype |
| **Comparing across tokenizers with loss** | not comparable | use BPB (tokenizer-agnostic) |
| **Local BPB as H100 predictor** | different scaling regime | treat local as relative-only (file 17) |

---

## 23.8 Takeaway

Good evaluation is most of good ML. The model and optimizer get the headlines, but the reason you
*know* gated-attn+value-residual is the champion, that the crossover is at ~7M tokens, and that the
chunked kernel is correct — is methodology: one variable per run, averaged over batches, repeated
across seeds, staged to save compute, scored with the right metric, and distrustful of numbers that
look too good. Every result in file 08 is only as strong as this file makes it.

**Next:** [`24-build-it-yourself-exercises.md`](24-build-it-yourself-exercises.md) — stop reading,
start running.
