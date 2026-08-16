# 13 — Initialization, Regularization, and Training Stability

The "boring" half of training — but it's what decides whether your loss curve descends smoothly or
explodes into NaN at step 200. This file covers **weight initialization, why zero-init works, weight
decay vs dropout, label smoothing, and μP** (the trick that lets your hyperparameters survive when
you scale the model up).

---

## 13.1 Why initialization matters at all

Before training, every weight is a random number. The *scale* of those random numbers decides
whether signals (forward) and gradients (backward) **grow or shrink** as they pass through 12
layers:

- Init too large → activations blow up → `inf`/NaN → dead run.
- Init too small → activations shrink toward 0 → gradients vanish → no learning.

The goal: keep the **variance of activations roughly constant** across layers. Classic schemes:

- **Xavier/Glorot** — scale init by `1/√(fan_in)`; good for tanh/sigmoid nets.
- **Kaiming/He** — scale by `√(2/fan_in)`; accounts for ReLU killing half the signal.
- **GPT-2/nanoGPT init** — normal with std `0.02`, and the *residual* output projections scaled by
  an extra `1/√(2·n_layer)` so that adding 12 layers' contributions to the residual stream doesn't
  make its variance grow with depth. (A residual stream sums L sublayer outputs; without the
  `1/√(2L)` scaling, its magnitude grows like √L.)

---

## 13.2 Zero-init projections (the one your champion uses)

File 03 introduced it; here's *why* it's safe and powerful. Initialize the **last** matrix of each
sublayer (attention's `out_proj`, FFN's `down`) to **exactly zero**:

```
  x = x + out_proj(attention(norm(x)))
        └─ out_proj = 0 at init  →  the whole term is 0  →  x = x  (identity)
```

At step 0, **every block is an identity function** — the network is a clean residual pass-through
from embedding to head. Then gradients gradually teach each block to add a *small, useful*
contribution. Benefits:
- No early-training explosion (every layer starts contributing nothing, ramps up gently).
- Effectively an automatic, learned form of "start shallow, grow deep."
- Combined with QK-norm (bounds attention logits) and RMSNorm (bounds activation scale), this is
  why your runs can use the relatively **high Muon LR of 0.025** without diverging.

It's the cheapest stabilizer in the stack — `cfg.zero_init` is on by default in nanolab.

---

## 13.3 Regularization: weight decay vs dropout

**Regularization** = anything that fights overfitting (the model memorizing training data instead
of generalizing — file 01's train/val gap).

### Weight decay (the one you use)
Pull every weight slightly toward zero each step: `w ← w − lr·wd·w`. Discourages the model from
relying on a few huge weights; favors smoother, more general solutions. Your champion uses
`weight_decay = 0.04` (Muon) / `0.1` (AdamW default). **Decoupled** (the "W" in AdamW) means the
decay is applied separately from the gradient-based update, which behaves better than the original
L2-in-the-loss formulation.

**Grouping (file 05 reminder):** decay 2D matrices only — *never* biases, norm gains, or 1D
params. Decaying a norm's scale just fights the norm's job.

### Dropout
Randomly zero a fraction of activations each forward pass, forcing redundancy. **Standard in GPT-2,
mostly dropped in modern LM pretraining** — at large data scale you're *underfitting*, not
overfitting, so dropout just slows learning. Your runs don't need it; it reappears only for
fine-tuning on small datasets. (Knowing *when not* to use a technique is half of ML.)

### Label smoothing
Instead of a one-hot target (true token = 1.0, all else = 0), use `0.9` for the true token and
spread `0.1` over the rest. Prevents the model from becoming over-confident (driving a logit to
+∞). A small calibration aid — related in spirit to the **softcap/temperature calibration** your
sprint trainer sweeps (file 07).

---

## 13.4 The stability stack, summarized

Your champion's training is stable because several cheap guards compose:

```
  RMSNorm (pre-norm) ── keeps activations at a sane scale every sublayer
  QK-norm            ── stops attention logits from exploding (file 03)
  zero-init proj     ── every block starts as identity (this file)
  residual 1/√(2L)   ── residual stream variance doesn't grow with depth
  warmup (~20 steps) ── LR ramps up while Adam's variance estimate settles (file 05)
  grad clip ≈ 0.3    ── one bad batch can't blow up a step (file 05)
  bf16 (not fp16)    ── no overflow, no loss-scaling fragility (file 06)
  grad-spike rollback── (APRDH) detect a spike, skip/roll back the step
```

Remove any one and you can usually still train; remove two or three and you start seeing the NaN
runs and loss spikes. They're insurance, individually cheap.

---

## 13.5 μP — making hyperparameters transfer across scale

The expensive problem: you tune the LR (0.025) on a *small* model, scale to a 10× wider model, and
the optimal LR is now *different* — so you'd have to re-sweep at the expensive scale. **μP (Maximal
Update Parametrization)** fixes this by scaling init, LR, and output multipliers with width so that
**the optimal hyperparameters stay the same as you widen the model.**

The mechanics nanolab implements (`cfg.mup`, `mup_base_width`):
```
  output_mult  = 1 / width          # shrink output as width grows
  hidden LR    ∝ 1 / width          # per-layer LR scaled by width ratio
  μP init      = init scaled to the base width
  apply_lr     = schedule treated as a multiplier on each group's initial_lr,
                 so per-group ratios survive (the refactor in your notes)
```

The picture: plot validation loss against learning rate for a small and a wide model.

```
  WITHOUT μP — the optimum MOVES, so a laptop sweep misleads at scale:
  loss │   small model          wide model
       │      ╲   ╱                ╲   ╱
       │       ╲ ╱                  ╲ ╱
       │        V                    V
       └────────┼────────────────────┼──────── learning rate
              1e-3                  2e-4        ← different optima → must re-sweep at scale 💸

  WITH μP — the optimum STAYS PUT (init/LR scaled by width), so small→large transfers:
  loss │      ╲   ╱   ╲   ╱
       │       ╲ ╱     ╲ ╱
       │        V       V
       └────────┼───────┼──────────────────── learning rate
              1e-3    1e-3               ← same optimum → tune once on the laptop ✓
```

Why you care: the whole "tune small, then scale to 8×H100" workflow (the competition shape) only
works if your laptop-tuned recipe *transfers*. μP is the principled way to make a 3070 Ti sweep
predict an H100 run — addressing the README's caveat that "local BPB does not predict the H100
ranking." μP narrows that gap for the hyperparameters (architecture/data still need scale-up
validation).

This is "worth it if you'll scale" in the guide's table — and the competition is exactly a
scale-up, so it's on the table.

**Next:** [`14-moe-and-adaptive-compute.md`](14-moe-and-adaptive-compute.md) — spending compute
*selectively*: mixture-of-experts and your APRDH adaptive-computation experiment.
