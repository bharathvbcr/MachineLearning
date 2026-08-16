# 05 — Optimizers, Learning Rates, and Schedules

The architecture decides *what* the model can represent. The optimizer decides *how fast and how
well* it learns. This file goes from SGD to **Muon** (the one your champion uses), explains the
**Newton–Schulz orthogonalization** trick, and covers learning-rate schedules, warmup, and
clipping — all tied to your champion settings and GPU sweep.

---

## 5.1 The job of an optimizer

Recall the update from file 01: `new_weight = old_weight − lr × gradient`. That's vanilla SGD.
Every fancier optimizer answers one question better: **"given the raw gradient, what update
should I actually apply?"** They differ in how much they *do for you* and how much memory they
cost.

| Optimizer | Core idea | Extra memory | LR sensitivity |
|---|---|---|---|
| **SGD + momentum** | step along a running average of gradients | ~1× params | very high |
| **AdamW** | per-parameter adaptive step from grad mean (m) + variance (v) | ~2× params | moderate |
| **Lion** | update = `sign(momentum)`; memory-light | ~1× | high |
| **Sophia** | Adam + cheap diagonal **Hessian** (curvature) estimate | ~2–3× | moderate |
| **Schedule-Free / Prodigy** | **estimate the LR for you** — no schedule | ~2× | low (auto) |
| **Muon** | **orthogonalize** the gradient momentum for 2D weights | ~1× (2D) | moderate |

nanolab implements **all seven** so you can swap with one flag (`optim.py`). Below: the three
that matter most for understanding.

---

## 5.2 SGD + momentum — feel the fragility

```
v = μ·v + grad           # momentum: a running average of gradients (μ ≈ 0.9)
w = w − lr·v
```

Momentum smooths out noisy gradients and powers through small bumps, like a ball rolling
downhill. The problem: **one global learning rate for every weight.** Some weights need big
steps, some tiny — SGD can't tell. Pick `lr` slightly too high → divergence; slightly too low →
crawls. This is why you internalize SGD then never use it for LMs.

---

## 5.3 AdamW — the workhorse

Adam gives **each weight its own effective step size**, adapted from the history of its gradient:

```
m = β1·m + (1−β1)·grad           # 1st moment: mean of gradient  (β1 = 0.9)
v = β2·v + (1−β2)·grad²          # 2nd moment: variance of gradient (β2 = 0.95 for LMs!)
w = w − lr · m / (sqrt(v) + ε)   # big steps where gradient is steady & small; small where noisy
w = w − lr · wd · w              # AdamW: decoupled weight decay (regularization)
```

Two "minute things" that trip everyone up:

- **β2 = 0.95, not the default 0.999, for LM pretraining.** LM gradients are noisier and
  non-stationary; 0.999 averages over too long a window and reacts too slowly. Your champion uses
  `β=(0.9, 0.95)` for the Adam-handled params.
- **Weight-decay grouping:** decay only the 2D weight matrices (linears, embeddings); **do NOT**
  decay biases, norm gains, or any 1D parameter. Decaying a norm's scale just fights the norm.
  This is the standard nanoGPT param-grouping pattern, and nanolab's `apply_lr` preserves
  per-group ratios so this survives the schedule.

"AdamW" = Adam with that decoupled decay. It's the baseline every result is measured against.

---

## 5.4 Muon — the high-leverage swap (your champion's optimizer)

Muon is the current speed frontier (~2× faster convergence than AdamW on the speedrun). The
idea: a weight **matrix's** gradient, naively applied, is often dominated by a few directions
(it's low-rank-ish), so the update pushes hard along a couple of axes and barely moves the rest.
Muon **orthogonalizes** the momentum matrix first — roughly, it makes the update "spread out"
evenly across all directions (semi-orthogonal), which turns out to be a much healthier step.

```
M = μ·M + grad                       # momentum, as usual
O = orthogonalize(M)  via Newton–Schulz   # ← the Muon trick: make the update semi-orthogonal
w = w − lr · O
```

### Newton–Schulz orthogonalization (the "minute" mechanism)

Computing a true orthogonalization needs an SVD — far too slow per step. Muon instead runs **~5
iterations of a Newton–Schulz polynomial** `X ← aX + b(XXᵀ)X + c(XXᵀ)²X`, which rapidly drives
the singular values of the momentum matrix toward 1 (i.e. toward orthogonal) using **only
matmuls** — perfectly GPU-friendly. It's an approximate, matmul-only "snap to orthogonal."

- Muon applies **only to 2D weight matrices.** Scalars, biases, embeddings, norms, and the LM
  head/SSM params are handled by **Adam** — this is the standard **Muon + Adam split**, and it's
  exactly how your champion is set up.

### The throughput-vs-convergence trade-off you measured

Newton–Schulz costs extra time *per step*. In the GPU optimizer sweep (124M, bs16/ctx1024):

Throughput, as a bar (bs16/ctx1024; longer = faster *per step*, NOT per unit of learning):

```
  adamw         11.4K tok/s  ###########################   opt step  34 ms
  sgd           10.8K tok/s  #########################▌    opt step  ~30 ms
  muon          10.8K tok/s  #########################▌    opt step 117 ms (Newton–Schulz)
  lion           ~9.8K       #######################
  sophia         ~9.2K       #####################▌        (+ periodic Hessian update)
  schedulefree   ~9.0K       #####################
  prodigy        ~8.6K       ####################
```

But "tok/s" is the wrong axis for choosing an optimizer — **steps-to-target-loss** is:

```
  raw throughput:        adamw  ████████████  fastest tok/s
  convergence per step:  muon   ████████████  fewer steps to the same loss
  wall-clock to target:  muon   ███████████▌  WINS — its better step beats its slower step time
```

Muon's step is ~3× slower than Adam's, **but it converges in far fewer steps**, so it wins on
*wall-clock-to-target-loss* despite lower raw tok/s. You also built a **batched Muon** that
stacks same-shape weights into one batched Newton–Schulz, cutting the opt time **496 → 109 ms** —
a real systems optimization on top of the algorithm.

### Your champion's exact optimizer settings (from the ablations)

```
Muon (2D matrices):  matrix_lr = 0.025   ← 0.027 HURT (BPB 2.100 vs 2.093)
                     momentum  = 0.99
                     β2        = 0.95
                     weight_decay = 0.04
Adam (scalars/embeds/head):  β = (0.9, 0.95)
Warmup:              ~20 steps
Grad clip:           ~0.3   (0.35 marginally better)
```

Note how *tight* the LR optimum is: 0.025 → 0.027 is a 8% bump and it measurably hurt. **The LR
is the single most sensitive knob.** That's why the next section exists.

---

## 5.5 Learning-rate schedules

The LR is not constant. A schedule changes it over training:

```
 lr
  │      ╭──────────╮
  │     ╱            ╲___                cosine: warm up, then smoothly decay to ~0
  │    ╱                 ╲__
  │   ╱  warmup              ╲___
  └──┴──────────────────────────── steps
```

- **Warmup (the first ~20–2000 steps):** start the LR near 0 and ramp up. **Why:** at init the
  weights are random, gradients are large and meaningless; a big early step can permanently
  damage the model (and Adam's variance estimate `v` hasn't warmed up yet, so its steps are
  unreliable). Warmup lets the statistics settle. Skipping warmup is a classic divergence cause.
- **Cosine decay:** the default — smooth cosine curve down to ~0. Used in all your bake-off/scale
  runs. (Recall: the long cosine is *why* mamba2 ended the laggard at 8.2M — its convergence
  shape interacted with the decay.)
- **WSD (Warmup-Stable-Decay):** warm up, hold flat, decay only at the end. Lets you *extend* a
  run without recommitting to a final step count.
- **Constant / plateau:** baselines; plateau drops the LR when val loss stops improving.

nanolab has all of these plus an **LR finder** (`schedules.py`) — a short run that sweeps LR
upward and plots loss, so you *find* the optimum instead of guessing. Given how sensitive 0.025
was, this matters.

---

## 5.6 Gradient clipping — the seatbelt

```
total_norm = sqrt(Σ grad²)               # magnitude of the whole gradient
if total_norm > clip:                    # your champion: clip ≈ 0.3
    grad = grad × (clip / total_norm)    # scale the whole thing down, preserving direction
```

One pathological batch can produce a giant gradient that, taken at face value, wrecks the model.
Clipping caps the update magnitude while keeping its direction. The **grad-norm you log every
step** (file 01, line E) is exactly `total_norm` — watching it is how you *see* instability:

- grad-norm spiking → LR too high or a bad batch; the clip is saving you.
- **grad-norm = 0** → something is broken upstream. This is literally how you caught the diffusion
  loss-collapse bug (file 08): the loss/grad-norm logged 0 instantly because the target was wrong.

The APRDH trainer goes further with **gradient-spike skip/rollback** — detect a spike, skip the
step (or roll back), rather than let it through.

---

## 5.7 LR-free optimizers (the "auto-correcting LR" idea)

**Schedule-Free AdamW** and **Prodigy** *estimate the learning rate for you* (Prodigy tracks a
`d/d0` distance-to-solution ratio and scales the LR by it), removing the schedule entirely. Given
how sensitive the LR was in your runs, this is appealing — but in the GPU sweep they were slower
per step than Muon/Adam and didn't beat the hand-tuned recipe, so the champion stayed Muon+Adam.
Worth knowing they exist; not yet worth the trade at this scale.

**Next:** [`06-precision-and-gpu.md`](06-precision-and-gpu.md) — fp16/bf16/fp8/fp4, MFU, fused
kernels, and the GPU-maximization run that 2.4×'d your throughput.
