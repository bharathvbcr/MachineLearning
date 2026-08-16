# 19 — Muon and Newton–Schulz, Numerically

File 05 explained *why* Muon orthogonalizes the gradient. This file shows *how* — the exact quintic
Newton–Schulz iteration from your `nanolab/optim.py`, with the real coefficients, the SVD intuition,
and a traced example. This is the math behind your champion's optimizer.

---

## 19.1 The claim, restated precisely

A weight **matrix's** momentum `M` has a singular value decomposition `M = U Σ Vᵀ`, where Σ holds
the "singular values" (the strengths of M's independent directions). Real gradient momentum is
**ill-conditioned**: a few singular values are huge, most are tiny. Plain SGD/Adam steps along `M`
therefore push hard on 2–3 directions and barely move the rest.

**Muon replaces `M` with `U Vᵀ`** — the same directions, but *all singular values set to 1*
(semi-orthogonal). The update becomes "balanced": equal-strength movement along every direction M
identified. Empirically this is a much healthier step (~2× over AdamW on the speedrun).

```
  M = U Σ Vᵀ   with Σ = diag(8.0, 2.0, 0.05, 0.01, ...)   ← ill-conditioned, lopsided
  orthogonalize → U Vᵀ   ≈ U diag(1,1,1,1,...) Vᵀ          ← balanced, every direction equal
```

The problem: a real SVD per weight per step is far too slow. Newton–Schulz approximates `U Vᵀ`
using **only matrix multiplies**.

---

## 19.2 The actual iteration (from your optim.py)

Your `zeropower_via_newtonschulz5` (optim.py:26–46):

```python
a, b, c = 3.4445, -4.7750, 2.0315          # the quintic coefficients
X = G.bfloat16()
X = X / (X.norm() + eps)                    # (1) normalize so singular values start in [0,1]
for _ in range(5):                          # (2) ~5 iterations
    A = X @ X.mT                            #     A = X Xᵀ
    B = b*A + c*A@A                         #     a quintic polynomial in X Xᵀ
    X = a*X + B @ X                         #     X ← a·X + (b·A + c·A²)·X
return X                                    # X now ≈ U Vᵀ (orthogonalized)
```

What it's doing: each step applies the polynomial `p(σ) = a·σ + b·σ³ + c·σ⁵` to **every singular
value σ of X simultaneously** (because the iteration is built from `X` and `XXᵀ`, it acts on Σ
without ever computing Σ). The coefficients `(3.4445, −4.7750, 2.0315)` are chosen to **pull any
σ in (0,1] up into a band around 1** — fast.

Here is the *actual* trajectory (computed by running `p` five times on a scalar; `█` marks σ):

```
  target band ≈ [0.7, 1.3] around 1.0 → "approximately orthogonal"
  σ start    step1  step2  step3  step4  step5
  0.05  →    0.17   0.57   1.20   0.94   0.76      tiny values rocket up, then settle
  0.30  →    0.91   0.80   0.97   0.73   1.08
  0.80  →    0.98   0.72   1.09   0.70   1.12      already-large values jitter near 1
            └ note it does NOT converge to exactly 1 — it OSCILLATES into a band near 1.
```

This is the key honesty point: Muon uses an **aggressive, non-monotone** quintic tuned for *speed*,
not clean convergence. After ~5 steps every σ sits in roughly [0.7, 1.3] — "approximately
orthogonal," which is all the optimizer needs. (A gentler coefficient set would converge monotonically
to 1 but take more steps; Muon trades exactness for fewer matmuls.) The whole thing uses only the
matmuls `X @ X.mT` and `B @ X` — no SVD, no eigendecomposition, perfectly GPU-friendly.

Two implementation "minute things" in your code:
- **Normalize first** (`X / X.norm()`): the polynomial only converges if singular values start
  ≤ 1, so the initial scale must be tamed.
- **Transpose trick** (`X = X.mT` when rows > cols, lines 38/45): the iteration is cheaper on the
  "tall" orientation, so it flips the matrix, runs, and flips back. Pure efficiency.

---

## 19.3 Why bf16 is fine here (precision callback)

The iteration runs in **bf16** (`X = G.bfloat16()`) even though file 06 warned that *accumulation*
wants fp32. The difference: Newton–Schulz is **self-correcting** — it's a fixed-point iteration
converging to orthogonal, so small rounding errors each step get washed out by the next step's pull
toward 1. It's not a long accumulation that drifts (like the SSM scan); it's a contraction toward a
stable target. So bf16's range+speed wins with no stability cost. *Knowing which computations
tolerate low precision and which don't is the whole precision skill.*

---

## 19.4 Batched Muon — your 496→109 ms optimization

The clever bit you built (optim.py:31, "batched bmm form"): a transformer has *many weight matrices
of the same shape* (e.g. all 12 layers' `q_proj` are `[768,768]`). Instead of 12 separate
Newton–Schulz calls, **stack them into one 3-D tensor `[12, 768, 768]`** and run the iteration once
— `X @ X.mT` becomes a batched matmul (`bmm`) over the stack. The GPU does all 12 orthogonalizations
in parallel.

```
  naive:   for each of 12 weights: newton_schulz(W)   → 12 kernel launches, 496 ms
  batched: newton_schulz(stack[12,768,768])           → 1 batched call,     109 ms
```

A 4.5× speedup on the optimizer step, from recognizing that same-shape weights can share one
batched call. This is also why Muon's per-step cost (117 ms in the sweep, file 05) is tolerable —
without batching it would be ~5× worse and Muon would lose the wall-clock race to Adam.

---

## 19.5 The Muon + Adam split (why not Muon for everything)

Newton–Schulz orthogonalization only makes sense for **2-D matrices** (it's about singular values
/ directions of a matrix). It's meaningless for:
- **1-D parameters** — norm gains, biases (no matrix structure).
- **Embeddings / the LM head** — these are lookup tables, not transformations; orthogonalizing them
  hurts.

So your optimizer routes 2-D transformation weights → **Muon**, and everything else → **Adam**
(file 05). This is the standard modded-nanoGPT split your champion uses:

```
  2-D weights (q/k/v/o, gate/up/down):  Muon  (matrix_lr 0.025, momentum 0.99)
  1-D + embeddings + head:               Adam  (β = 0.9, 0.95)
```

---

## 19.6 The takeaway

Muon is "Adam's adaptivity, but for the *geometry* of a weight matrix instead of per-element
variance." It costs ~5 matmuls per matrix per step (made cheap by batching), tolerates bf16 because
it self-corrects, applies only to 2-D weights, and converges in fewer steps — which is why it beats
AdamW on wall-clock despite a slower step (file 05's throughput-vs-convergence trade). Every piece
of that sentence is now something you can derive from the 6 lines of `zeropower_via_newtonschulz5`.

**Next:** [`20-rope-positional-math.md`](20-rope-positional-math.md) — the other piece of elegant
matrix math in your model: how rotation encodes position.
