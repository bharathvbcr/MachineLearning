# 14 — Mixture-of-Experts and Adaptive Computation

So far every token does the *same* amount of work through the *same* weights. This file covers two
ways to break that: **MoE** (many expert FFNs, few active per token — your `nanolab` MoE with
Switch load-balancing) and **adaptive compute** (spend *more* steps on hard tokens — your APRDH
experiment's marginal-gain controller). Both answer: *spend compute where it pays.*

---

## 14.1 The motivation: capacity vs compute

From file 10, the FFN holds most of the parameters and does most of the per-token compute. To make
a model "know more," you widen the FFN — but that makes *every* token more expensive. **MoE breaks
the link between capacity (total params) and compute (params used per token):**

```
  Dense FFN:   every token → the ONE big FFN            (all params active, expensive)
  MoE FFN:     every token → a router picks k of N FFNs  (huge total params, only k active)
```

DiffusionGemma is 26B total / 4B active — 6× the knowledge at 1× the per-token cost. That's the MoE
bargain: **capacity scales with N experts, compute scales with k active.**

Capacity (params) vs compute (active per token), drawn — the MoE bargain is the divergence:

```
                  CAPACITY (total params)        COMPUTE (active/token)      MEMORY (VRAM)
  dense FFN       ████                            ████                        ████
  MoE 8-expert    ████████████████████████████    ████ (top-1)                ████████████████████████████
                  └ 8× capacity ─────────────┘    └ same! ┘                   └ 8× — all experts resident →
                                                                                OOM at bs16 on your 8 GB GPU
```

The first two bars are the win (capacity up, compute flat); the third is the catch that bit you.

---

## 14.2 How MoE routing works (your `nanolab/model.py::MoE`)

```
  for each token's vector x:
     scores = router(x)                  # a Linear: [d] → [n_experts]
     probs  = softmax(scores)
     pick top-k experts (k=1 for Switch, k=2 common)
     out    = Σ  probs[e] · expert_e(x)  # weighted sum of the chosen experts' SwiGLU outputs
```

Your implementation (model.py:72–106): `moe_experts` SwiGLU experts, a router, top-k selection,
and the chosen experts process only their assigned tokens (`out[sel] += w * expert(xf[sel])`).
Each expert is a full SwiGLU FFN (file 03).

```
                       ┌─ router (softmax over experts) ─┐
        token x ───────┤                                  ├──▶ pick top-k
                       └──────────────────────────────────┘        │
              ┌──────────┬──────────┬──────────┬──────────┐         │
           expert0    expert1    expert2    expert3   ...           │  only the chosen
           (SwiGLU)   (SwiGLU)   (SwiGLU)   (SwiGLU)                 │  experts run
              └──────────┴────┬─────┴──────────┴──────────┘◀─────────┘
                              ▼
                       weighted sum → out
```

---

## 14.3 The load-balancing problem (and your Switch aux loss)

Left alone, the router **collapses**: it learns to send everything to its 2–3 favorite experts,
the rest never train, and you've wasted most of the capacity. The fix is an **auxiliary loss** that
penalizes imbalance and pushes usage toward uniform. Your code (model.py:106):

```
  aux = n_experts × Σ_e ( frac_to_e × mean_prob_e )
```

- `frac_to_e` = fraction of tokens actually routed to expert e (hard assignment).
- `mean_prob_e` = average router probability for expert e (soft).
- The product is **minimized when both are uniform** (1/N each), so minimizing `aux` spreads load.

This is the **Switch Transformer** load-balancing loss. It's added to the main loss with a small
weight (`cfg.moe_aux_weight`), surfaced as `model._moe_aux` so your training log can watch it. If
that number stays high, the router is collapsing — another "log it to see it" lever (file 16).

---

## 14.4 The cost MoE doesn't escape: memory

MoE saves *compute* but **not memory** — all N experts' weights sit in VRAM even though only k run.
This bit you directly in the GPU sweep:

> **`moe OOMs at bs16`** (8 dense experts) — use bs ≤ 8.

8 experts = 8× the FFN parameters resident. On an 8 GB laptop that's an instant OOM at a normal
batch. This is why MoE is a *datacenter* technique (where you also need **expert parallelism** —
file 06 — to spread experts across GPUs, with all-to-all communication between them). On a single
small GPU, MoE's capacity win is mostly theoretical. Good to have measured it yourself.

> **Why MoE + diffusion is a noted pairing (guide §2.1):** a diffusion model runs *many* forward
> passes per generated sequence (iterative denoising, file 15), so cheap-per-pass high-capacity
> compute is especially valuable — MoE gives exactly that. DiffusionGemma is an MoE for this reason.

---

## 14.5 Adaptive computation — APRDH's marginal-gain controller

MoE varies *which* weights a token uses. **Adaptive computation** varies *how much* — easy tokens
("the") get a quick pass, hard tokens (a surprising word) get more. Your **APRDH** experiment
(`train_toy_adaptive.py`) is a from-scratch take on this:

- **One weight-shared block applied recurrently 2–5 times** — depth via *reuse*, not more params
  (cf. file 03: naive recursive sharing hurt BPB 2.851; APRDH is the *routed*, careful version).
- **A marginal-gain compute controller** — at each recurrent pass, estimate "would another pass
  reduce loss enough to justify its cost?" and **halt** when marginal gain drops below a budget
  penalty. This is "Adaptive Computation Time" (ACT) / PonderNet-style halting.
- **Gumbel-routed gates** — the halt/route decisions are discrete (do another pass or not), which
  isn't differentiable. The **Gumbel-softmax** trick makes a discrete choice differentiable by
  adding calibrated noise and annealing from soft (continuous, trainable) to hard (discrete,
  deployable) — your **soft→hard Gumbel schedule with a tau floor**.

```
  token ──▶ block ──▶ controller: "gain ≥ cost?" ──no──▶ halt, emit
                ▲                          │ yes
                └──────── another pass ◀───┘   (2–5 passes, per token)
```

Plus the rest of the APRDH kitchen sink (file 08): projected GDN dual scan, MLA latent routing,
span-mixer patching, an n-gram "engram" hash memory, fast-weight adapters. It's the most
research-y thread — exploring *learned, input-dependent compute* end to end.

---

## 14.6 The unifying idea

```
  Dense:            fixed weights, fixed compute, every token       (simple, your champion)
  MoE:              VARY THE WEIGHTS per token (sparse experts)     (capacity ≫ compute)
  Adaptive compute: VARY THE DEPTH per token (halting)              (compute matches difficulty)
  APRDH:            both, plus learned routing/memory               (the experimental extreme)
```

All four are points on one axis: *how much does the model adapt its computation to the input?* The
champion sits at the simple, robust end — which, given your ablation results (recursive sharing
hurt, MoE OOMs on the laptop), was the right call **for this budget and hardware.** The adaptive
ideas are where you'd reach if the budget and hardware grew.

**Next:** [`15-diffusion-language-models.md`](15-diffusion-language-models.md) — the most different
idea in the project: dropping autoregression entirely.
