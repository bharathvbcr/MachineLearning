# 22 — Distributed Training and Scaling Systems

Everything you ran was on **one** GPU. The competition trains on **8×H100**. This file covers what
changes when you go multi-GPU: gradient accumulation (the single-GPU stand-in you used), data /
tensor / pipeline / expert parallelism, the communication bottleneck, and why the 8×H100 run lives
on the far side of crossovers your laptop can't reach (file 04, 17).

---

## 22.1 Gradient accumulation — simulating a big batch on one GPU

You met this in file 01. It's the single-GPU substitute for data parallelism:

```
  effective_batch = batch_size × grad_accum × block_size       (× n_gpus when distributed)
  phase1: 24 × 20 × 1024 = ~491,520 tokens per optimizer step   (all on one card)
```

Process `grad_accum` micro-batches, **sum** their gradients, step once. Mathematically identical to
one big batch — you just trade time for memory. The reason big effective batches matter: gradient
*noise* shrinks with batch size, so the optimizer sees a cleaner descent direction (up to a point —
the "critical batch size"). On 8 GPUs you'd get the same effective batch with `grad_accum` 8× lower,
in ~1/8 the wall-clock.

---

## 22.2 The four kinds of parallelism

When one GPU isn't enough — either the data is too slow to get through, or the *model* doesn't fit —
you split the work. Four axes, often combined ("3D parallelism"):

```
  DATA parallel:      same model copy on each GPU, different data shards.
                      Sync: all-reduce the GRADIENTS every step.            ← the common one
  TENSOR parallel:    split each WEIGHT MATRIX across GPUs (e.g. half the
                      attention heads on each). Sync: all-reduce ACTIVATIONS
                      within every layer.                                   ← needs fast interconnect
  PIPELINE parallel:  put different LAYERS on different GPUs; micro-batches
                      flow through like an assembly line. Sync: pass
                      activations between stages.                           ← "bubble" idle time
  EXPERT parallel:    put different MoE EXPERTS on different GPUs (file 14).
                      Sync: all-to-all to route tokens to their expert.     ← MoE only
```

For the competition's 8×H100 + a ~16 MB model, **data parallel** is the natural fit — the model is
tiny (fits on one card easily), so you just want 8× the throughput: 8 copies, 8 data shards,
all-reduce the gradients. Tensor/pipeline parallelism only matter when the *model* doesn't fit, which
isn't the Parameter Golf regime.

---

## 22.3 The all-reduce — what "sync the gradients" means

In data parallelism, after each GPU computes gradients on its own shard, they must be **averaged**
across all GPUs so every copy applies the same update (otherwise the 8 copies drift apart):

```
  GPU0 grad ─┐
  GPU1 grad ─┤
     ...     ├──▶ all-reduce (sum, then ÷ N) ──▶ every GPU gets the averaged gradient ──▶ step
  GPU7 grad ─┘
```

All-reduce moves a *full copy of the gradients* between GPUs every step. For a small model that's
cheap; at billions of params it's the bottleneck — which is file 06's "communication stall."

---

## 22.4 The communication bottleneck (and why it doesn't bite you yet)

File 06's three stall causes: memory, communication, precision. **Communication only exists at
multi-GPU scale**, and it can dominate:

- Gradient all-reduce (data parallel) — every step, every param.
- All-to-all expert routing (MoE) — can hit a ~1:1 compute-to-communication ratio cross-node.
- Pipeline bubbles — GPUs idle waiting for the previous stage.

The fixes (the frameworks you'd reach for, file 06):
- **Overlap compute and communication** — start all-reducing layer L's gradients while still
  computing layer L−1's. DeepSeek's **DualPipe** overlaps forward/backward so all-to-all is nearly
  hidden; **MegaScale**'s overlap alone added +6.2% MFU, +17.6% with everything stacked.
- **Topology-aware placement** — keep chatty GPUs on the same node (NVLink) not across the network.
- **Frameworks:** Megatron-LM, TorchTitan, NeMo implement this so you don't hand-roll it.

On your single 3070 Ti, **none of this applies** — your only enemy is memory (file 06). That's why
the GPU-maximization work was all fused kernels + checkpointing + batch tuning, not communication.

---

## 22.5 Why the 8×H100 run is a different world

Three multiplicative factors separate your laptop from the competition target:

```
  1. 8 GPUs            →  ~8× throughput (data parallel)
  2. H100 vs 3070 Ti   →  ~10–15× faster per GPU (more tensor cores, HBM3)
  3. FP8 (Hopper)      →  ~2× over your bf16 ceiling (file 06)
  ──────────────────────────────────────────────────────────────────
  combined: roughly 100–200× more tokens trained in the same wall-clock
```

The three factors stacking, on a log scale (each step multiplies the last):

```
  3070 Ti (you)        █                          1×     ~13.7K tok/s, bf16, 1 GPU
  + 8 GPUs             ████████                   ~8×
  + H100 per-GPU       ████████████████████████   ~100×
  + FP8                ██████████████████████████ ~150–200×
                       └ this is WHY local BPB ≠ H100 ranking: the H100 run lives past
                         every token-budget crossover (file 04) your laptop can reach.
```

To make it concrete: in 10 minutes your laptop sees ~8M tokens (the scale-run budget); the 8×H100
target sees on the order of **a billion+** — the far side of the ~7M crossover, where attention
has long since overtaken the recurrent mixers.

So in 10 minutes the H100 run sees **hundreds of times more tokens** than your laptop can. Recall
the crossover (file 04): mamba2/mingru win below ~7M tokens, attention wins above. The H100 run
lives *far* past every crossover your laptop can reach — which is the concrete, mechanical reason
**local BPB doesn't predict the H100 ranking** (file 17). Your laptop and the H100 are sampling
*different regions of the scaling curve*.

What still transfers (file 17): correctness, relative mechanics (does fused-CE help? does int6
survive?), and — via μP (file 13) — the hyperparameters. That's the whole justification for the
local scripts being "correctness and relative-comparison harnesses."

---

## 22.6 FlashAttention across the GPU generations (the systems ladder)

Tying file 06 to hardware, since this is where it pays off:

```
  3070 Ti (Ampere):  FlashAttention-2, bf16            ← your runs; SDPA fallback on Windows
  H100 (Hopper):     FlashAttention-3, FP8             ← 1.5–2× over FA-2, ~75% util; the sprint target
  B200 (Blackwell):  FlashAttention-4, FP4             ← softmax-bound redesign; the scale-up frontier
```

Your sprint trainer (`train_gpt_sprint_native.py`) is written for **FA-3** and uses an **SDPA
fallback** so the same code runs on your Ampere card (which has no FA-3) — file 06. That's the
bridge: develop and verify locally on FA-2/SDPA, deploy on FA-3/H100 where the real speed is.

---

## 22.7 Takeaway

Single-GPU training is a *memory* game; multi-GPU adds a *communication* game on top. The
competition's 8×H100 is "just" data-parallel for a tiny model, but the 100–200× token throughput it
buys is what makes it a fundamentally different experiment from your laptop — same code, same recipe
(if μP-transferred), different point on the scaling law. The laptop's job was never to win the
leaderboard; it was to make every choice *correct and verified* before the expensive run, which is
exactly what `preflight_h100.py` enforces (file 07).

**Next:** [`23-evaluation-methodology.md`](23-evaluation-methodology.md) — how to measure all of
this so the comparisons actually mean something.
