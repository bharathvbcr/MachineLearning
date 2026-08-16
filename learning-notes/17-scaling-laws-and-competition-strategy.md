# 17 — Scaling Laws and the Competition Strategy

This final file zooms out. Why is Parameter Golf shaped the way it is? What do **scaling laws** say
about trading parameters for data for compute, and how does every technique in files 01–16 map onto
the competition's three constraints? This is where the whole project becomes one coherent strategy.

---

## 17.1 Scaling laws — the empirical backbone of modern ML

The foundational finding (Kaplan 2020, then Chinchilla / Hoffmann 2022): a language model's loss
falls **predictably** as a power law in three quantities:

```
  L(N, D) ≈ E + A/N^α + B/D^β
            │     │        │
       irreducible  params  data (tokens)
       (entropy of  too few  too little
        language)   params   data
```

- **N** = parameters, **D** = training tokens, and compute **C ≈ 6·N·D**.
- The loss is bottlenecked by whichever of N or D is too small. Pour data into a tiny model and it
  plateaus (N-bound); train a huge model on scraps and it plateaus (D-bound).

### Chinchilla's punchline
For a **fixed compute budget**, there's an *optimal* split between N and D — and the rule of thumb
is roughly **~20 tokens per parameter**. GPT-3 was *undertrained* (too big for its data);
Chinchilla showed a smaller model trained on more tokens beats it at equal compute. This is why
modern small models (and yours) train on far more tokens per param than the 2020 norm.

### Where your crossover experiment fits
File 04's mixer crossover (~7M tokens) is a *micro* scaling law: it's the architecture-dependent
version of "data bottleneck vs capacity." Recurrent mixers have stronger priors (effectively a
better constant when D is tiny); attention has more capacity (better α-scaling) so it wins as D
grows. **You measured a scaling-law crossover on a laptop** — the same shape as the field's
N-vs-D curves, but along the architecture axis.

---

## 17.2 Parameter Golf reframes the scaling law

Standard scaling fixes compute and asks for the best N/D split. Parameter Golf fixes a **different**
constraint:

```
  Competition constraints:
    1. artifact (weights + code) must COMPRESS under 16 MB     ← caps effective N (after quantization)
    2. training must finish in < 10 minutes on 8×H100          ← caps compute C (so caps D)
    3. scored by BPB on held-out FineWeb                       ← the loss L, tokenizer-agnostic
```

So it's an **L(N) optimization under a compressed-size budget**: *minimize loss for a model that
fits in 16 MB and trains in 10 minutes.* Architecture is unconstrained — that's why it's an open,
interesting problem rather than "just use a transformer." Every file in these notes is a lever on
one of those three constraints:

| Constraint | The levers (which file) |
|---|---|
| **16 MB artifact** | slim vocab + tied embeddings (11), int6 per-row quant + GPTQ + zstd (07), no wasted params (recursive-sharing tested & rejected) (03) |
| **10 min on 8×H100** | MFU/throughput: FlashAttention-3, fused CE, FP8, Muon convergence (05,06), chunk-parallel kernels (04) |
| **lowest BPB** | gated attn + value residual + aux heads (03), the right mixer for the token budget (04), EMA + TTT + calibration (07) |

---

## 17.3 The "fill the budget" mindset

The champion artifact was **~1.34 MB** — *far* under the 16 MB cap. That's not a victory, it's
**headroom left on the table.** Under a fixed-size budget, the optimal strategy is to **spend every
byte** where it buys the most BPB:

```
  16 MB budget − 1.34 MB used = ~14.6 MB of unspent capacity
       → could fund: more layers, wider d_model, a bigger (but still slim) vocab,
                     more aux-head params, less aggressive quantization (int8 not int6)
       → each is a scaling-law bet: which marginal byte lowers BPB most?
```

The budget, drawn to scale (each `█` ≈ 0.4 MB of the 16 MB cap):

```
  used   [███▌                                       ]  1.34 MB  (champion artifact)
  unspent[    ████████████████████████████████████████] 14.66 MB ← headroom = opportunity
          0 MB                                       16 MB cap
  ── filling the budget is the GAME: every unspent byte is BPB you didn't buy ──
```

How the artifact bytes break down (and the levers on each — files 07, 11):

```
  embeddings  ███████  slim 1024 vocab + tied  ← the big save (vs ~26 MB at 50k vocab)
  transformer ████████ int6 per-row + GPTQ-clip
  code        ██        counts toward 16 MB too (preflight checks this)
  + EMA / calibration / TTT shave BPB at EVAL with ZERO artifact-size cost
```

This is the deepest strategic lesson: the competition is a **constrained allocation problem**.
Quantization (file 07) isn't just compression — it's what lets you *afford more parameters* under
the size cap. A model that quantizes to int6 cleanly can be 25% bigger than one that needs int8 for
the same artifact size. Architecture, quantization, and tokenizer choices all trade against each
other on the same 16 MB ledger.

---

## 17.4 The local-vs-H100 gap (and how μP narrows it)

The README's recurring caveat: **local BPB on a 3070 Ti does not predict the 8×H100 ranking.** Why?

- **Scale changes the answer.** The mixer crossover (file 04) *proves* this: at 2M tokens recurrent
  wins, at 8.2M attention wins. The H100 run trains on vastly more tokens in 10 minutes than the
  laptop ever can — so it lives on the *other side* of crossovers the laptop can't reach. A laptop
  A/B can rank two options backwards relative to scale.
- **What transfers anyway:** *correctness* (does the kernel compute the right thing?), *relative
  mechanics* (does fused CE save memory? does QAT survive int6?), and — with **μP** (file 13) — the
  *hyperparameters*. That's why the local scripts are explicitly "correctness and relative-
  comparison harnesses," and why μP is on the table for the scale-up.
- **The workflow this dictates:** tune the recipe small (Phases 0–2), lock what transfers, use μP
  for HP transfer, and treat the H100 run as the *real* experiment — validated by `preflight_h100.py`
  before paying for it (file 07). Don't trust a laptop BPB as a leaderboard predictor; trust it as
  a "does this work and roughly which direction is better" signal.

---

## 17.5 The whole project as one sentence per layer of abstraction

```
  Atoms:        dot products & matmuls move information between token vectors          (10)
  Mechanism:    attention (lookup) or recurrence (running summary) mixes the sequence  (02,04)
  Architecture: cheap modern tricks (gated attn + value residual) lower loss per param (03)
  Optimization: Muon + the right LR/schedule reach that loss in fewer steps            (05)
  Systems:      bf16 + fused kernels + memory discipline keep the GPU busy (25% MFU)   (06)
  Compression:  QAT int6 + GPTQ + zstd fit the result under 16 MB                      (07)
  Strategy:     scaling laws say spend every byte where it buys the most BPB           (17)
```

Each layer is a file. Together they're the answer to one question: **what is the lowest-BPB model
you can train in 10 minutes that compresses under 16 MB?** — and the honest answer is that you
learned to reason about every byte and every millisecond of that budget.

---

## 17.6 Where to go next (if you keep learning)

- **Read the source you built on:** modded-nanoGPT (the speedrun tricks), the Chinchilla paper, the
  Mamba-2 and Gated DeltaNet papers, the Muon write-up, FlashAttention-2/3.
- **Run the scale-up:** the natural next experiment is to push the crossover further — does a hybrid
  (6 GDN + 2 attn, file 04) beat pure attention at *competition* token counts? Only an H100 run
  answers that.
- **Close the loop on the budget:** take the champion and *spend the unused 14 MB* — try int8 over
  int6, a 2048 vocab, or 2 more layers, and measure which byte was worth most.
- **Phase 3 further:** the diffusion model (file 15) is undertrained; scale it and compare BPB-for-
  BPB against the AR champion.

← Back to [`00-README.md`](00-README.md) · Start of Part II: [`10-math-foundations-and-shapes.md`](10-math-foundations-and-shapes.md)
