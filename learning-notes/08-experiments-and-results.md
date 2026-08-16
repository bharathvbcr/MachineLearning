# 08 — All Results, In One Place

This is the consolidated scoreboard — every experiment in the workspace with its numbers and the
one lesson it teaches. Use it as the quick-reference; the earlier files explain the *why*.

> For suite-level methods, failures, and artifact paths, use the
> [`experiment-notes` lab notebook](../experiment-notes/00-INDEX.md), including replay guidance
> and High/Medium/Low evidence grades. The notebook index also provides the chronological project
> story, decision timeline, and reversals; every suite adds a source-grounded experiment narrative,
> aftermath, detailed observations, and explicit limits. This chapter remains the consolidated
> scoreboard rather than the decision-history layer.

---

## 8.1 The SOTA architecture ladder → champion

Staged short→mid→long ablation (`parameter-golf/logs/`, `run_ablation_3070ti.py`). Local 3070 Ti
proxy: 4 layers, 128 dim, 4 heads / 2 KV, vocab 1024, seq 256.

| Variant (long stage, 3000 steps, seeds 1337+42) | Calibrated BPB |
|---|---|
| **gated attention + value residual** | **1.985 ← CHAMPION** |
| value residual only | 1.987 |
| gated attention only | 2.089 |

BPB as a bar (shorter = better; note the gating-alone cliff and the recursive-sharing disaster):

```
  gated attn + value resid  1.985  ████████████████████  ★ CHAMPION
  value residual only       1.987  ████████████████████  (within noise of champion)
  aux heads (bigram+ve)     2.066  ████████████████████▊
  gated attention ALONE     2.089  █████████████████████  ← gating HURTS without value resid
  recursive weight-share    2.851  █████████████████████████████  ← catastrophic (sharing kills depth)
                                   └ the champion margin over gating-alone is real (0.104 BPB);
                                     the margin over value-resid-alone (0.002) is within seed noise.
```

Sub-ablation findings:
- Aux heads helped: **bigram_dim 48 + ve_dim 24 → BPB 2.066**.
- **XSA** (extra self-attn, last-4 layers) on; turning it **off cost +0.020 BPB**.
- **Recursive weight-sharing** (`recur_2x3`) **hurt: BPB 2.851** — sharing kills specialization.

**Lesson:** stack cheap individually-tested tricks, but **re-test combinations** (gating was bad
alone, good on top of value residual). → files 03, 05.

---

## 8.2 Mixer bake-off — low token budget (2M tokens)

`nanolab/out/bakeoff_<mixer>/`. Identical seed/opt/sched, bs8/ctx512, Muon, cosine, FineWeb-edu.

| Rank | Mixer | Best val loss |
|---|---|---|
| 🥇 | **minGRU** | 5.837 |
| 🥈 | Gated DeltaNet | 5.994 |
| 🥉 | Mamba-2 | 6.040 |
| 4 | Attention | 6.073 |
| 5 | MLA | 6.156 |

As a bar (shorter = lower loss = better; recurrent mixers sweep the top 3):

```
  minGRU         5.837  ██████████████████████          ← best
  Gated DeltaNet 5.994  ████████████████████████▌
  Mamba-2        6.040  █████████████████████████
  Attention      6.073  █████████████████████████▌      ← the "default" loses here
  MLA            6.156  ███████████████████████████      worst (its win is inference memory)
                        └ all 5 within 0.32 loss — the RANKING is the signal, not the values
```

**Lesson:** recurrent/SSM inductive bias **beats attention when data is scarce**. → file 04.

---

## 8.3 Token-budget crossover (8.2M tokens) — the headline

`nanolab/out/scale_<mixer>/`. Same config, run to 8.2M tokens, eval every 200 steps.

The real curves (`A`=attention, `G`=minGRU, `M`=mamba2; full data + gap chart in file 04):

```
val loss
 6.5 | A
 6.3 | G  *
 6.0 |    G  M *
 5.8 |       G  A  M
 5.6 |             A     M  M
 5.5 |             G  *        M
 5.2 |                         *  G   ← 7.4M: A drops below G
 5.1 |                            A
     +------------------------------
tok    0.8 1.6 2.5 3.3 4.1 4.9 5.7 6.6 7.4 8.2   (millions)
```

| Tokens | minGRU | Attention | Mamba-2 | gap (A−G) |
|---|---|---|---|---|
| 0.8M | **6.334** | 6.516 | 6.493 | +0.182 |
| 4.1M | **5.549** | 5.624 | 5.769 | +0.075 |
| 6.6M | **5.353** | 5.358 | 5.560 | +0.005 (tied) |
| 7.4M | 5.249 | **5.239** | 5.469 | −0.010 ← overtakes |
| 8.2M | 5.155 | **5.136** | 5.383 | −0.019 |

**Lesson:** **bias wins early, capacity wins late — the gap shrinks monotonically (0.182→0.005)
then flips between 6.6M and 7.4M tokens.** The single cleanest result in the project. → file 04.

---

## 8.4 GPU optimizer / mixer / FFN sweeps

`nanolab/out/gpu_sweep_{opt,mixer,ffn}.json` (124M, 3070 Ti).

```
OPTIMIZER (bs16/ctx1024):  adamw 11.4K ≥ sgd/muon 10.8K (muon opt 117ms, Newton–Schulz)
                           > lion/sophia/schedulefree/prodigy
FFN (bs16/ctx1024):        swiglu 10.3K > gelu 9.8K > relu2 9.2K;  moe OOMs at bs16
MIXER (bs8/ctx512):        mla 9.3K (fastest) > attention 7.9K > mingru 6.7K
                           >> mamba2 333 / gdn 238 tok/s  (sequential — needs chunk kernels!)
```

The mixer throughput gap, drawn on a **log scale** (the only way the sequential SSMs even show up):

```
  mla        9,300 tok/s  █████████████████████████████  (low-rank KV = smallest fwd)
  attention  7,900 tok/s  ███████████████████████████
  mingru     6,700 tok/s  █████████████████████████      (parallel scan)
  mamba2 seq   333 tok/s  ██                              ← 28× slower!
  gdn    seq   238 tok/s  █                               ← 33× slower (bwd alone 12.5 s)
  ── after chunk-parallel kernels (file 04/21): ──
  mamba2 chk 3,224 tok/s  ████████████████  (9.7× — back in the race)
  gdn    chk   482 tok/s  ███               (2×)
```

FFN bar (bs16/ctx1024):
```
  swiglu  10.3K  ████████████████████████  ★ best quality AND fastest
  gelu     9.8K  ███████████████████████
  relu2    9.2K  █████████████████████▌
  moe       —    ✗ OOM at bs16 (8 dense experts resident; use bs≤8)
```

**Lesson:** Muon costs more *per step* but converges faster → wins on wall-clock; the pure
sequential SSM kernels are 24–33× too slow and *must* be chunk-parallelized. → files 04, 05.

---

## 8.5 Chunk-parallel kernel speedups

Ported from the verified `verify_*.py` references into nanolab `mixers.py`.

| Kernel | Before → After | Unlocks |
|---|---|---|
| **Mamba-2 SSD** (bs8/512) | 333 → **3,224 tok/s (9.7×)** | trainable at ctx1024 |
| **GDN** (bs8/512) | 238 → 482 tok/s (2×), 4.7→2.6 GB | — |
| **GDN** (bs16/ctx1024) | **OOM → 1,100 tok/s @ 4.0 GB** | was impossible |

Both verified exact vs sequential reference (incl. non-divisible-T padding). Scan runs in **fp32**
(the bug a CPU-only test missed). **Lesson:** a custom kernel is worthless until verified against a
brute-force reference; accumulation needs fp32. → files 04, 06.

---

## 8.6 GPU maximization (the systems win)

`nanolab/out/gpu_max/` vs `gpu_baseline/`.

```
  baseline (sysmem thrash):  14% util,  57 W,  ~18 s/step,  >8 GB spilling
  gpu_max preset:            96–100%,  130 W,  13.7K tok/s,  25.5% MFU,  6.1 GB
                             └────────────── 2.4× throughput ──────────────┘
```

Stack: fused linear cross-entropy (chunks=16 optimum) + batched Muon (496→109 ms) + grad
checkpointing + TF32/flash-SDPA + GPU-resident loader + mem_fraction 0.92. **Lesson:** on 8 GB,
memory is the bottleneck; the thrash is invisible in the loss curve but obvious in util/power/
reserved-mem. ~25% MFU is the laptop ceiling for 124M. → file 06.

---

## 8.7 Two experiments the earlier files only touched

### APRDH — adaptive raw-byte recurrent architecture (`train_toy_adaptive.py`)

The most experimental thread: a **byte-level** model built around **one weight-shared block
applied recurrently 2–5 times** (depth via reuse, not more params), combining:

- a **projected Gated DeltaNet** with an exact chunked dual scan as a custom `autograd.Function`,
- **DeepSeek MLA** with top-k latent routing,
- **span-mixer patching** over byte spans (group bytes into learned "patches"),
- a tiny **n-gram hash "engram" memory** (cheap external lookup keyed by byte n-grams),
- **fast-weight adapters**, and
- a **marginal-gain compute controller** (Gumbel-routed) that *decides how much compute to spend
  per token* under a budget penalty — adaptive computation.

Training hardening: gradient-spike skip/rollback, soft→hard Gumbel routing schedules, tau floors.
This is where you explored **adaptive compute** and **learned routing** — ideas beyond the dense
transformer. (Recall from 8.1 that naive recursive weight-sharing hurt BPB; APRDH is the careful,
routed version of that idea.) Benchmarked via `run_toy_benchmarks.py` against RADA/DeltaHybrid.

### Diffusion LM conversion (`nanolab/diffusion.py`) — Phase 3

Converted the **autoregressive** model into a **masked-diffusion** LM (DiffuGPT/LLaDA-style):

```
  AR:         predict next token, left-to-right, causal mask
  Diffusion:  mask random tokens, predict them ALL at once, bidirectional, iterate to denoise
```

Mechanics: anneal the attention mask **causal → bidirectional**, an **absorbing-[MASK] objective**
with 1/t reweighting, complementary masking, and confidence-based **parallel decoding**. Result:
adapted the phase0 checkpoint on TinyStories in ~7 min, **val perplexity 19.5 → 8.2**, generating
coherent stories by iterative denoising.

> **The bug that taught the lesson:** the diffusion loss must target the **clean** tokens, not the
> masked input — otherwise the loss collapses to 0. It was caught *instantly* because the
> loss/grad-norm logged **0** (file 05/06: log everything, and a zero is a scream).

---

## 8.8 The whole project as one flowchart

```
  FineWeb / TinyStories data
        │
        ├─▶ Phase 0: watch it learn (TinyStories, char) ──▶ nanolab cpu_smoke / phase0
        │
        ├─▶ Phase 1: instrumented 124M base run (bf16, log MFU) ──▶ gpu_max (25.5% MFU)
        │
        ├─▶ Phase 2: A/B experiments (one lever per run)
        │       ├─ architecture ladder ──▶ gated attn + value resid (BPB 1.985)  [8.1]
        │       ├─ mixer bake-off ──▶ recurrent wins @ 2M  [8.2]
        │       ├─ scaling crossover ──▶ attention overtakes @ 7M  [8.3]
        │       └─ optimizer/precision sweeps  [8.4–8.6]
        │
        ├─▶ Phase 3 (opt): diffusion conversion ──▶ ppl 19.5→8.2  [8.7]
        │
        └─▶ Submission: QAT int6 + GPTQ + EMA + TTT ──▶ 1.34 MB artifact ──▶ preflight ──▶ 8×H100
```

**The throughline:** *one variable per run, logged from step one, verified against a reference.*
Every result above exists because the harness made the comparison clean and the logs made the
truth visible.

**Next / reference:** [`09-glossary.md`](09-glossary.md).
