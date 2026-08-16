# 06 — Numerical Precision and GPU Utilization

You asked specifically about "fp16 or something precision and more, even minute things." This
file is that — the bit-level reality of floating point, why **bf16** is the laptop default, what
**fp8/fp4** buy on newer hardware, and the systems lessons from your **GPU-maximization run that
went from 14% utilization to 25.5% MFU** (a 2.4× throughput gain).

---

## 6.1 What a floating-point number actually is

A float stores a number in three parts, like scientific notation in binary:

```
   ± 1.xxxxxxxx × 2^(eeeee)
   │     │            │
  sign  mantissa    exponent
        (precision)  (range)
```

- **Exponent bits** = **dynamic range** — how big and how small a number you can represent.
- **Mantissa bits** = **precision** — how many significant digits / how fine the resolution.

That split is the *entire* story of fp16 vs bf16. Here are the formats:

```
            sign  exponent  mantissa   total   range          precision
  FP32       1       8         23       32      huge           ~7 decimal digits   ← "full precision"
  TF32       1       8         10       19*     same as FP32   ~3 digits           (Ampere tensor cores)
  FP16       1       5         10       16      NARROW (±65k)  ~3 digits           ← the classic trap
  BF16       1       8          7       16      SAME as FP32   ~2–3 digits         ← your default
  FP8 (E4M3) 1       4          3        8      small          very coarse         (Hopper+)
  FP8 (E5M2) 1       5          2        8      wider          coarser             (Hopper+)
  FP4 (E2M1) 1       2          1        4      tiny           almost none         (Blackwell only)
```

(*TF32 is a 19-bit internal tensor-core mode that reads/writes fp32.)

The bit-split, drawn (`S`=sign, `E`=exponent/range, `M`=mantissa/precision):

```
FP32  S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM   8 exp + 23 mant   full range, ~7 digits
BF16  S EEEEEEEE MMMMMMM                   8 exp + 7  mant   SAME RANGE as fp32, ~2-3 digits ★ your default
FP16  S EEEEE MMMMMMMMMM                   5 exp + 10 mant   ←overflows at 65504! more digits, no range
FP8   S EEEE MMM                           4 exp + 3  mant   Hopper+ ; coarse
FP4   S EE M                               2 exp + 1  mant   Blackwell only ; almost no precision
       │ └ range ─┘ └─ precision ─┘
       └ note BF16 and FP32 have the SAME exponent width → BF16 never overflows where FP16 does.
```

---

## 6.2 FP16 vs BF16 — the single most important precision lesson

Both are 16 bits. They split those bits differently, and it changes everything:

- **FP16** spends bits on mantissa (precision) but has only a **5-bit exponent → max ~65,504**.
  In training, gradients and some activations easily exceed that → **overflow to `inf` → NaN →
  dead run.** FP16 training *requires* "loss scaling" (multiply the loss up before backward,
  divide after) to keep gradients in range. Fiddly and fragile.
- **BF16** keeps the **full 8-bit exponent of FP32** (same range, never overflows) and sacrifices
  mantissa instead (only 7 bits, ~2–3 digits). For deep learning, **range matters far more than
  precision** — a slightly noisy gradient is fine, an `inf` gradient is fatal. So bf16 needs **no
  loss scaling** and "just works."

> **This is why every nanolab run uses `torch.autocast(dtype=torch.bfloat16)` and not fp16.**
> Your RTX 3070 Ti is Ampere (SM 8.6), which has native bf16 tensor cores. bf16 is the default
> for all your runs; fp16's only advantage (1 more bit of mantissa) isn't worth its fragility.

### Mixed precision = the best of both

You don't pick one globally. **Autocast** runs the *matmuls* in bf16 (fast, on tensor cores) but
keeps the *master weights* and *reductions* (like the optimizer state and softmax accumulation)
in fp32. This is "Automatic Mixed Precision (AMP)." The model is ~2× faster and just as stable.

### Where you got bitten by precision (file 04 callback)

The **chunk-parallel SSD/GDN scans must run in fp32**, with autocast *disabled* inside, and the
backward must cast `grad_y` to fp32. Why: a recurrence *accumulates* over the sequence, and small
bf16 rounding errors compound step after step into real drift. A one-off matmul tolerates bf16; a
long accumulation does not. A CPU-only test (CPU was already fp32) didn't catch it — the GPU run
did. **Lesson: accumulation wants more precision than pointwise ops.**

---

## 6.3 FP8 and FP4 — the scale-up precision lever

Lower precision isn't just a memory saver — it **doubles or quadruples tensor-core throughput**,
because the hardware can pack more low-precision multiply-adds per cycle.

| Format | Hardware | Payoff |
|---|---|---|
| **BF16** | Ampere+ (your 3070 Ti) | the laptop default |
| **FP8 / MXFP8** | Hopper (H100), Blackwell | ~2× over FP16 at matched accuracy. **DeepSeek trained a 671B model in FP8.** |
| **NVFP4 / FP4** | **Blackwell only** (B200) | ~2–3× throughput, ½ the memory vs FP8; matches FP8 accuracy up to ~120B |

Your Ampere card **tops out at bf16** — no native fp8/fp4. This is the concrete reason the
competition's **8×H100** target (Hopper, fp8-capable) can train so much more in 10 minutes than
your laptop, and why "rent Blackwell" is the real scale-up move (it unlocks FP4 + FlashAttention-4).

---

## 6.4 MFU — the one number that tells you if the GPU is working

**MFU = Model FLOPs Utilization** = (FLOPs your model actually needs) ÷ (FLOPs the GPU could
theoretically do in that time). It answers: *"What fraction of my expensive tensor-core hardware
is doing useful model math, versus sitting idle?"*

```
  naive run:      20–40% MFU   ← tensor cores stalled most of the time
  well-tuned:     70%+ MFU     (big models, datacenter GPUs)
  your realistic ceiling: ~21–25% MFU for 124M on laptop Ampere
                  (small d=768 matmuls don't saturate the cores; no FP8)
```

Tensor cores stall for three reasons, each with its own fix (guide §7):

```
  ┌─ waiting on MEMORY (HBM)        → fused / IO-aware kernels   ← your single-GPU lever
  ├─ waiting on COMMUNICATION       → compute-comm overlap       ← multi-GPU only (N/A on laptop)
  └─ idle low-PRECISION paths       → FP8 / FP4                  ← Hopper/Blackwell only
```

On a single 8 GB laptop, **memory is the bottleneck**, every time.

---

## 6.5 FlashAttention — the canonical memory fix

Naive attention writes the full **T×T** score matrix to slow GPU memory (HBM), then reads it back
for softmax, then again for the value-multiply. That round-tripping — not the math — is the
bottleneck. **FlashAttention** keeps the computation **tiled in fast on-chip SRAM** and never
materializes the full matrix in HBM. Same result, far less memory traffic.

```
  FA-2  → what your Ampere card runs. Big win over naive; ~35% util on H100 (underuses Hopper)
  FA-3  → Hopper-only: async tensor cores + FP8 → 1.5–2× over FA-2, ~75% H100 util
  FA-4  → Blackwell (early 2026): softmax became the bottleneck, redesigned around it
```

Your sprint trainer targets **FlashAttention-3** for the H100 run; `train_gpt_sprint_core.py`
installs an **SDPA fallback** so the same code runs on your Windows/Ampere card (which has no
FA-3). **Liger Kernel** is the other big single-GPU win — it *fuses* RMSNorm + RoPE + SwiGLU +
cross-entropy into single kernels (~40% more throughput, ~55% less memory).

---

## 6.6 The sysmem-fallback trap — your hardest-won lesson

This is the most valuable systems thing you learned, and it's a Windows-specific landmine:

> When an allocation exceeds the 8 GB VRAM, the Windows/WDDM NVIDIA driver **silently spills to
> host RAM over PCIe (~25× slower)** instead of throwing an out-of-memory error.

The signature is diabolical because it looks like a *busy* GPU, not a broken one:

```
  HEALTHY run:    ~96–100% util,  ~130 W,   sub-second steps
  THRASHING run:  ~100% util,     ~57–60 W, ~18 SECONDS per step   ← looks like a hang
                  └─ the tell: `reserved` memory creeping past 8192 MiB
```

Same 100% utilization, but **half the power draw** (the cores are stalling on PCIe transfers,
not computing) and **20× slower steps**. At ctx1024/124M a with-grad forward+backward without
checkpointing needs ~16 GB — so this hits *fast*. The three fixes:

1. **`fused_ce=True`** — fused cross-entropy never builds the full `B·T·vocab` logits tensor (the
   biggest single allocation). The `fused_ce_chunks` knob is huge: chunks=2 → 1.4K tok/s @ 14 GB
   (thrash); **chunks=16 → 13.3K tok/s @ 4.2 GB (the optimum: fastest *and* leanest).**
2. **`grad_checkpoint=True`** — don't store all 12 layers' activations; recompute them in the
   backward pass. Trades ~30% compute for a big memory cut. **Required at ctx1024 on 8 GB.**
3. **`set_per_process_memory_fraction(0.92)`** — forces a clean `OutOfMemoryError` in milliseconds
   when a config is over budget, instead of silently thrashing for minutes. Baked into
   `sweep_gpu.py --mem_fraction` so over-budget variants fail fast.

---

## 6.7 RESULT — the GPU-maximization run (14% → 25.5% MFU)

You built `bench_gpu.py` / `probe_perf.py` to localize the thrash, then stacked the fixes:

```
  fused linear cross-entropy (Liger-style, chunked, numerically exact)
+ batched Muon (one Newton–Schulz over stacked weights: 496 → 109 ms)
+ gradient checkpointing
+ TF32 + flash-SDPA
+ GPU-resident dataloader (no CPU→GPU stall per batch)
+ the right batch / chunk combo
```

The before/after on the 3070 Ti:

```
                          util    power    tok/s     MFU      VRAM
  baseline (thrashing)    14%     57 W     ~thrash   ~low     >8 GB (spilling)
  gpu_max preset          96–100% ~130 W   13.7K     25.5%    6.1 GB
                          └──────── 2.4× throughput ────────┘
```

The validated peak: **bs32 + fused_ce_chunks=16 + mem_fraction 0.92 = 13.7K tok/s / 25.5% MFU @
6.1 GB** on real data. ~21–25% MFU is the realistic ceiling for a 124M model on laptop Ampere —
the matmuls (d=768) are too small to saturate the tensor cores, and there's no FP8. To go higher
you need bigger matmuls (a larger model) or better hardware (Hopper/Blackwell + FP8).

### The `fused_ce_chunks` sweep — one knob, the whole difference (124M, ctx1024)

The biggest single lever was how finely fused cross-entropy chunks the logits. Too few chunks → a
giant logits tensor → spill to host RAM → thrash. Bar length ∝ throughput:

```
chunks   tok/s                                          peak VRAM   state
   2     1.4K  ###                                        14.0 GB   THRASH (spilling to host RAM)
   4     7.0K  ##############                              8.4 GB   borderline
   8    13.3K  ##########################                 5.6 GB   healthy
  16    13.3K  ##########################                 4.2 GB   ★ OPTIMUM (fastest + leanest)
  32    13.2K  #########################                  4.2 GB   launch overhead creeps in
         └─ a 9.5× throughput swing and a 3.3× memory swing from ONE integer.
```

`chunks=16` frees the VRAM that then lets the **batch** grow (the second lever):

```
batch (chunks=16)   tok/s                                  reserved   MFU
   8                 9.8K  ###################                2.9 GB
  16                10.7K  #####################              4.3 GB
  24                11.2K  ######################             5.6 GB
  32                13.7K  ###########################        6.1 GB   ★ validated peak / 25.5% MFU
  (bs32 @ chunks=8) 9.0K   ##################                 8.1 GB   ← reserved>8GB → fallback, drops!
```

### MFU: the 2.4× climb, as a bar

```
baseline (thrash)   14%  MFU  ###                57 W,  ~18 s/step   ← looks busy, does ~nothing
+ fused_ce          ...        (removes the giant logits allocation)
+ grad_checkpoint   ...        (drops 12 layers of saved activations)
+ batched Muon      ...        (496→109 ms optimizer step)
+ bs32/chunks16    25.5% MFU  ##########################  130 W, sub-second   ← validated peak
                              └─────── 2.4× throughput, same GPU ───────┘
```

**The meta-lesson:** none of this was visible from the loss curve. It was visible from
**utilization, power draw, tok/s, and reserved-memory** logs. *Instrument the system, not just
the model.*

**Next:** [`07-quantization-and-compression.md`](07-quantization-and-compression.md) — squeezing
the trained model into the 16 MB budget.
