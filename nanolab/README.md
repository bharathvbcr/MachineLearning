# nanolab — modern small-LM training, one flag at a time

This is the **companion package** referenced throughout
[`../modern-small-lm-training-guide.md`](../modern-small-lm-training-guide.md).
It is a clean, instrumented, single-GPU (RTX 3070 Ti / 8 GB) implementation of
the guide's modern training stack — built so that **every architectural and
training lever the guide discusses is one CLI flag**, and every run changes
exactly one variable (guide §0).

It is deliberately separate from the sprawling competition trainers in
`../parameter-golf/` (`train_gpt_sprint_native.py`, `train_hypercascade.py`, …):
those hold the *fast, fused, submission-grade* paths; nanolab holds the
*readable, pedagogical, one-variable-per-run* path the guide is written around.
The winning defaults here are lifted from those runs (see "Baked-in findings").

```bash
# Phase 0 — watch a small model actually learn (an afternoon)
python -m nanolab.train --preset phase0 --mixer attention

# Phase 1 — instrumented 128M base run, 1–2B tokens (1–2 days)
python -m nanolab.train --preset phase1

# Phase 2 — short optimizer/LR experiments (30–60 min each)
python -m nanolab.train --preset phase2 --optimizer muon --schedule cosine

# CPU sanity check — no GPU needed, ~10s, watch loss fall
python -m nanolab.train --preset cpu_smoke
```

## Why a separate package

The guide's three principles (§0) are *isolate variables, instrument
everything, short runs one variable at a time*. The competition trainers can't
do that cleanly — they fuse dozens of tricks into one 100 KB file. nanolab makes
each trick a flag on a `Config`, so an A/B is literally one changed field.

## File ↔ guide-section map

| File | Guide § | What it implements |
|---|---|---|
| `config.py` | §0, §2.2, §8 | `Config` dataclass; `cpu_smoke`/`phase0`/`phase1`/`phase2` presets; CLI |
| `model.py` | §2 | GPT: RoPE, RMSNorm pre-norm, QK-norm, SwiGLU/ReLU²/GELU, tied embeddings, zero-init projections, μP-ready, grad-checkpointing |
| `mixers.py` | §2.5, §2.1 | Pluggable `--mixer`: `attention` (GQA+gated+value-residual), `mingru`, `mamba2`, `gdn`, `mla` (DeepSeek latent attention) |
| `special_tokens.py` | §2, §11 | Control tokens (`<think>`, `</think>`, `<|answer|>`) utilizing padded vocab space |
| `optim.py` | §4 | `Muon`+AdamW hybrid, `Lion`, `ScheduleFreeAdamW`, **`Sophia`** (clipped diagonal Hessian), **`Prodigy`** (LR-free), SGD |
| `schedules.py` | §5 | constant / cosine / WSD / ReduceLROnPlateau; **LR finder**; schedule-as-multiplier (preserves μP/Muon per-group LR) |
| `data.py` | §3 | char Shakespeare, **text8/enwik8** (bits-per-char), TinyStories/OpenWebText/FineWeb-edu, **curriculum** (sequence-length + compression-ratio difficulty) |
| `prep_fineweb.py` | §3 | Torch-free HF tokenizer (FineWeb parquet / generic streaming) |
| `sft_data.py` | §3, §11 | Phase-2 SFT data prep (GSM8K or synthetic arithmetic to prompt-masked bins) |
| `train.py` | §4.1, §6 | grad-accum loop, bf16 autocast, grad clip, **loss/val/lr/grad-norm/tok-s/MFU** logging, checkpoint+resume |
| `sft.py` | §6, §8 | Phase-2 Supervised Fine-Tuning: think-then-answer reasoning under prompt-masked loss |
| `experiments.py` | §6.3 | optimizer / schedule / mixer **bake-offs**; warmup & overfitting **ablations**; LR finder CLI |
| `reason.py` | §12 | Phase-1/2 inference harness: free-form reasoning + schema-guided JSON decoding |
| `diffusion.py` | §9 | Phase 3: AR → masked-diffusion conversion (annealing, 1/t objective, complementary masking, parallel decoding); **block diffusion / tri-mode** (block-causal attn → `--mode` full/block/selfspec) |
| `star.py` | §8, §24 | Phase-3 Self-Taught Reasoner (STaR) bootstrap loop (sample traces, filter correct, rationalize, SFT) |
| `bench_gpu.py` | §7 | GPU-utilization micro-bench (MFU / mem / fwd-bwd-opt breakdown) |
| `bench_modes.py` | §9 | wall-clock tok/s per decoding mode (ar / diffusion / block / selfspec, cached vs not) |
| `sweep_gpu.py` | §7 | runs the **whole registry** (every mixer / optimizer / FFN) on the GPU, ranked by throughput; VRAM cap so over-budget configs OOM cleanly |
| `probe_perf.py` | §7 | per-phase timing + peak-memory probe (localizes throughput regressions, e.g. sysmem-fallback thrash) |
| `utils.py` | §6.1 | seeding, device/dtype, JSONL + optional wandb logger |

## Tests

A fast, dependency-light regression suite (`tests.py`) locks in the core
invariants — fused-CE numerics + grads, every mixer/optimizer/schedule, μP
scaling, both curricula, MoE routing, the diffusion objective, checkpoint
round-trips. CPU-only, no pytest, exits non-zero on failure:

```bash
python -m nanolab.tests        # 14 checks, <30s
```

Run it after any change before a GPU run.

## The experiment bake-offs (guide §6.3)

Each holds seed/data/tokens fixed and changes one variable, then prints a table
ranked by val loss:

```bash
python -m nanolab.experiments optimizer --preset phase2   # SGD vs AdamW vs Lion vs Schedule-Free vs Muon (§4)
python -m nanolab.experiments schedule  --preset phase2   # constant vs cosine vs WSD vs plateau (§5.3)
python -m nanolab.experiments mixer     --preset phase2   # attention vs mingru vs mamba2 vs gdn (§2.5)
python -m nanolab.experiments warmup    --preset cpu_smoke  # warmup ablation — watch the early spike (§5.2)
python -m nanolab.experiments overfit   --preset cpu_smoke  # overfitting demo — val peels from train (§6.3)
python -m nanolab.experiments lrfind    --preset cpu_smoke  # LR finder — find the knee (§5.1)
```

## Baked-in findings (from this repo's old runs)

The defaults are not generic — they are the **empirically winning** choices from
the architecture ladder in `../parameter-golf/logs/ablations/.../champion.json`:

- **Optimizer:** Muon on 2D hidden weights @ `matrix_lr=0.025`, `momentum=0.99`;
  AdamW (`betas=(0.9,0.95)`) on embeddings/head/norms/scalars (guide §4.4).
- **Attention:** **gated attention + value residual** — the champion combination
  (calibrated BPB 1.985 at the long stage vs 2.089 for gated-attn-only).
- **Grad clip** ~0.3–1.0 global norm; **warmup** short (the long runs used ~20
  steps; the guide's 1–5%-of-steps rule is the default here).
- **QK-norm on**, **β₂=0.95** (not 0.999), **zero-init output projections on**.

See `../parameter-golf/` and the guide §4 for the reasoning.

## What's logged (guide §6.1)

Every run writes `out/<run_name>/metrics.jsonl` (one JSON record per log step)
plus `config.json`. Set `WANDB=1` to also stream to Weights & Biases. The console
shows, from step one:

```
step    100 | loss 3.1421 | lr 6.00e-04 | gnorm 0.84 | 12.3K tok/s | mfu 41.2%
-------- eval @ 250: train 3.05  val 3.07  ppl 21.5 --------
```

## Datasets (guide §3)

| `--dataset` | source | use |
|---|---|---|
| `shakespeare` | tiny-Shakespeare (auto-download, offline fallback) | char or BPE; smoke / Phase 0 |
| `tinystories` | `roneneldan/TinyStories` via HF + GPT-2 BPE | Phase 0 — coherent English from a tiny model |
| `hf` | any HF text dataset (`--hf_dataset`, `--hf_config`) | Phase 1 — FineWeb-edu `sample-10BT`, OpenWebText, Cosmopedia |
| `fineweb_bin` | `--fineweb_pattern '…*.bin'` | reuse this repo's pre-tokenized FineWeb SentencePiece shards |

> **Hard rule (§3):** train/val are tokenized into separate `train.bin`/`val.bin`
> — validation never leaks into training.

### Preparing HF data on Windows (torch + pyarrow gotcha)

Importing **torch and pyarrow/`datasets` in the same process segfaults on
Windows** (clashing native OpenMP/MKL DLLs), and the HF *streaming* parquet
reader segfaults on its own. So tokenize in a dedicated **torch-free** process
first, then train (a second process) just reads the `.bin`s:

```bash
# FineWeb-edu: downloads one parquet shard, reads it non-streaming
python -m nanolab.prep_fineweb --config sample-10BT --max_tokens 50000000
# TinyStories / OpenWebText / any HF text dataset: datasets streaming
python -m nanolab.prep_fineweb --hf_dataset roneneldan/TinyStories --max_tokens 30000000

python -m nanolab.train --preset phase1   # reuses the prepared .bin (never imports pyarrow)
```

`get_dataset` checks for existing `.bin`s **before** importing any native data
stack, so once prepared the trainer is safe. (On Linux/Mac the in-trainer prep
works directly; the standalone step is a Windows requirement.)

## Maximizing GPU utilization (guide §7)

`nanolab/bench_gpu.py` is a tight micro-benchmark for the §7 goal — *keep the
tensor cores busy*. It reports tok/s, MFU, peak memory, and a fwd/bwd/opt phase
breakdown so you can see exactly where time goes:

```bash
python -m nanolab.bench_gpu --batch_size 16 --fused_ce true --grad_checkpoint true
```

On an 8 GB RTX 3070 Ti Laptop, profiling a 124M model found two bottlenecks and
fixed both — the `gpu_max` preset bakes in the result:

| stage | tok/s | MFU | GPU util | power |
|---|---|---|---|---|
| baseline bs12, no fused-CE | thrash (OOM edge) | — | 14% | 57 W |
| clean bs4, per-param Muon | 4.8K | 8.9% | — | — |
| + **batched Muon** (opt 496→109 ms) | 9.0K | 16.7% | — | — |
| + **fused-CE** + **grad-ckpt** bs16 | 11.5K | 21.4% | 96–100% | ~130 W |
| + **bs24** + **VRAM cap** | 13.6K | 25.3% | ~100% | ~130 W |
| + **chunks=16** + **bs32** (`gpu_max`, validated on real data) | **13.7K** | **25.5%** | **~100%** | **~130 W** |

The levers, each a `Config` flag:

1. **Fused linear cross-entropy** (`--fused_ce true`, §7.1) — the `B·T·50304`
   logits tensor (~1.6 GB fp32, doubled by `cross_entropy`'s upcast) was forcing
   OOM-thrash above bs2–3. `FusedLinearCrossEntropy` in `model.py` tiles over
   tokens and computes the input/weight grads *in the forward pass*, so peak
   logit memory is `chunk·vocab`. This is the pure-PyTorch equivalent of
   LinkedIn's Liger fused CE — **no Triton/CUDA toolchain**, so it runs on
   Windows. Verified numerically exact (loss diff 0, grads ≤1e-8).
   The **chunk count is tuned** (`--fused_ce_chunks 16`, default): at bs24/ctx1024
   too few chunks blow up the fp32 intermediates (chunks=2 → 14 GB → thrash, 1.4K
   tok/s), too many add launch overhead (chunks=32 → 13.2K). **16 is the optimum**
   (13.3K tok/s) and uses 1.4 GB less than chunks=8 (4.2 vs 5.6 GB) — that freed
   VRAM is what lets the batch grow to **bs32** (13.7K tok/s, the new peak).
2. **Batched Muon** (`optim.py`) — the per-parameter Newton-Schulz loop launched
   ~100 tiny matmuls/step and ate ~60% of the step. Stacking same-shape weights
   into one `(B,m,n)` tensor and running a **batched** NS collapsed that to ~4
   launches → opt time **496→109 ms**.
3. **Gradient checkpointing** (`--grad_checkpoint true`, §7.4) — trades ~30%
   recompute for memory so the batch can grow, amortizing the optimizer and
   filling the tensor cores. Util climbs to ~100%, power doubles to ~130 W. It is
   **decisively worth it** here: measured, checkpointing+bs32 (13.7K tok/s / 5.1 GB)
   beats *no*-checkpointing at the largest batch that fits — bs6 no-ckpt is only
   12.6K (the 12-layer activations are so heavy that without recompute only a tiny
   batch fits, and its low MFU loses to the recompute it saves; bs8 no-ckpt already
   thrashes at 8.3 GB). So with checkpointing, **bs32 is the throughput peak** at
   ctx1024 (13.7K tok/s / 25.5% MFU / 6.1 GB reserved).
4. **VRAM cap** (`--mem_fraction 0.92`, `set_per_process_memory_fraction`) — on
   8 GB Windows/WDDM, an over-budget step silently spills to host RAM over PCIe
   (~25× slower: 100% util, ~60 W, multi-second steps that *look like a hang*).
   The cap converts that into a clean `OutOfMemoryError`, so you find the real
   batch ceiling in milliseconds. `probe_perf.py` localizes the thrash if it bites.
5. **TF32 + flash-SDPA + bf16** (`--tf32 true`, auto on CUDA) — free Ampere
   throughput (§7.3).
6. **GPU-resident dataloader** (`data.py`) — small datasets live on-device, so
   there is no per-step H2D copy stalling the GPU between steps; the real
   training then **matches or beats** the synthetic-data bench throughput
   (validated: 13.6K tok/s real vs 11.2K synthetic — CUDA-event timing in the
   bench includes sync overhead the steady-state loop amortizes).

```bash
python -m nanolab.train --preset gpu_max     # all of the above, on the 3070 Ti
```

> ~25% MFU is near the realistic ceiling for a 124M model on laptop Ampere —
> `d=768` matmuls are small and memory-bound, and Ampere has no FP8. GPU
> *utilization* is already ~100%; higher MFU needs a bigger `d_model` or
> Blackwell/FP8 (guide §7.3). The point of §7 — no idle tensor cores — is met.

### Sweeping the whole registry on the GPU

`nanolab/sweep_gpu.py` runs *every* mixer / optimizer / FFN through the same
`bench()` and ranks them — one apples-to-apples table per axis. It defaults to
`fused_ce + grad_checkpoint` and caps the allocator (`--mem_fraction 0.92`) so an
over-budget variant raises a clean `OutOfMemoryError` in milliseconds instead of
the **sysmem-fallback thrash** that otherwise looks like a multi-minute hang
(100% util, ~60 W, ~18 s/step — the WDDM driver silently spilling to host RAM
over PCIe; `probe_perf.py` is the tool that localizes it).

```bash
python -m nanolab.sweep_gpu optimizer --batch_size 16 --block_size 1024
python -m nanolab.sweep_gpu mixer     --batch_size 8  --block_size 512   # recurrent refs need a smaller batch
python -m nanolab.sweep_gpu ffn       --batch_size 16 --block_size 1024
```

**Optimizer** (124M attention, bs16/ctx1024) — fwd/bwd are identical, so the
throughput spread is *entirely* the optimizer-step cost:

| optimizer | tok/s | MFU | opt-step | note |
|---|---|---|---|---|
| adamw | 11.4K | 21.1% | 33.6 ms | cheapest competitive step |
| sgd | 10.8K | 20.1% | 16.9 ms | cheapest step, least memory (4.1 GB) |
| **muon** | 10.8K | 20.0% | **116.8 ms** | Newton-Schulz orthogonalization — costly/step but best per-step convergence (the champion default) |
| lion | 10.2K | 19.0% | 35.5 ms | |
| sophia | 10.1K | 18.8% | 38.0 ms | + periodic Hessian refresh |
| schedulefree | 9.8K | 18.2% | 54.1 ms | |
| prodigy | 9.8K | 18.2% | 73.0 ms | LR-free; most optimizer state (5.6 GB) |

> Muon costs ~3.5× AdamW's step but isn't throughput-bound here — it's the
> default because it reaches a lower loss in *fewer* steps (see the bake-offs).

**FFN** (bs16/ctx1024): `swiglu` 10.3K > `gelu` 9.8K > `relu2` 9.2K (ReLU²'s
`4d` hidden is wider than SwiGLU's `~2.67d`, so more FLOPs). `moe` (8 dense
SwiGLU experts ≈ 8× the FFN params, ~450M expert weights) is **memory-prohibitive
on 8 GB**: the expert weights (~1.8 GB fp32) *plus the optimizer's fp32 state for
every one of them* (Muon keeps 1 momentum buffer; AdamW would keep 2 — worse) sit
at ~8 GB **before activations**, so it OOMs at bs16 and thrashes at the edge even
at bs4. The lever that helps is **fewer experts** (`--moe_experts 4/2`) or a
smaller `d_model` — not the optimizer or batch.

**Mixer** (bs8/ctx512 — the small batch lets the slow references run):

| mixer | tok/s | MFU | note |
|---|---|---|---|
| mla | 9.3K | 15.4% | low-rank KV compression → *smallest* forward (147 ms) |
| attention | 7.9K | 13.8% | flash-SDPA baseline |
| mingru | 6.7K | 12.0% | parallel scan — fast |
| mamba2 | ~~333~~ **3.2K** | ~~0.6%~~ | **chunk-parallel SSD** — 9.7× faster (was a sequential ref) |
| gdn | ~~238~~ **1.6K** | ~~0.4%~~ | **vectorized WY chunked delta rule** — 6.7× faster (was a sequential ref) |

The attention-family mixers (`mla`, `attention`) and the parallel-scan `mingru`
are all fast. Both recurrent mixers were rewritten from O(T) sequential references
to **chunk-parallel kernels** (ported from the repo's verified `../parameter-golf`
scans), each kept alongside its `_sequential` reference and pinned by a regression
test that checks output *and* input-gradient (incl. non-chunk-divisible lengths):

- **`mamba2`** → `ssd_chunk_parallel` (from `train_hypercascade.py::_ssd_chunk_parallel`,
  `verify_scan.py` to 1e-5). Genuinely vectorized: two passes of `K=T/chunk`
  steps (≈64 at T=1024/C=32, not 1024) — pass 1 solves each `C×C` chunk in
  parallel as a decay-weighted `C·Bᵀ` attention, pass 2 carries the chunk-final
  state. Plain autograd (2K-deep graph). **bs8/ctx512: 333 → 3.2K tok/s (9.7×)**,
  and now trainable at the full **ctx1024 (bs8, chunk=32: ~2.0K tok/s / 6.0 GB)**
  where the O(T) reference was hopeless. (mamba2's `d_inner=2d` is wider than gdn,
  so use `--mixer_chunk 32` at ctx1024 to keep the `C×C` intra-chunk tiles small.)
- **`gdn`** → `gdn_chunked`, a **fully-vectorized WY / UT-transform** of the gated
  delta rule (no per-timestep loop — the repo's reference kernels all looped, so
  this was derived from scratch and checked against `_sequential`). Within a chunk
  the write vectors solve a unit-lower-triangular system `(I+M)U = R` (one batched
  `solve_triangular`); outputs and the chunk-final state are two decay-weighted
  matmuls; only the chunk carry stays sequential. Pure autograd. **bs8/ctx512:
  238 → 1.6K tok/s (6.7×); ctx1024 1.9K tok/s @ 2.7 GB.** Slightly behind SSD
  (the `(I+M)⁻¹` solve is extra work the SSM doesn't need) but firmly usable.

## Hardware notes (guide §1, §7)

- bf16 autocast + `torch.compile` (attention path) auto-enable on CUDA; both are
  no-ops on CPU so `cpu_smoke` runs anywhere.
- Recurrent mixers (`mamba2`, `gdn`) are **pure-PyTorch FP32 references** — no
  `mamba-ssm` / `flash-linear-attention` CUDA kernels required. They are correct
  but slow; the fast chunk-parallel versions live in
  `../parameter-golf/train_hypercascade.py` (verified to 1e-5 by `verify_scan.py`).
- `--grad_checkpoint true` trades compute for memory when you push `block_size`.
- MFU is reported against a 40 TFLOP/s bf16 peak; override with `PEAK_FLOPS=…`.

## Phase 2 (SFT) & Phase 3 (STaR) — Reasoning & Constrained Decoding

This section of the companion codebase implements teaching the small model to perform step-by-step reasoning ("thinking") and format its outputs using structured schemas:

1. **Schema-Guided Constrained JSON Decoding (`reason.py`)**:
   Enforces structural correctness (valid JSON schema) during generation. By masking out disallowed tokens at each decoding step (e.g. preventing unescaped control chars, forcing digit/comma structure for numbers, adhering to string/bool/enum constraints), the generated text is **always** parseable by `json.loads`. Reuses the KV cache for attention layers (incremental O(T) decode) or falls back to O(T^2) recompute for other mixers.
2. **Supervised Fine-Tuning (SFT) for Reasoning (`sft_data.py`, `sft.py`, `special_tokens.py`)**:
   Repurposes unused padded vocabulary slots (ids `50257..50259`) as reasoning delimiters (`<think>`, `</think>`, `<|answer|>`), eliminating the need to resize embedding tables. Prompt tokens are masked in cross-entropy (`ignore_index=-1`) so the model only trains on the reasoning trace, JSON answer, and `<|endoftext|>` token.
3. **Self-Taught Reasoner (STaR) Bootstrap (`star.py`)**:
   Stitches decoding and SFT into a bootstrap loop. Samples multiple reasoning traces per question, rejects incorrect ones based on the gold answer, utilizes **rationalization** (hinting the correct answer to generate a valid trace when stuck, but training only on the raw prompt), and fine-tunes on the successful trails.

### Commands

Prepare SFT datasets (GSM8K or offline synthetic fallback):
```bash
python3 -m nanolab.sft_data --dataset gsm8k --max_examples 2000
```

Fine-tune a base checkpoint on reasoning traces:
```bash
python3 -m nanolab.sft --base_run run128m_fineweb_2k --run sft128m_gsm8k --steps 800
```

Run inference with reasoning and schema constraints:
```bash
python3 -m nanolab.reason --run sft128m_gsm8k --special --question "Is the sky blue? Give your confidence."
```

Run the Self-Taught Reasoner (STaR) bootstrap loop:
```bash
python3 -m nanolab.star --base_run sft128m_gsm8k --dataset gsm8k --rationalize --rounds 3 --samples 4
```

## Phase 3 — diffusion conversion (guide §9)

`diffusion.py` adapts the trained **AR** checkpoint into a **masked / absorbing-
state diffusion** LM (DiffuGPT / LLaDA-style), on the *same* model:

- **Attention-mask annealing** (`--anneal_steps`) — `GPT.set_causal()` flips
  every layer causal → bidirectional, annealed so the adaptation from the AR
  checkpoint is cheap.
- **Masked objective** — pick a noise level t~U(0,1), mask that fraction to a
  dedicated `[MASK]` id (50257), predict the *clean* tokens there, loss
  reweighted by 1/t (the MDLM/LLaDA NLL bound).
- **LLaDA-2.0 add-ons** — `--complementary` masking (two opposite views per
  sequence) and **confidence-based parallel decoding** at sampling time
  (temperature controls diversity; `[MASK]` is never emitted).
- **Block diffusion / tri-mode** (`--block_len`, Nemotron-Labs-Diffusion-style) —
  train with **block-causal** attention (causal across blocks, bidirectional
  within) via `GPT.set_block_attention(block_len)` instead of annealing to full
  bidirectional. The block length is the single dial spanning all three modes —
  `1` = causal AR, `>=T` = full diffusion, in between = block diffusion — so one
  weight set serves them all, picked at `sample` time with `--mode`:
  - `diffusion` — full parallel denoise.
  - `block` — semi-AR: decode block by block, parallel within a block (throughput).
    `--cached` (default) keeps a **KV cache** of finalized blocks so they are never
    recomputed — identical output, ~10× fewer positions processed at gen_len 192 /
    block_len 16 (`forward_cached` / `forward_hidden_window`; verified exact in
    `tests.py::kv_cached_blockwise_matches_uncached`).
  - `selfspec` — diffusion drafts a block, one causal AR forward verifies it and
    accepts the longest greedy-matching prefix +1. **Lossless** vs greedy AR.
    `--cached` (default) keeps a persistent **causal** committed cache (textbook
    speculative decoding) — identical lossless output, ~7× fewer positions
    processed; verified in `tests.py::kv_cached_selfspec_is_lossless`.

```bash
# adapt an AR checkpoint -> diffusion (≈7 min for phase0 on the 3070 Ti)
python -m nanolab.diffusion train --preset phase0 \
    --init nanolab/out/phase0_tinystories/best.pt --max_steps 1500

# OR block diffusion (semi-AR) — basis for --mode block / selfspec
python -m nanolab.diffusion train --preset phase0 --block_len 32 \
    --init nanolab/out/phase0_tinystories/best.pt --max_steps 1500

# generate: full denoise / semi-AR block / lossless self-speculation
python -m nanolab.diffusion sample --preset phase0 \
    --ckpt nanolab/out/diffusion_phase0/final.pt --prompt "Once upon a time,"
python -m nanolab.diffusion sample --preset phase0 --mode block     --block_len 32 ...
python -m nanolab.diffusion sample --preset phase0 --mode selfspec  --block_len 32 ...
```

Adapting the 30M TinyStories model took masked-denoising val ppl 19.5 → 8.2 and
produces coherent stories by parallel denoising. (Instrumentation lesson, guide
§6: the first attempt logged `loss 0.0000` from step 0 — the masked input was
being used as its own CE target, so the model trivially echoed `[MASK]`; the
flat-zero loss caught it immediately.)

## Not yet wired (guide §10)

Scale-up (FSDP / μP-transfer / FP8 / FA-4 / DualPipe overlap) is documented in
the guide but out of scope for this single-GPU package; the `mup` flag and the
zero-init / QK-norm here are the prerequisites for the μP-transfer workflow.
