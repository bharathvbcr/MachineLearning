# 09 — Glossary (including the minute things)

Fast lookup. Grouped by theme. Bold term → one-line plain-English meaning → where it shows up in
your work.

## Core training
- **Token** — an integer id for a chunk of text (char, subword, or byte). Vocab = how many exist.
- **Embedding** — the lookup table mapping each token id to a vector. Big under a large vocab.
- **Logits** — the model's raw, un-normalized scores over the vocabulary, before softmax.
- **Cross-entropy loss** — `−log P(correct token)`; average surprise per token. Lower = better.
- **Perplexity** — `exp(loss)`; "effective number of tokens the model is choosing between."
- **BPB (bits-per-byte)** — loss converted to bits per byte of raw text; tokenizer-agnostic; the
  Parameter Golf score; also a compression ratio. Champion = **1.985**.
- **Gradient** — derivative of loss w.r.t. each weight; the direction to nudge weights downhill.
- **Backprop** — the chain rule computing all gradients in one backward pass (`loss.backward()`).
- **Autoregressive (AR)** — generate one token at a time, left to right, each conditioned on the
  past. Requires the causal mask.
- **Parameters (N) / Tokens (D) / FLOPs** — model size / data seen / compute (~`6·N·D`).
- **Train vs validation** — learn-on vs held-out-to-measure; never leak val into train.

## Attention
- **Q / K / V** — Query (what I seek) / Key (what I offer) / Value (what I hand over). Attention =
  `softmax(Q·Kᵀ/√d) · V`.
- **Softmax** — turns scores into a probability distribution; sharp, so attention can focus.
- **√d_head scaling** — divides scores so softmax doesn't saturate; small but load-bearing.
- **Causal mask** — sets future scores to −∞ so a token can't see ahead. Removed in diffusion.
- **Multi-head (MHA)** — several parallel attentions, each learning different relations.
- **O(T²)** — attention's quadratic cost in sequence length; the reason for every variant below.
- **Sliding-window attention (SWA)** — each token attends to only the last W tokens → O(T·W);
  long-range recovered across layers like a **CNN receptive field** (reach = L×W).
- **GQA** — query heads share fewer KV heads (champion: 8 Q / 4 KV); shrinks the KV cache.
- **MQA** — all query heads share 1 KV head; smallest cache, slight quality hit.
- **MLA** — compress KV into a small latent (DeepSeek); came last in your 2M bake-off (6.156).
- **KV cache** — stored past K/V vectors during generation; the thing GQA/MLA shrink.
- **RoPE** — rotary position embedding; encodes *relative* position by rotating Q/K. On by default.
- **QK-norm** — RMSNorm on Q and K before attention; stops attention logits from blowing up.

## Architecture
- **Residual / skip connection** — `x = x + sublayer(x)`; makes deep nets trainable.
- **Residual stream** — the running `x` every layer reads from and writes to.
- **Pre-norm** — normalize *before* the sublayer (modern); more stable than post-norm.
- **RMSNorm** — normalize by root-mean-square only (no mean subtract); cheaper than LayerNorm.
- **SwiGLU** — gated FFN `down(SiLU(x·Wg) ⊙ (x·Wu))`; more expressive per param than GELU MLP.
- **ReLU²** — `relu(x)²`; cheap gateless activation the modded-nanoGPT speedrun favors.
- **Gated attention** — multiply attention output by a learned sigmoid gate; champion ingredient.
- **Value residual** — feed early-layer value vectors into deep layers; champion ingredient.
- **Zero-init projections** — init each sublayer's output matrix to 0 → starts as identity; stable.
- **U-Net skips** — connect early layers directly to late layers (shortcut highways).
- **Tied embeddings** — share the input and output embedding table; halves that param cost.
- **Bigram hash / value embeddings** — tiny aux heads injecting n-gram stats (helped: BPB 2.066).
- **Recursive weight-sharing** — reuse one block's weights for fake depth; **hurt (BPB 2.851)**.
- **μP (maximal-update parameterization)** — init/LR scaling so hyperparameters transfer across
  model widths; tune small, apply big.

## Sequence mixers
- **Sequence mixer** — the sub-layer that moves info *between* positions (attention or an SSM).
- **SSM (state-space model)** — maintains a fixed-size state, decays + updates it per token; O(T).
- **Mamba-2 / selective SSM** — SSM whose decay/input/output are *input-dependent*; "selective."
- **ZOH discretization** — turning a continuous linear system into discrete steps (the dt, log_dA).
- **Gated DeltaNet (GDN)** — gated linear attention with a delta rule (overwrite old associations)
  → stronger recall than vanilla SSMs. Beat Mamba-2 in your bake-off.
- **minGRU** — minimal RNN whose gates depend only on input → parallel scan; won the 2M bake-off.
- **Linear attention** — rewrite attention as a recurrence via `(QKᵀ)V = Q(KᵀV)`; O(T).
- **Chunk-parallel scan** — split the sequence into chunks, solve each in parallel, carry boundary
  state. SSD **9.7×**, GDN **2×**; made SSMs trainable on your laptop.
- **Inductive bias** — built-in assumptions (e.g. sequentiality); wins when data is scarce.
- **Crossover** — the token count where attention's capacity overtakes recurrent bias (**~7M**).
- **Hybrid stack** — interleave cheap recurrent layers with a few attention layers (e.g. 6 GDN + 2
  attn); the practical best-of-both.

## Optimizers & schedules
- **SGD + momentum** — step along a running gradient average; one global LR; fragile.
- **AdamW** — per-parameter adaptive step from gradient mean (m) + variance (v) + decoupled decay.
- **β2 = 0.95** — Adam's variance-averaging window for LMs (not the default 0.999); reacts faster.
- **Weight-decay grouping** — decay 2D matrices only, not biases/norms/1D params.
- **Muon** — orthogonalize the gradient-momentum matrix → semi-orthogonal update; ~2× over AdamW.
- **Newton–Schulz** — matmul-only iteration (~5 steps) that snaps a matrix toward orthogonal;
  Muon's core. Batched Muon stacks weights → one NS call (496→109 ms).
- **Muon + Adam split** — Muon for 2D weights, Adam for scalars/embeds/head. Champion setup.
- **Lion / Sophia / Schedule-Free / Prodigy** — sign-based / 2nd-order-lite / LR-free variants.
- **Warmup** — ramp LR up from ~0 over the first steps; lets gradient statistics settle; ~20 steps.
- **Cosine / WSD / plateau** — LR decay shapes. Cosine = default; WSD lets you extend a run.
- **Gradient clipping** — cap the gradient norm (champion ≈ 0.3) so one bad batch can't wreck it.
- **Grad-norm** — the logged gradient magnitude; spikes = instability, **0 = something's broken**.
- **LR finder** — short run sweeping LR to locate the optimum (the LR optimum was tight: 0.025).

## Precision & systems
- **FP32** — 32-bit float, "full precision"; ~7 digits, huge range.
- **FP16** — 16-bit, 5-bit exponent → **overflows (max ~65k)**; needs loss-scaling; fragile.
- **BF16** — 16-bit, **full FP32 range**, less mantissa; no overflow, no loss-scaling. **Your
  default.**
- **TF32** — Ampere tensor-core 19-bit internal mode reading/writing fp32; near-free speedup.
- **FP8 (E4M3/E5M2)** — 8-bit, Hopper+; ~2× over FP16 (DeepSeek trained 671B in FP8).
- **FP4 (NVFP4)** — 4-bit, Blackwell only; ~2–3× throughput, ½ memory vs FP8.
- **Mixed precision / autocast** — matmuls in bf16, master weights + reductions in fp32.
- **Loss scaling** — multiply loss before backward to keep fp16 gradients in range (bf16 avoids).
- **MFU (Model FLOPs Utilization)** — fraction of the GPU's peak FLOPs doing useful model math;
  naive 20–40%, your laptop ceiling ~25%, datacenter 70%+.
- **HBM vs SRAM** — slow large GPU memory vs fast small on-chip memory; FlashAttention keeps work
  in SRAM.
- **FlashAttention (2/3/4)** — tiled attention that never materializes the T×T matrix in HBM.
  FA-2 = your card; FA-3 = Hopper; FA-4 = Blackwell.
- **Fused kernel / Liger** — combine many ops into one to cut memory traffic (~40% throughput).
- **Fused cross-entropy** — never builds the full B·T·vocab logits tensor; `chunks=16` optimum.
- **Gradient checkpointing** — recompute activations in backward instead of storing them; required
  at ctx1024 on 8 GB.
- **Gradient accumulation** — sum gradients over micro-batches to simulate a bigger batch.
- **Sysmem-fallback thrash** — Windows driver silently spills over-budget VRAM to host RAM (~25×
  slower). Signature: **100% util but ~57 W and ~18 s/step**; fix with mem_fraction 0.92.

## Quantization & submission
- **Quantization** — map float weights to small ints (int8 = 4×, int6 = 5.3× smaller).
- **Per-row scaling** — a separate quantization scale per weight row; much better than one global.
- **QAT** — simulate quantization *during* training so weights survive it (straight-through grad).
- **PTQ** — quantize after training; simpler, lossier.
- **GPTQ-lite clip search** — sweep clip thresholds to tame outliers and keep the bulk fine.
- **EMA weights** — slowly-averaged weights, shipped for a free generalization bump.
- **zstd / lzma** — entropy coders that compress the quantized ints further into the `.ptz`.
- **TTT (test-time training)** — legal on-the-fly adaptation to the eval text to shave BPB.
- **Logit calibration** — temperature/softcap scaling of logits for better-calibrated probs.
- **16 MB budget** — weights **+ code** must compress under this; champion artifact ≈ 1.34 MB.

## The experiments (shorthand)
- **nanolab** — the clean teaching trainer: one guide-lever = one flag; logs everything from step 1.
- **Parameter Golf** — the competition: best BPB for a model that compresses < 16 MB, trains < 10
  min on 8×H100.
- **SOTA architecture ladder** — the staged ablation that found the champion (gated attn + value
  resid, BPB 1.985).
- **Mixer bake-off / scaling crossover** — the 2M and 8.2M token mixer comparisons.
- **APRDH** — the experimental adaptive raw-byte recurrent model with learned compute routing.
- **DiffuGPT conversion** — turning the AR model into a masked-diffusion LM (ppl 19.5→8.2).
- **RADA / HyperCascade / DeltaHybrid** — the hybrid SSM+attention trainers.

---

← Back to [`00-README.md`](00-README.md)
