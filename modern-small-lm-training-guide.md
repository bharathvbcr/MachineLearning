# Modern Small-LM Training — Implementation Guide

*A hands-on reference for training a ~128M-parameter language model on a single consumer GPU (RTX 3070 Ti, 8 GB), with a modern architecture, a proper optimizer/tuning workflow, and an optional path to a diffusion model and to cloud scale-up.*

---

## 0. Scope & philosophy

You are building a **~128M decoder-only LM** primarily to *learn how training works* — optimizers, learning-rate dynamics, loss behavior — and secondarily as a base you can later convert to a diffusion model (DiffuGPT-style) or scale up.

Three principles drive every choice below:

1. **Isolate variables.** Learn the training mechanics on a plain autoregressive (next-token) model first. Don't debug "how training works" and "how diffusion works" at the same time. Diffusion is a clean Phase 3 (§9).
2. **Instrument everything.** The loss/LR/grad-norm curves *are* the curriculum. Logging is not optional.
3. **Short runs, one variable at a time.** For learning optimizer/LR behavior, compare 30–60 min runs with a fixed seed; don't wait for full convergence on every experiment.

> Why 128M is the right size: essentially everything that happens at 12B happens at 128M too — same optimizer behavior, same LR dynamics, same failure modes — but cheap and fast enough to iterate on an 8 GB card.

---

## 1. Hardware & environment

### 1.1 The 8 GB reality (RTX 3070 Ti)

| Item | Value | Note |
|---|---|---|
| VRAM | 8 GB GDDR6X | Shared with display if it's your main GPU |
| Precision | **BF16 supported** | Ampere has BF16 Tensor Cores — use it |
| FlashAttention | FA-2 supported | Cuts activation memory + speeds attention |
| Throughput | ~10k+ tokens/s for 128M | ⇒ roughly **~1 B tokens/day** |

**Memory budget for 128M, full-parameter AdamW (~16 bytes/param):**

| Tensor | Bytes/param | 128M total |
|---|---|---|
| BF16 weights | 2 | 0.25 GB |
| BF16 grads | 2 | 0.25 GB |
| FP32 Adam *m* | 4 | 0.5 GB |
| FP32 Adam *v* | 4 | 0.5 GB |
| FP32 master weights | 4 | 0.5 GB |
| **Static total** | **16** | **~2 GB** |

That leaves ~5–6 GB (after CUDA/driver overhead) for activations — plenty for a real batch at sequence length 256–1024.

### 1.2 Setup

- **Base codebase:** start from [`karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT) — ~300 readable lines, every piece visible. Graduate to [`KellerJordan/modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt) once you want the modern/speedrun stack.
- **Stack:** recent PyTorch (CUDA build), `flash-attn`, `wandb` (or TensorBoard), `datasets`/`tokenizers` from HuggingFace.
- **Levers to enable on 8 GB:** `torch.compile`, BF16 autocast, FlashAttention, **gradient accumulation** (to get a large *effective* batch), and **activation/gradient checkpointing** if you push sequence length.

### 1.3 Cloud option (for fast sweeps / scale-up)

The 3070 Ti is plenty for learning. When you want to run many optimizer/LR sweeps in parallel, an 8×H100 node is the modded-nanoGPT speedrun territory (GPT-2-quality for roughly tens of dollars per run, ~minutes wall-clock). For the eventual large model, rent **B200s** to unlock FP8/FP4 training (§7).

---

## 2. Architecture — the modern stack vs GPT-2 defaults

GPT-2 (2019) is the teaching baseline. Here is what a 2025–2026 model changes, and whether each change matters at 128M.

| Component | GPT-2 default | Modern choice | Worth it at 128M? |
|---|---|---|---|
| Positional encoding | Learned absolute | **RoPE** (rotary) | Yes — strictly better, cheap |
| Normalization | LayerNorm | **RMSNorm**, pre-norm | Yes |
| Norm stability | — | **QK-Norm** (RMSNorm on Q,K) | Yes — stabilizes training |
| Activation / FFN | GELU MLP | **SwiGLU** (gated) | Yes |
| Attention | MHA | **GQA** (grouped-query) / MLA | Optional (KV-cache trick; see note) |
| Sequence mixer | Attention (MHA) | **Attention** vs SSM / linear-recurrent | Attention is the sub-200M default — see §2.5 |
| Capacity | Dense | **MoE** (sparse experts) | Optional, adds complexity |
| Long context | — | Sliding-window + global layers | Skip for now |
| Init/stability | — | **Zero-init** output projections | Yes — cheap stabilizer |
| HP transfer | — | **μP** (maximal-update param.) | Yes if you'll scale (§10) |

### 2.1 The pieces, briefly

- **RoPE** — encodes *relative* position by rotating Q/K vectors; generalizes to longer sequences far better than learned absolute positions. (Use YaRN later if you extend context beyond the training length.)
- **RMSNorm** — normalizes by root-mean-square only (no mean subtraction); cheaper than LayerNorm and just as stable in pre-norm.
- **QK-Norm** — apply RMSNorm to queries and keys before attention. Tamps down attention-logit blowups; now standard in OLMo 2, Qwen3, etc.
- **SwiGLU** — gated FFN: `down(SiLU(gate(x)) * up(x))`. Replaces the 2-layer GELU MLP; better quality per parameter (use ~⅔ the hidden width to keep param count matched).
- **GQA / MLA** — both shrink the **KV cache during autoregressive decoding**. See the diffusion note below before adopting them blindly.
- **MoE** — many FFN "experts," few active per token (e.g. DiffusionGemma is 26B total / 4B active). Big capacity at low active-compute; especially attractive for diffusion (§9) because diffusion runs many forward passes.

### 2.2 Recommended 128M config (dense, modern)

```
model:
  n_layers: 12
  d_model: 768
  n_heads: 12
  head_dim: 64
  pos: rope
  norm: rmsnorm           # pre-norm
  qk_norm: true
  ffn: swiglu             # hidden ~= 2/3 * 4 * d_model, rounded
  vocab_size: 50257       # GPT-2 BPE (or a slimmer custom tokenizer)
  tie_embeddings: true    # share input/output embeddings (saves params)
  zero_init_proj: true
  block_size: 1024        # context length
# ~124–128M params
```

### 2.3 Speedrun architectural tricks (from modded-nanoGPT)

The current NanoGPT speedrun stack stacks several of the above plus: **rotary embeddings**, **ReLU² activations**, **zero-init projections**, **QK-norm**, and a **slimmer tokenizer** (smaller vocab → fewer head/embedding params and FLOPs). Cloning modded-nanoGPT gives you these already wired and Muon-tuned — modify rather than reinvent.

### 2.4 ⚠️ Architecture note for the diffusion path

Many "latest" optimizations are designed for **autoregressive, causal decoding**:

- **GQA / MLA** exist mainly to shrink the *growing KV cache* during left-to-right generation. A diffusion model does parallel denoising with **bidirectional** attention — no growing cache in the same sense — so the motivation weakens. Re-evaluate, don't copy.
- **Causal masking** is removed entirely in diffusion.
- **RoPE, RMSNorm, QK-Norm, SwiGLU, MoE** all transfer cleanly. MoE is *especially* good for diffusion.

So: adopt the general improvements freely; treat KV-cache-specific attention variants as AR-only until proven otherwise on a bidirectional model.

### 2.5 Sequence mixer — Transformer vs the alternatives (sub-200M)

Everything above assumes **attention** as the token mixer. The other 2026 option is a **linear-recurrent / state-space mixer** (Mamba-2, Gated DeltaNet, RWKV-7) or a hybrid of the two.

| Mixer | What it is | Cost in T | When it wins |
|---|---|---|---|
| **Attention** | full pairwise mixing | quadratic | **Default.** Strongest, best-tested at small scale on standard text |
| **Mamba-2** | selective state-space | linear | Long context; fast, constant-memory inference |
| **Gated DeltaNet** | gated linear attention | linear | Same, with stronger recall than vanilla SSMs |
| **minGRU** | minimal parallel RNN | linear | Pedagogy / zero-dependency tiny baselines |

**Recommendation:** at this scale a **well-tuned Transformer is the strongest, most reliable default** — the NanoGPT speedrun and Parameter Golf leaderboards both converged on optimized attention. Linear-recurrent mixers buy **cheaper long context and faster, constant-memory inference**, *not* better small-scale perplexity, so reach for them only when context length or throughput is the bottleneck. The decisive factor sub-200M is the **training recipe (optimizer / LR / data), not the mixer** — lock the recipe first (Phases 1–2), then A/B mixers with tokens + seed fixed.

> Diffusion caveat (§9): SSM / linear mixers are built around **causal** recurrence; a bidirectional diffusion model changes their story — re-evaluate rather than copy, same logic as the GQA/MLA note in §2.4.

In the companion **nanolab** repo this is one flag — same MLP, embeddings, optimizer, seed, and tokens, so the comparison is clean:

```
python train.py --preset phase0 --mixer attention   # default
python train.py --preset phase0 --mixer mingru        # zero-dep recurrent reference (sequential; slow at long T)
python train.py --preset phase0 --mixer mamba2         # pip install mamba-ssm causal-conv1d
python train.py --preset phase0 --mixer gdn            # pip install flash-linear-attention
```

To keep it honest, match param counts (`--n_embd` / `--n_layer`) across mixers.

---

## 3. Data & tokenization

**Pick the dataset for the job — not Shakespeare.** Tiny-Shakespeare is fine to watch the loop run, but it's too small and stylistically narrow for a model to learn general English. Two roles:

***Phase 0 — "watch a small model actually learn":***
- **TinyStories** (`roneneldan/TinyStories`) — synthetic short stories with a tiny vocabulary, built so even few-M-param models produce **coherent, grammatical English**. The single best upgrade from Shakespeare for a learning run.
- **enwik8 / text8** — the classic char-level benchmark (bits-per-char); more diverse than Shakespeare, drops into a char pipeline the same way.

***Phase 1 — real pretraining (token-level):***
- **FineWeb-edu** (`HuggingFaceFW/fineweb-edu`, e.g. config `sample-10BT`) — **the default for small models.** The educational-quality filter makes it ~8× more sample-efficient (matches the next-best open set at ~8× fewer tokens) — exactly what you want on a tight token budget.
- **Nemotron-CC-HQ** / **DCLM-baseline** — ranked #1 and #2 in systematic head-to-head open-dataset evals, but long-horizon and heavy; overkill for sub-200M on one GPU.
- **Cosmopedia** (`HuggingFaceTB/cosmopedia`) — synthetic textbook-style (Phi-flavored); strong for small models.
- **The Pile / C4 / SlimPajama** — older diverse references. **Math:** FineMath / MegaMath. **Code:** The Stack.

You do **not** need full convergence to learn — 1–2 B tokens gives a coherent model and rich curves.

- **Tokenizer:** GPT-2 BPE (50257) is the safe default. Vocab size trades head/embedding params vs bytes-per-token; a slimmer tokenizer saves parameters under a tight budget (a real Parameter Golf lever).
- **Data ordering (free speedups):** **curriculum** (easy→hard, by compression ratio / readability) can cut steps-to-baseline ~18–45%; **sequence-length curriculum / dataset decomposition** reports up to ~3× faster time-to-accuracy.
- **Hygiene:** deduplicate and quality-filter; keep a human-data anchor if you ever mix in synthetic data.
- **Hard rule:** never let validation data leak into training (this is *the* disqualifier in benchmarks like Parameter Golf).

> In **nanolab**: TinyStories and OpenWebText work as-is via `--hf_dataset`; FineWeb-edu / Cosmopedia / DCLM need a subset config — pass `name=<config>` to `load_dataset` in `data.py` (a two-line change).

---

## 4. Optimizers (core)

### 4.1 The loop

```
for step in range(max_steps):
    for micro in range(grad_accum_steps):       # gradient accumulation
        x, y = next(loader)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = cross_entropy(logits, y) / grad_accum_steps
        loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping
    lr = schedule(step)                                    # see §5
    for g in optimizer.param_groups: g["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    log(step, loss=loss.item()*grad_accum_steps, lr=lr, grad_norm=grad_norm, tok_s=...)
```

Everything you want to learn is visible here: the **loss** (the error signal), the **gradient** (its derivative w.r.t. weights), **clipping** (bounding bad updates), the **schedule** (how aggressively you step), and the **optimizer** (how the gradient becomes a weight update).

### 4.2 The optimizer spectrum (in order of "how much they do for you")

| Optimizer | Idea | Extra memory | LR sensitivity | Use it to learn… |
|---|---|---|---|---|
| **SGD + momentum** | Fixed step along momentum | ~1× params | Very high | How fragile fixed step sizes are |
| **AdamW** | Per-parameter adaptive step from grad mean (*m*) + variance (*v*); decoupled weight decay | ~2× params | Moderate | The workhorse; the "dynamic" you pictured |
| **Lion** | Update = sign(momentum); memory-light | ~1× params | High (use ~3–10× smaller LR, higher WD) | Sign-based updates |
| **Sophia** | Adam + cheap diagonal **Hessian** (curvature) estimate, clipped | ~2–3× params | Moderate | A taste of second-order |
| **Schedule-Free AdamW / Prodigy** | **Estimate the LR for you**; no schedule needed | ~2× params | Low (auto) | "Dynamic optimizer that corrects the LR" |
| **Muon** | Orthogonalize gradient momentum (Newton–Schulz) for 2D weights | ~1× params (2D) | Moderate | The current speed frontier (~2× over AdamW) |

### 4.3 AdamW — the baseline to internalize

```
optimizer = torch.optim.AdamW(
    param_groups,            # see weight-decay grouping below
    lr=6e-4,                 # peak LR; set via LR finder (§5.1)
    betas=(0.9, 0.95),       # NB: beta2=0.95 for LM pretraining, not 0.999
    eps=1e-8,
    weight_decay=0.1,        # decoupled
)
```

**Weight-decay grouping (standard nanoGPT pattern):** decay 2D weights (linears, embeddings); **do not** decay biases, norm weights, or 1D params.

```
decay, no_decay = [], []
for n, p in model.named_parameters():
    if p.ndim >= 2: decay.append(p)
    else:           no_decay.append(p)
param_groups = [
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
]
```

### 4.4 Muon — the high-leverage swap

Muon ("MomentUm Orthogonalized by Newton-Schulz") is ~2× more compute-efficient than AdamW at scale and was the single biggest jump in the NanoGPT speedrun. Rules:

- Apply Muon **only to 2D hidden-layer weights**.
- Keep **embeddings, the LM head, norms, biases, and scalars on AdamW**.

```
# Conceptual hybrid setup (use modded-nanoGPT's tested implementation)
hidden_2d   = [p for n,p in model.named_parameters()
               if p.ndim == 2 and "embed" not in n and "lm_head" not in n]
everything_else = [p for n,p in model.named_parameters() if p not in hidden_2d]

opt_muon  = Muon(hidden_2d, lr=0.02, momentum=0.95)     # spectral / Newton-Schulz
opt_adamw = torch.optim.AdamW(everything_else, lr=6e-4, betas=(0.9,0.95), weight_decay=0.1)
# step both each iteration
```

Successors to watch: **Turbo-Muon** (faster Newton-Schulz) and **MuonBP** (block-periodic orthogonalization). Side benefit: Muon-trained weights have fewer outlier activations, so they **quantize better** afterward.

### 4.5 Learning-rate-free optimizers (matches your "auto-correcting LR" interest)

**Schedule-Free AdamW** (Defazio et al.) and **Prodigy** (Mishchenko & Defazio) estimate the step size internally, so you can drop the cosine schedule entirely. Run one head-to-head against a hand-tuned AdamW+cosine and compare the curves — it's the clearest way to *see* what a schedule was doing for you.

---

## 5. Learning rate & schedules (tuning)

### 5.1 LR finder

Sweep LR upward (e.g. exponentially) over a few hundred steps and plot loss vs LR. The "knee" just before loss explodes marks the usable ceiling; pick your peak a notch below it.

### 5.2 Warmup — and *why*

Run once **without** warmup and watch the early loss spike or diverge (fresh Adam variance estimates are unreliable, so big early steps wreck things). Add linear warmup over the first ~1–5% of steps and watch it stabilize. That contrast teaches you what warmup is for.

### 5.3 Schedules to compare

| Schedule | Shape | When |
|---|---|---|
| **Constant** | Flat after warmup | Debugging; learning-rate-free opt |
| **Cosine decay** | Warmup → cosine to ~10% (or 0) | Default for fixed-length runs |
| **WSD** (warmup-stable-decay) | Warmup → long flat → short decay | Flexible run length; resume-friendly |
| **ReduceLROnPlateau** | Drop LR when val loss stalls | Simplest *reactive/self-correcting* schedule |

```
import math
def lr_cosine(step, peak=6e-4, warmup=2000, total=100000, floor_frac=0.1):
    if step < warmup:
        return peak * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))
```

### 5.4 Other tuning levers

- **Gradient clipping** at global norm ~1.0; watch the grad-norm log — clipping flattens spikes (your most direct "error-rate correction").
- **Batch size ↔ LR**: larger effective batch generally tolerates a larger LR (roughly square-root scaling); use gradient accumulation to grow the effective batch on 8 GB.
- **μP** (§10) lets you tune LR/init at 128M and *transfer* to a larger model — exactly the prototype→scale workflow.

---

## 6. Training dynamics & monitoring ("correcting error rates")

### 6.1 Log from run one

train loss, **val loss**, learning rate, **gradient norm**, tokens/s, and **MFU** (model-FLOPs utilization). Reading these is the point.

### 6.2 Reading the curves

| Symptom | Likely cause | Fix |
|---|---|---|
| Early loss spike / NaN | LR too high, no warmup | Add/extend warmup; lower peak LR |
| Periodic loss spikes | Bad batches, LR too high, no clipping | Gradient clipping; lower LR |
| Grad-norm spikes | Same | Clip; inspect data |
| Val loss rises while train falls | **Overfitting** | More data, regularization, early stop |
| Loss plateaus early | LR too low / schedule decayed too soon | LR finder; WSD |
| Loss flat from step 0 | Bug (LR=0, frozen params, bad masking) | Sanity-check the loop |

### 6.3 Deliberate experiments to run

- **Overfitting demo:** train on a tiny data slice, watch val peel away from train — clearest generalization lesson.
- **Warmup ablation** (§5.2).
- **Optimizer bake-off:** SGD vs AdamW vs Muon vs Schedule-Free, identical seed/data, compare curves.
- **Schedule bake-off:** constant vs cosine vs WSD vs plateau.

### 6.4 Hygiene

Fixed seed; **change one variable per run**; checkpoint regularly (and test resume); name runs by the variable you changed.

---

## 7. GPU utilization & systems-level speedups

The enemy is **idle tensor cores**. A naive run often sits at **20–40% MFU** (model-FLOPs utilization) — the expensive matmul hardware stalled most of the time; a well-tuned run hits **70%+**. Tensor cores stall for three reasons, each with its own fix:

| Stall cause | Fix family | Where it applies |
|---|---|---|
| Waiting on memory (HBM) | Fused / IO-aware kernels | Single GPU and up |
| Waiting on communication | Compute–comm overlap, parallelism | Multi-GPU only |
| Idle low-precision paths | FP8 / FP4 | Hopper / Blackwell |

### 7.1 Memory stalls → faster, fused kernels (helps you on the 3070 Ti now)

**Attention** is the classic culprit; the FlashAttention lineage keeps data in SRAM instead of round-tripping to HBM:

- **FlashAttention-2** — what your Ampere card runs. A big win over naive attention, but it underuses newer GPUs (~35% utilization on H100 vs 80–85% for optimized matmul).
- **FlashAttention-3** (Hopper) — async Tensor Cores + TMA via warp-specialization, interleaved matmul/softmax, and FP8 support: **1.5–2× over FA-2**, ~75% H100 utilization (740 TFLOPs FP16), close to 1.2 PFLOPs in FP8.
- **FlashAttention-4** (Blackwell, early 2026) — redesigned around a sharp insight: on B200/GB200 the tensor cores doubled while shared-memory bandwidth and the exponential (softmax) units did not, so the bottleneck **moved off the matmuls** onto softmax (forward) and shared-memory traffic (backward). FA-4 uses async MMA, larger tiles, software-emulated exponential, and 2-CTA mode, and beats cuDNN/Triton on B200.

For the **rest of the model**, kernel-fusion libraries are the biggest single-GPU win:

- **Liger Kernel** (LinkedIn, Triton) — fuses RMSNorm, RoPE, SwiGLU, and cross-entropy and uses in-place ops: ~**40% more throughput and ~55% less GPU memory** on Llama-3-8B fine-tuning. Drop-in for HuggingFace, and it runs on your Ampere card; the freed memory directly buys a bigger batch or longer sequence.
- **Litespark** — architecture-level attention/MLP kernel work reporting **2–6× throughput** and MFU climbing from **3–8% to 17–40%** in large configs.
- **`torch.compile`** (Triton backend, default in modern PyTorch) and **CUDA graphs** — fuse ops and strip Python/launch overhead nearly for free.

### 7.2 Communication stalls → overlap & parallelism (the big lever once you're multi-GPU)

This does **not** apply to a single GPU — only when you rent a node/cluster. At scale, gradient all-reduce, expert all-to-all, and pipeline transfers dominate; cross-node expert parallelism can leave a roughly 1:1 compute-to-communication ratio.

- **DualPipe** (DeepSeek-V3) — splits each chunk into attention / all-to-all dispatch / MLP / all-to-all combine, manually tunes the SM split between compute and communication, and overlaps forward and backward so all-to-all and pipeline comm are nearly fully hidden. Part of how DeepSeek pretrained 671B on 14.8T tokens in 2.788M H800-hours.
- **MegaScale** — 3D-parallel communication overlap alone added **+6.2% MFU**, with all systems optimizations stacking to **+17.6% MFU**.
- **Expert parallelism + topology-aware placement** — MFU case studies move from **43% → 71%**.
- Frameworks that implement this: Megatron-LM, TorchTitan, NVIDIA NeMo.

### 7.3 Precision → more throughput per tensor-core op

Lower precision is a *throughput* lever, not just a memory one — it doubles or quadruples tensor-core math.

| Format | Hardware | Use |
|---|---|---|
| **BF16** | Ampere+ (your 3070 Ti) | Default for all your runs |
| **FP8 / MXFP8** | Hopper (H100/H200), Blackwell | Production-ready; ~2× over FP16 at matched accuracy (DeepSeek trained 671B in FP8) |
| **NVFP4 / FP4** | **Blackwell only** (B200, RTX PRO 6000) | ~2–3× throughput, ½ memory vs FP8; matches FP8 accuracy to ~120B. No native FP4 on H100/A100 |

Your Ampere card tops out at BF16, so FP8/FP4 are a concrete reason to rent Blackwell for the scale-up.

### 7.4 What to actually do

- **On the 3070 Ti now:** BF16 + **FlashAttention-2** + **`torch.compile`** + **Liger Kernel** + **gradient checkpointing** + **gradient accumulation** + an efficient (pre-tokenized, packed) data loader. Then watch the **MFU** line in your logs climb — that's the whole point of this section.
- modded-nanoGPT already bakes in compiled paths and a FlexAttention-style kernel — one more reason to start from it.
- **At scale:** rent Blackwell to get FA-4 + FP4, use a framework with DualPipe-style **compute–communication overlap** (Megatron-LM / TorchTitan / NeMo), and add **expert parallelism** if you go MoE.

---

## 8. The experiment plan (phased)

| Phase | Goal | Scale | Time (3070 Ti) |
|---|---|---|---|
| **0** | See the loop run | char Shakespeare → **TinyStories** | An afternoon |
| **1** | Instrumented base run | 128M, 1–2 B tokens | 1–2 days |
| **2** | Optimizer & LR experiments | 128M, short 30–60 min runs | Iterative |
| **3** *(opt.)* | Diffusion conversion | same 128M | Days |

**Architecture A/B (optional):** once Phase 2's recipe is locked, swap `--mixer` (attention / mamba2 / gdn) on fixed tokens + seed to isolate what the mixer alone buys — see §2.5. Do this *after* the recipe is settled, not before.

---

## 9. Optional Phase 3 — convert to a diffusion model (DiffuGPT-style)

Once the AR mechanics feel obvious, layer the conversion on the **same** 128M model — you now understand the base, so you're only learning the diffusion part.

The recipe (from DiffuLLaMA/DiffuGPT — adapt a pretrained AR checkpoint via continual pre-training):

1. **Attention-mask annealing** — gradually switch causal → **bidirectional** attention. (This is why a custom architecture forfeits the cheap-adaptation discount at scale: there's no pretrained checkpoint of *your* design to anneal from — see §10.)
2. **Shift operation** — preserves AR-like training dynamics during the switch.
3. **Masked/diffusion objective** — predict masked tokens across noise levels (cutoff length ~256 is fine).

Modern efficiency add-ons (from LLaDA 2.0):

- **Complementary masking** — two opposite masked views of each sequence per batch, so every position is seen uncorrupted exactly once (fixes the "MLM only learns from ~15% of tokens" inefficiency).
- **Mask-ratio bandwidth** — train on the noise levels that give the most informative gradients.
- **Confidence/auxiliary loss** — sharpens predictions for parallel decoding.

Use [`HKUNLP/DiffuLLaMA`](https://github.com/HKUNLP/DiffuLLaMA) — it ships DiffuGPT scripts/configs on LLaMA-Factory. Reminder: the **base** conversion uses annealing; for later task-SFT, turn annealing off (the model is already bidirectional). Diffusion models also benefit disproportionately from **MoE** (low active params, many forward passes).

---

## 10. Scaling up (when ready)

### 10.1 The fork you must choose consciously

- **Custom architecture → train from scratch.** Full control, but no pretrained checkpoint exists, so you pay full from-scratch cost (trillions of tokens; ~$250k+ at the 8B scale, à la LLaDA).
- **Adapt an existing open checkpoint** (Gemma / Qwen / Llama / OLMo) → cheap (continual pre-training delivers ~80–90% of from-scratch quality at ~5–10% of cost), but you inherit *their* architecture.

You cannot have both "my own architecture" *and* "cheap 12B via adaptation." Pick deliberately.

### 10.2 Mechanics at scale

- **Sharding:** full-parameter training of a 12B needs ~16 bytes/param ≈ ~190 GB of optimizer+weight state — beyond any single GPU. Use **FSDP / ZeRO** across an 8×H100 node (640 GB) or larger.
- **Precision:** rent **B200s** for FP8/FP4 to cut the bill (§7).
- **μP:** tune hyperparameters at 128M and transfer to the large model — without it you re-tune blind at every scale.
- **Optimizer:** Muon scales (Moonshot's Moonlight 16B); use the distributed implementation.

---

## 11. Quick-start checklist

```
[ ] Clone nanoGPT (or companion repo nanolab); train char Shakespeare / TinyStories; watch loss fall  (Phase 0)
[ ] Wire wandb: loss / val_loss / lr / grad_norm / tok_s / MFU
[ ] Switch base to modded-nanoGPT (RoPE, RMSNorm, QK-norm, SwiGLU, Muon)
[ ] Prepare FineWeb-edu / OpenWebText shard
[ ] Train 128M for 1–2 B tokens, BF16 + FlashAttention + grad-accum        (Phase 1)
[ ] Maximize MFU: torch.compile + Liger Kernel + grad-checkpointing; watch the MFU log climb  (§7)
[ ] LR finder → set peak LR; add warmup; pick cosine/WSD                   (Phase 2)
[ ] Optimizer bake-off: SGD vs AdamW vs Muon vs Schedule-Free (fixed seed)
[ ] Architecture A/B: --mixer attention vs mamba2 / gdn (fixed tokens + seed)  (§2.5)
[ ] Schedule bake-off; gradient-clipping + overfitting demos
[ ] (Optional) DiffuGPT conversion: annealing + masked objective + complementary masking  (Phase 3)
[ ] (Later) scale-up: μP + FSDP + B200/FP8 + FA-4 + DualPipe-style overlap (§7, §10)
```

Golden rules: **one variable per run, fixed seed, log everything, short runs for learning.**

---

## 12. References

**Codebases**
- nanoGPT — https://github.com/karpathy/nanoGPT
- modded-nanoGPT (speedrun, Muon) — https://github.com/KellerJordan/modded-nanogpt
- DiffuLLaMA / DiffuGPT — https://github.com/HKUNLP/DiffuLLaMA
- OpenAI Parameter Golf — https://github.com/openai/parameter-golf

**Optimizers**
- Muon (blog, Keller Jordan, 2024) — https://kellerjordan.github.io/posts/muon/
- "Muon is Scalable for LLM Training" — arXiv:2502.16982
- Schedule-Free — Defazio et al., "The Road Less Scheduled" (2024)
- Prodigy — Mishchenko & Defazio (2023)
- Sophia — Liu et al. (2023), arXiv:2305.14342
- Lion — Chen et al. (2023), arXiv:2302.06675

**Architecture**
- RoPE — Su et al., arXiv:2104.09864 · RMSNorm — Zhang & Sennrich (2019) · SwiGLU — Shazeer, arXiv:2002.05202 · GQA — Ainslie et al., arXiv:2305.13245 · MLA — DeepSeek-V2
- μP (HP transfer) — Yang et al., "Tensor Programs V", arXiv:2203.03466
- Sebastian Raschka, LLM Architecture Gallery / "Big LLM Architecture Comparison" — https://sebastianraschka.com/llm-architecture-gallery/

**Sequence mixers (non-attention)**
- Mamba — Gu & Dao, arXiv:2312.00752 · Mamba-2 ("Transformers are SSMs") — Dao & Gu, arXiv:2405.21060 · `mamba-ssm`
- Gated DeltaNet — Yang et al., arXiv:2412.06464 · flash-linear-attention (`fla`) — https://github.com/fla-org/flash-linear-attention
- minGRU / minLSTM ("Were RNNs All We Needed?") — Feng et al., arXiv:2410.01201
- RWKV-7 — https://github.com/BlinkDL/RWKV-LM · Titans (test-time memory) — Behrouz et al., arXiv:2501.00663

**Diffusion LMs**
- DiffuLLaMA — arXiv:2410.17891 · LLaDA — arXiv:2502.09992 · LLaDA 2.0 (to 100B) — arXiv:2512.15745
- DiffusionGemma — https://ai.google.dev/gemma/docs/diffusiongemma

**Low-precision training**
- "Pretraining LLMs with NVFP4" (NVIDIA) — arXiv:2509.25149 · "FP4 All the Way" — arXiv:2505.19115

**Systems, kernels & GPU utilization**
- FlashAttention-3 (Hopper) — arXiv:2407.08608 · FlashAttention-4 (Blackwell, Together AI, 2026) — arXiv:2603.05451
- Liger Kernel (fused Triton kernels) — https://github.com/linkedin/Liger-Kernel (arXiv:2410.10989) · Litespark — arXiv:2510.02483
- DeepSeek-V3 (DualPipe, FP8, compute–comm overlap) — arXiv:2412.19437 · MegaScale (3D-parallel overlap) — NSDI 2024

**Efficiency / data**
- Chinchilla (compute-optimal ≈ 20 tokens/param) — Hoffmann et al. (2022)
- Curriculum learning for pretraining — arXiv:2506.11300 · Dataset Decomposition — arXiv:2405.13226

**Datasets**
- FineWeb & FineWeb-edu — Penedo et al. (2024) — https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Nemotron-CC — Su et al., arXiv:2412.02595 · DCLM (DataComp-LM) — Li et al., arXiv:2406.11794
- TinyStories — Eldan & Li, arXiv:2305.07759 · Cosmopedia — https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
- enwik8 / text8 (Hutter Prize) — http://prize.hutter1.net/ · The Pile — Gao et al., arXiv:2101.00027

**Community challenge**
- "What Parameter Golf taught us" (OpenAI, 2026) — https://openai.com/index/what-parameter-golf-taught-us/

---

*Built from a working session covering: AR→diffusion conversion economics, the "why billions of tokens" question, hardware feasibility (M5 Pro, 3070 Ti, H100/B200), 2026 training-efficiency research (Muon, complementary masking, NVFP4), GPU-utilization systems work (FlashAttention 3/4, Liger kernel fusion, DualPipe communication overlap), modern architecture choices including pluggable sequence mixers (attention / Mamba-2 / Gated DeltaNet / minGRU), dataset selection (TinyStories, FineWeb-edu), and OpenAI's Parameter Golf submissions.*
