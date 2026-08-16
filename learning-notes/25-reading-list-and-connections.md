# 25 — Reading List and How It All Connects

A map of where every idea in these notes comes from, so you can go to the primary sources. Grouped
by theme; each entry says *what to read it for* and which note file it backs. You don't need all of
it — the ★ items are the highest-leverage starting points.

---

## 25.1 The foundations (start here)

- ★ **"Attention Is All You Need"** (Vaswani et al., 2017) — the original transformer. Read for the
  Q/K/V mechanism and `√d` scaling (file 02, 10). Everything else is a modification of this.
- ★ **The Illustrated Transformer** (Jay Alammar, blog) — the gentlest visual intro to attention;
  pair it with file 02.
- ★ **nanoGPT** (Karpathy, GitHub) + his "Let's build GPT" video — the minimal, readable
  implementation your whole `nanolab` lineage descends from. Read the code alongside files 01, 10, 18.
- **The Annotated Transformer** (Harvard NLP) — line-by-line transformer in PyTorch.
- **Backpropagation** (Karpathy's micrograd / "Neural Networks: Zero to Hero") — file 18 in video form.

## 25.2 The modern architecture stack (file 03)

- **RoPE** — "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021).
  Read for the rotation-encodes-relative-position math (file 20). **YaRN** (2023) for context
  extension.
- **RMSNorm** — Zhang & Sennrich, 2019. **Pre-norm** — "On Layer Normalization in the Transformer
  Architecture" (Xiong et al., 2020).
- **QK-Norm** — used in OLMo 2, Qwen3 tech reports; "Scaling Vision Transformers" first popularized it.
- **SwiGLU** — "GLU Variants Improve Transformer" (Shazeer, 2020). One page, high signal.
- **GQA** — "GQA: Training Generalized Multi-Query Transformer" (Ainslie et al., 2023).
- **MLA** — the **DeepSeek-V2/V3** technical reports (file 02, 12).
- ★ **modded-nanoGPT** (Keller Jordan, GitHub) — the speedrun your champion tricks come from: value
  residual, zero-init, ReLU², U-Net skips, the Muon recipe (files 03, 05). The single most relevant
  repo to your competition work — it's already cloned in your workspace.

## 25.3 Optimizers (file 05, 19)

- ★ **Muon** — Keller Jordan's "Muon: An optimizer for the hidden layers of neural networks"
  write-up. Read for Newton–Schulz orthogonalization (file 19) and the Muon+Adam split.
- **Adam/AdamW** — Kingma & Ba 2014; Loshchilov & Hutter 2017 (decoupled decay). Internalize β2=0.95.
- **Lion** — "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023).
- **Sophia** — Liu et al., 2023 (cheap diagonal Hessian).
- **Schedule-Free** — Defazio et al., 2024. **Prodigy** — Mishchenko & Defazio, 2023 (LR-free).
- **μP** — "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
  (Yang et al., 2022) — the file 13/22 "tune small, scale up" guarantee.

## 25.4 Sequence mixers (files 04, 21)

- ★ **Mamba** (Gu & Dao, 2023) and ★ **Mamba-2 / SSD** ("Transformers are SSMs", Dao & Gu, 2024) —
  the selective state-space model and the chunk-parallel duality you implemented (file 21).
- **Gated DeltaNet** (Yang et al., 2024) and **DeltaNet** ("Parallelizing Linear Transformers with
  the Delta Rule") — the delta-rule recall mechanism (file 21).
- **Linear Attention** — "Transformers are RNNs" (Katharopoulos et al., 2020) — the `(QK)V = Q(KV)`
  associativity trick (file 21).
- **minGRU/minLSTM** — "Were RNNs All We Needed?" (Feng et al., 2024) — the parallel-scan minimal RNN.
- **Hybrids** — Jamba, Zamba, and the "DeltaNet-hybrid" recipes (6 GDN + 2 attn, file 04).
- **RWKV-7** — another linear-recurrent line worth knowing.

## 25.5 Precision, kernels, systems (file 06, 22)

- ★ **FlashAttention** 1/2/3 (Dao et al., 2022–2024) — the IO-aware attention that defines the
  memory story (file 06). FA-3 is the H100 target; FA-4 (2026) the Blackwell redesign.
- **Liger Kernel** (LinkedIn, GitHub) — the fused RMSNorm/RoPE/SwiGLU/CE kernels your fused-CE
  echoes (file 06).
- **Mixed precision** — "Mixed Precision Training" (Micikevicius et al., 2017) for loss scaling;
  **bf16** background from the Google Brain bfloat16 notes (file 06).
- **FP8** — the DeepSeek-V3 report (671B trained in FP8). **NVFP4** — NVIDIA Blackwell docs (file 06).
- **Parallelism** — Megatron-LM papers (tensor parallel), GPipe (pipeline), "DualPipe" in DeepSeek-V3,
  MegaScale (Jiang et al., 2024) for the MFU overlap numbers (file 22).

## 25.6 Scaling, MoE, diffusion (files 14, 15, 17)

- ★ **Chinchilla** — "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022) — the
  N-vs-D optimum and ~20 tokens/param rule (file 17). Pair with **Kaplan et al. 2020** (the original
  scaling laws).
- **Switch Transformer** (Fedus et al., 2021) — the top-1 MoE + load-balancing aux loss your
  `MoE` implements (file 14). **GShard**, **Mixtral** for top-2 routing.
- **Adaptive computation** — "Adaptive Computation Time" (Graves, 2016) and **PonderNet** (2021) —
  the halting idea behind APRDH's controller (file 14). **Gumbel-Softmax** (Jang et al., 2017) for
  differentiable discrete routing.
- **Diffusion LMs** — **DiffuGPT** / "Scaling up Masked Diffusion Models", **LLaDA** (2025), and
  **DiffusionGemma** — the masked-diffusion path you built (file 15).
- **Data** — the **FineWeb / FineWeb-edu** report (Penedo et al., 2024) for why edu-filtered data is
  ~8× more sample-efficient (file 01, 11).

## 25.7 The dependency map (how the ideas stack)

```
  scaling laws (17) ──────────── say WHY: lowest loss per param/compute/byte
        │
        ├── architecture (03) ── attention (02,10) ── RoPE (20)
        │        │                    └── mixers (04,21) ── chunk-parallel kernels
        │        └── champion tricks (gated attn, value resid) ← modded-nanoGPT
        │
        ├── optimization (05) ── Muon / Newton–Schulz (19) ── backprop (18)
        │
        ├── systems (06,22) ── precision (bf16/fp8) ── FlashAttention ── memory discipline
        │
        ├── compression (07) ── quant (int6) + GPTQ + zstd ── fits the 16 MB budget
        │
        └── methodology (16,23) ── one-lever runs, seeds, verify, log everything
                 │
                 └── makes every result above TRUSTWORTHY
```

Read the map bottom-up for "how do I *do* ML" (methodology first) or top-down for "why does the
field look like this" (scaling laws first). Your project touched every box — that's unusually
complete for a single workspace.

How much of each area the workspace actually exercised (●=built & measured, ◐=built, ○=referenced):

```
  Attention & variants (MHA/GQA/MLA)        ●●●●●  built all three, measured in bake-off
  Recurrent/SSM mixers (minGRU/Mamba2/GDN)  ●●●●●  built from scratch + chunk-parallel kernels
  Optimizers (Muon/Adam/+5 more)            ●●●●●  all 7 implemented, swept on GPU
  Precision & GPU systems (bf16/MFU)        ●●●●●  14%→25.5% MFU, sysmem-thrash diagnosed
  Quantization & compression (int6/GPTQ)    ●●●●◐  full pipeline, 1.34 MB artifact
  Scaling laws / token-budget crossover     ●●●●●  the ~7M crossover, measured precisely
  MoE & adaptive compute                    ●●●◐○  MoE built (OOM'd), APRDH experimental
  Diffusion LMs                             ●●●●◐  working DiffuGPT conversion (ppl 19.5→9)
  Distributed / multi-GPU                   ●◐○○○  grad-accum only; 8×H100 is the next step
  FP8/FP4 hardware                          ○○○○○  Ampere-capped; the reason to rent Blackwell
```

The two thin rows (distributed, FP8/FP4) are exactly the **hardware-gated** topics — everything
that *can* be learned on a single 8 GB laptop, you did.

## 25.8 The throughline, one last time

Files 01–25 answer one question at increasing depth: **what is the lowest-BPB language model you can
train in 10 minutes that compresses under 16 MB?** The answer required understanding prediction
(01), attention (02), the modern stack (03), mixers (04), optimizers (05), precision/systems (06),
compression (07) — then proving it with worked math (10, 18–21), the right systems (22), honest
evaluation (23), and the strategy that ties it together (17). You didn't just read this; you ran
it, debugged it (16), and have the logs to prove it (08). That's how you learn ML from scratch — not
from the top down, but by building something real and understanding every layer of why it works.

← Back to [`00-README.md`](00-README.md)
