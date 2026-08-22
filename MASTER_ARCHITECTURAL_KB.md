# Parameter Golf & Rust_MLKit — Master Architectural Knowledge Base

**Status:** Artifact-verified (code + JSON/metrics), deep audit 2026-07-20  
**Purpose:** Durable source for later distillation (talks, READMEs, decision memos, resumes). Prefer citing the evidence paths below over narrative docs when numbers disagree.  
**Scope:** Training (parameter-golf, nanolab, arch_02 metal-native) + inference (gemma-metal) on Apple M5 Pro / CUDA ablation hosts.

---

## How to read this document

### Evidence grades

| Grade | Meaning |
|-------|---------|
| **A — Locked artifact** | Number appears in champion JSON, gate JSON, or `metrics.jsonl` FINAL line |
| **B — Code + measured log** | Implementation in source; timing/throughput in bench txt/JSON but not a formal gate |
| **C — Design intent** | Present in code/architecture; no exclusive A/B metric |
| **D — Doc-only / folklore** | Appears in DECISIONS/README/notes; **not** found in bench/metrics artifacts — do not promote without rebench |

### Distillation rules

1. Prefer **A** numbers. If you need a single headline, use the artifact’s own `summary` field when present.
2. Never equate **peak** with **median** or **mean**.
3. Never equate **selection BPB** with **FINAL EMA BPB** (different stages).
4. Never attribute arch-ladder BPB to QAT (different suites).
5. Native gemma-metal ≠ MLX Phase-0. Label the runtime.

### Repo map (primary trees)

| Tree | Role |
|------|------|
| `parameter-golf/` | Sprint trainers, QAT/export, GDN verify refs, toy APRDH |
| `nanolab/` | Mixer bakeoffs, WY GDN / SSD Mamba, optimizer experiments |
| `Rust_MLKit/arch_02_value_resid/metal-native/` | Soft + FA_TILED + banked/Polar Muon training on Metal |
| `Rust_MLKit/gemma-metal/` | Gemma-4 E4B/31B Metal inference, DFlash port, FA kernels |
| `Rust_MLKit/reference/ablation_results/` | Locked ablation champions |
| `research/` | Exact-128M Polar funnel manifests |
| `experiment-notes/` | Narrative experiment writeups (secondary to artifacts) |

---

## Ten best achievements (detailed)

---

### 1. Architecture Ladder Champion — Gated Attention + Value Residual

**Grade:** A  
**Systems:** parameter-golf ablation → feeds `arch_02_value_resid`  
**Primary artifacts:**
- `Rust_MLKit/reference/ablation_results/champion_arch_ladder.json`
- `Rust_MLKit/reference/ablation_results/arch_ladder_long_results.json`
- Writeup: `experiment-notes/training/04-sota-arch-ladder.md`

#### Context
Hyperparameter tweaks alone were insufficient for the sprint quality target. Architectural levers (KV compression, value residual, gated attention, recursive weight-sharing) needed a fixed ablation protocol because some levers hurt alone and combinations do not compose linearly.

#### Protocol
- Suite: `sota_arch_ladder` via `parameter-golf/run_ablation_3070ti.py`
- Hardware (ablation): RTX 3070 Ti
- Stages: short (300) → mid (1000) → long (3000) with seeds `[1337, 42]`
- Ranking key: `calibrated_bpb`, then `sliding_bpb`, `step_avg_ms`, `artifact_bytes`
- Champion env overrides: **only** `GATED_ATTENTION=1`, `VALUE_RESIDUAL=1` (no QAT flags)

#### Technical content
- **Value residual:** carries almost all of the quality win vs control.
- **Gated attention alone:** collapses on the long board (~0.104 BPB worse than champion).
- **Combo `gated_value_resid`:** tiny edge over value-resid-only (~0.0027 BPB; seed-noise margin).
- **Anti-pattern `recur_2x3`:** recursive weight-sharing destroys depth specialization (short-stage calibrated BPB ~2.851) despite smaller artifact size.

#### Locked metrics
| Metric | Value | Grade |
|--------|-------|-------|
| Calibrated BPB (long mean) | **1.9847424805646139** | A |
| Sliding BPB | **1.98797577** | A |
| Artifact bytes (mean) | **1,341,003 (~1.34 MB)** | A |
| Step avg ms | **1441.175** | A |
| value_resid-only BPB | 1.987491 (Δ+0.002749) | A |
| gated_attn-only BPB | 2.088671 (Δ+0.103929) | A |

#### Distillation one-liner
> Two-seed arch ladder locked **gated attention + value residual** at **BPB 1.9847** and **1.34 MB** int6 artifact; gating without value residual fails hard.

#### Caveats for later distill
- This is a **toy/sprint proxy** ladder (4L / related sota preset), not a claim about 7B-class models.
- Int6 export size is packaging; **do not** say “QAT produced 1.9847” (see §5).

---

### 2. Metal-Native Soft-Split + FA_TILED Quality Track

**Grade:** A (quality), B (throughput on quality path)  
**Systems:** `Rust_MLKit/arch_02_value_resid/metal-native/`  
**Primary artifacts** (under `metal-native/out/`):
- `sota_f32_clipsoft_seed1337_20k_fa_tiled_softsplit_warmdown/metrics.jsonl` → FINAL EMA **1.89688**
- `sota_f32_clipsoft_seed1337_100k_fa_tiled_softsplit_wsd/metrics.jsonl` → FINAL EMA **1.882767**
- `sota_f32_clipsoft_seed42_20k_fa_tiled_softsplit_warmdown_reseed/metrics.jsonl` → **1.887607**
- Contrast speed path: `sota_f32_clipsoft_seed1337_harden/metrics.jsonl` → ~**72k tok/s**, FINAL EMA **2.050** (not quality)

#### Context
CUDA sprint quality (~1.99 BPB class) and Apple Silicon training throughput had to coexist on one native stack. Soft-split + tiled FlashAttention is the locked quality operating point.

#### Technical content
- Soft-split training with FA_TILED attention kernels.
- Soft √c clip and packed-encode reductions feed the same step graph.
- Banked / Polar Muon (see §3–4) sits under the same trainer.
- **Quality path ≠ harden speed path.** Harden removes work for throughput and regresses BPB.

#### Locked metrics
| Run | FINAL EMA BPB | Approx last-20 tok/s | Notes |
|-----|---------------|----------------------|-------|
| 20k softsplit warmdown (seed 1337) | **1.89688** | ~56.6k | Quality |
| 20k softsplit warmdown (seed 42) | **1.892465** | ~56.6k | Quality |
| 20k softsplit warmdown reseed (42) | **1.887607** | ~58.4k | Quality |
| 100k softsplit WSD (seed 1337) | **1.882767** | ~**60.4k** | Best Soft EMA found |
| harden (short) | 2.050173 | ~**72k** (peak 72473) | Speed-only; not quality claim |

#### Distillation one-liner
> Metal-native Soft+FA_TILED reaches **EMA BPB 1.883 @100k** (~60k tok/s); do not quote harden’s **72k tok/s** as the quality operating point.

#### Caveats
- Some directory names on disk may be awkward to open from shells (macOS normalization); open via Python/`os.listdir` if needed.
- 100k warmdown sibling dir was stopped early (`STOPPED_EARLY.txt`) — prefer **wsd** metrics for the 100k claim.

---

### 3. Polar Express Muon — Exact-128M Funnel Champion

**Grade:** A  
**Systems:** metal-native exact gate + train; research manifests  
**Primary artifacts:**
- `research/champion-run.json`
- `research/exact-128m-gate-polar.json`
- `out/champion_128m_seed1337/metrics.jsonl`
- Funnel notes: `experiment-notes/arch-metal/51-m5-128m-optimizer-funnel-preflight.md`

#### Context
Bank-batched NS5 removed the optimizer dispatch bottleneck, but which Muon-family polynomial wins **quality** at exact 128M required an equal-data funnel with resume parity gates.

#### Funnel shape (conceptual)
1. Native parity-qualified candidates (NS5, NS3, Polar, NorMuon, Muown, MONA, AdamW-family, …)
2. LR sweeps at smaller token budgets
3. Exact-scale advancement
4. Exact checkpoint resume gate (`exact_gate`)
5. Locked champion train (`--preset arch02-128m`, 2000 steps, B16/T256, seed 1337)

#### Winner configuration
- Optimizer: **`muon_polar_adamw`**
- Matrix LR: **0.05**
- Parameter count: **128,367,988**
- Data: FineWeb SP1024 paths in champion argv
- EMA decay: 0.997; final warmdown 350

#### Locked metrics
| Metric | Value | Grade |
|--------|-------|-------|
| Selection mean validation BPB | **2.1699185** | A |
| Exact gate `loss_delta` | **1.430511474609375e-6** | A |
| Exact gate `passed` | **true** | A |
| Resume loss atol | 1e-5 | A |
| Champion FINAL EMA sliding BPB | **2.015756** | **DISPUTED** |

> **Conflict (2026-08-22).** `research/champion-run.json` records
> `champion_final_ema_sliding_bpb = 2.010659` (seed 1337) and `2.040352` (seed 2026)
> for this run, against the **2.015756** recorded here, in
> `Rust_MLKit/docs/optimization_map.md` and in `metal-native/DECISIONS.md`. The two
> figures come from different audit passes and have not been reconciled, so the grade is
> downgraded from A to DISPUTED. Do not cite either number until the passes are
> identified. Tracked as D4 in `docs/ISSUES_AND_GAPS_2026-08-22.md`.

| Steady dispatch budget | ~1975–2019 typical; gate sample 1707 | A/B |

#### Distillation one-liner
> Equal-data funnel + exact resume gate selected **Polar Muon @ lr 0.05** for exact-128M (selection BPB **2.170**; full-run FINAL EMA **2.016**).

#### Caveats
- **Selection BPB ≠ FINAL EMA.** Quote both only with labels.
- `champion-run.json` may show `"locked": false` depending on unlock workflow — still the declared winner artifact.
- Audit8 dir (`out/champion_128m_seed1337_audit8`) can show slightly different FINAL EMA; prefer the path named in the manifest unless you intentionally re-lock.

---

### 4. Whole-Bank NS5 Muon (Systems Substrate)

**Grade:** B (code), D (8.8× timing)  
**Systems:** metal-native `optim.rs`; also Burn-port / Python banked Muon  
**Primary code:**
- `Rust_MLKit/arch_02_value_resid/metal-native/src/optim.rs` (`NS_A/B/C`, “4 bank dispatches”)
- `Rust_MLKit/DECISIONS.md` (banking design)
- Python: `parameter-golf/train_gpt_sprint_native.py` (Muon / NS5)

#### Context
Executing Newton-Schulz per 2D matrix on Apple Silicon produced severe command-queue serialization (~60+ tiny enqueues). Banking shape-aligned matrices into 3D stacks collapses dispatch count.

#### Mathematical content
Quintic Newton-Schulz (NS5) used in the Muon step:
\[
X_{k+1} = X_k \left( a I + b X_k^\top X_k + c (X_k^\top X_k)^2 \right)
\]
with coefficients in code:
\[
a = 3.4445,\quad b = -4.7750,\quad c = 2.0315
\]
(`NS_A`, `NS_B`, `NS_C` in `optim.rs`).

Four banks (typical):
- `qo_bank`
- `kv_bank`
- `mlp_up`
- `mlp_down`

Init uses a **convergent cubic** orthogonalization path (distinct from Muon quintic NS5). Orthogonality test gates in Burn-port are on the order of **&lt; 0.05**, not 1e-8.

#### Metrics
| Claim | Status |
|-------|--------|
| Coeffs a/b/c | **A/B — in source** |
| 4 bank dispatches | **B — in source comments + call sites** |
| ~60+ → 4 enqueue collapse | **B — design + DECISIONS; consistent with code structure** |
| **8.8× (4201 → 478.8 ms)** | **D — DECISIONS/optimization_map only; no bench JSON found** |
| Live Polar optim_ms ~400–450 | **A — champion metrics.jsonl** (post-banked regime, not A/B) |

#### Distillation one-liner
> Banked NS5 Muon collapses ~60+ matrix orthogonalizations into **4** shape-aligned dispatches (`a/b/c = 3.4445/−4.7750/2.0315`); Polar (#3) is the quality winner on that substrate.

#### Caveats
- Do not headline **8.8×** until rebench JSON exists.
- Python “Parallel Muon” (reduce-scatter → local NS → all-gather) is a distributed variant; Metal banking is the Apple path.

---

### 5. STE Fake-Quant + GPTQ-Lite Clip Search (Export / QAT Toolkit)

**Grade:** B (implementation), A (pack size on ladder artifacts), **overstated if tied to ladder BPB**  
**Systems:** `parameter-golf/train_gpt_sprint_native.py` (and arch_01 copy under Rust_MLKit)  
**Related suites:** separate QAT ablation candidates (`qat_010`… in ablation suites)

#### Context
Competition artifact budget is **16 MB** compressed. FP32 ~128M weights are ~512 MB. Low-bit export is mandatory; naive PTQ hurt BPB, so STE QAT + GPTQ-lite clip search were built.

#### Technical content
1. **STE fake-quant** in `CastedLinear.forward`:
   - Pattern: `w + (w_q - w).detach()`
   - Training updates stay in FP32; forward simulates quantized weights.
2. **Mixed categories:** int6 for `mlp`/`attn`; int8 for embeddings (export path).
3. **Clamp asymmetry:**
   - QAT fake-quant clamp often `[-32, 31]`
   - Export int6 uses `clip_range=31` → `[-31, 31]`
4. **GPTQ-lite scale clip search:** percentile sweep (e.g. `0.9990…1.0`) minimizing
   \[
   \min_\alpha \| W - \mathrm{quant}(W, \alpha) \|_2^2
   \]
5. **Packing:** int6 + lzma (`final_model.int6.ptz` class artifacts).
6. **Late QAT:** `LATE_QAT_THRESHOLD` (default 0.15) can auto-enable; default `QAT_ENABLED=0`.

#### Metrics
| Claim | Status |
|-------|--------|
| STE + GPTQ-lite code exists | **B — verified in trainer** |
| Ladder champion ~1.34 MB | **A — artifact_bytes** |
| Ladder champion trained with QAT | **FALSE / overstated** — champion env has no `QAT_ENABLED` |
| “BPB 1.9847 because of QAT” | **Do not claim** |

#### Distillation one-liner
> STE QAT + GPTQ-lite clip search + int6/lzma packing clear the **16 MB** budget (~**1.34 MB** artifacts); the **1.9847** ladder champion is an **arch** result, not a QAT result.

---

### 6. Verified Chunk-Parallel Sequence Mixers (WY GDN + SSD Mamba-2)

**Grade:** A/B  
**Systems:** `nanolab/mixers.py`; refs in `parameter-golf/verify_*.py`  
**Primary artifacts:**
- Baselines: `nanolab/out/gpu_sweep_mixer.json` (gdn **238.18**, mamba2 **333.23** tok/s)
- GDN after: `nanolab/out/ab_gdn/metrics.jsonl` (peak **1605.70** tok/s)
- Mamba after: `nanolab/out/bakeoff_mamba2/metrics.jsonl` (peak **2304.59**); `scale_mamba2` peak **2251.18**
- Code: `gdn_chunked`, `ssd_chunk_parallel` in `nanolab/mixers.py`

#### Context
Linear-time mixers (GDN, Mamba-2) beat attention on memory at short budgets, but naive sequential PyTorch scans destroy occupancy. Chunk-parallel forms recover throughput **with** sequential reference tests.

#### Technical content — Gated DeltaNet (WY / UT)
Recurrence (gated delta rule):
\[
h_t = (I - \alpha_t \otimes u_t v_t^\top)\, h_{t-1} + \alpha_t \otimes \beta_t x_t^\top
\]
- Production path: vectorized **WY / UT-transform** with `solve_triangular`.
- Default chunk often **C=32** in GDN helpers; config `mixer_chunk` may be 64 with `min(mixer_chunk, block_size)`.
- FP32 accumulation / autocast-off around the solve for stability.
- `parameter-golf/verify_gdn_wy.py` is a **chunked exact autograd reference**, not the full WY speed implementation.

#### Technical content — Mamba-2 SSD
- Zero-order hold / SSD chunk-parallel scan lives on the **Mamba** path (not GDN).
- Verified against sequential references (`verify_scan.py` family).

#### Locked / recorded metrics
| Mixer | Baseline tok/s | Best recorded after | Speedup vs baseline | Grade |
|-------|----------------|---------------------|---------------------|-------|
| GDN | **238.18** | **1605.70** (`ab_gdn`) | **~6.7×** | A |
| Mamba-2 | **333.23** | **~2305** (bakeoff) | **~6.9×** | A |
| Mamba-2 “3224 / 9.7×” | — | **not in results JSON** | — | D (README) |

#### Distillation one-liner
> Chunk-parallel **WY GDN (~6.7×)** and **SSD Mamba-2 (~6.9× vs recorded peaks)** restore linear-mixer throughput; do not cite README’s 9.7×/3224 without a matching artifact.

#### Caveats
- ZOH language belongs to Mamba SSD, not GDN.
- Throughput depends on batch/ctx; quote the artifact path.

---

### 7. Hazard Skip-Auto Shipping Default (E4B Hot Decode)

**Grade:** A  
**Systems:** gemma-metal runtime barriers  
**Primary artifact:** `Rust_MLKit/gemma-metal/bench/results/hazard_ab_e4b_20260719T065759Z.json`  
**Code:** `GemmaGpu::new_inference` → hazard skip-auto when env unset (`kernels.rs` / gpu init path)

#### Context
Always-on Device barriers after every dispatch are golden-safe for exactness/debug but serialize the Metal queue and cost ~5 tok/s on E4B Hot.

#### Technical content
| Arm | Env semantics | Behavior |
|-----|---------------|----------|
| hazard0 | `METAL_RUNTIME_HAZARD_BARRIERS=0` | Always-on barrier after every dispatch |
| hazard1 | `=1` | Skip auto; explicit RAW at phase edges |
| default | unset | Same as hazard1 (shipping) |

Capture / exactness lanes can force always-on while `HiddenCapture` is attached.

#### Locked metrics (real E4B Q4 Hot, 16 steps after prefill T=4, TRACE off, FUSE_LAYER off)
| Arm | tok/s | TTFT ms |
|-----|-------|---------|
| hazard0 (always-on) | **17.72** | 197.8 |
| hazard1 (skip-auto) | **22.54** | 148.1 |
| default (shipping) | **23.24** | 145.0 |
| Speedup hazard1 vs 0 | **1.272×** | — |

#### Distillation one-liner
> Shipping E4B decode uses hazard **skip-auto** (**22.5–23.2 tok/s**) vs always-on **17.7** (**~1.27×**); use always-on for golden/parity.

#### Caveats
- Product hazard can **FAIL** token-0 exactness vs always-on — documented trade, not a silent bug.
- Mini-graph lines in the same logs are **not** the gate metric.

---

### 8. MLX DFlash Block-Verify on Gemma-4-31B (Honest Statistics)

**Grade:** A (JSON), with explicit anti-cherry-pick rules  
**Systems:** MLX product path (Phase-0); native port separate  
**Primary artifacts:**
- Multi-prompt: `Rust_MLKit/gemma-metal/bench/results/run_20260713_152543_dflash_31b_mlx.json`
- NAX A/B: `Rust_MLKit/gemma-metal/bench/results/mlx032_nax_ab_31b.json`
- Native gates: `Rust_MLKit/gemma-metal/bench/results/latest_dflash_parity_gates.json`
- Narrative (secondary): `gemma-metal/docs/gates.md`

#### Context
Single-token decode (M=1) of 31B Q4 is memory-bound (~12–13 tok/s). Speculative block verification evaluates K draft tokens in one target forward (M=K), shifting GEMV→GEMM. On **mlx 0.32**, M>1 quantized GEMMs can use M5 Neural Accelerators (NAX).

#### Technical content
- Draft: `z-lab/gemma-4-31B-it-DFlash` (4-bit)
- Target: `mlx-community/gemma-4-31b-it-4bit` (plain baseline)
- Block sizes in artifacts vary (primary JSON uses `block_size: 16`; other sweeps use 5/8)
- Exact verify aims for greedy-identical streams (honest lane)
- Native `gemma-metal` ports the graph (`dflash.rs`, `step_verify`, device `HiddenCapture`) but is **not** at product speed

#### Honest metrics — primary multi-prompt JSON (mlx 0.31.2)
| Prompt | Plain tok/s | DFlash tok/s |
|--------|-------------|--------------|
| math | 13.22 | 15.66 |
| code | 11.16 | **32.13** (peak) |
| general | 13.67 | 13.65 |
| **Mean** | **12.68** | **20.48** |
| **Median (DFlash)** | — | **15.66** |
| **Summary speedup** | — | **1.61×** |

#### Honest metrics — mlx 0.32 NAX A/B
| mlx | decode_median | Notes |
|-----|---------------|-------|
| 0.31.2 | **18.64** | pre-NAX verify |
| 0.32.0 | **27.77** (~**1.49×** vs 0.31.2) | NAX on M>1 GEMMs |
| 0.32.0 wired | **27.81** | wired-memory no material effect |

#### Native gemma-metal DFlash (do not conflate)
| Field | Value |
|-------|-------|
| `real_31b.best_dflash_tok_s` | **~1.516** |
| mean accept @ best | ~**2.4** |
| `exact_vs_greedy` | **true** (parity lane) |
| `product_31b_decode_ge_15` | **false** |
| mini DFlash best | ~**705** tok/s (synthetic; still &lt; mini greedy) |

#### Distillation one-liner
> MLX DFlash on 31B: honest mean **~20.5 tok/s (1.61×)**; peak **~32** on code; mlx0.32 NAX median **~27.8**. Native port is exactness-capable but ~**1.5 tok/s** — not product-ready.

#### Anti-patterns (do not repeat)
- Calling peak **32** a “median ~31”
- Computing 2.5× as peak÷plain-mean
- Citing mini ~700 tok/s as product native 31B

---

### 9. Heterogeneous Dual FlashAttention-2 Metal Kernels

**Grade:** B  
**Systems:** gemma-metal attention  
**Primary code/docs:**
- Kernels: `kernels/flash_attn_swa_h256.metal`, `kernels/flash_attn_global_h512.metal`
- Wrappers: `flash_attn_swa_h256(_ex)`, `flash_attn_global_h512(_ex)` in `src/kernels.rs`
- Architecture note: `gemma-metal/docs/architecture.md`
- Bench harness timings in `src/bin/bench.rs` + `bench/results/*_bench.txt`

#### Context
Metal Performance Shaders attention head-dimension limits (≤64/128 class) cannot serve Gemma-4’s mixed layout: sliding-window layers at head_dim **256**, global layers at head_dim **512**.

#### Technical content
Online softmax in threadgroup tiles (FA-2 style):
\[
m_i = \max(m_{i-1}, S_i),\quad
d_i = d_{i-1} e^{m_{i-1}-m_i} + e^{S_i - m_i}
\]
Also folded into the path:
- Partial RoPE with rotary factor **0.25**
- QK-norm (scale often 1.0 in fused prep kernels)
- Draft-path variant: `flash_attn_swa_h128` for DFlash

FA still largely consumes **dense** KV buffers; not fully bound to `KvLayout` rings (densify-on-wrap where needed).

#### Metrics
| Claim | Status |
|-------|--------|
| Kernels exist + wired | **B — verified** |
| ~20–23 µs/call | **B — bench txt** (e.g. ~21.9–23.1 µs SWA h256); not a JSON field |
| Numerical tol **1e-5** | **FALSE for FA** — CPU-match tests use **`max_err < 2e-3`** |
| 1e-5 elsewhere | Appears on other kernel tests (e.g. gelu), not FA |

#### Distillation one-liner
> Custom FA-2 kernels for **h256 SWA** and **h512 global** (~**22 µs**, tol **2e-3**) unblock Gemma-4 head dims MPS cannot serve.

---

### 10. Native Hot Q4 E4B Decode Ladder (+ Supporting Substrate)

**Grade:** B (historical ladder), A for individual JSON points  
**Systems:** gemma-metal Hot path, KvLayout, dual-norm, fusion  
**Primary artifacts:**
- Early: `bench/results/run_e4b_gemma_metal_1783965523.json` → **4.7757** tok/s
- Peak class: `run_e4b_gemma_metal_1783981596.json` → **~25.10**; simd notes label “25.10 peak”
- Quiet class: multiple runs ~**23.6–23.9** (e.g. `1784048651` **23.91**)
- Supporting: `kv.rs` (`SlidingRing` / `GlobalFull`), `gpu_model.rs` dual-norm + `layer_scalar`, `HiddenCapture`

#### Context
Early native decode was dominated by host sync, host gelu, and unfused traffic. The stack moved toward GPU-resident Hot Q4 GEMV, fused MLP/KV store, dual FA, and static scratch/KV layout. Still below MLX E4B Phase-0 (~76 tok/s).

#### Supporting technical pieces (substrate, not separate Top-10 winners)

**KvLayout (Grade C/B)**  
- `SlidingRing` for SWA, `GlobalFull` for full attention  
- GPU-resident slots; shared consumers via base+offset  
- “Zero alloc after load” is a **plan/intent** (`scratch.rs`); no alloc-counter metric proving mid-decode zero forever  
- Session setup / capture / verify paths still allocate

**Gemma-4 dual-norm + `layer_scalar` (Grade B)**  
- Post-attn and post-MLP RMSNorm before residual add  
- `layer_scalar` multiplies full layer output (often folded into fused residual)  
- Observed upload range ~**0.0364–0.9805** (not 0.992)  
- E4B true dual-norm is **opt-in** (`GEMMA_METAL_E4B_DUAL_NORM`); 31B/no-PLE uses dual-norm for MLX parity  
- Wired through `step_verify` / GEMM verify for DFlash

**Device-side HiddenCapture (Grade B)**  
- Stage hidden states with `copy_f32_to_offset` / `copy_f32_range` into GPU `capture_row`  
- Defer host assemble to end of block — removes mid-layer CPU sync on happy path  
- **No** before/after tok/s JSON attributing a product win solely to capture  
- Exactness lane may still force always-on barriers while capturing

**Roofline redirect (Grade B/C)**  
- `bench/results/kernel_roofline_finding.json`: host ~**273 GB/s**; narrative that at ~21.5 tok/s ~**77%** of token time is overhead  
- Corrected false “4× GEMV peel headroom” story → prioritize fusion, hazard policy, speculation

#### Ladder metrics (cite as progression, not one A/B)
| Point | tok/s | Notes |
|-------|-------|-------|
| Early honest-partial | **4.78** | PLE skipped era |
| Peak | **~25.10** | Jul-13 peak class |
| Quiet shipping class | **~23.6–23.9** | Multiple quiet JSONs |
| Recent shipping variance | ~18–23 | Depends on flags/fusion |

#### Distillation one-liner
> Native E4B Hot climbed from **~4.8 → ~25 peak / ~24 quiet** tok/s via GPU-resident Q4 + fusion; still ~3× below MLX E4B, with hazard policy (#7) as a locked +1.27× lever.

---

## Demoted / do-not-promote claims

| Claim | Why demoted | What to say instead |
|-------|-------------|---------------------|
| APRDH “60% param reduction” | Toy runs incomplete/blocked; no locked packaged BPB | Describe architecture only, no compression metric |
| Prefetch “860→980 tok/s, 82%→97% GPU” | Not in cited DECISIONS span; training prefetch exists without those numbers | “Async span prefetch exists (training)” |
| Native DFlash ~31 tok/s | Actual best ~1.52 tok/s | “Exactness ported; product speed unmet” |
| MLX “median ~31 / 2.5×” | Peak cherry-pick | Mean **20.5 / 1.61×** or NAX median **27.8** |
| Muon “12×” or unverified “8.8×” | 12× outdated; 8.8× doc-only | “4-bank NS5; Polar quality winner” |
| Mamba “9.7× / 3224 tok/s” | Not in results JSON | “~6.9× to ~2305 tok/s recorded” |
| QAT → BPB 1.9847 | Wrong suite | “Arch ladder 1.9847; QAT is separate toolkit” |
| FA numerical tol 1e-5 | Tests use 2e-3 | “tol 2e-3” |
| `layer_scalar` up to 0.992 | Logs max ~0.9805 | “~0.036–0.981” |
| Staged capture restored 2.4× alone | No attributing JSON; larger restores from gelu/tanh/hazard | “Removes mid-layer sync for DFlash conditioning” |
| KvLayout “saved 1.2 ms/token” | No supporting metric | “Static GPU KV layout after load” |

---

## Cross-cutting architecture map

```
parameter-golf ablations
  └─ gated_value_resid champion (BPB 1.9847, 1.34 MB)
        └─ arch_02 metal-native Soft + FA_TILED (EMA 1.883 @100k)
              ├─ banked NS5 Muon (4 dispatches)
              └─ Polar Muon funnel winner (exact-128M)

nanolab mixers
  ├─ WY GDN chunk-parallel (~6.7×)
  └─ SSD Mamba-2 chunk-parallel (~6.9× recorded)

gemma-metal inference
  ├─ Dual FA h256/h512
  ├─ KvLayout + Hot Q4 decode ladder (~4.8 → ~25)
  ├─ Hazard skip-auto shipping (+1.27× vs always-on)
  ├─ Dual-norm + layer_scalar (MLX parity)
  ├─ HiddenCapture (DFlash conditioning)
  ├─ MLX DFlash product (honest ~20.5 mean / ~27.8 NAX median)
  └─ Native DFlash (exact PASS, ~1.5 tok/s — open problem)
```

---

## Open problems (honest frontier)

1. **Native DFlash product speed** — exactness PASS; accept/tok/s far below MLX; `product_31b_decode_ge_15` false.  
2. **E4B absolute tok/s** — quiet ~24 vs MLX ~76; roofline says overhead/dispatch dominates once GEMV is near bandwidth.  
3. **FA ↔ KvLayout binding** — still densify/dense-buffer oriented.  
4. **NAX from native** — TensorOps/NAX Int4 unbound; verify is simdgroup Q4 GEMM.  
5. **QAT×export H100 suites** — toolkit exists; not the locked arch-ladder story.  
6. **Muon timing A/B JSON** — rebench 4-bank vs per-matrix to replace DECISIONS folklore.

---

## Suggested distillation recipes

### For a resume / “10 achievements” slide
Use §§1,2,3,6,7,8 (honest MLX stats),9,10 — omit unverified 8.8× and native-31-tok/s.

### For a training-only memo
§§1–6; emphasize Soft 1.883 and Polar funnel gates.

### For an inference-only memo
§§7–10 + demoted native DFlash honesty; lead with hazard A/B JSON and MLX DFlash means.

### For a “numbers only” cheat sheet
```
Arch ladder:     BPB 1.984742 · 1,341,003 B
Soft 100k:       EMA 1.882767 · ~60k tok/s
Polar select:    BPB 2.1699185 · gate Δloss 1.43e-6 · FINAL 2.015756  [DISPUTED — see D4]
GDN:             238.18 → 1605.70 tok/s
Mamba2:          333.23 → ~2305 tok/s (recorded)
Hazard E4B:      17.72 → 22.54 tok/s (1.272×); default 23.24
MLX DFlash:      plain 12.68 · dflash mean 20.48 · peak 32.13 · 1.61×
MLX NAX 0.32:    median 27.77 vs 18.64
Native DFlash:   best 1.516 tok/s · exact PASS · product FAIL
FA:              h256/h512 · ~22 µs · tol 2e-3
E4B ladder:      4.78 → ~25.1 peak / ~23.9 quiet
```

---

## Changelog of this KB

| Date | Change |
|------|--------|
| 2026-07-20 | Initial artifact-verified master KB after deep audit (code + JSON/metrics, not docs alone). Replaces earlier narrative “Ten Achievements” drafts that mixed peak/median, QAT causality, and unmet native DFlash gates. |

---

## Appendix — key absolute paths

```
/Users/bharath/Code/parameter_golf/Rust_MLKit/reference/ablation_results/champion_arch_ladder.json
/Users/bharath/Code/parameter_golf/research/champion-run.json
/Users/bharath/Code/parameter_golf/research/exact-128m-gate-polar.json
/Users/bharath/Code/parameter_golf/out/champion_128m_seed1337/metrics.jsonl
/Users/bharath/Code/parameter_golf/Rust_MLKit/arch_02_value_resid/metal-native/out/sota_f32_clipsoft_seed1337_100k_fa_tiled_softsplit_wsd/metrics.jsonl
/Users/bharath/Code/parameter_golf/Rust_MLKit/arch_02_value_resid/metal-native/src/optim.rs
/Users/bharath/Code/parameter_golf/nanolab/out/gpu_sweep_mixer.json
/Users/bharath/Code/parameter_golf/nanolab/out/ab_gdn/metrics.jsonl
/Users/bharath/Code/parameter_golf/Rust_MLKit/gemma-metal/bench/results/hazard_ab_e4b_20260719T065759Z.json
/Users/bharath/Code/parameter_golf/Rust_MLKit/gemma-metal/bench/results/run_20260713_152543_dflash_31b_mlx.json
/Users/bharath/Code/parameter_golf/Rust_MLKit/gemma-metal/bench/results/mlx032_nax_ab_31b.json
/Users/bharath/Code/parameter_golf/Rust_MLKit/gemma-metal/bench/results/latest_dflash_parity_gates.json
/Users/bharath/Code/parameter_golf/Rust_MLKit/gemma-metal/docs/architecture.md
/Users/bharath/Code/parameter_golf/MASTER_ARCHITECTURAL_KB.md  (this file)
```
