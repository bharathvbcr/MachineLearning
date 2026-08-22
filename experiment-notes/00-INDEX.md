# Experiment suite index

This lab notebook records methods, variants, failures, results, reproduction paths, and concrete artifacts for 37 experiment suites. Of these, 30 are done, five are partial, one is blocked, and one is planned. Confidence grades describe the evidence actually preserved, not the ambition of the experiment.

Copy [`_TEMPLATE.md`](_TEMPLATE.md) for new suites. Status values: `done` / `partial` / `planned` / `blocked`.

## Headline findings

- On a 3070 Ti (one seed, bs8) attention overtook minGRU between **6.6M and 7.4M tokens**. On GH200 bs32 n=5 the same pair flips at **~1.05M** then **~12.4M** (independent 20M prefix recovered 12.34M). GH200 bs8 n=5 has **no flip by 7.38M**. Short rankings lie; the token of the flip is recipe-dependent.
- Value residual supplied the architecture ladder’s robust gain; gated attention’s additional **~0.0027 BPB** edge was within seed noise, while gating alone was **~0.104 BPB** worse than the champion.
- Lean auxiliary heads improved calibrated BPB from **2.093 to 2.066** while shrinking the export.
- Compact 4L×128 + higher LR matched or beat the nearly 2× larger/slower depth proxy at long horizon; short-stage depth rankings did not survive promotion.
- Sequential Mamba-2/GDN kernels were **24–33×** too slow for fair wall-clock comparisons; verified chunk-parallel kernels recovered up to **9.7×** and required fp32 accumulation.
- On the 8 GB 3070 Ti, residency/fusion/checkpointing raised the 124M path to **~13.7K tok/s and 25.5% MFU**; memory spill, not loss, exposed the baseline failure.
- The 10k 124M run’s recipe remained the reliable laptop long-run winner; the lower-LR 20k arm regressed after its early best, so horizon alone was not the cause.
- MLX 0.32 DFlash block=5 is the measured 31B product path at **~31.7 tok/s median** with exact greedy parity; native DFlash remains below acceptance, parity, and speed gates.
- Native E4B rose from **4.78 to ~25 tok/s**, but roofline analysis attributes roughly **77%** of quiet token time to dispatch/overhead rather than GEMV.
- Metal-native training demonstrates a speed/quality split: row FA gives **56.6 ms/step**, while FA_TILED plus WSD reaches **1.8828 EMA BPB**; these are different operating points.
- Soft FA_TILED campaign: **16M Soft 20k EMA 1.7591** (new 16M lock); Soft attn sota golden 20k **1.8969** remains sota Soft crown; mingru Soft 20k 1.9933; Polar Soft 3k closed.

## How to interpret evidence

- **High:** completed comparative stages with replication or independent confirmation, plus parity/validation where relevant. Small effects may still be labeled noise.
- **Medium:** direct measurements exist, but usually one seed, few prompts, mixed schedules, session noise, or incomplete baseline coverage limits generalization.
- **Low:** planned/blocked/failed stages, modeled rather than end-to-end outcomes, noisy sessions, incomplete parity, or superseded snapshots. Low confidence is still useful diagnostic evidence.

## Training

| Note | Suite | Status | Confidence | Hardware | Finding |
|------|-------|--------|------------|----------|---------|
| [01-sota-local](training/01-sota-local.md) | Grad clip / Muon / LR / XSA / lean aux | done | High | RTX 3070 Ti (local ablation via `run_ablation_3070ti.py`) | Lean aux heads won at calibrated BPB 2.066 versus 2.093 control. |
| [02-sota-lean-followup](training/02-sota-lean-followup.md) | Aux floor / LR / XSA follow-ups | partial | Low | RTX 3070 Ti (local ablation via `run_ablation_3070ti.py`) | Keep the aux floor and raise LR to 0.027 (mid calibrated BPB 2.232). |
| [03-sota-depth-proxy](training/03-sota-depth-proxy.md) | Depth×width×MLP tradeoffs | done | High | RTX 3070 Ti (local ablation via `run_ablation_3070ti.py`) | LR-up 4L×128 beat the nearly 2× larger 8L×128 proxy at BPB 2.0715. |
| [04-sota-arch-ladder](training/04-sota-arch-ladder.md) | Gated attn + value resid champion | done | High | RTX 3070 Ti (local ablation via `run_ablation_3070ti.py`) | Gated attention + value residual won at calibrated BPB 1.985. |
| [05-sota-arch-followup-value-resid](training/05-sota-arch-followup-value-resid.md) | Cross kv1/LN/VE/XSA on champion | partial | Low | RTX 3070 Ti (local ablation via `run_ablation_3070ti.py`) | A 2.6938 short hint failed promotion; the 1.985 long champion remains held. |
| [06-toy-aprdh](training/06-toy-aprdh.md) | Adaptive raw-byte vs RADA/DeltaHybrid | blocked | Low | RTX 3070 Ti (`device:cuda`, single-GPU; logs spell out target) | Missing `apply_rotary_emb` blocked comparable APRDH runs; an interrupted run reached ~3.72 BPB. |
| [07-h100-conductor-planned](training/07-h100-conductor-planned.md) | H100 conductor stubs (`sota_h100_depth`, `sota_ttt_optimizer`, `sota_qat_export`) | planned | Low | Target: 8×H100 sprint (`runner: sprint`, `WINDOWS_SAFE_MODE=0`); local 3070 Ti outs absent | Three calibrated-BPB H100 follow-ups are defined, but no sprint outputs exist. |

## Nanolab

| Note | Suite | Status | Confidence | Hardware | Finding |
|------|-------|--------|------------|----------|---------|
| [10-phase0-smoke](nanolab/10-phase0-smoke.md) | `cpu_smoke` + `phase0_tinystories` | done | Medium | RTX 3070 Ti Laptop 8 GB (phase0); CPU path also exercised | Smoke and phase0 established descending loss and clean logs before A/Bs. |
| [11-phase1-fineweb](nanolab/11-phase1-fineweb.md) | Instrumented 124M base | done | Medium | RTX 3070 Ti Laptop 8 GB | The first FineWeb run established the MFU/tok-s measurement contract. |
| [12-mixer-ab-tinystories](nanolab/12-mixer-ab-tinystories.md) | `ab_{attention,mingru,mamba2,gdn}` | done | Medium | RTX 3070 Ti Laptop 8 GB | Cheap one-lever A/Bs reproduced the low-data recurrent-mixer advantage. |
| [13-mixer-bakeoff-2M](nanolab/13-mixer-bakeoff-2M.md) | FineWeb bakeoff | done | Medium | RTX 3070 Ti Laptop 8 GB | minGRU led the 2M-token board at best val loss 5.837. |
| [14-scale-crossover-8M](nanolab/14-scale-crossover-8M.md) | Headline crossover | done | High | RTX 3070 Ti Laptop 8 GB | Attention overtook minGRU between 6.6M and 7.4M tokens. |
| [15-crossover-followups](nanolab/15-crossover-followups.md) | `xs_*` + `ext_*` | done | Medium | RTX 3070 Ti Laptop 8 GB | Follow-ups preserved the crossover; only its token timing shifted with scale. |
| [16-optimizer-quality-bakeoff](nanolab/16-optimizer-quality-bakeoff.md) | `optbake_*` (quality, not throughput) | done | Medium | RTX 3070 Ti Laptop 8 GB | Lion/AdamW led short-run quality; Prodigy/Schedule-Free needed retuning. |
| [17-gpu-throughput-sweeps](nanolab/17-gpu-throughput-sweeps.md) | `gpu_sweep_{opt,mixer,ffn}.json` | done | Medium | RTX 3070 Ti Laptop 8 GB (124M probe) | Sequential SSM kernels were 24–33× too slow for fair quality comparisons. |
| [18-gpu-maximization](nanolab/18-gpu-maximization.md) | baseline → `gpu_max` / validate / bs32 | done | Medium | RTX 3070 Ti Laptop 8 GB | Memory residency capped the 8 GB 124M run near 25% MFU. |
| [19-chunk-parallel-kernels](nanolab/19-chunk-parallel-kernels.md) | Mamba-2/GDN speedups | done | High | RTX 3070 Ti Laptop 8 GB | Verified chunk-parallel kernels required fp32 accumulation. |
| [20-run128m-long](nanolab/20-run128m-long.md) | `run128m_{2k,10k,20k}` | done | Medium | RTX 3070 Ti Laptop 8 GB | The 10k run at 6e-4 beat the lower-LR 20k continuation on best val. |
| [21-diffusion-lm](nanolab/21-diffusion-lm.md) | `diffusion_phase0` + `diffusion128_block32` | done | Medium | RTX 3070 Ti Laptop 8 GB | Masked diffusion worked once loss targeted clean tokens instead of collapsing to zero. |
| [22-gh200-crossover-50m](nanolab/22-gh200-crossover-50m.md) | GH200 50M n=5 mixer grid | done | High (pair) / Medium (zoo) | Lambda GH200 (ParameterGolf) | Short rankings lie: minGRU overtakes ~1.05M, attention ~12.4M under 50M cosine; 50M mean attention 4.222. |
| [23-locked20-attn-mingru](nanolab/23-locked20-attn-mingru.md) | Locked attn vs minGRU 20M n=5 | done | High | Lambda GH200 (ParameterGolf) | Early 1.05M flip replicates; late flip moves to 14.6M on a 20M cosine. |
| [24-matched20-prefix](nanolab/24-matched20-prefix.md) | 20M stop, 50M cosine | done | High | Lambda GH200 | Independent prefix recovered flips at 1.04M and 12.34M. |
| [25-gh200-bs8](nanolab/25-gh200-bs8.md) | GH200 bs8 n=5 @ 8.192M | done | High (to 7.38M) | Lambda GH200 | No flip by 7.38M; minGRU leads from eval 1 on every seed. |
| [26-matched32-hybrids](nanolab/26-matched32-hybrids.md) | 8 arms bs32 eval_iters=20 50M | done | High | Lambda GH200 | Attn 4.222 ties hybrid_mingru 4.232; Mamba hybrid 4.333 not 4.60. |

## Gemma-metal

| Note | Suite | Status | Confidence | Hardware | Finding |
|------|-------|--------|------------|----------|---------|
| [30-phase0-runtime-baselines](gemma-metal/30-phase0-runtime-baselines.md) | Ollama / mlx-lm floors vs gates | done | Medium | Apple M5 Pro · 20 GPU · 64 GB unified · macOS 26.x | Measured floors were ~56–76 tok/s for E4B and ~12.3 tok/s for 31B. |
| [31-native-decode-speed-ladder](gemma-metal/31-native-decode-speed-ladder.md) | Host-KV → Hot qmv climb | done | Medium | Apple M5 Pro · 20 GPU · 64 GB | Native E4B peaked near 25 tok/s; dispatch overhead became the limiting lever. |
| [32-native-fusion-2026-07-14](gemma-metal/32-native-fusion-2026-07-14.md) | SESSION_RESULTS steps 1–5 | done | Low | Apple M5 Pro (noisy session: load ~4–5; Cursor helpers) | The bf16 producer fuse was the only clear gain, reaching ~24.9 tok/s on E4B. |
| [33-kernel-roofline-overhead](gemma-metal/33-kernel-roofline-overhead.md) | Roofline vs e2e | done | Medium | Apple M5 Pro · ~273 GB/s unified peak | Roughly 77% of quiet decode time was overhead rather than GEMV. |
| [34-mlx-dflash-product](gemma-metal/34-mlx-dflash-product.md) | MLX DFlash product path | done | High | Apple M5 Pro · mlx **0.32.0** (`~/.venvs/dflash32`) | MLX 0.32 DFlash block=5 delivered ~31 tok/s median with exact greedy parity. |
| [35-mlx-dflash-block-tuning](gemma-metal/35-mlx-dflash-block-tuning.md) | block=5 fixed vs adaptive | done | Medium | Apple M5 Pro · mlx 0.32.0 | Fixed block=5 remained best; adaptive sizing did not fix acceptance. |
| [36-native-dflash-parity-accept](gemma-metal/36-native-dflash-parity-accept.md) | exactness / mean_accept / hazards | partial | Low | Apple M5 Pro · gemma-metal Hot 31B + DFlash draft | Mini gates passed, but honest 31B acceptance and speed remained below MLX. |
| [37-golden-token-parity](gemma-metal/37-golden-token-parity.md) | greets + intermediates vs MLX | partial | Low | Apple M5 Pro · mlx golden vs gemma-metal Hot 31B | MLX goldens were exact while native greet16 remained 0/16. |
| [38-clustered-mtp-e4b](gemma-metal/38-clustered-mtp-e4b.md) | Legacy Phase 5 MTP | done | Medium | Apple M5 Pro · E4B Hot + `google/gemma-4-E4B-it-assistant` | 75% acceptance at ~10–12 tok/s proved wiring but not a throughput win. |
| [39-mlx-serve-ttft](gemma-metal/39-mlx-serve-ttft.md) | prompt-cache + SSE + 3-turn TTFT | done | Medium | Apple M5 Pro · `serve_dflash.py` MLX track | Decode held ~36 tok/s, but cached-prefix tax still dominated short-turn TTFT. |
| [40-ddtree-frontier](gemma-metal/40-ddtree-frontier.md) | Modeled defer | done | Low | Apple M5 Pro · modeled from mlx-0.32 verify microbench | Modeled 31B MLX DDTree gain was ≤~1.02×, so the path was deferred. |
| [41-audit-deep-2026-07-14](gemma-metal/41-audit-deep-2026-07-14.md) | Cross-cutting accept≈0 audit snapshot | done | Low | Apple M5 Pro · 20 GPU · 64 GB · ~273 GB/s | The audit localized accept≈0 to GEMM residual algebra; MLX remained the measured product path. |

## Arch-metal

| Note | Suite | Status | Confidence | Hardware | Finding |
|------|-------|--------|------------|----------|---------|
| [50-arch02-metal-native-train](arch-metal/50-arch02-metal-native-train.md) | Soft/FA_TILED BPB + ms/step gates | partial | Medium | Apple M5 Pro, 20-core GPU, 64 GB unified memory, macOS 26.5.2 | Row FA gates at 56.6 ms/~72k tok/s; FA_TILED + WSD reached EMA BPB 1.8828 at 100k. |
| [51-m5-128m-optimizer-funnel-preflight](arch-metal/51-m5-128m-optimizer-funnel-preflight.md) | Exact-128M Polar funnel + champion | done | Medium–High | Apple M5 Pro | Polar Muon @ lr 0.05 won equal-data funnel; champion EMA ~2.01 @2k; long20k 1.8155. |
| [52-context-crossover-metal](arch-metal/52-context-crossover-metal.md) | Attn vs Mamba-2 vs MinGRU tok/s | partial | Low | Apple M5 Pro | Early note claimed ~238k SSM tok/s; audited as stub-inflated pending honest E2E. |
| [53-loop-research-soft-fatiled-2026-07-23](arch-metal/53-loop-research-soft-fatiled-2026-07-23.md) | Soft FA_TILED / Polar / mingru / **16M Soft 20k** | partial | Medium–High | Apple M5 Pro | **16M Soft 20k EMA 1.7591**; Soft attn sota 20k still 1.897; Polar Soft closed. |

## The project story

### From a local proxy to a disciplined ladder

The notebook begins with a constraint rather than a grand architecture claim: useful choices had to be made on a single RTX 3070 Ti before the intended 8×H100 sprint environment was available. The local `sota` proxy compressed the problem to 4 layers, width 128, sequence length 256, and 4,096 training tokens per step. That proxy could not prove what would win at full scale, but it could reject weak ideas cheaply—provided that short screens were promoted through longer stages and judged by calibrated BPB rather than by a convenient training loss.

The first local suite tested ordinary training knobs alongside auxiliary-head size. The surprise was that the most structural-looking change was also the cheapest: reducing the bigram and VE heads to 48 and 24 dimensions improved two-seed long calibrated BPB from 2.093 to 2.066 and reduced the export from about 1.42 MB to 1.33 MB. A follow-up tried to continue shrinking those heads, but 40/20 and 32/16 were worse even at the short stage. The evidence therefore described a floor, not a monotonic “smaller is better” law. Raising LR to 0.027 led the one-seed mid board, but the absent long stage kept that choice provisional.

Depth produced the first major stage reversal. Six- and eight-layer candidates led at 300 and 1,000 steps, making deeper shapes look attractive if the experiment had stopped early. At 3,000 steps across seeds 1337 and 42, the compact 4L×128 control with the higher LR reached 2.0715 calibrated BPB, essentially tied with the 8L×128/MLP×2.6 candidate at 2.0728. The tie was broken decisively by cost: the compact model was about 1.7× smaller and 1.8× faster per step. The project did not conclude that depth never helps; it concluded that this proxy did not justify paying for it, and preserved a separate H100 depth suite for the regime where that decision actually belongs.

The architecture ladder then moved from training controls to value flow and attention structure. Gated attention, value residual, KV compression, VE placement, and recursive sharing entered the same short→mid→long process. The long result was not the simple additive story one might expect. `gated_value_resid` ranked first at 1.984742, but `value_resid` alone was only 0.002749 BPB behind—within the suite’s seed-noise boundary—while gated attention alone fell to 2.088671. Value residual carried the supported gain; gating was safe only in combination. Recursive sharing made a very small artifact but a disastrous 2.851 short calibrated BPB, showing that parameter reuse could destroy effective depth specialization. The follow-up’s XSA1 hint at 2.6938 never survived promotion because all promoted mid and long runs returned code 1. Holding the prior champion was therefore a scientific decision, not a lack of ambition.

Two branches remained deliberately unresolved. APRDH asked whether carefully routed adaptive reuse could succeed where naive recursive sharing failed, but missing `apply_rotary_emb` crashed the comparable rows and an interrupted run only showed raw-byte BPB descending to about 3.72. The H100 conductor encoded depth, TTT×optimizer, and QAT×export suites without pretending that definitions were results. Together these notes mark the boundary of the local training evidence: a strong ablation method, a supported value-residual result, and explicit unexecuted votes at target scale.

### Nanolab: making the comparison visible

Nanolab restarted from first principles. A CPU character smoke and a 6L×384 TinyStories run established clean descent, stable logs, and working precision/optimizer paths. The next 124M FineWeb run added the measurement contract—validation loss, tok/s, MFU, parameters, and memory behavior—because later systems work would be impossible to interpret if “faster” and “better” were allowed to collapse into one number.

Mixer experiments then climbed a budget ladder. At roughly 461K TinyStories tokens, minGRU narrowly led attention, while slow Mamba-2 and GDN paths already exposed very low MFU. At 2.048M matched FineWeb tokens, minGRU led at 5.837 best validation loss; GDN and Mamba-2 also finished ahead of attention’s 6.073, with MLA last at 6.156. The observation was a low-budget ranking. Calling it recurrent “inductive bias” was explicitly interpretation, because one seed and unequal kernel efficiency left alternative explanations.

The 8.192M extension changed the story. Attention trailed minGRU by 0.182 loss at 0.8M tokens, by 0.075 at 4.1M, and by only 0.005 at 6.6M. Between 6.6M and 7.4M the sign flipped; at 8.2M attention led by 0.019. Two further pairs preserved the shape. A smaller 6L×384 pair crossed near step 1,000, while a longer 124M pair crossed near step 900 and eventually opened about a 0.32 gap. The durable laptop result was not a universal crossover token count. It was that a short-run winner could become a long-run loser, and that model/budget selection had to be treated jointly.

The GH200 50M n=5 grid then showed that even the *token* of the flip is recipe-dependent. On a matched attention/minGRU pair (bs32, `eval_iters=20`, 50M cosine) minGRU overtook at ~1.05M and attention overtook for good at ~12.4M (per-seed late 12.03–12.58M); an independent 20M prefix recovered 1.04M and 12.34M. Inside suite 14’s 6.6–8.2M window minGRU still led by 0.13–0.18 at bs32. A short 20M cosine moved the late flip to 14.6M. GH200 bs8 n=5 never flipped by 7.38M and started with minGRU ahead. Shipping “7M crossover, replicated” would have been a false reading. At matched bs32 / 50M, attention 4.222 ties last-2-attention minGRU 4.232; the mixed-batch zoo had underrated the Mamba hybrid.

The optimizer work reinforced metric separation. In a matched 4.096M-token quality bakeoff, Lion and AdamW led at 5.557 and 5.565, Muon followed at 5.643, and default Prodigy soft-diverged. In the fixed-shape throughput probe, AdamW was cheaper per step and Muon paid a Newton–Schulz tax. Those results do not contradict each other: they answer quality-at-recorded-defaults and execution-cost questions. Nor did either justify ranking optimizers without tuning; Schedule-Free and Prodigy were evidence about these settings, not their theoretical ceiling.

The mixer throughput probe uncovered a more severe confound. Attention ran near 7.9K tok/s and MLA near 9.3K, while sequential Mamba-2 and GDN managed only 333 and 238 tok/s. A fair wall-clock bakeoff was impossible at a 24–33× kernel disadvantage. Verified chunk-parallel scans raised Mamba-2 to 3,224 tok/s, raised GDN to 482 tok/s in the first port and later about 1.6K on the vectorized WY path, and turned a bs16/ctx1024 GDN OOM into a 1,100 tok/s run at 4.0 GB. Output and input-gradient checks, non-divisible sequence tests, and fp32 accumulation were part of the result; a fast but numerically drifting kernel would not have counted.

GPU maximization told a similar systems story at the whole-run level. The baseline could show a plausible loss curve while spilling beyond 8 GB, using about 14% GPU and taking roughly 18 seconds per step. Fused linear cross-entropy, batched Muon, checkpointing, TF32/flash SDPA, a GPU-resident loader, and a 0.92 memory fraction kept the workload resident. The bs32 path reached about 13.7K tok/s and 25.5% MFU. The measured lesson was not that 25.5% is a universal ceiling; it was that residency and telemetry transformed this specific laptop run.

Long training then complicated the intuitive “more steps wins” story. The 10k recipe at 6e-4 reached a reliable best validation loss of 3.621. The 20k arm used a lower LR and matrix LR, reached its best around step 4,500, then regressed toward 3.79. Because horizon and schedule changed together, the notebook refuses to blame duration alone. Finally, the diffusion branch transferred an autoregressive checkpoint to masked bidirectional denoising, moving TinyStories validation perplexity from 19.49 toward about 8.2. Its most useful failure was a zero loss caused by targeting the masked input rather than clean tokens: instrumentation turned an apparently perfect number into an immediate bug signature.

### From CUDA experiments to Metal product gates

The M5 inference work began with honest baselines: context 4,096, generation 128, greedy temperature zero, and thinking disabled. mlx-lm delivered about 76 tok/s on E4B; Ollama delivered roughly 56 tok/s on E4B and 12.3 tok/s on 31B. Those measurements became product floors and prevented native work from celebrating improvements that still lost to an installed runtime.

The Rust/Metal native ladder was nevertheless substantial. Removing host KV densification and per-dispatch synchronization, keeping weights and KV hot on GPU, packing asynchronous work, fusing MLP/KV operations, and testing MLX-style Q4 paths raised E4B from 4.78 to roughly 25 tok/s. But the final qdot, bfloat2, and Interleaved4 peels were flat or regressive, and 31B remained 6.83 tok/s. A noisy fusion session found one modest win—bf16 producer fusion at 24.90 E4B tok/s—while dual-norm, mid-commit, K+V, and argmax changes mostly held correctness without producing a stable speed shift.

The roofline measurement reversed the next engineering priority. Hot-resident Q4 GEMV kernels were already delivering about 62–100% of the M5 Pro’s roughly 273 GB/s peak depending on shape. At 21.5 tok/s, streaming approximately 2.86 GB of weights accounted for about 10.6 ms of a 46.5 ms token; roughly 36 ms, or 77%, remained in encoding, barriers, casts, and approximately 780 dispatches. An earlier “20% peak, 4× GEMV headroom” narrative had measured upload behavior and was corrected. Kernel rewriting could offer perhaps 1.3× on common shapes, not the required 2–4×. Fewer dispatches and multi-token verification became the credible path.

MLX 0.32 DFlash supplied that path. A quantized draft proposed blocks and the 31B target verified them exactly, preserving the greedy token stream. Across eight prompt types, fixed block=5 produced a 31.7 tok/s median versus roughly 12.5 plain, with every prompt above 15 tok/s and all but creative prose above 25. Version 0.32 itself mattered: NAX acceleration improved the measured block-8 median from 18.64 to 27.77 tok/s. Block tuning could not erase the prose weakness; block=3 improved prose only about 3.7%, while code still wanted block=5. The evidence pointed to draft acceptance, not adaptive block policy.

The native DFlash port did not inherit MLX’s result merely by reproducing its algorithm. Synthetic steered mini gates passed, but real 31B runs remained near 1.91 tok/s, acceptance around 0.77–1.0 rather than MLX’s roughly 3, and streams could mode-lock so an exactness pass became vacuous. Golden comparison made the failure concrete: MLX greedy and DFlash matched exactly, while native `greet16` matched 0 of 16 tokens and collapsed to repeated high IDs. A source-first audit localized one historical mean-acceptance-zero failure to missing Gemma 4 dual-norm residual algebra and an unused `layer_scalar` in GEMM verification. Landing that fix improved the snapshot but did not clear parity or speed gates; stale and NaN-derived claims were explicitly superseded.

Other speculative branches clarified what acceptance does and does not buy. Clustered E4B MTP achieved 75% acceptance but only 10–12 tok/s because verification still called the backbone per draft token. DDTree’s CPU core passed 19/19 tests, yet the measured MLX M>1 verification curve produced a modeled 31B gain of at most about 1.02×; the M=10 cost cliff consumed the extra accepted tokens. Both were deferred, not declared impossible. Meanwhile the MLX server held decode near 36 tok/s and grew its prompt cache across three turns, but TTFT stayed around 363–370 ms because each short turn still prefetched 15–23 new tokens. Product performance had split again into decode, acceptance, parity, and prefill.

### The parity frontier

The final Metal-native training suite closes the loop between architecture and systems. A Rust + Metal 4 implementation of `arch_02` separated the fastest row-wise flash path from the best-quality tiled path. Row FA sustained 56.6 ms per step and roughly 72K tok/s. FA_TILED cost about 69 ms per step but, with Soft-split clipping and horizon-appropriate warmdown, reached 1.8969 EMA BPB at 20k and 1.8828 at 100k. Faster GEMM accumulation reduced binders but regressed 3k quality; relaxed hazards reached NaN by step 3. The decision was to keep separate speed and quality operating points while 16M scaling, TensorOps flash backward, and CUDA parity remain open.

Across all three tracks, the project’s durable method is more important than any single winner: stage cheap ideas, preserve failures, use the metric that matches the decision, and require parity before interpreting speed. The notebook repeatedly found that plausible shortcuts—short-stage rankings, raw throughput, high acceptance, faster kernels, or longer schedules—could point in the wrong direction when their hidden constraint was finally measured.

## Decision timeline

| Date / order | Decision | Evidence | Consequence |
|--------------|----------|----------|-------------|
| 2026-03-27 | Keep lean aux 48/24 | 2.066 vs 2.093 calibrated BPB; smaller export | Made lean aux the local base |
| 2026-03-28 | Stop shrinking aux; treat LR-up as provisional | 40/20 and 32/16 lost short; LR-up led mid; no long | Preserved floor and open confirmation |
| 2026-03-31 | Prefer compact 4L×128 proxy | Long BPB tie plus ~1.8× step and ~1.7× size advantage | Deferred real depth vote to H100 |
| 2026-04-01 | Keep value residual; hold combo champion cautiously | 1.9847 combo vs 1.9875 value-only; gating-only 2.0887 | Made value flow the supported architecture lever |
| 2026-04-02 | Reject promotion from failed follow-up | XSA1 short hint; all promoted mid/long rc=1 | Held suite-04 champion |
| 2026-06-15 | Use token budget as an experimental axis | minGRU led at 2M; attention crossed at 6.6–7.4M | Triggered two crossover replications |
| 2026-08-21 | Treat the 7M overtake as recipe-local, not replicated | GH200 n=5: flips at 1.05M and 12.4M; suite-14 window still minGRU-led | Locked a 20M attn-vs-minGRU follow-up; forbade mixing bs8/bs32 tables |
| 2026-06-15 | Fix SSM kernels before wall-clock comparisons | 238–333 tok/s sequential paths | Built verified chunk-parallel scans |
| 2026-06-15 | Optimize residency as a stack | Baseline spill/14% util vs 13.7K tok/s/25.5% MFU | Enabled laptop long runs |
| 2026-06-18 | Keep 10k recipe as reliable long-run result | Lower-LR 20k arm regressed after early best | Avoided a false “more steps” conclusion |
| 2026-07-13 | Judge native inference against live floors | mlx E4B ~76; Ollama 31B ~12.3 | Locked ≥15/≥25 and parity gates |
| 2026-07-13 | Stop treating GEMV as the main bottleneck | Kernels 62–100% peak; modeled overhead ~77% | Shifted effort to fusion/speculation |
| 2026-07-13/14 | Ship MLX 0.32 DFlash block=5 path | 31.7 tok/s median; exact greedy parity | Established the 31B product track |
| 2026-07-14 | Keep native DFlash partial | Low honest acceptance, mode-lock, 0/16 golden prefix | Prioritized algebra and intermediate parity |
| 2026-07-14 | Defer DDTree on 31B MLX | Modeled ≤1.02× at measured verify costs | Retained core for cheaper verify/E4B |
| 2026-07-11–13 | Separate Metal training speed and quality gates | Row FA 56.6 ms; FA_TILED/WSD 1.8828 BPB | Kept two operating points and open parity work |

## Reversals and surprises

1. **Less auxiliary capacity helped—until it did not.** Moving to 48/24 improved BPB and size, but further cuts to 40/20 and 32/16 lost at the short stage.
2. **Depth won the screens and lost the decision.** Deeper models led short and mid, then the compact LR-up control tied them at long horizon while being markedly faster and smaller.
3. **Gating was not an independent win.** Gated attention alone was about 0.104 BPB worse than the champion; value residual supplied nearly all of the combination’s gain.
4. **A promising XSA hint became no result.** `gated_value_resid_xsa1` led the short board, but promotion produced return-code failures rather than confirmation.
5. **The low-budget mixer champion changed with scale — and the flip token moved with the recipe.** minGRU led attention by 0.182 at 0.8M on the 3070; attention overtook between 6.6M and 7.4M there. On GH200 n=5 the same pair flipped at 1.05M then 12.4M; the 7M window did not replicate.
6. **Optimizer quality and optimizer speed disagreed.** Lion/AdamW led the short quality board, while AdamW was cheapest per step and Muon’s systems case depended on batching and convergence.
7. **Loss hid the GPU failure.** The spilling baseline could appear numerically healthy while utilization, power, step time, and memory showed a practically unusable run.
8. **Longer training was not the winning recipe.** The lower-LR 20k arm reached an early best then regressed; schedule changed with horizon, preventing a simplistic duration claim.
9. **A zero loss was evidence of a bug, not success.** Diffusion training targeted masked inputs until logs exposed the trivial objective; clean tokens were the required target.
10. **The kernel roofline contradicted the optimization narrative.** Direct GEMV measured 62–100% of peak and corrected an earlier upload-derived claim of about 20%; dispatch dominated end-to-end time.
11. **Higher speculative acceptance did not guarantee speed.** Clustered MTP accepted 75% yet ran below greedy because verification remained per-token.
12. **Adaptive block sizing could not rescue prose.** The best prose-specific change was only about 3.7%, while code preferred the fixed product block.
13. **An exactness pass could be vacuous.** Native streams mode-locked to few unique tokens, so `exact_vs_greedy=true` did not imply parity with MLX goldens.
14. **Fewer binders did not guarantee better training.** GEMM accumulation improved speed and binder count but regressed Soft-clipped BPB; relaxed hazard barriers produced NaNs.

## Open questions / next experiments

- Complete the two-seed long stage for `sota_lean_followup`; the current LR-up choice is only a mid-stage result.
- Diagnose the promoted-stage return-code failures in `sota_arch_followup_value_resid`, then retest the short XSA1 hint against the held long champion.
- Restore `apply_rotary_emb`, rerun APRDH through packaging, and include RADA/DeltaHybrid in the same summary before making any architecture comparison.
- Execute the defined H100 depth, TTT×optimizer, and QAT×export suites in the intended sprint environment.
- For native 31B DFlash, match MLX draft proposals/intermediates, eliminate mode-locked streams, and clear exactness, acceptance, and ≥15/≥25 tok/s together.
- Extend native golden capture past the current 0/16 greet mismatch to target-hidden, fc-out, h-context, and proposed-token checkpoints.
- Finish the Metal-native 16M 20k run and production multi-block TensorOps flash backward/LSE path; recheck quality parity against the CUDA reference.
