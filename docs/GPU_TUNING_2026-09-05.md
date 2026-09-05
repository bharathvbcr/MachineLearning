# GH200 tuning sprint — 2026-09-05

**Question.** The E28–E35 program (`docs/architecture-review-2026-09-04/inputs/04-efficiency-reconciliation-memo.md` §7)
was priced from elapsed times of past suites divided by their tenancy. Before
spending ~$110 of GH200 time on it, measure the knobs that set how long it takes,
and separate the ones that change only speed from the ones that change what a
board measures.

**Box.** NVIDIA GH200 480GB (94.5 GiB usable to the runner), aarch64, torch 2.7.0
/ CUDA 12.8, $2.29/h. Measured dense bf16 ceiling **752.8 TFLOP/s** (8192³ matmul,
50 iterations) — used as the MFU denominator below rather than a spec number.
No throttling during the sprint: SM clocks pinned at 1980/1980 MHz, 542 W of a
900 W limit, 56 °C, `clocks_throttle_reasons.active = 0x0`.

**Method.** Two independent harnesses, which agree where they overlap (attention:
130.4K tok/s from the arm sweep, 131.0K from the tenancy driver):

* `nanolab.sweep_gpu arm` — one row per `crossover_replicate` ARM, configured
  through `job_config` so hybrids and per-arm overrides are priced exactly as the
  suite runner would launch them. Synthetic in-GPU batches, so a row is compute,
  not the dataloader.
* `scripts/tune_tenancy.py` — N concurrent processes of one arm over a **shared
  wall-clock window**, so a high-tenancy row cannot be flattered by start-up skew.

Nothing here changes a recipe. Where a speed-up would change one, that is stated
as its price.

---

## The headline

At the 50M board shape a 124M model reaches **13.8% MFU** (104 of 752.8 TFLOP/s)
with the GPU busy essentially 100% of the wall clock — the kernels are small, not
the GPU idle. That single fact explains every result below.

**Tenancy is not one number.** `workers` is a recipe field every board sets by
hand — 2 for the ladder, 3 for E18/E19/E20, 4 for the recall grid — and the knee
had never been measured. Thirteen of the fifteen token-matched suites on disk ran
at 2 or 3. What the measurement shows:

1. **Without MPS, concurrent processes time-slice the GPU rather than sharing
   SMs.** For an arm that already saturates (attention, the minGRU hybrids)
   tenancy ≥ 2 is a flat **~9% loss**. For a *dispatch-bound* arm it is a
   **1.53× gain**, because time-slicing recovers the idle gaps between one job's
   many small kernel launches. GDN is the dispatch-bound case (2.7% MFU).
2. **There is no MPS daemon running on the box, and starting one changes the
   answer** for the saturating arms too.

So the correct rule is not "use tenancy N" but: *tenancy pays in proportion to
the GPU idle time a single job leaves behind.* An arm's MFU predicts its sign.

---

## The five findings, in order of what they change

### 1. `torch.compile` is 1.94x and it is switched off everywhere

| arm | eager | compiled | speed-up | compile time |
|---|---:|---:|---:|---:|
| `attention` | 113.6K tok/s | 220.9K | **1.94x** | 31 s |
| `mingru` | 91.8K | 179.6K | **1.96x** | 29 s |
| `moe_e8k1` | 55.9K | 60.5K | 1.08x | 0 s |
| `gdn` | 26.0K | 25.2K | 0.97x | 1 s |
| `hybrid_mingru8_attn4` | 98.1K | 98.1K | 1.00x (fell back) | 29 s |

`compile=False` is hardcoded in `job_config` and `current_recipe`, and `train.py`
additionally refuses to compile anything but a pure-attention stack. Both date
from an Inductor stall on aarch64 that torch 2.7.0 does not reproduce.

The old gate was wrong in both directions. It *excluded* a pure minGRU stack,
which gains the most of anything measured; and it would have *admitted* the
hybrids, which blow Dynamo's recompile limit (`hit config.recompile_limit (8)`,
`last reason: GLOBAL_STATE changed: grad_mode` — the eval/train switch) and fall
back to eager. The condition that actually predicts the win is **one mixer kind
throughout**, which is what the gate now tests.

This is a numerics change: Inductor fuses and re-associates. `compile` is now a
recorded recipe field (`CROSSOVER_COMPILE`, default off), so a compiled run
cannot pool with an eager one by accident. What it costs in nats is being
measured by `scripts/tune_compile_board.sh` at the time of writing.

### 2. Tenancy's sign is predicted by the arm's MFU

The repo treats `workers` as a per-board choice and thirteen of the fifteen
token-matched suites on disk record 2 or 3. It is not a choice that can be made
once: with no MPS daemon, concurrent processes time-slice, which *costs* ~9% for
an arm that already saturates the device and *recovers* idle launch gaps for one
that does not. Ranked by MFU the relationship is monotone:

| arm | MFU at t=1 | best tenancy, no MPS | gain |
|---|---:|---|---:|
| `gdn` | 2.7% | 3 | **1.53x** |
| `moe_e8k1_raw` | 4.5% | 4 | **1.33x** |
| `w384_mingru_lr40` | 7.0% | 1 | 1.00x |
| `w384_attention_lr80` | 7.8% | 1 | 1.00x |
| `mingru` | 13.0% | 1 | 1.00x |
| `hybrid_mingru8_attn4` | 13.2% | 1 | 1.00x |
| `attention` | 13.8% | 1 | 1.00x |

Two controls: the sweep run in **descending** tenancy order reproduces the
ascending one (130.5K at t=1, ~119K at 2–6), so this is not an ordering or
warm-up artifact; and no throttling occurred (SM clocks pinned at 1980/1980 MHz,
542 W of 900 W, 56 °C, `clocks_throttle_reasons.active = 0x0`).

**The recall grid is the other regime.** At ~9.5M parameters and seq 31 the same
cell takes 613 s at `--workers 1` and 229 s at 4 — a **2.68x** gain. So
`mqar_suite`'s existing default of 4 for E28/E32 is right, and the 124M result
must not be generalised to it.

### 3. MPS refunds the tenancy tax, without touching any recipe

`workers` is a recipe field (`current_recipe`), so a board reusing an existing
directory cannot change it. MPS is not a recipe field, and it is worth most
exactly where tenancy is locked:

| arm | t=2 no MPS → MPS | t=3 no MPS → MPS |
|---|---|---|
| `attention` | 118.8K → 142.5K (**1.20x**) | 119.0K → 144.5K (**1.21x**) |
| `hybrid_mingru8_attn4` | 100.7K → 122.6K (**1.22x**) | 100.7K → 123.8K (**1.23x**) |
| `gdn` | 44.2K → 51.8K (1.17x) | 45.1K → 57.4K (**1.27x**) |

Against *serial* the same daemon is worth only ~1.10x, because serial was already
the best non-MPS configuration. It needs no code change: `launch` copies
`os.environ` into every worker, so exporting `CUDA_MPS_PIPE_DIRECTORY` suffices.

Two cautions. Verify the daemon is **serving** (`echo get_server_list |
nvidia-cuda-mps-control` returning a pid) rather than merely present — `pgrep`
races the daemon's fork, and that race mislabelled a stage of this sprint before
the gate was made fail-closed. And MPS weakens fault isolation.

### 4. The sampler silently switches token streams on free VRAM

`data.should_gpu_resident` returns the GPU-resident path only when free VRAM
exceeds **9.27 GiB** at `Batcher` construction. That path seeds a **CUDA**
generator; the memmap fallback seeds a **CPU** one. Measured, same seed and split:

```
free 93.8 GiB → gpu_resident → first batch [247, 82, 2057, 12, 44708, 1410, 290, 262]
free  7.4 GiB → memmap       → first batch [393, 7734, 416, 4737, 257, 1545, 11, 1641]
```

Construction happens *before* the model is built, so which path a worker takes
depends on how many co-resident workers have already allocated — a race, not a
setting. This is the one place tenancy is not recipe-neutral, and it fails
silently: the fallback needs *less* VRAM, so it converts an OOM into a run that
looks fine and trains on different tokens.

### 5. Two knobs that are simply mis-set for a 94 GiB card

**Fused cross-entropy.** `crossover50m` sets `fused_ce=True, fused_ce_chunks=16`,
which exists to avoid materialising a 16384 x 50304 logits tensor — a real
constraint on the 8 GB card this repo started on. Unfused is **1.29x** faster on
attention (145.6K vs 113.1K) and 1.24x on the hybrid, for +10.8 GB.

**GDN chunk width.** `mixer_chunk` is 32; width 128 runs **1.98x** faster
(53.2K vs 26.9K) for +7.7 GB. Adoptable only if the width changes the
floating-point order and not the operator — the first check of that was vacuous
(zero-initialised output projection made both sides all zeros) and is being
redone by `scripts/tune_gdnchunk2.py`.

### What was ruled out

* **Deferring the eval loop's per-iteration `.item()`**: 1.00x. Eval is not
  sync-bound, and the idea is dead. (297.9 → 297.2 ms per eval.)
* **Tenancy rescuing itself on real jobs.** The synthetic harness omits evals,
  checkpointing and start-up, which biases it *against* tenancy. Measured
  end-to-end on real suite jobs it does not recover: at 20M tokens, 844 s at
  `--workers 1` against 871 s at 3; at 5M, 549 s against 555 s.

---

## Does any of this change what a board measures?

The gate for every recipe-neutral claim above. Same arms and seeds, 5M tokens,
one directory per condition; **a1 and a2 are the same configuration run twice**,
so their difference is the floor any other difference must clear.

| comparison | mean abs `final_val` difference | max |
|---|---:|---:|
| a1 vs a2 — identical config, identical seeds | **0.0014** | 0.0031 |
| a1 vs t3 — `--workers` 1 against 3 | **0.0012** | 0.0022 |

Tenancy's effect on the reported number is *smaller than re-running the same job*.
That settles it: tenancy is recipe-neutral in practice, not just in principle.

It also gives the repo a number it did not have. The trainer is **not run-to-run
deterministic** — two identical jobs land ~0.0014 nats apart in `final_val` (max
0.0031 over ten runs), which is what non-deterministic cuBLAS/cuDNN backward
kernels cost. For scale, the paired gap the architecture review reports for the
8+4 hybrid is **-0.0176 nats**, about 12x that floor, so its conclusion is not at
risk; but any future per-seed claim below ~0.003 nats is indistinguishable from
running the same job again. Measured at 5M tokens; the floor at 50M is unmeasured.

The MPS condition (`m3`) is still missing: it was skipped twice because the
daemon was not serving. It was *skipped rather than mislabelled*, which is the
behaviour the gate was built for, and it is being rerun.

## What eval_iters buys, and what it costs

Eval is ~16% of a 50M run (61 evaluations x 20 iterations x a forward pass), and
`eval_iters` is a recipe field. Two trained checkpoints at the same seed, scored
on the same 200 val batches:

* per-batch SD 0.0736 nats; **correlation between the two arms 0.9957** — because
  `Batcher` seeds val with `cfg.seed + 1`, both arms see identical batches
* the pairing therefore removes ~90% of the noise: the paired difference has
  SD 0.0068 against 0.0736 for a single arm

| `eval_iters` | SE of one arm's mean | SE of the **paired difference** | eval cost |
|---|---:|---:|---:|
| 5 | 0.0329 | 0.0031 | 25% |
| 10 | 0.0233 | 0.0022 | 50% |
| **20 (current)** | 0.0165 | **0.0015** | 100% |
| 40 | 0.0116 | 0.0011 | 200% |

Note the first column: a *single* arm's mean at `eval_iters=20` has SE 0.0165,
which is the same size as the 8+4 effect. Absolute `final_val` is far noisier
than the paired difference — which is exactly why the boards pair, and is
independent support for that choice.

At `eval_iters=5` the paired difference is still resolved at ~5.7 SE and eval
drops from 16% of a run to 4%. That is a recipe change (it moves every board's
markers), so it is a decision, not a recommendation.

---

## What to change, and what it costs

Ordered by (value × confidence). "Recipe-neutral" means the loss curve is
provably unchanged — safe even for boards that pool with existing runs.

### 1. Start an MPS daemon. It is the only free ~20% here — recipe-neutral

`workers` is a **recipe field**: `current_recipe()` records it and `lock_recipe`
refuses a directory whose recorded tenancy differs. So for the boards that must
reuse an existing directory — the E27 ladder (locked at 2), E30a into
`crossover50m_ratioplace32` (locked at 3) — tenancy cannot be changed at all.

MPS is **not** a recipe field. It is a scheduling layer outside the training
process, and it is exactly where those boards are losing the most:

| tenancy the recipe locks | without MPS | with MPS | gain |
|---|---:|---:|---:|
| 2 (E27 ladder) | 118.8K tok/s | 142.5K | **1.20x** |
| 3 (E18/E19/E20/E30a) | 119.0K | 144.5K | **1.21x** |

Against *serial* the same daemon is worth only ~1.10x, because serial was already
the best non-MPS configuration. The 20% is the tenancy tax being refunded.

Operationally it needs no code change: `launch` copies `os.environ` into every
worker (`crossover_replicate.py:1674`), so exporting `CUDA_MPS_PIPE_DIRECTORY`
in the launching shell is enough. Two cautions: the daemon must be verified
*serving* (`echo get_server_list | nvidia-cuda-mps-control` returning a pid —
`pgrep` races its fork, which mislabelled a stage of this sprint), and MPS
weakens fault isolation, so a client that dies hard can take the server with it.

### 2. Pick tenancy per arm, from the arm's MFU — recipe-neutral, new dirs only

The sign of the tenancy effect is predictable from how much GPU idle time a
single job leaves behind:

* **Saturating arms** (attention, the minGRU hybrids; ~13–14% MFU but the GPU
  busy ~100% of the wall clock): tenancy ≥ 2 without MPS is a flat ~9% loss.
* **Dispatch-bound arms** (GDN, 2.7% MFU): tenancy 3 is a **1.53x gain** even
  without MPS, because time-slicing recovers the gaps between its many small
  kernel launches.

### 3. VRAM: budget from the real number, not the bench number

A bench row understates a real job by ~4.1 GiB, paid **per co-resident job**:

| term | size | why the bench misses it |
|---|---:|---|
| peak reserved (measured) | per arm, table A | — |
| CUDA context | **2.24 GiB** | per process, not a PyTorch allocation |
| resident corpus | **1.85 GiB** | `Batcher` copies 497.5M tokens as int32 |

The context figure comes from the tenancy-6 OOM trace: PyTorch reported 15.31 GiB
in use while the process held 17.55 GiB. Bharath's uncommitted `nanolab/vram.py`
anchors (attention 17.0, minGRU 23.3 GiB) reconcile with the bench rows once both
terms are added — independent support for both.

### 4. The sampler switches on free VRAM, and the two paths are not the same run

`data.should_gpu_resident` returns GPU-resident only when free VRAM exceeds
**9.27 GiB** at `Batcher` construction. That path seeds a **CUDA** generator; the
memmap fallback seeds a **CPU** one. Same seed, different token stream.
Construction happens before the model is built, so which path a worker takes
depends on how many co-resident workers have already allocated — a race, not a
setting. This is the one place where tenancy is *not* recipe-neutral.

---

## What this changes for the interrupted E27 ladder

E27's state is unchanged from the handoff and was re-verified at the start of this
sprint: `crossover_ladder1536` has 2 jobs stuck at `running` (orphaned 06:05Z,
both with `ckpt.pt`, so they resume) and 8 `pending`; `crossover_ladder_probe1536`
has 4 done and 2 failed to CUDA OOM.

The sprint reproduced E27's failure directly. At the board shape,
`hybrid_mingru8_attn4` at tenancy 6 OOMed 6/6 workers, and `attention` at tenancy
6 reached **90.5 GB reserved of a 94.5 GiB card** — with no margin for the CUDA
context and resident corpus each real job also carries. The two w1536 minGRU
probe cells that failed were 45.9 + 48.4 GiB co-resident at `workers: 2`; that is
the same arithmetic one width up.

The repair the handoff prescribes is right, and the measurement now says *why*:

* The probe rerun belongs at `--workers 1` in a new directory — the existing
  probe dir records `workers: 2` and `lock_recipe` refuses a different tenancy.
  At width 1536 tenancy 1 is not a sacrifice: two minGRU jobs do not fit at all.
* The attention half must stay at `workers: 2`, because `crossover_ladder1536`
  records that and the recipe is locked. That costs ~9% against serial for a
  saturating arm, which is a price already paid, not a new decision.
* The orphaned `running` jobs must be reset to `pending` before any relaunch:
  `claim_job` only claims `pending`, so they are invisible to a new worker
  regardless of tenancy.

The tenancy finding does **not** retroactively invalidate any board. `workers`
changes throughput, and — subject to the sampler caveat below — not the loss
curve. What it invalidates is the arithmetic that converted elapsed time into
per-job GPU-minutes by dividing by tenancy: for a saturating arm at `workers: 3`
that divisor overstates per-job cost by ~2.7x.

---

## What is measured, what is inferred, and what is not verified

**Verified** (a number in this document came from a run on the box today):
the per-arm cost table; the tenancy curves with and without MPS; the absence of
thermal or power throttling during the sprint; the 752.8 TFLOP/s bf16 ceiling;
the reproduction of E27's OOM class at tenancy 6; that 13 of 15 token-matched
suites on disk record `workers` of 2 or 3; that the suite runner copies its
environment to workers, so MPS needs no code change.

**Inferred** (arithmetic on measured numbers, not itself observed):
the re-priced program totals; the claim that the ~9% context-switch tax roughly
cancels start-up overlap at 50M but not at 5M; the ~4.1 GiB gap between a bench
row and a real job's VRAM.

**Not verified — and the limits that matter:**

* **The cost model uses synthetic batches.** `bench_gpu.synth_batch` returns
  `y == x`, a target the model can read off its own input, so loss collapses to
  ~0 within the warm-up. For dense arms that changes nothing about timing, but it
  is why `moe_e4k1` and `moe_e4k1_raw` — identical compute — differed by 17%:
  a router that trains dispatches differently from one that cannot. The MoE rows
  are therefore re-measured on real corpus batches, and only those should be used
  to price E29.
* **The tenancy driver runs one arm at a time, in lockstep, with no evaluation,
  checkpointing or data loading.** Real jobs are staggered and heterogeneous and
  leave GPU-idle gaps that a co-resident job can fill. That biases the synthetic
  measurement *against* tenancy; the validation stage measures real wall clock at
  1 vs 3 workers to bound it, at a token budget where start-up is a much larger
  fraction than it is at 50M — so its ratio must not be read as the 50M ratio.
* **45-second windows.** Long enough for 357 attention steps and 76 GDN steps.
  Run-to-run spread across workers was 1.00–1.07x; no row is a single sample of
  a noisy quantity, but none is a confidence interval either.
* **Absolute throughput drifted ~13% between harnesses, and the ratios are what
  survive.** The first arm sweep read attention at 130.4K tok/s and the tenancy
  driver at 131.0K; the later `tune_compile` and `tune_fusedce` probes read
  113.6K and 113.1K on the same `bench()` code path, while a real training job
  running concurrently with this writing reports 131.1K. Every speed-up quoted
  here is a **ratio measured within one probe**, where both arms of the
  comparison saw the same conditions, so the 1.94x, 1.29x and 1.98x figures are
  unaffected. The absolute levels are not interchangeable across probes, and the
  cause — allocator state accumulated across many model builds in one process,
  versus a fresh process — is being re-measured (`drift` stage).

* **One box, one driver version.** Nothing here transfers to the M5 Pro items
  (memo items 10, 12, 13) or to a different CUDA/driver stack.

---

## Appendix: the measured tables

## A. Per-arm cost at the board shape (batch 32, ctx 512, tenancy 1)

| arm | tok/s | ms/step | peak alloc | peak resv | params | 50M job |
|---|---:|---:|---:|---:|---:|---:|
| `attn6_w512` | 286.9K | 57 | 5.7 GB | 5.9 GB | 45.1M | 2.9 min |
| `attn6_w576` | 259.1K | 63 | 6.3 GB | 6.6 GB | 52.9M | 3.2 min |
| `w384_attention_lr80` | 215.2K | 76 | 7.5 GB | 7.8 GB | 40.6M | 3.9 min |
| `w384_attention_lr10` | 214.2K | 76 | 7.5 GB | 7.8 GB | 40.6M | 3.9 min |
| `w384_hybrid_mingru8_attn4_lr40` | 192.5K | 85 | 9.7 GB | 10.3 GB | 46.5M | 4.3 min |
| `w384_mingru_lr40` | 177.3K | 92 | 10.9 GB | 11.6 GB | 49.4M | 4.7 min |
| `attention_novr` | 136.8K | 120 | 14.2 GB | 14.8 GB | 123.7M | 6.1 min |
| `attention_untied_novr` | 135.8K | 121 | 14.6 GB | 15.2 GB | 162.3M | 6.1 min |
| `mingru_x1` | 132.0K | 124 | 15.5 GB | 16.1 GB | 130.7M | 6.3 min |
| `hybrid_mingru_periodic_x1` | 131.4K | 125 | 15.3 GB | 15.8 GB | 128.9M | 6.3 min |
| `hybrid_mingru8_attn4_x1` | 131.2K | 125 | 15.2 GB | 15.8 GB | 128.3M | 6.4 min |
| `attention` | 130.4K | 126 | 14.4 GB | 15.1 GB | 123.7M | 6.4 min |
| `attention_untied` | 129.4K | 127 | 14.9 GB | 15.5 GB | 162.3M | 6.4 min |
| `moe_e1k1_raw` | 118.6K | 138 | 15.1 GB | 15.7 GB | 123.7M | 7.0 min |
| `moe_e1k1` | 118.4K | 138 | 15.1 GB | 15.7 GB | 123.7M | 7.0 min |
| `hybrid_mingru8_attn4` | 110.4K | 148 | 19.0 GB | 19.7 GB | 147.2M | 7.5 min |
| `hybrid_mingru_periodic` | 108.3K | 151 | 19.6 GB | 20.4 GB | 150.2M | 7.7 min |
| `hybrid_mingru10_attn2` | 106.3K | 154 | 20.2 GB | 20.9 GB | 153.1M | 7.8 min |
| `hybrid_mingru_bookend` | 106.2K | 154 | 20.2 GB | 21.0 GB | 153.1M | 7.8 min |
| `hybrid_mingru11_attn1` | 104.4K | 157 | 20.8 GB | 21.6 GB | 156.0M | 8.0 min |
| `mingru` | 102.5K | 160 | 21.3 GB | 22.3 GB | 159.0M | 8.1 min |
| `moe_e4k1` | 73.4K | 223 | 16.9 GB | 19.4 GB | 293.6M | 11.4 min |
| `moe_e4k1_raw` | 62.6K | 262 | 16.9 GB | 19.4 GB | 293.6M | 13.3 min |
| `w1536_attention_lr80` | 61.8K | 265 | 29.6 GB | 32.4 GB | 417.5M | 13.5 min |
| `moe_e8k1` | 55.0K | 298 | 19.1 GB | 25.9 GB | 520.1M | 15.2 min |
| `w1536_mingru_lr20` | 48.6K | 337 | 44.0 GB | 46.2 GB | 558.6M | 17.1 min |
| `w1536_mingru_lr80` | 48.6K | 337 | 44.0 GB | 46.2 GB | 558.6M | 17.1 min |
| `w1536_mingru_lr40` | 48.6K | 337 | 44.0 GB | 46.2 GB | 558.6M | 17.2 min |
| `moe_e8k1_raw` | 42.6K | 385 | 19.2 GB | 25.3 GB | 520.1M | 19.6 min |
| `hybrid_gdn_periodic` | 36.1K | 454 | 22.8 GB | 23.8 GB | 123.8M | 23.1 min |
| `hybrid_gdn_periodic_pub` | 36.1K | 454 | 22.8 GB | 23.8 GB | 123.8M | 23.1 min |
| `gdn_pub` | 27.8K | 588 | 25.6 GB | 26.8 GB | 123.8M | 29.9 min |
| `gdn` | 27.8K | 589 | 25.6 GB | 26.8 GB | 123.8M | 30.0 min |

## B. Aggregate throughput vs tenancy

| arm | MPS | t=1 | t=2 | t=3 | t=4 | t=6 | best |
|---|---|---:|---:|---:|---:|---:|---|
| `attention` | off | 131.0K | 118.8K | 119.0K | 119.0K | 118.9K | **1** |
| `hybrid_mingru8_attn4` | off | 110.8K | 100.7K | 100.7K | 100.8K | **OOM** | **1** |
| `gdn` | off | 29.4K | 44.2K | 45.1K | — | — | **3** |
| `attention` | on | 129.6K | 142.5K | 144.5K | 144.5K | 142.8K | **4** |
| `hybrid_mingru8_attn4` | on | 110.1K | 122.6K | 123.8K | 123.6K | **OOM** | **3** |
| `gdn` | on | 29.9K | 51.8K | 57.4K | — | — | **3** |
| `attention` (descending order) | off | 130.5K | 118.9K | 119.0K | 118.8K | 118.9K | **1** |

## C. torch.compile

| arm | eager | compiled | speed-up | compile time | max abs logit diff |
|---|---:|---:|---:|---:|---:|
| `attention` | 113.6K | 220.9K | **1.94x** | 31 s | None |
| `mingru` | 91.8K | 179.6K | **1.96x** | 29 s | None |
| `hybrid_mingru8_attn4` | 98.1K | 98.1K | **1.00x** | 29 s | None |
| `gdn` | 26.0K | 25.2K | **0.97x** | 1 s | None |
| `moe_e8k1` | 55.9K | 60.5K | **1.08x** | 0 s | None |

## D. Probes

**Sampler path (data.should_gpu_resident)**

- corpus 497,500,000 tokens = 1.85 GiB as int32; the GPU-resident path needs **> 9.27 GiB free** at Batcher construction
- with an empty GPU: `gpu_resident`; squeezed to 7.41 GiB free: `memmap`
- same seed, same split, first batch identical: **False**
- HAZARD: same seed, different tokens -- the sampler that runs depends on free VRAM at construction

**GDN chunk width (`mixer_chunk`, default 32)**

| chunk | tok/s | ms/step | peak | max abs vs O(T) reference |
|---|---:|---:|---:|---:|
| 16 | 14.5K | 1129 | 25.6 GB | 0.00e+00 |
| 32 | 26.9K | 610 | 25.6 GB | 0.00e+00 |
| 64 | 46.3K | 354 | 27.9 GB | 0.00e+00 |
| 128 | 53.2K | 308 | 33.3 GB | 0.00e+00 |
| 256 | 44.3K | 370 | 44.6 GB | 0.00e+00 |

**Eval loop host sync**

- as written: 298 ms/eval; one deferred transfer: 297 ms/eval (**1.00x**)
- means bit-identical: **False**; saves 0 s per 50M run (61 evals)

**MoE arms re-timed on real corpus batches**

| arm | tok/s (real) | ms/step | peak |
|---|---:|---:|---:|
| `attention` | 114.2K | 144 | 16.4 GB |
| `moe_e1k1` | 104.6K | 157 | 17.0 GB |
| `moe_e1k1_raw` | 104.8K | 156 | 17.0 GB |
| `moe_e4k1` | 54.9K | 298 | 18.9 GB |
| `moe_e4k1_raw` | 61.4K | 267 | 18.9 GB |
| `moe_e8k1` | 38.4K | 427 | 21.2 GB |
| `moe_e8k1_raw` | 44.2K | 371 | 21.1 GB |


## E. Per-arm tenancy, derived from the rows above

| arm | MFU @ t=1 | t=1 | best without MPS | best with MPS | recommend |
|---|---:|---:|---|---|---|
| `attention` | 13.8% | 131.0K | 1 @ 131.0K (1.00x) | 4 @ 144.5K (1.10x) | **4 (MPS)** (1.10x) |
| `hybrid_mingru8_attn4` | 13.2% | 110.8K | 1 @ 110.8K (1.00x) | 3 @ 123.8K (1.12x) | **3 (MPS)** (1.12x) |
| `mingru` | 13.0% | 102.9K | 1 @ 102.9K (1.00x) | — | **1** (1.00x) |
| `w384_attention_lr80` | 7.8% | 235.6K | 1 @ 235.6K (1.00x) | — | **1** (1.00x) |
| `w384_mingru_lr40` | 7.0% | 176.9K | 1 @ 176.9K (1.00x) | — | **1** (1.00x) |
| `moe_e8k1_raw` | 4.5% | 50.9K | 4 @ 67.6K (1.33x) | — | **4 (no MPS)** (1.33x) |
| `gdn` | 2.7% | 29.4K | 3 @ 45.1K (1.53x) | 3 @ 57.4K (1.96x) | **3 (MPS)** (1.96x) |

## F. Fused cross-entropy chunking

| arm | setting | tok/s | ms/step | peak resv | loss on fixed input |
|---|---|---:|---:|---:|---:|
| `attention` | fused/4 | 115.6K | 142 | 18.3 GB | 0.002640 |
| `attention` | fused/8 | 115.2K | 142 | 16.0 GB | 0.002617 |
| `attention` | fused/16 | 113.1K | 145 | 15.1 GB | 0.002414 |
| `attention` | fused/32 | 109.4K | 150 | 14.5 GB | 0.002742 |
| `attention` | unfused | 145.6K | 113 | 25.9 GB | 0.002606 |
| `hybrid_mingru8_attn4` | fused/4 | 99.6K | 164 | 22.9 GB | 0.000640 |
| `hybrid_mingru8_attn4` | fused/8 | 99.3K | 165 | 20.6 GB | 0.000628 |
| `hybrid_mingru8_attn4` | fused/16 | 97.8K | 168 | 19.7 GB | 0.000622 |
| `hybrid_mingru8_attn4` | fused/32 | 95.0K | 172 | 19.1 GB | 0.000628 |
| `hybrid_mingru8_attn4` | unfused | 121.1K | 135 | 30.5 GB | 0.000630 |

## G. What eval_iters buys

Two trained arms at seed 1337 (`cx32loop_attention_s1337` vs `cx32loop_attn6_s1337`), scored on the same 200 val batches.

- per-batch SD: 0.0736 and 0.0733 nats; correlation between the arms **0.9957**
- paired difference -0.0765, SD 0.0068 (the pairing removes most of the batch noise)

| eval_iters | SE of one arm's mean | SE of the paired difference | eval cost |
|---|---:|---:|---:|
| 5 | 0.0329 | 0.0031 | 25% |
| 10 | 0.0233 | 0.0022 | 50% |
| 20 | 0.0165 | 0.0015 | 100% |
| 40 | 0.0116 | 0.0011 | 200% |
