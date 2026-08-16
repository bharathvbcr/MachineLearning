# 24 — Build It Yourself: Exercises

Reading builds recognition; running builds understanding. These exercises use the tools already in
your workspace (`nanolab`, the `verify_*` scripts, `nvidia-smi`) so every concept in files 01–23
becomes something you *watch happen*. Ordered from zero-GPU to GPU to analysis. Each has a **goal**,
a **command**, and **what to look for**.

> Environment reminder (from your memory notes): CPU-only work runs on the base Python; **GPU runs
> need the conda env** `C:\conda-data\envs\cuda_torch_env\python.exe` (torch 2.5.1+CUDA). Monitor
> with `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader`.

---

## Track A — No GPU needed (understand the loop)

### A1. Watch a model learn from scratch
- **Goal:** see the training loop (file 01) actually descend.
- **Command:** `python -m nanolab.train --preset cpu_smoke`
- **Look for:** loss falling from ~4.35 → ~2.60 over 40 steps. That descent *is* backprop + the
  optimizer (files 01, 18). If loss were flat from step 0, that's the "bug" signature (file 16).

  What a real descent looks like (your `gpu_max` run, loss vs step — bar ∝ loss):
  ```
    step   0  loss 15.02  ##############################   ← random init: max confusion
    step  10  loss  6.93  #############                    ← the "easy wins" (frequent tokens)
    step  20  loss  4.99  #########
    step  40  loss  4.09  ########                          ← steep early drop flattening out
    step  60  loss  3.54  #######
    step 110  loss  3.14  ######                            ← diminishing returns (log-like curve)
  ```
  The shape — fast early, then flattening — is universal: the model grabs the cheap statistical
  structure first, then grinds for the hard long tail. The MFU line climbs the opposite way
  (0.12 → 0.22 by step 20) as the GPU warms into steady state.

### A2. Run the regression tests and read them
- **Goal:** see what "verified" means (file 23).
- **Command:** `python -m nanolab.tests`
- **Look for:** 14 CPU assert-checks passing — fused-CE numerics, every mixer fwd/bwd, μP scaling,
  checkpoint roundtrip. Open `tests.py` and read the `gdn_chunked_matches_sequential` test: that's
  the 1e-5 kernel verification (files 04, 21) in code.

### A3. Hand-trace, then verify
- **Goal:** connect files 10/18 to reality.
- **Do:** pick the cross-entropy example in file 10 (§10.6), compute it on paper, then reproduce it
  in a Python REPL with `torch.nn.functional.cross_entropy`. Then do the attention example (§10.5).
- **Look for:** your by-hand numbers matching PyTorch's. Nothing builds confidence like this.

---

## Track B — The headline experiments (GPU, the ones you already ran)

### B1. Reproduce the mixer bake-off
- **Goal:** the inductive-bias result (file 04, 08).
- **Command:** `python -m nanolab.experiments` bake-off, or individually:
  `... train --preset phase0 --mixer mingru` vs `--mixer attention` (same seed/tokens).
- **Look for:** at ~2M tokens, recurrent mixers (mingru 5.837) beating attention (6.073). Confirm
  the ranking, not the absolute loss (the models are undertrained babble).

### B2. Find the crossover yourself
- **Goal:** watch bias-vs-capacity flip (file 04, 17).
- **Command:** run `--mixer mingru` and `--mixer attention` to ~8M tokens, eval every 200 steps,
  plot val loss vs tokens from `out/scale_*/metrics.jsonl`.
- **Look for:** the lines crossing near **7M tokens** (attn 5.239 < mingru 5.249). Seeing two curves
  cross is seeing a scaling law with your own eyes.

### B3. The architecture A/B
- **Goal:** the champion ingredients (file 03).
- **Do:** toggle gated attention and value residual independently, fixed seed/tokens.
- **Look for:** value-residual helps, gating *alone* hurts, the combination wins — the non-additivity
  lesson (file 03, 16). This is why you re-test combinations.

---

## Track C — Systems (GPU, watch the hardware)

### C1. Induce and then fix the sysmem thrash
- **Goal:** feel file 06/16's worst trap.
- **Do:** run `phase1`-ish at `ctx1024` with `grad_checkpoint=false, fused_ce=false`, big batch,
  while watching `nvidia-smi`. Then flip on `fused_ce=true (chunks=16)`, `grad_checkpoint=true`,
  `mem_fraction=0.92`.
- **Look for:** the broken state = **100% util, ~57 W, ~18 s/step, reserved > 8 GB**. The fixed
  state = **~130 W, sub-second steps, ~25% MFU**. The power-draw number is the tell, not util.

### C2. Probe the fused-CE chunk knob
- **Goal:** see one hyperparameter swing memory 3× (file 06).
- **Command:** `python -u -m nanolab.probe_perf --batch_size 32 --block_size 1024
  --grad_checkpoint true --fused_ce true --fused_ce_chunks {2,4,8,16,32}`
- **Look for:** chunks=2 → ~1.4K tok/s @ 14 GB (thrash) vs chunks=16 → ~13.3K tok/s @ 4.2 GB. One
  knob, the whole difference between thrashing and the validated peak.

### C3. Measure the kernel speedup
- **Goal:** quantify chunk-parallel (files 04, 21).
- **Do:** bench mamba2/gdn sequential vs chunked at bs8/ctx512 (the `sweep_gpu.py` / `probe_perf.py`
  `--mixer_chunk` path).
- **Look for:** SSD 333 → 3224 tok/s (9.7×), GDN OOM → trainable at ctx1024. The reason the bake-off
  was even possible.

---

## Track D — Verify the math (any device)

### D1. Verify a recurrence kernel
- **Command:** `python verify_scan.py`, `python verify_gdn.py`, `python verify_gdn_wy.py` (in
  `parameter-golf/`).
- **Look for:** chunk-parallel output matching the brute-force sequential reference to **1e-5**.
  This is the discipline from file 16: never trust an unverified custom kernel.

### D2. Reproduce Newton–Schulz on paper
- **Goal:** file 19 made concrete.
- **Do:** take a 2×2 matrix, run `X = a·X + (b·A + c·A²)·X` with `a,b,c = 3.4445,−4.7750,2.0315`
  for 5 steps (after normalizing), and check its singular values approach 1 (`torch.linalg.svdvals`).
- **Look for:** the singular values converging to ~1 — that's orthogonalization by matmul only.

---

## Track E — Go beyond (the experimental threads)

### E1. Convert to diffusion
- **Command:** the `nanolab/diffusion.py` path — adapt a `phase0` checkpoint on TinyStories.
- **Look for:** val perplexity 19.5 → 8.2 in ~7 min, and coherent stories by iterative denoising
  (file 15). Deliberately break the target (use masked input) once to *see* the loss collapse to 0
  (file 16) — then fix it. Breaking it on purpose teaches more than reading about it.

### E2. Spend the budget headroom
- **Goal:** the competition strategy (file 17).
- **Do:** take the champion config and try int8 instead of int6, or a 2048 vocab, or +2 layers;
  measure BPB *and* artifact size.
- **Look for:** which marginal byte buys the most BPB. The champion's 1.34 MB leaves ~14 MB unspent
  — this is the real game.

---

## A suggested 1-week path

```
  Day 1:  A1, A2, A3            (the loop + verification + hand-math)
  Day 2:  B1, B3                (mixer bias + architecture A/B)
  Day 3:  B2                    (the crossover — let it run, then plot)
  Day 4:  C1, C2               (the thrash and its fix — the systems core)
  Day 5:  C3, D1, D2           (kernels + the math behind them)
  Day 6:  E1                    (diffusion — the conceptual leap)
  Day 7:  E2                    (strategy — spend the budget)
```

By the end you'll have *watched* every major claim in files 01–23 happen on your own hardware.
That's the difference between knowing the words and knowing the thing.

**Next:** [`25-reading-list-and-connections.md`](25-reading-list-and-connections.md) — where each
idea came from, to go deeper.
