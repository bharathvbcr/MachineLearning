# 16 — Debugging and Failure Modes: A Field Guide

Theory tells you how things *should* work. This file is the opposite: everything that **broke** in
your project, the **logged signal** that exposed it, and the fix. It's organized as symptom →
diagnosis → fix, because that's how you'll actually use it. Every entry is a real event from your
runs — this is the most practically valuable file in the set.

> **The meta-lesson, stated once:** every bug below was caught by a *logged number looking wrong*,
> not by reading code. The loss, grad-norm, tok/s, power draw, and reserved-memory lines are your
> instruments. A trainer that doesn't log these is flying blind. nanolab logs all of them from
> step one — that is not decoration, it's the debugging strategy.

---

## 16.0 The diagnosis flowchart (start here when a run looks wrong)

```mermaid
graph TD
    START["Run looks wrong"] --> L{"loss?"}
    L -->|"= exactly 0"| BUG1["trivial target / leak<br/>(diffusion: targeted masked input)"]
    L -->|"NaN"| BUG2["divergence → LR too high?<br/>fp16 overflow? bad init?"]
    L -->|"flat from step 0"| BUG3["nothing learning →<br/>grad not flowing / data broken"]
    L -->|"descending fine"| SPD{"slow?"}
    SPD -->|"100% util, ~57W, 18s/step"| BUG4["SYSMEM THRASH<br/>reserved>8GB → fix: fused_ce+ckpt+mem_frac"]
    SPD -->|"backward takes seconds"| BUG5["O(T) recurrence /<br/>giant retained graph → chunk-parallel"]
    SPD -->|"throughput fine"| GN{"grad-norm?"}
    GN -->|"= 0"| BUG3
    GN -->|"spiking"| BUG6["instability → clip / lower LR"]
    GN -->|"stable"| OK["healthy — now compare A/B<br/>(≥2 seeds, one lever)"]
```

## 16.1 The numbers to watch, and what each one screams

| Logged signal | Healthy | What a bad value means |
|---|---|---|
| **loss** | descends, noisy | flat from step 0 = nothing's learning; **exactly 0** = trivial task (bug); NaN = divergence |
| **val loss** | tracks train, small gap | rising while train falls = overfitting; flat = underfitting/bug |
| **grad-norm** | small, stable | spiking = instability; **0** = no gradient reaching weights (bug); NaN = exploded |
| **lr** | follows the schedule | wrong shape = schedule misconfigured |
| **tok/s** | steady, high | dropping = memory thrash or input stall |
| **MFU** | near your ceiling (~25%) | low = GPU idle (memory/launch bound) |
| **power draw** (nvidia-smi) | ~130 W (3070 Ti) | **~57 W at 100% util = sysmem thrash** |
| **reserved mem** | < 8192 MiB | creeping past 8 GB = about to spill to host RAM |

---

## 16.2 Failure: loss collapses to exactly 0 (the diffusion bug)

- **Symptom:** loss and grad-norm both log **0** almost immediately.
- **Diagnosis:** the task became *trivial*. In your diffusion run, the loss targeted the **masked
  input** instead of the **clean tokens**, so the model just learned "output `[MASK]` where you see
  `[MASK]`" — a perfect, useless score.
- **Fix:** target the clean tokens (`diffusion_loss(..., x_clean, ...)`, file 15).
- **General rule:** **a loss of exactly 0 is never a success.** It means a leak, a trivial target,
  or a label/identity shortcut. Be *more* suspicious of a too-good number than a bad one.

---

## 16.3 Failure: 100% GPU util but it crawls (the sysmem thrash)

- **Symptom:** `nvidia-smi` shows **100% utilization** but only **~57 W** (vs ~130 W healthy), and
  steps take **~18 seconds**. Looks like a hang or a busy GPU — it's neither.
- **Diagnosis:** the allocation exceeded 8 GB and the **Windows/WDDM driver silently spilled VRAM
  to host RAM over PCIe (~25× slower)** instead of throwing OOM. The tell: `reserved` memory creeps
  past 8192 MiB. The low power proves the cores are *stalling on transfers*, not computing.
- **Fix (stack, file 06):** `fused_ce=True` (chunks=16), `grad_checkpoint=True`, and
  `set_per_process_memory_fraction(0.92)` so over-budget configs **OOM cleanly in ms** instead of
  thrashing for minutes. Result: 14% → 25.5% MFU, 13.7K tok/s.
- **General rule:** on Windows/8 GB, **watch power draw and reserved memory, not just utilization.**
  Util can be 100% while the GPU does almost nothing useful.

---

## 16.4 Failure: custom kernel is correct on CPU, wrong on GPU (the fp32 scan)

- **Symptom:** the chunk-parallel GDN/SSD scan passed a CPU test but drifted on GPU.
- **Diagnosis:** under bf16 autocast, the **recurrence accumulated rounding error** over the
  sequence. CPU ran fp32 by default, so the CPU test never exercised the bug. Specifically the
  backward's `grad_y` came in as bf16.
- **Fix:** force **fp32 inside the scan** (disable autocast in fwd/bwd, cast `grad_y` to fp32 —
  mixers.py:394). Verified exact vs the sequential reference, including non-divisible-T padding.
- **General rules:**
  1. **Accumulation wants more precision than pointwise ops** (file 06) — recurrences, softmax
     sums, optimizer state stay fp32.
  2. **Always verify a custom kernel against a brute-force reference** (`verify_scan.py`,
     `verify_gdn*.py` to 1e-5). A fast kernel you haven't verified is a fast way to be wrong.
  3. **CPU tests miss precision bugs** — test on the device and dtype you'll actually run.

---

## 16.5 Failure: recurrent mixer is unusably slow (the O(T) trap)

- **Symptom:** mamba2 at **333 tok/s**, gdn at **238 tok/s** — 24–33× slower than attention; gdn's
  backward pass alone took **12.5 seconds**. 0.4–0.6% MFU.
- **Diagnosis:** the naive recurrence is a Python `for t in range(T)` loop — sequential, and it
  builds a T-deep autograd graph that the backward must traverse.
- **Fix:** **chunk-parallel scanning** (file 04): SSD 333→3224 tok/s (9.7×), GDN 238→482 (2×), and
  it made ctx1024 trainable at all. Store only T/C small carries, not a T-deep graph.
- **General rule:** if a backward pass takes seconds, you have a **sequential dependency or a giant
  retained graph.** Vectorize the recurrence (parallel/associative scan) or checkpoint it.

---

## 16.6 Failure: a "better" hyperparameter quietly hurts (LR sensitivity)

- **Symptom:** bumping Muon LR 0.025 → 0.027 (an 8% change) raised BPB 2.093 → 2.100. `recur_2x3`
  weight-sharing tanked BPB to 2.851. Gated attention *alone* (2.089) was worse than plain value
  residual (1.987).
- **Diagnosis:** not bugs — **the optimum is narrow and effects interact.** A change that "should"
  help can hurt, and a trick that helps in combination can hurt alone (file 03).
- **Fix:** the **ablation ladder methodology** — change **one variable per run**, fixed seed +
  tokens, **multiple seeds** at the long stage (1337 + 42) to separate signal from noise, and a
  staged short→mid→long funnel so you don't pay for long runs on bad candidates.
- **General rule:** **trust measured A/B over intuition**, run ≥2 seeds before believing a small
  gap, and always re-test the *combination*, not just each piece.

---

## 16.7 Failure: the process segfaults on startup (Windows data stack)

- **Symptom:** the trainer segfaults when loading data — no Python traceback, just a native crash.
- **Diagnosis:** **torch + pyarrow/`datasets` in one process clash at the native-DLL level** on
  Windows; HF streaming parquet also segfaults alone.
- **Fix:** tokenize in a **dedicated torch-free process** (`prep_fineweb.py`) that writes `.bin`
  files; the trainer's `get_dataset` checks for existing `.bin`s **before importing any native data
  stack**, so torch and pyarrow never co-load (file 11).
- **General rule:** a crash with **no Python traceback** is a native/DLL conflict — isolate the
  conflicting libraries into separate processes.

---

## 16.8 Failure: the whole Mac freezes and there is no crash log (wired unified memory)

- **Symptom:** running DiffusionGemma 26B-A4B (8-bit MLX) through the benchmark harness **froze the
  entire machine** — beachball, no SSH, hard reboot required. Crucially, `/Library/Logs/
  DiagnosticReports` contained **zero `.panic` files** afterwards.
- **Diagnosis:** *no panic log means it never crashed* — it starved. On Apple Silicon the GPU and CPU
  share one memory pool, and mlx-vlm wraps generation in `wired_limit`, which unconditionally calls
  `mx.set_wired_limit(max_recommended_working_set_size)` = **51.8 GiB on a 64 GiB machine**
  (`mlx_vlm/generate/common.py:139`, applied at `diffusion.py:1179`). **Wired pages cannot be evicted
  by the kernel.** Meanwhile Ollama's default `keep_alive` is **5 minutes**, so the previously
  benchmarked model (18–36 GB) was *still resident* when MLX wired another 26 GB. 36 + 26 > 64, and
  the OS had nothing left to reclaim — so instead of swapping slowly, it stopped responding entirely.
- **The two guards that look like they'd help and don't:**
  - `mx.set_memory_limit` is documented as a *guideline* that raises only "if the memory limit is
    exceeded and there is no more RAM (**including swap**)" — it fires *after* the machine is dead.
    Its default is 1.5× the working set (~78 GB on a 64 GiB box), i.e. above physical RAM by design.
  - The wired limit **cannot be lowered**: no env var, no parameter, and it is re-set inside the
    library on *every* generation, so anything you set beforehand is overwritten.
- **Fix:** you cannot constrain it in-process, so **isolate it into a killable child**. `bench_mlx.py`
  (in the Benchmark project) evicts Ollama first via `keep_alive: 0` and *proves* `/api/ps` is empty,
  preflights the budget against both the Metal working set and live `vm_stat` availability, then runs
  MLX in a subprocess under a watchdog that `SIGKILL`s the **process group** on a memory floor or
  swap-growth breach. The parent never allocates, so it can always act.
- **General rule:** **a freeze with no panic log is memory starvation, not a crash.** And when a
  library sets a *global system* limit internally, in-process guards are theatre — the only real
  containment is a separate process you can kill. Note the symmetry with **16.3**: same root cause
  (allocation exceeding physical memory, silently absorbed instead of thrown), but the CUDA/WDDM
  version *degrades* to a 25× PCIe crawl, while the unified-memory version *wires* the pages and
  takes the OS down with it. Slow is a gift; unevictable is not.

---

## 16.9 Failure: the harness is wrong, not the model (measurement integrity)

- **Symptom:** DiffusionGemma scored **0.67** on a 3-prompt benchmark, failing the coding task
  with `exec failed: invalid syntax (<string>, line 1)`. The obvious read: the model can't code.
- **Diagnosis:** the model's code was *perfect*. The grader `exec()`'d the raw response, which
  began with a markdown fence and Gemma's `<|channel>thought<channel|>` control marker — so
  "line 1" was never Python at all. Worse, that checker had **never once executed**: every prior
  benchmark had been run with `--quick`, which only runs prompt 1. The bug had sat latent since
  the day it was written, and the first real invocation was also its first test.
- **Two more measurement bugs found in the same pass:**
  - `forwards_per_token` divided denoising steps by **canvas size** instead of *emitted tokens*.
    The canvas is what was *allocated* (256 slots for a 10-token answer), so unused slots were
    counted as free output — flattering the model **16×**. A metric whose denominator you did
    not choose deliberately is a metric that lies.
  - `skip_special_tokens=True` looked like it fixed the marker leak, and it did nothing: those
    markers aren't in the tokenizer's `all_special_ids`. **Confirming the flag exists is not
    confirming the flag works** — only the output proves it.
- **Fix:** strip control tokens and fences at one canonical owner before scoring; divide by
  emitted tokens and report `canvas_utilization` alongside so the gap stays visible; add a test
  per bug that fails against the pre-fix code. Re-scored, the true accuracy was **1.0**.
- **General rule:** **a bad score is a claim about your harness until you have read the raw
  output.** Grade the grader first — print what the model actually said before believing what
  the scorer said about it. And a code path that has never run is not "working", it is
  *unobserved*: `--quick` had quietly meant "skip the only test that exercises the checker."

---

## 16.10 The general debugging workflow you converged on

```
  1. LOG everything from step 1 (loss, val, grad-norm, lr, tok/s, MFU, + nvidia-smi power/mem).
  2. A weird NUMBER is the entry point — 0 loss, 0 grad-norm, 57 W, 18 s/step, reserved > 8 GB.
  3. ISOLATE: change one variable; reproduce on the real device + dtype (not CPU).
  4. VERIFY custom math against a brute-force reference to 1e-5.
  5. For "is this real?" — ≥2 seeds, staged short→long, fixed tokens.
  6. Distrust too-good numbers (0 loss, val ≪ train) more than bad ones.
  7. Before believing a BAD score, read the raw output — grade the grader first.
```

This is the actual scientific method applied to training. Most of your real findings — the
crossover, the champion, the kernel speedups, the MFU win — exist because the harness made step 3
cheap and the logs made step 2 possible.

**Next:** [`17-scaling-laws-and-competition-strategy.md`](17-scaling-laws-and-competition-strategy.md)
— zooming out to why the whole competition is shaped the way it is.
