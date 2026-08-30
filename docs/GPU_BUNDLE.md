# GPU bundle — E1, E2, D10

One runner, `scripts/gpu_bundle.py`, covering the outstanding rented-GPU suites.
**394 jobs** by default. Resumable: finished runs are detected and skipped, so an
interrupted session costs time and no data.

Every figure below is **derived, not typed**. Regenerate it with:

```bash
python3 scripts/gpu_bundle.py --cost
```

That command reads the committed per-job `metrics.jsonl` of suites 22–26, takes the
median `tok_s` per (mixer, batch, context), and prices this matrix against it. Paper
§10 notes that no suite reports GPU-hours because the trainer discarded elapsed time;
it never discarded per-step throughput, and those run records are in the repository.

## Why this matrix is 394 jobs and not 64

The earlier version of this bundle would have spent real money and produced a result
that could not be claimed. Four defects, each fixed and each with a test that fails
against the pre-fix runner:

**1. Three of the four pre-registered readouts had no cells.** Every E1 job ran at
the suite-24 recipe. But §8.4's decision table is mostly *about other recipes*: rows
2 and 3 compare the 20M cosine (suite 23) against the 50M cosine under µP, and row 4
asks whether the batch-8 arm (suite 25) develops a crossing under µP. Neither recipe
was in the matrix. The run would have answered **one readout of four** and reported
"not measured" against its own pre-registered rule. The matrix now carries SP and µP
cells at all three recipes.

**2. The learning rate every µP cell inherits was decided by one seed.** `e1_proxy`
is the upstream dependency of the entire arm — a wrong value mis-tunes both arms on
the exact axis §8.4 exists to control, which is the D7 failure one axis over. D7's
own record is the argument: *"three times we drew a conclusion from two seeds that a
third seed overturned"*, at a per-cell seed spread the same size as the basin it was
resolving. The sweep now runs **three seeds**, and the transfer is published only if
the minimum is **bracketed and sign-consistent against both neighbours on every
seed**. Proxy jobs are the cheapest here; this costs ~1.9 GPU-h.

**3. Each µP arm was measured at a single learning rate.** §8.3's mechanism is
**offset flat basins**: two arms each measured at one point can sit on opposite sides
of their own optima, and the ordering that falls out is an artifact of where they
were measured. That is the defect this arm exists to remove, and the design
reproduced it with a different number. `e1_mup_basin` adds four more points per arm
at 0.25/0.5/2/4× the transferred value, so each arm gets a **five-point LR curve at
the target width**.

**4. µP transfer was asserted, never measured.** The arm rests on "the LR tuned at
width 256 is optimal at 768," implemented in `nanolab/optim.py` as
`matrix_lr / (d_model / mup_base_width)`. That is the **Adam** µP hidden-layer rule,
and the group it is applied to is **Muon** — whose update is orthogonalized and
already carries a `max(1, m/n)**0.5` RMS-match factor, and which the Muon literature
reports transfers across width *without* a 1/width factor. We do not change the rule:
changing a documented parametrization on the strength of an argument is what §8.3 is
a cautionary tale about. The five-point target-width curve **measures** it. If the
transferred value is the interior minimum, transfer is verified and the crossing
token comes with an LR-sensitivity band. If it is an end point, the transfer failed
and the arm is reported as a failed transfer rather than as a µP measurement. The
16× span is chosen to bracket a 3× rule error with room on both sides.

## The grid has already been extended once, and that is part of the record

The first sweep ran `PROXY_MATRIX_LRS = 0.0016 … 0.05` (six points, factor-2) at
three seeds. Result:

- **minGRU: bracketed at `matrix_lr = 0.00625`**, beating both neighbours on **3/3
  seeds**. Published.
- **attention: still falling at the low edge** — 5.265662 at 0.0016, rising monotone
  to 5.668418 at 0.05. Excluded from `transfer.json`; its µP cells stayed blocked.

That is the bracketing gate doing its job before it cost anything, and it is also a
result in its own right: **the two arms want very different hidden learning rates.**
minGRU's optimum is at 0.00625; attention's is below 0.0016, at least 4× lower, and
against the inherited suite value of 0.025 attention gives up 0.335 nats at this
width. Tuning **per arm** was not a precaution — transferring one arm's optimum to
both would have mis-tuned the other by a factor of at least four, which is the
unequal-tuning error that retired the §5 funnel's champion.

The grid was extended **downward to 0.0002** (nine points, 250× span) and re-run.
**Both arms got the new points**, not just attention: D7's own list of what made its
rounds unequal includes *"one candidate given lower learning rates than the other,"*
and these two curves are compared against each other. The 36 finished jobs were
skipped by the resume logic; the extension cost 18 jobs.

§8.3 records that its own grid went from twelve jobs to fifty-two across five rounds,
each round revealing the previous one had compared something unequal, and states that
trajectory rather than presenting the final design as the plan. This section does the
same. If attention's minimum is still at an edge after the extension, extend again —
these are the cheapest jobs in the bundle.

## The µP transfer failed, and that is a result

The basin was specified to detect a failed transfer rather than to confirm a
successful one. It detected one. At the s24 recipe, 20M tokens, n=3 (n=5 at ×1):

| µP multiplier | applied LR attn/minGRU | attention | minGRU |
|---|---|---|---|
| ×0.25 | 1.33e-4 / 5.21e-4 | 5.6986 | 5.1451 |
| ×0.5 | 2.67e-4 / 1.04e-3 | 5.5332 | 5.0865 |
| **×1 (transferred)** | 5.33e-4 / 2.08e-3 | 5.2900 | 5.0169 |
| ×2 | 1.07e-3 / 4.17e-3 | 5.1288 | 4.9933 |
| ×4 | 2.13e-3 / 8.33e-3 | **5.1034** | 5.0079 |
| — **SP baseline, same box, same recipe** | 0.025 global | **4.7454** | 4.9359 |

The curve is monotone in the multiplier with its minimum at the **high edge** — not
bracketed — and **every µP cell is worse than the SP cell measured beside it**. The
transferred value is not the target-width optimum, so the µP cells at ×1 are a
mis-tuned arm rather than a parametrization measurement, and they are reported that
way. The direction is the one the competing rule predicts: dividing an
already-orthogonalized, RMS-matched Muon update by `width_mult` overshoots downward.

Without this suite, the run would have published "no crossing under µP" — which is
what the ×1 cells show, on all five seeds — as a finding about parametrization when
it is a finding about mis-tuning.

`BASIN_MULTS` was extended to 32× (a 128× span) to bracket it.

## The basin publishes an anchor, the way the proxy publishes a transfer

Once bracketed, the basin locates each arm's target-width optimum, and that becomes
`anchor.json` — the same shape of artifact as `transfer.json`, gated the same way
(published only when the optimum is bracketed **and** sign-consistent, so a
direction can never be inherited as an optimum).

**Measured: attention 4×, minGRU 2×**, both 3/3 sign-consistent against both
neighbours. With the optimum located, the two candidate width rules stop being an
argument and become a measurement:

```
optim_py_divisor   off by [4x, 2x]          worst 4.0x   SAME direction on every arm
muon_no_divisor    off by [1.333x, 0.667x]  worst 1.5x   straddles 1.0
```

The grid is factor-2, so an error inside 2× is not resolvable. The no-divisor rule's
errors straddle 1.0 and sit inside one grid step — unbiased, within resolution. The
`matrix_lr / width_mult` rule in `optim.py` is off in the **same direction on both
arms**, by 2–4×. **Muon's learning rate transfers across width without the 1/width
divisor**, to within this grid's resolution, and applying the Adam µP hidden rule to
an already-orthogonalized, RMS-matched update is wrong at this width ratio.

`optim.py` is deliberately **not** changed. The finding is reported; the µP suites
are re-anchored by passing the measured `matrix_lr` explicitly. Changing a
parametrization rule in code, mid-experiment, on the strength of one width ratio is
what §8.3 is a cautionary tale about — and it would invalidate every basin cell that
produced the finding.

`e1_mup` stays at 1×: a failed transfer is a result, and re-pointing those job ids at
a better learning rate would erase it. `e1_mup_tuned` is the anchored cell, and
**the pre-registered readouts read the tuned cell, never the transfer cell** — reading
them off a 2–4×-mis-tuned arm would report mis-tuning as µP's answer, which is the
error this whole arm exists to remove.

## And the SP side needed the same treatment

Setting µP at its own optimum against SP at an inherited value would measure tuning
quality and report it as parametrization — this arm's own error, committed on the
other side of the table. `e1_sp_basin` sweeps SP's `matrix_lr` at the target width,
per arm, spanning 64× around the inherited 0.025 (which `e1_sp_rerun` already
measures at n=5, so only six of the seven points are new jobs).

It also settles something the paper has been carrying as an open concern rather than
a number. §8.1 names the inherited global learning rate — never re-tuned at these
recipes — as the largest uncontrolled factor in the paper, and §8.2 states that seed
agreement cannot bound it, because a wrong global LR is a systematic error all five
seeds share. `--analyse` prints the price directly: how many × off the inherited
value is, and what it costs in nats, paired per seed, exactly as §8.3 priced the
funnel's inherited learning rates.

## What each suite settles

| suite | jobs | closes |
|---|---|---|
| `e1_proxy` | 90 | Proxy-width **matrix-LR** sweep at `mup_base_width`, **per arm**, **5 seeds**. Locates a peak to transfer; it does not rank arms. Raised from 3 seeds on 2026-08-27: at three, minGRU's argmin sat inside the seed spread and `--analyse` refused to price its inherited LR. |
| `e1_sp_rerun` | 10 | SP cells of the 2×2 at the **s24** recipe, on *this* box. Hardware control, and — on a GH200 — the drift check against suite 24's published values. |
| `e1_mup` | 10 | µP at s24 at the **transferred** LR. This is the pre-registered transfer cell; it is kept at 1× because the fact that the transfer missed is a result. |
| `e1_mup_tuned` | 10 | µP at s24 at its **measured** target-width optimum, n=5. This is µP's actual answer, and the readouts read it. |
| `e1_mup_basin_spattn` | 42 | `e1_mup_basin` with SP's attention temperature. Its "TRANSFER MISSED" verdict was measured through the broken temperature, so it is not evidence about µP transfer. Re-measures whether the interior minimum moves to 1.0×. |
| `e1_mup_sched20_spattn` | 10 | `e1_mup_sched20` with SP's attention temperature. Rows 2-3 need a µP crossing at **both** schedules; the s24 half went from 0 crossings to 5/5 at 10.12M under this one-term ablation, so this is the s23 half. |
| `e1_mup_bs8_spattn` | 10 | `e1_mup_bs8` with SP's attention temperature. Row 4's "no" was measured through the broken temperature, so it was not an answer about batch 8. |
| `e1_mup_spattn` | 10 | µP at the transferred LR with SP's `1/sqrt(d)` attention temperature. Ablates µP's `1/d` rule, which without its companion q/k init leaves attention at 99.8% of uniform entropy at init (SP: 89%) and is the arm-asymmetric term behind every **NOT A VALID COMPARATOR** verdict. |
| `e1_mup_basin` | 42 | s24 µP either side of that LR (0.25× … 32×), 3 seeds. **Verifies the transfer** and gives the crossing token's LR sensitivity. |
| `e1_sp_basin` | 90 | s24 **SP** matrix-LR sweep at the target width, **5 seeds** (raised from 3 on 2026-08-27: at three, minGRU's argmin won on the mean and lost a seed, so `--analyse` refused to price its inherited LR). Prices the inherited 0.025 against SP's own optimum, and anchors the SP side of the comparison. |
| `e1_sp_sched20` | 10 | SP at the **s23** recipe (20M stop under a 20M cosine). |
| `e1_mup_sched20` | 10 | µP at s23. With `e1_mup`, this is **readout rows 2 and 3**. |
| `e1_sp_bs8` | 10 | SP at the **s25** recipe (batch 8, 8.19M). Reproduces suite 25's no-crossing result on this box, as row 4's control. |
| `e1_mup_bs8` | 10 | µP at s25. **Readout row 4**. |
| `e1_perlayer_sp` | 10 | Per-layer SP prescription (Everett et al.). **An approximation — see caveats.** |
| `e1_embed_lr` | 10 | Embedding-LR-only ablation (Kalra & Barkeshli): `embed_lr_mult = 768/256 = 3`. |
| `e2_matched32_50m` | 10 | Suite 26's missing attention/minGRU cells at 50M, batch 32. **GH200-gated.** |
| `d10_horizon` | 6 | **Opt-in** (`--with-d10`). Matched 10k vs 20k at one LR, 3 seeds, uninterrupted. |

The three recipes, each the exact recipe of a published suite:

| | batch | stop | cosine horizon | published crossing (SP) |
|---|---|---|---|---|
| **s24** | 32 | 20M | 50M | late 12.34M |
| **s23** | 32 | 20M | 20M | late 14.58M |
| **s25** | 8 | 8.19M | own | **none through 7.38M** |

## The hardware control, and why it is now enforced

PAPER §8.4's 2×2 puts new µP cells against suite 24's SP cells. **Suite 24 ran on a
GH200.** On any other box that 2×2 is confounded by hardware and proves nothing —
PAPER §7.1 refuses exactly this comparison, because the same architecture pair
differs by ~0.18–0.3 nats at matched token markers across two GPUs.

**`e2_matched32_50m` is gated on GH200 hardware.** Its cells are not a standalone
measurement: they are **merged into suite 26's published board**
(`experiment-notes/nanolab/artifacts/26-matched32_lock.json`), whose other eight rows
were measured on a Lambda ParameterGolf GH200. Filling that board's two
`"source": "suite22"` rows from an A100 replaces a same-box caveat with a
cross-hardware one — it leaves the board **worse than it found it**. An earlier
version of this document recommended running "E1+E2 on 8× A100," which was wrong for
E2 for exactly the reason it had already established for E1.

The runner previously said it "does not check that the box is that GH200, because it
cannot." It cannot identify the individual machine — but it can identify the **model**,
via `torch.cuda.get_device_name(0)`, and that is what the control is about. Preflight
now fails closed off a GH200 and records `device_name` / `is_gh200` in the ledger.
`--allow-cross-hardware-board` proceeds and records the caveat.

- **On a GH200** — the default and the recommendation. Everything valid.
- **`--sp-cells suite24`** drops the SP re-run and now **requires** a GH200 by device
  name. Recommended against even there: the re-run is 30 cheap jobs that convert an
  assumption about the environment into a **measured** drift number against suites
  23/24/25, and it is the only way to detect a PyTorch/driver change.

## The corpus is part of the recipe, not a capacity requirement

`nanolab/data/` is gitignored, so a fresh box has no tokenized corpus. An earlier
version of this document said E1 and E2 "need ~50M tokens' worth of headroom and the
existing 497.5M-token corpus is ample." That understates it.

The `Batcher` samples windows uniformly **with replacement**. A 20M-token job over a
50M-token corpus and the same job over a 497.5M-token corpus are two different
training distributions — 0.4 epochs against 0.04. Every cell here is compared against
a published suite, so the corpus must be **the same corpus**, not merely a large
enough one. Preflight fails closed when `train.bin` is not 497,500,000 tokens.

**Copy the reference corpus rather than re-tokenizing it** — a fresh tokenization of
the same nominal size is not guaranteed byte-identical (shard selection, dataset
revision):

```bash
rsync -az nanolab/data/HuggingFaceFW_fineweb-edu/ BOX:~/MLSystemsLab/nanolab/data/HuggingFaceFW_fineweb-edu/
```

`d10_horizon` is different: its 20k arm requests **655.4M tokens** against the
497.5M-token corpus, so it would revisit training data — 1.32 epochs — while the 10k
arm at 0.66 would not. That is a second variable moving alongside the one the pair
exists to isolate. Preflight fails closed on it; `--allow-data-repeat` accepts it
instead and records `data_epochs` per job so the repeat cannot go unreported.

## Order of operations

```bash
python3 scripts/gpu_bundle.py --plan          # the matrix, with blockers
python3 scripts/gpu_bundle.py --cost          # price it before renting
python3 scripts/gpu_bundle.py --check-docs    # do the docs still match the matrix?
python3 scripts/gpu_bundle.py --preflight     # gate the box BEFORE it bills
python3 scripts/gpu_bundle.py --smoke         # 40-step check, isolated subtree
python3 scripts/gpu_bundle.py --only e1_proxy --workers 4 --oversubscribe
#   read the sweep: both arms must be bracketed AND sign-consistent, then:
python3 scripts/gpu_bundle.py --workers 1     # everything remaining
python3 scripts/gpu_bundle.py --report        # ledger, GPU-hours, sweep
python3 scripts/gpu_bundle.py --analyse       # crossing tokens + the 8.4 readouts
```

`--workers` defaults to one job per visible GPU and **refuses** to exceed that. Pass
`--oversubscribe` if you mean it. Each child gets its own `CUDA_VISIBLE_DEVICES`.

**Concurrency.** On a 96 GiB GH200 the memory headroom is large (weights + gradients
+ optimizer state is 1.07–1.33 GiB) and these models are launch-bound, so
oversubscription buys a lot: a 256-wide proxy job holds 70.3k tok/s with **four**
running at once, against a measured 59.5k tok/s for a single 768-wide job.

```bash
python3 scripts/gpu_bundle.py --only e1_proxy --workers 4 --oversubscribe
python3 scripts/gpu_bundle.py --workers 3 --oversubscribe      # everything else
```

Concurrency does not touch the loss curves — separate processes, separate CUDA
contexts, per-job seeds — so it cannot affect a crossing token. It touches only
`mean_tok_s` and `elapsed_s`, and there the right choice is **to match the reference
suites rather than to minimise contention**. `ISOLATE_STAGES` in
`nanolab/crossover_replicate.py` records `"workers": 2` for suites 23, 25 and 26, and
suite 22's documented launch is `--workers 4`: the 59.5k tok/s in this cost model is
itself a contended rate. A single-tenant run here would produce throughput numbers
that are *not* comparable to the suites this arm is measured against.

Gap E6 — that suites 22–26 discarded their wall clock — is closed by recording
`elapsed_s` at all, not by recording it single-tenant. The ledger stores the worker
count in its `meta`, so a contended rate can never be read as an uncontended one.

`--smoke` writes into `nanolab/out/gpu_bundle/_smoke/`, a **separate subtree**.
`--preflight` runs automatically before any real launch.

## Cost and instance choice

Total work, derived: **≈ 36.40 GH200-hours** across the 394 jobs, of which 1.9 h is
extrapolated rather than measured (the proxy runs at width 256; no committed run
covers it, so `--cost` applies a labelled factor). Every other job is priced against
a measured rate. The longest single job is 18 minutes.

| instance | $/GPU-hr | full bundle |
|---|---|---|
| **1× GH200 96 GB — $2.29/hr** | **2.29** | **$57 / 24.9 h** |
| 4× H100 80 GB SXM5 — $16.36/hr | 4.09 | $95–103 / 5.8–6.3 h |
| 8× A100 40 GB SXM4 — $15.92/hr | 1.99 | $64–81 / 4.1–5.1 h |
| 2× H100 80 GB SXM5 — $8.38/hr | 4.19 | $66–72 / 7.9–8.6 h |
| 1× H100 80 GB SXM5 — $4.29/hr | 4.29 | $64–70 / 14.8–16.3 h |
| 1× H100 80 GB PCIe — $3.29/hr | 3.29 | $59–72 / 18.1–21.7 h |
| 1× A100 40 GB SXM4 — $1.99/hr | 1.99 | $50–66 / 25.2–33.3 h |
| 1× A10 24 GB — $1.29/hr | 1.29 | $64–95 / 49.4–73.6 h |

**Recommendation: 1× GH200 96 GB.** It is the cheapest option on the list *and* the
only correct one. Every other row's range is an assumed throughput ratio against the
GH200; the GH200's is 1.0 by definition, because every measured rate in the cost
model was recorded on that hardware. The 4× H100 finishes in a third of the time for
twice the money and invalidates E2.

**Do not use the 1× A10** — 24 GB is the smallest headroom here and it is the slowest
option per dollar of work.

**Memory.** Weights + gradients + optimizer state is **1.07 GiB** for the 768-dim
attention arm and **1.33 GiB** for minGRU, computed exactly from the parameter split.
That figure is almost irrelevant to how many jobs fit. **Measured peak device memory
per concurrent trainer on the GH200 is 17.0 GiB (attention) and 23.3 GiB (minGRU)** —
activations and the caching allocator are an order of magnitude larger than the
parameter state.

The gap between the two arms is what bites. Sizing `--workers 4` against attention's
17 GiB put four minGRU jobs on the card at **95.5 of 97.9 GiB — 97.6% full**, one
allocation from an OOM that would have killed four mid-flight jobs. The runner now
computes a cap from the **heaviest job actually queued** (`JOB_VRAM_GIB`, measured
not derived) against 85% of device VRAM, and refuses to exceed it. `--oversubscribe`
does **not** waive it — that flag says "co-locate them", not "fill the card";
`--ignore-vram` is the separate, explicit escape hatch.

On this GH200 the cap is **3 workers** for any suite containing a 768-dim minGRU
cell. An unmeasured shape is sized at the **worst known value**, not a plausible
guess — batch 8 is obviously lighter than batch 32, but it has not been measured, so
it is capped with everything else. Measuring it is what relaxes the cap; arguing
about it is not. `--smoke` on the rented box is how to measure a new shape.

## Three caveats that affect how results may be reported

**1. `e1_perlayer_sp` is an approximation, not a reproduction.** Everett et al.
(arXiv:2407.05872, Table 1) state their prescription for **pure Adam**: embedding LR
width-constant, hidden and readout ∝ 1/√n. Our Muon-family runs are a hybrid — 2-D
hidden matrices go to Muon, whose update is normalised by construction, and only
embeddings and scalars reach AdamW. They give no exponent for that combination.
Additionally, `_split_params` groups `tok_emb`/`pos_emb`/`lm_head` together, and under
`tie_embeddings=True` the embedding and readout are the *same tensor*, so their two
different exponents cannot both be honoured; we apply the embedding rule. To
reproduce their prescription proper, run this arm with `optimizer="adamw"` — which
then no longer matches suites 22–26 and is a different experiment.

**2. `d10_horizon` is not a reproduction of suite 20.** That would need the original
RTX 3070 Ti. This trio answers the *scientific* question — does a longer horizon help
at fixed LR? — on different hardware. Suite 20's own claim stays withdrawn either
way, because its 20k arm is eight resumed segments with a token counter off by 6.7×.
Report this as a new measurement, not a replication. It is **opt-in**: at three seeds
it is roughly 3× everything else combined, in jobs no parallelism shortens, and it is
not on §8.4's critical path.

**3. The µP arm inherits an unsettled embedding learning rate.** `e1_proxy` tunes
`matrix_lr` at the proxy width and transfers it, which is the µP contract for the
hidden layers. `cfg.lr` — the embedding/scalar rate — stays at the suite value, which
is correct *by the rule* under µP's width-constant embedding prescription, but is
itself inherited rather than tuned. That is a stated limitation of this arm, not a
defect in it. `e1_embed_lr` is the cheapest probe of whether it matters.

## What the runner refuses to do

Each of these was a real defect, and each now fails loudly instead:

- **Transfer an unbracketed optimum.** If either arm's proxy minimum lands at an end
  of `PROXY_MATRIX_LRS`, that arm is excluded from `transfer.json` and its µP cells
  stay blocked. A boundary minimum measures "lower is better within this range,"
  not an optimum. D7 needed four grid extensions to find its bottom and twice reported
  an edge as an optimum before the next point contradicted it.
- **Transfer an optimum decided by seed noise.** A minimum that wins on the mean but
  loses to a neighbour on any seed is not published either. That is the n=2 result
  §8.3 had overturned three times.
- **Run a µP suite with an inherited learning rate.** All four µP suites are blocked
  until the sweep publishes. The previous runner never passed the swept value into
  the µP cells at all, so they would have inherited the preset's 0.025 whatever the
  sweep found.
- **Merge a cross-hardware cell into a published board.** `e2_matched32_50m` fails
  preflight off a GH200.
- **Train on a different corpus than the suites it is compared against.**
- **Silently resume a half-finished run.** `Logger` opens `metrics.jsonl` in *append*
  mode, so re-running a job that died mid-flight produces one file with two `start`
  records — the `run128m_20k` shape that made suite 20 unusable. Such directories are
  marked `partial` or `suspect`, skipped, and never counted; `--reset-partial`
  archives them under `_archived/<timestamp>/` so the re-run starts clean.
- **Read an unverified run as a measurement.** A job counts as `done` only if it
  exited 0, **and** wrote exactly one `start`/`done` pair, **and** produced a
  `best_val`, **and** the `config.json` it wrote matches the recipe that was
  requested, field by field (PAPER §3.3's fingerprint check).
- **Lose a ledger.** `--only` used to serialize only the current invocation's jobs,
  so running suites one at a time erased the earlier ones — gap D8 by a second
  mechanism. The ledger now merges by id.
- **Rank on `best_val`.** PAPER §3.2: minimum-over-all-evaluations is not a ranking.
  Every proxy cell stops at the same token budget, so the sweep ranks on the **last**
  eval, and every job's full evaluation curve is stored in the ledger.
- **Report a readout it could not measure.** `--analyse` marks a row `unanswered`,
  never `pass`, when its cells did not run.

## Reading the result

```bash
python3 scripts/gpu_bundle.py --analyse
```

Computes the attention-vs-minGRU crossing token **per seed** — paired within a seed,
then aggregated, never the crossing of the mean curve — for every cell, with Student-t
intervals, and evaluates §8.4's four pre-registered rows against them. Writes
`nanolab/out/gpu_bundle/crossings.json`, **generated from the ledger, not
transcribed**: §8.3's record is the reason, where a hand-written conclusion survived
the round that superseded it and asserted the opposite of the data.

§8.4 states the decision rule **before** the run. Do not revise it after seeing the
numbers.

## The documents are checked against the code

```bash
python3 scripts/gpu_bundle.py --check-docs
```

Verifies the job totals, the per-suite counts in the table above, and the derived
GH200-hours in **this file and `ISSUES_AND_GAPS`** against the matrix the runner
actually builds. Exit 1 on any disagreement.

This exists because the repository has already been bitten twice. Gap D3 was a
document asserting the opposite of its artifact. And on the SP-cells question this
file said the 2×2's SP cells were "**not** rerun" while `ISSUES_AND_GAPS` §3.3 said
re-running them was "not optional" — **the two documents disagreed and the code
followed the wrong one**. Both grids in this bundle have since been extended
mid-run, which moves every total in both files. The code is the source; the check
says so.

## Tests

```bash
python3 scripts/test_gpu_bundle.py     # 259 checks, exit 0 = clean
```

No GPU, no trainer and no network: synthetic run directories, plus end-to-end cases
that drive the real `main()` with a fake trainer. Every defect listed above has a
test that fails against the pre-fix runner.
