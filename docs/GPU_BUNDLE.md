# GPU bundle — E1, E2, D10

One runner, `scripts/gpu_bundle.py`, covering the outstanding rented-GPU suites.
**64 jobs.** Resumable: finished runs are detected and skipped, so an interrupted
session costs time and no data.

Every figure below is **derived, not typed**. Regenerate it with:

```bash
python3 scripts/gpu_bundle.py --cost
```

That command reads the committed per-job `metrics.jsonl` of suites 22–26, takes the
median `tok_s` per (mixer, batch, context), and prices this matrix against it. Paper
§10 notes that no suite reports GPU-hours because the trainer discarded elapsed time;
it never discarded per-step throughput, and those run records are in the repository.

## What each suite settles

| suite | jobs | closes |
|---|---|---|
| `e1_proxy` | 12 | Proxy-width **matrix-LR** sweep at `mup_base_width`, **per arm**, 1 seed. Locates a peak to transfer; it does not rank arms. |
| `e1_sp_rerun` | 10 | The **SP cells** of PAPER §8.4's 2×2, measured on *this* box. See "the hardware control" below — this is not optional off a GH200. |
| `e1_mup` | 10 | The µP cells of the 2×2. **Blocked** until `e1_proxy` publishes a bracketed optimum per arm. |
| `e1_perlayer_sp` | 10 | Per-layer SP prescription (Everett et al.). **An approximation — see caveats.** |
| `e1_embed_lr` | 10 | Embedding-LR-only ablation (Kalra & Barkeshli): `embed_lr_mult = 768/256 = 3`. |
| `e2_matched32_50m` | 10 | Suite 26's missing attention/minGRU cells at 50M, batch 32. Lifts that board off Medium-High. |
| `d10_horizon` | 2 | Matched 10k vs 20k at **one** learning rate, uninterrupted. |

## The hardware control, and why the matrix grew by 10 jobs

PAPER §8.4's 2×2 puts new µP cells against suite 24's SP cells. **Suite 24 ran on a
GH200.** On any other box that 2×2 is confounded by hardware and proves nothing —
PAPER §7.1 refuses exactly this comparison, because the same architecture pair
differs by ~0.18–0.3 nats at matched token markers across two GPUs.

An earlier version of this document and of the runner both said the SP cells were
"suite 24 and are **not** rerun," while `docs/ISSUES_AND_GAPS_2026-08-22.md` §3.3 said
re-running all four cells on a non-GH200 box was "not optional." The two documents
disagreed and the code followed the wrong one. `e1_sp_rerun` is the fix.

- **On an A100 or H100 box** — the default. All four cells measured together.
- **On the GH200 that ran suite 24** — pass `--sp-cells suite24` to drop the re-run.
  The runner prints what that assumes; it does not check that the box is that GH200,
  because it cannot.

§8.4's own cost paragraph already said "twenty jobs at 20M tokens for the main 2×2,"
which is both rows. The table in that section said the SP row was already run. The
cost line was right.

## Order of operations

```bash
python3 scripts/gpu_bundle.py --plan          # the matrix, with blockers
python3 scripts/gpu_bundle.py --preflight     # gate the box BEFORE it bills
python3 scripts/gpu_bundle.py --smoke         # 40-step check, isolated subtree
python3 scripts/gpu_bundle.py --only e1_proxy --workers 4
#   read the sweep, confirm both arms bracketed, then:
python3 scripts/gpu_bundle.py --only e1_sp_rerun --workers 4
python3 scripts/gpu_bundle.py --only e1_mup --workers 4
python3 scripts/gpu_bundle.py --workers 4     # everything remaining
python3 scripts/gpu_bundle.py --report        # ledger, GPU-hours, sweep
python3 scripts/gpu_bundle.py --cost          # re-derive the estimate, any time
```

`--workers` defaults to one job per visible GPU and **refuses** to exceed that: more
concurrent trainers than devices co-locates them and OOMs at this model size. Pass
`--oversubscribe` if you mean it. Each child gets its own `CUDA_VISIBLE_DEVICES`.

`--smoke` writes into `nanolab/out/gpu_bundle/_smoke/`, a **separate subtree**. It
used to write into the same directories the matrix resumes from, so a 40-step smoke
run left a real `done` record with a real `best_val` and the full run then skipped
that job — one per suite, `e1_mup` among them. Following the documented procedure
corrupted the matrix. It no longer can.

`--preflight` runs automatically before any real launch. It checks the corpus, CUDA
device count, disk, and whether any run directory is left `partial` or multi-segment;
pass `--skip-preflight` to override it, which you should not need.

## Cost and instance choice

Total work, derived: **≈ 13.9 GH200-hours** across the 64 jobs, of which 7.0 h is
extrapolated rather than measured (the `d10_horizon` pair runs at context 1024 and
the proxy runs at width 256; no committed run covers either, so `--cost` applies
labelled factors).

The decisive fact is **the critical path, not the total**. `d10_horizon`'s 20k job is
one serial run of ≈ 4.4 GH200-hours. No number of GPUs shortens it, and it alone is
47% of the bundle's compute in 2 of its 64 jobs.

| instance | $/GPU-hr | full bundle | E1+E2 only (no `d10_horizon`) |
|---|---|---|---|
| 8× A100 40 GB SXM4 — $15.92/hr | 1.99 | $132–171 / 8.3–10.7 h | **$41–49 / 2.6–3.1 h** |
| 1× A100 40 GB SXM4 — $1.99/hr | 1.99 | **$48–63** / 24–32 h | $26–34 / 13–17 h |
| 8× A100 80 GB SXM4 — $22.32/hr | 2.79 | $166–217 / 7.4–9.7 h | $53–64 / 2.4–2.9 h |
| 1× H100 80 GB PCIe — $3.29/hr | 3.29 | $57–69 / 17–21 h | $32–38 / 9.6–11.5 h |
| 4× H100 80 GB SXM5 — $16.36/hr | 4.09 | $84–92 / **5.2–5.6 h** | $45–48 / 2.8–2.9 h |
| 2× H100 80 GB SXM5 — $8.38/hr | 4.19 | $64–70 / 7.6–8.3 h | $38–41 / 4.5–4.9 h |
| 1× H100 80 GB SXM5 — $4.29/hr | 4.29 | $61–67 / 14–16 h | $34–37 / 8.0–8.7 h |
| 1× A10 24 GB — $1.29/hr | 1.29 | $61–91 / 47–70 h | $33–49 / 25–38 h |

Brackets come from an **assumed** per-GPU throughput ratio against the GH200 (H100
SXM5 ≈ 0.95–1.05×, since GH200 carries an H100 die; A100 40 GB ≈ 0.45–0.60×). The
assumption-free version of the same question is the break-even multiple `--cost`
prints: an H100 SXM5 at $4.29/hr must be faster than **2.16×** an A100 40 GB, per
GPU, to cost less for the same work.

**Recommendation, in two parts, because they are two decisions:**

1. **Run E1+E2 first, on 8× A100 40 GB.** 62 jobs, none longer than 18 minutes, all
   independent — this is the case the 8-GPU box is for. **≈ $41–49, under 3 hours.**
   This is the work PAPER §8.4's pre-registered readouts depend on.
2. **Decide `d10_horizon` separately.** It costs more than the rest of the bundle
   combined on a multi-GPU box, because you rent 8 GPUs while one job monopolises
   one of them for 8–11 hours. If you want it, **2× H100 SXM5** is the sane home for
   the pair ($64–70 for everything, 7.6–8.3 h) — or run it alone on **1× A100 40 GB**
   for ~$25 and walk away for a day. Suite 20's horizon claim is already withdrawn
   in *both* directions, so this pair adds a new measurement rather than settling a
   live question.

**Do not use the 8× A100 80 GB.** It is the same A100 generation at 40% more per
GPU-hour, and nothing in this bundle needs 80 GB. **Do not use the 1× A10** — 24 GB
is the smallest headroom here and it is the slowest option per dollar of work.

**Memory, measured where it can be:** weights + gradients + optimizer state is
**1.07 GiB** for the 768-dim attention arm and **1.33 GiB** for minGRU (which carries
more matrix parameters), computed exactly from the parameter split. Activations are
the rest and cannot be measured on a machine without a GPU — `--smoke` on the rented
box is the check, and it exercises the widest job in each suite. The `d10_horizon`
arms run at context **1024**, double every other job, and are the memory high-water
mark.

## Data preparation is a required, billed step

`nanolab/data/` is gitignored, so a fresh box has no tokenized corpus and every job
would otherwise stall on first use. Tokenize before launching:

```bash
python -m nanolab.prep_fineweb --config sample-10BT --max_tokens 50000000
```

E1 and E2 need ~50M tokens' worth of headroom and the existing 497.5M-token corpus is
ample. **`d10_horizon` is different**: its 20k arm requests **655.4M tokens** against
a 497.5M-token corpus, so it would revisit training data — 1.32 epochs — while the 10k
arm at 0.66 epochs would not. That is a second variable moving alongside the one the
pair exists to isolate. Preflight **fails closed** on it and prints the exact prep
command; `--allow-data-repeat` accepts it instead and records `data_epochs` per job in
the ledger so the repeat cannot go unreported.

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
RTX 3070 Ti. This pair answers the *scientific* question — does a longer horizon help
at fixed LR? — on different hardware. Suite 20's own claim stays withdrawn either way,
because its 20k arm is eight resumed segments with a token counter off by 6.7×.
Report this as a new measurement, not a replication.

**3. The µP arm inherits an unsettled embedding learning rate.** `e1_proxy` tunes
`matrix_lr` at the proxy width and transfers it, which is the µP contract for the
hidden layers. `cfg.lr` — the embedding/scalar rate — stays at the suite value, which
is correct *by the rule* under µP's width-constant embedding prescription, but is
itself inherited rather than tuned. That is a stated limitation of this arm, not a
defect in it.

## What the runner refuses to do

Each of these was a real defect, and each now fails loudly instead:

- **Transfer an unbracketed optimum.** If either arm's proxy minimum lands at an end
  of `PROXY_MATRIX_LRS`, that arm is excluded from `transfer.json` and its `e1_mup`
  cells stay blocked. A boundary minimum measures "lower is better within this range,"
  not an optimum. D7 needed four grid extensions to find its bottom and twice reported
  an edge as an optimum before the next point contradicted it. If you see this,
  extend the grid and re-run the sweep — the extra jobs are the cheapest in the bundle.
- **Run `e1_mup` with an inherited learning rate.** The previous runner never passed
  the swept value into the µP cells at all, so they would have inherited the preset's
  0.025 whatever the sweep found — an arm mis-tuned on the exact axis under test,
  which is the D7 failure one axis over.
- **Silently resume a half-finished run.** `Logger` opens `metrics.jsonl` in *append*
  mode, so re-running a job that died mid-flight produces one file with two `start`
  records — the `run128m_20k` shape that made suite 20 unusable. Such directories are
  marked `partial` or `suspect`, skipped, and never counted; `--reset-partial`
  archives them under `_archived/<timestamp>/` so the re-run starts clean.
- **Read an unverified run as a measurement.** A job counts as `done` only if it
  exited 0, **and** wrote exactly one `start`/`done` pair, **and** produced a
  `best_val`, **and** the `config.json` it wrote matches the recipe that was
  requested, field by field (PAPER §3.3's fingerprint check, which this bundle
  previously skipped). Any failure records a reason in the ledger.
- **Lose a ledger.** `--only` used to serialize only the current invocation's jobs,
  so running suites one at a time erased the earlier ones — gap D8 by a second
  mechanism. The ledger now merges by id.
- **Rank on `best_val`.** PAPER §3.2: minimum-over-all-evaluations is not a ranking.
  Every proxy cell stops at the same token budget, so the sweep ranks on the **last**
  eval, and every job's full evaluation curve is stored in the ledger so crossing
  tokens can be computed without re-reading the run directories.

## Pre-registered readouts

§8.4 of the paper states the decision rule for the 2×2 **before** the run: which
outcome supports which reading of §4.3's schedule effect. Do not revise it after
seeing the numbers.

## Tests

```bash
python3 scripts/test_gpu_bundle.py     # 140 checks, exit 0 = clean
```

No GPU, no trainer and no network: synthetic run directories, plus end-to-end cases
that drive the real `main()` with a fake trainer. Every defect listed above has a
test that fails against the pre-fix runner.
