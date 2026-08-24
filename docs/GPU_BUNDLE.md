# GPU bundle — E1, E2, D10

One runner, `scripts/gpu_bundle.py`, covering the three outstanding rented-GPU suites.
54 jobs. Resumable: finished runs are detected and skipped, so an interrupted session
costs time and no data.

## Instance: 8× A100 40 GB, $15.92/hr

The cost algebra is unusual on this price list and worth stating, because the obvious
choice is wrong:

```
work = W GPU-hours
1x A100 40GB :  W hours   x $1.99  = $1.99W
8x A100 40GB :  W/8 hours x $15.92 = $1.99W     <- identical total
```

Both are $1.99 per GPU-hour, so the 8× box costs the same and finishes eight times
sooner. The only premium is roughly one hour of setup billed at the 8× rate, ~$14.
This workload is 52 independent jobs with no cross-job communication, so there is
nothing for a single fast GPU to win.

**Do not use the 1× A10.** 24 GB will not hold the 768-dim µP target with optimizer
state, and at roughly 0.4× an A100's throughput for $1.29/hr it works out near
$3.20 per A100-equivalent hour — the worst value of the three options.

**40 GB is sufficient**, verified rather than assumed: the widest model here is the
768-dim target at batch 32 / context 512, ~124M parameters, ~15 GB including
optimizer state and activations. The proxy sweep runs narrower still, at 256 dim.

**Estimate: ~32–37 A100-GPU-hours → ~4–4.5 h wall on 8 GPUs, plus ~1 h setup ≈ $88.**
Derived from suite 22–26 timings scaled by the A100/GH200 ratio; treat it as an
estimate, not a measurement. The `d10_horizon` pair dominates: 655M + 328M tokens at
context 1024 is roughly a third of the total.

## Run it

```bash
python3 scripts/gpu_bundle.py --plan       # 52 jobs by suite, no work
python3 scripts/gpu_bundle.py --smoke      # 40-step check, one job per suite
python3 scripts/gpu_bundle.py --only e1_mup
python3 scripts/gpu_bundle.py              # everything
```

Run `--smoke` first on the rented box. It exercises every code path at 40 steps in a
few minutes and will surface a data-path or driver problem before the full matrix
starts billing.

Results land in `nanolab/out/gpu_bundle/`, with `ledger.json` rewritten after every
job. A job counts as `done` only if it exited 0 **and** wrote a `best_val`; either
alone marks it `failed` with a reason. A run that could not be measured must never
read as a measured run.

## What each suite settles

| suite | jobs | closes |
|---|---|---|
| `e1_mup` | 10 | µP cells of PAPER §8.4's 2×2. The SP cells are suite 24 and are **not** rerun. |
| `e1_proxy` | 12 | Proxy-width **matrix-LR** sweep at `mup_base_width`, **per arm**, 1 seed. Locates a peak to transfer; it does not rank arms. |
| `e1_perlayer_sp` | 10 | Per-layer SP prescription (Everett et al.). **An approximation — see caveats.** |
| `e1_embed_lr` | 10 | Embedding-LR-only ablation (Kalra & Barkeshli): `embed_lr_mult = 768/256 = 3`. |
| `e2_matched32_50m` | 10 | Suite 26's missing attention/minGRU cells at 50M, batch 32. Lifts that board from Medium-High. |
| `d10_horizon` | 2 | Matched 10k vs 20k at **one** learning rate, uninterrupted. |

**Run `e1_proxy` first, and read its output before launching `e1_mup`.** Under
`muon_ns5_adamw` the 2-D hidden matrices are driven by `cfg.matrix_lr`, not
`cfg.lr` -- an earlier version of this sweep varied `cfg.lr` and would have left
the hidden LR, the quantity muP exists to transfer, pinned at an inherited 0.025
across every point. The sweep now varies `matrix_lr` over 0.0016..0.05, per arm,
and the runner refuses to call a grid edge an optimum: if either arm's minimum
lands at an end of the range, extend `PROXY_MATRIX_LRS` and re-run before
spending anything on the 2x2. `cfg.lr` stays at the suite value, which is correct
under muP's width-constant embedding rule but is itself inherited rather than
tuned -- a stated limitation of this arm.

Every `e1_*` and `e2_*` job takes its cadence from
`crossover_replicate.scale_to_token_budget`, so eval markers line up with suites 22–26
and the loss-vs-token curves stay comparable.

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

**3. The µP arm inherits an unsettled learning rate.** `e1_proxy` tunes at the proxy
width and transfers, which is the µP contract. But the D7 re-tune (`research/
d7-lr-retune.json`) found *both* optimizer finalists 4–8× above their optima at 128M,
with neither optimum bracketed. If the proxy sweep's peak lands at the edge of
`PROXY_LRS`, the transferred LR is a boundary value, not an optimum, and §8.4's
pre-registered readouts should be read with that stated — the same failure the D7 grid
hit twice.

## Pre-registered readouts

§8.4 of the paper states the decision rule for `e1_mup` **before** the run: which
outcome supports which reading of §4.3's schedule effect. Do not revise it after
seeing the numbers.
