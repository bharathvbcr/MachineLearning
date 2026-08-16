# 51: M5 128M optimizer funnel preflight

## Executive summary

- **Question:** Can the exact 128M Arch02 engine expose a broad, honest native
  optimizer study and pass the correctness, resume, memory, dispatch, and
  telemetry gates required before one 2,000-step champion run?
- **Result:** Yes. The equal-data funnel completed end-to-end and selected
  **Polar Muon** (`muon_polar_adamw`, matrix LR **0.05**). The unlocked
  champion run finished 2,000 exact-128M steps at final EMA BPB **2.015756**.
  Cross-scale comparison to the 3070 Ti sota-ladder **1.9944** is **withdrawn**
  (different shape); same-shape CUDA 128M is pending `logs/cuda128m` from
  `cuda_ref_128m.sh`. MiMuon and SOAP remain
  visible exclusions, not silent substitutions.
- **Implication:** At this token budget and LR-selection protocol, Polar is the
  honest native winner over NorMuon/NS5/MONA. The 16M ranking was not stable
  across scales: MONA led the two-seed 16M stage but finished last at exact
  128M/500. Absolute-zero swap in `exact_gate` was a false fail on ambient
  macOS residual; the gate now measures induced swap delta.
- **Status:** `funnel-complete / champion-complete`; evidence confidence
  **High** for systems correctness and **Medium–High** for optimizer ranking
  under this equal-data protocol.

## Inputs from the journey

This work preserves the local evidence instead of treating new optimizer papers
as automatic upgrades:

- The 4.096M-token width-768 bakeoff put Lion and AdamW ahead of Muon on quality,
  so both remain mandatory serious controls.
- The native Arch02 work showed that same-shape Muon bank batching is a strong
  systems fit on this M5 Pro, so NS5 remains the matrix-optimizer anchor.
- The Muon numerics note requires stable Frobenius normalization and bounded
  polynomial error; NS3, NS5, Polar, NorMuon, MONA, and Muown each have separate
  native state and parity fixtures.
- The stability notes and long-run native evidence preserve Soft-split clipping:
  Muon-family updates use `sqrt(c)`, adaptive updates use `c`, with global
  threshold 0.3.

## Native optimizer surface

| Status | Candidates |
|---|---|
| Native parity-qualified | Muon NS5, Muon NS3, Polar Express Muon, NorMuon, Muown, MONA, AdamW, Lion, Cautious AdamW, Cautious Lion, momentum SGD, Sophia, Schedule-Free AdamW, Prodigy |
| Python oracle only / systems-blocked | MiMuon (exact singular-gap routing needs SVD), SOAP (basis refresh needs eigendecomposition) |

Metal/MPS exposes the matrix multiplication and factorization primitives needed
by the implemented arms but no GPU SVD/eigensolver for the two blocked arms. A
CPU Accelerate fallback would introduce forced device synchronization and is
therefore rejected for this single-GPU engine.

## Numerical screen

All 16 candidates completed the identical 20-step CPU toy screen without a
nonfinite failure. This is a numerical gate, not a quality conclusion. The top
four toy BPBs were Cautious Lion 3.9006, Lion 3.9129, Cautious AdamW 3.9358, and
Polar 4.0101; the ordering must not bypass the 16M LR sweep.

Artifact: `nanolab/out/funnel/toy-results.json`.

## Exact-scale systems result

The exact gate runs two uninterrupted B16/T256 steps, then repeats step two from
a complete checkpoint made after step one.

| Metric | Result | Gate |
|---|---:|---:|
| Parameters | 128,367,988 | exact preset |
| Loss delta | 0 | ≤1e-5 |
| Gradient-norm delta | 7.15e-7 | ≤1e-4 |
| Sampled master-weight max delta | 2.41e-6 | ≤1e-4 |
| Current physical footprint | 13,477.6 MiB (13.16 GiB) | <52 GiB |
| Dispatches | 1,707 | <10,000 |
| Swap before/after | 0 / 0 MiB | no pressure |

The first attempt also found a scale-only checkpoint defect: the NPY writer made
one syscall for every float and advanced at roughly 2 MB/s. Bulk contiguous
little-endian reads/writes replaced it before the passing rerun.

## Research telemetry

On requested log steps, weights are snapshotted on the GPU and reduced to scalar
summaries after the update. JSONL now contains optimizer and total step time,
dispatch count, current physical footprint, gradient and update norms by matrix /
embedding / auxiliary role, sampled orthogonality error, row log drift, a
Frobenius spectral-drift proxy, and nonfinite counts. No full tensor is copied to
the host for research logging.

## Equal-data funnel and champion lock

`nanolab/native_funnel.py` maintains an atomic, resumable job ledger:

1. Five-point logarithmic LR sweep on the 16M model, 100 steps, seed 1337.
2. Every stable LR winner for 500 steps, seed 1337.
3. Best validation candidates plus mandatory AdamW and NS5 anchors for 1,000
   steps, seeds 42 and 2026.
4. Top four at exact 128M for 500 steps.
5. Top two at exact 128M for 1,000 steps, seeds 42 and 2026.
6. Select by mean equal-token validation BPB, then time-to-loss, physical
   footprint, and step time when confidence intervals overlap.

The final manifest encodes B16/T256, 2,000 steps (8.192M tokens), seed 1337,
350-step final warmdown, EMA 0.997, and evaluation plus complete checkpoint every
250 steps. It remained locked until the funnel and winner-specific exact gate
passed; Polar then unlocked and completed the champion run above.

### End-to-end funnel canary

The first live ledger job completed on FineWeb after the implementation gates:
Muon NS5 at matrix LR 0.00625, 16M preset, seed 1337, 100 steps. It remained
finite, ended at live BPB 2.8201 and final EMA BPB 3.507494 (EMA deliberately
lags on this short horizon), averaged 579.7 ms over logged steps and 43.9 ms in
the profiled optimizer, peaked at 5,463.5 MiB current physical, used at most
1,096 dispatches, and reported zero nonfinite telemetry values.

The canary found and corrected an orchestration hazard: the initial runner used
an existing release binary without checking source freshness. That stale binary
became nonfinite and also exposed `null`/NaN collector handling. Funnel launches
now always run Cargo's incremental release build, archive any prior attempt,
stop the trainer at the first nonfinite loss/global norm, sanitize nonfinite JSON
to strict `null`, and classify such arms as failed. The archived stale-binary
attempt is not optimizer evidence.

## Completed funnel results

### Five-point 16M LR sweep (100 steps, seed 1337)

The LR stage completed **68/70** finite jobs. Both failures were Prodigy
(`lr=0.5` and `lr=4.0`); every other candidate completed all five settings.
Best final EMA BPB per candidate:

| Rank | Candidate | Selected LR | EMA BPB | Finite points |
|---:|---|---:|---:|---:|
| 1 | NorMuon | 0.1 | 3.353160 | 5/5 |
| 2 | MONA | 0.05 | 3.369066 | 5/5 |
| 3 | Polar Muon | 0.05 | 3.372584 | 5/5 |
| 4 | Muon NS5 | 0.1 | 3.374745 | 5/5 |
| 5 | Muon NS3 | 0.1 | 3.376012 | 5/5 |
| 6 | Cautious AdamW | 0.0012 | 3.570897 | 5/5 |
| 7 | AdamW | 0.0012 | 3.593993 | 5/5 |
| 8 | Muown | 0.00625 | 3.641739 | 5/5 |
| 9 | Sophia | 0.0012 | 3.643193 | 5/5 |
| 10 | Cautious Lion | 0.00048 | 3.739487 | 5/5 |
| 11 | momentum SGD | 0.1 | 3.753697 | 5/5 |
| 12 | Lion | 0.00048 | 3.754734 | 5/5 |
| 13 | Schedule-Free AdamW | 0.0024 | 3.777532 | 5/5 |
| 14 | Prodigy | 0.25 | 11.889982 | 3/5 |

The table is a per-candidate LR choice, not a cross-optimizer conclusion; EMA
lag is large at 100 steps.

### Stable 500 (16M, seed 1337)

Thirteen candidates completed; Prodigy diverged at step 86. Results are ordered
by final EMA BPB:

| Rank | Candidate | LR | EMA BPB | Logged step ms | Optimizer ms | Peak physical MiB |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Polar Muon | 0.05 | **2.407025** | 516.3 | 41.1 | 5463.5 |
| 2 | NorMuon | 0.1 | **2.467975** | 517.9 | 37.9 | 5463.5 |
| 3 | Muon NS5 | 0.1 | **2.480362** | 521.9 | 41.2 | 5463.4 |
| 4 | Muon NS3 | 0.1 | **2.509174** | 511.4 | 27.0 | 5464.0 |
| 5 | Muown | 0.00625 | 2.796907 | 539.2 | 46.1 | 5448.5 |
| 6 | MONA | 0.05 | 2.888677 | 526.7 | 36.7 | 5463.4 |
| 7 | Lion | 0.00048 | 5.805887 | 488.9 | 3.8 | 5447.3 |
| 8 | Cautious Lion | 0.00048 | 6.000390 | 486.6 | 4.5 | 5463.4 |
| 9 | Schedule-Free AdamW | 0.0024 | 6.593316 | 483.3 | 3.7 | 5463.4 |
| 10 | AdamW | 0.0012 | 8.489102 | 498.2 | 4.1 | 5463.3 |
| 11 | Sophia | 0.0012 | 8.945090 | 486.5 | 3.6 | 5463.3 |
| 12 | Cautious AdamW | 0.0012 | 9.253303 | 493.0 | 4.6 | 5463.5 |
| 13 | momentum SGD | 0.1 | 12.503120 | 484.6 | 3.8 | 5463.4 |
| — | Prodigy | 0.25 | **failed** | — | — | 5463.3 |

Prodigy showed a global gradient norm of 275.3 by step 50, update norm 85.8 on
matrix roles, MLP-up bank norm 5421.6, and then `loss=NaN` / `grad_norm=NaN` at
step 86. This is a numerical exclusion, not a missing result.

**Interpretation:** the 100-step sweep selected aggressive adaptive/control LRs
that aged poorly over 500 steps, while the Muon-family arms continued improving.
The previous journey evidence for Lion/AdamW is not erased: AdamW remains a
mandatory 1,000-step anchor, and the two new seeds will test whether the current
ranking transfers. Polar, NorMuon, NS5, NS3, Muown, and MONA advance as the top
six; AdamW advances as the mandatory control. Prodigy is excluded.

### Two-seed 1,000 (16M, seeds 42 and 2026)

All 14 jobs completed with finite telemetry. Mean BPB and the 95% interval
computed from the two seeds:

| Rank | Candidate | Seed 42 | Seed 2026 | Mean BPB | 95% interval half-width | Mean logged step ms |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MONA | 2.183040 | 2.178756 | **2.180898** | 0.004198 | 528.2 |
| 2 | Polar Muon | 2.179047 | 2.185596 | **2.182322** | 0.006418 | 550.4 |
| 3 | NorMuon | 2.236222 | 2.228217 | 2.232220 | 0.007845 | 542.2 |
| 4 | Muon NS5 | 2.225931 | 2.239343 | 2.232637 | 0.013144 | 567.1 |
| 5 | Muon NS3 | 2.262745 | 2.255586 | 2.259166 | 0.007016 | 543.7 |
| 6 | Muown | 2.583694 | 2.585490 | 2.584592 | 0.001760 | 556.3 |
| 7 | AdamW control | 11.378829 | 11.179040 | **11.278935** | 0.195793 | 511.4 |

MONA and Polar have overlapping intervals. The declared tie-breaker keeps MONA
first: mean time to the common logged training-loss target was 26.94 s versus
28.07 s for Polar. Their memory footprints were effectively tied (~5.34 GiB).

AdamW completed numerically but is a quality failure at LR 0.0012. Its 100-step
LR choice looked competitive only because short-horizon EMA lag hid the later
instability: BPB worsened from 3.594 at 100 steps to 8.489 at 500 and a two-seed
mean of 11.279 at 1,000. This does not overturn the older AdamW journey evidence;
it demonstrates that the current short LR-selection horizon chose an LR that
does not transfer. The mandatory control obligation is satisfied, and AdamW
does not advance to exact scale.

**Exact-scale advancement:** MONA, Polar, NorMuon, and NS5 advance to the
128M/500-step stage at seed 1337. NS3 misses the cutoff by mean BPB; Muown and
AdamW are clearly separated.

### Exact 128M / 500 (seed 1337)

All four jobs completed finite. Ranking by validation BPB:

| Rank | Candidate | LR | Val BPB | Logged step ms | Optimizer ms | Peak physical MiB | Max dispatches |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Polar Muon | 0.05 | **2.420958** | 3014.8 | 451.2 | 13779.7 | 1981 |
| 2 | NorMuon | 0.1 | 2.567474 | 2925.5 | 451.7 | 13780.8 | 1985 |
| 3 | Muon NS5 | 0.1 | 2.601096 | 2915.5 | 437.5 | 13780.5 | 1981 |
| 4 | MONA | 0.05 | 2.950696 | 3147.0 | 495.0 | 13779.8 | 1981 |

**Interpretation:** scale reverses the 16M two-seed order. MONA’s 16M lead does
not transfer; Polar opens a clear gap (+0.15 BPB over NorMuon). Footprints and
dispatch counts are effectively tied. Polar and NorMuon advance to exact
128M/1,000 with seeds 42 and 2026.

### Exact 128M / 1,000 (seeds 42 and 2026)

| Rank | Candidate | Seed 42 | Seed 2026 | Mean BPB | 95% interval half-width | Mean logged step ms |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Polar Muon | 2.167282 | 2.172555 | **2.169919** | 0.005168 | 2831.8 |
| 2 | NorMuon | 2.202106 | 2.200184 | 2.201145 | 0.001884 | 2872.9 |

Intervals do not overlap. Polar is the champion without invoking systems
tie-breakers. Selection mean BPB **2.1699185** is stored in
`research/champion-run.json` with tuned LR **0.05**.

### Winner exact gate and swap-pressure fix

The first Polar exact-gate attempt failed with `passed: false` while resume
deltas, footprint, and dispatches all cleared. Both `swap_before_mb` and
`swap_after_mb` were **0.25** — ambient macOS residual, not run-induced
pressure (`vm.swapusage` matched; swapins were 0).

`exact_gate` previously required absolute `max(swap) == 0`, which is
machine-state brittle and rejects honest winners after any prior swapout.
The gate and `unlock_from_gate` now require **non-increasing** used swap
(`swap_delta_mb <= 0` within 1e-6 MiB), report `swap_delta_mb` /
`swap_pressure` / `failures`, and still fail if the run grows swap.

Polar winner gate after the fix:

| Metric | Result | Gate |
|---|---:|---:|
| Loss delta | 1.43e-6 | ≤1e-5 |
| Gradient-norm delta | 0 | ≤1e-4 |
| Sampled master-weight max delta | 2.56e-6 | ≤1e-4 |
| Current physical footprint | 13,483.4 MiB (13.17 GiB) | <52 GiB |
| Dispatches | 1,707 | <10,000 |
| Swap before / after / delta | 0.25 / 0.25 / **0.0** | no induced pressure |

Unlock succeeded; champion manifest is unlocked with full 2,000-step argv.

Artifact: `research/exact-128m-gate-polar.json`.

### Champion run (exact 128M, 2,000 steps, seed 1337)

Polar Muon at matrix LR 0.05 completed the locked recipe (B16/T256, 8.192M
tokens, final warmdown 350, EMA 0.997, eval/checkpoint every 250):

| Metric | Result |
|---|---:|
| Final EMA sliding BPB | **2.015756** |
| Same-shape CUDA 128M reference | **null** (pending `logs/cuda128m`; sota-ladder 1.9944 is not a twin) |
| Mean logged step ms | 2831.0 |
| Peak current physical | 13,780.1 MiB (13.46 GiB) |
| Steady-step dispatches | ~1,975–1,981 |
| Checkpoint-step dispatches | 9,475 (still <10,000) |

Output: `out/champion_128m_seed1337/` (checkpoints through step 2000,
`metrics.jsonl`).

## Verification

- Native Rust/Metal: **57/57** release library tests pass.
- Python correctness/orchestration oracle: **29/29** tests pass (includes
  induced-swap unlock rejection).
- Exact checkpoint/replay: NS5 preflight pass; Polar winner gate pass after
  induced-swap semantics.
- Intentional NaN: reported as a numerical failure.
- Funnel ledger: LR sweep **68 completed / 2 failed**; stable-500 **13
  completed / 1 failed**; advance-1000 **14/14**; exact-128M/500 **4/4**;
  exact-128M/1000 **4/4**; two systems-blocked candidates retained.
- Champion: Polar Muon unlocked and 2,000-step run complete.

## Artifacts

- `research/optimizer-study.json`
- `research/native-optimizer-funnel.json`
- `research/champion-run.json`
- `research/exact-128m-gate.json`
- `research/exact-128m-gate-polar.json`
- `out/champion_128m_seed1337/`
- `nanolab/optimizer_funnel.py`
- `nanolab/native_funnel.py`
- `Rust_MLKit/arch_02_value_resid/metal-native/src/research.rs`
- `Rust_MLKit/arch_02_value_resid/metal-native/src/bin/exact_gate.rs`
