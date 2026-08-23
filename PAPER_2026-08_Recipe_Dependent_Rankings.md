# Recipe-Dependent Rankings: Method Orderings in Language-Model Screens Are Properties of the Measurement, Not the Methods

**Bharath Chandra Vaddaram** · independent researcher · <bharath.vbcr@gmail.com>

`parameter_golf` / `nanolab` · **Reconstructed draft, 2026-08-23**

**Code and artifacts:** <https://github.com/bharathvbcr/MachineLearning>
**Reproduce every figure below:** `python3 paper/derive_figures.py`
**License:** text CC BY 4.0; code and data artifacts MIT.
**Competing interests:** none. **Funding:** none; all compute self-funded on
personal hardware (RTX 3070 Ti Laptop 8 GB, Apple M5 Pro, and one rented
NVIDIA GH200 instance).

---

## Note on this reconstruction

The source file of this manuscript was lost. It was removed from the working
tree and stripped from every commit by a history rewrite of this repository;
no copy survives on disk or in any reachable object.

What survived is the part that decides whether the paper is true: the runs.
`nanolab/out/crossover50m/`, `crossover8m_bs8/`, `crossover20m_locked/`,
`crossover20m_matched_lr/`, `crossover50m_matched32/` and
`research/native-optimizer-funnel.json` are intact, five seeds per arm.

This text was therefore rebuilt **from those artifacts rather than from
recollection**. `paper/derive_figures.py` recomputes every number stated here,
and `--check` fails if the manuscript states a figure the script does not
derive. Where the two disagreed, the script won — §5 records the one place
that happened.

Two things were not recovered and are not reproduced here: the catalogue of
six further reversal axes assembled from earlier single-seed work, and the
nineteen-item reference list. Their absence narrows the paper's scope; it does
not weaken the results below, which stand on runs in this repository.

---

## Abstract

Short-horizon architecture comparisons are the default currency of empirical
language-model work. We report a controlled attempt to replicate one such
comparison — our own — and find that its headline was a property of the
measurement recipe rather than of the architectures compared.

A single-seed run on an RTX 3070 Ti at batch size 8 placed the point where
softmax attention overtakes minGRU between 6.6M and 7.4M training tokens on a
12-layer / 768-dim model. Re-running the matched pair on an NVIDIA GH200 at
**n = 5 seeds** through a 50M-token horizon does not reproduce that location.
The mean validation curves cross **twice**: minGRU overtakes attention at
**1.05M** tokens, and attention overtakes minGRU for good at **12.35M**,
finishing **0.227** nats ahead at 50M (attention 4.222 [4.204, 4.240], minGRU
4.449 [4.423, 4.475]).

Isolating the confounds one at a time, five seeds each: holding batch size at
the original 8 produces **no crossing at all** through 7.38M tokens on any
seed. Truncating the run to a 20M-token budget while letting the cosine
schedule follow the truncated step count moves the late crossing to **14.58M**;
restoring the 50M cosine horizon and stopping at the same 20M budget recovers
**12.33M**, within interpolation error of the full-length run's 12.35M. The
crossing point is thus a function of the learning-rate horizon and the batch
size, not of the mixers.

The same phenomenon appears on an independent axis. In a five-stage optimizer
funnel on Apple silicon, MONA ranks **first** at 16M parameters (2.180898 BPB)
and **last of four** at an exact 128M-parameter model (2.950696); the eventual
128M champion, Polar Express Muon, ranked **third** in the initial
learning-rate screen.

The transferable claim is not "attention wins late", and not merely that the
crossing token moves. It is that **method orderings in short screens are
functions of batch size, learning-rate horizon and model scale**, and that a
ranking measured at one recipe carries no licence to be reported as a property
of the methods.

---

## 1. The original measurement, and why it was suspect

The claim under test was recorded from a single seed, on one consumer GPU, at
batch size 8: attention overtakes minGRU somewhere between 6.6M and 7.4M
tokens. Nothing about that measurement was wrong as an observation. What was
wrong was the sentence built on it, which named a property of two
architectures.

Three things made it worth re-running rather than citing: one seed, one batch
size, and a horizon short enough that the cosine schedule's tail sat inside the
window being compared.

## 2. Protocol

Twelve layers, 768 model dimensions, 12 heads, RoPE, RMSNorm with QK
normalisation, SwiGLU. Identical data pipeline and identical token budgets
across arms; the only difference between the two arms is the sequence mixer.
Five seeds — 42, 100, 777, 1337, 2026 — for every arm of every condition
reported here. Validation loss is evaluated on a fixed grid of token counts,
and the curves compared are means across those five seeds. Crossing points are
located by linear interpolation between adjacent evaluation points.

Confidence intervals on the final losses are the ones carried in each run's
`summary.json`. Crossings are reported to the precision the evaluation grid
supports and no further: adjacent grid points are roughly 0.82M tokens apart,
so a crossing is located to within that interval, not to the digit.

## 3. The replication

Through 50M tokens on a GH200, n = 5:

| | mean crossing | which mixer leads after |
|---|---|---|
| first crossing | **1.05M tokens** | minGRU |
| second crossing | **12.35M tokens** | attention |

The single-seed result did not survive. It reported one crossing, in a window
where the replication finds minGRU already ahead and staying ahead for another
five million tokens.

Final validation loss at 49.99M tokens:

| arm | mean | 95% CI |
|---|---|---|
| attention | **4.222** | [4.204, 4.240] |
| minGRU | **4.449** | [4.423, 4.475] |

Attention finishes **0.227** nats ahead. That much of the original conclusion
holds. Its location does not.

## 4. Isolating the confounds

Each condition below changes exactly one thing and re-runs five seeds.

| condition | run | result |
|---|---|---|
| batch 8, the original batch size | `crossover8m_bs8` | **no crossing** through 7.38M tokens |
| 20M budget, cosine schedule truncated with it | `crossover20m_locked` | crossings at 1.05M and **14.58M** |
| 20M budget, 50M cosine horizon kept | `crossover20m_matched_lr` | crossings at 1.04M and **12.33M** |

The batch-size row is the sharpest. At the original batch size, on the new
hardware, the crossing the original measurement reported **does not occur at
all** within the window that measurement covered — on any of five seeds.

The two 20M rows differ only in whether the cosine schedule is told about the
truncation. That single choice moves the late crossing by **2.25M tokens**,
and the row that keeps the original 50M horizon lands within interpolation
error of the full-length run. The crossing point is a property of the
learning-rate schedule as much as of the mixers.

## 5. The ten-arm board, and a correction to this paper

Ranking ten mixer configurations at 50M tokens, five seeds each:

| rank | arm | final | 95% CI |
|---|---|---|---|
| 1 | attention | **4.222** | [4.204, 4.240] |
| 2 | hybrid minGRU×10 + attn×2 | **4.275** | [4.245, 4.305] |
| 3 | hybrid GDN periodic | 4.280 | [4.247, 4.313] |
| 4 | hybrid GDN bookend | 4.293 | [4.258, 4.328] |
| 5 | hybrid GDN×10 + attn×2 | 4.309 | [4.274, 4.344] |
| 6 | GDN | 4.429 | [4.402, 4.456] |
| 7 | minGRU | 4.449 | [4.423, 4.475] |
| 8 | hybrid Mamba×10 + attn×2 | 4.604 | [4.517, 4.692] |
| 9 | MLA | 4.606 | [4.579, 4.634] |
| 10 | Mamba-2 | 4.759 | [4.680, 4.837] |

**This table corrects the pre-loss version of this paper.** That version
reported attention first at 4.222 [4.204, 4.240] — which reproduces exactly —
"but statistically tied with a 10×minGRU + 2×attention hybrid at 4.232
[4.210, 4.254]". The second half does not reproduce. The runner-up computes to
**4.275 [4.245, 4.305]** from these runs, and its interval is **disjoint** from
attention's. The 4.232 figure appears in no surviving run.

The nearest thing to it is the matched batch-32 sweep
(`crossover50m_matched32`), which does place a minGRU hybrid on top at 4.214
[4.207, 4.221] — but that sweep contains **no attention arm**, so it cannot
support a claim about a tie with attention.

A paper arguing that rankings are artifacts of the measurement had a ranking
claim of its own that depended on which condition was tabulated. It is recorded
here rather than quietly amended, because it is the same error the paper is
about.

## 6. The second axis: an optimizer funnel

A five-stage funnel on Apple silicon, 16 candidates enrolled — 14 ran; two
(`mimuon_adamw`, `soap_adamw`) were blocked because exact singular-gap and
SOAP-basis routing need a GPU eigensolver that Metal does not expose, and a CPU
fallback would force a host synchronisation the funnel forbids.

- **MONA** ranks **first** at 16M parameters (2.180898 BPB, two seeds) and
  **last of four** at an exact 128M-parameter model (2.950696).
- **Polar Express Muon** ranks **third** in the initial learning-rate screen
  and is the eventual 128M champion (2.169919 BPB).

Neither ordering is stable across the scale at which it was measured. The
funnel's own confidence intervals were also corrected: at its two-seed stages
the interval had been computed with a normal quantile, understating the
half-width by roughly 6.5× (t₁ = 12.706 against z = 1.960). With the correct
Student-*t* multiplier the champion-selection intervals **overlap**, and the
selection rests on a sign-consistent 2-of-2 ordering rather than on separated
intervals. The recorded champion is unchanged.

## 7. What this does and does not establish

It establishes that on this model, this data and these mixers, the crossing
token moves with batch size and with the learning-rate horizon, and that at
the original batch size it does not exist within the original window. It
establishes that two independent axes — a mixer comparison and an optimizer
funnel — each produce orderings that reverse under a change of scale or recipe.

It does not establish a general law about attention and recurrence. Every run
here is one architecture family, one data pipeline, one parameter scale, and
one hardware generation per condition. All §4 conditions use standard
parametrisation with a single global Muon learning rate, so a reader cannot
separate "the crossing moved" from "the learning rate was no longer right for
one arm" — that separation needs a µP sweep this work does not contain.

The claim we would defend is narrow and, we think, sufficient: **a method
ordering measured at one recipe is a measurement, not a property**, and
reporting it as the latter is the error this paper was written to document —
including where this paper committed it.

## 8. Reproduction

```bash
python3 paper/derive_figures.py           # every figure in this document
python3 paper/derive_figures.py --json    # machine-readable
python3 paper/derive_figures.py --check   # fail if the prose has drifted
```

The script reads `nanolab/out/crossover*/` and
`research/native-optimizer-funnel.json` directly. It contains no literals
copied from this text; the numbers above were written from its output.
